"""Build the four-low-view sliding-window cache for active-view RL."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping

import cv2
import numpy as np
import torch
import yaml

from .vpoclip_adapter import VPOCLIPAdapter
from .body_relative_geometry import BodyRelativeResolver, direction_labels, json_safe, wrap_degrees

ROOT = Path(__file__).resolve().parents[1]
LOW_CAMERAS = ("C001", "C003", "C005", "C007")
SAMPLE_RE = re.compile(r"^(A\d+)_P(\d+)_G(\d+)_C(\d+)$")


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["_config_path"] = str(path)
    return config


def resolve(config: Mapping[str, Any], value: str | Path) -> Path:
    path = Path(value)
    base = Path(str(config["_config_path"])).parent
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _subject(name: str) -> str:
    match = re.search(r"_P(\d+)_", Path(name).stem)
    if not match:
        raise ValueError(f"cannot parse subject from {name}")
    return f"P{int(match.group(1)):03d}"


def _skeleton_path(root: Path, stem: str, subject: str) -> Path:
    band = "P001-P050" if int(subject[1:]) <= 50 else "P051-P100"
    direct = root / f"Skeleton({band})" / band / f"{stem}.csv"
    if direct.exists():
        return direct
    matches = list(root.glob(f"**/{stem}.csv"))
    if not matches:
        raise FileNotFoundError(f"skeleton CSV not found for {stem}")
    return matches[0]


def _load_angles(path: Path) -> Dict[str, float]:
    result: Dict[str, float] = {}
    if not path.exists():
        return result
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                if row.get("estimated_relative_yaw_deg", "").strip():
                    result[row["sample_id"]] = float(row["estimated_relative_yaw_deg"])
            except (KeyError, ValueError):
                continue
    return result


def _angle(angles: Mapping[str, float], base: str, camera: str) -> tuple[float, str]:
    key = f"{base}_{camera}"
    if key in angles:
        return float(angles[key]), "csv"
    # Kept only for migration from the old P001-P005 CSV.
    return {"C001": 0.0, "C002": 0.0, "C003": -45.0, "C004": -45.0,
            "C005": 45.0, "C006": 45.0, "C007": 135.0, "C008": 135.0}.get(camera, 0.0), "camera_order_fallback"


def _window_specs(video: Path, seconds: float, stride: float, frames: int, maximum: int | None,
                  total_override: int | None = None) -> list[dict[str, Any]]:
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 20.0)
    cap.release()
    if total_override is not None:
        total = min(total, int(total_override))
    if total <= 0:
        raise ValueError(f"cannot read frame count from {video}")
    window = max(frames, int(round(seconds * fps)))
    step = max(1, int(round(stride * fps)))
    if total <= window:
        starts = [0]
    else:
        starts = list(range(0, total - window + 1, step))
        if starts[-1] != total - window:
            starts.append(total - window)
    if maximum and maximum > 0 and len(starts) > maximum:
        starts = [starts[int(round(x))] for x in np.linspace(0, len(starts) - 1, maximum)]
    result = []
    for index, start in enumerate(starts):
        end = min(total, start + window)
        samples = np.linspace(start, max(start, end - 1), frames).round().astype(np.int64)
        result.append({
            "window_index": index,
            "start_frame": int(start),
            "end_frame_exclusive": int(end),
            "sample_frames": [int(x) for x in samples],
        })
    return result


def _csv_geometry_fallback(episode: Mapping[str, Any]) -> dict[str, Any]:
    """Build per-recording geometry when the raw 3-D skeleton is unavailable.

    ``angle_deg`` is the recording-specific camera forward-axis yaw relative to
    C001 from ``low_view_camera_angles_split55.csv``.  It is not a global
    camera-ID direction table.  The common 180-degree optical-axis reversal
    cancels in the relative differences used by the candidate scorer and move
    cost, so the values can be used as a conservative direction-set fallback.
    """

    values = []
    for view in episode["views"]:
        try:
            value = float(view.get("angle_deg", 0.0))
        except (TypeError, ValueError):
            value = 0.0
        values.append(wrap_degrees(value) if math.isfinite(value) else 0.0)
    results = []
    for view, bearing in zip(episode["views"], values):
        label4, label4_zh, label8, label8_zh = direction_labels(bearing)
        results.append({
            "camera": str(view["camera"]),
            "relative_bearing_deg": float(bearing),
            "bearing_mad_deg": 0.0,
            "bearing_inlier_frames": 0,
            "direction_4": label4,
            "direction_4_zh": label4_zh,
            "direction_8": label8,
            "direction_8_zh": label8_zh,
            "camera_center_x_m": float("nan"),
            "camera_center_y_m": float("nan"),
            "camera_center_z_m": float("nan"),
            "fit_median_residual_m": float("nan"),
            "fit_p90_residual_m": float("nan"),
            "geometry_confidence": 0.35,
            "geometry_source": "csv_optical_axis_relative_yaw_fallback",
        })
    ordered = sorted(results, key=lambda result: result["relative_bearing_deg"])
    rank = {result["camera"]: index for index, result in enumerate(ordered)}
    for result in results:
        result["signed_bearing_rank"] = rank[result["camera"]]
    return {
        "body_yaw_deg_c001": 0.0,
        "body_yaw_mad_deg": 180.0,
        "body_yaw_inlier_frames": 0,
        "body_yaw_confidence": 0.35,
        "camera_order_by_signed_bearing": [result["camera"] for result in ordered],
        "geometry_fallback": "csv_optical_axis_relative_yaw",
        "views": results,
    }


def _source_rows(config: Mapping[str, Any], split: str) -> list[tuple[str, int]]:
    data = config["data"]
    source_root = data["source_split_dir"]
    if isinstance(source_root, dict):
        source_root = source_root[split]
    source = resolve(config, source_root)
    prefix = data["source_prefix"][split]
    names = np.load(source / f"{prefix}_sample_names.npy", allow_pickle=True).astype(str)
    labels = np.load(source / f"{prefix}_labels.npy").astype(np.int64)
    allowed = set(data["subject_groups"][split])
    excluded = set(int(x) for x in data.get("exclude_labels_by_split", {}).get(split, data.get("exclude_labels", [])))
    allowed_labels = data.get("allowed_labels_by_split", {}).get(split)
    allowed_labels = set(int(x) for x in allowed_labels) if allowed_labels is not None else None
    allowed_recordings = data.get("allowed_recording_ids")
    recording_filter_labels = None
    if allowed_recordings is None and data.get("recording_manifest"):
        manifest = resolve(config, data["recording_manifest"])
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        recording_filter_labels = set(payload["unseen_classes_zero_based"])
        allowed_recordings = []
        for item in payload.get("selected", []):
            allowed_recordings.extend(item.get("complete_recording_ids", []))
    allowed_recordings = (set(str(x) for x in allowed_recordings)
                          if allowed_recordings is not None else None)
    rows = []
    for name, label in zip(names, labels):
        name = str(name)
        stem = Path(name).stem
        recording_id = stem.rsplit("_", 1)[0]
        if _subject(name) not in allowed or int(label) in excluded:
            continue
        if allowed_labels is not None and int(label) not in allowed_labels:
            continue
        if (allowed_recordings is not None
                and (recording_filter_labels is None or int(label) in recording_filter_labels)
                and recording_id not in allowed_recordings):
            continue
        rows.append((name, int(label)))
    return rows


def build_episode_manifest(config: Dict[str, Any], split: str) -> Path:
    """Create one row per synchronized person/action/repetition/window."""
    out = resolve(config, config["cache"]["root"]) / split
    out.mkdir(parents=True, exist_ok=True)
    rgb_root = resolve(config, config["data"]["rgb_root"])
    skeleton_root = resolve(config, config["data"]["skeleton_root"])
    angles = _load_angles(resolve(config, config["data"]["angle_csv"]))
    grouped: dict[str, list[tuple[str, int]]] = {}
    for name, label in _source_rows(config, split):
        grouped.setdefault(Path(name).stem.rsplit("_", 1)[0], []).append((name, label))
    by_family: dict[str, list[tuple[str, int]]] = {}
    for base, entries in grouped.items():
        by_family.setdefault(base.rsplit("_", 1)[0], []).extend(entries)
    candidate_cameras = tuple(config["cache"].get("candidate_cameras", LOW_CAMERAS))
    allow_cross_group = bool(config["data"].get("allow_cross_group_views", False))
    allow_any_four = bool(config["data"].get("allow_available_four_views", False))
    allow_variable = bool(config["data"].get("allow_variable_views", False))
    minimum_views = int(config["data"].get("minimum_views", 2))
    if minimum_views < 2 or minimum_views > len(candidate_cameras):
        raise ValueError(
            "minimum_views must be between 2 and the number of candidate cameras"
        )
    merge_actions = set()
    for value in config["data"].get("merge_group_actions", []):
        text = str(value).strip().upper()
        merge_actions.add(text if text.startswith("A") else f"A{int(text):03d}")
    processed_merged_families: set[str] = set()
    episodes: list[dict[str, Any]] = []
    for base, entries in sorted(grouped.items()):
        source_name, label = entries[0]
        subject = _subject(source_name)
        parts = base.split("_")
        if len(parts) != 3 or not parts[0].startswith("A") or not parts[1].startswith("P") or not parts[2].startswith("G"):
            continue
        action, person, group = parts
        episode_cameras = candidate_cameras
        by_camera = {Path(name).stem.rsplit("_", 1)[1]: (name, lab, Path(name).stem.rsplit("_", 1)[0]) for name, lab in entries}
        selected = []
        family = base.rsplit("_", 1)[0]
        if action in merge_actions:
            # A049/A050 are recorded as one low camera per Group.  Treat all
            # Groups for the same A/P recording family as one logical
            # episode, and emit one representative for each low-camera slot.
            # The first base is the deterministic owner; otherwise the same
            # merged family would be duplicated once per Group.
            if family in processed_merged_families:
                continue
            family_entries = by_family.get(family, [])
            family_by_camera: dict[str, list[tuple[str, int, str]]] = {}
            for name, lab in family_entries:
                camera = Path(name).stem.rsplit("_", 1)[1]
                family_by_camera.setdefault(camera, []).append(
                    (name, lab, Path(name).stem.rsplit("_", 1)[0])
                )
            for camera in episode_cameras:
                if camera in family_by_camera:
                    selected.append(sorted(family_by_camera[camera], key=lambda x: x[2])[0])
            if len(selected) >= minimum_views:
                episode_cameras = tuple(camera for camera, _, _ in selected)
                processed_merged_families.add(family)
            else:
                continue
        elif all(camera in by_camera for camera in episode_cameras):
            selected = [by_camera[camera] for camera in episode_cameras]
        elif allow_cross_group:
            family_entries = by_family.get(family, [])
            family_by_camera: dict[str, list[tuple[str, int, str]]] = {}
            for name, lab in family_entries:
                cam = Path(name).stem.rsplit("_", 1)[1]
                family_by_camera.setdefault(cam, []).append((name, lab, Path(name).stem.rsplit("_", 1)[0]))
            if all(camera in family_by_camera for camera in episode_cameras):
                selected = [sorted(family_by_camera[camera], key=lambda x: x[2])[0] for camera in episode_cameras]
        if not selected and allow_any_four:
            family_entries = by_family.get(family, [])
            family_by_camera: dict[str, list[tuple[str, int, str]]] = {}
            for name, lab in family_entries:
                cam = Path(name).stem.rsplit("_", 1)[1]
                family_by_camera.setdefault(cam, []).append((name, lab, Path(name).stem.rsplit("_", 1)[0]))
            available = sorted(family_by_camera, key=lambda x: int(x[1:]))
            if len(available) >= 4:
                selected = [sorted(family_by_camera[camera], key=lambda x: x[2])[0] for camera in available[:4]]
                episode_cameras = tuple(camera for camera, _, _ in selected)
        if not selected and allow_variable:
            # Keep incomplete ordinary recordings in their original Group
            # whenever at least two low cameras exist.  This is the intended
            # 2-of-2/3-of-3 path for A010/A012/A013 and the corresponding
            # partially recorded Groups; it never infers a missing camera.
            available = [camera for camera in episode_cameras if camera in by_camera]
            if len(available) >= minimum_views:
                selected = [by_camera[camera] for camera in available]
                episode_cameras = tuple(available)
        if len(selected) < minimum_views or (not allow_variable and len(selected) != 4):
            continue
        views = []
        try:
            for name, lab, actual_base in selected:
                camera = Path(name).stem.rsplit("_", 1)[1]
                stem = Path(name).stem
                video = rgb_root / name
                try:
                    skeleton = _skeleton_path(skeleton_root, stem, subject)
                except FileNotFoundError:
                    if not bool(config["data"].get("allow_missing_skeleton", False)):
                        raise
                    skeleton = None
                if not video.exists():
                    raise FileNotFoundError(video)
                angle, angle_source = _angle(angles, actual_base, camera)
                if bool(config["data"].get("require_recording_angles", False)) and angle_source != "csv":
                    raise ValueError(
                        "missing recording-specific camera angle for "
                        f"{actual_base}_{camera}; refusing camera-order fallback"
                    )
                cap = cv2.VideoCapture(str(video))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 20.0)
                cap.release()
                views.append({"camera": camera, "video": str(video), "sample_name": str(name), "skeleton": str(skeleton) if skeleton is not None else None, "angle_deg": angle, "angle_source": angle_source, "frame_count": frame_count, "fps": fps, "source_base": actual_base})
        except FileNotFoundError:
            continue
        common_total = min(int(v["frame_count"] * float(views[0]["fps"]) / max(float(v["fps"]), 1e-6)) for v in views)
        for window in _window_specs(
            Path(views[0]["video"]),
            float(config["data"]["window_seconds"]),
            float(config["data"]["stride_seconds"]),
            int(config["data"]["frames"]),
            config["data"].get("max_windows_per_clip"),
            total_override=common_total,
        ):
            episodes.append({
                "episode_id": len(episodes),
                "subject": subject,
                "action": action,
                "group": (
                    "MERGED"
                    if action in merge_actions
                    else group if all(v["source_base"] == base for v in views) else "MIXED"
                ),
                "label": int(label),
                "base_sample": family + "_MERGED" if action in merge_actions else base,
                "window": window,
                "views": views,
            })
            if config["data"].get("max_episodes") and len(episodes) >= int(config["data"]["max_episodes"]):
                break
        if config["data"].get("max_episodes") and len(episodes) >= int(config["data"]["max_episodes"]):
            break
    manifest = out / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in episodes:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "manifest_summary.json").write_text(json.dumps({
        "split": split,
        "episodes": len(episodes),
        "views": LOW_CAMERAS,
        "variable_views": allow_variable,
        "minimum_views": minimum_views,
        "merge_group_actions": sorted(merge_actions),
    }, indent=2), encoding="utf-8")
    print(f"manifest {split}: {len(episodes)} episodes -> {manifest}", flush=True)
    return manifest


def _load_raw_module():
    path = ROOT / "test_raw_end_to_end.py"
    spec = importlib.util.spec_from_file_location("vpoclip_raw_pipeline", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_video(
    path: Path,
    width: int = 640,
    height: int = 480,
    sample_frames: list[int] | tuple[int, ...] | None = None,
) -> np.ndarray:
    """Decode a video, retaining only requested frames when provided.

    The old implementation materialized every 640x480 frame even though the
    VPOCLIP input uses only 13 uniformly sampled frames.  That made cache
    generation memory-bound and left the GPU idle.  Sequential decoding still
    preserves the exact frame content while keeping memory proportional to the
    requested window.
    """
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    if sample_frames is None:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    else:
        requested = [max(0, int(index)) for index in sample_frames]
        unique_targets = sorted(set(requested))
        captured: dict[int, np.ndarray] = {}
        target_index = 0
        frame_index = 0
        while target_index < len(unique_targets):
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index == unique_targets[target_index]:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                captured[frame_index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                target_index += 1
            frame_index += 1
        if target_index != len(unique_targets):
            cap.release()
            raise ValueError(f"could not decode requested frames from {path}")
        frames = [captured[index] for index in requested]
    cap.release()
    if not frames:
        raise ValueError(f"could not decode {path}")
    return np.stack(frames, axis=0)


def _rtmpose_window(body: Any, frames: np.ndarray, raw: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pose = np.zeros((3, len(frames), 17, 2), dtype=np.float32)
    for time_index, frame in enumerate(frames):
        keypoints, scores = body(frame)
        keypoints, scores = np.asarray(keypoints), np.asarray(scores)
        if keypoints.ndim != 3 or len(keypoints) == 0:
            continue
        order = np.argsort(scores.sum(axis=1))[::-1][:2]
        keypoints, scores = keypoints[order], scores[order]
        height, width = frame.shape[:2]
        for person in range(min(2, len(keypoints))):
            pose[0, time_index, :, person] = keypoints[person, :, 0] / width * 2.0 - 1.0
            pose[1, time_index, :, person] = keypoints[person, :, 1] / height * 2.0 - 1.0
            pose[2, time_index, :, person] = scores[person]
    pose_tensor = torch.from_numpy(pose)
    joint_xy = torch.from_numpy(pose[:2, :, :, 0].transpose(1, 2, 0).copy())
    score = pose[2]
    quality = torch.tensor([
        float((score > 0).mean()),
        float(score[score > 0].mean()) if (score > 0).any() else 0.0,
        float(score.max()),
        1.0,
    ])
    return pose_tensor, joint_xy, quality


def _precomputed_rtmpose_window(
    pose_array: Any,
    pose_index: Mapping[str, int],
    sample_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read the already extracted 13-frame RTMPose tensor for one video.

    The stored format is ``[3, 13, 17, 2]`` = x/y/score, time, COCO-17
    joints, person slots.  Keeping the lookup by the complete sample name is
    important: camera IDs are physical labels, not a direction ordering.
    """
    try:
        index = pose_index[str(sample_name)]
    except KeyError as exc:
        raise FileNotFoundError(f"RTMPose cache has no sample {sample_name}") from exc
    pose_np = np.asarray(pose_array[index], dtype=np.float32)
    if pose_np.shape != (3, 13, 17, 2):
        raise ValueError(f"unexpected RTMPose shape for {sample_name}: {pose_np.shape}")
    pose = torch.from_numpy(pose_np)
    joint = torch.from_numpy(pose_np[:2, :, :, 0].transpose(1, 2, 0).copy())
    score = pose[2]
    quality = torch.tensor([
        float((score > 0).float().mean()),
        float(score[score > 0].mean()) if (score > 0).any() else 0.0,
        float(score.max()),
        1.0,
    ])
    return pose, joint, quality


def _runtime_models(config: Mapping[str, Any], device: torch.device):
    raw = _load_raw_module()
    raw_cfg = config["raw"]
    args = SimpleNamespace(
        x3d_root=str(resolve(config, raw_cfg["x3d_root"])),
        x3d_config=str(resolve(config, raw_cfg["x3d_config"])),
        x3d_checkpoint=str(resolve(config, raw_cfg["x3d_checkpoint"])),
        x3d_layer=raw_cfg.get("x3d_layer", "s5"),
        ctrgcn_root=str(resolve(config, raw_cfg["ctrgcn_root"])),
        ctrgcn_config=str(resolve(config, raw_cfg["ctrgcn_config"])),
        ctrgcn_weights=str(resolve(config, raw_cfg["ctrgcn_weights"])),
        ctrgcn_hook_layer=raw_cfg.get("ctrgcn_hook_layer", "l4"),
        yolo_repo=str(resolve(config, raw_cfg["yolo_repo"])),
        yolo_weights=str(resolve(config, raw_cfg["yolo_weights"])),
        yolo_size=640, yolo_conf=0.25, yolo_iou=0.45, yolo_half=True,
        no_yolo=False, object_grid_size=6, object_value="presence", object_max_distance_weight=10.0,
        yolo_detect_every=1,
    )
    x3d_model, x3d_capture, x3d_hook, x3d_cfg = raw.load_x3d_model(args, device)
    pose_model, pose_capture, pose_hook = raw.load_ctrgcn_model(args, device)
    object_runner = raw.ObjectMapRunner(raw.load_yolo_model(args, device), device, args)
    adapter = VPOCLIPAdapter.from_config(
        str(resolve(config, config["recognizer"]["config"])),
        str(resolve(config, config["recognizer"]["checkpoint"])),
        str(device),
    )
    if raw_cfg.get("rtmpose_cache_root"):
        # The raw RTMPose tensors are already available for every source
        # split.  Reusing them keeps this cache aligned with VPOCLIP's
        # pre-extracted pose input and avoids running an identical detector
        # once per frame during the cache build.
        body = None
    else:
        import onnxruntime as ort
        from rtmlib import Body
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDAExecutionProvider unavailable for RTMPose")
        body = Body(mode=raw_cfg.get("rtmpose_mode", "balanced"), backend="onnxruntime", device="cuda")
    return raw, args, x3d_model, x3d_capture, x3d_cfg, pose_model, pose_capture, object_runner, body, adapter, x3d_hook, pose_hook


def export_cache(config: Dict[str, Any], manifest_path: Path) -> Path:
    """Run frozen VPOCLIP on all aligned views and write an atomic cache."""
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty manifest: {manifest_path}")
    output = manifest_path.parent
    tmp = output / ".cache_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    device = torch.device(str(config["runtime"].get("device", "cuda:0")))
    runtime = _runtime_models(config, device)
    raw, args, x3d_model, x3d_capture, x3d_cfg, pose_model, pose_capture, object_runner, body, adapter, x3d_hook, pose_hook = runtime
    n, views = len(rows), len(LOW_CAMERAS)
    arrays = {
        "z": np.lib.format.open_memmap(tmp / "z.npy", mode="w+", dtype=np.float32, shape=(n, views, 512)),
        "logits": np.lib.format.open_memmap(tmp / "logits.npy", mode="w+", dtype=np.float32, shape=(n, views, 55)),
        "pose": np.lib.format.open_memmap(tmp / "pose.npy", mode="w+", dtype=np.float16, shape=(n, views, 442)),
        # Multiplicative RS maps can exceed float16 range when several
        # same-class detections overlap.  Keep this cache in float32 so the
        # frozen VPOCLIP object branch receives finite, lossless values.
        "object_map": np.lib.format.open_memmap(tmp / "object_map.npy", mode="w+", dtype=np.float32, shape=(n, views, 1800)),
        "image_quality": np.lib.format.open_memmap(tmp / "image_quality.npy", mode="w+", dtype=np.float32, shape=(n, views, 4)),
        "view_geometry": np.lib.format.open_memmap(tmp / "view_geometry.npy", mode="w+", dtype=np.float32, shape=(n, views, 4)),
        "reachable": np.lib.format.open_memmap(tmp / "reachable.npy", mode="w+", dtype=np.bool_, shape=(n, views, views)),
        "view_valid": np.lib.format.open_memmap(tmp / "view_valid.npy", mode="w+", dtype=np.bool_, shape=(n, views)),
        "move_cost": np.lib.format.open_memmap(tmp / "move_cost.npy", mode="w+", dtype=np.float32, shape=(n, views, views)),
        "labels": np.lib.format.open_memmap(tmp / "labels.npy", mode="w+", dtype=np.int64, shape=(n,)),
        "target_columns": np.lib.format.open_memmap(tmp / "target_columns.npy", mode="w+", dtype=np.int64, shape=(n,)),
    }
    batch_size = int(config["raw"].get("batch_size", 8))
    video_cache: dict[tuple[str, tuple[int, ...]], np.ndarray] = {}
    decode_workers = int(config['raw'].get('decode_workers', 1))
    pose_cache: dict[tuple, tuple] = {}
    precomputed_pose = None
    precomputed_pose_index: dict[str, int] = {}
    if config["raw"].get("rtmpose_cache_root"):
        split_key = {"dqn_train": "dqn", "val": "val", "test": "test"}
        pose_root = resolve(config, config["raw"]["rtmpose_cache_root"])
        pose_prefix = split_key[manifest_path.parent.name]
        precomputed_pose = np.load(
            pose_root / f"{pose_prefix}_x.npy", mmap_mode="r", allow_pickle=False
        )
        pose_names = np.load(
            pose_root / f"{pose_prefix}_sample_names.npy", allow_pickle=True
        ).astype(str)
        precomputed_pose_index = {name: index for index, name in enumerate(pose_names)}
    geometry_resolver = BodyRelativeResolver()
    metadata_rows = []
    try:
        for start in range(0, n, batch_size):
            batch_rows = rows[start:start + batch_size]
            requests = []
            for row in batch_rows:
                frame_key = tuple(int(index) for index in row["window"]["sample_frames"])
                for view in row["views"]:
                    key = (view["video"], frame_key)
                    if key not in video_cache:
                        requests.append(key)
            missing = list(dict.fromkeys(requests))
            if decode_workers > 1:
                with ThreadPoolExecutor(max_workers=decode_workers) as pool:
                    decoded = pool.map(
                        lambda request: _read_video(
                            Path(request[0]), sample_frames=request[1]
                        ),
                        missing,
                    )
                    for key, frames in zip(missing, decoded):
                        video_cache[key] = frames
            videos, poses, joints, objects, qualities = [], [], [], [], []
            row_view_counts: list[int] = []
            for row in batch_rows:
                frame_ids = row["window"]["sample_frames"]
                actual_count = len(row["views"])
                if actual_count < 1 or actual_count > views:
                    raise ValueError(f"invalid view count {actual_count} in {row['base_sample']}")
                row_view_counts.append(actual_count)
                # The tensors retain four physical camera slots for backward
                # compatibility.  A missing slot repeats an existing input
                # only for batching, then is zeroed and masked before writing;
                # it can never become a legal action downstream.
                for slot in range(views):
                    view = row["views"][min(slot, actual_count - 1)]
                    frame_key = tuple(int(index) for index in frame_ids)
                    video_key = (view["video"], frame_key)
                    if video_key not in video_cache:
                        video_cache[video_key] = _read_video(
                            Path(view["video"]), sample_frames=frame_key
                        )
                    frames = video_cache[video_key]
                    x3d_size = int(config["raw"].get("x3d_input_size", 0) or getattr(x3d_cfg.TRANSFORM.TEST, "TENSOR_RESIZE_SIZE", 0) or getattr(x3d_cfg.TRANSFORM.TEST, "TEST_CROP_SIZE", 0) or 182)
                    videos.append(raw.x3d_tensor_from_frames(frames, x3d_size, x3d_cfg.TRANSFORM.MEAN, x3d_cfg.TRANSFORM.STD))
                    if precomputed_pose is not None:
                        pose, joint, quality = _precomputed_rtmpose_window(
                            precomputed_pose,
                            precomputed_pose_index,
                            view["sample_name"],
                        )
                    elif config['raw'].get('reuse_pose_frames', False):
                        parts = []
                        for fid, frame in zip(frame_ids, frames):
                            key = (view['video'], min(int(fid), len(frames)-1))
                            if key not in pose_cache:
                                pose_cache[key] = _rtmpose_window(body, frame[None], raw)
                            parts.append(pose_cache[key])
                        pose = torch.cat([p[0] for p in parts], dim=1)
                        joint = torch.cat([p[1] for p in parts], dim=0)
                        score = pose[2]
                        quality = torch.tensor([float((score > 0).float().mean()), float(score[score > 0].mean()) if (score > 0).any() else 0., float(score.max()), 1.])
                    else:
                        pose, joint, quality = _rtmpose_window(body, frames, raw)
                    poses.append(pose); joints.append(joint); qualities.append(quality); objects.append(frames[len(frames) // 2])
            video = torch.stack(videos).to(device, non_blocking=True)
            skeleton = torch.stack(poses).to(device, non_blocking=True)
            joint = torch.stack(joints).to(device, non_blocking=True)
            object_map = object_runner(objects)
            if not torch.isfinite(object_map).all():
                raise RuntimeError("YOLO object map contains NaN/Inf before cache write")
            with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                video_feature = raw.x3d_features_from_model(x3d_model, x3d_capture, video, args)
                pose_feature = raw.ctrgcn_pose_from_model(pose_model, pose_capture, skeleton, args)
                encoded = adapter.encode_all_views({
                    "video": video_feature.unsqueeze(0), "pose": pose_feature.unsqueeze(0),
                    "object": object_map.unsqueeze(0), "joint_xy": joint.unsqueeze(0),
                })
            encoded_z = encoded["z"][0].cpu().numpy().reshape(len(batch_rows), views, 512)
            encoded_logits = encoded["logits"][0].cpu().numpy().reshape(len(batch_rows), views, 55)
            cached_pose = joint.cpu().numpy().reshape(len(batch_rows), views, 442)
            cached_objects = object_map.cpu().numpy().reshape(len(batch_rows), views, 1800)
            cached_quality = torch.stack(qualities).numpy().reshape(len(batch_rows), views, 4)
            for offset, (row, actual_count) in enumerate(zip(batch_rows, row_view_counts)):
                if actual_count < views:
                    encoded_z[offset, actual_count:] = 0.0
                    encoded_logits[offset, actual_count:] = 0.0
                    cached_pose[offset, actual_count:] = 0.0
                    cached_objects[offset, actual_count:] = 0.0
                    cached_quality[offset, actual_count:] = 0.0
                arrays["z"][start + offset] = encoded_z[offset]
                arrays["logits"][start + offset] = encoded_logits[offset]
                arrays["pose"][start + offset] = cached_pose[offset]
                arrays["object_map"][start + offset] = cached_objects[offset]
                arrays["image_quality"][start + offset] = cached_quality[offset]
                try:
                    resolved_geometry = geometry_resolver.resolve(row)
                except (FileNotFoundError, TypeError):
                    if not bool(config["data"].get("allow_missing_skeleton", False)):
                        raise
                    resolved_geometry = _csv_geometry_fallback(row)
                angle_degrees = np.asarray([
                    float(view["relative_bearing_deg"])
                    for view in resolved_geometry["views"]
                ])
                angles = np.radians(angle_degrees)
                geometry = np.stack((np.sin(angles), np.cos(angles)), axis=1)
                cost = np.zeros((views, views), dtype=np.float32)
                for source_id in range(actual_count):
                    for dest_id in range(actual_count):
                        cost[source_id, dest_id] = abs(
                            wrap_degrees(angle_degrees[dest_id] - angle_degrees[source_id])
                        ) / 180.0
                geometry_padded = np.zeros((views, 4), dtype=np.float32)
                geometry_padded[:actual_count, :2] = geometry
                arrays["view_geometry"][start + offset] = geometry_padded
                reachable = np.zeros((views, views), dtype=np.bool_)
                reachable[:actual_count, :actual_count] = True
                # Padded camera slots are invalid actions, but the cache
                # contract still requires every fixed slot to have a reset
                # self-edge.  `view_valid` remains the authority for whether
                # a padded slot can actually be selected.
                np.fill_diagonal(reachable, True)
                arrays["reachable"][start + offset] = reachable
                arrays["view_valid"][start + offset] = False
                arrays["view_valid"][start + offset, :actual_count] = True
                arrays["move_cost"][start + offset] = cost
                arrays["image_quality"][start + offset, :actual_count, 3] = float(
                    resolved_geometry["body_yaw_confidence"]
                )
                arrays["labels"][start + offset] = row["label"]
                arrays["target_columns"][start + offset] = row["label"]
                row["body_relative_geometry"] = {
                    key: value for key, value in resolved_geometry.items() if key != "views"
                }
                for original, resolved_view in zip(row["views"], resolved_geometry["views"]):
                    original.update(resolved_view)
                metadata_rows.append(row)
            if len(video_cache) > 32:
                video_cache.clear()
                pose_cache.clear()
            if (start + len(batch_rows)) % max(10 * batch_size, 1) == 0 or start + len(batch_rows) == n:
                print(f"cache {output.name}: {start + len(batch_rows)}/{n}", flush=True)
    finally:
        x3d_hook.remove(); pose_hook.remove()
    for value in arrays.values(): value.flush()
    np.save(tmp / "text_prototypes.npy", adapter.model.text_features.detach().float().cpu().numpy())
    (tmp / "metadata.json").write_text(json.dumps(json_safe({
        "format": "active_view_cache_v2", "episodes": metadata_rows, "num_views": 4,
        "classes": 55, "embedding_dim": 512, "input_resolution": [640, 480],
        "variable_views": True,
        "minimum_views": int(config["data"].get("minimum_views", 2)),
        "merge_group_actions": sorted({
            str(value).upper() if str(value).upper().startswith("A") else f"A{int(value):03d}"
            for value in config["data"].get("merge_group_actions", [])
        }),
        "windowing": {"seconds": config["data"]["window_seconds"], "stride_seconds": config["data"]["stride_seconds"], "frames": config["data"]["frames"]},
        "protocol": "strict_zsl_50_5", "angle_csv": str(resolve(config, config["data"]["angle_csv"])),
        "geometry_contract": {
            "definition": (
                "recording-specific candidate bearing; relative to anatomical body "
                "forward when 3-D skeleton geometry is available, otherwise the "
                "per-recording source-camera optical-axis yaw relative to C001"
            ),
            "sign": "negative=body-left, zero=front, positive=body-right",
            "camera_id_is_not_angular_order": True,
            "fallback_definition": (
                "per-recording source camera forward-axis yaw relative to C001; "
                "not an estimate of human anatomical yaw"
            ),
            "missing_skeleton_fallback": "per-recording CSV optical-axis relative yaw; confidence=0.35",
        },
    }), ensure_ascii=False, allow_nan=False), encoding="utf-8")
    for value in arrays.values(): del value
    for old in output.iterdir():
        if old.name not in {"manifest.jsonl", "manifest_summary.json", ".cache_tmp"}:
            if old.is_dir(): shutil.rmtree(old)
            else: old.unlink()
    for item in tmp.iterdir(): item.rename(output / item.name)
    tmp.rmdir()
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config_rl.yaml")))
    parser.add_argument("--split", choices=("dqn_train", "val", "test"), required=True)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.max_episodes:
        config["data"]["max_episodes"] = args.max_episodes
    manifest = build_episode_manifest(config, args.split)
    if not args.manifest_only:
        export_cache(config, manifest)


if __name__ == "__main__":
    main()
