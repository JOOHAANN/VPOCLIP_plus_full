"""Build the four-low-view sliding-window cache for active-view RL."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
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
from .body_relative_geometry import BodyRelativeResolver, json_safe, wrap_degrees

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
        if all(camera in by_camera for camera in episode_cameras):
            selected = [by_camera[camera] for camera in episode_cameras]
        elif allow_cross_group:
            family_entries = by_family.get(base.rsplit("_", 1)[0], [])
            family_by_camera: dict[str, list[tuple[str, int, str]]] = {}
            for name, lab in family_entries:
                cam = Path(name).stem.rsplit("_", 1)[1]
                family_by_camera.setdefault(cam, []).append((name, lab, Path(name).stem.rsplit("_", 1)[0]))
            if all(camera in family_by_camera for camera in episode_cameras):
                selected = [sorted(family_by_camera[camera], key=lambda x: x[2])[0] for camera in episode_cameras]
        if not selected and allow_any_four:
            family_entries = by_family.get(base.rsplit("_", 1)[0], [])
            family_by_camera: dict[str, list[tuple[str, int, str]]] = {}
            for name, lab in family_entries:
                cam = Path(name).stem.rsplit("_", 1)[1]
                family_by_camera.setdefault(cam, []).append((name, lab, Path(name).stem.rsplit("_", 1)[0]))
            available = sorted(family_by_camera, key=lambda x: int(x[1:]))
            if len(available) >= 4:
                selected = [sorted(family_by_camera[camera], key=lambda x: x[2])[0] for camera in available[:4]]
                episode_cameras = tuple(camera for camera, _, _ in selected)
        if len(selected) != 4:
            continue
        views = []
        try:
            for name, lab, actual_base in selected:
                camera = Path(name).stem.rsplit("_", 1)[1]
                stem = Path(name).stem
                video = rgb_root / name
                skeleton = _skeleton_path(skeleton_root, stem, subject)
                if not video.exists():
                    raise FileNotFoundError(video)
                angle, angle_source = _angle(angles, actual_base, camera)
                cap = cv2.VideoCapture(str(video))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 20.0)
                cap.release()
                views.append({"camera": camera, "video": str(video), "skeleton": str(skeleton), "angle_deg": angle, "angle_source": angle_source, "frame_count": frame_count, "fps": fps, "source_base": actual_base})
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
                "group": group if all(v["source_base"] == base for v in views) else "MIXED",
                "label": int(label),
                "base_sample": base,
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
    (out / "manifest_summary.json").write_text(json.dumps({"split": split, "episodes": len(episodes), "views": LOW_CAMERAS}, indent=2), encoding="utf-8")
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


def _read_video(path: Path, width: int = 640, height: int = 480) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
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
        "move_cost": np.lib.format.open_memmap(tmp / "move_cost.npy", mode="w+", dtype=np.float32, shape=(n, views, views)),
        "labels": np.lib.format.open_memmap(tmp / "labels.npy", mode="w+", dtype=np.int64, shape=(n,)),
        "target_columns": np.lib.format.open_memmap(tmp / "target_columns.npy", mode="w+", dtype=np.int64, shape=(n,)),
    }
    batch_size = int(config["raw"].get("batch_size", 8))
    video_cache: dict[str, np.ndarray] = {}
    decode_workers = int(config['raw'].get('decode_workers', 1))
    pose_cache: dict[tuple, tuple] = {}
    geometry_resolver = BodyRelativeResolver()
    metadata_rows = []
    try:
        for start in range(0, n, batch_size):
            batch_rows = rows[start:start + batch_size]
            missing = list(dict.fromkeys(v['video'] for r in batch_rows for v in r['views'] if v['video'] not in video_cache))
            if decode_workers > 1:
                with ThreadPoolExecutor(max_workers=decode_workers) as pool:
                    for path, decoded in zip(missing, pool.map(lambda p: _read_video(Path(p)), missing)):
                        video_cache[path] = decoded
            videos, poses, joints, objects, qualities = [], [], [], [], []
            for row in batch_rows:
                frame_ids = row["window"]["sample_frames"]
                for view in row["views"]:
                    if view["video"] not in video_cache:
                        video_cache[view["video"]] = _read_video(Path(view["video"]))
                    source = video_cache[view["video"]]
                    frames = source[[min(int(i), len(source) - 1) for i in frame_ids]]
                    x3d_size = int(config["raw"].get("x3d_input_size", 0) or getattr(x3d_cfg.TRANSFORM.TEST, "TENSOR_RESIZE_SIZE", 0) or getattr(x3d_cfg.TRANSFORM.TEST, "TEST_CROP_SIZE", 0) or 182)
                    videos.append(raw.x3d_tensor_from_frames(frames, x3d_size, x3d_cfg.TRANSFORM.MEAN, x3d_cfg.TRANSFORM.STD))
                    if config['raw'].get('reuse_pose_frames', False):
                        parts = []
                        for fid, frame in zip(frame_ids, frames):
                            key = (view['video'], min(int(fid), len(source)-1))
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
            arrays["z"][start:start + len(batch_rows)] = encoded["z"][0].cpu().numpy().reshape(len(batch_rows), views, 512)
            arrays["logits"][start:start + len(batch_rows)] = encoded["logits"][0].cpu().numpy().reshape(len(batch_rows), views, 55)
            arrays["pose"][start:start + len(batch_rows)] = joint.cpu().numpy().reshape(len(batch_rows), views, 442)
            arrays["object_map"][start:start + len(batch_rows)] = object_map.cpu().numpy().reshape(len(batch_rows), views, 1800)
            arrays["image_quality"][start:start + len(batch_rows)] = torch.stack(qualities).numpy().reshape(len(batch_rows), views, 4)
            for offset, row in enumerate(batch_rows):
                resolved_geometry = geometry_resolver.resolve(row)
                angle_degrees = np.asarray([
                    float(view["relative_bearing_deg"])
                    for view in resolved_geometry["views"]
                ])
                angles = np.radians(angle_degrees)
                geometry = np.stack((np.sin(angles), np.cos(angles)), axis=1)
                cost = np.zeros((views, views), dtype=np.float32)
                for source_id in range(views):
                    for dest_id in range(views):
                        cost[source_id, dest_id] = abs(
                            wrap_degrees(angle_degrees[dest_id] - angle_degrees[source_id])
                        ) / 180.0
                arrays["view_geometry"][start + offset] = np.concatenate((geometry, np.zeros((views, 2), dtype=np.float32)), axis=1)
                arrays["reachable"][start + offset] = True
                arrays["move_cost"][start + offset] = cost
                arrays["image_quality"][start + offset, :, 3] = float(
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
        "windowing": {"seconds": config["data"]["window_seconds"], "stride_seconds": config["data"]["stride_seconds"], "frames": config["data"]["frames"]},
        "protocol": "strict_zsl_50_5", "angle_csv": str(resolve(config, config["data"]["angle_csv"])),
        "geometry_contract": {
            "definition": "candidate camera center bearing relative to anatomical body forward",
            "sign": "negative=body-left, zero=front, positive=body-right",
            "camera_id_is_not_angular_order": True,
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
