"""Body-relative camera bearings and explicit direction labels for ETRI views."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import yaml


LOW_CAMERAS = ("C001", "C003", "C005", "C007")
REQUIRED_JOINTS = (0, 1, 4, 8, 12, 16, 20)


def json_safe(value: Any) -> Any:
    """Replace non-finite diagnostic floats with JSON null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def wrap_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def circular_distance(a: float, b: float) -> float:
    return abs(wrap_degrees(a - b))


def circular_cluster_median(values: Sequence[float]) -> tuple[float, float, int]:
    if not values:
        raise ValueError("cannot summarize an empty angle sequence")
    best = min(
        values,
        key=lambda candidate: (
            -sum(circular_distance(candidate, value) <= 20.0 for value in values),
            sum(min(circular_distance(candidate, value), 45.0) for value in values),
        ),
    )
    inliers = [value for value in values if circular_distance(value, best) <= 30.0]
    unwrapped = [best + wrap_degrees(value - best) for value in inliers]
    estimate = statistics.median(unwrapped)
    mad = statistics.median(abs(value - estimate) for value in unwrapped)
    return wrap_degrees(estimate), float(mad), len(inliers)


def direction_labels(angle: float) -> tuple[str, str, str, str]:
    """Return four-way and eight-way labels; positive angles are body-right."""
    angle = wrap_degrees(angle)
    absolute = abs(angle)
    if absolute < 45.0:
        label4 = "front"
        label4_zh = "正面"
    elif absolute >= 135.0:
        label4 = "back"
        label4_zh = "背面"
    elif angle > 0:
        label4 = "right"
        label4_zh = "右侧"
    else:
        label4 = "left"
        label4_zh = "左侧"

    if absolute < 22.5:
        label8, label8_zh = "front", "正前"
    elif absolute < 67.5:
        label8, label8_zh = (
            ("front_right", "右前") if angle > 0 else ("front_left", "左前")
        )
    elif absolute < 112.5:
        label8, label8_zh = ("right", "正右") if angle > 0 else ("left", "正左")
    elif absolute < 157.5:
        label8, label8_zh = (
            ("back_right", "右后") if angle > 0 else ("back_left", "左后")
        )
    else:
        label8, label8_zh = "back", "正后"
    return label4, label4_zh, label8, label8_zh


def load_skeleton_xyz(path: Path) -> Dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load the best tracked body row per frame from an ETRI Kinect CSV."""
    frames: Dict[int, tuple[np.ndarray, np.ndarray]] = {}
    scores: Dict[int, int] = {}
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            try:
                frame = int(row[0])
                points = np.full((25, 3), np.nan, dtype=np.float64)
                valid = np.zeros(25, dtype=np.bool_)
                for joint in range(25):
                    offset = 3 + joint * 10
                    point = np.asarray(row[offset : offset + 3], dtype=np.float64)
                    tracking = float(row[offset + 9])
                    if np.isfinite(point).all() and point[2] > 0.0 and tracking > 0.0:
                        points[joint] = point
                        valid[joint] = True
            except (IndexError, ValueError):
                continue
            score = int(valid.sum())
            if score > scores.get(frame, -1):
                frames[frame] = (points, valid)
                scores[frame] = score
    return frames


def _rigid_fit(source: np.ndarray, destination: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0)
    destination_center = destination.mean(axis=0)
    u, _, vt = np.linalg.svd(
        (source - source_center).T @ (destination - destination_center)
    )
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    translation = destination_center - rotation @ source_center
    residual = np.linalg.norm(source @ rotation.T + translation - destination, axis=1)
    return rotation, translation, residual


def fit_camera_to_reference(
    reference: Mapping[int, tuple[np.ndarray, np.ndarray]],
    source: Mapping[int, tuple[np.ndarray, np.ndarray]],
) -> Dict[str, Any] | None:
    """Robustly fit source-camera XYZ into C001 coordinates."""
    common = sorted(set(reference) & set(source))
    if len(common) < 10:
        return None
    common = common[int(0.2 * len(common)) : max(int(0.8 * len(common)), 1)]
    source_points, reference_points = [], []
    for frame in common:
        ref_xyz, ref_valid = reference[frame]
        src_xyz, src_valid = source[frame]
        valid = ref_valid & src_valid
        if int(valid.sum()) >= 10:
            source_points.append(src_xyz[valid])
            reference_points.append(ref_xyz[valid])
    if not source_points:
        return None
    source_xyz = np.concatenate(source_points)
    reference_xyz = np.concatenate(reference_points)
    keep = np.ones(len(source_xyz), dtype=np.bool_)
    rotation = np.eye(3)
    translation = np.zeros(3)
    residual = np.zeros(len(source_xyz))
    for _ in range(4):
        if int(keep.sum()) < 100:
            return None
        rotation, translation, _ = _rigid_fit(source_xyz[keep], reference_xyz[keep])
        residual = np.linalg.norm(
            source_xyz @ rotation.T + translation - reference_xyz, axis=1
        )
        threshold = min(0.75, float(np.quantile(residual, 0.85)))
        keep = residual <= max(threshold, 0.05)
    forward_yaw = math.degrees(math.atan2(float(rotation[0, 2]), float(rotation[2, 2])))
    return {
        "rotation": rotation,
        "camera_center": translation,
        "forward_yaw_deg": wrap_degrees(forward_yaw),
        "median_residual_m": float(np.median(residual[keep])),
        "p90_residual_m": float(np.quantile(residual[keep], 0.90)),
        "points": int(keep.sum()),
    }


def body_frame(points: np.ndarray, valid: np.ndarray) -> Dict[str, np.ndarray | float] | None:
    """Build an anatomical ground-plane frame from shoulders, hips and torso."""
    if not valid[[0, 4, 8, 12, 16, 20]].all():
        return None
    center = np.mean(points[[0, 1, 20]], axis=0)
    # Kinect's named anatomical joints resolve the front/back sign.  With this
    # release, up x anatomical-right points toward the actor's facing direction.
    right = 0.5 * ((points[8] - points[4]) + (points[16] - points[12]))
    up = points[20] - points[0]
    right[1] = 0.0
    right_norm = float(np.linalg.norm(right))
    if right_norm < 1e-5:
        return None
    right /= right_norm
    forward = np.cross(up, right)
    forward[1] = 0.0
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm < 1e-5:
        return None
    forward /= forward_norm
    yaw = wrap_degrees(math.degrees(math.atan2(float(forward[0]), float(forward[2]))))
    return {"center": center, "right": right, "forward": forward, "yaw_deg": yaw}


class BodyRelativeResolver:
    """Resolve one synchronized A/P/G clip and all of its temporal windows."""

    def __init__(self) -> None:
        self.base_sample: str | None = None
        self.reference_camera: str | None = None
        self.reference: Dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.transforms: Dict[str, Dict[str, Any] | None] = {}

    def _prepare(self, episode: Mapping[str, Any]) -> None:
        base = str(episode["base_sample"])
        if base == self.base_sample:
            return
        view_list = list(episode["views"])
        views = {str(view["camera"]): view for view in view_list}
        reference_view = views.get("C001", view_list[0])
        reference_camera = str(reference_view["camera"])
        reference = load_skeleton_xyz(Path(reference_view["skeleton"]))
        transforms: Dict[str, Dict[str, Any] | None] = {
            reference_camera: {
                "rotation": np.eye(3),
                "camera_center": np.zeros(3),
                "forward_yaw_deg": 0.0,
                "median_residual_m": 0.0,
                "p90_residual_m": 0.0,
                "points": 0,
            }
        }
        for camera, view in views.items():
            if camera == reference_camera:
                continue
            transforms[camera] = fit_camera_to_reference(
                reference, load_skeleton_xyz(Path(view["skeleton"]))
            )
        self.base_sample = base
        self.reference_camera = reference_camera
        self.reference = reference
        self.transforms = transforms

    def resolve(self, episode: Mapping[str, Any]) -> Dict[str, Any]:
        self._prepare(episode)
        frame_ids = [int(frame) + 1 for frame in episode["window"]["sample_frames"]]
        body_frames = []
        for frame_id in frame_ids:
            item = self.reference.get(frame_id) or self.reference.get(frame_id - 1)
            if item is not None:
                frame = body_frame(*item)
                if frame is not None:
                    body_frames.append(frame)
        if body_frames:
            body_yaw, body_mad, body_inliers = circular_cluster_median(
                [float(frame["yaw_deg"]) for frame in body_frames]
            )
            usable_fraction = len(body_frames) / max(len(frame_ids), 1)
            body_confidence = float(
                np.clip(usable_fraction * math.exp(-body_mad / 30.0), 0.0, 1.0)
            )
        else:
            body_yaw, body_mad, body_inliers, body_confidence = 0.0, 180.0, 0, 0.0

        view_results = []
        for view in episode["views"]:
            camera = str(view["camera"])
            transform = self.transforms.get(camera)
            bearings = []
            source = "skeleton_camera_center"
            if transform is not None:
                camera_center = np.asarray(transform["camera_center"], dtype=np.float64)
                for frame in body_frames:
                    direction = camera_center - np.asarray(frame["center"], dtype=np.float64)
                    direction[1] = 0.0
                    norm = float(np.linalg.norm(direction))
                    if norm > 1e-5:
                        direction /= norm
                        bearings.append(
                            wrap_degrees(
                                math.degrees(
                                    math.atan2(
                                        float(np.dot(direction, frame["right"])),
                                        float(np.dot(direction, frame["forward"])),
                                    )
                                )
                            )
                        )
            else:
                camera_center = np.full(3, np.nan)

            if not bearings and body_frames:
                # Explicit low-confidence fallback: assume the camera optical
                # axis points toward the actor, then reverse it to actor->camera.
                source = "optical_axis_fallback"
                camera_yaw = wrap_degrees(float(view.get("angle_deg", 0.0)) + 180.0)
                direction = np.asarray(
                    [math.sin(math.radians(camera_yaw)), 0.0, math.cos(math.radians(camera_yaw))]
                )
                for frame in body_frames:
                    bearings.append(
                        wrap_degrees(
                            math.degrees(
                                math.atan2(
                                    float(np.dot(direction, frame["right"])),
                                    float(np.dot(direction, frame["forward"])),
                                )
                            )
                        )
                    )
            if bearings:
                bearing, bearing_mad, bearing_inliers = circular_cluster_median(bearings)
            else:
                bearing, bearing_mad, bearing_inliers = 0.0, 180.0, 0
                source = "unavailable"
            if transform is None:
                transform_confidence = 0.35 if source == "optical_axis_fallback" else 0.0
                fit_median = float("nan")
                fit_p90 = float("nan")
            else:
                fit_median = float(transform["median_residual_m"])
                fit_p90 = float(transform["p90_residual_m"])
                transform_confidence = float(np.clip(math.exp(-fit_median / 0.35), 0.2, 1.0))
            confidence = float(np.clip(body_confidence * transform_confidence, 0.0, 1.0))
            label4, label4_zh, label8, label8_zh = direction_labels(bearing)
            view_results.append({
                "camera": camera,
                "relative_bearing_deg": float(bearing),
                "bearing_mad_deg": float(bearing_mad),
                "bearing_inlier_frames": int(bearing_inliers),
                "direction_4": label4,
                "direction_4_zh": label4_zh,
                "direction_8": label8,
                "direction_8_zh": label8_zh,
                "camera_center_x_m": float(camera_center[0]),
                "camera_center_y_m": float(camera_center[1]),
                "camera_center_z_m": float(camera_center[2]),
                "fit_median_residual_m": fit_median,
                "fit_p90_residual_m": fit_p90,
                "geometry_confidence": confidence,
                "geometry_source": source,
            })
        # Ascending signed bearing: left -> front -> right -> back wrap.  Keep
        # this metadata only for reporting; model tensor rows remain camera IDs.
        ordered = sorted(view_results, key=lambda result: result["relative_bearing_deg"])
        rank = {result["camera"]: index for index, result in enumerate(ordered)}
        for result in view_results:
            result["signed_bearing_rank"] = rank[result["camera"]]
        return {
            "body_yaw_deg_c001": float(body_yaw),
            "body_yaw_mad_deg": float(body_mad),
            "body_yaw_inlier_frames": int(body_inliers),
            "body_yaw_confidence": float(body_confidence),
            "camera_order_by_signed_bearing": [result["camera"] for result in ordered],
            "views": view_results,
        }


SIDECAR_FIELDS = (
    "split", "episode_id", "base_sample", "window_index", "camera",
    "relative_bearing_deg", "direction_4", "direction_4_zh", "direction_8",
    "direction_8_zh", "signed_bearing_rank", "body_yaw_deg_c001",
    "body_yaw_confidence", "geometry_confidence", "geometry_source",
    "camera_center_x_m", "camera_center_y_m", "camera_center_z_m",
    "bearing_mad_deg", "fit_median_residual_m", "fit_p90_residual_m",
)


def _sidecar_rows(split: str, episode: Mapping[str, Any], result: Mapping[str, Any]) -> Iterable[Dict[str, Any]]:
    for view in result["views"]:
        yield {
            "split": split,
            "episode_id": episode["episode_id"],
            "base_sample": episode["base_sample"],
            "window_index": episode["window"]["window_index"],
            "camera": view["camera"],
            "relative_bearing_deg": f"{view['relative_bearing_deg']:.3f}",
            "direction_4": view["direction_4"],
            "direction_4_zh": view["direction_4_zh"],
            "direction_8": view["direction_8"],
            "direction_8_zh": view["direction_8_zh"],
            "signed_bearing_rank": view["signed_bearing_rank"],
            "body_yaw_deg_c001": f"{result['body_yaw_deg_c001']:.3f}",
            "body_yaw_confidence": f"{result['body_yaw_confidence']:.4f}",
            "geometry_confidence": f"{view['geometry_confidence']:.4f}",
            "geometry_source": view["geometry_source"],
            "camera_center_x_m": f"{view['camera_center_x_m']:.5f}",
            "camera_center_y_m": f"{view['camera_center_y_m']:.5f}",
            "camera_center_z_m": f"{view['camera_center_z_m']:.5f}",
            "bearing_mad_deg": f"{view['bearing_mad_deg']:.3f}",
            "fit_median_residual_m": f"{view['fit_median_residual_m']:.5f}",
            "fit_p90_residual_m": f"{view['fit_p90_residual_m']:.5f}",
        }


def apply_to_cache(cache_root: Path, split: str) -> Path:
    """Rewrite only geometry/cost/quality metadata; preserve visual features."""
    root = cache_root / split
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    episodes = metadata["episodes"]
    geometry = np.array(np.load(root / "view_geometry.npy", mmap_mode="r"), copy=True)
    movement = np.array(np.load(root / "move_cost.npy", mmap_mode="r"), copy=True)
    quality = np.array(np.load(root / "image_quality.npy", mmap_mode="r"), copy=True)
    resolver = BodyRelativeResolver()
    rows = []
    for index, episode in enumerate(episodes):
        result = resolver.resolve(episode)
        angles = np.asarray([view["relative_bearing_deg"] for view in result["views"]])
        radians = np.radians(angles)
        geometry[index, :, 0] = np.sin(radians)
        geometry[index, :, 1] = np.cos(radians)
        for source in range(4):
            for destination in range(4):
                movement[index, source, destination] = abs(
                    wrap_degrees(angles[destination] - angles[source])
                ) / 180.0
        quality[index, :, 3] = float(result["body_yaw_confidence"])
        episode["body_relative_geometry"] = {
            key: value for key, value in result.items() if key != "views"
        }
        for original, resolved in zip(episode["views"], result["views"]):
            original.update(resolved)
        rows.extend(_sidecar_rows(split, episode, result))
        if (index + 1) % 500 == 0 or index + 1 == len(episodes):
            print(f"body geometry {split}: {index + 1}/{len(episodes)}", flush=True)

    if not np.isfinite(geometry).all() or not np.isfinite(movement).all() or not np.isfinite(quality).all():
        raise RuntimeError(f"non-finite rewritten arrays for {split}")
    for name in ("view_geometry.npy", "move_cost.npy", "image_quality.npy", "metadata.json"):
        source = root / name
        backup = root / f"{source.stem}.pre_body_relative{source.suffix}"
        if not backup.exists():
            shutil.copy2(source, backup)
    for name, array in (
        ("view_geometry.npy", geometry),
        ("move_cost.npy", movement),
        ("image_quality.npy", quality),
    ):
        temporary = root / f"{Path(name).stem}.body_relative.tmp.npy"
        np.save(temporary, array)
        temporary.replace(root / name)
    metadata["geometry_contract"] = {
        "definition": "candidate camera center bearing relative to anatomical body forward",
        "sign": "negative=body-left, zero=front, positive=body-right",
        "tensor": "view_geometry[...,0:2]=[sin(delta),cos(delta)]",
        "confidence": "image_quality[...,3]=body_yaw_confidence",
        "camera_rows": list(LOW_CAMERAS),
        "camera_id_is_not_angular_order": True,
    }
    temporary_meta = root / "metadata.body_relative.tmp.json"
    temporary_meta.write_text(
        json.dumps(json_safe(metadata), ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary_meta.replace(metadata_path)
    sidecar = root / "body_relative_directions.csv"
    with sidecar.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SIDECAR_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return sidecar


def catalog_manifest(manifest: Path, output: Path, split: str) -> Path:
    episodes = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    resolver = BodyRelativeResolver()
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SIDECAR_FIELDS)
        writer.writeheader()
        for index, episode in enumerate(episodes):
            writer.writerows(_sidecar_rows(split, episode, resolver.resolve(episode)))
            if (index + 1) % 500 == 0 or index + 1 == len(episodes):
                print(f"body geometry catalog {split}: {index + 1}/{len(episodes)}", flush=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config_rl.yaml")))
    parser.add_argument("--splits", nargs="+", default=("dqn_train", "val"))
    parser.add_argument("--catalog-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cache_root_value = Path(config["cache"]["root"])
    cache_root = (
        cache_root_value.resolve()
        if cache_root_value.is_absolute()
        else (config_path.parent / cache_root_value).resolve()
    )
    for split in args.splits:
        root = cache_root / split
        if args.catalog_only or not (root / "metadata.json").exists():
            catalog_manifest(root / "manifest.jsonl", root / "body_relative_directions.csv", split)
        else:
            print(apply_to_cache(cache_root, split), flush=True)


if __name__ == "__main__":
    main()
