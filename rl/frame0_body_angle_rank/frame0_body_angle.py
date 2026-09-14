"""Resolve the per-video frame-0 human-relative camera angle.

The required angle is measured in each camera's own Kinect coordinate frame:

    body-relative bearing = angle(body-forward, person->camera)

The body forward/right frame comes from the 25-joint 3-D skeleton.  Since the
camera origin is (0, 0, 0) in that video's skeleton coordinate frame, this
does not assume that C001/C003/C005/C007 are in angular filename order and it
also works for the merged A049/A050 groups.  A missing skeleton is an error;
the historical optical-axis CSV is intentionally never used as a fallback.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


LOW_CAMERAS = ("C001", "C003", "C005", "C007")
REQUIRED_JOINTS = (0, 1, 4, 8, 12, 16, 20)


class MissingSkeletonError(FileNotFoundError):
    """Raised when an exact anatomical angle cannot be computed."""


def wrap_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def load_angle_csv(path: Path) -> dict[str, dict[str, str]]:
    """Load the existing CSV only as a path index, never as an angle source."""

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "sample_id" not in reader.fieldnames:
            raise ValueError(f"angle index has no sample_id column: {path}")
        rows = {str(row["sample_id"]): dict(row) for row in reader}
    return rows


def load_skeleton_xyz(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Select the rightmost body by depth-image torso X at the first frame."""

    frames: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    scores: dict[int, int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            try:
                frame = int(row[0])
                if frames and frame > min(frames):
                    break
                points = np.full((25, 3), np.nan, dtype=np.float64)
                valid = np.zeros(25, dtype=bool)
                for joint in range(25):
                    offset = 3 + joint * 10
                    xyz = np.asarray(row[offset : offset + 3], dtype=np.float64)
                    tracking = float(row[offset + 9])
                    if xyz.shape == (3,) and np.isfinite(xyz).all() and tracking > 0.0:
                        points[joint] = xyz
                        valid[joint] = True
            except (IndexError, TypeError, ValueError):
                continue
            if not valid.all() or body_frame(points, valid) is None:
                continue
            image_x = [float(row[3 + j * 10 + 3]) for j in (0, 1, 20)]
            score = float(np.mean(image_x))
            if not np.isfinite(score):
                continue
            if score > scores.get(frame, -float('inf')):
                frames[frame] = (points, valid)
                scores[frame] = score
    return frames


def body_frame(points: np.ndarray, valid: np.ndarray) -> dict[str, Any] | None:
    """Build the anatomical ground-plane frame used by the prior ETRI code."""

    if points.shape != (25, 3) or valid.shape != (25,):
        return None
    if not valid[list(REQUIRED_JOINTS)].all():
        return None

    # Joint layout follows the existing ETRI Kinect loader in this repository:
    # 0/1/20 define torso/head center, 4/8 shoulders, 12/16 hips.
    center = np.mean(points[[0, 1, 20]], axis=0)
    right = 0.5 * ((points[8] - points[4]) + (points[16] - points[12]))
    up = points[20] - points[0]
    right[1] = 0.0
    right_norm = float(np.linalg.norm(right))
    if right_norm < 1e-5:
        return None
    right = right / right_norm

    forward = np.cross(up, right)
    forward[1] = 0.0
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm < 1e-5:
        return None
    forward = forward / forward_norm
    return {
        "center": center,
        "right": right,
        "forward": forward,
        "forward_yaw_deg_in_camera": wrap_degrees(
            math.degrees(math.atan2(float(forward[0]), float(forward[2])))
        ),
    }


def first_body_frame(
    frames: Mapping[int, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, Any], int, str]:
    """Return the video frame-0 body frame and the source row index.

    ETRI releases have appeared with either zero-based or one-based skeleton
    row numbering.  We prefer row 0, then row 1 only when row 0 is absent or
    unusable, and record that fallback in the audit output.
    """

    for frame_id in sorted(frames):
        indexing = 'earliest_complete_valid_frame'
        item = frames.get(frame_id)
        if item is None:
            continue
        result = body_frame(*item)
        if result is not None:
            return result, frame_id, indexing
    raise ValueError("no valid anatomical body frame at skeleton row 0/1")


def body_relative_camera_angle(body: Mapping[str, Any]) -> float:
    """Return signed angle from human forward to the camera direction.

    Positive means body-right; negative means body-left.  The vector from the
    body center to the camera origin is ``-center`` in a camera-local frame.
    """

    direction = -np.asarray(body["center"], dtype=np.float64).copy()
    direction[1] = 0.0
    norm = float(np.linalg.norm(direction))
    if norm < 1e-5:
        raise ValueError("camera and body center are coincident in skeleton frame")
    direction /= norm
    right = np.asarray(body["right"], dtype=np.float64)
    forward = np.asarray(body["forward"], dtype=np.float64)
    return wrap_degrees(
        math.degrees(
            math.atan2(float(np.dot(direction, right)), float(np.dot(direction, forward)))
        )
    )


def skeleton_path_for_view(
    view: Mapping[str, Any],
    angle_rows: Mapping[str, Mapping[str, str]],
) -> Path:
    """Resolve one video to its corresponding released skeleton CSV."""

    sample_name = str(view.get("sample_name") or view.get("video") or "")
    sample_id = Path(sample_name).stem
    row = angle_rows.get(sample_id)
    if row is None:
        raise MissingSkeletonError(f"no angle-index row for video {sample_id!r}")
    raw = str(row.get("skeleton_path") or "").strip()
    if not raw:
        raise MissingSkeletonError(f"empty skeleton path for {sample_id}")
    path = Path(raw)
    if not path.exists():
        raise MissingSkeletonError(
            f"missing 3-D skeleton for {sample_id}: {path}; "
            "the frame-0 body-angle experiment refuses CSV optical-angle fallback"
        )
    return path


def resolve_view(
    view: Mapping[str, Any],
    angle_rows: Mapping[str, Mapping[str, str]],
    skeleton_cache: dict[Path, dict[int, tuple[np.ndarray, np.ndarray]]],
) -> dict[str, Any]:
    """Resolve one metadata view with an exact frame-0 3-D body angle."""

    path = skeleton_path_for_view(view, angle_rows)
    if path not in skeleton_cache:
        skeleton_cache[path] = load_skeleton_xyz(path)
    frames = skeleton_cache[path]
    try:
        body, source_frame, indexing = first_body_frame(frames)
    except ValueError as exc:
        raise ValueError(f'{path}: {exc}') from exc
    angle = body_relative_camera_angle(body)
    sample_id = Path(str(view.get("sample_name") or view.get("video"))).stem
    return {
        "sample_id": sample_id,
        "camera": str(view.get("camera", "")),
        "skeleton_path": str(path),
        "angle_deg": float(angle),
        "angle_source": "frame0_3d_anatomical_body_forward_to_camera",
        "skeleton_frame_row": int(source_frame),
        "skeleton_indexing": indexing,
        "body_forward_yaw_in_camera_deg": float(body["forward_yaw_deg_in_camera"]),
        "body_center_camera_xyz": [float(x) for x in np.asarray(body["center"])],
    }


def resolve_episode(
    episode: Mapping[str, Any],
    angle_rows: Mapping[str, Mapping[str, str]],
    source_valid: np.ndarray,
    episode_index: int,
    skeleton_cache: dict[Path, dict[int, tuple[np.ndarray, np.ndarray]]],
) -> dict[str, Any]:
    """Resolve and sort one episode from negative angle to positive angle."""

    views = list(episode.get("views", []))
    resolved: list[dict[str, Any]] = []
    for original_index, view in enumerate(views):
        if original_index >= len(source_valid) or not bool(source_valid[original_index]):
            continue
        try:
            item = resolve_view(view, angle_rows, skeleton_cache)
        except ValueError as exc:
            print(json.dumps({'unavailable_view': str(view.get('sample_name')), 'episode': episode_index, 'reason': str(exc)}), flush=True)
            continue
        item["original_view_index"] = int(original_index)
        item["original_camera"] = str(view.get("camera", ""))
        item["video"] = str(view.get("video", ""))
        item["sample_name"] = str(view.get("sample_name", ""))
        resolved.append(item)

    # If metadata omitted a view entry but the source mask says it is valid,
    # fail loudly rather than silently changing the cache's index semantics.
    if int(np.asarray(source_valid, dtype=bool).sum()) > len(views):
        raise ValueError(
            f"episode {episode_index} metadata/mask mismatch: "
            f"mask_valid={int(np.asarray(source_valid, dtype=bool).sum())}, "
            f"resolved={len(resolved)}"
        )

    resolved.sort(key=lambda item: (float(item["angle_deg"]), int(item["original_view_index"])))
    for rank, item in enumerate(resolved):
        item["angle_rank"] = int(rank)
        item["action_index"] = int(rank)
        item["angle_rank_definition"] = "ascending signed body-relative bearing"

    rank_to_original = [-1, -1, -1, -1]
    original_to_rank = [-1, -1, -1, -1]
    angle_by_rank = [None, None, None, None]
    for item in resolved:
        rank = int(item["angle_rank"])
        original = int(item["original_view_index"])
        rank_to_original[rank] = original
        original_to_rank[original] = rank
        angle_by_rank[rank] = float(item["angle_deg"])

    return {
        "episode_index": int(episode_index),
        "base_sample": str(episode.get("base_sample", "")),
        "views": resolved,
        "rank_to_original": rank_to_original,
        "original_to_rank": original_to_rank,
        "angle_deg_by_rank": angle_by_rank,
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "split", "episode_index", "base_sample", "angle_rank", "action_index",
        "original_view_index", "original_camera", "sample_id", "angle_deg",
        "angle_source", "skeleton_path", "skeleton_frame_row", "skeleton_indexing",
        "body_forward_yaw_in_camera_deg",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False), encoding="utf-8")
