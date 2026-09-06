#!/usr/bin/env python
"""Create CLIPGCN joint-xy arrays for action-detail raw tensor splits.

The action-detail raw tensor cache is built for X3D/CTR-GCN training. CLIPGCN
alignment also needs per-frame 2D joint coordinates. Windowed samples can reuse
the existing window caches; full-video samples are reconstructed from the
original skeleton CSVs with the same uniform-13 sampling used by the raw cache.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


WINDOW_NAME_RE = re.compile(r"__w\d+_f\d+-\d+")
SPECIAL_LABELS = {34, 37, 45, 46, 47}  # A035, A038, A046, A047, A048.


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", default="/workspace/X3D/data/clipgcn_action_detail_raw_13f_zsl_50_5")
    parser.add_argument("--src-root", default="/workspace/CLIPGCN/data")
    parser.add_argument("--short-window-dir", default="/workspace/CLIPGCN/data/windowed_2s1s_13f")
    parser.add_argument("--special-window-dir", default="/workspace/CLIPGCN/data/windowed_start_anchor_2s1s_13f")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test", "test_seen"])
    parser.add_argument("--frames", type=int, default=13)
    parser.add_argument("--depth-width", type=float, default=512.0)
    parser.add_argument("--depth-height", type=float, default=424.0)
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def csv_name_from_video(name: str) -> str:
    return str(Path(name).with_suffix(".csv"))


def joint_columns(header, joints=25):
    index = {name: idx for idx, name in enumerate(header)}
    xyz_columns = [
        (index[f"joint{joint}_3dX"], index[f"joint{joint}_3dY"], index[f"joint{joint}_3dZ"])
        for joint in range(1, joints + 1)
    ]
    depth_columns = [
        (index[f"joint{joint}_depthX"], index[f"joint{joint}_depthY"])
        for joint in range(1, joints + 1)
    ]
    state_columns = [index.get(f"joint{joint}_trackingState") for joint in range(1, joints + 1)]
    return index["frameNum"], index.get("trackingID"), index.get("bodyindexID"), xyz_columns, depth_columns, state_columns


def read_main_track(csv_path: Path, joints=25):
    tracks = defaultdict(dict)
    tracked_score = Counter()
    with csv_path.open("r", newline="", errors="replace") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        frame_col, tracking_col, body_col, xyz_cols, depth_cols, state_cols = joint_columns(header, joints)

        for row in reader:
            if len(row) < len(header):
                continue
            try:
                frame = int(float(row[frame_col])) - 1
            except ValueError:
                continue
            if frame < 0:
                continue

            if tracking_col is not None and row[tracking_col]:
                track_id = row[tracking_col]
            elif body_col is not None and row[body_col]:
                track_id = f"body_{row[body_col]}"
            else:
                track_id = "single"

            xyz = np.zeros((joints, 3), dtype=np.float32)
            depth_xy = np.zeros((joints, 2), dtype=np.float32)
            valid_value = False
            for joint_idx, (xyz_col, depth_col) in enumerate(zip(xyz_cols, depth_cols)):
                try:
                    xyz[joint_idx] = (float(row[xyz_col[0]]), float(row[xyz_col[1]]), float(row[xyz_col[2]]))
                    depth_xy[joint_idx] = (float(row[depth_col[0]]), float(row[depth_col[1]]))
                except ValueError:
                    pass
                valid_value = valid_value or bool(np.any(xyz[joint_idx]) or np.any(depth_xy[joint_idx]))
                state_col = state_cols[joint_idx]
                if state_col is not None:
                    try:
                        tracked_score[track_id] += int(float(row[state_col]))
                    except ValueError:
                        pass
            if valid_value:
                tracks[track_id][frame] = (xyz, depth_xy)

    if not tracks:
        raise ValueError(f"No valid skeleton track in {csv_path}")

    def rank_track(item):
        track_id, frames = item
        return len(frames), tracked_score[track_id], str(track_id)

    _track_id, frame_to_values = max(tracks.items(), key=rank_track)
    return frame_to_values


def nearest_frame(frame_to_values, frame_idx):
    if frame_idx in frame_to_values:
        return frame_to_values[frame_idx]
    nearest = min(frame_to_values, key=lambda item: abs(item - frame_idx))
    return frame_to_values[nearest]


def normalize_depth_xy(depth_xy, width, height):
    out = np.nan_to_num(depth_xy.copy(), nan=0.0, posinf=width - 1.0, neginf=0.0)
    out[..., 0] = (out[..., 0] / (width - 1.0)) * 2.0 - 1.0
    out[..., 1] = (out[..., 1] / (height - 1.0)) * 2.0 - 1.0
    return np.clip(out, -1.0, 1.0)


def video_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError(f"Cannot read frame count from {path}")
        return total_frames
    finally:
        cap.release()


def is_window_name(name: str) -> bool:
    return WINDOW_NAME_RE.search(Path(name).stem) is not None


def load_window_lookup(directory: Path, splits: list[str]):
    lookup = {}
    arrays = []
    for split in splits:
        names_path = directory / f"{split}_sample_names.npy"
        joint_path = directory / f"{split}_joint_xy.npy"
        if not names_path.exists() or not joint_path.exists():
            continue
        names = np.load(names_path, allow_pickle=True)
        joint_xy = np.load(joint_path, mmap_mode="r")
        arrays.append(joint_xy)
        for row, name in enumerate(names):
            lookup.setdefault(str(name), (joint_xy, int(row)))
    return lookup


class FullVideoJointXYBuilder:
    def __init__(self, src_root: Path, frames: int, depth_width: float, depth_height: float):
        self.src_root = src_root
        self.frames = frames
        self.depth_width = depth_width
        self.depth_height = depth_height
        self._track_cache = {}
        self._video_info_cache = {}

    def build(self, rel_video: str) -> np.ndarray:
        rel_csv = csv_name_from_video(rel_video)
        csv_path = self.src_root / rel_csv
        video_path = self.src_root / rel_video
        if rel_csv not in self._track_cache:
            self._track_cache[rel_csv] = read_main_track(csv_path)
        if rel_video not in self._video_info_cache:
            self._video_info_cache[rel_video] = video_frame_count(video_path)

        total_frames = self._video_info_cache[rel_video]
        sample_frames = np.linspace(0, total_frames - 1, self.frames).round().astype(np.int64)
        depth_xy = []
        for frame_idx in sample_frames:
            _xyz, frame_depth_xy = nearest_frame(self._track_cache[rel_csv], int(frame_idx))
            depth_xy.append(frame_depth_xy)
        return normalize_depth_xy(np.stack(depth_xy, axis=0), self.depth_width, self.depth_height)


def load_labels(target_dir: Path, split: str) -> np.ndarray:
    return np.load(target_dir / f"{split}_labels.npy", mmap_mode="r")


def load_names(target_dir: Path, split: str) -> np.ndarray:
    return np.load(target_dir / f"{split}_sample_names.npy", allow_pickle=True)


def write_split(split: str, args, short_lookup, special_lookup, full_builder):
    target_dir = Path(args.target_dir)
    output_path = target_dir / f"{split}_joint_xy.npy"
    if output_path.exists() and not args.overwrite:
        print(f"Skipping existing {output_path}", flush=True)
        return

    labels = load_labels(target_dir, split)
    names = load_names(target_dir, split)
    dtype = np.float32 if args.dtype == "float32" else np.float16
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=dtype,
        shape=(len(names), args.frames, 25, 2),
    )

    source_counts = {"full": 0, "short": 0, "special": 0}
    for row, name_value in enumerate(names):
        name = str(name_value)
        label = int(labels[row])
        if is_window_name(name):
            lookup = special_lookup if label in SPECIAL_LABELS else short_lookup
            fallback_lookup = short_lookup if lookup is special_lookup else special_lookup
            entry = lookup.get(name) or fallback_lookup.get(name)
            if entry is None:
                raise KeyError(f"{split}:{row} window sample was not found in window caches: {name}")
            source_array, source_row = entry
            output[row] = source_array[source_row].astype(dtype, copy=False)
            source_counts["special" if label in SPECIAL_LABELS else "short"] += 1
        else:
            output[row] = full_builder.build(name).astype(dtype, copy=False)
            source_counts["full"] += 1

    del output
    print(f"Wrote {output_path}: {len(names)} rows, sources={source_counts}", flush=True)


def main():
    args = parse_args()
    target_dir = Path(args.target_dir)
    if not target_dir.exists():
        raise FileNotFoundError(f"Target directory does not exist: {target_dir}")

    lookup_splits = ["train", "val", "test"]
    short_lookup = load_window_lookup(Path(args.short_window_dir), lookup_splits)
    special_lookup = load_window_lookup(Path(args.special_window_dir), lookup_splits)
    full_builder = FullVideoJointXYBuilder(
        src_root=Path(args.src_root),
        frames=args.frames,
        depth_width=args.depth_width,
        depth_height=args.depth_height,
    )

    for split in args.splits:
        write_split(split, args, short_lookup, special_lookup, full_builder)


if __name__ == "__main__":
    main()
