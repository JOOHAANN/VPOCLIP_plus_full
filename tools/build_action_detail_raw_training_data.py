#!/usr/bin/env python
"""Build action-detail raw training data for X3D and CTR-GCN.

This script creates model-training inputs, not CLIPGCN fusion features:

* X3D: ``train_float16.npy``, ``val_float16.npy``, ``test_float16.npy`` plus
  labels/manifests in a CLIPGCN tensor-data directory.
* CTR-GCN: one ``.npz`` with ``x_train/y_train/x_val/y_val/x_test/y_test``.

The action source is selected from ``Action details collection.xlsx``:
L actions use the original full-video 13-frame samples; S actions use 2s/1s
window samples; special start-anchor actions use a separate S cache whose first
sampled frame is the first frame of the original full video.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import cv2
import numpy as np

CLIPGCN_ROOT = Path(__file__).resolve().parents[1]
if str(CLIPGCN_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(CLIPGCN_ROOT / "tools"))

from build_windowed_clipgcn_dataset import nearest_frame, read_main_track  # noqa: E402


SPLITS = ("train", "val", "test")
SPECIAL_ACTIONS = ("A035", "A038", "A046", "A047", "A048")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx", default="/workspace/CLIPGCN/Action details collection.xlsx")
    parser.add_argument("--src-root", default="/workspace/CLIPGCN/data")
    parser.add_argument("--full-x3d-dir", default="/workspace/X3D/data/clipgcn_tensor_cs_70_10_20")
    parser.add_argument("--short-window-dir", default="/workspace/CLIPGCN/data/windowed_2s1s_13f")
    parser.add_argument("--special-window-dir", default="/workspace/CLIPGCN/data/windowed_start_anchor_2s1s_13f")
    parser.add_argument("--x3d-out-dir", default="/workspace/X3D/data/clipgcn_action_detail_raw_13f")
    parser.add_argument("--ctrgcn-out", default="/workspace/CTR-GCN/data/etri/ETRI_P1_P230_action_detail_raw_13f.npz")
    parser.add_argument("--frames", type=int, default=13)
    parser.add_argument("--size", type=int, default=160)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--num-class", type=int, default=55)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_action_id(value: str) -> str:
    match = re.fullmatch(r"A0*(\d+)", str(value).strip().upper())
    if match is None:
        raise ValueError(f"Invalid action id: {value}")
    return f"A{int(match.group(1)):03d}"


def action_to_label(action: str) -> int:
    return int(normalize_action_id(action)[1:]) - 1


def label_to_action(label: int) -> str:
    return f"A{int(label) + 1:03d}"


def action_from_path(path: str | Path) -> str:
    match = re.search(r"A0*(\d+)", Path(path).name)
    if match is None:
        raise ValueError(f"Cannot parse action from {path}")
    return f"A{int(match.group(1)):03d}"


def label_from_name(name: str | Path) -> int:
    return action_to_label(action_from_path(name))


def csv_name_from_video(rel_video: str) -> str:
    return str(Path(rel_video).with_suffix(".csv"))


def subject_from_name(name: str | Path) -> str:
    for part in Path(str(name)).parts:
        if re.fullmatch(r"P\d+", part):
            return part
    match = re.search(r"P(\d+)", str(name))
    if match:
        return f"P{int(match.group(1)):02d}" if int(match.group(1)) < 200 else f"P{int(match.group(1))}"
    return "P000"


def one_hot(labels: np.ndarray, num_class: int) -> np.ndarray:
    y = np.zeros((len(labels), num_class), dtype=np.float32)
    if len(labels):
        y[np.arange(len(labels)), labels.astype(np.int64)] = 1.0
    return y


def xlsx_cell_text(cell, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    value = cell.find("{*}v")
    if cell_type == "s":
        return shared_strings[int(value.text)] if value is not None and value.text is not None else ""
    if cell_type == "inlineStr":
        return "".join(text.text or "" for text in cell.findall(".//{*}t")).strip()
    return value.text.strip() if value is not None and value.text is not None else ""


def read_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    return ["".join(text.text or "" for text in item.findall(".//{*}t")).strip() for item in root.findall("{*}si")]


def read_xlsx_rows(path: Path) -> Iterable[list[str]]:
    with zipfile.ZipFile(path) as archive:
        shared_strings = read_shared_strings(archive)
        sheet_names = sorted(name for name in archive.namelist() if name.startswith("xl/worksheets/sheet"))
        if not sheet_names:
            raise ValueError(f"No worksheet found in {path}")
        root = ET.fromstring(archive.read(sheet_names[0]))
        for row in root.findall(".//{*}sheetData/{*}row"):
            values = [xlsx_cell_text(cell, shared_strings) for cell in row.findall("{*}c")]
            if any(value.strip() for value in values):
                yield values


def load_action_rules(xlsx_path: Path) -> dict[int, str]:
    rules: dict[int, str] = {}
    for values in read_xlsx_rows(xlsx_path):
        action = None
        rule = None
        for value in values:
            stripped = value.strip()
            if re.fullmatch(r"A0*\d+", stripped, flags=re.IGNORECASE):
                action = normalize_action_id(stripped)
            if stripped.upper() in {"L", "S"}:
                rule = stripped.upper()
        if action is not None and rule is not None:
            rules[action_to_label(action)] = rule

    missing = sorted(set(range(55)) - set(rules))
    if missing:
        raise ValueError(f"Missing action rules for {[label_to_action(label) for label in missing]}")
    for action in SPECIAL_ACTIONS:
        rules[action_to_label(action)] = "SPECIAL"
    return rules


def read_manifest(path: Path) -> tuple[np.ndarray, np.ndarray]:
    names = []
    labels = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().rsplit(" ", 1)
            if len(parts) != 2:
                continue
            names.append(parts[0])
            labels.append(int(parts[1]))
    return np.asarray(names, dtype=object), np.asarray(labels, dtype=np.int64)


def reconstruct_full_x3d_split(full_dir: Path, split: str):
    metadata = json.loads((full_dir / "metadata.json").read_text(encoding="utf-8"))
    src_root = Path(metadata["source_root"])
    split_subjects = set(metadata["split_subjects"][split])
    excluded_actions = set(metadata.get("excluded_actions") or [])
    videos = sorted(
        path
        for path in src_root.glob("P*/**/*.mp4")
        if action_from_path(path) not in excluded_actions and path.relative_to(src_root).parts[0] in split_subjects
    )
    labels = np.load(full_dir / f"{split}_labels.npy", mmap_mode="r")
    if len(videos) != len(labels):
        raise RuntimeError(f"{split}: reconstructed {len(videos)} full videos but labels have {len(labels)} rows")
    names = np.asarray([str(path.relative_to(src_root)) for path in videos], dtype=object)
    inferred = np.asarray([label_from_name(name) for name in names], dtype=np.int64)
    valid = (labels[:] >= 0) & (labels[:] == inferred)
    return names, labels, valid


@dataclass(frozen=True)
class Selection:
    source: str
    rows: np.ndarray


class RawSource:
    def __init__(self, name: str, directory: Path, source_type: str):
        self.name = name
        self.directory = directory
        self.source_type = source_type

    def video(self, split: str):
        return np.load(self.directory / f"{split}_float16.npy", mmap_mode="r")

    def labels(self, split: str):
        return np.load(self.directory / f"{split}_labels.npy", mmap_mode="r")

    def names(self, split: str):
        if self.source_type == "full":
            return reconstruct_full_x3d_split(self.directory, split)[0]
        return np.load(self.directory / f"{split}_sample_names.npy", allow_pickle=True)

    def valid_rows(self, split: str):
        labels = self.labels(split)
        if self.source_type == "full":
            return reconstruct_full_x3d_split(self.directory, split)[2]
        names = self.names(split)
        inferred = np.asarray([label_from_name(name) for name in names], dtype=np.int64)
        return (labels[:] >= 0) & (labels[:] == inferred)

    def skeleton_x(self, split: str):
        if self.source_type == "full":
            raise RuntimeError("Full-video skeleton rows are generated from CSV on demand.")
        return np.load(self.directory / f"{split}_skeleton_x.npy", mmap_mode="r")


def rows_for_rule(source: RawSource, split: str, wanted_labels: set[int]) -> np.ndarray:
    labels = source.labels(split)
    valid = source.valid_rows(split)
    return np.flatnonzero(valid & np.isin(labels[:], list(wanted_labels))).astype(np.int64)


class FullSkeletonBuilder:
    def __init__(self, src_root: Path, frames: int):
        self.src_root = src_root
        self.frames = frames
        self._track_cache = {}

    def _total_frames(self, video_path: Path) -> int:
        cap = cv2.VideoCapture(str(video_path))
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                raise ValueError(f"Cannot read frame count from {video_path}")
            return total
        finally:
            cap.release()

    def build(self, rel_video: str) -> np.ndarray:
        rel_csv = csv_name_from_video(rel_video)
        csv_path = self.src_root / rel_csv
        if rel_csv not in self._track_cache:
            self._track_cache[rel_csv] = read_main_track(csv_path)
        frame_to_values = self._track_cache[rel_csv]
        total_frames = self._total_frames(self.src_root / rel_video)
        sample_frames = np.linspace(0, total_frames - 1, self.frames).round().astype(np.int64)

        xyz_frames = []
        for frame_idx in sample_frames:
            xyz, _depth_xy = nearest_frame(frame_to_values, int(frame_idx))
            xyz_frames.append(xyz)
        skeleton = np.zeros((self.frames, 2, 25, 3), dtype=np.float32)
        skeleton[:, 0, :, :] = np.stack(xyz_frames, axis=0)
        return skeleton.reshape(self.frames, 150)


def collect_split_plan(split: str, rules: dict[int, str], sources: dict[str, RawSource]) -> list[Selection]:
    l_labels = {label for label, rule in rules.items() if rule == "L"}
    s_labels = {label for label, rule in rules.items() if rule == "S"}
    special_labels = {label for label, rule in rules.items() if rule == "SPECIAL"}
    plan = [
        Selection("full", rows_for_rule(sources["full"], split, l_labels)),
        Selection("short", rows_for_rule(sources["short"], split, s_labels)),
        Selection("special", rows_for_rule(sources["special"], split, special_labels)),
    ]
    return [selection for selection in plan if len(selection.rows) > 0]


def write_split(split: str, plan: list[Selection], sources: dict[str, RawSource], args, skeleton_builder):
    out_dir = Path(args.x3d_out_dir)
    full_skeleton_rows: dict[int, np.ndarray] = {}
    skipped_rows = []
    validated_plan = []
    for selection in plan:
        source = sources[selection.source]
        names = source.names(split)
        valid_rows = []
        if selection.source == "full":
            for row in selection.rows:
                name = str(names[row])
                try:
                    full_skeleton_rows[int(row)] = skeleton_builder.build(name)
                    valid_rows.append(int(row))
                except Exception as exc:
                    skipped_rows.append(
                        {
                            "source": selection.source,
                            "row": int(row),
                            "sample_name": name,
                            "error": str(exc),
                        }
                    )
        else:
            valid_rows = [int(row) for row in selection.rows]
        if valid_rows:
            validated_plan.append(Selection(selection.source, np.asarray(valid_rows, dtype=np.int64)))

    plan = validated_plan
    total_rows = sum(len(selection.rows) for selection in plan)
    if total_rows == 0:
        raise RuntimeError(f"{split}: no rows selected")

    first_video = next(sources[selection.source].video(split) for selection in plan if len(selection.rows) > 0)
    video_out = np.lib.format.open_memmap(
        out_dir / f"{split}_float16.npy",
        mode="w+",
        dtype=first_video.dtype,
        shape=(total_rows,) + first_video.shape[1:],
    )
    labels_out = np.lib.format.open_memmap(out_dir / f"{split}_labels.npy", mode="w+", dtype=np.int64, shape=(total_rows,))
    skeleton_x = np.zeros((total_rows, args.frames, 150), dtype=np.float32)
    sample_names = []
    skeleton_names = []

    offset = 0
    for selection in plan:
        source = sources[selection.source]
        video = source.video(split)
        names = source.names(split)
        rows = selection.rows
        for start in range(0, len(rows), args.chunk_size):
            end = min(start + args.chunk_size, len(rows))
            dst_slice = slice(offset + start, offset + end)
            src_rows = rows[start:end]
            video_out[dst_slice] = video[src_rows]
            inferred_labels = np.asarray([label_from_name(names[row]) for row in src_rows], dtype=np.int64)
            labels_out[dst_slice] = inferred_labels

        if selection.source == "full":
            for local_idx, row in enumerate(rows):
                skeleton_x[offset + local_idx] = full_skeleton_rows[int(row)]
        else:
            source_skeleton = source.skeleton_x(split)
            for start in range(0, len(rows), args.chunk_size):
                end = min(start + args.chunk_size, len(rows))
                skeleton_x[offset + start : offset + end] = source_skeleton[rows[start:end]]

        sample_names.extend(str(names[row]) for row in rows)
        skeleton_names.extend(csv_name_from_video(str(names[row])) for row in rows)
        offset += len(rows)

    labels_array = np.asarray(labels_out[:], dtype=np.int64)
    del video_out, labels_out

    np.save(out_dir / f"{split}_sample_names.npy", np.asarray(sample_names, dtype=object))
    with (out_dir / f"{split}_manifest.txt").open("w", encoding="utf-8") as handle:
        for name, label in zip(sample_names, labels_array):
            handle.write(f"{name} {int(label)}\n")

    return {
        "x": skeleton_x,
        "y": one_hot(labels_array, args.num_class),
        "sample_name": np.asarray(skeleton_names, dtype=object),
        "subject": np.asarray([subject_from_name(name) for name in sample_names], dtype=object),
        "action": np.asarray([label_to_action(label) for label in labels_array], dtype=object),
        "frames": np.full(total_rows, args.frames, dtype=np.int64),
        "metadata": {
            "num_samples": int(total_rows),
            "by_source": {selection.source: int(len(selection.rows)) for selection in plan},
            "skipped_invalid_full_skeleton": skipped_rows,
            "num_skipped_invalid_full_skeleton": int(len(skipped_rows)),
            "class_ids": sorted(int(label) for label in np.unique(labels_array).tolist()),
        },
    }


def main():
    args = parse_args()
    x3d_out_dir = Path(args.x3d_out_dir)
    ctrgcn_out = Path(args.ctrgcn_out)
    if x3d_out_dir.exists() and any(x3d_out_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{x3d_out_dir} is not empty. Pass --overwrite to replace it.")
        shutil.rmtree(x3d_out_dir)
    x3d_out_dir.mkdir(parents=True, exist_ok=True)
    ctrgcn_out.parent.mkdir(parents=True, exist_ok=True)

    rules = load_action_rules(Path(args.xlsx))
    sources = {
        "full": RawSource("full", Path(args.full_x3d_dir), "full"),
        "short": RawSource("short", Path(args.short_window_dir), "window"),
        "special": RawSource("special", Path(args.special_window_dir), "window"),
    }
    skeleton_builder = FullSkeletonBuilder(Path(args.src_root), args.frames)

    split_payload = {}
    split_meta = {}
    for split in SPLITS:
        print(f"Planning/writing {split}", flush=True)
        plan = collect_split_plan(split, rules, sources)
        payload = write_split(split, plan, sources, args, skeleton_builder)
        split_payload[split] = payload
        split_meta[split] = payload["metadata"]
        print(json.dumps({split: payload["metadata"]}, ensure_ascii=False), flush=True)

    save_npz = np.savez_compressed
    save_npz(
        ctrgcn_out,
        x_train=split_payload["train"]["x"],
        y_train=split_payload["train"]["y"],
        train_sample_name=split_payload["train"]["sample_name"],
        train_subject=split_payload["train"]["subject"],
        train_action=split_payload["train"]["action"],
        train_frames=split_payload["train"]["frames"],
        x_val=split_payload["val"]["x"],
        y_val=split_payload["val"]["y"],
        val_sample_name=split_payload["val"]["sample_name"],
        val_subject=split_payload["val"]["subject"],
        val_action=split_payload["val"]["action"],
        val_frames=split_payload["val"]["frames"],
        x_test=split_payload["test"]["x"],
        y_test=split_payload["test"]["y"],
        test_sample_name=split_payload["test"]["sample_name"],
        test_subject=split_payload["test"]["subject"],
        test_action=split_payload["test"]["action"],
        test_frames=split_payload["test"]["frames"],
    )

    full_metadata = json.loads((Path(args.full_x3d_dir) / "metadata.json").read_text(encoding="utf-8"))
    metadata = {
        "name": "clipgcn_action_detail_raw_13f",
        "description": "Raw model-training data mixed from full-video L actions, 2s/1s S actions, and start-anchor special actions.",
        "xlsx": str(Path(args.xlsx)),
        "source_root": str(Path(args.src_root)),
        "full_x3d_dir": str(Path(args.full_x3d_dir)),
        "short_window_dir": str(Path(args.short_window_dir)),
        "special_window_dir": str(Path(args.special_window_dir)),
        "ctrgcn_npz": str(ctrgcn_out),
        "shape": {
            split: [split_meta[split]["num_samples"], 3, args.frames, args.size, args.size]
            for split in SPLITS
        },
        "classes": full_metadata["classes"],
        "class_to_idx": full_metadata["class_to_idx"],
        "split_rule": full_metadata.get("split_rule"),
        "split_mode": full_metadata.get("split_mode"),
        "split_seed": full_metadata.get("split_seed"),
        "split_subjects": full_metadata.get("split_subjects"),
        "cache_dtype": "float16",
        "action_rules": {label_to_action(label): rule for label, rule in sorted(rules.items())},
        "splits": split_meta,
    }
    (x3d_out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (ctrgcn_out.with_suffix("").as_posix() + "_metadata.json")
    Path(ctrgcn_out.with_suffix("").as_posix() + "_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
