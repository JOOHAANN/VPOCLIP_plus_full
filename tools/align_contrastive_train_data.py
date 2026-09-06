#!/usr/bin/env python
"""Prepare X3D-aligned trimodal contrastive training files.

The X3D tensor cache stores samples in the deterministic video-list order, while
its manifest was written in worker completion order. This script reconstructs
the real X3D order from metadata, filters out failed video decodes and skeleton
samples missing from the CTR-GCN npz, then writes aligned video/object files and
an aligned skeleton npz for CTR-GCN feature extraction.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Align X3D video, YOLO object, and CTR-GCN skeleton data.")
    parser.add_argument("--x3d-data-dir", default="/workspace/X3D/data/clipgcn_tensor_cs_70_10_20")
    parser.add_argument("--contrastive-dir", default="/workspace/CLIPGCN/data/contrastive_train_data")
    parser.add_argument("--skeleton-npz", default="/workspace/CTR-GCN/data/etri/ETRI_P1_P230_CS_raw_uniform13.npz")
    parser.add_argument(
        "--aligned-skeleton-npz",
        default="/workspace/CTR-GCN/data/etri/ETRI_P1_P230_CS_raw_uniform13_x3d_train_aligned.npz",
    )
    parser.add_argument("--video-feature", default="train_res5_model_007000.npy")
    parser.add_argument("--video-labels", default="train_res5_model_007000.labels.npy")
    parser.add_argument("--video-valid-indices", default="train_res5_model_007000.valid_indices.npy")
    parser.add_argument("--object-file", default="train_frame7_yolov5m_objects.npy")
    parser.add_argument("--prefix", default="aligned_tmp_", help="Temporary output prefix inside contrastive-dir.")
    return parser.parse_args()


def action_id(path):
    match = re.match(r"(A\d+)_", Path(path).name)
    if match is None:
        raise ValueError(f"Cannot parse action id from {path}")
    return match.group(1)


def participant_id(path, src_root):
    return Path(path).relative_to(src_root).parts[0]


def csv_name_from_video(rel_video):
    return str(Path(rel_video).with_suffix(".csv"))


def video_name_from_csv(rel_csv):
    return str(Path(rel_csv).with_suffix(".mp4"))


def reconstruct_train_videos(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    src_root = Path(metadata["source_root"])
    train_subjects = set(metadata["split_subjects"]["train"])
    excluded_actions = set(metadata.get("excluded_actions") or [])
    class_to_idx = metadata["class_to_idx"]

    videos = sorted(
        path
        for path in src_root.glob("P*/**/*.mp4")
        if action_id(path) not in excluded_actions and participant_id(path, src_root) in train_subjects
    )
    rel_videos = np.array([str(path.relative_to(src_root)) for path in videos], dtype=object)
    expected_labels = np.array([class_to_idx[action_id(path)] for path in videos], dtype=np.int64)
    return metadata, rel_videos, expected_labels


def skeleton_lookup(npz):
    names = np.concatenate([npz["train_sample_name"], npz["test_sample_name"]], axis=0)
    x = np.concatenate([npz["x_train"], npz["x_test"]], axis=0)
    y = np.concatenate([npz["y_train"], npz["y_test"]], axis=0)
    subjects = np.concatenate([npz["train_subject"], npz["test_subject"]], axis=0)
    actions = np.concatenate([npz["train_action"], npz["test_action"]], axis=0)
    frames = np.concatenate([npz["train_frames"], npz["test_frames"]], axis=0)
    return {video_name_from_csv(name): idx for idx, name in enumerate(names)}, {
        "names": names,
        "x": x,
        "y": y,
        "subjects": subjects,
        "actions": actions,
        "frames": frames,
    }


def copy_feature_rows(src_path, dst_path, rows):
    src = np.load(src_path, mmap_mode="r")
    dst = np.lib.format.open_memmap(dst_path, mode="w+", dtype=src.dtype, shape=(len(rows),) + src.shape[1:])
    chunk = 128
    for start in range(0, len(rows), chunk):
        end = min(start + chunk, len(rows))
        dst[start:end] = src[rows[start:end]]
    del dst


def filter_object_payload(src_path, dst_path, original_rows, aligned_rel_videos, labels):
    payload = np.load(src_path, allow_pickle=True).item()
    out = {}
    for key, value in payload.items():
        if isinstance(value, np.ndarray) and value.shape[:1] == (len(payload["presence"]),):
            out[key] = value[original_rows]
        else:
            out[key] = value
    out["manifest"] = np.array([[name, int(label)] for name, label in zip(aligned_rel_videos, labels)], dtype=object)
    out["aligned_sample_names"] = aligned_rel_videos.astype(object)
    out["alignment_note"] = "Filtered to X3D valid samples that also have skeleton CSV in CTR-GCN raw uniform13 npz."
    np.save(dst_path, out, allow_pickle=True)


def main():
    args = parse_args()
    x3d_data_dir = Path(args.x3d_data_dir)
    contrastive_dir = Path(args.contrastive_dir)
    contrastive_dir.mkdir(parents=True, exist_ok=True)

    metadata, rel_videos, expected_labels = reconstruct_train_videos(x3d_data_dir / "metadata.json")
    train_labels = np.load(x3d_data_dir / "train_labels.npy", mmap_mode="r")
    valid_indices = np.flatnonzero(train_labels >= 0)
    if not np.array_equal(expected_labels[valid_indices], train_labels[valid_indices]):
        raise RuntimeError("Reconstructed X3D video order does not match train_labels.npy")

    skeleton_npz = np.load(args.skeleton_npz, allow_pickle=True)
    skel_by_video, skel = skeleton_lookup(skeleton_npz)

    keep_valid_rows = []
    keep_original_rows = []
    keep_skeleton_rows = []
    missing_skeleton = []

    valid_rel_videos = rel_videos[valid_indices]
    for x3d_feature_row, original_row in enumerate(valid_indices):
        rel_video = str(rel_videos[original_row])
        skel_row = skel_by_video.get(rel_video)
        if skel_row is None:
            missing_skeleton.append(rel_video)
            continue
        keep_valid_rows.append(x3d_feature_row)
        keep_original_rows.append(int(original_row))
        keep_skeleton_rows.append(int(skel_row))

    keep_valid_rows = np.asarray(keep_valid_rows, dtype=np.int64)
    keep_original_rows = np.asarray(keep_original_rows, dtype=np.int64)
    keep_skeleton_rows = np.asarray(keep_skeleton_rows, dtype=np.int64)
    aligned_rel_videos = rel_videos[keep_original_rows]
    aligned_labels = train_labels[keep_original_rows].astype(np.int64)

    skeleton_labels = np.where(skel["y"][keep_skeleton_rows] > 0)[1]
    if not np.array_equal(aligned_labels, skeleton_labels):
        mismatch = np.flatnonzero(aligned_labels != skeleton_labels)[:10]
        raise RuntimeError(f"Skeleton labels do not match X3D labels at aligned rows {mismatch.tolist()}")

    video_src = contrastive_dir / args.video_feature
    video_dst = contrastive_dir / f"{args.prefix}{args.video_feature}"
    label_dst = contrastive_dir / f"{args.prefix}{args.video_labels}"
    valid_dst = contrastive_dir / f"{args.prefix}{args.video_valid_indices}"
    object_src = contrastive_dir / args.object_file
    object_dst = contrastive_dir / f"{args.prefix}{args.object_file}"
    sample_names_dst = contrastive_dir / f"{args.prefix}x3d_aligned_sample_names.npy"
    index_dst = contrastive_dir / f"{args.prefix}alignment_indices.npz"

    copy_feature_rows(video_src, video_dst, keep_valid_rows)
    np.save(label_dst, aligned_labels)
    np.save(valid_dst, keep_original_rows)
    filter_object_payload(object_src, object_dst, keep_original_rows, aligned_rel_videos, aligned_labels)
    np.save(sample_names_dst, aligned_rel_videos.astype(object))

    aligned_skeleton_npz = Path(args.aligned_skeleton_npz)
    aligned_skeleton_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        aligned_skeleton_npz,
        x_train=skel["x"][keep_skeleton_rows],
        y_train=skel["y"][keep_skeleton_rows],
        train_sample_name=np.array([csv_name_from_video(name) for name in aligned_rel_videos], dtype=object),
        train_subject=skel["subjects"][keep_skeleton_rows],
        train_action=skel["actions"][keep_skeleton_rows],
        train_frames=skel["frames"][keep_skeleton_rows],
    )

    np.savez(
        index_dst,
        x3d_feature_rows=keep_valid_rows,
        x3d_original_rows=keep_original_rows,
        skeleton_rows=keep_skeleton_rows,
        sample_names=aligned_rel_videos.astype(object),
        missing_skeleton=np.array(missing_skeleton, dtype=object),
    )

    summary = {
        "x3d_total_rows": int(len(rel_videos)),
        "x3d_valid_rows": int(len(valid_indices)),
        "aligned_rows": int(len(keep_original_rows)),
        "missing_skeleton_rows": int(len(missing_skeleton)),
        "missing_skeleton": missing_skeleton,
        "video_tmp": str(video_dst),
        "labels_tmp": str(label_dst),
        "valid_indices_tmp": str(valid_dst),
        "object_tmp": str(object_dst),
        "sample_names_tmp": str(sample_names_dst),
        "alignment_indices": str(index_dst),
        "aligned_skeleton_npz": str(aligned_skeleton_npz),
        "source_metadata": metadata_path_str(metadata),
    }
    summary_path = contrastive_dir / f"{args.prefix}alignment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def metadata_path_str(metadata):
    return {
        "source_root": metadata.get("source_root"),
        "split_mode": metadata.get("split_mode"),
        "split_seed": metadata.get("split_seed"),
    }


if __name__ == "__main__":
    main()
