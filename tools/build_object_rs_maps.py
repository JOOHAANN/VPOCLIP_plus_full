#!/usr/bin/env python
"""Convert object detections into 6x6 RS maps.

When per-instance detections are available, maps for repeated objects of the
same class are multiplied as described in the ADL paper:

    G = E
    G' = G * E'

The older cached files only contain one center per object class
(`presence [B,50]`, `center_xyz [B,50,3]`), so they cannot recover multiple
instances that were discarded upstream. Those files are still supported as the
single-instance-per-class fallback.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths",
        nargs="+",
        default=[
            "/workspace/CLIPGCN/data/contrastive_train_data/trimodal_train_object.npy",
            "/workspace/CLIPGCN/data/contrastive_zsl_splits/50_5/seen_object.npy",
            "/workspace/CLIPGCN/data/contrastive_zsl_splits/50_5/unseen_object.npy",
            "/workspace/CLIPGCN/data/contrastive_zsl_splits/45_10/seen_object.npy",
            "/workspace/CLIPGCN/data/contrastive_zsl_splits/45_10/unseen_object.npy",
            "/workspace/CLIPGCN/data/contrastive_zsl_splits/40_15/seen_object.npy",
            "/workspace/CLIPGCN/data/contrastive_zsl_splits/40_15/unseen_object.npy",
            "/workspace/CLIPGCN/data/contrastive_zsl_splits/35_20/seen_object.npy",
            "/workspace/CLIPGCN/data/contrastive_zsl_splits/35_20/unseen_object.npy",
        ],
        help="Object .npy files containing dicts with presence and center_xyz.",
    )
    parser.add_argument("--grid-size", type=int, default=6)
    parser.add_argument(
        "--value",
        choices=["presence", "confidence"],
        default="presence",
        help="Value multiplied by the distance map. presence matches the original [B,50] object feature.",
    )
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument(
        "--max-distance-weight",
        type=float,
        default=10.0,
        help="Clamp inverse-distance grids to avoid 1e6 spikes when an object lands on a grid point.",
    )
    parser.add_argument(
        "--backup-dir",
        default="/workspace/CLIPGCN/data/object_dict_backups_before_rs",
        help="Where to copy original object dict files before overwriting.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing map arrays too.")
    return parser.parse_args()


def spatial_grid(grid_size):
    x = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
    y = np.linspace(1.0, -1.0, grid_size, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    return grid_x, grid_y


def backup_path_for(path, backup_dir):
    return backup_dir / (path.parent.name + "__" + path.name)


def load_object_dict(path):
    value = np.load(path, allow_pickle=True)
    if value.shape != ():
        if value.ndim == 4:
            return None
        raise ValueError(f"{path} is neither an object dict nor an object map; shape={value.shape}")
    obj = value.item()
    if not isinstance(obj, dict):
        raise ValueError(f"{path} does not contain an object dict")
    return obj


def load_object_dict_or_backup(path, backup_dir, overwrite):
    try:
        obj = load_object_dict(path)
    except ValueError:
        if not overwrite:
            raise
        obj = None
    if obj is not None:
        return obj, path
    if not overwrite:
        return None, path

    backup_path = backup_path_for(path, backup_dir)
    if not backup_path.exists():
        raise ValueError(f"{path} already contains a map array and backup was not found: {backup_path}")
    backup_obj = load_object_dict(backup_path)
    if backup_obj is None:
        raise ValueError(f"Backup does not contain an object dict: {backup_path}")
    return backup_obj, backup_path


def distance_grid(grid_x, grid_y, x_obj, y_obj, max_distance_weight):
    dx = grid_x[None, None, :, :] - x_obj[:, :, None, None]
    dy = grid_y[None, None, :, :] - y_obj[:, :, None, None]
    distance_map = 1.0 / (np.sqrt(dx * dx + dy * dy) + 1e-6)
    return np.clip(distance_map, 0.0, max_distance_weight)


def find_instance_payload(obj):
    """Return (class_ids, center_xy, values) if obj stores per-instance data."""

    class_key = next(
        (key for key in ("instance_class_ids", "detection_class_ids", "det_class_ids", "classes") if key in obj),
        None,
    )
    center_key = next(
        (key for key in ("instance_center_xy", "detection_center_xy", "det_center_xy", "centers_xy") if key in obj),
        None,
    )
    if class_key is None or center_key is None:
        return None

    class_ids = np.asarray(obj[class_key])
    center_xy = np.asarray(obj[center_key], dtype=np.float32)
    if class_ids.ndim != 2 or center_xy.ndim != 3 or center_xy.shape[-1] < 2:
        return None

    value_key = next(
        (key for key in ("instance_confidence", "detection_confidence", "det_confidence", "scores") if key in obj),
        None,
    )
    values = np.asarray(obj[value_key], dtype=np.float32) if value_key is not None else np.ones_like(class_ids, dtype=np.float32)
    valid_key = next(
        (key for key in ("instance_valid", "detection_valid", "det_valid") if key in obj),
        None,
    )
    valid = np.asarray(obj[valid_key]).astype(bool) if valid_key is not None else class_ids >= 0
    return class_ids, center_xy[..., :2], values, valid


def build_maps_from_instances(obj, grid_size, value_key, dtype, max_distance_weight):
    instance_payload = find_instance_payload(obj)
    if instance_payload is None:
        return None

    det_class_ids, center_xy, det_values, valid = instance_payload
    class_ids = np.asarray(obj["class_ids"], dtype=np.int64)
    class_to_slot = {int(class_id): slot for slot, class_id in enumerate(class_ids.tolist())}
    batch_size = det_class_ids.shape[0]
    num_objects = len(class_ids)
    grid_x, grid_y = spatial_grid(grid_size)

    output = np.zeros((batch_size, num_objects, grid_size, grid_size), dtype=np.float32)
    instance_counts = np.zeros((batch_size, num_objects), dtype=np.int16)

    for sample_idx in range(batch_size):
        per_class_grids = {}
        for det_idx in range(det_class_ids.shape[1]):
            if not valid[sample_idx, det_idx]:
                continue
            slot = class_to_slot.get(int(det_class_ids[sample_idx, det_idx]))
            if slot is None:
                continue

            x_obj = 2.0 * center_xy[sample_idx, det_idx, 0] - 1.0
            y_obj = 1.0 - 2.0 * center_xy[sample_idx, det_idx, 1]
            dx = grid_x - x_obj
            dy = grid_y - y_obj
            e_grid = 1.0 / (np.sqrt(dx * dx + dy * dy) + 1e-6)
            e_grid = np.clip(e_grid, 0.0, max_distance_weight)
            if value_key == "confidence":
                e_grid = e_grid * float(det_values[sample_idx, det_idx])

            if slot in per_class_grids:
                per_class_grids[slot] = per_class_grids[slot] * e_grid
            else:
                per_class_grids[slot] = e_grid
            instance_counts[sample_idx, slot] += 1

        for slot, grid in per_class_grids.items():
            output[sample_idx, slot] = grid

    obj["_instance_counts_for_metadata"] = instance_counts
    return output.astype(dtype, copy=False)


def build_maps_from_class_summary(obj, grid_size, value_key, dtype, max_distance_weight):
    presence = np.asarray(obj["presence"], dtype=np.float32)
    center_xy = np.asarray(obj["center_xyz"], dtype=np.float32)[..., :2]
    values = presence if value_key == "presence" else np.asarray(obj["confidence"], dtype=np.float32)

    batch_size, num_objects = presence.shape
    grid_x, grid_y = spatial_grid(grid_size)
    output = np.zeros((batch_size, num_objects, grid_size, grid_size), dtype=dtype)

    chunk = 512
    for start in range(0, batch_size, chunk):
        end = min(start + chunk, batch_size)
        chunk_presence = presence[start:end]
        chunk_values = values[start:end]
        x_obj = 2.0 * center_xy[start:end, :, 0] - 1.0
        y_obj = 1.0 - 2.0 * center_xy[start:end, :, 1]

        distance_map = distance_grid(grid_x, grid_y, x_obj, y_obj, max_distance_weight)
        maps = chunk_values[:, :, None, None] * distance_map
        maps *= chunk_presence[:, :, None, None]
        output[start:end] = maps.astype(dtype, copy=False)

    return output


def build_maps(obj, grid_size, value_key, dtype, max_distance_weight):
    maps = build_maps_from_instances(obj, grid_size, value_key, dtype, max_distance_weight)
    if maps is not None:
        return maps, "per_instance_multiplicative"
    return build_maps_from_class_summary(obj, grid_size, value_key, dtype, max_distance_weight), "per_class_single_center"


def write_metadata(path, obj, output_shape, args, mode):
    instance_counts = obj.get("_instance_counts_for_metadata")
    metadata = {
        "output": str(path),
        "shape": list(output_shape),
        "dtype": args.dtype,
        "source_format": mode,
        "value": args.value,
        "grid_size": args.grid_size,
        "max_distance_weight": args.max_distance_weight,
        "coordinate_transform": {
            "input_xy": "[0,1] normalized image coordinates",
            "x_rs": "2*x - 1",
            "y_rs": "1 - 2*y",
            "grid_x": "linspace(-1, 1, 6)",
            "grid_y": "linspace(1, -1, 6)",
        },
        "formula": "single instance: G=E; repeated same-class instances: G'=G*E'; absent classes are zero",
        "absent_objects": "all-zero 6x6 map because presence=0",
        "class_ids": np.asarray(obj.get("class_ids", [])).astype(int).tolist(),
        "class_names": [str(x) for x in np.asarray(obj.get("class_names", [])).tolist()],
    }
    if instance_counts is not None:
        metadata["multi_instance_summary"] = {
            "max_instances_per_class": int(instance_counts.max()) if instance_counts.size else 0,
            "sample_class_pairs_with_multiple_instances": int((instance_counts > 1).sum()),
        }
    elif mode == "per_class_single_center":
        metadata["multi_instance_note"] = (
            "This source file only stores one center per class, so same-class "
            "multi-object multiplication cannot be reconstructed from it."
        )
    metadata_path = path.with_suffix("").as_posix() + ".rs_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    return metadata_path


def main():
    args = parse_args()
    backup_dir = Path(args.backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    dtype = np.float32 if args.dtype == "float32" else np.float16

    for raw_path in args.paths:
        path = Path(raw_path)
        obj, source_path = load_object_dict_or_backup(path, backup_dir, args.overwrite)
        if obj is None:
            print(f"skip existing map: {path}")
            continue

        backup_path = backup_path_for(path, backup_dir)
        if not backup_path.exists():
            shutil.copy2(path, backup_path)

        maps, mode = build_maps(obj, args.grid_size, args.value, dtype, args.max_distance_weight)
        np.save(path, maps)
        metadata_path = write_metadata(path, obj, maps.shape, args, mode)
        print(f"{path}: {maps.shape} {maps.dtype}; source={source_path}; backup={backup_path}; metadata={metadata_path}")


if __name__ == "__main__":
    main()
