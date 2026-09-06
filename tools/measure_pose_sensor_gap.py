"""Measure how much accuracy is lost by swapping the skeleton source.

The model is trained on Kinect v2 skeletons from the ETRI csv files, but on a
robot the skeleton comes from an RGB pose estimator. Both are run here over the
same clips, with the same video and object streams, so the only thing that
changes is where the joints come from.

Kinect gives metric 3D (Z is absolute distance from the camera, ~1.7 m),
MediaPipe gives normalised image coordinates plus a hip-relative Z centred on
zero, so the deployed Z never even lands in the range the model was trained on.

Usage:
  cd /workspace/VPOCLIP_plus
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
    python tools/measure_pose_sensor_gap.py --max-samples 400
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from test_raw_end_to_end import (  # noqa: E402
    CTRGCN_ROOT,
    X3D_ROOT,
    ObjectMapRunner,
    ctrgcn_pose_from_model,
    ctrgcn_tensor_and_joint_xy_from_csv,
    load_clipgcn_model,
    load_ctrgcn_model,
    load_split_manifest,
    load_x3d_model,
    load_yolo_model,
    read_uniform_rgb_frames,
    x3d_features_from_model,
    x3d_tensor_from_frames,
)
from train import get_device, get_path_from_config, load_config  # noqa: E402
from webcam_realtime import mediapipe_landmarks_to_ntu25  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Kinect vs RGB pose estimator accuracy gap.")
    parser.add_argument("--config", default="config_55_0_objconcat.yaml")
    parser.add_argument("--clipgcn-checkpoint", default=None)
    parser.add_argument("--split-dir", default="./data/contrastive_test_data")
    parser.add_argument("--class-split-dir", default="./data/contrastive_zsl_splits/50_5")
    parser.add_argument("--prefix", default="trimodal_test")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--max-samples", type=int, default=400)
    parser.add_argument("--frames", type=int, default=13)
    parser.add_argument("--output", default="./work_dir/pose_sensor_gap.json")

    parser.add_argument("--mediapipe-model", default=str(ROOT / "checkpoints" / "pose_landmarker_full.task"))
    parser.add_argument("--mediapipe-min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--mediapipe-min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--mediapipe-min-visibility", type=float, default=0.2)

    parser.add_argument("--x3d-root", default=str(X3D_ROOT))
    parser.add_argument("--x3d-config", default=str(X3D_ROOT / "configs" / "x3d-s_clipgcn_tensor_cross_subject_70_10_20_182.yaml"))
    parser.add_argument("--x3d-checkpoint", default=str(X3D_ROOT / "outputs" / "x3d-s_clipgcn_tensor_cs_70_10_20_182" / "model_007000.pth"))
    parser.add_argument("--x3d-layer", default="s5")
    parser.add_argument("--ctrgcn-root", default=str(CTRGCN_ROOT))
    parser.add_argument("--ctrgcn-config", default=str(CTRGCN_ROOT / "work_dir" / "etri_p1_p230_13frames" / "xsub" / "ctrgcn_joint_raw" / "config.yaml"))
    parser.add_argument("--ctrgcn-weights", default=str(CTRGCN_ROOT / "work_dir" / "etri_p1_p230_13frames" / "xsub" / "ctrgcn_joint_raw" / "runs-50-2700.pt"))
    parser.add_argument("--ctrgcn-hook-layer", default="l4")
    parser.add_argument("--yolo-repo", default="/workspace/yolov5")
    parser.add_argument("--yolo-weights", default="/workspace/yolov5/runs/train/coco_custom50_fixed2/weights/best.pt")
    parser.add_argument("--yolo-size", type=int, default=640)
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-iou", type=float, default=0.45)
    parser.add_argument("--yolo-half", action="store_true")
    parser.add_argument("--yolo-detect-every", type=int, default=1)
    parser.add_argument("--no-yolo", action="store_true")
    parser.add_argument("--object-grid-size", type=int, default=6)
    parser.add_argument("--object-value", choices=["presence", "confidence"], default="presence")
    parser.add_argument("--object-max-distance-weight", type=float, default=10.0)
    return parser.parse_args()


class MediaPipeSkeleton:
    """Same 33 -> NTU-25 mapping the realtime demo uses, run offline on frames.

    mediapipe 0.10.35 dropped the ``mp.solutions`` API the demo still calls, so
    this goes through the Tasks API instead. The landmark order and the
    normalised x/y plus hip-relative z are unchanged, so the mapping is shared.
    """

    def __init__(self, args):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        self.min_visibility = float(args.mediapipe_min_visibility)
        self.pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(args.mediapipe_model)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=args.mediapipe_min_detection_confidence,
                min_tracking_confidence=args.mediapipe_min_tracking_confidence,
            )
        )
        self.timestamp = 0

    def close(self):
        self.pose.close()

    def __call__(self, frames):
        joints, joint_xy = [], []
        detected = 0
        last_3d = np.zeros((25, 3), dtype=np.float32)
        last_xy = np.zeros((25, 2), dtype=np.float32)
        for frame in frames:
            image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(frame),
            )
            # Frames are sampled from a clip, so any monotonic clock works.
            self.timestamp += 33
            result = self.pose.detect_for_video(image, self.timestamp)
            if result.pose_landmarks:
                landmarks = np.asarray(
                    [[p.x, p.y, p.z, p.visibility] for p in result.pose_landmarks[0]],
                    dtype=np.float32,
                )
                last_3d, last_xy = mediapipe_landmarks_to_ntu25(landmarks, self.min_visibility)
                detected += 1
            joints.append(last_3d.copy())
            joint_xy.append(last_xy.copy())

        joints = np.stack(joints, axis=0)
        # CTR-GCN wants [C, T, V, M]; the second person stays empty, as it does
        # on the robot where only one body is tracked.
        skeleton = np.zeros((3, joints.shape[0], 25, 2), dtype=np.float32)
        skeleton[:, :, :, 0] = joints.transpose(2, 0, 1)
        return (
            torch.from_numpy(skeleton).float(),
            torch.from_numpy(np.stack(joint_xy, axis=0)).float(),
            detected,
        )


def predict(models, x3d_clip, yolo_frame, skeleton, joint_xy, candidate_labels, device, args):
    x3d_model, x3d_captured, ctrgcn_model, ctrgcn_captured, object_runner, clipgcn_model = models
    with torch.inference_mode():
        object_maps = object_runner([yolo_frame])
        video_features = x3d_features_from_model(x3d_model, x3d_captured, x3d_clip.unsqueeze(0).to(device), args)
        pose_features = ctrgcn_pose_from_model(ctrgcn_model, ctrgcn_captured, skeleton.unsqueeze(0).to(device), args)
        logits = clipgcn_model(video_features, pose_features, object_maps, joint_xy.unsqueeze(0).to(device))
    return int(candidate_labels[int(logits[0].argmax())])


def main():
    args = parse_args()
    config_path = str(ROOT / args.config)
    config = load_config(config_path)
    device = get_device(config["runtime"].get("device"))

    class_split_dir = get_path_from_config(config_path, args.class_split_dir)
    metadata = json.load(open(Path(class_split_dir) / "metadata.json"))
    candidate_labels = sorted(metadata["seen_classes"] + metadata["unseen_classes"])

    split_dir = get_path_from_config(config_path, args.split_dir)
    data_root = get_path_from_config(config_path, args.data_root)
    items = load_split_manifest(split_dir, args.prefix, data_root, max_samples=args.max_samples)
    print(f"clips: {len(items)}   candidates: {len(candidate_labels)}")

    x3d_model, x3d_captured, x3d_hook, x3d_cfg = load_x3d_model(args, device)
    ctrgcn_model, ctrgcn_captured, ctrgcn_hook = load_ctrgcn_model(args, device)
    object_runner = ObjectMapRunner(load_yolo_model(args, device), device, args)
    clipgcn_model, checkpoint_path, candidate_labels = load_clipgcn_model(
        args, config, config_path, device, candidate_labels
    )
    candidate_labels = [int(label) for label in candidate_labels]
    models = (x3d_model, x3d_captured, ctrgcn_model, ctrgcn_captured, object_runner, clipgcn_model)
    print(f"checkpoint: {checkpoint_path}")

    pose_estimator = MediaPipeSkeleton(args)
    records = []
    start = time.time()
    for index, item in enumerate(items):
        frames = read_uniform_rgb_frames(item["video_path"], args.frames)
        x3d_clip = x3d_tensor_from_frames(
            frames,
            int(x3d_cfg.TRANSFORM.TEST.TENSOR_RESIZE_SIZE),
            x3d_cfg.TRANSFORM.MEAN,
            x3d_cfg.TRANSFORM.STD,
        )
        yolo_frame = frames[len(frames) // 2]

        kinect_skeleton, kinect_xy = ctrgcn_tensor_and_joint_xy_from_csv(item["csv_path"], args.frames)
        mp_skeleton, mp_xy, detected = pose_estimator(frames)

        # The skeleton feeds CTR-GCN, joint_xy places the pose maps on the RS
        # grid. Crossing them says which of the two paths carries the gap.
        record = {
            "sample_name": item["sample_name"],
            "label": int(item["label"]),
            "mediapipe_detected_frames": detected,
        }
        for name, (skeleton, xy) in {
            "kinect_pred": (kinect_skeleton, kinect_xy),
            "mediapipe_pred": (mp_skeleton, mp_xy),
            "kinect_skel_mp_xy": (kinect_skeleton, mp_xy),
            "mp_skel_kinect_xy": (mp_skeleton, kinect_xy),
        }.items():
            record[name] = predict(models, x3d_clip, yolo_frame, skeleton, xy,
                                   candidate_labels, device, args)
        records.append(record)

        if (index + 1) % 25 == 0:
            rate = (index + 1) / (time.time() - start)
            accs = {
                key: np.mean([r[key] == r["label"] for r in records])
                for key in ("kinect_pred", "mediapipe_pred", "kinect_skel_mp_xy", "mp_skel_kinect_xy")
            }
            print(f"  [{index + 1}/{len(items)}] "
                  + "  ".join(f"{k.replace('_pred', '')}={v:.3f}" for k, v in accs.items())
                  + f"  ({rate:.1f} clip/s)", flush=True)

    pose_estimator.close()
    x3d_hook.remove()
    ctrgcn_hook.remove()

    labels = np.array([r["label"] for r in records])
    kinect = np.array([r["kinect_pred"] for r in records]) == labels
    mediapipe = np.array([r["mediapipe_pred"] for r in records]) == labels
    detected = np.array([r["mediapipe_detected_frames"] for r in records])

    summary = {
        "num_samples": len(records),
        "checkpoint": str(checkpoint_path),
        "kinect_top1": float(kinect.mean()),
        "mediapipe_top1": float(mediapipe.mean()),
        "kinect_skel_mp_xy_top1": float((np.array([r["kinect_skel_mp_xy"] for r in records]) == labels).mean()),
        "mp_skel_kinect_xy_top1": float((np.array([r["mp_skel_kinect_xy"] for r in records]) == labels).mean()),
        "gap": float(kinect.mean() - mediapipe.mean()),
        "agreement": float(np.mean([r["kinect_pred"] == r["mediapipe_pred"] for r in records])),
        "kinect_right_mediapipe_wrong": int((kinect & ~mediapipe).sum()),
        "mediapipe_right_kinect_wrong": int((mediapipe & ~kinect).sum()),
        "mediapipe_frames_detected_mean": float(detected.mean()),
        "clips_with_no_detection": int((detected == 0).sum()),
    }

    output = get_path_from_config(config_path, args.output)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "records": records}, handle, indent=2)

    print("\n================ pose sensor gap ================")
    print(f"clips                : {summary['num_samples']}")
    print(f"Kinect skel + Kinect xy : {summary['kinect_top1'] * 100:.2f}%")
    print(f"MP     skel + MP     xy : {summary['mediapipe_top1'] * 100:.2f}%")
    print(f"Kinect skel + MP     xy : {summary['kinect_skel_mp_xy_top1'] * 100:.2f}%   (only joint_xy swapped)")
    print(f"MP     skel + Kinect xy : {summary['mp_skel_kinect_xy_top1'] * 100:.2f}%   (only skeleton swapped)")
    print(f"gap                  : {summary['gap'] * 100:.2f} points")
    print(f"prediction agreement : {summary['agreement'] * 100:.2f}%")
    print(f"kinect right / mp wrong : {summary['kinect_right_mediapipe_wrong']}")
    print(f"mp right / kinect wrong : {summary['mediapipe_right_kinect_wrong']}")
    print(f"mediapipe frames detected: {summary['mediapipe_frames_detected_mean']:.2f}/{args.frames}"
          f"   clips with none: {summary['clips_with_no_detection']}")
    print(f"\nsaved: {output}")


if __name__ == "__main__":
    main()
