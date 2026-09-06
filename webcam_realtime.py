#!/usr/bin/env python
"""Realtime webcam inference for CLIPGCN.

The trained CLIPGCN model expects pre-extracted video, pose, object and joint
features. This script reads frames from a local webcam, then runs X3D, YOLO,
MediaPipe/CTR-GCN, and CLIPGCN online.
"""

# the right is belonged to the RICELab Unige

import argparse
import os
import time
from collections import deque

import cv2
import numpy as np
import torch

from gating import UNKNOWN_LABEL, UnknownGate, label_name
from model import load_action_descriptions
from test import (
    apply_unseen_score_scale,
    load_split_classes,
    load_split_metadata,
    logits_to_cosine,
)
from test_raw_end_to_end import (
    CTRGCN_ROOT,
    X3D_ROOT,
    ObjectMapRunner,
    ctrgcn_pose_from_model,
    load_clipgcn_model,
    load_ctrgcn_model,
    load_x3d_model,
    load_yolo_model,
    x3d_features_from_model,
    x3d_tensor_from_frames,
)
from train import get_device, get_path_from_config, load_config, print_device_info


def parse_args():
    parser = argparse.ArgumentParser(description="Run CLIPGCN realtime inference from a local webcam.")
    parser.add_argument("--config", default="config_50_5.yaml", help="Path to the CLIPGCN YAML config.")
    parser.add_argument(
        "--class-split-dir",
        default=None,
        help="Directory whose metadata.json defines seen/unseen classes. Defaults to config data.train.data_dir.",
    )
    parser.add_argument(
        "--candidate-scope",
        choices=["unseen", "seen", "all"],
        default="all",
        help="Which action text labels are valid predictions.",
    )
    parser.add_argument(
        "--unseen-score-scale",
        type=float,
        default=1.3,
        help="Multiplier applied to unseen class confidence scores before top-k prediction.",
    )
    parser.add_argument("--clipgcn-checkpoint", default=None, help="Optional CLIPGCN checkpoint override.")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV webcam index.")
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--frames", type=int, default=13, help="Rolling frame window. Must match training.")
    parser.add_argument("--predict-every", type=int, default=13, help="Run recognition once every N captured frames.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--pose-source",
        choices=["mediapipe", "zero"],
        default="mediapipe",
        help="Realtime pose source. Use zero only for debugging or when MediaPipe is unavailable.",
    )
    parser.add_argument("--mediapipe-model-complexity", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--mediapipe-min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--mediapipe-min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--mediapipe-min-visibility", type=float, default=0.2)
    parser.add_argument("--window-name", default="CLIPGCN realtime")
    parser.add_argument("--headless", action="store_true", help="Print predictions without opening a display window.")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU fallback when the config requests CUDA.")
    parser.add_argument(
        "--cudnn-benchmark",
        action="store_true",
        help="Enable cudnn benchmark for fixed-size realtime inference on CUDA.",
    )

    parser.add_argument("--x3d-root", default=str(X3D_ROOT))
    parser.add_argument(
        "--x3d-config",
        default=str(X3D_ROOT / "configs" / "x3d-s_clipgcn_tensor_cross_subject_70_10_20_182.yaml"),
    )
    parser.add_argument(
        "--x3d-checkpoint",
        default=str(X3D_ROOT / "outputs" / "x3d-s_clipgcn_tensor_cs_70_10_20_182" / "model_007000.pth"),
    )
    parser.add_argument("--x3d-layer", default="s5")

    # Kept for compatibility with helpers imported from test_raw_end_to_end.py.
    parser.add_argument("--ctrgcn-root", default=str(CTRGCN_ROOT))
    parser.add_argument("--ctrgcn-config", default=str(CTRGCN_ROOT / "work_dir" / "etri_p1_p230_13frames" / "xsub" / "ctrgcn_joint_raw" / "config.yaml"))
    parser.add_argument("--ctrgcn-weights", default=str(CTRGCN_ROOT / "work_dir" / "etri_p1_p230_13frames" / "xsub" / "ctrgcn_joint_raw" / "runs-50-2700.pt"))
    parser.add_argument("--ctrgcn-hook-layer", default="l4")

    parser.add_argument("--yolo-repo", default="/workspace/yolov5")
    parser.add_argument("--yolo-weights", default="/workspace/yolov5/runs/train/coco_custom50_fixed2/weights/best.pt")
    parser.add_argument("--yolo-size", type=int, default=640)
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-iou", type=float, default=0.45)
    parser.add_argument("--yolo-half", action="store_true", help="Run YOLO in FP16 on CUDA.")
    parser.add_argument(
        "--yolo-detect-every",
        type=int,
        default=1,
        help="Run YOLO once every N predictions and reuse the previous object map in between.",
    )
    parser.add_argument("--no-yolo", action="store_true", help="Use zero object maps instead of YOLO.")
    parser.add_argument("--object-grid-size", type=int, default=6)
    parser.add_argument("--object-value", choices=["presence", "confidence"], default="presence")
    parser.add_argument("--object-max-distance-weight", type=float, default=10.0)

    # Gating overrides. Everything defaults to the inference.gating block of the
    # YAML; these flags exist to sweep thresholds without editing the config.
    parser.add_argument("--no-gating", action="store_true", help="Disable open-set gating entirely.")
    parser.add_argument("--novelty-method", choices=["entropy", "prototype"], default=None)
    parser.add_argument("--entropy-threshold", type=float, default=None)
    parser.add_argument("--cosine-threshold", type=float, default=None)
    parser.add_argument("--static-threshold", type=float, default=None)
    parser.add_argument("--hysteresis", type=int, default=None)
    return parser.parse_args()


def build_gate(config, candidate_labels, args):
    gate_config = dict((config.get("inference") or {}).get("gating") or {})
    novelty = dict(gate_config.get("novelty") or {})
    motion = dict(gate_config.get("motion") or {})

    if args.no_gating:
        gate_config["enabled"] = False
    if args.novelty_method is not None:
        novelty["method"] = args.novelty_method
    if args.entropy_threshold is not None:
        novelty["entropy_threshold"] = args.entropy_threshold
    if args.cosine_threshold is not None:
        novelty["cosine_threshold"] = args.cosine_threshold
    if args.static_threshold is not None:
        motion["static_threshold"] = args.static_threshold
    if args.hysteresis is not None:
        gate_config["hysteresis"] = args.hysteresis

    gate_config["novelty"] = novelty
    gate_config["motion"] = motion
    return UnknownGate(candidate_labels, gate_config)


def load_action_names(config, config_path):
    """Short action names for the overlay, e.g. 'fallen on the floor'."""

    text_config = config["data"]["text"]
    labels, names, _records = load_action_descriptions(
        xlsx_path=get_path_from_config(config_path, text_config["xlsx"]),
        text_column="action_description",
        id_column=text_config.get("id_column", "ID"),
        label_offset=text_config.get("label_offset", 1),
        prompt_template="{action_description}",
    )
    return {int(label): name for label, name in zip(labels, names)}


def validate_args(args):
    if args.frames != 13:
        raise ValueError(
            "This CLIPGCN checkpoint expects 13-frame features. "
            "Keep --frames 13 unless you retrain/update the fusion model."
        )
    if args.predict_every <= 0:
        raise ValueError("--predict-every must be positive.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if args.unseen_score_scale <= 0:
        raise ValueError("--unseen-score-scale must be positive.")


def open_camera(args):
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index {args.camera_index}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    return cap


class MediaPipePoseSource:
    """Converts MediaPipe's 33 landmarks into the NTU/Kinect 25-joint layout."""

    def __init__(self, args):
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError(
                "MediaPipe is not installed in this environment. "
                "Install it in the clipgcn environment, or run with --pose-source zero."
            ) from exc

        self.min_visibility = float(args.mediapipe_min_visibility)
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=args.mediapipe_model_complexity,
            enable_segmentation=False,
            min_detection_confidence=args.mediapipe_min_detection_confidence,
            min_tracking_confidence=args.mediapipe_min_tracking_confidence,
        )
        self.last_joints_3d = np.zeros((25, 3), dtype=np.float32)
        self.last_joint_xy = np.zeros((25, 2), dtype=np.float32)

    def close(self):
        self.pose.close()

    def process(self, frame_rgb):
        results = self.pose.process(frame_rgb)
        if not results.pose_landmarks:
            return {
                "joints_3d": self.last_joints_3d.copy(),
                "joint_xy": self.last_joint_xy.copy(),
                "detected": False,
            }

        landmarks = np.asarray(
            [
                [landmark.x, landmark.y, landmark.z, landmark.visibility]
                for landmark in results.pose_landmarks.landmark
            ],
            dtype=np.float32,
        )
        joints_3d, joint_xy = mediapipe_landmarks_to_ntu25(landmarks, self.min_visibility)
        self.last_joints_3d = joints_3d
        self.last_joint_xy = joint_xy
        return {
            "joints_3d": joints_3d,
            "joint_xy": joint_xy,
            "detected": True,
        }


def mediapipe_landmarks_to_ntu25(landmarks, min_visibility):
    def point(index):
        if landmarks[index, 3] < min_visibility:
            return None
        x = landmarks[index, 0] * 2.0 - 1.0
        y = landmarks[index, 1] * 2.0 - 1.0
        z = landmarks[index, 2]
        if not np.isfinite([x, y, z]).all():
            return None
        return np.asarray([x, y, z], dtype=np.float32)

    def average(*indices):
        values = [point(index) for index in indices]
        values = [value for value in values if value is not None]
        if not values:
            return None
        return np.mean(np.stack(values, axis=0), axis=0).astype(np.float32)

    def midpoint(a, b):
        if a is None:
            return b
        if b is None:
            return a
        return ((a + b) * 0.5).astype(np.float32)

    left_shoulder = point(11)
    right_shoulder = point(12)
    left_hip = point(23)
    right_hip = point(24)
    shoulder_center = midpoint(left_shoulder, right_shoulder)
    hip_center = midpoint(left_hip, right_hip)
    spine_mid = midpoint(shoulder_center, hip_center)

    ntu_points = [
        hip_center,  # 1 spine base
        spine_mid,  # 2 spine mid
        shoulder_center,  # 3 neck
        average(0, 7, 8),  # 4 head
        left_shoulder,
        point(13),
        point(15),
        point(19),
        right_shoulder,
        point(14),
        point(16),
        point(20),
        left_hip,
        point(25),
        point(27),
        point(31),
        right_hip,
        point(26),
        point(28),
        point(32),
        shoulder_center,  # 21 spine shoulder
        point(19),
        point(21),
        point(20),
        point(22),
    ]

    joints_3d = np.zeros((25, 3), dtype=np.float32)
    for index, value in enumerate(ntu_points):
        if value is not None:
            joints_3d[index] = value
    joint_xy = np.clip(joints_3d[:, :2], -1.0, 1.0).astype(np.float32)
    return joints_3d, joint_xy


def build_zero_pose_inputs(batch_size, frames, device):
    # Laptop webcam RGB has no Kinect/ETRI 25-joint skeleton stream.
    pose = torch.zeros(batch_size, 2, 64, frames, 25, dtype=torch.float32, device=device)
    joint_xy = torch.zeros(batch_size, frames, 25, 2, dtype=torch.float32, device=device)
    return pose, joint_xy


def build_mediapipe_skeleton_inputs(skeleton_buffer, device):
    joints_3d = np.stack([sample["joints_3d"] for sample in skeleton_buffer], axis=0).astype(np.float32)
    joint_xy = np.stack([sample["joint_xy"] for sample in skeleton_buffer], axis=0).astype(np.float32)

    skeleton = np.zeros((3, joints_3d.shape[0], 25, 2), dtype=np.float32)
    skeleton[:, :, :, 0] = joints_3d.transpose(2, 0, 1)
    detected_frames = sum(1 for sample in skeleton_buffer if sample["detected"])

    return (
        torch.from_numpy(skeleton).unsqueeze(0).to(device=device, dtype=torch.float32),
        torch.from_numpy(joint_xy).unsqueeze(0).to(device=device, dtype=torch.float32),
        detected_frames,
    )


def run_prediction(
    frame_buffer,
    skeleton_buffer,
    args,
    device,
    x3d_cfg,
    x3d_model,
    x3d_captured,
    ctrgcn_model,
    ctrgcn_captured,
    object_map_runner,
    clipgcn_model,
    candidate_labels,
    unseen_labels,
    use_amp,
    gate,
):
    frames = list(frame_buffer)
    x3d_clip = x3d_tensor_from_frames(
        frames,
        int(x3d_cfg.TRANSFORM.TEST.TENSOR_RESIZE_SIZE),
        x3d_cfg.TRANSFORM.MEAN,
        x3d_cfg.TRANSFORM.STD,
    ).unsqueeze(0).to(device, non_blocking=True)
    yolo_frame = frames[len(frames) // 2]
    detected_pose_frames = 0
    if args.pose_source == "mediapipe":
        skeletons, joint_xy, detected_pose_frames = build_mediapipe_skeleton_inputs(skeleton_buffer, device)
    else:
        pose_features, joint_xy = build_zero_pose_inputs(batch_size=1, frames=args.frames, device=device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
        object_maps = object_map_runner([yolo_frame])
        video_features = x3d_features_from_model(x3d_model, x3d_captured, x3d_clip, args)
        if args.pose_source == "mediapipe":
            pose_features = ctrgcn_pose_from_model(ctrgcn_model, ctrgcn_captured, skeletons, args)
        logits = clipgcn_model(video_features, pose_features, object_maps, joint_xy)
        prediction_scores, used_scale = apply_unseen_score_scale(
            clipgcn_model,
            logits,
            candidate_labels,
            unseen_labels=unseen_labels,
            unseen_score_scale=args.unseen_score_scale,
        )
        # Gating works on plain cosine similarities: the unseen rescaling above
        # is a GZSL calibration and would make the thresholds meaningless.
        gate_info = gate(
            logits_to_cosine(clipgcn_model, logits.float())[0],
            object_map=object_maps,
            joints=[sample["joints_3d"] for sample in skeleton_buffer] or None,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    top_k = min(args.top_k, len(candidate_labels))
    top_scores, top_indices = torch.topk(prediction_scores[0], k=top_k)
    labels = [int(candidate_labels[index]) for index in top_indices.detach().cpu().tolist()]
    scores = [float(score) for score in top_scores.detach().cpu().tolist()]
    return {
        "labels": labels,
        "scores": scores,
        "elapsed": elapsed,
        "used_scale": used_scale,
        "detected_pose_frames": detected_pose_frames,
        "gate": gate_info,
    }


def draw_overlay(frame_bgr, prediction, frame_count, args, names=None):
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (620, 290), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.55, frame_bgr, 0.45, 0, frame_bgr)

    cv2.putText(
        frame_bgr,
        "CLIPGCN realtime",
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        f"frames={args.frames}  unseen_scale={args.unseen_score_scale:g}  pose={args.pose_source}",
        (18, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (120, 220, 255),
        1,
        cv2.LINE_AA,
    )

    if prediction is None:
        text = f"warming up: {frame_count}/{args.frames}"
        cv2.putText(frame_bgr, text, (18, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 220, 255), 2, cv2.LINE_AA)
        return frame_bgr

    latency_ms = prediction["elapsed"] * 1000.0
    gate = prediction["gate"]
    is_unknown = gate["label"] == UNKNOWN_LABEL

    cv2.putText(
        frame_bgr,
        label_name(gate["label"], names).upper(),
        (18, 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (60, 60, 255) if is_unknown else (80, 255, 120),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        f"{gate['reason']}  person={int(gate['person'])}  H={gate['entropy']:.2f}"
        f"  cos={gate['cosine']:.2f}  motion={gate['motion']:.4f}",
        (18, 122),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (170, 200, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        f"inference {latency_ms:.1f} ms  pose_frames {prediction['detected_pose_frames']}/{args.frames}",
        (18, 146),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    for rank, (label, score) in enumerate(zip(prediction["labels"], prediction["scores"]), start=1):
        y = 146 + rank * 24
        color = (80, 255, 120) if rank == 1 and not is_unknown else (230, 230, 230)
        cv2.putText(
            frame_bgr,
            f"{rank}. {label_name(label, names)}: {score:.4f}",
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
    return frame_bgr


def print_prediction(prediction, names=None):
    if prediction is None:
        return
    gate = prediction["gate"]
    parts = [
        f"{rank}. {label_name(label, names)}={score:.4f}"
        for rank, (label, score) in enumerate(zip(prediction["labels"], prediction["scores"]), start=1)
    ]
    print(
        f"[{prediction['elapsed'] * 1000.0:.1f} ms] "
        f"{label_name(gate['label'], names).upper()} ({gate['reason']}, "
        f"H={gate['entropy']:.2f}, cos={gate['cosine']:.2f}, motion={gate['motion']:.4f}) | "
        + " | ".join(parts),
        flush=True,
    )


def main():
    args = parse_args()
    validate_args(args)

    config_path = os.path.abspath(args.config)
    config = load_config(config_path)
    requested_device = str(config["runtime"].get("device"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "The config requests CUDA, but torch.cuda.is_available() is False. "
            "Pass --allow-cpu intentionally, or run in an environment with GPU access."
        )
    device = get_device(config["runtime"].get("device"))
    print_device_info(device)
    if args.cudnn_benchmark and device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    class_split_dir = args.class_split_dir or get_path_from_config(config_path, config["data"]["train"]["data_dir"])
    class_split_dir = get_path_from_config(config_path, class_split_dir)
    split_metadata = load_split_metadata(class_split_dir)
    unseen_labels = [int(label) for label in split_metadata.get("unseen_classes", [])]
    candidate_labels = load_split_classes(class_split_dir, args.candidate_scope)

    x3d_model, x3d_captured, x3d_hook, x3d_cfg = load_x3d_model(args, device)
    ctrgcn_model = None
    ctrgcn_captured = None
    ctrgcn_hook = None
    if args.pose_source == "mediapipe":
        ctrgcn_model, ctrgcn_captured, ctrgcn_hook = load_ctrgcn_model(args, device)
    yolo_model = load_yolo_model(args, device)
    object_map_runner = ObjectMapRunner(yolo_model, device, args)
    clipgcn_model, checkpoint_path, candidate_labels = load_clipgcn_model(
        args,
        config,
        config_path,
        device,
        candidate_labels,
    )
    candidate_labels = [int(label) for label in candidate_labels]
    use_amp = bool(config["runtime"].get("amp", False)) and device.type == "cuda"
    gate = build_gate(config, candidate_labels, args)
    action_names = load_action_names(config, config_path)

    print("Realtime CLIPGCN webcam inference")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  class_split_dir: {class_split_dir}")
    print(f"  candidate_scope: {args.candidate_scope}")
    print(f"  candidate_labels: {candidate_labels}")
    print(f"  unseen_labels: {unseen_labels}")
    print(f"  unseen_score_scale: {args.unseen_score_scale:g}")
    print(f"  pose_source: {args.pose_source}")
    print(f"  camera_index: {args.camera_index}")
    if gate.enabled:
        threshold = (
            f"entropy>{gate.entropy_threshold:g}"
            if gate.method == "entropy"
            else f"cosine<{gate.cosine_threshold:g}"
        )
        print(f"  gating: {gate.method} ({threshold}), hysteresis={gate.hysteresis}")
        print(f"  static branch: motion<{gate.static_threshold:g} -> unknown or {gate.static_labels}")
    else:
        print("  gating: disabled")
    print("  quit: press q or ESC")

    pose_source = MediaPipePoseSource(args) if args.pose_source == "mediapipe" else None
    cap = open_camera(args)
    frame_buffer = deque(maxlen=args.frames)
    skeleton_buffer = deque(maxlen=args.frames)
    prediction = None
    captured_frames = 0

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                raise RuntimeError("Failed to read a frame from the webcam.")

            captured_frames += 1
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_buffer.append(frame_rgb)
            if pose_source is not None:
                skeleton_buffer.append(pose_source.process(frame_rgb))

            pose_ready = args.pose_source == "zero" or len(skeleton_buffer) == args.frames
            should_predict = (
                len(frame_buffer) == args.frames
                and pose_ready
                and captured_frames % args.predict_every == 0
            )
            if should_predict:
                prediction = run_prediction(
                    frame_buffer,
                    skeleton_buffer,
                    args,
                    device,
                    x3d_cfg,
                    x3d_model,
                    x3d_captured,
                    ctrgcn_model,
                    ctrgcn_captured,
                    object_map_runner,
                    clipgcn_model,
                    candidate_labels,
                    unseen_labels,
                    use_amp,
                    gate,
                )
                if args.headless:
                    print_prediction(prediction, action_names)

            if not args.headless:
                draw_overlay(frame_bgr, prediction, len(frame_buffer), args, action_names)
                cv2.imshow(args.window_name, frame_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
    finally:
        cap.release()
        x3d_hook.remove()
        if ctrgcn_hook is not None:
            ctrgcn_hook.remove()
        if pose_source is not None:
            pose_source.close()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
