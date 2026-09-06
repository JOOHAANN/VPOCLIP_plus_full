"""End-to-end batch=1 latency benchmark for the full CLIPGCN framework.

Loads the REAL models (no offline features):
  1. YOLOv5m custom50  -> frame-7 detection -> object RS map (50,6,6)
  2. X3D-S             -> s5 hook          -> video feature (13,192,6,6)
  3. CTR-GCN           -> l4 hook          -> pose feature (2,64,13,25)
  4. CLIPGCN fusion    -> 512-d embedding  -> cosine over 55 text prototypes

All inputs are staged in RAM/GPU BEFORE the timed loop (no disk I/O measured).
Each stage is wrapped in cuda.synchronize() pairs; batch size is 1 throughout.

Usage:
  cd /workspace/CLIPGCN
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
    /root/miniconda3/envs/clipgcn/bin/python tools/benchmark_e2e_latency.py
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

YOLO_ROOT = "/workspace/yolov5"
X3D_ROOT = "/workspace/X3D"
CTR_ROOT = "/workspace/CTR-GCN"
CLIPGCN_ROOT = "/workspace/CLIPGCN"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def parse_args():
    p = argparse.ArgumentParser(description="Batch=1 end-to-end latency benchmark.")
    p.add_argument("--yolo-weights", default=f"{YOLO_ROOT}/runs/train/coco_custom50_fixed2/weights/best.pt")
    p.add_argument("--x3d-config", default=f"{X3D_ROOT}/configs/x3d-s_clipgcn_tensor_cross_subject_70_10_20.yaml")
    p.add_argument("--x3d-checkpoint", default=f"{X3D_ROOT}/outputs/x3d-s_clipgcn_tensor_cs_70_10_20/model_final.pth")
    p.add_argument("--ctr-config", default=f"{CTR_ROOT}/config/etri-p1-p230/ctrgcn_joint_raw_13.yaml")
    p.add_argument("--ctr-weights", default=f"{CTR_ROOT}/work_dir/etri_p1_p230_13frames/xsub/ctrgcn_joint_raw/runs-65-3510.pt")
    p.add_argument("--clipgcn-config", default=f"{CLIPGCN_ROOT}/config_55_0.yaml")
    p.add_argument("--clipgcn-checkpoint",
                   default=f"{CLIPGCN_ROOT}/work_dir/clipgcn_contrastive_55_0/run_20260712_093442/best_model.pth")
    p.add_argument("--sample-source", default="/workspace/X3D/data/clipgcn_tensor_cs_70_10_20/val_float16.npy")
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--yolo-imgsz", type=int, default=640)
    p.add_argument("--conf-thres", type=float, default=0.25)
    p.add_argument("--iou-thres", type=float, default=0.45)
    return p.parse_args()


# ---------------------------------------------------------------- model loading
def load_yolo(weights, device, imgsz):
    sys.path.insert(0, YOLO_ROOT)
    from models.common import DetectMultiBackend
    from utils.general import check_img_size

    model = DetectMultiBackend(weights, device=device, dnn=False, fp16=True)
    imgsz = check_img_size(imgsz, s=model.stride)
    return model, imgsz


def load_x3d(config_path, checkpoint, device):
    sys.path.insert(0, X3D_ROOT)
    cwd = os.getcwd()
    os.chdir(X3D_ROOT)  # vendored yacs + relative paths
    from tsn.config import get_cfg_defaults
    from tsn.model.recognizers.build import build_recognizer

    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_path)
    cfg.defrost()
    cfg.NUM_GPUS = 1 if device.type == "cuda" else 0
    cfg.MODEL.PRETRAINED = checkpoint
    cfg.freeze()
    model = build_recognizer(cfg, device=device)
    model.eval()
    os.chdir(cwd)

    feats = {}

    def hook(_m, _i, output):
        feats["s5"] = output

    module = model
    for part in "backbone.s5".split("."):
        if not hasattr(module, part):
            module = model  # fallback: search plain 's5'
            for p2 in "s5".split("."):
                module = getattr(module, p2)
            break
        module = getattr(module, part)
    module.register_forward_hook(hook)
    return model, feats


def load_ctrgcn(config_path, weights, device):
    sys.path.insert(0, CTR_ROOT)
    import importlib

    with open(config_path) as f:
        ctr_cfg = yaml.safe_load(f)
    module_name, class_name = ctr_cfg["model"].rsplit(".", 1)
    Model = getattr(importlib.import_module(module_name), class_name)
    model = Model(**ctr_cfg["model_args"]).to(device)

    state = torch.load(weights, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }
    model.load_state_dict(state)
    model.eval()

    feats = {}

    def hook(_m, _i, output):
        feats["l4"] = output

    model.l4.register_forward_hook(hook)

    # CTR-GCN 的 model 包与 CLIPGCN 的 model.py 同名, 用完清掉避免冲突
    for name in list(sys.modules):
        if name == "model" or name.startswith("model."):
            del sys.modules[name]
    sys.path.remove(CTR_ROOT)
    return model, feats


def load_clipgcn(config_path, checkpoint, device):
    sys.path.insert(0, CLIPGCN_ROOT)
    cwd = os.getcwd()
    os.chdir(CLIPGCN_ROOT)
    import test as clipgcn_test

    with open(config_path) as f:
        config = yaml.safe_load(f)
    model = clipgcn_test.load_model(config, config_path, device, checkpoint)
    candidate_labels = list(range(55))
    candidate_labels, _ = clipgcn_test.build_text_bank(model, config, config_path, candidate_labels)
    model.eval()
    os.chdir(cwd)
    return model


# ---------------------------------------------------------------- stages
def make_object_rs_map(det, img_hw, device, grid_size=6, max_w=10.0):
    """YOLO detections -> (1,1,50,6,6) RS map, mirroring build_object_rs_maps."""
    rs = torch.zeros(1, 1, 50, grid_size, grid_size, device=device)
    if det is None or len(det) == 0:
        return rs
    h, w = img_hw
    gx = torch.linspace(-1, 1, grid_size, device=device)
    gy = torch.linspace(1, -1, grid_size, device=device)
    grid_x = gx.view(1, -1).expand(grid_size, -1)
    grid_y = gy.view(-1, 1).expand(-1, grid_size)
    best_conf = {}
    for *xyxy, conf, cls in det.tolist():
        c = int(cls)
        if conf <= best_conf.get(c, 0.0):
            continue
        best_conf[c] = conf
        cx = ((xyxy[0] + xyxy[2]) * 0.5) / w
        cy = ((xyxy[1] + xyxy[3]) * 0.5) / h
        x_rs, y_rs = 2 * cx - 1, 1 - 2 * cy
        dist = torch.sqrt((grid_x - x_rs) ** 2 + (grid_y - y_rs) ** 2)
        rs[0, 0, c] = torch.clamp(1.0 / (dist + 1e-6), 0.0, max_w)
    return rs


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "需要GPU"
    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---------------- 加载全部模型 (不计时) ----------------
    print("loading models ...")
    yolo, imgsz = load_yolo(args.yolo_weights, device, args.yolo_imgsz)
    from utils.general import non_max_suppression  # yolov5, path already inserted
    from utils.augmentations import letterbox

    x3d, x3d_feats = load_x3d(args.x3d_config, args.x3d_checkpoint, device)
    ctr, ctr_feats = load_ctrgcn(args.ctr_config, args.ctr_weights, device)
    clipgcn = load_clipgcn(args.clipgcn_config, args.clipgcn_checkpoint, device)
    print("models loaded.")

    # ---------------- 预置输入 (不计时, 无磁盘IO进入计时段) ----------------
    cache = np.load(args.sample_source, mmap_mode="r")
    clip_np = np.array(cache[0], dtype=np.float32)               # (3,13,160,160)
    clip = torch.from_numpy(clip_np).unsqueeze(0).to(device)     # X3D 输入

    frame7 = clip_np[:, 6]                                       # (3,160,160) 第7帧
    frame7 = (frame7 * IMAGENET_STD + IMAGENET_MEAN).clip(0, 1)
    frame7_uint8 = np.ascontiguousarray(
        (np.transpose(frame7, (1, 2, 0)) * 255).astype(np.uint8))  # HWC RGB, 在内存

    skel = torch.randn(1, 3, 13, 25, 2, device=device)           # CTR-GCN 输入(数值不影响耗时)
    joint_xy = torch.rand(1, 13, 25, 2, device=device)           # 融合的关节坐标

    def stage_yolo():
        img = letterbox(frame7_uint8, imgsz, stride=yolo.stride, auto=yolo.pt)[0]
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        im = torch.from_numpy(img).to(device).half() / 255.0
        pred = yolo(im.unsqueeze(0), augment=False, visualize=False)
        det = non_max_suppression(pred, args.conf_thres, args.iou_thres, max_det=300)[0]
        return make_object_rs_map(det, frame7_uint8.shape[:2], device)

    def stage_x3d():
        with torch.no_grad():
            # 复刻特征抽取管线: 160 -> 182 resize 后过骨干, s5 输出 6x6
            x = clip.transpose(1, 2).reshape(13, 3, 160, 160)
            x = torch.nn.functional.interpolate(x, size=(182, 182), mode="bilinear", align_corners=False)
            x = x.reshape(1, 13, 3, 182, 182).transpose(1, 2).contiguous()
            x3d(x)
        f = x3d_feats["s5"]                                      # (1,192,13,6,6)
        return f.permute(0, 2, 1, 3, 4).contiguous()             # (1,13,192,6,6)

    def stage_ctr():
        with torch.no_grad():
            ctr(skel)
        f = ctr_feats["l4"]                                      # 期望 (N*M,64,13,25) 或类似
        if f.ndim == 4 and f.shape[0] == 2:
            f = f.unsqueeze(0)                                   # (1,2,64,13,25)
        elif f.ndim == 4:
            f = f.view(1, -1, *f.shape[1:])
        return f.contiguous()

    def stage_fusion(video_f, pose_f, obj_rs):
        with torch.no_grad():
            logits = clipgcn(video_f, pose_f, obj_rs, joint_xy)
        return logits.argmax(dim=1)

    # ---------------- warmup ----------------
    print(f"warmup x{args.warmup} ...")
    with torch.no_grad():
        for _ in range(args.warmup):
            obj = stage_yolo(); vf = stage_x3d(); pf = stage_ctr(); stage_fusion(vf, pf, obj)
    torch.cuda.synchronize()

    # ---------------- 分阶段计时 ----------------
    stages = {"yolo(det+RSmap)": [], "x3d(s5)": [], "ctrgcn(l4)": [], "fusion+cosine": []}
    print(f"timing x{args.iters} (batch=1) ...")
    with torch.no_grad():
        for _ in range(args.iters):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            obj = stage_yolo()
            torch.cuda.synchronize(); t1 = time.perf_counter()
            vf = stage_x3d()
            torch.cuda.synchronize(); t2 = time.perf_counter()
            pf = stage_ctr()
            torch.cuda.synchronize(); t3 = time.perf_counter()
            stage_fusion(vf, pf, obj)
            torch.cuda.synchronize(); t4 = time.perf_counter()
            stages["yolo(det+RSmap)"].append(t1 - t0)
            stages["x3d(s5)"].append(t2 - t1)
            stages["ctrgcn(l4)"].append(t3 - t2)
            stages["fusion+cosine"].append(t4 - t3)

    # ---------------- 报告 ----------------
    print("\n================ batch=1 端到端延迟 (不含磁盘IO) ================")
    total_mean = 0.0
    for name, ts in stages.items():
        arr = np.array(ts) * 1000
        total_mean += arr.mean()
        print(f"{name:18s}: {arr.mean():7.2f} ms  (std {arr.std():.2f}, p50 {np.percentile(arr,50):.2f}, p95 {np.percentile(arr,95):.2f})")
    print("-" * 60)
    print(f"{'TOTAL':18s}: {total_mean:7.2f} ms/clip  ->  {1000.0/total_mean:.1f} clips/s")
    print("\n注: YOLO阶段含 letterbox预处理+H2D拷贝+NMS+RS map构建;")
    print("    骨架关键点获取(相机/姿态估计)不在测量范围内; 文本原型为部署期常量, 已预计算。")


if __name__ == "__main__":
    main()
