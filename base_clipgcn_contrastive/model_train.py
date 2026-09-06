import argparse
import copy
import json
import os
import random
import re
import time
from collections import defaultdict

import torch
import torch.utils.data as Data
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

from model import build_model

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def load_config(config_path):
    # 读取 YAML，把训练参数集中放到 config.yaml 里管理。
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Please install PyYAML first: pip install pyyaml") from exc

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_path_from_config(config_path, path):
    # YAML 里的相对路径默认相对于 config.yaml 所在目录。
    if path is None or os.path.isabs(path):
        return path
    config_dir = os.path.dirname(os.path.abspath(config_path))
    return os.path.join(config_dir, path)


class CocoCaptionDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None, caption_mode="all"):
        self.image_dir = image_dir
        self.transform = transform
        self.caption_mode = caption_mode

        if self.caption_mode not in ["all", "first", "random"]:
            raise ValueError("Unsupported caption_mode: {}".format(self.caption_mode))

        with open(annotation_file, "r", encoding="utf-8") as f:
            coco = json.load(f)

        # COCO caption json 里 images 存文件名，annotations 存 caption。
        id_to_filename = {}
        for image_info in coco["images"]:
            id_to_filename[image_info["id"]] = image_info["file_name"]

        image_id_to_captions = defaultdict(list)
        for ann in coco["annotations"]:
            image_id = ann["image_id"]
            if image_id in id_to_filename:
                image_id_to_captions[image_id].append(ann["caption"])

        self.samples = []
        for image_id, file_name in id_to_filename.items():
            captions = image_id_to_captions[image_id]
            if len(captions) == 0:
                continue

            image_path = self.get_image_path(file_name)
            if not os.path.exists(image_path):
                continue

            if self.caption_mode == "all":
                for caption in captions:
                    self.samples.append((image_path, caption))
            elif self.caption_mode == "first":
                self.samples.append((image_path, captions[0]))
            else:
                self.samples.append((image_path, captions))

        if len(self.samples) == 0:
            raise ValueError("No image-caption samples found in {} and {}".format(image_dir, annotation_file))

    def get_image_path(self, file_name):
        # 兼容 COCO_..._000000123456.jpg 和 000000123456.jpg 两种命名。
        image_path = os.path.join(self.image_dir, file_name)
        if os.path.exists(image_path):
            return image_path

        match = re.search(r"(\d{12}\.jpg)$", file_name)
        if match is not None:
            image_path = os.path.join(self.image_dir, match.group(1))

        return image_path

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, caption_data = self.samples[index]
        if self.caption_mode == "random":
            caption = random.choice(caption_data)
        else:
            caption = caption_data

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, caption


def build_transform(transform_config):
    # 图片最终要变成 [3, 224, 224] 这种 tensor，才能送进 image encoder。
    transform_list = [
        transforms.Resize(tuple(transform_config["resize"])),
        transforms.ToTensor(),
    ]

    normalize_config = transform_config.get("normalize", {})
    if normalize_config.get("enabled", False):
        transform_list.append(transforms.Normalize(mean=normalize_config["mean"], std=normalize_config["std"]))

    return transforms.Compose(transform_list)


def train_val_data_process(config, config_path):
    # Dataset 每次返回一张 image tensor 和一条 caption 字符串。
    dataset_config = config["data"]["dataset"]
    loader_config = config["data"]["dataloader"]
    transform = build_transform(config["data"]["transforms"])

    train_data = CocoCaptionDataset(
        image_dir=get_path_from_config(config_path, dataset_config["train_image_dir"]),
        annotation_file=get_path_from_config(config_path, dataset_config["train_annotation"]),
        transform=transform,
        caption_mode=dataset_config["train_caption_mode"],
    )

    val_data = CocoCaptionDataset(
        image_dir=get_path_from_config(config_path, dataset_config["val_image_dir"]),
        annotation_file=get_path_from_config(config_path, dataset_config["val_annotation"]),
        transform=transform,
        caption_mode=dataset_config["val_caption_mode"],
    )

    train_data_loader = Data.DataLoader(
        train_data,
        batch_size=loader_config["batch_size"],
        shuffle=loader_config["train_shuffle"],
        num_workers=loader_config["num_workers"],
        pin_memory=loader_config["pin_memory"],
        drop_last=loader_config["drop_last"],
    )

    val_data_loader = Data.DataLoader(
        val_data,
        batch_size=loader_config["batch_size"],
        shuffle=loader_config["val_shuffle"],
        num_workers=loader_config["num_workers"],
        pin_memory=loader_config["pin_memory"],
        drop_last=False,
    )

    return train_data_loader, val_data_loader


def get_device(device_name=None):
    if device_name is None:
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"

    if device_name.startswith("cuda"):
        if not torch.cuda.is_available():
            print("CUDA is not available, using CPU instead.")
            return torch.device("cpu")

        if ":" in device_name:
            gpu_id = int(device_name.split(":")[1])
            if gpu_id >= torch.cuda.device_count():
                raise ValueError("GPU {} does not exist. Available GPU count: {}".format(gpu_id, torch.cuda.device_count()))

    return torch.device(device_name)


def print_device_info(device):
    if torch.cuda.is_available():
        print("Available GPUs:")
        for gpu_id in range(torch.cuda.device_count()):
            print("  cuda:{} - {}".format(gpu_id, torch.cuda.get_device_name(gpu_id)))
    print("Using device: {}".format(device))


def progress_bar(data_loader, desc):
    if tqdm is None:
        print("{}...".format(desc))
        return data_loader
    return tqdm(data_loader, desc=desc, leave=False)


def train_model_process(model, train_data_loader, val_data_loader, config, config_path, device=None):
    if device is None:
        device = get_device()

    train_config = config["train"]
    optimizer_config = train_config["optimizer"]
    output_config = config["outputs"]

    # CLIP text encoder 已冻结，所以只优化 requires_grad=True 的参数。
    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer_config["name"] == "Adam":
        optimizer = torch.optim.Adam(
            params,
            lr=optimizer_config["lr"],
            weight_decay=optimizer_config["weight_decay"],
        )
    elif optimizer_config["name"] == "AdamW":
        optimizer = torch.optim.AdamW(
            params,
            lr=optimizer_config["lr"],
            weight_decay=optimizer_config["weight_decay"],
        )
    else:
        raise ValueError("Unsupported optimizer: {}".format(optimizer_config["name"]))

    num_epochs = train_config["epochs"]
    use_amp = config["runtime"]["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")

    train_loss_all = []
    val_loss_all = []
    since = time.time()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        train_loss = 0.0
        val_loss = 0.0
        train_num = 0
        val_num = 0

        model.train()
        train_bar = progress_bar(train_data_loader, "Train epoch {}/{}".format(epoch + 1, num_epochs))

        for images, captions in train_bar:
            images = images.to(device)
            # captions 是字符串列表，不能调用 .to(device)。
            captions = list(captions)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                # CLIP-COCO 的核心训练目标：让同一位置的 image 和 caption 更相似。
                loss = model.contrastive_loss(images, captions)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = images.size(0)
            train_loss += loss.item() * batch_size
            train_num += batch_size

            if tqdm is not None:
                train_bar.set_postfix(loss=train_loss / train_num)

        model.eval()
        with torch.no_grad():
            val_bar = progress_bar(val_data_loader, "Val epoch {}/{}".format(epoch + 1, num_epochs))

            for images, captions in val_bar:
                images = images.to(device)
                captions = list(captions)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss = model.contrastive_loss(images, captions)

                batch_size = images.size(0)
                val_loss += loss.item() * batch_size
                val_num += batch_size

                if tqdm is not None:
                    val_bar.set_postfix(loss=val_loss / val_num)

        train_loss_all.append(train_loss / train_num)
        val_loss_all.append(val_loss / val_num)

        print("train_loss: {:.4f} epoch: {}".format(train_loss_all[-1], epoch))
        print("val_loss: {:.4f} epoch: {}".format(val_loss_all[-1], epoch))

        if val_loss_all[-1] < best_loss:
            best_loss = val_loss_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print("Training Time {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

    model.load_state_dict(best_model_wts)

    work_dir = get_path_from_config(config_path, output_config["work_dir"])
    os.makedirs(work_dir, exist_ok=True)
    save_path = get_path_from_config(config_path, output_config["best_model"])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(best_model_wts, save_path)

    train_process = {
        "epoch": range(num_epochs),
        "train_loss_all": train_loss_all,
        "val_loss_all": val_loss_all,
    }

    return train_process


def matplot_acc_loss(train_process, save_path=None, show=True):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed, skip plotting.")
        return

    plt.figure(figsize=(6, 4))

    plt.plot(train_process["epoch"], train_process["train_loss_all"], "ro-", label="train_loss")
    plt.plot(train_process["epoch"], train_process["val_loss_all"], "bo-", label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200)
        print("Training curve saved to {}".format(save_path))
    if show:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Train CLIP-COCO contrastive model.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_path = get_path_from_config(os.path.abspath(__file__), args.config)
    config = load_config(config_path)

    device = get_device(config["runtime"]["device"])
    print_device_info(device)

    net = build_model(
        text_model_name=config["model"]["text_encoder"]["name"],
        device=device,
        download_root=get_path_from_config(config_path, config["model"]["text_encoder"]["download_root"]),
    )

    train_data_loader, val_data_loader = train_val_data_process(config, config_path)

    train_process = train_model_process(
        net,
        train_data_loader,
        val_data_loader,
        config,
        config_path,
        device=device,
    )

    curve_path = get_path_from_config(config_path, config["outputs"]["train_curve"])
    os.makedirs(os.path.dirname(curve_path), exist_ok=True)
    matplot_acc_loss(train_process, save_path=curve_path, show=not config["runtime"]["no_show"])
