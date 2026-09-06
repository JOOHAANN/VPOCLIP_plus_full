"""
CLIP-COCO contrastive learning scaffold.

目标架构：
    image encoder: 你自己写、自己训练
    text encoder : 使用已经预训练好的 OpenAI CLIP text encoder，并冻结参数

本文件故意把关键步骤都写成中文注释，方便你顺着注释改。
训练时你通常只需要：
    1. 准备一个 DataLoader，每个 batch 返回 images 和 captions
    2. 调用 model(images, captions)
    3. 对 logits_per_image / logits_per_text 做交叉熵
    4. optimizer 只更新 image_encoder 和 logit_scale
"""

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# 1. 导入 CLIP 包
# ---------------------------------------------------------------------------
# 你的 clipgcn 环境里已经安装了本地 CLIP：
#   pip install -e /workspace/CLIP
# 所以这里可以直接 import。
import clip


def l2_normalize(features):
    """把特征归一化到单位长度，这样点积就等价于 cosine similarity。"""

    return F.normalize(features, dim=-1)



class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1conv=False, strides=1):
        super(Residual, self).__init__()
        self.ReLU = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(in_channels=num_channels,  out_channels=num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        if use_1conv:
            self.conv3 = nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
    def forward(self, x):
        y = self.ReLU(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y = self.ReLU(y+x)
        return y


class CustomImageEncoder(nn.Module):
    """
    你要重点修改的部分：自己写 image encoder。

    输入:
        images 是图片张量，形状是 [batch_size, 3, image_size, image_size]

    输出:
        image_features 是图像特征，形状是 [batch_size, embed_dim]

    关键要求:
        1. 输出维度必须等于 CLIP text encoder 的输出维度 embed_dim。
           例如 ViT-B/32 的 embed_dim 是 512。
        2. forward 只返回未归一化的特征；归一化在外层 ContrastiveModel 统一做。
        3. 这个模块的参数是需要训练的，所以不要在这里 no_grad。

    下面给了一个很小的 CNN baseline，能跑通训练流程。
    你可以把 self.backbone / self.projection 换成自己的 ResNet、ViT、CNN-GCN 等。
    """

    def __init__(self, embed_dim):
        super(CustomImageEncoder, self).__init__()

        # Step 1: 写图像特征提取网络。
        # 这里的例子会把 RGB 图片逐步下采样，然后用 AdaptiveAvgPool2d 压成 [B, C, 1, 1]。
        # 你真正做实验时，可以把这个 nn.Sequential 整块替换成自己的网络。
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.b2 = nn.Sequential(Residual(64, 64, use_1conv=False, strides=1),
                                Residual(64, 64, use_1conv=False, strides=1))

        self.b3 = nn.Sequential(Residual(64, 128, use_1conv=True, strides=2),
                                Residual(128, 128, use_1conv=False, strides=1))

        self.b4 = nn.Sequential(Residual(128, 256, use_1conv=True, strides=2),
                                Residual(256, 256, use_1conv=False, strides=1))

        self.b5 = nn.Sequential(Residual(256, 512, use_1conv=True, strides=2),
                                Residual(512, 512, use_1conv=False, strides=1))
        
        self.b6 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))

        # Step 2: 写 projection head，把 backbone 输出投影到 CLIP 文本特征空间。
        # 如果你的 backbone 最后输出通道不是 256，就把第一个 Linear 的输入维度改掉。
        self.projection = nn.Sequential(
            nn.Flatten(),  # [B, 256, 1, 1] 变成 [B, 256]
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, embed_dim),  # 最终必须是 [B, embed_dim]
        )

    def forward(self, images):
        # Step 3: 在 forward 里串起来。
        # 注意：这里不做 softmax，不做分类头，也不做 CrossEntropyLoss。
        # 对比学习只需要输出一个向量，让它和 text_features 做相似度。
        x = self.b1(images)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        image_features = self.projection(x)
        return image_features


class FrozenCLIPTextEncoder(nn.Module):
    """
    预训练 CLIP text encoder 的封装。

    你需要知道的接口只有三个：
        clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        tokens = clip.tokenize(["a dog", "a cat"], truncate=True).to(device)
        text_features = clip_model.encode_text(tokens)

    说明：
        - clip.tokenize 会把字符串变成 token id，shape = [B, 77]
        - encode_text 会输出文本向量，shape = [B, embed_dim]
        - 这里会 freeze CLIP 的所有参数，只把它当作固定的 teacher / target space
    """

    def __init__(self, model_name="ViT-B/32", device=None, download_root=None):
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        # Step 1: 加载 CLIP。第一次运行如果本地没有权重，会下载到 ~/.cache/clip
        # 或者你传入的 download_root。
        self.clip_model, _ = clip.load(
            model_name,
            device=device,
            jit=False,
            download_root=download_root,
        )

        # Step 2: 冻结 text encoder。训练时不会更新 CLIP 参数，只更新你自己的 image encoder。
        self.clip_model.eval()
        for parameter in self.clip_model.parameters():
            parameter.requires_grad = False

        # Step 3: 记录输出维度。image encoder 的输出必须和这个维度一致。
        self.embed_dim = int(self.clip_model.text_projection.shape[1])

    def get_device(self):
        # 返回 CLIP text encoder 当前所在的 device，例如 cpu、cuda、cuda:0。
        # 这样即使后面调用 model.to("cuda")，这里也能拿到最新 device。
        return next(self.clip_model.parameters()).device

    def forward(self, captions):
        # Step 4: captions 可以是一个字符串，也可以是一个 batch 的字符串列表。
        # truncate=True 可以避免 COCO caption 偶尔过长时报错。
        with torch.no_grad():
            tokens = clip.tokenize(captions, truncate=True).to(self.get_device())

            # Step 5: 输出 text_features。这里仍然返回未归一化特征，外层统一 normalize。
            text_features = self.clip_model.encode_text(tokens)
            return text_features.float()


class CLIPCOCOContrastiveModel(nn.Module):
    """
    最外层对比学习模型。

    forward 的输出:
        logits_per_image: shape = [B, B]
        logits_per_text : shape = [B, B]

    第 i 张图片和第 i 条 caption 是正样本。
    同一个 batch 里的其他 caption / image 自动作为负样本。
    """

    def __init__(self, text_model_name="ViT-B/32", device=None, download_root=None):
        super().__init__()

        self.text_encoder = FrozenCLIPTextEncoder(
            model_name=text_model_name,
            device=device,
            download_root=download_root,
        )
        self.image_encoder = CustomImageEncoder(embed_dim=self.text_encoder.embed_dim)

        # CLIP 原论文常用 temperature = 0.07，所以 logit_scale 初始为 log(1/0.07)。
        # 这是可训练参数，训练时会自动学习相似度分数的尺度。
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

        # 把你自己写的 image encoder 和 logit_scale 放到 text encoder 同一个 device。
        # 否则在有 CUDA 的机器上会出现 images 在 GPU、image_encoder 权重还在 CPU 的错误。
        self.to(self.text_encoder.get_device())

    def train(self, mode=True):
        # 正常训练你自己的 image encoder。
        super().train(mode)

        # 但 CLIP text encoder 是冻结的，所以始终保持 eval 状态。
        self.text_encoder.clip_model.eval()
        return self

    def forward(self, images, captions):
        # Step 1: 保证图片和 text encoder 在同一个 device。
        images = images.to(self.text_encoder.get_device())

        # Step 2: 分别得到 image/text features。
        # image_features 会有梯度，用来训练你的 image encoder。
        # text_features 没有梯度，因为 CLIP text encoder 已被冻结。
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(captions)

        # Step 3: 归一化后做矩阵乘法，得到 batch 内两两相似度。
        image_features = l2_normalize(image_features)
        text_features = l2_normalize(text_features)

        # Step 4: logit_scale.exp() 相当于 1 / temperature。
        # clamp 是为了防止训练中温度尺度爆掉。
        scale = self.logit_scale.exp().clamp(max=100)
        logits_per_image = scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text

    def contrastive_loss(self, images, captions):
        """
        一个最小训练 loss 示例。

        labels = [0, 1, 2, ..., B-1]
        表示第 i 张图片应该匹配第 i 条 caption。
        """

        logits_per_image, logits_per_text = self(images, captions)
        batch_size = logits_per_image.shape[0]
        labels = torch.arange(batch_size, device=logits_per_image.device)

        image_to_text_loss = F.cross_entropy(logits_per_image, labels)
        text_to_image_loss = F.cross_entropy(logits_per_text, labels)
        return (image_to_text_loss + text_to_image_loss) / 2


def build_model(text_model_name="ViT-B/32", device=None, download_root=None):
    """
    给训练脚本用的构造函数。

    示例:
        model = build_model(device="cuda")
        loss = model.contrastive_loss(images, captions)
        loss.backward()
        optimizer.step()
    """

    return CLIPCOCOContrastiveModel(
        text_model_name=text_model_name,
        device=device,
        download_root=download_root,
    )


def train_step_example(model, optimizer, images, captions):
    """
    单步训练伪代码，方便你写 model_train.py 时照着搬。

    注意：
        captions 必须和 images 一一对应。
        例如 images[0] 对应 captions[0]，images[1] 对应 captions[1]。
    """

    model.train()

    # CLIP text encoder 被冻结，但 model.train() 会把所有子模块切到 train。
    # 所以这里再把它切回 eval，避免 dropout/bn 之类状态改变。
    model.text_encoder.clip_model.eval()

    optimizer.zero_grad(set_to_none=True)
    loss = model.contrastive_loss(images, list(captions))
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(device=device)

    print("device:", next(model.image_encoder.parameters()).device)

    # torchsummary 只能处理“输入是张量”的模型。
    # 整个 CLIPCOCOContrastiveModel 的 forward 需要 images 和 captions 两个输入，
    # captions 是字符串列表，所以不要直接 summary(model, ...)。
    # 如果想看结构，summary 你自己写的 image_encoder 就可以。
    from torchsummary import summary
    summary(model.image_encoder, (3, 224, 224), device=device)

    # # 如果想测试整个对比学习模型，需要同时给图片和文字。
    # # 这里构造一个假的 batch，只检查 forward 能不能跑通。
    # model.eval()
    # images = torch.randn(2, 3, 224, 224)
    # captions = ["a dog on the grass", "a cat on the sofa"]
    # with torch.no_grad():
    #     logits_per_image, logits_per_text = model(images, captions)

    # print("logits_per_image:", logits_per_image.shape)
    # print("logits_per_text:", logits_per_text.shape)
