import numpy as np
import torch

def Load_video_feature():
    # 这里假设你的视频特征已经保存在一个 NumPy 文件中
    video_feature_np = np.load('video_feature_raw.npy')  # 替换为你的文件路径
    video_feature_torch = torch.from_numpy(video_feature_np).float()  # 转换为 PyTorch 张量并确保类型为 float
    return video_feature_torch

