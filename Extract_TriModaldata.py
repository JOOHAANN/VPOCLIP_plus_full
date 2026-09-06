import numpy as np
import torch


def load_pose_feature():
    pose_feature_np = np.load("data/contrastive_train_data/trimodal_train_pose.npy")
    return torch.from_numpy(pose_feature_np).float()

def load_joint_location():
    joint_location_np = np.load("data/contrastive_train_data/trimodal_train_joint_xy.npy")
    return torch.from_numpy(joint_location_np).float()


def load_object_feature():
    object_feature_np = np.load("data/contrastive_train_data/trimodal_train_object.npy")
    object_feature = torch.from_numpy(object_feature_np).float()
    if object_feature.ndim == 4:
        object_feature = object_feature.unsqueeze(1)
    return object_feature


def load_video_feature():
    video_feature_np = np.load("data/contrastive_train_data/trimodal_train_video.npy")
    return torch.from_numpy(video_feature_np).float()

if __name__ == "__main__":
    
    joint_location = load_joint_location()
    pose_feature = load_pose_feature()
    object_feature = load_object_feature()
    video_feature = load_video_feature()

    print(pose_feature)
