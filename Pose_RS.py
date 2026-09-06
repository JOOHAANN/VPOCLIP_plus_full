import torch


def get_spatial_map(grid_size, device=None, dtype=torch.float32):
    x = torch.linspace(-1, 1, grid_size, device=device, dtype=dtype)
    y = torch.linspace(1, -1, grid_size, device=device, dtype=dtype)

    Y, X = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((X, Y), dim=-1)


def get_distance_map(spatial_map, x_s, y_s, max_weight=10.0):
    x_s = torch.as_tensor(
        x_s,
        device=spatial_map.device,
        dtype=spatial_map.dtype
    )
    y_s = torch.as_tensor(
        y_s,
        device=spatial_map.device,
        dtype=spatial_map.dtype
    )

    distances = torch.sqrt(
        (spatial_map[..., 0] - x_s[..., None, None]) ** 2
        + (spatial_map[..., 1] - y_s[..., None, None]) ** 2
    )

    weights = 1.0 / (distances + 1e-6)
    return torch.clamp(weights, max=max_weight)


def get_RS_map(pose_feature_map, x_s, y_s):
    # pose_feature_map: [B, T, J, H, W]
    # x_s/y_s: [B, T, J]
    grid_size = pose_feature_map.shape[-1]

    spatial_map = get_spatial_map(
        grid_size,
        device=pose_feature_map.device,
        dtype=pose_feature_map.dtype
    )

    distance_map = get_distance_map(spatial_map, x_s, y_s)

    pose_feature_map = torch.nan_to_num(pose_feature_map, nan=0.0, posinf=0.0, neginf=0.0)
    return pose_feature_map * distance_map


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    pose_feature_map = torch.randn(
        6, 6,
        device=device
    )

    spatial_map = get_spatial_map(
        grid_size=6,
        device=device
    )

    distance_map = get_distance_map(
        spatial_map,
        x_s=0.2,
        y_s=-0.3
    )

    RS_map = get_RS_map(
        pose_feature_map,
        x_s=0.2,
        y_s=-0.3
    )

    print("spatial_map shape:", spatial_map.shape)
    print("distance_map shape:", distance_map.shape)
    print("RS_map shape:", RS_map.shape)
    print("RS_map device:", RS_map.device)
