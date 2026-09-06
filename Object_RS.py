import torch


def get_spatial_map(grid_size, device=None, dtype=torch.float32):
    x = torch.linspace(-1, 1, grid_size, device=device, dtype=dtype)
    y = torch.linspace(1, -1, grid_size, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1)


def get_distance_map(spatial_map, x_o, y_o):
    x_o = torch.as_tensor(x_o, device=spatial_map.device, dtype=spatial_map.dtype)
    y_o = torch.as_tensor(y_o, device=spatial_map.device, dtype=spatial_map.dtype)
    distances = torch.sqrt(
        (spatial_map[..., 0] - x_o[..., None, None]) ** 2
        + (spatial_map[..., 1] - y_o[..., None, None]) ** 2
    )
    return 1.0 / (distances + 1e-6)


def get_RS_map(object_feature, object_xy, presence=None, max_distance_weight=10.0):
    """Build object RS maps.

    Args:
        object_feature: [B, O] object values, usually presence/confidence.
        object_xy: [B, O, 2] normalized object centers in [-1, 1].
        presence: optional [B, O] binary mask; absent objects become zero maps.

    Returns:
        [B, O, 6, 6] by default when the caller uses 50 object categories.
    """

    if object_xy.ndim != 3 or object_xy.shape[-1] != 2:
        raise ValueError(f"Expected object_xy [B,O,2], got {tuple(object_xy.shape)}")

    grid_size = 6
    spatial_map = get_spatial_map(
        grid_size,
        device=object_xy.device,
        dtype=object_xy.dtype,
    )
    distance_map = get_distance_map(spatial_map, object_xy[..., 0], object_xy[..., 1])
    distance_map = torch.clamp(distance_map, max=max_distance_weight)

    values = object_feature.to(device=object_xy.device, dtype=object_xy.dtype)
    output = values[..., None, None] * distance_map
    if presence is not None:
        output = output * presence.to(device=object_xy.device, dtype=object_xy.dtype)[..., None, None]
    return output


def get_multiplicative_RS_map(instance_class_slots, instance_xy, num_classes=50, confidence=None, valid=None, grid_size=6, max_distance_weight=10.0):
    """Build object maps from per-instance detections with same-class products.

    Args:
        instance_class_slots: [B, K] class slot in [0, num_classes).
        instance_xy: [B, K, 2] normalized centers in [-1, 1].
        confidence: optional [B, K] multiplier for each instance.
        valid: optional [B, K] mask.

    Returns:
        [B, num_classes, grid_size, grid_size]
    """

    if instance_class_slots.ndim != 2 or instance_xy.ndim != 3 or instance_xy.shape[-1] != 2:
        raise ValueError(
            f"Expected class slots [B,K] and xy [B,K,2], got {tuple(instance_class_slots.shape)} and {tuple(instance_xy.shape)}"
        )

    device = instance_xy.device
    dtype = instance_xy.dtype
    batch_size, num_instances = instance_class_slots.shape
    output = torch.zeros(batch_size, num_classes, grid_size, grid_size, device=device, dtype=dtype)
    seen = torch.zeros(batch_size, num_classes, device=device, dtype=torch.bool)
    spatial_map = get_spatial_map(grid_size, device=device, dtype=dtype)

    if valid is None:
        valid = (instance_class_slots >= 0) & (instance_class_slots < num_classes)
    if confidence is None:
        confidence = torch.ones(batch_size, num_instances, device=device, dtype=dtype)

    for instance_idx in range(num_instances):
        slots = instance_class_slots[:, instance_idx].long()
        mask = valid[:, instance_idx] & (slots >= 0) & (slots < num_classes)
        if not torch.any(mask):
            continue

        e_grid = get_distance_map(
            spatial_map,
            instance_xy[:, instance_idx, 0],
            instance_xy[:, instance_idx, 1],
        )
        e_grid = torch.clamp(e_grid, max=max_distance_weight)
        e_grid = e_grid * confidence[:, instance_idx, None, None].to(device=device, dtype=dtype)

        sample_indices = torch.arange(batch_size, device=device)[mask]
        class_indices = slots[mask]
        old_values = output[sample_indices, class_indices]
        first_instance = ~seen[sample_indices, class_indices]
        output[sample_indices, class_indices] = torch.where(
            first_instance[:, None, None],
            e_grid[mask],
            old_values * e_grid[mask],
        )
        seen[sample_indices, class_indices] = True

    return output
