"""v4 state builder using the released 3-D skeleton yaw.

The historical v4 state builder derives two orientation channels from the
RTMPose image-plane torso.  This isolated replacement keeps the same network
shape and causal trajectory contract, but fills those two channels from the
frame-0 anatomical yaw stored in ``body_yaw_deg.npy``.  The scalar is repeated
over the 13 temporal slots because this experiment defines the label from the
video's frame 0; the RTMPose pose remains the source of the person
trajectory/image position channels.
"""

from __future__ import annotations

import torch

from .. import multistep_angle_object_trajectory_policy_v1 as v1


def build_trajectory_state_v4_true_yaw(
    raw: object,
    episodes: torch.Tensor,
    history: torch.Tensor,
    history_count: torch.Tensor,
) -> dict[str, torch.Tensor]:
    episodes = episodes.long()
    history = history.long()
    count = history_count.long().clamp(1, history.shape[1])
    batch = episodes.shape[0]
    observed = (
        torch.arange(history.shape[1], device=episodes.device)[None, :]
        < count[:, None]
    )
    observed_f = observed.float()
    denominator = count.float().clamp_min(1.0)[:, None]
    current = history.gather(1, (count - 1)[:, None]).squeeze(1)

    observed_pose = raw.pose[episodes[:, None], history]
    base_person = v1.person_trajectory_from_pose(observed_pose)
    # ``body_yaw_deg.npy`` is explicitly stored in degrees.  Convert before
    # evaluating trigonometric functions; using ``sin(degrees)`` would make
    # the v4 branch depend on the numeric unit rather than the actual yaw.
    yaw_deg = raw.body_yaw_deg[episodes[:, None], history]
    yaw_rad = torch.deg2rad(
        torch.nan_to_num(yaw_deg, nan=0.0, posinf=0.0, neginf=0.0)
    )
    yaw_sincos = torch.stack((yaw_rad.sin(), yaw_rad.cos()), -1)
    yaw_sincos = yaw_sincos[:, :, None, :].expand(
        -1, -1, base_person.shape[2], -1
    )
    person_per_view = torch.cat((base_person, yaw_sincos), -1)
    person = (person_per_view * observed_f[:, :, None, None]).sum(1) / denominator[:, :, None]

    object_track = raw.object_position_track[episodes[:, None], history]
    object_xy = object_track[..., 1:3]
    relative_xy = object_xy - person_per_view[..., :2][..., None, :]
    object_sensor = torch.cat((object_track, relative_xy), -1)
    object_sensor = (
        object_sensor * observed_f[:, :, None, None, None]
    ).sum(1) / denominator[:, :, None, None]

    observed_views = torch.zeros(
        (batch, v1.NUM_VIEWS), dtype=torch.bool, device=episodes.device
    )
    observed_views.scatter_(1, history, observed)
    current_relative_angle = raw.geometry[episodes, current, :2]
    geometry = raw.geometry[episodes, :, :2]
    sin_delta = (
        geometry[..., 0] * current_relative_angle[:, None, 1]
        - geometry[..., 1] * current_relative_angle[:, None, 0]
    )
    cos_delta = (geometry * current_relative_angle[:, None]).sum(-1)
    valid = raw.valid[episodes] & raw.reachable[episodes, current] & (~observed_views)
    candidate = torch.cat(
        (
            geometry,
            torch.stack((sin_delta, cos_delta), -1),
            current_relative_angle[:, None].expand(-1, v1.NUM_VIEWS, -1),
            valid[..., None].float(),
        ),
        -1,
    )
    return {
        "person_trajectory": person,
        "object_trajectory": object_sensor,
        "current_relative_angle": current_relative_angle,
        "candidate": candidate,
        "mask": valid,
    }
