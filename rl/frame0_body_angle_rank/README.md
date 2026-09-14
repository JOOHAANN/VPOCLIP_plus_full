# Frame-0 anatomical body-angle ranked active-view experiment

This is an isolated experiment.  It does not rewrite the historical `v1`–`v9`
code, caches, checkpoints, or anchor reports.

## Label/action definition

For every valid video independently, the first usable skeleton row for video
frame 0 is loaded from the released 25-joint 3-D skeleton.  The body frame is
formed from shoulders, hips, torso and head.  In that camera-local coordinate
system the camera is the origin, so the vector from the body centre to the
camera is `-body_center`.  The signed angle is:

```text
angle = atan2(dot(person_to_camera, body_right),
              dot(person_to_camera, body_forward))
```

Negative is body-left and positive is body-right.  The four valid views of one
episode are sorted by this numeric signed angle.  The policy action is the
ordinal rank `0`, `1`, `2`, or `3`; `rank_to_original.npy` and the per-episode
metadata map that action back to the original camera/video.  The original
camera IDs are never treated as angular order.

No old optical-axis angle is used as a fallback.  If an original skeleton is
missing, cache construction stops before writing any destination arrays.

## Models

The run includes `v1`, the preserved legacy angle+object implementation under
the explicit name `v2_legacy`, `v3`, `v4`, `v5`, `v6`, `object_only`, and
`geometry_only`.  The repository contains no independent trajectory-v2 module
or checkpoint; the manifest records this rather than inventing a canonical v2.

All variants use the same frozen VPOCLIP cache and offline seen-class utility
target as their historical counterpart.  Only the view-axis ordering and the
angle feature contract change.  VPOCLIP logits/z are not selector state.

## Commands

```bash
cd /home/youhan/ws/VPOCLIP_plus_full
python3 -m rl.frame0_body_angle_rank.build_ranked_cache
bash rl/frame0_body_angle_rank/run_all.sh
```

When the protected ETRI skeleton files are not yet present, the resumable
driver can wait without using GPU:

```bash
bash rl/frame0_body_angle_rank/wait_for_skeleton_and_run.sh
```

It detects the extracted P001–P050 and P051–P100 ranges and then automatically
builds the ranked cache and launches the variants.

If the ETRI session is refreshed, the supplied archive helper can download and
validate both protected skeleton archives; it intentionally keeps the zip
files for recovery:

```bash
bash rl/frame0_body_angle_rank/download_skeleton_archives_after_login.sh
```

The outer run executes one model process at a time with a 16% per-process GPU
memory cap, leaving headroom for another GPU workload.  Logs and outputs are:

```text
/home/youhan/ws/VPOCLIP_plus_full/logs/frame0_body_angle_rank_v1_v6/
/home/youhan/ws/VPOCLIP_plus_full/work_dir/frame0_body_angle_rank_v1_v6/
```
