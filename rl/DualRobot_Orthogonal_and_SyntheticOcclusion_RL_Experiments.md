# 两阶段实验计划：双机器人几何协同 + 虚拟遮挡强化学习

## 总体顺序

严格按照下面顺序执行，不要并行推进，也不要在第一阶段结果未完成时自动开始第二阶段的大规模 RL：

```text
Experiment 1
双机器人 / 双视角几何协同
Preferred Orthogonal（非学习）
        ↓
完成 equal-budget benchmark + bootstrap
        ↓
Experiment 2
2D View-Consistent Synthetic Occlusion
        ↓
先做 active-view diagnostic
        ↓
只有 diagnostic 成功后
        ↓
训练 RL
```

两个实验回答不同问题：

- **Experiment 1**：在原始 ETRI、几乎无遮挡的条件下，简单的 class-agnostic 几何互补双视角是否已经足够有效？
- **Experiment 2**：如果人为加入具有视角一致性的遮挡，使“移动”真正改变可观测性，RL 是否终于能够学习出有意义的避障/绕遮挡视角策略？

---

# Experiment 1 — Dual-Robot Preferred Orthogonal Cooperative Perception

## 1. 实验目标

使用 ETRI 同步多视角数据离线模拟两台机器人：

```text
Robot A:
先确定观察位置，作为 anchor robot

Robot B:
根据 Robot A 与人体的相对观察方向
使用 Preferred Orthogonal 几何规则
选择最互补的第二观察位置

Robot A + Robot B:
同时观察同一 temporal window
同时进行 HAR
最后做双视角融合
```

本实验的主动布位算法 **不使用 RL，不训练 viewpoint policy**。

主动选位部分属于：

```text
class-agnostic
human-centric
geometry-based
non-learning viewpoint coordination
```

VPOCLIP 识别器仍然是 learning-based。

---

## 2. 重要科学问题

必须区分两个不同收益：

### Extra-sensor gain

```text
Dual-Robot Random Pair
-
Single-Robot Fixed
```

这个收益主要来自“多一个相机/机器人”。

### Coordination gain

```text
Preferred Orthogonal Pair
-
Dual-Robot Random Pair
```

这个才是几何协同算法本身的贡献。

最终报告中不能只强调：

```text
Ours - Single Fixed
```

必须同时报告：

```text
Ours - Random Pair
Ours - Fixed Pair
```

---

## 3. 数据与协议

保持当前已有 protocol：

```text
Seen train
Seen validation / subject-held-out validation
Strict true unseen test
```

当前 true unseen：

```text
[0, 2, 26, 34, 50]
```

重要：

```text
禁止使用 true unseen 调：
- camera angle
- orthogonal target angle
- fusion weight
- threshold
- motion-cost weight
- model checkpoint
```

true unseen 只用于最终 locked evaluation。

---

## 4. ETRI 双机器人离线模拟的含义

ETRI 原始数据是固定多相机同步采集，因此这里不要在论文/报告中声称已经进行了真实双机器人运动实验。

当前阶段准确称为：

```text
dual-robot sensing proxy
或
dual-agent synchronized multi-view simulation using ETRI cameras
```

每一个 ETRI camera view 代表机器人可到达的离散观察位置。

如果以后有真实机器人，再做 physical deployment validation。

---

## 5. Camera Relative Angle 配置

ETRI 如果没有 camera calibration，不要伪造 intrinsic/extrinsic。

建立一个配置文件：

```text
camera_view_angles.yaml
```

例如：

```yaml
view0: 0
view1: 90
view2: 180
view3: 270
```

但这里的具体映射必须根据：

```text
ETRI camera ordering
已有 metadata
人工快速检查几个同步样本
```

确定。

如果只能知道离散环绕顺序，也可以使用：

```text
0 / 90 / 180 / 270
```

作为 pseudo-angle。

报告中明确称为：

```text
relative camera-view angle
```

不要称为 calibrated world pose。

---

## 6. Robot A — Anchor Placement

主实验为了和之前结果公平比较：

```text
Robot A = fixed View0
```

额外做 robustness：

```text
Robot A = random valid anchor
```

但主表优先固定 View0。

---

## 7. Robot B — Preferred Orthogonal

设 Robot A 相对人体方向：

\[
\theta_A
\]

候选 Robot B 位置：

\[
\theta_j
\]

计算 circular angular separation：

\[
\Delta\theta_j
=
\min(
|\theta_j-\theta_A|,
360-|\theta_j-\theta_A|
)
\]

Preferred Orthogonal 选择：

\[
j^*
=
\arg\min_j
|\Delta\theta_j-\theta^*|
\]

默认：

\[
\theta^*=90^\circ
\]

如果多个 candidate 并列：

1. 优先 movement cost 小的；
2. 如果没有 movement cost，使用固定 deterministic tie-break；
3. 不允许查看 HAR GT 或 future candidate logits 来 tie-break。

---

## 8. 可选几何版本

只做小规模 ablation，不要过度搜索：

```text
Target angle:
60°
90°
120°
```

参数只能在 validation 上选择。

主方法优先保持：

```text
90° Preferred Orthogonal
```

以保持方法简单和可解释。

---

## 9. 双视角 HAR Fusion

为了单独测“布位”贡献，第一版 fusion 不要复杂化。

主 baseline：

\[
z_{fused}
=
\frac{z_A+z_B}{2}
\]

其中：

```text
z_A = Robot A logits
z_B = Robot B logits
```

额外可测试：

```text
max confidence fusion
weighted logits fusion
```

但 fusion 参数不能用 true unseen 调。

暂时不要引入新的大型 cross-view network，否则无法判断收益来自布位还是 fusion。

---

## 10. Experiment 1 必须比较的 Baselines

至少包含：

### 1. Single-Robot Fixed

```text
Robot A at View0
1 view
```

### 2. Single-Robot Random-2

模拟单机器人：

```text
View0
+
随机第二视角
```

用于和之前 Random-2 结果对齐。

### 3. Dual-Robot Fixed Pair

固定两个 camera pair，例如：

```text
View0 + predefined View1
```

用于判断“固定双机”本身的收益。

### 4. Dual-Robot Random Pair

```text
Robot A = View0
Robot B = random legal candidate
```

这是最重要的 equal-budget baseline。

### 5. Dual-Robot Maximum Separation

选择与 Robot A 角度差最大的 view。

用于区分：

```text
90° complementary
vs
单纯越远越好
```

### 6. Dual-Robot Preferred Orthogonal

本实验 proposed method。

### 7. All Valid Views

全部 view 融合。

### 8. Oracle-2

利用 GT 离线选择最优第二视角。

只能作为 upper bound。

---

## 11. Experiment 1 主结果表

Codex 最终必须生成：

| Method | # Views | Seen Top-1 | True Unseen Top-1 | Δ vs Single Fixed | Δ vs Random Pair | 95% CI vs Random Pair |
|---|---:|---:|---:|---:|---:|---:|
| Single Fixed | 1 |  |  |  |  |  |
| Single Random-2 | 2 |  |  |  |  |  |
| Dual Fixed Pair | 2 |  |  |  |  |  |
| Dual Random Pair | 2 |  |  |  |  |  |
| Max Separation Pair | 2 |  |  |  |  |  |
| Preferred Orthogonal Pair | 2 |  |  |  |  |  |
| All Views | 4 |  |  |  |  |  |
| Oracle-2 | 2 |  |  |  |  |  |

---

## 12. Experiment 1 统计测试

对：

```text
Preferred Orthogonal
vs
Dual Random Pair
```

必须做 paired bootstrap：

```text
>= 5000 bootstrap samples
```

报告：

```text
accuracy difference
95% CI
```

同时做：

```text
Preferred Orthogonal
vs
Dual Fixed Pair
```

---

## 13. Per-Class Analysis

true unseen `[0,2,26,34,50]` 分类别报告：

```text
Single Fixed
Random Pair
Preferred Orthogonal
Oracle-2
```

防止总体提升完全由单一 action class 驱动。

---

## 14. Experiment 1 Go / No-Go

### Positive result

如果出现：

```text
Preferred Orthogonal > Random Pair
```

且：

```text
多个 split / class 中大多数为正
CI 明显向正方向移动
```

则 geometry-based dual-robot coordination 值得继续作为主方法。

### Weak but useful result

如果：

```text
Preferred Orthogonal 仅比 Random Pair 高 0.5~1 pp
```

但：

```text
稳定
并且 movement / placement cost 更低
```

仍然保留，论文重点转向：

```text
accuracy-cost tradeoff
```

### Negative result

如果：

```text
Preferred Orthogonal ~= Random Pair
```

且所有 split 都无稳定优势，则不要包装成“学习成功”。

继续执行 Experiment 2，验证是否是原始 ETRI 缺乏 active-perception pressure。

---

# Experiment 2 — View-Consistent Synthetic Occlusion + RL

## 15. 实验目标

验证下面假设：

> ETRI 原始环境为了清晰观察老人行为，遮挡较少，因此不同 viewpoint 之间的 action value 差异不足，导致之前的 RL / NBV / VoI 方法难以学到有效移动策略。

本实验人为生成：

```text
view-dependent
multi-view consistent
human-centric synthetic occlusion
```

使：

```text
某些视角被遮挡
侧向移动后遮挡减少
相反方向移动后遮挡增加
```

然后重新检查：

```text
Oracle gap
recoverable ratio
visibility-to-HAR relationship
RL performance
```

---

# 16. 本实验不要直接做 3D calibration

当前 ETRI 如果没有 camera intrinsic/extrinsic：

```text
不要尝试伪造 calibration
不要先做 3D reconstruction
不要先上 NeRF / Gaussian Splatting
```

第一版使用：

```text
2D top-down pseudo-world
+
relative camera angles
+
virtual occluder
```

---

# 17. Top-Down Pseudo-World

定义人体中心：

\[
H=(0,0)
\]

四个 camera：

\[
C_i=
R[
\cos\theta_i,
\sin\theta_i
]
\]

其中：

```text
theta_i
```

来自 Experiment 1 的 `camera_view_angles.yaml`。

不需要真实米制位置。

例如：

```text
Human = origin
Camera radius R = 1.0
```

---

# 18. Virtual Occluder

每个 synchronized multi-view clip 只采样一次：

```text
occluder_angle
occluder_distance
occluder_width
severity
target_region
```

然后整个 clip 固定。

示例：

```text
occluder_angle = -30°
occluder_distance = 0.4
severity = medium
target_region = upper_body
```

相同 sample 的所有 view 必须共享同一个 virtual occluder。

禁止每个 view 独立随机 mask。

---

# 19. 计算 View-Dependent Occlusion

第一版可以使用 line-of-sight distance。

对于 camera \(C_i\) 到人体 \(H\) 的视线段：

\[
L_i=C_i\rightarrow H
\]

计算 virtual occluder \(O\) 到该线段的最短距离：

\[
d_i = distance(O,L_i)
\]

定义 occlusion severity：

\[
s_i
=
s_{max}
\exp
\left(
-\frac{d_i^2}{2\sigma_d^2}
\right)
\]

同时检查 occluder 是否位于：

```text
camera -> human
```

之间。

如果不在视线前方：

```text
s_i = 0
```

---

## 20. 遮挡方向

使用有符号 lateral distance / 2D cross product 判断：

```text
occluder 在当前 camera 视野中位于人体左边
或
位于人体右边
```

然后 mask 从对应方向进入 human bbox。

目标效果：

```text
Front view:
中/重度遮挡

Left view:
更严重

Right view:
轻微或完全消失
```

这必须由同一个 virtual occluder 几何产生，而不是硬编码每个 camera 的 mask。

---

# 21. Human-Centric 2D Mask

优先从已有：

```text
person bbox
或
2D skeleton
```

获得每帧 human bbox：

```text
x1(t), y1(t), x2(t), y2(t)
```

如果没有 person bbox：

```text
从有效 2D keypoints 的 min/max
生成 bbox
再扩大 5~10% margin
```

mask width：

```text
mask_width
=
severity * human_bbox_width
```

mask height 第一版：

```text
0.6 ~ 0.8 * human_bbox_height
```

优先遮：

```text
upper body
hands
face / torso interaction region
```

而不是随机背景。

---

# 22. Temporal Consistency

同一个 clip：

```text
occluder world parameters 固定
```

每帧只根据 person bbox 更新 mask 在 image plane 的位置。

为避免 bbox 抖动：

```text
对 bbox center / size 做 EMA smoothing
```

禁止每帧随机改变 mask。

---

# 23. Mask Appearance

按阶段：

### Debug

```text
gray rectangle
```

### Main synthetic benchmark

推荐：

```text
blurred / textured rectangle
```

避免纯黑块造成过强 synthetic artifact。

可以从非人体背景 crop texture 后填充 mask。

第一轮不要做复杂 object cutout。

---

# 24. 不要重新编码全部 MP4

优先：

```text
on-the-fly masking
```

在 Dataset / dataloader 中：

```python
frames = load_clip(...)
frames = apply_view_consistent_occlusion(...)
model_input = preprocess(frames)
```

不要默认保存大量新视频。

可以只缓存：

```text
sample_id
occluder_angle
occluder_distance
severity
target_region
random_seed
```

确保所有方法使用完全相同 synthetic episode。

---

# 25. 极其重要：避免“无遮挡辅助模态泄漏”

在当前 VPOCLIP pipeline 中检查：

```text
RGB
Skeleton
Object
```

是否来自原始未遮挡缓存。

如果 RGB 被 mask，但 skeleton / object 仍然来自原始 clear video，会形成信息泄漏：

```text
视觉上被挡
但模型仍通过未遮挡 skeleton/object 看见完整人体
```

这是不允许的。

Codex 必须首先检查当前数据流，然后选择下面一种一致实现：

### Preferred

mask 在所有 feature extraction 之前加入：

```text
masked frame
-> pose/object extraction
-> VPOCLIP
```

重新生成受遮挡输入。

### 如果 pose/object 重建成本过高

根据 mask region：

```text
落在 mask 内的 keypoints -> missing / zero / invalid
被 mask 大面积覆盖的 object -> drop / invalid
```

不能继续使用完整无遮挡 pose/object。

### 如果做不到一致 corruption

先只运行一个明确标注的：

```text
RGB-only synthetic occlusion diagnostic
```

不要把其结果误称为 full multimodal VPOCLIP result。

---

# 26. Occlusion Severity Levels

固定四档：

```text
None
Mild
Medium
Heavy
```

建议初始范围：

```text
Mild:
15~30% target-region occlusion

Medium:
30~50%

Heavy:
50~70%
```

具体参数只在 seen validation 上确认。

---

# 27. RL 之前必须先做 Diagnostic

在任何 PPO/RL 训练前，分别对：

```text
None
Mild
Medium
Heavy
```

计算：

```text
View0
Random-2
Preferred Orthogonal-2
Oracle-2
Recoverable ratio
```

主表：

| Occlusion | View0 | Random-2 | Orthogonal-2 | Oracle-2 | Oracle-Random Gap | Recoverable Ratio |
|---|---:|---:|---:|---:|---:|---:|
| None | | | | | | |
| Mild | | | | | | |
| Medium | | | | | | |
| Heavy | | | | | | |

---

# 28. Active-View Signal Diagnostic

因为 synthetic generator 知道每个 candidate 的实际遮挡程度，可定义：

```text
least_occluded_candidate
```

然后计算：

```text
P(least_occluded_candidate == HAR_best_candidate)
```

其中：

```text
HAR_best_candidate
```

只用于离线 diagnostic，由 GT utility 确定。

如果有 3 个候选，random action agreement 约为：

```text
33.3%
```

希望 synthetic occlusion 后：

```text
least-occluded vs HAR-best agreement
明显 > random
```

例如达到：

```text
50%+
```

更理想：

```text
60~80%
```

---

# 29. Experiment 2 RL Go / No-Go Gate

只有同时满足大部分以下条件才开始 RL：

```text
1. Oracle-Random gap 相比 None 明显扩大
2. Recoverable ratio 明显提高
3. least-occluded candidate 与 HAR-best candidate 的 agreement 明显高于 random
4. Orthogonal / low-occlusion view 在 synthetic occlusion 下出现稳定优势
```

如果这些都没出现：

```text
不要训练 PPO
```

说明 synthetic mask 没有创造出合理 active-view pressure，应先修改遮挡生成器。

---

# 30. RL Problem Formulation

如果 diagnostic 通过，构造真正 sequential RL。

### Episode

```text
start at one view
observe masked clip
choose:
    STOP
    move to another valid view
observe new masked synchronized view
optionally move again
STOP
```

最多：

```text
2 moves
```

第一版不要无限 horizon。

---

# 31. RL State

只允许使用当前/历史已经观察到的信息：

```text
current VPOCLIP feature / logits
current entropy
current top1-top2 margin

current observed occlusion ratio
current observed mask side
current human bbox / visible keypoints

current camera relative angle
visited-view mask
history fused logits
number of views used
movement cost
```

注意：

```text
current observed occlusion ratio / side
```

可以从当前 synthetic mask 或当前 image-derived mask 计算，因此是 causal observable information。

---

# 32. RL 严禁输入

禁止：

```text
GT action label
future candidate RGB
future candidate logits
future candidate VPOCLIP feature

future candidate occlusion severity
future candidate visibility
latent occluder_angle
latent occluder_position

HAR Oracle action
```

特别注意：

**不能直接把 synthetic generator 的隐藏 occluder_angle 给 policy。**

否则 policy 是在读取模拟器真值，不是真正 active vision。

---

# 33. RL Action

四视角数据中：

```text
STOP
move_to_view0
move_to_view1
move_to_view2
move_to_view3
```

已经访问的 view：

```text
mask as illegal
```

当前所在 view：

```text
不能原地重复选择
```

---

# 34. RL Reward

推荐第一版：

\[
r_t
=
\alpha\Delta HAR_t
+
\beta\Delta Obs_t
-
\lambda C_{move}
-
\mu C_{view}
\]

其中：

### HAR evidence gain

训练 seen class 时：

\[
\Delta HAR_t
=
Margin_{t+1}^{GT}
-
Margin_t^{GT}
\]

只用于 reward，不作为 state。

### Observable quality gain

可以使用：

\[
\Delta Obs
=
Occ_t-Occ_{t+1}
\]

即遮挡减少。

但不要让 visibility reward 完全主导。

建议从：

```text
alpha = 1.0
beta = 0.2
lambda = 0.05
mu = 0.02
```

这类量级开始，再在 validation 上有限调整。

不要在 true unseen 调 reward。

---

# 35. Terminal Reward

STOP 后：

```text
correct final HAR:
+ positive reward

incorrect:
0 或 small negative
```

第一版保持简单。

---

# 36. RL Algorithm

如果现有项目已经有 PPO infrastructure：

```text
优先继续 PPO
```

但从小网络开始：

```text
MLP 64-64
```

然后最多比较：

```text
128-128
```

不要直接上大 Transformer。

当前 experiment 的目标是验证：

```text
遮挡使 active movement 变得可学习
```

不是证明大网络更强。

---

# 37. RL Training Data

只在 seen classes 上训练。

Synthetic occluder 每个 epoch / episode 随机采样：

```text
direction
distance
severity
target region
```

形成 domain randomization。

训练时混合：

```text
None
Mild
Medium
Heavy
```

例如：

```text
10% None
30% Mild
35% Medium
25% Heavy
```

具体比例只能在 validation 调。

---

# 38. 防止 RL 学 camera ID shortcut

必须：

```text
随机 start view
随机 occluder direction
随机 severity
随机 subject/sample order
```

不要总是：

```text
View0 被挡
View2 最清楚
```

否则 policy 只会记：

```text
View0 -> View2
```

而不是学“避开遮挡”。

---

# 39. RL Baselines

Synthetic occlusion 下至少比较：

```text
Fixed View
Random Move
Preferred Orthogonal
Greedy Current-Visibility heuristic
PPO / RL
Oracle
```

### Greedy Current-Visibility heuristic

只能根据当前 observation：

```text
mask side
camera geometry
```

使用简单规则选择侧向移动。

这是很重要的 baseline，用于判断 RL 是否真的优于 hand-designed avoidance。

---

# 40. RL 主结果表

| Occlusion | Fixed | Random | Orthogonal | Visibility Heuristic | RL | Oracle |
|---|---:|---:|---:|---:|---:|---:|
| None | | | | | | |
| Mild | | | | | | |
| Medium | | | | | | |
| Heavy | | | | | | |

同时报告：

```text
average views
average moves
average occlusion after move
recoverable success rate
```

---

# 41. True Unseen 测试

最终在 locked true unseen：

```text
[0,2,26,34,50]
```

上测试 synthetic occlusion transfer。

重要：

```text
相同 sample
相同 occluder seed
相同 severity
```

所有方法必须 paired evaluation。

---

# 42. RL 成功判据

理想结果不是只看：

```text
RL > Fixed
```

而是：

```text
RL > Random
RL > Preferred Orthogonal
RL > simple visibility heuristic
```

尤其是在：

```text
Medium
Heavy
```

遮挡下。

如果：

```text
RL 只比 Fixed 高
但 ~= Random / heuristic
```

则不能声称 RL 学到了有效主动视角策略。

---

# 43. 两个 Experiment 的最终联合表

最后生成一个总表：

| Setting | Method | True Unseen Top-1 | Δ vs Equal-Budget Random | Avg Views | Notes |
|---|---|---:|---:|---:|---|
| Original ETRI | Single Fixed | | | 1 | |
| Original ETRI | Dual Random Pair | | | 2 | |
| Original ETRI | Dual Preferred Orthogonal | | | 2 | |
| Synthetic Mild | Random | | | | |
| Synthetic Mild | RL | | | | |
| Synthetic Medium | Random | | | | |
| Synthetic Medium | RL | | | | |
| Synthetic Heavy | Random | | | | |
| Synthetic Heavy | RL | | | | |

---

# 44. Codex 最终必须回答的科学问题

Experiment 1：

```text
1. 双机器人同步双视角相对单机器人提升多少？
2. 其中多少是“多一个 sensor”的收益？
3. Preferred Orthogonal 相对 equal-budget Random Pair 到底提升多少？
4. 这个提升是否跨 unseen classes 稳定？
```

Experiment 2：

```text
1. View-consistent synthetic occlusion 是否扩大 Oracle-Random gap？
2. Recoverable ratio 是否随着遮挡增加？
3. 遮挡较少的 candidate 是否更可能成为 HAR-best candidate？
4. RL 是否学会朝减小遮挡的方向移动？
5. RL 是否在 true unseen action 上超过 Random / Orthogonal / heuristic？
6. 原始 ETRI 中 RL 失败是否可以部分解释为 active-perception pressure 不足？
```

---

# 45. 自动执行顺序

Codex 严格执行：

```text
PHASE 1
检查数据视角映射
建立 relative camera angle config

PHASE 2
运行 Experiment 1
不训练 viewpoint model

PHASE 3
输出 Experiment 1 benchmark + bootstrap

PHASE 4
实现 2D top-down virtual occluder
先可视化若干 synchronized samples
确认：
front / left / right 的遮挡变化符合几何规律

PHASE 5
检查 multimodal leakage
保证 masked RGB 不会同时配合完全无遮挡的 pose/object

PHASE 6
只运行 synthetic occlusion diagnostic
不训练 RL

PHASE 7
检查 RL Go / No-Go Gate

PHASE 8
只有 Gate 通过才训练 PPO

PHASE 9
输出 synthetic RL benchmark

PHASE 10
生成联合总结表与日志
```

---

# 46. 输出目录建议

在 repository root 下建立：

```text
work_dir/
  exp01_dual_robot_orthogonal/
    config/
    results/
    figures/
    logs/

  exp02_synthetic_occlusion_rl/
    config/
    masks_debug/
    diagnostics/
    checkpoints/
    results/
    figures/
    logs/
```

最终至少保存：

```text
summary.json
per_sample_predictions.csv
per_class_results.csv
bootstrap_results.json
config.yaml
run.log
```

Experiment 2 额外保存：

```text
synthetic_occlusion_config.json
occlusion_diagnostic.csv
rl_training_curve.csv
```

---

# 47. 禁止事项

Codex 不要：

```text
使用 true unseen 调参数
把 Oracle 信息输入 policy
给 policy 输入 future candidate 信息
每个 view 独立随机 mask
把 synthetic occluder latent world position直接作为 policy state
发现 Gate 失败后仍自动训练 RL
为了提升数字擅自改变 VPOCLIP checkpoint
把 ETRI 固定 camera 结果描述成真实双机器人实验
```

---

# 48. 最终决策逻辑

```text
Experiment 1 成功：
    保留 Dual-Robot Preferred Orthogonal
    作为原始无遮挡环境的简单稳健主方案

Experiment 1 弱：
    不立即否定
    继续 Experiment 2

Experiment 2 diagnostic 成功 + RL 成功：
    说明 RL 需要足够强的 active-perception pressure
    可形成“无遮挡几何协同 + 遮挡环境学习型主动视角”的完整故事

Experiment 2 diagnostic 成功 + RL 失败：
    遮挡确实创造了 active-view opportunity
    但当前 RL/state 仍不足
    比较 geometry / heuristic 方法，不要强行宣称 RL 成功

Experiment 2 diagnostic 失败：
    synthetic occlusion 设计没有创造合理的 active-view problem
    停止 RL，先修正遮挡模拟器
```

---

# 49. 最终研究叙事

如果两个实验都得到合理结果，可以形成这样的故事：

> In the original ETRI setting, where occlusion is limited, a simple class-agnostic complementary dual-view formation is competitive and robust for unseen-action recognition. When viewpoint-dependent occlusion is introduced, active movement becomes more consequential; under this setting, we test whether an RL policy can learn to move toward views that recover task-relevant evidence.

注意：

- 原始 ETRI 结果用于说明简单几何协同的价值；
- synthetic occlusion 用于研究 active viewpoint learning 在真正存在遮挡压力时是否有必要；
- 不要把 synthetic occlusion 当成真实场景等价物；
- 最终若要冲 robotics venue，最好后续补少量真实机器人 + 真实遮挡物实验。
