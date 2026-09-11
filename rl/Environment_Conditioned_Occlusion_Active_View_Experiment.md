# 改进方案：Environment-Conditioned Active View Policy with View-Consistent Synthetic Occlusion

版本建议：`multistep_environment_occlusion_policy_v2`

## 0. 本轮目标

本轮不要继续把问题定义成：

```text
“这个动作类别是什么 -> 应该移动到哪个视角？”
```

新的核心问题是：

```text
“当前人体在哪里、怎么运动、周围遮挡/物体在哪里、当前相机在哪里
 -> 移动到哪个方向能够获得更完整、更互补的观察？”
```

因此策略学习的是：

\[
\boxed{
\text{human-environment spatial state}
\rightarrow
\text{view acquisition strategy}
}
\]

而不是：

\[
\boxed{
\text{action semantics}
\rightarrow
\text{view preference}
}
\]

最终目标仍然是 unseen/open-vocabulary HAR，但 viewpoint policy 必须尽可能 **class-agnostic**。

---

# 1. 为什么现在值得做 Synthetic Occlusion

当前 ETRI 完整数据集本身的拍摄目的就是清楚记录老人行为，真实视频中严重遮挡较少。

此前实验已经反复出现：

```text
Random 与 learned policy 差距很小
大量样本 view-insensitive
Oracle 明显高于 Random，但最佳视角难以从当前 observation 预测
```

一个合理假设是：

\[
\boxed{
\text{原始 ETRI 缺少足够强的 active-perception pressure}
}
\]

也就是说，在大量 episode 中：

```text
换左边
换右边
换后面
```

都能看到差不多的信息。

于是：

\[
Q(s,a_1)\approx Q(s,a_2)\approx Q(s,a_3)
\]

policy 很难获得明确监督。

本轮通过 **view-consistent synthetic occlusion** 人为构造：

```text
当前视角受遮挡
侧向视角遮挡减弱
另一个方向重新获得 hand/object/body evidence
```

从而验证：

> 如果环境真的让“移动方向”决定 observability，当前 human/object/geometry policy 是否能够学到可迁移的空间策略？

---

# 2. 本轮最重要的原则

## 2.1 不能做普通 Random Erasing

禁止只做：

```text
random black square
random cutout
每个 view 独立随机 mask
```

因为这种 augmentation 没有建立：

\[
\text{camera motion}
\rightarrow
\text{occlusion change}
\]

的因果关系。

如果 View0 的 mask 与 View1/View2/View3 完全独立随机，policy 无法从当前状态推断往哪边走。

---

## 2.2 必须做 View-Consistent Occlusion

同一个 episode 只采样一个 virtual occluder：

```text
occluder bearing
occluder width
occluder height
occluder strength
target vertical region
render style
```

然后根据每个 camera 相对于人体的真实 `relative_angle` 生成不同程度的 2D 遮挡。

核心：

\[
\boxed{
\text{同一个虚拟障碍物，在不同 camera 下产生不同 occlusion}
}
\]

---

# 3. 保留当前 trajectory-conditioned policy 的核心结构

当前 v1 中已经存在：

```text
person trajectory
object trajectory
sample-level camera relative angle
candidate-current delta angle
history of observed views
action mask
```

本轮不要推翻这些设计。

继续保持：

```text
VPOCLIP frozen
policy 不输入 action class ID
policy 不输入 GT label
policy 不输入 future candidate RGB
policy 不输入 future candidate logits
policy 不输入 future candidate object/pose
```

新的 v2 主要增加：

```text
explicit observed occlusion/environment state
+
synthetic view-consistent occlusion benchmark
+
occlusion-aware training utility
```

---

# 4. Phase 0：先构造 Synthetic Occlusion Benchmark，不训练 policy

这是最重要的一步。

先回答：

> synthetic occlusion 是否真的增加了视角选择价值？

如果没有，就不要训练任何新网络。

---

# 5. 使用现有 sample-level 相机方位

当前每个样本已经有：

```text
view_geometry[..., 0:2]
=
[sin(relative_angle), cos(relative_angle)]
```

不要使用固定 camera slot 语义。

每个 recording 中先恢复：

\[
\theta_v = \operatorname{atan2}(\sin\theta_v,\cos\theta_v)
\]

得到每个有效 view 的 human-relative bearing。

virtual occluder 也定义为 human-relative bearing：

\[
\theta_{occ}
\]

每个 episode 随机采样一次。

---

# 6. 第一版：2D Pseudo-Geometric Occlusion

当前不要求完整 3D reconstruction。

利用：

```text
human-relative camera angle
person bbox / torso center
pose trajectory
object trajectory
```

构造一个近似几何一致的遮挡模型。

---

# 7. Occlusion Severity

对 camera \(v\) 定义：

\[
\Delta\theta_v
=
wrap(\theta_v-\theta_{occ})
\]

遮挡强度：

\[
r_v
=
r_{max}
\exp
\left(
-\frac{\Delta\theta_v^2}{2\sigma_{occ}^2}
\right)
\]

解释：

```text
camera 与 occluder bearing 接近
-> severe occlusion

camera 转到侧面
-> moderate occlusion

camera 与 occluder bearing 差异很大
-> weak/no occlusion
```

建议第一版：

```text
sigma_occ = 35° ~ 55°
```

不要直接固定一个值，放到 pseudo-unseen validation 上选择。

---

# 8. Occlusion Level

至少构建三级 benchmark：

### Mild

```text
r_max ≈ 0.25 ~ 0.35
```

### Medium

```text
r_max ≈ 0.45 ~ 0.55
```

### Heavy

```text
r_max ≈ 0.65 ~ 0.75
```

这里的比例不要简单理解成整张图面积。

优先定义为：

```text
person / interaction region 的遮挡比例
```

---

# 9. Mask 不要主要遮背景

目标区域必须和 human observation 有关系。

至少实现 3 类 occlusion target。

## A. Torso-centered

根据 COCO17：

```text
shoulders: 5, 6
hips:      11, 12
```

构造 torso bbox。

适合模拟：

```text
柜子
桌边
门框
家具
```

挡住人体中部。

---

## B. Upper-body / hand region

利用：

```text
shoulder
elbow
wrist
face/head proxy
```

优先遮挡：

```text
hand
face
upper torso
```

因为很多 HAR 动作依赖这些区域。

---

## C. Hand-object interaction region

使用当前 object trajectory：

```text
[presence, x, y, confidence]
```

对每帧寻找：

```text
离 left/right wrist 最近
且 confidence 足够高
```

的 object。

构造：

```text
hand + nearest object
```

联合 region。

注意：

```text
不能使用 GT action class 来决定遮哪里
```

---

# 10. Mask 的横向偏移也要随视角变化

不要只让 mask 面积变化。

定义 mask 相对人体中心的 horizontal offset：

\[
x_{offset,v}
=
k_x \sin(\Delta\theta_v)
\]

于是：

```text
正面相对 occluder:
mask 靠近人体中心

相机向一侧移动:
mask 在图像中逐渐移出人体关键区域
```

mask width：

\[
w_v
=
w_{base}\cdot r_v
\]

mask height：

\[
h_v
=
h_{base}\cdot
(0.6+0.4r_v)
\]

第一版不要求严格物理投影，但必须保证：

> mask 的变化由统一 occluder bearing + camera bearing 决定，而不是每个 view 独立随机。

---

# 11. Temporal Consistency

同一个 view 的 13 个 frames：

```text
不能每帧重新随机 mask
```

occluder 参数在整个 episode 中固定：

```text
theta_occ
width_base
height_base
render_style
```

person 在 13 帧中运动时，mask 可以根据 torso / interaction region 做平滑更新。

要求：

```text
mask center trajectory 连续
mask size trajectory 连续
不能 frame-to-frame 闪烁
```

建议加入 EMA：

\[
M_t =
\beta M_{t-1}
+
(1-\beta)\hat M_t
\]

第一版：

```text
beta ≈ 0.7
```

---

# 12. Render Style

不要只使用纯黑色矩形。

否则 VPOCLIP / policy 可能学到 synthetic artifact。

至少随机化：

```text
neutral gray
local blur
background-color fill
texture patch
```

训练时随机 renderer。

测试时至少做：

```text
seen renderer
held-out renderer
```

检查策略是不是只记住 mask 外观。

---

# 13. 不要让所有视角都被毁掉

生成一个 occlusion episode 后，必须检查：

```text
至少一个有效 view 的关键人体区域仍然较清楚
```

否则这个 episode 是：

```text
unrecoverable by construction
```

不能为 active-view learning 提供有意义的 target。

建议约束：

\[
\min_v r_v < r_{clear}
\]

例如：

```text
至少一个 view 的 task-relevant occlusion < 20%
```

---

# 14. Phase 0 必须重新跑 VPOCLIP Cache

mask 必须真正作用到输入视频，再重新经过冻结的 VPOCLIP。

不要：

```text
只修改 policy feature
但 HAR logits 仍用无遮挡视频
```

否则 reward/utility 与 synthetic environment 不一致。

对每个：

```text
sample
view
occlusion seed
```

重新生成：

```text
masked video / frames
VPOCLIP logits
pose if possible
object tracks if possible
```

---

# 15. Pose / Object 如何处理

## Preferred

如果计算资源允许：

```text
对 masked frames 重新跑 pose / object detector
```

这样 observed state 会自然体现：

```text
遮挡导致关键点缺失
object confidence 下降
trajectory 变差
```

这是最真实的版本。

---

## Minimum viable version

如果重新跑 detector 成本太高：

继续使用原始 pose/object trajectory，但必须额外把：

```text
synthetic mask descriptors
```

显式加入 policy state。

同时在报告中明确：

> pose/object trajectories are privileged clean sensor tracks in the first synthetic benchmark.

之后再做 masked-detector 版本。

---

# 16. 新增 Occlusion / Environment State

每个已经观察的 view、每个时间段保存：

```text
mask_area_ratio
mask_person_overlap
mask_torso_overlap
mask_upper_body_overlap
mask_left_hand_overlap
mask_right_hand_overlap
mask_interaction_overlap

mask_center_x_relative_to_person
mask_center_y_relative_to_person
mask_width_relative_to_person
mask_height_relative_to_person
```

建议 shape：

```text
occlusion_trajectory: [13, 10~12]
```

---

# 17. 不允许输入 Future Candidate Occlusion

这是非常重要的因果约束。

policy 在当前 state 中只能看到：

```text
已经访问 view 的 mask / occlusion descriptor
```

不能直接输入：

```text
candidate view 的真实 mask_area_ratio
candidate view 的真实 visibility
candidate view 的 future occlusion
```

否则就等于把答案告诉 policy。

---

# 18. 可选的 Map-Aware Variant

如果未来你希望模拟机器人已经有环境地图，可以单独做第二个版本。

Map-aware state 可加入：

```text
estimated obstacle bearing
estimated obstacle distance
obstacle width
```

但必须单独报告：

```text
No-map policy
Map-aware policy
```

主结果优先使用 No-map，避免 privileged information 争议。

---

# 19. v2 Policy State

建议：

\[
s_t=
[
H_t,
O_t,
E_t,
G_t,
History_t
]
\]

其中：

### Human \(H_t\)

继续使用：

```text
person_trajectory [13,4]
=
[x, y, vx, vy]
```

### Object \(O_t\)

继续使用：

```text
object_trajectory [13,50,6]
=
[presence, x, y, confidence, rel_x, rel_y]
```

### Environment / Occlusion \(E_t\)

新增：

```text
occlusion_trajectory [13,D_occ]
```

### Geometry \(G_t\)

继续使用：

```text
candidate angle
current angle
candidate-current delta
second harmonic
```

### History

继续限制：

```text
visited views
current view
remaining valid candidates
```

---

# 20. 网络改动：新增一个低容量 Occlusion Branch

不要直接换大 Transformer。

建议：

```text
Occlusion branch:

Conv1d(D_occ -> 16, kernel=3, padding=1)
GroupNorm
GELU
Flatten
Linear(16*13 -> 32)
LayerNorm
GELU
```

得到：

```text
occlusion context = 32-D
```

原 context：

```text
person 32-D
object 64-D
angle/current 16-D
```

新 context：

```text
person 32
+ object 64
+ occlusion 32
+ angle 16
```

再投影到：

```text
context 64-D
```

candidate branch 继续保持当前 64-D。

不要先增大主体容量。

---

# 21. 必须做的 State Ablation

至少比较：

```text
A. Geometry only

B. Geometry + Person trajectory

C. Geometry + Person + Object

D. Geometry + Occlusion

E. Geometry + Person + Occlusion

F. Geometry + Person + Object + Occlusion
```

核心问题不是“大网络有没有提高”，而是：

> explicit environment/occlusion state 是否真正产生跨 unseen class 的 view-selection signal？

---

# 22. Phase 0：先测 Active-Perception Pressure

在训练 v2 policy 前，对：

```text
Original
Mild
Medium
Heavy
```

分别计算固定 view0：

```text
Single
Random-2
Random-3
Preferred Orthogonal-2
Preferred Orthogonal-3
Oracle-2
All valid
Recoverable ratio
View-sensitive ratio
```

---

# 23. Gate 0：Synthetic Benchmark 是否有效

只有满足下面大部分条件才继续训练 policy。

## Condition A

随着：

```text
Original -> Mild -> Medium
```

Single-view accuracy 应合理下降。

不能出现：

```text
mask 越重 accuracy 反而整体提高
```

---

## Condition B

Oracle-Random gap 应明显扩大。

当前原始数据约有明显 Oracle gap。

希望 Medium occlusion 下：

```text
Oracle-2 - Random-2
```

至少比 Original 再扩大约：

```text
+3 percentage points
```

或相对增长：

```text
>= 30%
```

---

## Condition C

Recoverable ratio 应明显上升。

当前原始实验大约只有少数样本真正 recoverable。

建议目标：

```text
Medium occlusion recoverable >= 25%
```

更理想：

```text
30%+
```

---

## Condition D

不能所有视角一起崩。

如果：

```text
Oracle 也接近 Random / Single
```

说明 mask 太重，active movement 也救不了。

这种 benchmark 无效。

---

# 24. Occlusion Seed / Layout Split

不能在训练和测试重复同一个 mask pattern。

每个 episode 的 occluder 参数随机采样。

建议：

```text
Train occlusion seeds:
1000-1999

Validation:
2000-2199

Test:
3000-3299
```

并确保：

```text
theta_occ
width
strength
renderer
```

都有随机变化。

进一步做：

```text
held-out renderer
held-out occlusion strength range
```

检验 spatial strategy 泛化。

---

# 25. Policy 训练集

仍然只用 seen classes。

推荐训练 mixture：

```text
25% original
25% mild
35% medium
15% heavy
```

不要全部训练成 Heavy。

目的：

> policy 学到“什么时候 geometry/environment matter”，而不是只适应一个固定遮挡强度。

---

# 26. VPOCLIP 先冻结

第一阶段：

```text
VPOCLIP frozen
```

这是必要的。

因为我们要先隔离：

> active view policy 是否能因为 environment state 而选择更好的视角？

如果同时 fine-tune VPOCLIP：

```text
recognizer 可能学会直接适应 mask
```

反而把 active-view pressure 消掉。

只有 policy 已经成功后，再单独做：

```text
masked-VPOCLIP robustness fine-tuning
```

作为附加实验。

---

# 27. Utility 必须基于 Masked HAR Logits 重新计算

继续沿用当前 delta-margin utility：

\[
m(s)
=
L_{GT}(s)
-
\max_{c\neq GT}L_c(s)
\]

candidate utility：

\[
u(s,a)
=
m(s\cup a)-m(s)
\]

但所有 \(L\) 必须来自：

```text
masked / synthetic-occlusion VPOCLIP cache
```

不能继续用 original clean logits。

---

# 28. 训练 Target 与 State 的边界

允许：

```text
future candidate masked logits
GT label
```

用于离线生成 training target：

\[
u(s,a)
\]

但禁止进入 policy state。

推理时只允许：

```text
observed person trajectory
observed object trajectory
observed occlusion state
current/history geometry
candidate geometry
```

---

# 29. Loss 先保持 v1

第一版不要同时修改太多变量。

继续：

\[
\mathcal L
=
\mathcal L_{listwise}
+
0.5\mathcal L_{pairwise}
\]

soft target：

\[
p_a
=
softmax(u_a/T_u)
\]

初始：

```text
T_u = 0.20
```

pairwise：

```text
只比较 |u_i-u_j| > 0.02
```

保持与 v1 尽量一致。

---

# 30. 增加 View-Sensitivity Weight（建议做 ablation）

可以额外测试：

\[
VS(s)
=
\max_a u(s,a)-\min_a u(s,a)
\]

sample weight：

\[
w_s
=
clip(VS(s),w_{min},w_{max})
\]

比较：

```text
unweighted
vs
view-sensitivity weighted
```

如果 synthetic occlusion 真正产生 action difference，VS weighting 应该比原始 ETRI 更有意义。

---

# 31. 第一阶段仍然不要 PPO

先训练当前这种 supervised/offline ranking policy。

理由：

> 本轮首先验证“environment state 是否能够预测 useful movement”。

如果连监督 candidate utility 都不能学到，就没有理由直接上 PPO。

只有 supervised policy 在固定 view0 / unseen 下稳定超过 Random 后，再做真正 RL。

---

# 32. 主要评测协议：固定 View0

必须把：

```text
fixed view0
```

作为主协议。

原因：

此前 policy 在其他起点有小幅正提升，但 fixed view0 没有稳定超过 random。

本轮最重要的问题就是：

> synthetic occlusion + environment-conditioned state 能不能让 fixed view0 也产生稳定正收益？

其他：

```text
view1
view2
view3
```

作为补充。

---

# 33. 必须比较的 Baselines

每个 occlusion level 比较：

```text
Single
Random-2
Random-3
Preferred Orthogonal-2
Preferred Orthogonal-3

Geometry-only learned scorer
Current v1 trajectory policy
v2 + occlusion branch

Oracle-2
Oracle-3
All valid
```

---

# 34. Random 必须多 Seed

继续沿用当前严格做法：

```text
>= 10 random evaluation seeds
```

不要用一次随机路径和 policy 比。

报告：

```text
mean
std
paired difference
bootstrap CI
```

---

# 35. 主要成功指标

## Primary

fixed view0、true unseen、Medium occlusion：

\[
Policy - Random
\]

建议只有达到：

```text
>= +1.0 pp
```

且 3 个 training seeds 都方向一致，才认为出现有意义进展。

更理想：

```text
>= +2.0 pp
```

---

## Statistical

paired bootstrap：

```text
95% CI
```

最好：

```text
lower bound > 0
```

如果 CI 仍跨 0，但 3 seeds 全为正，可记为 promising，不算最终成功。

---

# 36. 更重要的诊断：Action Agreement 不是主指标

不要主要优化：

```text
是否命中唯一 oracle candidate
```

因为多个 candidate 可能得到相近 HAR 效果。

主要报告：

```text
final Top-1
candidate regret
utility regret
Policy - Random
```

其中：

\[
Regret
=
u_{oracle}-u_{selected}
\]

---

# 37. 检查 Policy 是否真的“绕开遮挡”

新增以下分析。

对每次 policy movement 计算：

```text
observed occlusion before move
observed occlusion after move
```

统计：

\[
\Delta Occ
=
Occ_{before}-Occ_{after}
\]

比较：

```text
Policy
Random
Preferred Orthogonal
```

如果 policy 真学到了 spatial avoidance，应看到：

```text
Policy 的平均 occlusion reduction > Random
```

---

# 38. Side-Step Behavior Analysis

按照 current mask 在人体左/右的位置分组：

```text
mask predominantly left of person
mask predominantly right of person
mask centered
```

检查 policy 选择的 relative candidate angle。

希望看到：

```text
左侧遮挡 -> policy 更倾向移动到能解遮挡的一侧
右侧遮挡 -> 方向相反
```

这比单纯 Top-1 更能证明：

> policy 学到了 environment-conditioned movement strategy。

---

# 39. Counterfactual Test

这是本轮非常重要的实验。

同一个：

```text
human trajectory
object trajectory
camera geometry
```

保持不变，只改变：

```text
occluder bearing
```

例如：

```text
theta_occ = -45°
theta_occ = +45°
```

检查 policy action 是否跟着系统性改变。

如果 action 基本不变：

> policy 仍主要依赖 camera prior / trajectory shortcut。

如果 action 随 obstacle side 改变：

> environment branch 真正在控制决策。

---

# 40. Occlusion-Feature Shuffle Test

测试时随机打乱不同 sample 之间的：

```text
occlusion descriptors
```

保持其他 state 不变。

如果 policy performance 明显下降，说明它确实使用 environment state。

如果几乎不变：

> occlusion branch 被忽略。

---

# 41. Geometry Shuffle Test

同理打乱 candidate geometry。

如果 performance 不降：

> 网络可能在记 camera slot。

必须检查并阻止 camera-ID shortcut。

---

# 42. Camera Slot Invariance

因为当前每个 recording 的 view0/1/2/3 没有固定左前右后语义，本轮继续确保：

```text
不把 slot ID 当方位输入
```

可额外随机 permutation view slots 后评测。

只要 geometry 和数据一起正确 permutation，policy 输出应该保持几何等价。

---

# 43. Seen-to-Unseen Generalization

训练只使用 seen action classes。

true unseen：

```text
[A001, A003, A027, A035, A051]
class id [0,2,26,34,50]
```

继续保持最终 5-way protocol。

但 mask 参数也必须 unseen：

```text
test occlusion seeds 未在 train 出现
```

所以最终是双重泛化：

\[
\boxed{
\text{unseen action}
+
\text{unseen occlusion layout}
}
\]

---

# 44. 如果 v2 成功，再做真正 RL

只有满足：

```text
fixed view0
true unseen
Medium occlusion
Policy > Random by meaningful margin
```

后，才进入 PPO / sequential RL。

---

# 45. RL Stage 的 State

继续使用：

```text
person trajectory
object trajectory
observed occlusion trajectory
current/history geometry
visited views
```

可以增加：

```text
current VPOCLIP uncertainty
```

但不要输入 action class。

---

# 46. RL Actions

4-view dataset 中：

```text
STOP
move to unvisited view1
move to unvisited view2
move to unvisited view3
```

非法 / 已访问动作 mask 掉。

---

# 47. RL Reward

在 synthetic occlusion 环境下，visibility finally has causal meaning。

建议：

\[
r_t
=
\alpha \Delta HAR_t
+
\beta \Delta Occ_t
+
\gamma \Delta InteractionVisibility_t
-
\lambda C_{move}
-
\mu C_{view}
\]

其中：

```text
Delta HAR:
识别 evidence improvement

Delta Occ:
真实 observed occlusion reduction

Delta InteractionVisibility:
hand/object/upper-body 关键区域恢复

C_move:
移动角度 / 距离代价

C_view:
每多观察一个 view 的固定成本
```

---

# 48. RL 不要让 Visibility 主导

虽然 synthetic occlusion 让 visibility 变得有意义，但最终任务仍然是 HAR。

所以：

```text
Delta HAR / final HAR reward
```

必须是主任务。

visibility 只是 shaped reward。

避免 policy 只追求：

```text
“看见最多人体”
```

却不提高识别。

---

# 49. 如果 Supervised v2 仍失败

如果 benchmark 已经满足 Gate 0：

```text
Oracle-Random gap 明显扩大
recoverable ratio 明显增加
```

但 v2 仍然：

```text
Policy ~= Random
```

则进一步判断：

### Case A

occlusion state shuffle 不影响 policy：

```text
环境分支没有学到
```

需要改 representation / optimization。

### Case B

occlusion state 能预测 oracle utility，但 network 学不好：

```text
capacity / training issue
```

这时才考虑更大网络。

### Case C

即使 oracle utility 与 observed environment state 相关性仍接近 0：

```text
当前 state 仍不足
```

此时不要继续堆网络。

---

# 50. 更大网络什么时候允许

只有当至少满足：

```text
environment features 与 utility 有明显相关性
v2 small model 在 seen/validation 明显优于 geometry baseline
true unseen 已出现稳定正 gap
```

才允许测试：

```text
context 128-D
person 64-D
object 128-D
occlusion 64-D
candidate 128-D
```

之后再考虑：

```text
GRU / TCN
small Transformer
```

不是现在第一步。

---

# 51. Optional Phase：Calibration-Based 3D Occluder

如果后续可以取得 camera calibration：

```text
intrinsics
extrinsics
```

再升级为真正 3D virtual cuboid。

在 human/world coordinate 中放：

```text
3D cuboid / vertical plane
```

分别投影到每个 camera：

\[
M_v=\Pi(P_v,O)
\]

这样 synthetic occlusion 会更接近真实机器人绕遮挡物观察。

但这不是本轮 MVP 的前置条件。

---

# 52. 最终推荐执行顺序

```text
STEP 0
复现当前 v1 baseline

STEP 1
实现 view-consistent synthetic occlusion generator

STEP 2
生成 Original / Mild / Medium / Heavy masked cache

STEP 3
重新跑 frozen VPOCLIP logits

STEP 4
评估 Oracle-Random gap / recoverable ratio
-> Gate 0

STEP 5
增加 observed occlusion/environment branch

STEP 6
训练 supervised v2 ranker

STEP 7
做 state ablation

STEP 8
fixed view0 true-unseen evaluation

STEP 9
做 occlusion shuffle / counterfactual / geometry shuffle

STEP 10
如果 v2 确实稳定超过 Random
再进入 sequential PPO
```

---

# 53. Codex 直接执行任务

```text
请基于：

rl/multistep_angle_object_trajectory_policy_v1.py

建立新版本：

rl/multistep_environment_occlusion_policy_v2.py

并建立 synthetic occlusion 数据生成与评测脚本。

不要直接训练 PPO。

==================================================
A. SYNTHETIC OCCLUSION GENERATOR
==================================================

新增：

rl/build_view_consistent_occlusion_cache.py

要求：

1. 使用每个 sample 的真实 view_geometry sin/cos；
2. 每个 episode 采样一个 human-relative occluder bearing；
3. 同一 episode 的 4 个 views 共用同一 occluder；
4. mask strength 根据 camera-occluder angular difference 变化；
5. 13 个 frames temporal consistent；
6. mask 优先覆盖 person / upper-body / hand-object region；
7. 不能根据 GT action class 决定 mask；
8. 至少一个 view 保持相对 clear；
9. 保存 occlusion metadata；
10. train/val/test 使用不同 occlusion seeds。

实现：
- Mild
- Medium
- Heavy

==================================================
B. MASKED VPOCLIP CACHE
==================================================

使用冻结的当前 VPOCLIP。

对 masked clips 重新计算：

- per-view logits
- fused evaluation cache

如果算力允许：
重新跑 pose/object detector。

如果暂时不允许：
保留 clean pose/object，
但必须显式记录这是 clean-track synthetic setting。

==================================================
C. PHASE-0 DIAGNOSTIC
==================================================

在训练新 policy 前，对：

Original
Mild
Medium
Heavy

计算 fixed view0：

- Single
- Random-2
- Random-3
- Preferred Orthogonal-2
- Preferred Orthogonal-3
- Oracle-2
- Oracle-3
- All valid
- recoverable ratio
- view-sensitive ratio
- Oracle-Random gap

如果 Medium 没有明显增加：
Oracle-Random gap
以及 recoverable ratio，

停止，不训练新 policy。
先修 synthetic occlusion。

==================================================
D. OCCLUSION STATE
==================================================

对 observed view 构造：

occlusion_trajectory [13,D]

至少包括：

- mask area ratio
- person overlap
- torso overlap
- upper-body overlap
- left-hand overlap
- right-hand overlap
- interaction overlap
- mask center relative x/y
- relative width/height

严禁输入 future candidate 的真实 occlusion。

==================================================
E. POLICY V2
==================================================

保留 v1：

- person branch
- object branch
- candidate geometry
- history
- action mask

新增低容量 occlusion branch：

Conv1d(D_occ -> 16)
GroupNorm
GELU
Flatten
Linear -> 32
LayerNorm
GELU

context:
person 32
object 64
occlusion 32
angle 16
-> context 64

candidate branch 暂时不扩大。

==================================================
F. TRAINING TARGET
==================================================

使用 masked VPOCLIP logits 重新计算：

m(s)
u(s,a) = m(s+a) - m(s)

继续：

listwise
+
0.5 * pairwise

pair:
|u_i-u_j| > 0.02

第一版保持 v1 主要超参数，
不要同时改变太多变量。

==================================================
G. TRAIN MIX
==================================================

建议：

25% original
25% mild
35% medium
15% heavy

只在 seen action classes 训练。

true unseen 禁止调参。

==================================================
H. STATE ABLATION
==================================================

必须比较：

A Geometry
B Geometry + Person
C Geometry + Person + Object
D Geometry + Occlusion
E Geometry + Person + Occlusion
F Geometry + Person + Object + Occlusion

==================================================
I. MAIN EVALUATION
==================================================

主协议：

fixed view0
true unseen 5-way

同时补充 fixed1/2/3。

Random 使用 >=10 evaluation seeds。

Policy 使用 >=3 training seeds。

比较：

Single
Random
Preferred Orthogonal
v1 trajectory policy
v2 environment-conditioned policy
Oracle
All valid

输出：

Top-1
Policy-Random
paired bootstrap 95% CI
utility regret
movement cost

==================================================
J. BEHAVIOR DIAGNOSTICS
==================================================

必须增加：

1. occlusion reduction after move
2. mask-left / mask-right 分组 action histogram
3. counterfactual occluder-bearing flip
4. occlusion-feature shuffle
5. geometry shuffle
6. camera-slot permutation test

目标不是只看 accuracy，
而是证明 policy 真正在使用 environment state。

==================================================
K. SUCCESS CONDITION
==================================================

主目标：

fixed view0
true unseen
Medium occlusion

希望：

Policy - Random >= +1.0 pp

更理想：
>= +2.0 pp

并且：
3 个 training seeds 方向一致。

如果 paired bootstrap 95% CI lower bound > 0，
可认为结果稳定。

==================================================
L. DO NOT TRAIN PPO YET
==================================================

只有 supervised v2 在上述主协议下稳定超过 Random，
才开始 sequential PPO。

如果 v2 仍约等于 Random，
先分析：
- environment utility correlation
- occlusion shuffle
- counterfactual behavior

不要自动扩大网络。

==================================================
FINAL REPORT
==================================================

最终必须回答：

1. synthetic occlusion 是否扩大 Oracle-Random gap？
2. recoverable ratio 是否明显增加？
3. explicit occlusion state 是否提升 unseen view selection？
4. policy 是否真的学会绕开遮挡，而不是记 camera prior？
5. fixed view0 是否终于稳定超过 Random？
6. 改进来自 environment conditioning，还是单纯 geometry？
7. 是否已经有足够证据进入 PPO？
```

---

# 54. 本轮最重要的论文假设

建议明确写成：

> **The failure of learned active-view policies on the original ETRI setting may be caused not only by weak policy models, but by insufficient active-perception pressure: most actions are already well observed from multiple cameras.**

然后验证：

> **When viewpoint-dependent occlusion makes observability depend on camera motion, a class-agnostic policy conditioned on human–environment spatial dynamics should learn transferable viewpoint behavior without using action semantics.**

---

# 55. 如果本轮成功，方法最终可描述为

\[
\boxed{
\textbf{Environment-Conditioned Class-Agnostic Active View Acquisition}
}
\]

策略学习：

\[
Q_\theta(
\text{human dynamics},
\text{environment/occlusion},
\text{view geometry},
\text{history},
a
)
\]

而不是：

\[
Q_\theta(
\text{action identity},
a
)
\]

这才是本项目希望得到的核心迁移机制：

\[
\boxed{
\text{transfer spatial observation strategy across unseen actions}
}
\]
