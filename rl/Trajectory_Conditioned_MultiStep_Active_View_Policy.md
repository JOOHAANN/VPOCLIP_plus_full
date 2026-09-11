# 轨迹条件的多步主动视角策略

版本：multistep_angle_object_trajectory_policy_v1

本文档记录当前已经实现并完成测试的方案：使用样本级相机方位、人体轨迹和 object detection 连续轨迹，逐步选择下一个尚未观察的视角，并将已经观察到的视角融合用于动作识别。

本文档中的“策略”沿用项目中的 RL/主动视角称呼，但当前实现不是端到端 PPO，也不是在线环境中的经典 DQN。它是一个低容量的候选视角评分网络，使用 seen 类上的离线识别收益构造排序目标进行训练；VPOCLIP 始终冻结。这样做的目的，是先隔离“传感器状态能否预测有用视角”这个问题。

## 1. 研究目标

对于一个动作样本，机器人从一个低视角开始观察。策略在每一步只能使用当前已经获得的传感器信息，从尚未观察的有效候选视角中选择下一个视角：

\[
a_t=\arg\max_{a\in\mathcal A_t}Q_\theta(s_t,a)
\]

其中：

- a_t 是视角槽位序号，不是动作类别，也不是连续角度；
- s_t 是当前和历史已观察视角形成的因果状态；
- A_t 只包含有效、可到达、且尚未访问的视角；
- VPOCLIP 负责动作识别，策略只负责视角选择。

当前数据每个样本最多有 4 个有效低视角，因此有意义的移动上限是 3 次：

    0 次移动：1 个视角融合
    1 次移动：2 个视角融合
    2 次移动：3 个视角融合
    3 次移动：4 个视角融合

超过 3 次只能重复访问已经看过的视角。在离线缓存中重复视角不会产生新观测，因此不能作为新的视觉信息实验。

## 2. 数据和划分

### 2.1 输入视频和时间采样

- 原始视频参考分辨率：640 × 480；
- 每个视角使用与 VPOCLIP 缓存一致的 13 个时间帧；
- object detection 在这 13 个帧上逐帧运行；
- 轨迹缓存不是重新定义一套抽帧规则，而是复用 VPOCLIP cache 的 sample_frames。

object 轨迹缓存的单个样本形状为：

    [4 views, 13 frames, 50 object classes, 4 channels]

4 个原始 object 通道为：

    [presence, x, y, confidence]

x/y 是归一化图像坐标，范围为 [-1, 1]，y 轴向下。

### 2.2 类别划分

训练只使用 seen 类。验证集使用预先指定的 pseudo-unseen seen 类进行模型选择；真实 unseen 只在最后测试时使用，不能参与调参。

真实 unseen 采用严格的 5-way 评测：

    [A001, A003, A027, A035, A051]
    class id = [0, 2, 26, 34, 50]

评测时的预测类别先限制到这 5 类，不是在 55 类中做 general prediction。

### 2.3 视角方位映射

视角槽位 0/1/2/3 不具有固定的“左/前/右/后”语义。每个 recording 使用自己的相机方位信息：

    view_geometry[..., 0:2] = [sin(relative_angle), cos(relative_angle)]

因此，策略使用的是每个样本、每个视角自己的相对方位，而不是把 camera ID 当作方位标签。若某个 recording 的 view3 在人体右侧，另一个 recording 的 view3 可能在左侧，网络都通过 geometry 向量区分。

样本级几何也用于计算候选之间的相对角度：

\[
\sin(\Delta\theta_v)=
\sin\theta_v\cos\theta_c-\cos\theta_v\sin\theta_c
\]

\[
\cos(\Delta\theta_v)=
\sin\theta_v\sin\theta_c+\cos\theta_v\cos\theta_c
\]

其中 c 是当前视角，v 是候选视角。

## 3. 策略状态

状态严格遵守因果约束：策略只能看当前和历史已经观察过的内容，不能读取未来视角的 RGB、VPOCLIP 特征或未来遮挡信息。

### 3.1 人体轨迹

缓存的 pose 是 13 × 17 × 2 = 442 维。每一帧取 COCO17 的四个躯干点：

    左右肩：5, 6
    左右髋：11, 12

计算躯干中心和速度，形成：

    person_trajectory: [13, 4]
    [center_x, center_y, velocity_x, velocity_y]

如果已经观察了多个视角，当前实现对已观察视角的轨迹做平均，形成历史传感器摘要。没有使用未来视角的 pose。

### 3.2 Object 连续轨迹

原始 object 轨迹为：

    [13, 50, 4]
    [presence, x, y, confidence]

进一步加入 object 相对于人体躯干中心的位置：

    relative_x = object_x - person_center_x
    relative_y = object_y - person_center_y

最终送入策略的 object 状态为：

    object_trajectory: [13, 50, 6]
    [presence, x, y, confidence, relative_x, relative_y]

这使策略能够区分：

- 某个物体是否出现；
- 物体在图像中的绝对位置；
- 物体相对于人体的位置；
- 物体位置是否随时间变化；
- 检测置信度是否稳定。

### 3.3 候选视角几何状态

每一步对 4 个候选视角都构造 geometry。基础输入包括：

    candidate view geometry       [sin(theta_v), cos(theta_v)]
    candidate-current delta       [sin(delta), cos(delta)]
    current view geometry         [sin(theta_c), cos(theta_c)]

之后加入二阶 harmonic 表达：

    [2*sin(theta)*cos(theta), cos(theta)^2 - sin(theta)^2]

最终候选几何特征为 10 维，合法性不作为普通数值特征，而是通过 action mask 单独处理。

候选合法性为：

    raw.valid
    AND raw.reachable[current_view]
    AND candidate_not_already_observed

非法候选的 Q 值在 argmax 前设为负无穷。

### 3.4 状态中明确排除的内容

当前轨迹策略不输入：

    RGB/video embedding
    VPOCLIP 512-D z
    skeleton embedding
    semantic belief
    current VPOCLIP logits
    动作类别 ID
    未来视角的特征、logits 或 object 信息
    GT action label

VPOCLIP logits 只在训练时构造离线收益标签，以及最终评测时计算 HAR；不进入策略输入。

## 4. 网络结构

策略网络是一个低容量、因子化的候选评分器。

### 4.1 人体分支

    Conv1d(4 -> 16, kernel=3, padding=1)
    GroupNorm
    GELU
    Flatten
    Linear(16*13 -> 32)
    LayerNorm
    GELU

输出人体上下文向量 32-D。

### 4.2 Object 分支

    50*6 = 300 input channels
    Conv1d(300 -> 64, kernel=3, padding=1)
    GroupNorm
    GELU
    Flatten
    Linear(64*13 -> 64)
    LayerNorm
    GELU

输出 object 上下文向量 64-D。

### 4.3 几何和候选分支

    current angle [2] -> Linear -> 16-D
    person 32-D + object 64-D + angle 16-D -> context 64-D
    candidate geometry [10] -> candidate MLP -> 64-D

动作分数为上下文和候选向量的低容量匹配分数，再加一个小型 residual 分支：

\[
Q(s,a)=
\frac{h(s)^T g(a)}{\sqrt{64}}
+0.20\,r(h(s),g(a))
\]

网络最终输出 4 个候选视角分数：

    [Q(view0), Q(view1), Q(view2), Q(view3)]

它不输出动作类别，也不输出连续坐标。

## 5. 多步决策过程

### 5.1 移动 1 次

给定起点 v0：

    s1 = state(history=[v0])
    a1 = argmax Q(s1, a)
    path = [v0, a1]
    final_logits = mean(logits[v0], logits[a1])

融合 2 个视角。

### 5.2 移动 2 次

第一步选出 a1 后，机器人获得第二个视角的传感器信息：

    s2 = state(history=[v0, a1])
    a2 = argmax Q(s2, a)
    path = [v0, a1, a2]
    final_logits = mean(logits[v0], logits[a1], logits[a2])

第二步不是重新从单视角预测，而是使用前两个已观察视角形成的历史状态。

### 5.3 移动 3 次

    s3 = state(history=[v0, a1, a2])
    a3 = argmax Q(s3, a)
    path = [v0, a1, a2, a3]
    final_logits = mean(logits[v0], logits[a1], logits[a2], logits[a3])

如果四个视角全部有效，所有合法路径最终观察到的集合都相同。因此四视角平均 logits 与访问顺序无关，策略不能再改变分类信息，只能影响移动顺序和 movement cost。

## 6. 训练目标

### 6.1 Seen 训练样本

当前训练使用完整的四视角 seen episode：

    training episodes = 2734
    training class bank = seen classes

只使用四视角完整样本训练，缺失视角样本不进入主训练池；评测阶段保留真实有效视角 mask。

每个 batch 随机生成历史阶段：

    history_count = 1: 45%
    history_count = 2: 45%
    history_count = 3: 10%

因此网络主要学习“一个已观察视角后选第二个”和“两个已观察视角后选第三个”，第三次移动/第四视角预测只占较小比例。

### 6.2 离线 delta-margin utility

对 seen 类动作类别 bank，定义当前融合状态的 GT margin：

\[
m(s)=L_{GT}(s)-\max_{c\ne GT}L_c(s)
\]

对于候选视角 a，将它的 VPOCLIP logits 加入当前历史融合，构造：

\[
u(s,a)=m(s\cup a)-m(s)
\]

u(s,a) 表示该候选对 seen 类识别 margin 的离线提升。

重要的因果边界：

- u(s,a) 使用未来候选的 logits 生成训练 target；
- 未来候选 logits 不进入 state；
- 推理时策略只有人体、object 和几何传感器；
- GT label 不进入 state。

### 6.3 Listwise + pairwise 排序损失

先将候选 utility 转成 soft target distribution：

\[
p_a=\operatorname{softmax}(u_a/0.20)
\]

然后使用 listwise 交叉熵，使高 utility 候选获得更高分。

同时对 utility 差异明显的候选增加 pairwise 排序约束：

    只比较 |u_i - u_j| > 0.02 的候选对
    要求 Q_i 与 Q_j 的顺序和 utility 顺序一致
    差距越大，要求的 Q 分离越大

总损失为：

\[
\mathcal L=
\mathcal L_{listwise}+0.5\mathcal L_{pairwise}
\]

history_count=3 的阶段使用较小权重 0.10，因为当前主要目标是前两次视角选择。

### 6.4 当前训练超参数

    epochs              = 40
    updates_per_epoch   = 64
    batch_size          = 16384
    learning_rate       = 1e-4
    optimizer           = AdamW
    weight_decay        = 0.04
    scheduler            = CosineAnnealingLR
    training seeds      = 20260909, 20260910, 20260911
    movement cost       = not included in training loss

movement cost 只在评测时记录，当前 v1 不用 movement cost 影响动作选择。

## 7. 评测协议

### 7.1 起点

分别固定：

    fixed0
    fixed1
    fixed2
    fixed3

另外可以单独报告 random start，但不能把 random start 和固定 view0 结果混为同一个公平协议。

### 7.2 对比方法

对每个起点和移动次数比较：

    Single       只使用起始视角
    Random       随机选择同样次数的未访问视角
    Policy       当前轨迹策略逐步选择视角
    Oracle       使用 GT 类别选择最终 margin 最好的路径，仅作上界
    All valid    所有有效视角平均，仅作为全视角参考

Random 评测需要使用多个 evaluation seeds，避免某一个随机路径样本偶然偏好或不利。

### 7.3 融合方式

当前融合不是拼接或重新投影 512-D embedding，而是直接对冻结 VPOCLIP 的 class logits 做平均：

\[
L_{fused}=
\frac{1}{|H|}\sum_{v\in H}L_v
\]

最终在对应类别 bank 内取最大得分作为预测类别。

## 8. 当前实验结果

以下为真实 unseen 5-way、414 个样本、3 个训练 seed，并用 10 个随机评测 seed 反复测试后的结果。表中的 Policy 是 3 个训练 checkpoint 的均值，Random 是 10 个随机路径 seed 的均值。

### 8.1 移动 1 次：融合 2 个视角

| 起点 | Single | Random | Policy | Policy - Random |
|---|---:|---:|---:|---:|
| view0 | 49.28% | 52.46 ± 0.49% | 52.33 ± 0.46% | -0.13 |
| view1 | 49.52% | 50.87 ± 0.63% | 51.29 ± 0.11% | +0.42 |
| view2 | 48.79% | 51.04 ± 0.77% | 52.09 ± 0.30% | +1.05 |
| view3 | 50.24% | 53.12 ± 0.70% | 53.70 ± 0.41% | +0.59 |

四个起点平均：Policy 52.36%，Random 51.87%，提升约 +0.48 个百分点。

### 8.2 移动 2 次：融合 3 个视角

| 起点 | Single | Random | Policy | Policy - Random |
|---|---:|---:|---:|---:|
| view0 | 49.28% | 53.74 ± 0.53% | 53.30 ± 0.30% | -0.44 |
| view1 | 49.52% | 52.61 ± 0.72% | 53.62 ± 0.20% | +1.01 |
| view2 | 48.79% | 52.46 ± 0.62% | 52.74 ± 0.50% | +0.27 |
| view3 | 50.24% | 53.57 ± 0.50% | 53.78 ± 0.89% | +0.21 |

四个起点平均：Policy 53.36%，Random 53.10%，提升约 +0.26 个百分点。

### 8.3 移动 3 次：融合 4 个视角

所有 4 个视角有效时，Random、Policy 和 All-valid 的最终分类准确率都为：

    53.14%

原因不是策略失效，而是三次移动后四个视角都已经被融合，视角访问顺序不能改变平均 logits。当前测试中 Policy 主要体现为移动成本略低于随机路径。

## 9. 结果解释

当前结果支持以下较谨慎的结论：

1. 人体/物体连续轨迹和样本级相机几何包含一定的可迁移视角选择信号；
2. 这个信号在 view1–view3 起点下更明显；
3. 固定 view0 是最严格、最公平的主协议，但当前策略在该协议下没有稳定超过随机；
4. 多次随机评测后，单个 seed 带来的偶然优势消失，不能只报告一次随机路径；
5. 移动 2 次通常比移动 1 次分类更高，但第三个视角也可能引入冲突信息，因此不是单调收益；
6. 移动 3 次在四视角数据上主要是路径规划问题，而不是视角选择识别问题。

当前策略更像是“利用可观测传感器先验做弱视角排序”，还不是一个能够稳定预测 HAR 最佳视角的通用策略。

## 10. 不能混淆的三个概念

### 10.1 Policy 不是类别分类器

策略输出：

    view index 0/1/2/3

VPOCLIP 输出：

    55 个动作类别 logits

在 unseen 测试时，VPOCLIP 的 logits 再限制到 5 个真实 unseen 类别。

### 10.2 训练 target 不等于策略 state

未来视角 logits 可以用于离线计算“这个视角对 seen 类识别是否有帮助”的 target，但不能作为策略输入。否则会发生未来信息泄漏，测到的是 privileged selector，而不是可部署策略。

### 10.3 三次移动不等于三次新的候选选择收益

三次移动访问四个固定视角后，所有路径都观察相同的四个数据块。因此：

- 分类结果由四视角集合决定；
- 访问顺序只影响运动成本和到达时间；
- 如果要研究第四次以后仍有新信息，必须增加中间相机位姿、实时视频或新的传感器观测。

## 11. 当前代码和结果位置

主训练策略：

    /home/youhan/ws/VPOCLIP_plus_full/rl/multistep_angle_object_trajectory_policy_v1.py

object 轨迹提取：

    /home/youhan/ws/VPOCLIP_plus_full/rl/build_object_position_tracks.py

多 seed 评测：

    /home/youhan/ws/VPOCLIP_plus_full/rl/evaluate_trajectory_policy_seed_sweep.py
    /home/youhan/ws/VPOCLIP_plus_full/rl/evaluate_one_two_move_random_seed_sweep.py

移动 1 次/2 次评测：

    /home/youhan/ws/VPOCLIP_plus_full/rl/evaluate_trajectory_policy_one_two_moves.py

移动 3 次、四视角融合评测：

    /home/youhan/ws/VPOCLIP_plus_full/rl/evaluate_trajectory_policy_four_views.py

训练 checkpoint：

    /home/youhan/ws/VPOCLIP_plus_full/work_dir/multistep_angle_object_trajectory_policy_v1/seed_20260909/best.pt
    /home/youhan/ws/VPOCLIP_plus_full/work_dir/multistep_angle_object_trajectory_policy_v1/seed_20260910/best.pt
    /home/youhan/ws/VPOCLIP_plus_full/work_dir/multistep_angle_object_trajectory_policy_v1/seed_20260911/best.pt

当前主要结果：

    /home/youhan/ws/VPOCLIP_plus_full/work_dir/multistep_angle_object_trajectory_policy_v1/summary.json
    /home/youhan/ws/VPOCLIP_plus_full/work_dir/multistep_angle_object_trajectory_policy_v1/one_two_move_test/random_seed_sweep/summary.json
    /home/youhan/ws/VPOCLIP_plus_full/work_dir/multistep_angle_object_trajectory_policy_v1/four_view_three_move_test/summary.json

## 12. 后续可研究方向

如果继续改进，优先级应为：

1. 在固定 view0 主协议下使用多 seed 验证，而不是只追求某个起点的最好数字；
2. 加入 STOP 动作，让策略可以在第二视角已经足够好时停止移动；
3. 将“识别收益”和“移动成本”作为独立报告维度，再决定是否放入 reward；
4. 用更多 held-out seen classes 检查轨迹到视角收益的可预测性；
5. 如果要研究 3 次以上移动，提供真实的新观测，而不是重复四个缓存视角；
6. 只有在状态可预测性明确成立后，再考虑 PPO/DQN 等真正在线 RL 算法。

