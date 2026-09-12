# Simple Weighted Logit Fusion：两个轻量实验

本轮只做两个简单 fusion 实验，不继续增加复杂模型。

共同融合形式：

\[
z_{fused}=\sum_{v=1}^{K} w_v z_v,\qquad \sum_v w_v=1
\]

其中 \(z_v\) 是每个 view 的分类 logits。

---

## 0. 非常重要：两个实验完全分开

请新建两个独立目录：

```text
work_dir/fusion_lightweight_gating_v1/
work_dir/fusion_rule_based_v1/
```

如需新脚本：

```text
rl/fusion_lightweight_gating_v1.py
rl/fusion_rule_based_v1.py
```

日志也分开：

```text
logs/fusion_lightweight_gating_v1.log
logs/fusion_rule_based_v1.log
```

要求：

```text
不要覆盖以前任何代码
不要覆盖旧 checkpoint
不要覆盖旧 cache
不要覆盖旧 log
不要覆盖旧 work_dir
不要修改旧 RL
不要重新训练 VPOCLIP
```

两个实验只读取现有 checkpoint / cache，输出写入各自新目录。

---

# 1. 共同 Baseline

当前 Equal Mean Fusion：

\[
w_v=rac{1}{K}
\]

\[
z_{mean}=rac{1}{K}\sum_v z_v
\]

所有新方法都必须和这个 baseline 在完全相同的 view set 上比较。

测试：

```text
Random-2
Preferred-Orthogonal-2
Random-3
Preferred-Orthogonal-3
All-valid views
```

---

# 2. Experiment A：Lightweight Gating Network

目录：

```text
work_dir/fusion_lightweight_gating_v1/
```

目标：

用一个很小的共享网络，根据每个已经观察到的 view 的质量和几何信息，输出一个 scalar weight。

---

## 2.1 每个 view 的输入

### Classification

```text
entropy              1
margin               1
max probability      1
```

建议：

```text
entropy = normalized entropy
margin = top1 - top2 probability margin
max probability = max softmax probability
```

### Temporal stability

只使用已有 cache 里已经有的字段，优先：

```text
entropy_std
top1_switch_rate
logit_variance_mean
margin_std
```

总共使用 2~4 个即可。

没有的字段不要伪造。

### Geometry

```text
sin(azimuth)
cos(azimuth)

sin(elevation)
cos(elevation)
```

共 4 维。

### Relative diversity

每个 view 相对于其他已观察 views 的平均 angular distance：

\[
D_v=rac{1}{K-1}\sum_{j
eq v} d_{angle}(v,j)
\]

normalize 到 0~1。

### Cross-view JS

每个 view 和其他 view 的 prediction distribution 做平均 JS divergence：

\[
JS_v=rac{1}{K-1}\sum_{j
eq v}JS(p_v,p_j)
\]

---

## 2.2 最终输入维度

大约：

```text
entropy                 1
margin                  1
max probability         1
temporal stability      2~4
azimuth sin/cos         2
elevation sin/cos       2
relative diversity      1
cross-view JS           1
--------------------------
total                 ~11-13
```

---

## 2.3 网络

保持非常轻量：

```text
Input D
  ↓
Linear(D, 16)
ReLU
  ↓
Linear(16, 1)
  ↓
score e_v
```

所有 views 共用同一个网络。

不要：

```text
Transformer
大 MLP
512-D feature 输入
完整 logits 输入
camera-specific network
```

---

## 2.4 权重与融合

对一个 sample 的所有已观察 views：

\[
w_v=softmax(e_v)
\]

最终：

\[
z_{fused}=\sum_v w_v z_v
\]

VPOCLIP 全部 freeze，只训练这个小 gating network。

Loss：

\[
L=CE(z_{fused},y_{GT})
\]

最后一层初始化接近 0，使初始：

\[
w_vpprox 1/K
\]

也就是从当前 equal mean 开始学习。

---

# 3. Experiment B：Rule-Based Weighted Fusion

目录：

```text
work_dir/fusion_rule_based_v1/
```

这个实验：

```text
不训练任何网络
```

只使用：

```text
entropy
margin
skeleton visibility
body-camera orientation
```

计算 view weight。

---

## 3.1 Entropy Score

normalized entropy：

\[
H_v^{norm}\in[0,1]
\]

定义：

\[
S_H(v)=1-H_v^{norm}
\]

即 entropy 越低，score 越高。

---

## 3.2 Margin Score

使用 probability top1-top2 margin：

\[
M_v=p_{(1)}-p_{(2)}
\]

normalize 到 0~1：

\[
S_M(v)=M_v^{norm}
\]

---

## 3.3 Skeleton Visibility

如果当前 cache 有 valid / visible joint mask：

\[
S_{vis}(v)=rac{visible\ joints}{total\ joints}
\]

范围 0~1。

不重新训练 pose model。

---

## 3.4 Body Orientation

定义角度：

```text
0°   = 人正面对摄像头
90°  = 人侧面对摄像头
180° = 人完全背对摄像头
```

希望：

```text
30° ~ 90° 最佳
背对摄像头明显扣分
```

使用简单 rule：

```text
if angle < 30:
    score = 0.8 + 0.2 * angle / 30

elif angle <= 90:
    score = 1.0

elif angle <= 150:
    score = 1.0 - 0.6 * (angle - 90) / 60

else:
    score = 0.2
```

即：

```text
0°      -> 0.8
30~90°  -> 1.0
150°    -> 0.4
180°    -> 0.2
```

---

# 4. Rule-Based 三个简单版本

## Rule-1：Classification Only

\[
S_v=0.5S_H+0.5S_M
\]

即：

```text
entropy 50%
margin  50%
```

---

## Rule-2：Skeleton / Orientation Only

\[
S_v=0.5S_{vis}+0.5S_{ori}
\]

即：

```text
visibility  50%
orientation 50%
```

---

## Rule-3：Combined

\[
S_v=
0.30S_H
+
0.30S_M
+
0.20S_{vis}
+
0.20S_{ori}
\]

即：

```text
entropy      30%
margin       30%
visibility   20%
orientation  20%
```

这是主要 rule-based 方法。

---

## 4.1 Rule weight

最终：

\[
w_v=rac{S_v}{\sum_j S_j}
\]

然后：

\[
z_{fused}=\sum_v w_vz_v
\]

禁止在 true unseen 上修改这些系数。

如果要改，只能根据 seen validation。

---

# 5. Evaluation

比较：

```text
Equal Mean Fusion
Lightweight Gating
Rule-1 Classification
Rule-2 Skeleton/Orientation
Rule-3 Combined
```

使用完全相同的：

```text
Random-2
Orthogonal-2
Random-3
Orthogonal-3
All-valid
```

分别报告：

```text
Seen
True unseen
```

---

# 6. 输出表

| View Set | Mean | Lightweight Net | Rule-Cls | Rule-Geo | Rule-Combined |
|---|---:|---:|---:|---:|---:|
| Random-2 | | | | | |
| Orthogonal-2 | | | | | |
| Random-3 | | | | | |
| Orthogonal-3 | | | | | |
| All-valid | | | | | |

每个方法报告：

```text
Top-1
Delta vs Equal Mean
paired bootstrap 95% CI
```

同时保存：

```text
sample_id
view_ids
entropy per view
margin per view
visibility per view
orientation per view
final weights
final prediction
```

---

# 7. Codex 直接执行指令

```text
本轮只做两个新的 fusion 实验。

非常重要：
两个实验必须放在两个全新的独立目录中。

创建：

work_dir/fusion_lightweight_gating_v1/
work_dir/fusion_rule_based_v1/

如需脚本：

rl/fusion_lightweight_gating_v1.py
rl/fusion_rule_based_v1.py

日志：

logs/fusion_lightweight_gating_v1.log
logs/fusion_rule_based_v1.log

不要覆盖或修改任何旧实验。

==================================================
EXPERIMENT A
LIGHTWEIGHT GATING NETWORK
==================================================

每个 view 输入：

- normalized entropy
- top1-top2 probability margin
- max probability

temporal stability:
从已有 cache 选择 2~4 个：
- entropy_std
- top1_switch_rate
- logit_variance_mean
- margin_std

geometry:
- sin azimuth
- cos azimuth
- sin elevation
- cos elevation

plus:
- relative angular diversity
- mean cross-view JS divergence

输入总维度约 11~13。

网络：

D
-> Linear(16)
-> ReLU
-> Linear(1)

所有 views 共用同一个 network。

weights = softmax(view_scores)

z_fused = sum(weight_v * logits_v)

freeze VPOCLIP。
只训练这个 gating network。

loss:
cross entropy of fused logits.

最后一层初始化接近 0，
使初始 weights 接近 uniform。

==================================================
EXPERIMENT B
RULE-BASED WEIGHTED FUSION
==================================================

不训练网络。

每个 view：

1. S_H = 1 - normalized_entropy

2. S_M = normalized top1-top2 probability margin

3. S_vis = visible_joint_ratio

4. orientation score:

0 deg   = facing camera
90 deg  = side
180 deg = back to camera

if angle < 30:
    score = 0.8 + 0.2 * angle / 30
elif angle <= 90:
    score = 1.0
elif angle <= 150:
    score = 1.0 - 0.6 * (angle - 90) / 60
else:
    score = 0.2

测试：

Rule-1:
0.5 * S_H
+ 0.5 * S_M

Rule-2:
0.5 * S_vis
+ 0.5 * S_ori

Rule-3:
0.30 * S_H
+ 0.30 * S_M
+ 0.20 * S_vis
+ 0.20 * S_ori

最终：

w_v = S_v / sum(S)

z_fused = sum(w_v * logits_v)

禁止使用 true unseen 调系数。

==================================================
COMMON EVALUATION
==================================================

比较：

Equal Mean
Lightweight Gating
Rule-1
Rule-2
Rule-3

view sets:

Random-2
Orthogonal-2
Random-3
Orthogonal-3
All-valid

分别报告：

Seen
True unseen

输出：

Top-1
Delta vs Mean
paired bootstrap 95% CI

保存每个 sample 的最终 weights。

==================================================
IMPORTANT
==================================================

不要继续 PPO。
不要上 Transformer。
不要做 feature-level fusion。
不要重新训练 VPOCLIP。

这轮跑完以后只输出结果和分析，
不要自动继续更复杂的 fusion。
```

---

# 8. 本轮目标

Experiment A 回答：

> 一个 10 多维输入、只有一层 hidden layer 的轻量 gating network，是否能比 equal mean 更好？

Experiment B 回答：

> 完全不用网络，只用 entropy + margin + skeleton visibility + body orientation，是否已经足以得到更合理的 view weights？

如果两个方法都几乎等于 Equal Mean，就先停，不要马上上复杂网络。
