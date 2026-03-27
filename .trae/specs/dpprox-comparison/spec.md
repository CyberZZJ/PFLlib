# DP-Prox vs DP-Prox-Native 差分隐私 FedProx 对比实验规格说明

## Why

基于 MDPI 2023 论文 "Consideration of FedProx in Privacy Protection"，实现两种差分隐私 FedProx 方案进行对比：

1. **DP-Prox（基准线）**：标准的后处理差分隐私（梯度裁剪 + 高斯噪声）
2. **DP-Prox-Native（新方案）**：原生差分隐私（隐私损失作为正则项融入优化目标）

目标：验证原生差分隐私方案是否能提供更好的隐私-效用权衡。

## What Changes

- **新增客户端实现**：
  - `clientdpprox.py`：DP-Prox 客户端（后处理差分隐私）
  - `clientdpproxnative.py`：DP-Prox-Native 客户端（原生差分隐私）
- **新增服务器实现**：
  - `serverdpprox.py`：DP-Prox 服务器
  - `serverdpproxnative.py`：DP-Prox-Native 服务器
- **修改 main.py**：添加差分隐私相关参数和算法路由

## Impact

- **受影响的文件**：
  - 新增：`system/flcore/clients/clientdpprox.py`
  - 新增：`system/flcore/clients/clientdpproxnative.py`
  - 新增：`system/flcore/servers/serverdpprox.py`
  - 新增：`system/flcore/servers/serverdpproxnative.py`
  - 修改：`system/main.py`

## ADDED Requirements

### Requirement: DP-Prox 客户端实现（基准线）

The system SHALL implement DP-Prox client following the standard differential privacy mechanism:

#### Scenario: 梯度裁剪和噪声添加
- **WHEN** 客户端完成反向传播计算梯度
- **THEN** 执行以下步骤：
  1. 计算梯度范数：`grad_norm = ||g||_2`
  2. 梯度裁剪：`g_clipped = g * min(1, C / grad_norm)`
  3. 添加高斯噪声：`g_noisy = g_clipped + N(0, (σC)²I)`

#### 参数配置
- `dp_clip_norm (C)`：梯度裁剪阈值，默认 1.0
- `dp_noise_multiplier (σ)`：噪声倍数，默认 1.0
- `dp_epsilon (ε)`：隐私预算，默认 10.0
- `dp_delta (δ)`：隐私参数，默认 1e-5

#### 隐私参数关系
```
ε ≥ 2ln(1.25/δ) · C²/σ²
```

### Requirement: DP-Prox-Native 客户端实现（新方案）

The system SHALL implement DP-Prox-Native client with privacy loss as regularization term:

#### Scenario: 原生隐私正则化
- **WHEN** 客户端进行本地训练
- **THEN** 优化目标为：
```
min_w h_k(w; w^t) = F_k(w) + (μ/2)||w - w^t||² + λ · R_privacy(w)
```

其中：
- `F_k(w)`：本地经验损失
- `(μ/2)||w - w^t||²`：FedProx 近端项
- `λ · R_privacy(w)`：原生隐私损失正则项

#### 隐私正则项实现
采用梯度 L2 范数平滑正则项：
```python
R_privacy(w) = (1/n) * Σ ||∇L_i(w)||²
```

#### 参数配置
- `lambda_privacy (λ)`：隐私正则项系数，默认 0.05
- `use_supplementary_noise`：是否使用补充噪声，默认 False
- `supplementary_noise_scale`：补充噪声尺度，默认 0.001

### Requirement: 实验配置一致性

The system SHALL ensure identical experimental conditions for fair comparison:

#### 实验参数
- **数据集**：FashionMNIST（30 客户端，non-iid 分布）
- **全局轮数**：30 轮
- **本地轮数**：3 轮
- **学习率**：0.01
- **近端系数 (μ)**：0.01
- **批量大小**：10
- **设备**：GPU (CUDA)

#### 差分隐私参数
- **DP-Prox**：
  - C = 1.0, σ = 1.0, ε = 10.0, δ = 1e-5
- **DP-Prox-Native**：
  - λ = 0.05, use_supplementary_noise = False

### Requirement: 性能指标记录

The system SHALL record comprehensive metrics for comparison:

#### 准确率指标
- 每轮测试准确率
- 最佳测试准确率
- 最终测试准确率
- AUC 分数

#### 收敛速度指标
- 达到 70% 准确率所需轮次
- 达到 80% 准确率所需轮次

#### 时间指标
- 总训练时间
- 平均每轮时间

#### 隐私指标（DP-Prox）
- 隐私预算消耗
- 噪声添加量

## MODIFIED Requirements

### Requirement: main.py 参数扩展

添加差分隐私相关命令行参数：

```python
# DP-Prox 参数
parser.add_argument('-dp', '--dp_enabled', action='store_true')
parser.add_argument('-dp_clip', '--dp_clip_norm', type=float, default=1.0)
parser.add_argument('-dp_noise', '--dp_noise_multiplier', type=float, default=1.0)
parser.add_argument('-dp_eps', '--dp_epsilon', type=float, default=10.0)
parser.add_argument('-dp_delta', '--dp_delta', type=float, default=1e-5)

# DP-Prox-Native 参数
parser.add_argument('-lambda_priv', '--lambda_privacy', type=float, default=0.05)
parser.add_argument('-sup_noise', '--use_supplementary_noise', action='store_true')
parser.add_argument('-sup_noise_scale', '--supplementary_noise_scale', type=float, default=0.001)
```

## REMOVED Requirements

无 - 本实验不删除任何现有功能

## Acceptance Criteria

### AC-1: DP-Prox 实现正确性
- **Given**: DP-Prox 客户端实现完成
- **When**: 运行 30 轮训练
- **Then**: 成功完成训练，梯度裁剪和噪声添加正常工作
- **Verification**: `programmatic` - 检查训练日志和结果文件

### AC-2: DP-Prox-Native 实现正确性
- **Given**: DP-Prox-Native 客户端实现完成
- **When**: 运行 30 轮训练
- **Then**: 成功完成训练，隐私正则项正常工作
- **Verification**: `programmatic` - 检查训练日志和结果文件

### AC-3: 对比实验公平性
- **Given**: 两个实验完成
- **When**: 对比结果
- **Then**: 实验配置完全一致，仅差分隐私机制不同
- **Verification**: `human-judgment` - 检查参数配置

### AC-4: 性能对比分析
- **Given**: 两个实验结果
- **When**: 生成对比报告
- **Then**: 包含准确率、收敛速度、隐私保护强度对比
- **Verification**: `human-judgment` - 报告内容完整

## Open Questions

- [ ] DP-Prox-Native 的隐私正则项系数 λ 如何选择最优值？
- [ ] 是否需要实现隐私预算追踪机制？
- [ ] 补充噪声对 DP-Prox-Native 的影响有多大？
