# FedProx-Beam 算法规格说明

## Why

基于 FedProx 算法框架，结合 FedBeam 的 Beam Search 多分支探索策略，创建一种新的联邦学习算法。该算法保留 FedProx 的 proximal term 正则化特性来处理 non-iid 数据，同时引入 Beam Search 策略维护多个全局模型分支，通过动态调整每个分支的 μ 值来优化收敛性能。

## What Changes

- **新增算法文件**：创建 `serverproxbeam.py` 和 `clientproxbeam.py`，结合 FedProx 和 FedBeam 的核心思想
- **Beam Search 策略**：维护 2 个全局模型分支（Beam Width=2），每轮生成 4 个候选分支
- **动态 μ值调整**：替代 FedBeam 的 TopK 压缩率调整，改为根据训练趋势动态调整每个分支的 proximal term 参数 μ
- **客户端异质性处理**：保留 FedBeam 的强/弱客户端区分策略
- **简化通信**：不实现 TopK 模型压缩，直接传输完整模型参数
- **评估策略**：在公共验证集上评估 4 个候选分支，选择 Top-2 最佳分支

## Impact

- **受影响的核心文件**：
  - 新增：`system/flcore/servers/serverproxbeam.py`
  - 新增：`system/flcore/clients/clientproxbeam.py`
  - 依赖：`system/flcore/optimizers/fedoptimizer.py` (PerturbedGradientDescent)
  - 依赖：`system/flcore/servers/serverbase.py` (基类)
  - 依赖：`system/flcore/clients/clientprox.py` (基类)

## ADDED Requirements

### Requirement: FedProx-Beam 服务器实现
The system SHALL provide a new server implementation that:
- 维护 2 个全局模型及其对应的 μ值
- 每轮训练生成 4 个候选分支（通过不同 μ值组合的客户端更新聚合）
- 在公共验证集上评估 4 个候选分支的准确率
- 计算每个候选分支的 Beam 得分（准确率 × 父分支得分）
- 选择 Top-2 候选分支作为下一轮的全局模型
- 根据训练趋势动态调整每个分支的 μ值

#### Scenario: 服务器训练流程
- **WHEN** 每轮训练开始
- **THEN** 服务器执行以下步骤：
  1. 选择参与客户端（按 join_ratio）
  2. 发送 Top-2 全局模型及其 μ值（强客户端 2 个，弱客户端 1 个）
  3. 接收客户端上传的训练后模型
  4. 构建 4 个候选分支（简单平均聚合）
  5. 在公共验证集上评估 4 个候选分支
  6. 计算 Beam 得分，选择 Top-2
  7. 根据准确率变化趋势调整 μ值
  8. 更新全局模型和 μ值

### Requirement: FedProx-Beam 客户端实现
The system SHALL provide a new client implementation that:
- 接收全局模型和对应的 μ值
- 使用 PerturbedGradientDescent 优化器进行本地训练（带 proximal term）
- 强客户端训练 2 个模型（接收 2 个全局模型），弱客户端训练 1 个模型
- 上传训练后的模型（不压缩）

#### Scenario: 客户端本地训练
- **WHEN** 客户端接收到全局模型和 μ值
- **THEN** 客户端执行以下步骤：
  1. 对每个接收的全局模型，使用对应的 μ值初始化本地模型
  2. 使用 PerturbedGradientDescent 优化器进行本地训练
  3. 训练完成后保存模型参数
  4. 上传所有训练后的模型到服务器

### Requirement: 动态 μ值调整策略
The system SHALL provide adaptive μ adjustment based on training trends:
- 当候选分支准确率相比父分支提升超过阈值时：增加 μ值（加强正则化）
- 当候选分支准确率相比父分支下降超过阈值时：减少 μ值（减弱正则化）
- 当准确率变化在阈值范围内时：微调 μ值（双向调整）

#### Scenario: μ值调整
- **WHEN** 候选分支评估完成
- **THEN** 对每个选中的候选分支：
  - 计算趋势：trend = accuracy_current - accuracy_parent
  - 如果 trend > trend_up_threshold: μ_new = μ_old * μ_factor_aggressive (>1)
  - 如果 trend < trend_down_threshold: μ_new = μ_old * μ_factor_conservative (<1)
  - 否则：μ_new = μ_old * (1 ± μ_factor_stable)
  - 确保 μ值在合理范围内 [μ_min, μ_max]

### Requirement: 客户端异质性处理
The system SHALL differentiate between strong and weak clients:
- 按训练数据量排序，前 30% 为强客户端
- 强客户端：接收 2 个全局模型，训练 2 个模型，上传 2 个模型
- 弱客户端：接收 1 个全局模型，训练 1 个模型，上传 1 个模型

## MODIFIED Requirements

### Requirement: 评估逻辑
**修改自 FedBeam**: 不评估客户端本地准确率，只在公共验证集上评估全局模型
- 每轮评估 4 个候选分支在公共验证集上的准确率
- 保存最佳公共验证集准确率

### Requirement: 模型聚合
**修改自 FedBeam**: 不使用 TopK 压缩，直接进行简单平均聚合
- 强客户端上传 2 个完整模型
- 弱客户端上传 1 个完整模型
- 服务器对相同类型的模型进行简单平均聚合

## REMOVED Requirements

### Requirement: TopK 模型压缩
**Reason**: 本实现专注于探索 Beam Search 策略和 proximal term 参数调整，暂时不实现模型压缩功能
**Migration**: 后续版本可以添加 TopK 压缩作为可选功能

### Requirement: 压缩率动态调整
**Reason**: 替换为 μ值动态调整策略
**Migration**: 原有的 compression_factor_* 参数替换为 μ_factor_* 参数
