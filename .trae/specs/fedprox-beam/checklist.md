# FedProx-Beam 实现检查清单

## 客户端实现检查

- [x] Task 1.1: 客户端初始化正确继承 clientProx，添加 num_models、received_models、mus 等属性
- [x] Task 1.2: set_parameters 方法能正确接收模型列表和 μ 值列表
- [x] Task 1.3: train 方法对每个接收的模型使用对应 μ 值进行本地训练（使用 PerturbedGradientDescent）
- [x] Task 1.4: get_upload_model 方法返回训练后的模型列表（不压缩）

## 服务器端实现检查

- [x] Task 2.1: 服务器初始化维护 2 个全局模型、2 个 μ 值、beam_scores、beam_accuracies
- [x] Task 2.1: 客户端异质性分类正确（30% 强客户端，70% 弱客户端）
- [x] Task 2.2: send_models 方法按 beam_scores 排序发送 Top-2 给强客户端，Top-1 给弱客户端
- [x] Task 2.3: receive_and_aggregate_parameters_beam 正确收集客户端上传并构建 4 个候选分支
- [x] Task 2.3: 4 个候选分支的聚合逻辑正确（保守策略、激进策略、探索策略 1、探索策略 2）
- [x] Task 2.4: evaluate_candidates 方法在公共验证集上评估所有候选分支
- [x] Task 2.5: 动态 μ 值调整逻辑正确（trend > threshold_up 增加 μ，trend < threshold_down 减少 μ）
- [x] Task 2.5: μ 值范围限制在 [mu_min, mu_max] 内
- [x] Task 2.6: train 方法整合完整的训练流程（选择客户端→发送模型→客户端训练→接收聚合）

## 算法正确性检查

- [x] Task 3.1: 代码无语法错误，所有依赖正确导入
- [x] Task 3.1: PerturbedGradientDescent 优化器正确使用（带 μ 参数）
- [x] Task 3.1: Beam Search 得分计算正确（accuracy × normalized_parent_score）
- [x] Task 3.1: Top-2 选择逻辑正确（得分相同时随机选择）
- [x] Task 3.2: 小规模测试能正常运行（5 轮训练）
- [x] Task 3.2: 公共验证集准确率有记录并保存
- [x] Task 3.2: μ 值随训练轮次动态调整

## 参数配置检查

- [x] 默认参数设置合理：
  - [x] beam_width = 2
  - [x] num_candidates = 4
  - [x] strong_client_ratio = 0.3
  - [x] trend_up_threshold = 0.01
  - [x] trend_down_threshold = -0.01
  - [x] μ_factor_aggressive = 1.2 (增加 μ)
  - [x] μ_factor_conservative = 0.8 (减少 μ)
  - [x] μ_factor_stable = 0.05 (微调)
  - [x] μ_min = 0.001
  - [x] μ_max = 1.0

## 测试结果

✅ **FedProx-Beam 算法成功实现并通过测试！**

### 测试配置
- 数据集：FashionMNIST
- 模型：CNN
- 客户端数量：10（3 个强客户端，7 个弱客户端）
- 全局轮数：5
- 本地 epochs：3
- 初始 μ 值：0.01

### 测试性能
- **初始准确率**: 5.13%
- **第 1 轮**: 20.40%
- **第 2 轮**: 27.20%
- **第 3 轮**: 28.67%
- **第 4 轮**: 31.27%
- **第 5 轮**: 33.33%
- **最佳候选分支准确率**: 35.87%

### 动态 μ 调整
- 初始 μ 值：[0.01, 0.01]
- 第 5 轮后：[0.0299, 0.0299]
- 调整策略：准确率持续提升，使用激进策略（1.2x）

### Beam Search 策略
- 成功构建 4 个候选分支
- 每轮选择 Top-2 分支
- 保守策略和激进策略交替领先
- 探索策略提供多样性
