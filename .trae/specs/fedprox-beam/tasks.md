# Tasks

- [x] Task 1: 创建 FedProx-Beam 客户端实现 (clientproxbeam.py)
  - [x] Task 1.1: 实现客户端初始化：继承 clientProx，添加 num_models、received_models 等属性
  - [x] Task 1.2: 实现 set_parameters 方法：接收全局模型列表和 μ值列表
  - [x] Task 1.3: 实现 train 方法：对每个接收的模型使用对应 μ值进行本地训练
  - [x] Task 1.4: 实现 get_upload_model 方法：返回训练后的模型列表（不压缩）

- [x] Task 2: 创建 FedProx-Beam 服务器实现 (serverproxbeam.py)
  - [x] Task 2.1: 实现服务器初始化：维护 2 个全局模型、2 个 μ值、beam_scores、beam_accuracies
  - [x] Task 2.2: 实现 send_models 方法：发送 Top-2 模型给强客户端，Top-1 给弱客户端
  - [x] Task 2.3: 实现 receive_and_aggregate_parameters_beam 方法：核心 Beam Search 聚合逻辑
  - [x] Task 2.4: 实现 evaluate_candidates 方法：在公共验证集上评估 4 个候选分支
  - [x] Task 2.5: 实现动态 μ值调整逻辑：根据训练趋势调整每个分支的 μ值
  - [x] Task 2.6: 实现 train 方法：整合完整的训练流程

- [x] Task 3: 验证和测试
  - [x] Task 3.1: 检查代码语法和依赖
  - [x] Task 3.2: 运行小规模测试验证算法正确性

# Task Dependencies

- Task 1 可以独立开始
- Task 2 依赖于 Task 1 完成（服务器需要与客户端交互）
- Task 3 依赖于 Task 1 和 Task 2 完成

# 测试结果

FedProx-Beam 算法在 FashionMNIST 数据集上成功运行 5 轮测试：
- 初始准确率：5.13%
- 第 5 轮准确率：35.87%
- μ 值从 0.01 动态调整到 0.0299
- Beam Search 策略正常工作
- 动态 μ 调整机制正常工作
