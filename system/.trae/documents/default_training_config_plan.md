# 默认训练配置计划

## 配置要求
- **客户端数量**：30个（固定）
- **本地训练轮数**：1次（固定）
- **全局轮数**：根据实验需求（默认50轮）
- **批次大小**：10（默认）
- **学习率**：0.01（默认）
- **近端正则化系数**：0.01（默认）
- **模型**：CNN（默认）
- **设备**：cuda（默认）

## 实验计划

## [ ] Task 1: 完成当前DP-Prox优化测试
- **Priority**: P0
- **Depends On**: None
- **Description**:
  - 完成当前正在运行的DP-Prox 50轮实验
  - 验证优化效果
  - 分析性能指标
- **Success Criteria**:
  - 实验成功运行50轮
  - 每轮训练时间 < 60秒
  - 模型准确率达到75%以上
- **Test Requirements**:
  - `programmatic` TR-1.1: 实验成功运行50轮
  - `programmatic` TR-1.2: 每轮训练时间 < 60秒
  - `programmatic` TR-1.3: 模型准确率达到75%以上

## [/] Task 2: 运行DP-Native-FedProx 50轮实验
- **Priority**: P0
- **Depends On**: Task 1
- **Description**:
  - 使用默认配置运行DP-Native-FedProx 50轮实验
  - 验证隐私保护效果
  - 分析性能指标
- **Success Criteria**:
  - 实验成功运行50轮
  - 每轮训练时间 < 60秒
  - 模型准确率达到75%以上
- **Test Requirements**:
  - `programmatic` TR-2.1: 实验成功运行50轮
  - `programmatic` TR-2.2: 每轮训练时间 < 60秒
  - `programmatic` TR-2.3: 模型准确率达到75%以上

## [ ] Task 3: 运行FedProx 50轮实验（基线）
- **Priority**: P0
- **Depends On**: Task 2
- **Description**:
  - 使用默认配置运行FedProx 50轮实验作为基线
  - 验证默认配置的性能
  - 为后续实验提供参考
- **Success Criteria**:
  - 实验成功运行50轮
  - 每轮训练时间 < 60秒
  - 模型准确率达到80%以上
- **Test Requirements**:
  - `programmatic` TR-3.1: 实验成功运行50轮
  - `programmatic` TR-3.2: 每轮训练时间 < 60秒
  - `programmatic` TR-3.3: 模型准确率达到80%以上

## [ ] Task 4: 生成实验对比报告
- **Priority**: P1
- **Depends On**: Task 1, Task 2, Task 3
- **Description**:
  - 对比三种方案的性能指标
  - 分析隐私保护对模型性能的影响
  - 生成详细的实验对比报告
- **Success Criteria**:
  - 生成详细的对比分析
  - 提供性能对比图表
  - 总结三种方案的优缺点和适用场景
- **Test Requirements**:
  - `programmatic` TR-4.1: 对比三种方案的准确率、AUC、训练时间
  - `human-judgement` TR-4.2: 分析隐私保护对模型性能的影响

## 执行策略
1. **配置一致性**：所有实验使用相同的客户端数量（30）和本地训练轮数（1）
2. **参数标准化**：使用默认参数配置，确保实验结果的可比性
3. **顺序执行**：按顺序运行三种方案的实验，确保资源合理利用
4. **详细记录**：记录每轮训练的性能指标和耗时
5. **对比分析**：生成详细的对比报告，分析三种方案的优缺点

## 预期结果
- 三种方案都能成功运行50轮
- 每轮训练时间控制在60秒以内
- 模型性能达到预期目标
- 生成详细的实验对比报告，为后续研究提供参考