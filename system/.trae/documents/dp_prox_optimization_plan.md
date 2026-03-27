# DP-Prox 优化计划

## 优化目标
1. 满足差分隐私的隐私安全要求
2. 优化输出开销，减少不必要的打印信息
3. 降低训练时间，恢复到之前的性能水平
4. 确保模型能够正常收敛

## 优化计划

## [ ] Task 1: 优化输出开销
- **Priority**: P0
- **Depends On**: None
- **Description**:
  - 减少不必要的打印信息
  - 只保留关键的训练指标和耗时信息
  - 避免每轮打印所有客户端的训练耗时
- **Success Criteria**:
  - 输出信息简洁明了
  - 每轮训练时间减少至少30%
  - 保留必要的训练指标和耗时信息
- **Test Requirements**:
  - `programmatic` TR-1.1: 每轮训练时间减少至少30%
  - `human-judgement` TR-1.2: 输出信息简洁明了，只包含关键信息

## [ ] Task 2: 优化差分隐私参数
- **Priority**: P0
- **Depends On**: None
- **Description**:
  - 调整噪声尺度和裁剪阈值
  - 确保满足差分隐私的隐私安全要求
  - 同时保证模型能够正常收敛
- **Success Criteria**:
  - 隐私预算 ε < 10（强隐私保护）
  - 模型准确率达到75%以上
  - 训练过程稳定，无NaN值
- **Test Requirements**:
  - `programmatic` TR-2.1: 隐私预算 ε < 10
  - `programmatic` TR-2.2: 模型准确率达到75%以上

## [ ] Task 3: 优化数值稳定性
- **Priority**: P1
- **Depends On**: Task 2
- **Description**:
  - 优化NaN值处理逻辑
  - 减少不必要的数值检查和裁剪
  - 确保训练过程稳定
- **Success Criteria**:
  - 训练过程无NaN值
  - 计算开销减少
  - 模型收敛稳定
- **Test Requirements**:
  - `programmatic` TR-3.1: 训练过程无NaN值
  - `programmatic` TR-3.2: 计算开销减少10%以上

## [/] Task 4: 测试优化效果
- **Priority**: P1
- **Depends On**: Task 1, Task 2, Task 3
- **Description**:
  - 运行优化后的DP-Prox 50轮实验
  - 对比优化前后的性能
  - 验证隐私保护效果
- **Success Criteria**:
  - 实验成功运行50轮
  - 训练时间恢复到之前的水平
  - 模型性能保持稳定
  - 隐私保护效果满足要求
- **Test Requirements**:
  - `programmatic` TR-4.1: 实验成功运行50轮
  - `programmatic` TR-4.2: 每轮训练时间 < 60秒
  - `programmatic` TR-4.3: 模型准确率达到75%以上

## 优化策略

### 输出开销优化
1. 只在服务器端打印每轮的整体耗时
2. 客户端只打印关键信息，避免每轮打印所有客户端的训练耗时
3. 使用日志级别控制，只输出必要的信息

### 差分隐私参数优化
1. 噪声尺度：初始值设为0.1，每轮衰减5%
2. 裁剪阈值：设为1.0
3. 动态调整噪声尺度，平衡隐私保护和模型性能

### 数值稳定性优化
1. 只在必要时进行NaN值检查
2. 减少不必要的裁剪操作
3. 优化计算逻辑，减少冗余计算

## 预期结果
- 每轮训练时间恢复到60秒以内
- 模型准确率达到75%以上
- 隐私预算 ε < 10
- 输出信息简洁明了，只包含关键信息
- 训练过程稳定，无NaN值