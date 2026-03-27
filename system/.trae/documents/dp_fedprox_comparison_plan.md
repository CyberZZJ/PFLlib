# FedProx、DP-Prox、DP-Native-FedProx 50轮实验对比计划

## 实验目标
运行50轮的FedProx、DP-Prox、DP-Native-FedProx实验，对比三种方案的性能和隐私保护效果。

## 实验配置

### 通用配置
- **数据集**：FashionMNIST
- **客户端数量**：30
- **全局轮数**：50
- **本地轮数**：3
- **学习率**：0.01
- **近端正则化系数**：0.01
- **批次大小**：10
- **模型**：CNN
- **设备**：cuda

### 差分隐私配置
- **DP-Prox**：
  - 噪声尺度：1.0（动态调整）
  - 裁剪阈值：1.0
  - 隐私参数 δ：1e-5

- **DP-Native-FedProx**：
  - 隐私正则系数：0.05
  - 补充噪声尺度：1.0（动态调整）
  - 裁剪阈值：1.0
  - 隐私参数 δ：1e-5

## 实验计划

## [/] Task 1: 运行 FedProx 50轮实验
- **Priority**: P0
- **Depends On**: None
- **Description**:
  - 运行标准 FedProx 50轮实验
  - 记录训练过程和性能指标
  - 作为基线对比
- **Success Criteria**:
  - 实验成功运行50轮
  - 记录完整的性能指标
  - 生成结果文件
- **Test Requirements**:
  - `programmatic` TR-1.1: 实验成功运行50轮
  - `programmatic` TR-1.2: 记录准确率、AUC、训练时间等指标

## [/] Task 2: 运行 DP-Prox 50轮实验
- **Priority**: P0
- **Depends On**: None
- **Description**:
  - 运行改进后的 DP-Prox 50轮实验
  - 记录训练过程和性能指标
  - 记录噪声尺度变化
- **Success Criteria**:
  - 实验成功运行50轮
  - 记录完整的性能指标
  - 生成结果文件
- **Test Requirements**:
  - `programmatic` TR-2.1: 实验成功运行50轮
  - `programmatic` TR-2.2: 记录准确率、AUC、训练时间、噪声尺度等指标

## [ ] Task 3: 运行 DP-Native-FedProx 50轮实验
- **Priority**: P0
- **Depends On**: None
- **Description**:
  - 运行改进后的 DP-Native-FedProx 50轮实验
  - 记录训练过程和性能指标
  - 记录噪声尺度变化
- **Success Criteria**:
  - 实验成功运行50轮
  - 记录完整的性能指标
  - 生成结果文件
- **Test Requirements**:
  - `programmatic` TR-3.1: 实验成功运行50轮
  - `programmatic` TR-3.2: 记录准确率、AUC、训练时间、噪声尺度等指标

## [ ] Task 4: 分析实验结果
- **Priority**: P1
- **Depends On**: Task 1, Task 2, Task 3
- **Description**:
  - 对比三种方案的性能指标
  - 分析隐私保护对模型性能的影响
  - 生成对比报告
- **Success Criteria**:
  - 生成详细的对比分析
  - 提供性能对比图表
  - 总结三种方案的优缺点
- **Test Requirements**:
  - `programmatic` TR-4.1: 对比三种方案的准确率、AUC、训练时间
  - `human-judgement` TR-4.2: 分析隐私保护对模型性能的影响

## 实验步骤
1. **准备阶段**：确认实验配置和代码正确性
2. **运行 FedProx**：作为基线实验
3. **运行 DP-Prox**：测试后处理差分隐私方案
4. **运行 DP-Native-FedProx**：测试原生差分隐私方案
5. **分析阶段**：对比三种方案的性能和隐私保护效果
6. **报告生成**：生成详细的实验对比报告

## 预期结果
- 三种方案都能成功运行50轮
- 获得详细的性能对比数据
- 分析隐私保护对模型性能的影响
- 总结三种方案的优缺点和适用场景