# 差分隐私方案改进计划

## 改进目标
1. **DP-Prox**：增加噪声尺度以增强隐私保护
2. **DP-Native-FedProx**：建立差分隐私理论保证，进一步修改

## 改进计划

## [ ] Task 1: 增强 DP-Prox 隐私保护
- **Priority**: P0
- **Depends On**: None
- **Description**:
  - 增加噪声尺度以减小隐私预算
  - 实现动态噪声调整机制
  - 验证增强后的隐私保护效果
- **Success Criteria**:
  - 隐私预算 ε 降低到合理范围（如 ε ≤ 10）
  - 统计验证通过
  - 模型性能保持在可接受范围内
- **Test Requirements**:
  - `programmatic` TR-1.1: 验证隐私预算计算
  - `programmatic` TR-1.2: 验证统计距离是否符合要求
  - `programmatic` TR-1.3: 评估模型性能

## [ ] Task 2: 为 DP-Native-FedProx 建立差分隐私理论保证
- **Priority**: P0
- **Depends On**: None
- **Description**:
  - 分析输入梯度正则化的差分隐私理论基础
  - 实现基于输入梯度正则化的差分隐私保证
  - 结合梯度裁剪和噪声添加增强隐私保护
- **Success Criteria**:
  - 建立输入梯度正则化的差分隐私理论保证
  - 实现增强的隐私保护机制
  - 验证隐私保护效果
- **Test Requirements**:
  - `programmatic` TR-2.1: 验证理论保证的正确性
  - `programmatic` TR-2.2: 评估增强后的隐私保护效果
  - `programmatic` TR-2.3: 评估模型性能

## [ ] Task 3: 实现动态噪声调整机制
- **Priority**: P1
- **Depends On**: Task 1
- **Description**:
  - 在训练过程中动态调整噪声尺度
  - 初期使用较大噪声，后期逐渐减小
  - 验证动态噪声调整的效果
- **Success Criteria**:
  - 实现动态噪声调整机制
  - 验证调整效果
  - 保持模型性能稳定
- **Test Requirements**:
  - `programmatic` TR-3.1: 验证动态噪声调整的实现
  - `programmatic` TR-3.2: 评估调整效果

## [x] Task 4: 验证改进效果
- **Priority**: P1
- **Depends On**: Task 1, Task 2
- **Description**:
  - 运行完整的隐私保护验证
  - 对比改进前后的效果
  - 生成改进报告
- **Success Criteria**:
  - 验证改进后的隐私保护效果
  - 对比改进前后的性能
  - 生成详细的改进报告
- **Test Requirements**:
  - `programmatic` TR-4.1: 运行完整的隐私保护验证
  - `programmatic` TR-4.2: 对比改进前后的效果

## 实施步骤
1. **准备阶段**：分析当前实现，确定改进方向
2. **DP-Prox 改进**：增加噪声尺度，实现动态噪声调整
3. **DP-Native-FedProx 改进**：建立理论保证，增强隐私保护
4. **验证阶段**：运行完整的隐私保护验证
5. **报告生成**：汇总改进效果，生成改进报告

## 预期结果
- DP-Prox 的隐私保护强度显著增强
- DP-Native-FedProx 具有严格的差分隐私理论保证
- 两个方案在隐私保护和模型性能之间取得更好的平衡