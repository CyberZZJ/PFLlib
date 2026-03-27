# 差分隐私方案验证计划

## 验证目标
验证 DP-Prox 和 DP-Native-FedProx 方案是否满足差分隐私要求，特别是噪声分布是否符合高斯分布。

## [ ] Task 1: 分析 DP-Prox 噪声生成机制
- **Priority**: P0
- **Depends On**: None
- **Description**:
  - 分析 clientdpprox.py 中的噪声生成代码
  - 验证噪声分布是否符合高斯分布
  - 验证是否满足 (ε, δ)-差分隐私的数学要求
- **Success Criteria**:
  - 噪声生成符合高斯分布
  - 满足差分隐私的数学条件
- **Test Requirements**:
  - `programmatic` TR-1.1: 生成噪声样本并验证其分布
  - `programmatic` TR-1.2: 计算隐私预算 ε 是否符合设置

## [ ] Task 2: 分析 DP-Native-FedProx 噪声生成机制
- **Priority**: P0
- **Depends On**: None
- **Description**:
  - 分析 clientdpproxnative.py 中的隐私正则项和噪声生成
  - 验证补充噪声是否符合高斯分布
  - 验证输入梯度正则化是否提供差分隐私保护
- **Success Criteria**:
  - 补充噪声符合高斯分布
  - 隐私正则化满足差分隐私要求
- **Test Requirements**:
  - `programmatic` TR-2.1: 验证补充噪声的分布
  - `programmatic` TR-2.2: 分析输入梯度正则化的隐私保护效果

## [ ] Task 3: 运行噪声分布验证实验
- **Priority**: P1
- **Depends On**: Task 1, Task 2
- **Description**:
  - 生成大量噪声样本
  - 进行正态性检验
  - 计算统计指标验证分布特性
- **Success Criteria**:
  - 噪声样本通过正态性检验
  - 均值和方差符合预期
- **Test Requirements**:
  - `programmatic` TR-3.1: 生成 10000 个噪声样本
  - `programmatic` TR-3.2: 执行 Shapiro-Wilk 正态性检验
  - `programmatic` TR-3.3: 计算均值、方差、偏度、峰度

## [ ] Task 4: 验证差分隐私数学条件
- **Priority**: P1
- **Depends On**: Task 1, Task 2
- **Description**:
  - 根据高斯机制的数学公式验证隐私保证
  - 计算实际的隐私预算 ε
  - 验证噪声尺度设置是否合理
- **Success Criteria**:
  - 隐私预算计算正确
  - 噪声尺度符合差分隐私要求
- **Test Requirements**:
  - `programmatic` TR-4.1: 计算理论隐私预算
  - `programmatic` TR-4.2: 验证噪声尺度与隐私预算的关系

## [x] Task 5: 生成验证报告
- **Priority**: P2
- **Depends On**: Task 3, Task 4
- **Description**:
  - 汇总验证结果
  - 分析两个方案的隐私保护效果
  - 提供改进建议
- **Success Criteria**:
  - 详细的验证报告
  - 明确的结论和建议
- **Test Requirements**:
  - `human-judgement` TR-5.1: 报告内容完整清晰
  - `human-judgement` TR-5.2: 结论有数据支持

## 验证方法
1. **代码分析**：检查噪声生成和隐私保护机制的实现
2. **统计验证**：生成噪声样本并进行分布检验
3. **数学验证**：验证差分隐私的数学条件
4. **实验验证**：运行小规模实验验证隐私保护效果

## 预期结果
- 确认两个方案都满足差分隐私要求
- 验证噪声分布符合高斯分布
- 提供详细的验证报告和改进建议