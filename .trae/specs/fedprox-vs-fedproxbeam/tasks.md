# FedProx vs FedProx-Beam 对比实验 - 实现计划

## [ ] Task 1: 准备实验环境和数据集
- **Priority**: P0
- **Depends On**: None
- **Description**: 
  - 确保 PFLlib 环境配置正确
  - 确认 FashionMNIST 非独立同分布数据集已生成（30 客户端）
  - 确认公共验证数据集已准备就绪
  - 验证 FedProx 和 FedProx-Beam 算法均已正确实现
- **Acceptance Criteria Addressed**: AC-1
- **Test Requirements**:
  - `programmatic` TR-1.1: 验证 FedProx 算法可正常运行
  - `programmatic` TR-1.2: 验证 FedProx-Beam 算法可正常运行
  - `programmatic` TR-1.3: 确认数据集文件存在且格式正确
- **Notes**: 使用与 FedProx 实验相同的随机种子确保可重复性

## [ ] Task 2: 运行 FedProx 基线实验
- **Priority**: P0
- **Depends On**: Task 1
- **Description**: 
  - 使用与已知结果完全相同的参数配置运行 FedProx
  - 参数: --algorithm FedProx --dataset FashionMNIST --num_clients 30 --num_global_rounds 50 --local_epochs 3 --learning_rate 0.01 --mu 0.01 --batch_size 10 --device cuda --seed 42
  - 记录完整训练过程数据（准确率、时间、内存使用等）
  - 验证是否能达到 84.93% 的准确率结果
- **Acceptance Criteria Addressed**: AC-1
- **Test Requirements**:
  - `programmatic` TR-2.1: 成功完成 50 轮训练
  - `programmatic` TR-2.2: 记录完整的准确率变化曲线
  - `programmatic` TR-2.3: 记录时间和资源使用数据
- **Notes**: 作为基线，必须复现原实验结果

## [ ] Task 3: 运行 FedProx-Beam 对比实验
- **Priority**: P0
- **Depends On**: Task 1
- **Description**: 
  - 在完全相同的实验环境下运行 FedProx-Beam
  - 参数: --algorithm FedProxBeam --dataset FashionMNIST --num_clients 30 --num_global_rounds 50 --local_epochs 3 --learning_rate 0.01 --mu 0.01 --batch_size 10 --device cuda --seed 42 --trend_up 0.01 --trend_down -0.01 --cf_agg 1.2 --cf_con 0.8 --cf_sta 0.05
  - 记录完整训练过程数据（准确率、时间、内存使用、μ值变化等）
- **Acceptance Criteria Addressed**: AC-1
- **Test Requirements**:
  - `programmatic` TR-3.1: 成功完成 50 轮训练
  - `programmatic` TR-3.2: 记录准确率变化曲线和μ值演化过程
  - `programmatic` TR-3.3: 记录候选分支选择历史
- **Notes**: 保持所有参数与 FedProx 相同，仅算法不同

## [ ] Task 4: 收集和整理实验数据
- **Priority**: P0
- **Depends On**: Task 2, Task 3
- **Description**: 
  - 从两个实验的结果文件中提取关键指标
  - 整理准确率变化数据用于绘图
  - 整理时间、内存等资源使用数据
  - 整理 FedProx-Beam 的μ值变化和候选分支选择数据
- **Acceptance Criteria Addressed**: AC-2
- **Test Requirements**:
  - `programmatic` TR-4.1: 提取完整的准确率时间序列数据
  - `programmatic` TR-4.2: 验证数据完整性（无缺失值）
  - `programmatic` TR-4.3: 生成数据对比表格
- **Notes**: 数据格式应便于后续可视化和分析

## [ ] Task 5: 生成对比分析报告
- **Priority**: P0
- **Depends On**: Task 4
- **Description**: 
  - 创建准确率对比曲线图（FedProx vs FedProx-Beam）
  - 生成性能指标对比表
  - 分析收敛速度差异
  - 分析μ值动态调整的影响
  - 评估稳定性差异
- **Acceptance Criteria Addressed**: AC-3
- **Test Requirements**:
  - `human-judgment` TR-5.1: 准确率对比图表清晰可读
  - `human-judgment` TR-5.2: 性能指标表格完整准确
  - `human-judgment` TR-5.3: 收敛速度分析客观合理
- **Notes**: 图表应突出关键差异点

## [ ] Task 6: 撰写结论和建议
- **Priority**: P1
- **Depends On**: Task 5
- **Description**: 
  - 总结 FedProx-Beam 相对于 FedProx 的优势
  - 量化性能提升（准确率、收敛速度等）
  - 提出算法改进建议
  - 识别潜在问题和限制
- **Acceptance Criteria Addressed**: AC-4
- **Test Requirements**:
  - `human-judgment` TR-6.1: 结论基于数据分析得出
  - `human-judgment` TR-6.2: 优势量化具体明确
  - `human-judgment` TR-6.3: 建议具有可操作性
- **Notes**: 客观评估，避免夸大或低估

## [ ] Task 7: 验证实验可重复性
- **Priority**: P2
- **Depends On**: Task 5
- **Description**: 
  - 重新运行部分实验验证结果一致性
  - 检查是否存在随机性导致的显著差异
  - 确保实验结果可靠
- **Acceptance Criteria Addressed**: AC-1
- **Test Requirements**:
  - `programmatic` TR-7.1: 重复实验结果相似
  - `human-judgment` TR-7.2: 结果差异在可接受范围内
- **Notes**: 如果差异过大，需要分析原因
