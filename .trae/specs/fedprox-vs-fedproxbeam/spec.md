# FedProx vs FedProx-Beam 对比实验规格说明

## Why

基于 FedProx 在 FashionMNIST 数据集上 50 轮完整实验的最佳结果（84.93% 准确率），在**完全相同的实验环境和参数配置**下运行 FedProx-Beam 算法，对比验证 Beam Search 策略结合动态μ值调整是否能带来：
1. **更高的最终准确率**（目标：超越 84.93%）
2. **更快的收敛速度**（目标：在更少轮次达到相同准确率）
3. **更好的稳定性**（目标：更小的准确率波动）

## What Changes

- **新增对比实验**：在相同配置下运行 FedProx-Beam 50 轮训练
- **控制变量**：保持数据集、客户端数量、超参数、硬件环境完全一致
- **评估指标**：公共验证集准确率、收敛速度、训练时间、GPU 内存使用
- **结果分析**：定量对比两种算法的性能差异

## Impact

- **受影响的文件**：
  - 现有：`system/flcore/servers/serverprox.py` (FedProx 基线)
  - 现有：`system/flcore/servers/serverproxbeam.py` (FedProx-Beam)
  - 现有：`system/main.py` (启动脚本)
  - 新增：实验结果对比分析报告

## ADDED Requirements

### Requirement: 实验环境一致性
The system SHALL ensure identical experimental conditions:
- **数据集**: FashionMNIST (30 个客户端，non-iid 分布，Dirichlet α=0.1)
- **客户端总数**: 30 个
- **每轮参与客户端**: 30 个 (100% 参与率)
- **公共验证集**: 1500 个样本 (10 个类别，每类 150 个样本)
- **全局训练轮数**: 50 轮
- **本地训练轮数**: 3 轮
- **批量大小 (batch_size)**: 10
- **学习率**: 0.01
- **初始μ值**: 0.01
- **运行设备**: GPU (CUDA)
- **随机种子**: 固定相同种子确保可重复性

### Requirement: FedProx-Beam 参数配置
The system SHALL configure FedProx-Beam with:
- **Beam Width**: 2 (维护 2 个全局模型分支)
- **候选分支数**: 4 (每轮生成 4 个候选)
- **动态μ调整参数**:
  - `trend_up_threshold`: 0.01
  - `trend_down_threshold`: -0.01
  - `mu_factor_aggressive`: 1.2
  - `mu_factor_conservative`: 0.8
  - `mu_factor_stable`: 0.05
  - `mu_min`: 0.001
  - `mu_max`: 1.0
- **客户端异质性**: 30% 强客户端 (num_models=2), 70% 弱客户端 (num_models=1)

### Requirement: 性能指标记录
The system SHALL record comprehensive metrics:
- **准确率指标**:
  - 每轮公共验证集准确率
  - 最佳公共验证集准确率及对应轮次
  - 最终公共验证集准确率
  - AUC 分数
- **收敛速度指标**:
  - 达到 70% 准确率所需轮次
  - 达到 80% 准确率所需轮次
  - 达到 84.93% (FedProx 最佳) 所需轮次
- **时间指标**:
  - 总训练时间
  - 平均每轮时间
  - 单轮时间分布
- **资源指标**:
  - GPU 内存使用峰值
  - CPU 使用率
- **FedProx-Beam 特有指标**:
  - 每轮μ值变化轨迹
  - 每轮选择的候选分支索引
  - 每轮 Beam 得分

### Requirement: 对比分析报告
The system SHALL generate a comprehensive comparison report including:
- **性能对比表**: 并排展示 FedProx 和 FedProx-Beam 的关键指标
- **准确率曲线图**: 50 轮训练过程中准确率变化对比
- **收敛速度分析**: 达到关键阈值所需轮次对比
- **μ值演化分析**: FedProx-Beam 的μ值动态调整过程
- **稳定性分析**: 准确率波动对比
- **效率分析**: 时间开销和计算资源使用对比
- **优势总结**: FedProx-Beam 相对于 FedProx 的具体改进

#### Scenario: 实验成功标准
- **WHEN** FedProx-Beam 完成 50 轮训练
- **THEN** 满足以下至少一项：
  1. 最终准确率 > 84.93% (FedProx 最佳)
  2. 达到 80% 准确率的轮次 < FedProx
  3. 准确率曲线更平滑（波动更小）

## MODIFIED Requirements

### Requirement: 实验启动命令
**FedProx 基线实验**:
```bash
python main.py --algorithm FedProx --dataset FashionMNIST --num_clients 30 \
               --num_global_rounds 50 --local_epochs 3 --learning_rate 0.01 \
               --mu 0.01 --batch_size 10 --device cuda --seed 42
```

**FedProx-Beam 对比实验**:
```bash
python main.py --algorithm FedProxBeam --dataset FashionMNIST --num_clients 30 \
               --num_global_rounds 50 --local_epochs 3 --learning_rate 0.01 \
               --mu 0.01 --batch_size 10 --device cuda --seed 42 \
               --trend_up 0.01 --trend_down -0.01 --cf_agg 1.2 --cf_con 0.8 --cf_sta 0.05
```

## REMOVED Requirements

无 - 本实验不修改 FedProx 或 FedProx-Beam 的现有实现，仅进行对比测试

## Acceptance Criteria

### AC-1: FedProx-Beam 实验运行
- **Given**: 实验环境和参数已配置
- **When**: 运行 FedProx-Beam 训练命令
- **Then**: 成功完成 50 轮全局训练，无错误或崩溃
- **Verification**: `programmatic` - 检查训练日志和结果文件

### AC-2: 性能数据收集
- **Given**: 训练完成
- **When**: 读取结果文件
- **Then**: 获取完整的准确率、时间、μ值变化等数据
- **Verification**: `programmatic` - 验证数据完整性和格式

### AC-3: 对比分析报告
- **Given**: FedProx 和 FedProx-Beam 的实验数据
- **When**: 生成对比报告
- **Then**: 包含性能对比表、曲线图、收敛分析等
- **Verification**: `human-judgment` - 报告内容完整、分析合理

### AC-4: 结论验证
- **Given**: 对比分析报告
- **When**: 分析实验结果
- **Then**: 明确回答 FedProx-Beam 是否在准确率、收敛速度、稳定性上有优势
- **Verification**: `human-judgment` - 基于数据得出结论

## Open Questions

- [ ] FedProx-Beam 的额外计算开销（评估 4 个候选分支）是否会影响单轮训练时间？
- [ ] 动态μ值调整是否会导致训练后期μ值过大或过小？
- [ ] Beam Search 策略在 non-iid 数据下的探索能力是否真能带来性能提升？
- [ ] 强/弱客户端的区分对 FedProx-Beam 的性能影响有多大？
