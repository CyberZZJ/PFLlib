# DP-Prox vs DP-Prox-Native 对比实验 - 实现计划

## [x] Task 1: 实现 DP-Prox 客户端（基准线）
- **Priority**: P0
- **Depends On**: None
- **Description**: 
  - 复制 clientprox.py 为 clientdpprox.py
  - 添加差分隐私参数：dp_enabled, dp_clip_norm, dp_noise_multiplier, dp_epsilon, dp_delta
  - 在训练循环中实现梯度裁剪和噪声添加
  - 实现标准的高斯机制：(ε, δ)-DP
- **Acceptance Criteria Addressed**: AC-1
- **Test Requirements**:
  - `programmatic` TR-1.1: 梯度裁剪功能正常
  - `programmatic` TR-1.2: 高斯噪声添加功能正常
  - `programmatic` TR-1.3: 隐私参数关系正确
- **Notes**: 完全遵循论文 "Consideration of FedProx in Privacy Protection" 的标准实现

## [x] Task 2: 实现 DP-Prox 服务器
- **Priority**: P0
- **Depends On**: Task 1
- **Description**: 
  - 复制 serverprox.py 为 serverdpprox.py
  - 继承 FedProx 的所有功能
  - 添加差分隐私参数传递
- **Acceptance Criteria Addressed**: AC-1
- **Test Requirements**:
  - `programmatic` TR-2.1: 服务器能正确初始化 DP-Prox 客户端
  - `programmatic` TR-2.2: 参数传递正确
- **Notes**: 服务器端无需额外处理，差分隐私在客户端完成

## [x] Task 3: 实现 DP-Prox-Native 客户端（新方案）
- **Priority**: P0
- **Depends On**: None
- **Description**: 
  - 复制 clientprox.py 为 clientdpproxnative.py
  - 添加隐私正则项参数：lambda_privacy, use_supplementary_noise, supplementary_noise_scale
  - 修改优化目标：添加隐私损失正则项 λ · R_privacy(w)
  - 实现梯度 L2 范数平滑正则项
  - 修改反向传播逻辑：隐私正则项参与梯度计算
- **Acceptance Criteria Addressed**: AC-2
- **Test Requirements**:
  - `programmatic` TR-3.1: 隐私正则项正确计算
  - `programmatic` TR-3.2: 隐私正则项参与反向传播
  - `programmatic` TR-3.3: 补充噪声功能可选
- **Notes**: 核心创新点 - 隐私损失作为正则项原生融入优化目标

## [x] Task 4: 实现 DP-Prox-Native 服务器
- **Priority**: P0
- **Depends On**: Task 3
- **Description**: 
  - 复制 serverprox.py 为 serverdpproxnative.py
  - 继承 FedProx 的所有功能
  - 添加隐私正则项参数传递
- **Acceptance Criteria Addressed**: AC-2
- **Test Requirements**:
  - `programmatic` TR-4.1: 服务器能正确初始化 DP-Prox-Native 客户端
  - `programmatic` TR-4.2: 参数传递正确
- **Notes**: 服务器端无需额外处理，隐私正则项在客户端完成

## [x] Task 5: 修改 main.py 添加算法支持
- **Priority**: P0
- **Depends On**: Task 2, Task 4
- **Description**: 
  - 添加 DP-Prox 和 DP-Prox-Native 算法路由
  - 添加差分隐私相关命令行参数
  - 确保参数正确传递给服务器和客户端
- **Acceptance Criteria Addressed**: AC-3
- **Test Requirements**:
  - `programmatic` TR-5.1: DP-Prox 算法能正确启动
  - `programmatic` TR-5.2: DP-Prox-Native 算法能正确启动
  - `programmatic` TR-5.3: 参数传递正确
- **Notes**: 确保命令行参数名称清晰易懂

## [ ] Task 6: 运行 DP-Prox 基准线实验（30 轮）
- **Priority**: P0
- **Depends On**: Task 5
- **Description**: 
  - 使用 FashionMNIST 数据集
  - 30 个客户端，non-iid 分布
  - 30 轮全局训练，3 轮本地训练
  - 差分隐私参数：C=1.0, σ=1.0, ε=10.0, δ=1e-5
  - 记录完整训练过程数据
- **Acceptance Criteria Addressed**: AC-3
- **Test Requirements**:
  - `programmatic` TR-6.1: 成功完成 30 轮训练
  - `programmatic` TR-6.2: 记录完整准确率数据
  - `programmatic` TR-6.3: 记录隐私预算消耗
- **Notes**: 作为基准线，需要确保实现正确

## [ ] Task 7: 运行 DP-Prox-Native 对比实验（30 轮）
- **Priority**: P0
- **Depends On**: Task 5
- **Description**: 
  - 使用完全相同的实验配置
  - 隐私正则项参数：λ=0.05, use_supplementary_noise=False
  - 记录完整训练过程数据
- **Acceptance Criteria Addressed**: AC-3
- **Test Requirements**:
  - `programmatic` TR-7.1: 成功完成 30 轮训练
  - `programmatic` TR-7.2: 记录完整准确率数据
  - `programmatic` TR-7.3: 隐私正则项正常工作
- **Notes**: 与 DP-Prox 保持完全相同的实验条件

## [ ] Task 8: 收集和整理实验数据
- **Priority**: P0
- **Depends On**: Task 6, Task 7
- **Description**: 
  - 从两个实验的结果文件中提取关键指标
  - 整理准确率变化数据用于绘图
  - 整理时间、内存等资源使用数据
  - 计算收敛速度指标
- **Acceptance Criteria Addressed**: AC-4
- **Test Requirements**:
  - `programmatic` TR-8.1: 提取完整的准确率时间序列数据
  - `programmatic` TR-8.2: 验证数据完整性
  - `programmatic` TR-8.3: 生成数据对比表格
- **Notes**: 数据格式应便于后续可视化和分析

## [ ] Task 9: 生成对比分析报告
- **Priority**: P0
- **Depends On**: Task 8
- **Description**: 
  - 创建准确率对比曲线图
  - 生成性能指标对比表
  - 分析收敛速度差异
  - 分析隐私保护强度差异
  - 评估两种方案的优劣
- **Acceptance Criteria Addressed**: AC-4
- **Test Requirements**:
  - `human-judgment` TR-9.1: 准确率对比图表清晰可读
  - `human-judgment` TR-9.2: 性能指标表格完整准确
  - `human-judgment` TR-9.3: 分析客观合理
- **Notes**: 图表应突出关键差异点

## [ ] Task 10: 撰写结论和建议
- **Priority**: P1
- **Depends On**: Task 9
- **Description**: 
  - 总结 DP-Prox-Native 相对于 DP-Prox 的优势
  - 量化性能提升（准确率、收敛速度等）
  - 提出算法改进建议
  - 识别潜在问题和限制
- **Acceptance Criteria Addressed**: AC-4
- **Test Requirements**:
  - `human-judgment` TR-10.1: 结论基于数据分析得出
  - `human-judgment` TR-10.2: 优势量化具体明确
  - `human-judgment` TR-10.3: 建议具有可操作性
- **Notes**: 客观评估，避免夸大或低估

# Task Dependencies
- [Task 2] depends on [Task 1]
- [Task 4] depends on [Task 3]
- [Task 5] depends on [Task 2, Task 4]
- [Task 6] depends on [Task 5]
- [Task 7] depends on [Task 5]
- [Task 8] depends on [Task 6, Task 7]
- [Task 9] depends on [Task 8]
- [Task 10] depends on [Task 9]
