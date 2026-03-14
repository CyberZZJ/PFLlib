# FedProx FashionMNIST 联邦学习实验 - 实现计划

## [x] Task 1: 检查PFLlib环境配置
- **Priority**: P0
- **Depends On**: None
- **Description**: 
  - 检查PFLlib框架是否正确安装
  - 验证必要的依赖项是否存在
  - 确保代码库能够正常运行
- **Acceptance Criteria Addressed**: AC-1, AC-2
- **Test Requirements**:
  - `programmatic` TR-1.1: 运行简单的PFLlib示例代码，验证环境配置正确
  - `human-judgement` TR-1.2: 确认所有必要的依赖项已安装
- **Notes**: 环境配置是后续所有任务的基础

## [/] Task 2: 使用现有代码生成30个客户端的FashionMNIST非独立同分布数据集
- **Priority**: P0
- **Depends On**: Task 1
- **Description**: 
  - 使用PFLlib框架中现有的数据生成功能
  - 配置数据生成参数，设置30个客户端
  - 选择合适的non-iid数据分布策略（如Dirichlet分布）
  - 生成并保存数据集
- **Acceptance Criteria Addressed**: AC-1
- **Test Requirements**:
  - `programmatic` TR-2.1: 成功生成30个客户端的数据集文件
  - `programmatic` TR-2.2: 验证数据集的non-iid特性（如标签分布不均）
- **Notes**: 建议使用Dirichlet分布生成non-iid数据，参数α设置为0.1以确保数据异构性

## [ ] Task 3: 创建公共验证数据集
- **Priority**: P0
- **Depends On**: Task 2
- **Description**: 
  - 从FashionMNIST测试集中划分出公共验证数据集
  - 确保验证数据集具有代表性，覆盖所有类别
  - 保存验证数据集供后续评估使用
- **Acceptance Criteria Addressed**: AC-1
- **Test Requirements**:
  - `programmatic` TR-3.1: 成功创建并保存公共验证数据集
  - `programmatic` TR-3.2: 验证验证数据集包含所有10个类别
- **Notes**: 可以使用FashionMNIST的官方测试集作为公共验证数据集

## [ ] Task 4: 配置FedProx训练参数
- **Priority**: P0
- **Depends On**: Task 3
- **Description**: 
  - 设置全局训练轮数为50
  - 配置FedProx的近端正则化参数μ
  - 设置本地训练轮数、批量大小等超参数
  - 配置模型架构（如CNN）
  - 配置使用GPU进行训练
- **Acceptance Criteria Addressed**: AC-2
- **Test Requirements**:
  - `programmatic` TR-4.1: 验证配置文件或命令行参数设置正确
  - `human-judgement` TR-4.2: 确认超参数设置合理
- **Notes**: 建议设置μ=0.01，本地训练轮数=3，批量大小=10

## [ ] Task 5: 使用现有FedProx代码运行训练
- **Priority**: P0
- **Depends On**: Task 4
- **Description**: 
  - 使用PFLlib框架中现有的FedProx实现
  - 执行训练命令，启动FedProx训练过程
  - 监控训练过程，确保无错误发生
  - 记录训练过程中的性能指标
- **Acceptance Criteria Addressed**: AC-2
- **Test Requirements**:
  - `programmatic` TR-5.1: 成功完成50轮全局训练
  - `programmatic` TR-5.2: 训练过程中无错误或崩溃
- **Notes**: 训练过程可能需要较长时间，取决于硬件性能

## [ ] Task 6: 评估模型在公共验证集上的性能
- **Priority**: P0
- **Depends On**: Task 5
- **Description**: 
  - 使用公共验证数据集评估训练后的全局模型
  - 生成包含训练过程和公共验证集上最终准确率的评估报告
  - 分析模型在non-iid数据下的收敛情况
- **Acceptance Criteria Addressed**: AC-3
- **Test Requirements**:
  - `programmatic` TR-6.1: 生成包含公共验证集上的准确率、损失等指标的评估报告
  - `programmatic` TR-6.2: 验证评估报告数据完整准确
- **Notes**: 评估报告应包含训练轮次与公共验证集性能指标的关系图，不需要评估客户端上的准确率

## [ ] Task 7: 分析FedProx在non-iid下的收敛原理
- **Priority**: P1
- **Depends On**: Task 6
- **Description**: 
  - 基于实验结果分析FedProx的收敛性能
  - 解释近端正则化项如何帮助模型在non-iid数据下收敛
  - 对比FedProx与其他联邦学习算法在non-iid场景下的表现
- **Acceptance Criteria Addressed**: AC-4
- **Test Requirements**:
  - `human-judgement` TR-7.1: 提供详细的算法收敛原理分析
  - `human-judgement` TR-7.2: 基于实验结果验证分析的正确性
- **Notes**: 重点分析近端正则化项如何限制客户端模型与全局模型的偏差