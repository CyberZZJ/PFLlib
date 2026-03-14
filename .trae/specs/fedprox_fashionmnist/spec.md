# FedProx FashionMNIST 联邦学习实验 - 产品需求文档

## Overview
- **Summary**: 实现一个联邦学习实验，使用FedProx算法在30个客户端的FashionMNIST非独立同分布(non-iid)数据集上进行训练，并使用公共验证数据集评估全局模型准确率。
- **Purpose**: 验证FedProx算法在非独立同分布数据下的收敛性能，特别是其近端正则化项如何帮助模型在客户端数据异构情况下保持稳定收敛。
- **Target Users**: 联邦学习研究人员和开发者，用于评估和理解FedProx算法的性能。

## Goals
- 生成30个客户端的FashionMNIST非独立同分布数据集
- 创建公共验证数据集用于评估全局模型准确率
- 使用FedProx算法进行50轮训练
- 评估并分析训练过程中的模型性能

## Non-Goals (Out of Scope)
- 不修改FedProx算法的核心实现
- 不添加新的联邦学习算法
- 不进行超参数调优实验
- 不使用真实分布式环境，仅在单机模拟环境中运行

## Background & Context
- 联邦学习是一种分布式机器学习范式，允许多个客户端在不共享原始数据的情况下协作训练模型
- 非独立同分布(non-iid)数据是联邦学习中的常见挑战，会导致模型收敛困难
- FedProx算法通过引入近端正则化项，帮助模型在non-iid数据下更好地收敛
- FashionMNIST是一个常用的图像分类数据集，包含10个类别的时尚物品图像

## Functional Requirements
- **FR-1**: 生成30个客户端的FashionMNIST非独立同分布数据集
- **FR-2**: 创建公共验证数据集用于评估全局模型准确率
- **FR-3**: 配置并运行FedProx算法进行50轮训练
- **FR-4**: 记录并分析训练过程中的模型性能指标

## Non-Functional Requirements
- **NFR-1**: 训练过程应稳定运行，无错误或崩溃
- **NFR-2**: 使用GPU进行训练以提高性能
- **NFR-3**: 训练时间应在合理范围内（取决于硬件性能）
- **NFR-4**: 结果应可重现，使用固定的随机种子

## Constraints
- **Technical**: 使用现有的PFLlib框架和代码，不修改核心实现，仅使用现有功能
- **Business**: 实验应在单机环境中完成，不使用分布式计算资源
- **Dependencies**: 依赖PFLlib框架中现有的FedProx实现和数据处理功能

## Assumptions
- PFLlib框架已正确安装并配置
- 系统具有足够的计算资源（CPU或GPU）来运行实验
- FashionMNIST数据集可以正常下载和处理

## Acceptance Criteria

### AC-1: 数据集生成
- **Given**: PFLlib环境已配置
- **When**: 运行数据生成脚本
- **Then**: 成功生成30个客户端的FashionMNIST非独立同分布数据集和公共验证数据集
- **Verification**: `programmatic`
- **Notes**: 数据集应保存在指定位置，便于后续训练使用

### AC-2: FedProx训练
- **Given**: 数据集已生成
- **When**: 配置并运行FedProx训练脚本
- **Then**: 成功完成50轮全局训练，无错误
- **Verification**: `programmatic`
- **Notes**: 训练过程应记录损失和准确率等指标

### AC-3: 模型评估
- **Given**: 训练完成
- **When**: 使用公共验证数据集评估模型
- **Then**: 生成包含训练过程和公共验证集上最终准确率的评估报告
- **Verification**: `programmatic`
- **Notes**: 报告应包含公共验证集上的准确率、损失等关键指标的变化趋势，不需要评估客户端上的准确率

### AC-4: 算法理解
- **Given**: 实验完成
- **When**: 分析实验结果
- **Then**: 能够解释FedProx如何在non-iid数据下实现收敛
- **Verification**: `human-judgment`
- **Notes**: 应基于实验结果和算法原理进行分析

## Open Questions
- [ ] 具体使用哪种non-iid数据分布策略？（如按标签分配、Dirichlet分布等）
- [ ] 是否需要调整FedProx的近端正则化参数μ？
- [ ] 如何设置本地训练轮数和批量大小等超参数？