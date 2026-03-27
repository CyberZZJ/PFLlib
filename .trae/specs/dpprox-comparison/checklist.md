# DP-Prox vs DP-Prox-Native 对比实验 - 验证检查清单

## 实现阶段

### DP-Prox 客户端实现

- [x] **文件创建**: clientdpprox.py 文件已创建
- [x] **类继承**: 正确继承 clientProx 类
- [x] **差分隐私参数**: 添加 dp_enabled, dp_clip_norm, dp_noise_multiplier, dp_epsilon, dp_delta 参数
- [x] **梯度裁剪**: 实现 `clip_gradient()` 方法，按范数裁剪梯度
- [x] **高斯噪声**: 实现 `add_gaussian_noise()` 方法，添加校准噪声
- [x] **训练循环**: 在反向传播后调用 `_apply_differential_privacy()` 方法
- [x] **隐私参数关系**: 满足 ε ≥ 2ln(1.25/δ) · C²/σ²

### DP-Prox 服务器实现

- [x] **文件创建**: serverdpprox.py 文件已创建
- [x] **类继承**: 正确继承 FedProx 类
- [x] **参数传递**: 差分隐私参数正确传递给客户端
- [x] **初始化**: 服务器能正确初始化 DP-Prox 客户端

### DP-Prox-Native 客户端实现

- [x] **文件创建**: clientdpproxnative.py 文件已创建
- [x] **类继承**: 正确继承 clientProx 类
- [x] **隐私正则项参数**: 添加 lambda_privacy, use_supplementary_noise, supplementary_noise_scale 参数
- [x] **隐私正则项计算**: 实现 `compute_input_gradient_regularization()` 方法
- [x] **优化目标修改**: 损失函数包含隐私正则项 λ · R_privacy(w)
- [x] **反向传播**: 隐私正则项参与梯度计算
- [x] **补充噪声**: 可选的补充噪声功能（默认关闭）

### DP-Prox-Native 服务器实现

- [x] **文件创建**: serverdpproxnative.py 文件已创建
- [x] **类继承**: 正确继承 FedProx 类
- [x] **参数传递**: 隐私正则项参数正确传递给客户端
- [x] **初始化**: 服务器能正确初始化 DP-Prox-Native 客户端

### main.py 修改

- [x] **算法路由**: 添加 DP-Prox 和 DP-Prox-Native 算法路由
- [x] **DP-Prox 参数**: 添加 -dp_clip, -dp_noise, -dp_eps, -dp_delta 参数
- [x] **DP-Prox-Native 参数**: 添加 -lambda_priv, -sup_noise, -sup_noise_scale 参数
- [x] **参数传递**: 所有参数正确传递给服务器和客户端

## 实验执行阶段

### DP-Prox 基准线实验

- [ ] **实验启动**: 使用正确命令启动 DP-Prox 训练
  ```bash
  python main.py --algorithm DPProx --dataset FashionMNIST --num_clients 30 \
                 --global_rounds 30 --local_epochs 3 --local_learning_rate 0.01 \
                 --mu 0.01 --batch_size 10 --device cuda \
                 --dp_enabled --dp_clip_norm 1.0 --dp_noise_multiplier 1.0 \
                 --dp_epsilon 10.0 --dp_delta 1e-5
  ```
- [ ] **训练完成**: 成功完成 30 轮全局训练，无错误或崩溃
- [ ] **梯度裁剪**: 日志显示梯度裁剪正常工作
- [ ] **噪声添加**: 日志显示噪声添加正常工作
- [ ] **数据完整性**: 准确率、时间、资源使用数据完整记录
- [ ] **日志保存**: 训练日志和结果文件妥善保存

### DP-Prox-Native 对比实验

- [ ] **实验启动**: 使用正确命令启动 DP-Prox-Native 训练
  ```bash
  python main.py --algorithm DPProxNative --dataset FashionMNIST --num_clients 30 \
                 --global_rounds 30 --local_epochs 3 --local_learning_rate 0.01 \
                 --mu 0.01 --batch_size 10 --device cuda \
                 --lambda_privacy 0.05
  ```
- [ ] **训练完成**: 成功完成 30 轮全局训练，无错误或崩溃
- [ ] **隐私正则项**: 日志显示隐私正则项正常工作
- [ ] **数据完整性**: 准确率、时间、资源使用数据完整记录
- [ ] **日志保存**: 训练日志和结果文件妥善保存

### 实验配置一致性

- [ ] **数据集相同**: 两个实验使用相同的 FashionMNIST 数据集
- [ ] **客户端数量相同**: 两个实验都是 30 个客户端
- [ ] **训练轮数相同**: 两个实验都是 30 轮全局训练
- [ ] **本地轮数相同**: 两个实验都是 3 轮本地训练
- [ ] **学习率相同**: 两个实验都是 0.01
- [ ] **近端系数相同**: 两个实验都是 μ=0.01
- [ ] **批量大小相同**: 两个实验都是 10
- [ ] **设备相同**: 两个实验都使用 GPU (CUDA)

## 数据收集和分析阶段

### 数据提取

- [ ] **准确率数据**: 从两个实验结果文件中提取每轮准确率
- [ ] **最佳准确率**: 提取最佳准确率和对应轮次
- [ ] **最终准确率**: 提取最终准确率
- [ ] **AUC 分数**: 提取 AUC 分数
- [ ] **训练时间**: 提取总训练时间和平均每轮时间
- [ ] **收敛速度**: 计算达到 70% 和 80% 准确率所需轮次

### DP-Prox 特有数据

- [ ] **隐私预算**: 记录隐私预算消耗
- [ ] **噪声添加量**: 记录噪声添加量
- [ ] **梯度裁剪统计**: 记录梯度裁剪统计信息

### DP-Prox-Native 特有数据

- [ ] **隐私正则项值**: 记录隐私正则项值变化
- [ ] **梯度范数**: 记录梯度范数变化
- [ ] **补充噪声**: 如果启用，记录补充噪声信息

## 对比分析阶段

### 准确率对比

- [ ] **准确率曲线图**: 绘制 DP-Prox vs DP-Prox-Native 的 30 轮准确率曲线
- [ ] **最佳准确率对比**: 对比两种方案的最佳准确率
- [ ] **最终准确率对比**: 对比两种方案的最终准确率
- [ ] **AUC 对比**: 对比两种方案的 AUC 分数

### 收敛速度对比

- [ ] **达到 70% 轮次**: 对比达到 70% 准确率所需轮次
- [ ] **达到 80% 轮次**: 对比达到 80% 准确率所需轮次
- [ ] **收敛曲线**: 对比收敛曲线的平滑度

### 效率对比

- [ ] **总训练时间**: 对比总训练时间
- [ ] **平均每轮时间**: 对比平均每轮时间
- [ ] **资源使用**: 对比 GPU 内存使用

### 隐私保护强度对比

- [ ] **DP-Prox 隐私保证**: 分析 DP-Prox 的 (ε, δ)-DP 保证
- [ ] **DP-Prox-Native 隐私保证**: 分析 DP-Prox-Native 的隐私保护强度
- [ ] **隐私-效用权衡**: 对比两种方案的隐私-效用权衡

## 结论验证阶段

### 性能提升验证

- [ ] **准确率提升**: DP-Prox-Native 是否有更高的准确率？
- [ ] **收敛加速**: DP-Prox-Native 是否有更快的收敛速度？
- [ ] **稳定性提升**: DP-Prox-Native 是否有更平滑的收敛曲线？

### 隐私保护验证

- [ ] **隐私强度**: DP-Prox-Native 是否提供足够的隐私保护？
- [ ] **隐私-效用权衡**: DP-Prox-Native 是否有更好的隐私-效用权衡？
- [ ] **理论保证**: DP-Prox-Native 是否有理论隐私保证？

### 综合评价

- [ ] **优势总结**: 总结 DP-Prox-Native 相对于 DP-Prox 的优势
- [ ] **劣势总结**: 总结 DP-Prox-Native 相对于 DP-Prox 的劣势
- [ ] **适用场景**: 分析两种方案的适用场景
- [ ] **改进建议**: 提出算法改进建议

## 最终交付物

- [ ] **实验数据**: 完整的实验结果文件（H5 格式）
- [ ] **训练日志**: 详细的训练过程日志
- [ ] **对比报告**: 包含图表和分析的完整对比报告
- [ ] **结论总结**: 清晰明确的结论和建议
