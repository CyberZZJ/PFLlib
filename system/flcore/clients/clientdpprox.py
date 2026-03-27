import torch
import numpy as np
import time
import copy
from flcore.clients.clientprox import clientProx


def clip_gradient(update, clip_norm):
    """
    梯度裁剪：L2范数裁剪
    
    参数:
        update: 模型更新（张量）
        clip_norm: 裁剪阈值
    
    返回:
        裁剪后的更新
    """
    update_norm = torch.norm(update)
    if update_norm > clip_norm and update_norm > 1e-8:
        update = update * (clip_norm / update_norm)
    return update


def add_gaussian_noise(update, noise_scale, clip_norm):
    """
    高斯噪声添加
    
    参数:
        update: 模型更新（张量）
        noise_scale: 噪声乘数
        clip_norm: 裁剪阈值
    
    返回:
        添加噪声后的更新
    """
    noise = torch.randn_like(update) * noise_scale * clip_norm
    return update + noise


class clientDPProx(clientProx):
    """
    后处理差分隐私 FedProx 客户端类
    在 FedProx 基础上添加差分隐私保护
    核心特点：训练完成后对模型更新进行梯度裁剪和噪声添加
    """
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        """
        初始化差分隐私 FedProx 客户端
        
        参数:
            args: 配置参数
            id: 客户端 ID
            train_samples: 训练样本数量
            test_samples: 测试样本数量
        """
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # 差分隐私参数
        self.dp_clip_norm = getattr(args, 'dp_clip_norm', 1.0)
        self.dp_noise_multiplier = getattr(args, 'dp_noise_multiplier', 1.0)
        self.dp_epsilon = getattr(args, 'dp_epsilon', 10.0)
        self.dp_delta = getattr(args, 'dp_delta', 1e-5)

    def train(self, round_num=0):
        """
        客户端本地训练方法（带差分隐私保护）
        
        训练逻辑：
        1. 执行标准的 FedProx 本地训练
        2. 计算模型更新：model_update = current_params - global_params
        3. 对更新进行 L2 范数裁剪
        4. 添加高斯噪声
        5. 将带噪更新还原为模型参数
        
        参数:
            round_num: 当前全局轮数，用于动态调整噪声尺度
        """
        # 加载本地训练数据
        trainloader = self.load_train_data()
        start_time = time.time()
        local_training_start = time.time()

        # 设置模型为训练模式
        self.model.train()

        # 设置本地训练轮数
        max_local_epochs = self.local_epochs
        # 如果是慢速客户端，随机减少训练轮数（模拟设备异构性）
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        # 本地多轮训练
        for epoch in range(max_local_epochs):
            for x, y in trainloader:
                # 数据预处理
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                # 模拟慢速客户端延迟
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                
                # 前向传播
                output = self.model(x)
                # 计算交叉熵损失
                loss = self.loss(output, y)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 应用近端正则化的梯度更新
                self.optimizer.step(self.global_params, self.device)

        # 学习率衰减
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        # 差分隐私后处理
        self._apply_differential_privacy(round_num)

        # 确保模型参数没有 NaN
        with torch.no_grad():
            for param in self.model.parameters():
                param.data = torch.nan_to_num(param.data, nan=0.0, posinf=1.0, neginf=-1.0)

        # 记录训练时间开销
        total_time = time.time() - start_time
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += total_time

    def _apply_differential_privacy(self, round_num):
        """
        应用差分隐私保护
        
        步骤：
        1. 计算模型更新（当前参数 - 全局参数）
        2. 将所有参数展平为一个向量
        3. 对更新进行 L2 范数裁剪
        4. 添加高斯噪声
        5. 将带噪更新还原为模型参数
        
        参数:
            round_num: 当前全局轮数，用于动态调整噪声尺度
        """
        # 计算模型更新并展平
        update_list = []
        for param, global_param in zip(self.model.parameters(), self.global_params):
            update = param.data - global_param.data
            update_list.append(update.view(-1))
        
        # 将所有更新拼接为一个向量
        update_flat = torch.cat(update_list, dim=0)
        
        # 动态调整噪声尺度
        # 初期使用较小噪声，后期逐渐减小
        initial_noise_scale = max(0.1, self.dp_noise_multiplier)  # 保持初始噪声尺度适中
        noise_scale = initial_noise_scale * (0.95 ** round_num)  # 每轮衰减 5%
        noise_scale = max(0.05, noise_scale)  # 确保噪声尺度不会太小
        
        # 梯度裁剪
        update_flat = clip_gradient(update_flat, self.dp_clip_norm)
        
        # 添加高斯噪声
        update_flat = add_gaussian_noise(update_flat, noise_scale, self.dp_clip_norm)
        
        # 确保没有 NaN 值
        update_flat = torch.nan_to_num(update_flat, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 将带噪更新还原为模型参数
        offset = 0
        for param, global_param in zip(self.model.parameters(), self.global_params):
            param_shape = param.data.shape
            param_numel = param.data.numel()
            # 从展平的更新中提取对应部分
            noisy_update = update_flat[offset:offset + param_numel].view(param_shape)
            # 确保噪声更新没有 NaN
            noisy_update = torch.nan_to_num(noisy_update, nan=0.0, posinf=1.0, neginf=-1.0)
            # 更新模型参数：global_params + noisy_update
            param.data = global_param.data.clone() + noisy_update
            # 确保参数没有 NaN
            param.data = torch.nan_to_num(param.data, nan=0.0, posinf=1.0, neginf=-1.0)
            offset += param_numel
        
        # 打印噪声尺度信息（每10轮打印一次）
        if round_num % 10 == 0 and self.id == 0:
            print(f"Client {self.id} - Noise scale: {noise_scale:.4f}")
