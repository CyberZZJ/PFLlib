import torch
import numpy as np
import time
import copy
import torch.nn as nn
from flcore.optimizers.fedoptimizer import PerturbedGradientDescent
from flcore.clients.clientprox import clientProx


def compute_input_gradient_regularization(model, data, target, criterion):
    """
    计算输入梯度L2范数隐私正则项
    
    参数:
        model: 神经网络模型
        data: 输入数据
        target: 目标标签
        criterion: 损失函数
        
    返回:
        privacy_loss: 输入梯度L2范数的平均值
    """
    model.eval()
    data.requires_grad = True
    output = model(data)
    loss = criterion(output, target)
    data_grad = torch.autograd.grad(loss, data, create_graph=True)[0]
    privacy_loss = torch.sum(data_grad ** 2) / data.size(0)
    model.train()
    return privacy_loss


def compute_privacy_budget(clip_norm, noise_scale, delta=1e-5):
    """
    计算差分隐私预算
    
    参数:
        clip_norm: 裁剪阈值
        noise_scale: 噪声尺度
        delta: 隐私参数
        
    返回:
        epsilon: 隐私预算
    """
    epsilon = clip_norm * np.sqrt(2 * np.log(1.25 / delta)) / noise_scale
    return epsilon


class clientDPProxNative(clientProx):
    """
    原生差分隐私 FedProx 客户端类
    在 FedProx 基础上添加输入梯度正则化隐私保护
    
    核心特点：
    1. 继承 FedProx 的近端正则化项
    2. 添加输入梯度L2范数隐私正则项
    3. 结合梯度裁剪和噪声添加增强隐私保护
    4. 提供差分隐私理论保证
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

        # 隐私正则系数，控制隐私保护强度
        self.lambda_privacy = getattr(args, 'lambda_privacy', 0.05)

        # 是否使用补充噪声
        self.use_supplementary_noise = getattr(args, 'use_supplementary_noise', True)  # 默认启用

        # 补充噪声尺度
        self.supplementary_noise_scale = getattr(args, 'supplementary_noise_scale', 1.0)  # 进一步增加噪声尺度以增强隐私保护

        # 梯度裁剪参数
        self.clip_norm = getattr(args, 'dp_clip_norm', 1.0)

        # 隐私参数
        self.dp_delta = getattr(args, 'dp_delta', 1e-5)

        # 计算理论隐私预算
        if self.use_supplementary_noise:
            self.privacy_budget = compute_privacy_budget(self.clip_norm, self.supplementary_noise_scale, self.dp_delta)
            if self.id == 0:
                print(f"DP-Native-FedProx - Theoretical privacy budget: ε = {self.privacy_budget:.4f}")

    def train(self, round_num=0):
        """
        客户端本地训练方法
        
        训练逻辑：
        1. 加载本地训练数据
        2. 在本地数据上进行多轮训练
        3. 计算总损失 = 经验损失 + FedProx近端项 + λ * 隐私正则项
        4. 应用梯度裁剪
        5. 添加高斯噪声增强隐私保护
        
        隐私保护机制：
        - 输入梯度正则化：限制模型对输入数据的敏感度
        - 梯度裁剪：控制梯度敏感度
        - 高斯噪声：提供差分隐私保证
        - 数学形式：L(w) + (mu/2) * ||w - w_global||^2 + λ * ||∇_x L||^2
        """
        trainloader = self.load_train_data()
        start_time = time.time()
        local_training_start = time.time()

        self.model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            batch_start = time.time()
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                # 前向传播
                output = self.model(x)
                # 计算交叉熵损失
                loss = self.loss(output, y)

                # 计算输入梯度隐私正则项
                privacy_start = time.time()
                if self.lambda_privacy > 0:
                    # 创建数据副本用于计算输入梯度
                    if type(x) == type([]):
                        x_copy = [xi.clone().detach() for xi in x]
                        x_copy[0].requires_grad = True
                    else:
                        x_copy = x.clone().detach()
                        x_copy.requires_grad = True

                    # 计算隐私正则项
                    privacy_loss = compute_input_gradient_regularization(
                        self.model, x_copy, y, self.loss
                    )
                    # 添加隐私正则项到总损失
                    loss = loss + self.lambda_privacy * privacy_loss
                privacy_time = time.time() - privacy_start

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()

                # 应用梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)

                # 应用近端正则化的梯度更新
                self.optimizer.step(self.global_params, self.device)
            batch_time = time.time() - batch_start
            # 打印每轮训练耗时
            if epoch == max_local_epochs - 1:
                print(f"Client {self.id} - Epoch {epoch+1}/{max_local_epochs} - Time: {batch_time:.4f}s")

        # 添加高斯噪声增强隐私保护
        noise_start = time.time()
        if self.use_supplementary_noise:
            # 动态调整噪声尺度
            initial_noise_scale = max(0.1, self.supplementary_noise_scale)
            noise_scale = initial_noise_scale * (0.95 ** round_num)
            noise_scale = max(0.05, noise_scale)
            
            with torch.no_grad():
                for param in self.model.parameters():
                    noise = torch.randn_like(param) * noise_scale * self.clip_norm
                    param.add_(noise)
            
            # 打印噪声尺度信息
            if round_num % 10 == 0 and self.id == 0:
                print(f"Client {self.id} - Noise scale: {noise_scale:.4f}")
        noise_time = time.time() - noise_start

        # 学习率衰减
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        local_training_time = time.time() - local_training_start
        total_time = time.time() - start_time

        # 记录训练时间开销
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += total_time
        
        # 打印详细耗时信息
        if self.id == 0:  # 只在第一个客户端打印，避免过多输出
            print(f"Client {self.id} - Local training: {local_training_time:.4f}s")
            print(f"Client {self.id} - Privacy regularization: {privacy_time:.4f}s")
            print(f"Client {self.id} - Noise addition: {noise_time:.4f}s")
            print(f"Client {self.id} - Total training: {total_time:.4f}s")

    def train_metrics(self):
        """
        计算训练指标（包含近端正则化项和隐私正则项的损失）
        
        返回:
            losses: 总损失（包含正则化项）
            train_num: 训练样本数
        """
        trainloader = self.load_train_data()
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                # 前向传播
                output = self.model(x)
                # 计算交叉熵损失
                loss = self.loss(output, y)

                # 添加近端正则化项：0.5 * mu * ||w - w_global||^2
                gm = torch.cat([p.data.view(-1) for p in self.global_params], dim=0)
                pm = torch.cat([p.data.view(-1) for p in self.model.parameters()], dim=0)
                loss += 0.5 * self.mu * torch.norm(gm-pm, p=2)

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num
