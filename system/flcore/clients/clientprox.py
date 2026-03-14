import torch
import numpy as np
import time
import copy
import torch.nn as nn
from flcore.optimizers.fedoptimizer import PerturbedGradientDescent
from flcore.clients.clientbase import Client


class clientProx(Client):
    """
    FedProx 客户端类
    实现 FedProx 算法的客户端本地训练逻辑
    核心特点：使用近端正则化项限制本地模型与全局模型的偏差
    """
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        """
        初始化 FedProx 客户端
        
        参数:
            args: 配置参数
            id: 客户端 ID
            train_samples: 训练样本数量
            test_samples: 测试样本数量
        """
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # 保存近端系数 mu，用于控制正则化强度
        self.mu = args.mu

        # 保存全局模型参数的深拷贝
        # 这些参数在每次接收全局模型时更新
        # 用于计算近端正则化项：||w - w_global||^2
        self.global_params = copy.deepcopy(list(self.model.parameters()))

        # 定义交叉熵损失函数
        self.loss = nn.CrossEntropyLoss()
        
        # 使用 PerturbedGradientDescent 优化器
        # 该优化器在标准 SGD 基础上添加了近端正则化梯度
        self.optimizer = PerturbedGradientDescent(
            self.model.parameters(), lr=self.learning_rate, mu=self.mu)
        
        # 学习率衰减调度器
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )

    def train(self):
        """
        客户端本地训练方法
        
        训练逻辑：
        1. 加载本地训练数据
        2. 在本地数据上进行多轮训练
        3. 使用 PerturbedGradientDescent 优化器
        4. 每次更新时应用近端正则化项
        
        近端正则化的作用：
        - 限制本地模型参数不要偏离全局模型太远
        - 在 non-iid 数据下提高收敛稳定性
        - 数学形式：L(w) + (mu/2) * ||w - w_global||^2
        """
        # 加载本地训练数据
        trainloader = self.load_train_data()
        start_time = time.time()

        # 设置模型为训练模式
        self.model.train()

        # 设置本地训练轮数
        max_local_epochs = self.local_epochs
        # 如果是慢速客户端，随机减少训练轮数（模拟设备异构性）
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        # 本地多轮训练
        for epoch in range(max_local_epochs):
            # 遍历所有批次数据
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
                # optimizer.step 会同时考虑损失梯度和近端项梯度
                self.optimizer.step(self.global_params, self.device)

        # 学习率衰减
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        # 记录训练时间开销
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def set_parameters(self, model):
        """
        从服务器接收新的全局模型参数
        
        参数:
            model: 服务器发送的全局模型
        """
        # 同时更新模型参数和全局参数备份
        for new_param, global_param, param in zip(model.parameters(), self.global_params, self.model.parameters()):
            # 更新全局参数备份（用于近端正则化计算）
            global_param.data = new_param.data.clone()
            # 更新模型参数
            param.data = new_param.data.clone()

    def train_metrics(self):
        """
        计算训练指标（包含近端正则化项的损失）
        
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
                # 数据预处理
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
