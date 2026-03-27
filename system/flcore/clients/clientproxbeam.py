import torch
import numpy as np
import time
import copy
import torch.nn as nn
from flcore.optimizers.fedoptimizer import PerturbedGradientDescent
from flcore.clients.clientprox import clientProx
from utils.compression import compress_model_for_communication


from flcore.clients.clientbeam import clientBeam


class clientProxBeam(clientProx):
    """
    FedProx-Beam 客户端类
    继承自 clientProx，结合 FedProx 的近端正则化、 Beam 的多模型训练机制和 TOPK 压缩
    核心特点：
    - 保留 FedProx 的 proximal term 功能，限制本地模型与全局模型的偏差
    - 支持多模型并行训练（强客户端训练多个模型，弱客户端训练单个模型）
    - 支持 TOPK 压缩以减少通信开销
    """
    
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        """
        初始化 FedProx-Beam 客户端
        
        参数:
            args: 配置参数
            id: 客户端 ID
            train_samples: 训练样本数量
            test_samples: 测试样本数量
        """
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # num_models: 控制客户端类型
        # 默认为 1（弱客户端），强客户端会被设为 2
        # 弱客户端只接收和训练 1 个模型
        # 强客户端接收和训练 2 个模型
        self.num_models = 1
        
        # received_models: 存储从服务器接收的全局模型列表
        # 强客户端会收到 2 个不同的全局模型
        # 弱客户端只收到 1 个全局模型
        self.received_models = []
        
        # trained_models: 存储本地训练后的模型列表
        # 训练完成后，每个接收的模型都会产生一个对应的训练后模型
        # 强客户端会有 2 个训练后的模型
        # 弱客户端只有 1 个训练后的模型
        self.trained_models = []
        
        # mus: 存储每个接收模型对应的 μ 值列表
        # μ 值控制近端正则化的强度
        # 不同的接收模型可能有不同的 μ 值
        self.mus = []
        
        # topk_ratios: 存储每个模型的 TOPK 压缩比率列表
        # 用于控制模型参数的压缩程度
        self.topk_ratios = []
        
        # 定义交叉熵损失函数（继承自 clientProx）
        self.loss = nn.CrossEntropyLoss()

    def set_parameters(self, models, mus, topk_ratios=None):
        """
        从服务器接收新的全局模型参数、 μ 值和 TOPK 压缩比率
        
        参数:
            models: 服务器发送的全局模型列表
                   - 强客户端收到 2 个模型
                   - 弱客户端收到 1 个模型
            mus: 与模型列表对应的 μ 值列表
                每个 μ 值用于控制对应模型的近端正则化强度
            topk_ratios: 与模型列表对应的 TOPK 压缩比率列表
                        每个 ratio 控制对应模型的压缩程度
        
        方法说明:
        1. 保存接收的模型列表到 self.received_models
        2. 保存 μ 值列表到 self.mus
        3. 保存 TOPK 压缩比率列表到 self.topk_ratios
        4. 使用第一个模型初始化当前模型（self.model）
        5. 同时更新全局参数备份（self.global_params）用于近端正则化计算
        """
        # 保存接收的模型列表
        if isinstance(models, list):
            self.received_models = models
        else:
            self.received_models = [models]
        
        # 保存对应的 μ 值列表
        self.mus = mus if isinstance(mus, list) else [mus]
        
        # 保存对应的 TOPK 压缩比率列表
        if topk_ratios is None:
            self.topk_ratios = [1.0] * len(self.received_models)  # 默认不压缩
        else:
            self.topk_ratios = topk_ratios if isinstance(topk_ratios, list) else [topk_ratios]
        
        # 使用第一个模型初始化当前模型和全局参数备份
        if len(self.received_models) > 0:
            first_model = self.received_models[0]
            
            # 同时更新模型参数和全局参数备份
            for new_param, global_param, param in zip(
                first_model.parameters(), 
                self.global_params, 
                self.model.parameters()
            ):
                # 更新全局参数备份（用于近端正则化计算）
                global_param.data = new_param.data.clone()
                # 更新模型参数
                param.data = new_param.data.clone()

    def train(self):
        """
        客户端本地训练方法
        
        训练逻辑：
        1. 对每个接收的模型，使用对应的 μ 值进行本地训练
        2. 训练逻辑与 clientProx 相同（使用 PerturbedGradientDescent 优化器）
        3. 每次训练前需要更新 self.mu 和 self.global_params
        4. 保存所有训练后的模型到 self.trained_models
        
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
        
        # 清空之前训练的模型列表
        self.trained_models = []
        
        # 设置本地训练轮数
        max_local_epochs = self.local_epochs
        # 如果是慢速客户端，随机减少训练轮数（模拟设备异构性）
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)
        
        # 对每个接收的模型进行训练
        for idx, initial_model in enumerate(self.received_models):
            # 加载当前要训练的模型状态
            self.model.load_state_dict(initial_model.state_dict())
            
            # 更新全局参数备份为当前模型的全局参数
            for global_param, init_param in zip(self.global_params, initial_model.parameters()):
                global_param.data = init_param.data.clone()
            
            # 更新当前模型的 μ 值（如果提供了多个 μ 值）
            if idx < len(self.mus):
                self.mu = self.mus[idx]
                # 重新初始化优化器以使用新的 μ 值
                self.optimizer = PerturbedGradientDescent(
                    self.model.parameters(), lr=self.learning_rate, mu=self.mu)
            
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
            
            # 保存训练后的模型（深拷贝）
            self.trained_models.append(copy.deepcopy(self.model))
        
        # 记录训练时间开销
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def compress_model(self, model, ratio):
        """
        使用 TOPK 压缩模型参数
        
        参数:
            model: 要压缩的模型
            ratio: 压缩比率（保留前 ratio 比例的参数）
        
        返回:
            compressed_data: 压缩后的模型数据字典
        """
        try:
            compressed_data = compress_model_for_communication(
                model,
                compression_ratio=ratio
            )
            # 添加元数据
            compressed_data['_is_compressed'] = True
            compressed_data['_client_id'] = self.id
            compressed_data['_compression_ratio'] = ratio
            return compressed_data
        except Exception as e:
            print(f"客户端 {self.id} 压缩失败: {e}")
            return copy.deepcopy(model)
    
    def get_upload_model(self):
        """
        获取要上传到服务器的模型（支持 TOPK 压缩）
        
        返回:
            compressed_models: 压缩后的模型列表
                             - 强客户端返回 2 个压缩后的模型
                             - 弱客户端返回 1 个压缩后的模型
        
        方法说明:
        - 对每个训练后的模型使用对应的 TOPK 压缩比率进行压缩
        - 压缩比率从 self.topk_ratios 获取
        - 如果压缩比率为 1.0，则不压缩（返回完整模型）
        - 服务器会根据客户端类型（强/弱）接收相应数量的模型
        """
        compressed_models = []
        
        # 对每个训练后的模型进行压缩
        for idx, trained_model in enumerate(self.trained_models):
            if idx < len(self.topk_ratios):
                ratio = self.topk_ratios[idx]
            else:
                ratio = 1.0  # 默认不压缩
            
            # 如果压缩比率为 1.0，则不压缩
            if ratio >= 1.0:
                compressed_models.append(trained_model)
            else:
                # 使用 TOPK 压缩
                compressed_data = self.compress_model(trained_model, ratio)
                compressed_models.append(compressed_data)
        
        return compressed_models
