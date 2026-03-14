import torch
from torch.optim import Optimizer


class PerAvgOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(PerAvgOptimizer, self).__init__(params, defaults)

    def step(self, beta=0):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if(beta != 0):
                    p.data.add_(other=d_p, alpha=-beta)
                else:
                    p.data.add_(other=d_p, alpha=-group['lr'])


class SCAFFOLDOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(SCAFFOLDOptimizer, self).__init__(params, defaults)

    def step(self, server_cs, client_cs):
        for group in self.param_groups:
            for p, sc, cc in zip(group['params'], server_cs, client_cs):
                p.data.add_(other=(p.grad.data + sc - cc), alpha=-group['lr'])


class pFedMeOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1, mu=0.001):
        defaults = dict(lr=lr, lamda=lamda, mu=mu)
        super(pFedMeOptimizer, self).__init__(params, defaults)

    def step(self, local_model, device):
        group = None
        weight_update = local_model.copy()
        for group in self.param_groups:
            for p, localweight in zip(group['params'], weight_update):
                localweight = localweight.to(device)
                # approximate local model
                p.data = p.data - group['lr'] * (p.grad.data + group['lamda'] * (p.data - localweight.data) + group['mu'] * p.data)

        return group['params']


class APFLOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(APFLOptimizer, self).__init__(params, defaults)

    def step(self, beta=1, n_k=1):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = beta * n_k * p.grad.data
                p.data.add_(-group['lr'], d_p)


class PerturbedGradientDescent(Optimizer):
    """
    扰动梯度下降优化器（FedProx 核心优化器）
    
    在标准梯度下降的基础上添加了近端正则化项的梯度
    用于 FedProx 算法，处理 non-iid 数据下的收敛问题
    
    更新规则：
    w = w - lr * (grad_loss + mu * (w - w_global))
    
    其中：
    - grad_loss: 原始损失函数的梯度
    - mu: 近端系数，控制正则化强度
    - (w - w_global): 当前参数与全局参数的偏差
    """
    def __init__(self, params, lr=0.01, mu=0.0):
        """
        初始化优化器
        
        参数:
            params: 模型参数
            lr: 学习率
            mu: 近端系数，默认为 0（退化为标准 SGD）
        """
        default = dict(lr=lr, mu=mu)
        super().__init__(params, default)

    @torch.no_grad()
    def step(self, global_params, device):
        """
        执行一步参数更新
        
        参数:
            global_params: 全局模型参数（用于计算近端项）
            device: 计算设备
        """
        for group in self.param_groups:
            # 遍历所有参数张量
            for p, g in zip(group['params'], global_params):
                g = g.to(device)
                
                # 计算扰动梯度：grad_loss + mu * (w - w_global)
                # 第一项是原始损失梯度，第二项是近端正则化梯度
                # 近端项的作用是将参数拉向全局参数，防止偏离太远
                d_p = p.grad.data + group['mu'] * (p.data - g.data)
                
                # 应用梯度更新：w = w - lr * d_p
                p.data.add_(d_p, alpha=-group['lr'])
