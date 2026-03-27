import time
import inspect
from flcore.clients.clientdpprox import clientDPProx
from flcore.servers.serverprox import FedProx


class DPProx(FedProx):
    """
    差分隐私 FedProx 联邦学习服务器类
    继承自 FedProx，使用差分隐私保护的客户端进行训练
    差分隐私处理在客户端完成，服务器端无需额外处理
    支持动态参数更新和单轮训练模式
    """
    def __init__(self, args, times):
        """
        初始化 DP-Prox 服务器
        
        参数:
            args: 配置参数，包含全局轮数、评估间隔、客户端数量等
            times: 运行次数标识
        """
        super().__init__(args, times)
        self.current_round = 0

        self.set_slow_clients()
        self.set_clients(clientDPProx)

        print(f"\nDP-Prox Server initialized with {len(self.clients)} clients")
    
    def update_params(self, params: dict):
        """
        动态更新差分隐私参数
        
        参数:
            params: 包含新参数的字典，可包含:
                - mu: 近端项系数
                - clip_norm: 梯度裁剪阈值
                - noise_multiplier: 噪声乘数
                - epsilon: 隐私预算
                - delta: 隐私参数
        """
        if 'mu' in params:
            self.args.mu = params['mu']
            print(f"[参数更新] mu -> {params['mu']}")
        
        if 'clip_norm' in params:
            self.args.dp_clip_norm = params['clip_norm']
            print(f"[参数更新] clip_norm -> {params['clip_norm']}")
        
        if 'noise_multiplier' in params:
            self.args.dp_noise_multiplier = params['noise_multiplier']
            print(f"[参数更新] noise_multiplier -> {params['noise_multiplier']}")
        
        if 'epsilon' in params:
            self.args.dp_epsilon = params['epsilon']
            print(f"[参数更新] epsilon -> {params['epsilon']}")
        
        if 'delta' in params:
            self.args.dp_delta = params['delta']
            print(f"[参数更新] delta -> {params['delta']}")
        
        for client in self.clients:
            if hasattr(client, 'update_dp_params'):
                client.update_dp_params(params)
    
    def train_one_round(self):
        """
        执行单轮训练，用于自适应调参系统
        """
        s_t = time.time()
        self.selected_clients = self.select_clients()
        self.send_models()
        
        for client in self.selected_clients:
            sig = inspect.signature(client.train)
            if 'round_num' in sig.parameters:
                client.train(round_num=self.current_round)
            else:
                client.train()
        
        self.receive_models()
        self.aggregate_parameters()
        
        self.Budget.append(time.time() - s_t)
        self.current_round += 1
    
    def train(self):
        """
        联邦学习训练方法，重写父类方法以传递轮数参数
        """
        for round in range(self.global_rounds):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            
            for client in self.selected_clients:
                sig = inspect.signature(client.train)
                if 'round_num' in sig.parameters:
                    client.train(round_num=round)
                else:
                    client.train()
            
            self.receive_models()
            self.aggregate_parameters()
            
            if (round + 1) % self.eval_gap == 0:
                self.evaluate()
            
            self.Budget.append(time.time() - s_t)
            print('------------------------- time cost -------------------------', self.Budget[-1])
        
        self.evaluate()
        print('Total time cost:', sum(self.Budget))
