import time
from flcore.clients.clientprox import clientProx
from flcore.servers.serverbase import Server
from threading import Thread


class FedProx(Server):
    """
    FedProx 联邦学习服务器类
    实现 FedProx 算法的服务器端逻辑，负责协调客户端训练和模型聚合
    """
    def __init__(self, args, times):
        """
        初始化 FedProx 服务器
        
        参数:
            args: 配置参数，包含全局轮数、评估间隔、客户端数量等
            times: 运行次数标识
        """
        super().__init__(args, times)

        # 设置慢速客户端（用于模拟异构设备）
        self.set_slow_clients()
        # 创建客户端实例，使用 clientProx 类
        self.set_clients(clientProx)

        # 打印客户端参与信息
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # 初始化训练时间记录列表
        self.Budget = []


    def train(self):
        """
        FedProx 主训练流程
        
        训练逻辑：
        1. 每轮选择一部分客户端参与训练
        2. 发送全局模型参数到选中的客户端
        3. 客户端在本地数据上进行训练（使用近端正则化）
        4. 接收客户端上传的模型参数
        5. 聚合所有客户端模型，更新全局模型
        6. 在公共验证集上评估全局模型性能
        """
        # 遍历所有全局训练轮次
        for i in range(self.global_rounds+1):
            s_t = time.time()
            
            # 步骤 1: 选择本轮参与训练的客户端
            # 根据 join_ratio 随机选择一定比例的客户端
            self.selected_clients = self.select_clients()
            
            # 步骤 2: 发送全局模型到所有客户端（不仅仅是选中的）
            # 每个客户端都会接收最新的全局模型参数
            self.send_models()

            # 步骤 3: 按评估间隔评估全局模型性能
            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                # 在公共验证集上评估全局模型
                self.evaluate()

            # 步骤 4: 选中的客户端在本地进行训练
            # 每个客户端使用自己的本地数据，并应用近端正则化项
            for client in self.selected_clients:
                client.train()

            # 可选的并行训练方式（当前使用串行）
            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            # 步骤 5: 接收客户端上传的模型参数
            self.receive_models()
            
            # 可选的隐私攻击评估（DLG 攻击）
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            
            # 步骤 6: 聚合所有客户端的模型参数
            # 使用加权平均，权重为各客户端的训练样本数
            self.aggregate_parameters()

            # 记录本轮训练时间
            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            # 检查是否满足自动停止条件
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        # 训练结束，输出最佳性能
        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        # 保存训练结果和全局模型
        self.save_results()
        self.save_global_model()

        # 如果有新客户端，进行微调评估
        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientProx)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
