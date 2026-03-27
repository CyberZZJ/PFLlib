import time
import numpy as np
import random
import copy
import torch
import os
import h5py
from flcore.clients.clientproxbeam import clientProxBeam
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data
from sklearn.metrics import accuracy_score


class FedProxBeam(Server):
    """
    FedProx-Beam 联邦学习服务器类
    
    结合 FedProx 和 Beam Search 的混合算法：
    - FedProx: 通过近端正则化项限制本地模型与全局模型的偏差，提高 non-iid 数据下的收敛稳定性
    - Beam Search: 维护多个候选模型分支，通过评估选择最优的模型组合
    
    核心机制：
    1. 客户端异质性：按训练数据量排序，前 30% 为强客户端（num_models=2），其余为弱客户端（num_models=1）
    2. 双模型维护：服务器维护两个全局模型及其对应的 μ 值
    3. Beam 搜索：每轮构建 4 个候选分支，评估后选择 Top-2 作为下一轮的父模型
    4. 动态 μ 调整：根据模型性能趋势动态调整近端正则化强度
    """
    
    def __init__(self, args, times):
        """
        初始化 FedProx-Beam 服务器
        
        参数:
            args: 配置参数，包含全局轮数、评估间隔、客户端数量、μ 值等
            times: 运行次数标识
        
        初始化内容:
        1. 调用父类 Server 的初始化方法
        2. 设置慢速客户端（模拟设备异构性）
        3. 创建客户端实例（使用 clientProxBeam 类）
        4. 按训练数据量排序客户端，前 30% 标记为强客户端
        5. 初始化两个全局模型及其对应的 μ 值
        6. 初始化 Beam 搜索相关的得分和准确率
        7. 设置动态 μ 调整的参数
        """
        # 调用父类初始化
        super().__init__(args, times)
        
        # 设置慢速客户端（用于模拟设备异构性）
        self.set_slow_clients()
        
        # 创建客户端实例，使用 clientProxBeam 类
        # clientProxBeam 支持接收多个模型和多个 μ 值
        self.set_clients(clientProxBeam)
        
        # 客户端异质性分类：按训练数据量排序
        # 训练数据量多的客户端计算能力强，标记为强客户端
        sorted_clients = sorted(self.clients, key=lambda c: c.train_samples, reverse=True)
        num_strong = int(len(self.clients) * 0.3)  # 前 30% 为强客户端
        strong_ids = set(c.id for c in sorted_clients[:num_strong])
        
        # 初始化客户端贡献度字典（用于加权平均）
        self.contributions = {c.id: 1.0 / len(self.clients) for c in self.clients}
        
        # 标记客户端类型：强客户端 num_models=2，弱客户端 num_models=1
        for c in self.clients:
            if c.id in strong_ids:
                c.num_models = 2  # 强客户端：接收和训练 2 个模型
            else:
                c.num_models = 1  # 弱客户端：只接收和训练 1 个模型
        
        # 打印初始化信息
        print(f"\nFedProx-Beam 初始化完成:")
        print(f"  强客户端 (30%): {len(strong_ids)} 个 (接收 2 个模型)")
        print(f"  弱客户端 (70%): {len(self.clients) - len(strong_ids)} 个 (接收 1 个模型)")
        print(f"  参与率：{self.join_ratio} / {self.num_clients}")
        
        # 初始化两个全局模型（深拷贝）
        # global_models[0] 和 global_models[1] 是两个不同的模型分支
        self.global_models = [copy.deepcopy(args.model), copy.deepcopy(args.model)]
        
        # 初始化两个 μ 值（近端正则化强度参数）
        # μ 值越大，本地模型越接近全局模型
        # μ 值越小，本地模型有更多自由度
        self.mus = [args.mu, args.mu]  # 初始值都为 args.mu
        
        # 初始化两个 TOPK 压缩比率
        # 压缩比率控制模型参数的压缩程度
        # ratio = 1.0 表示不压缩，ratio < 1.0 表示压缩
        self.topk_ratios = [args.topk_ratio, args.topk_ratio]  # 初始值都为 args.topk_ratio
        
        # Beam 搜索累积得分（初始值设为 0.5, 0.5 表示两个分支初始权重相等）
        self.beam_scores = [0.5, 0.5]
        
        # 上一轮评估准确率（用于计算趋势）
        # beam_accuracies[0] 对应 global_models[0] 的准确率
        # beam_accuracies[1] 对应 global_models[1] 的准确率
        self.beam_accuracies = [0.0, 0.0]
        
        # 动态 μ 调整参数
        # 当准确率提升超过阈值时，使用激进策略调整 μ
        self.trend_up_threshold = args.trend_up_threshold
        # 当准确率下降超过阈值时，使用保守策略调整 μ
        self.trend_down_threshold = args.trend_down_threshold
        # 激进因子：准确率提升时加大压缩（调整 μ）
        self.mu_factor_aggressive = args.compression_factor_aggressive
        # 保守因子：准确率下降时减少压缩（调整 μ）
        self.mu_factor_conservative = args.compression_factor_conservative
        # 稳定因子：准确率稳定时双向微调
        self.mu_factor_stable = args.compression_factor_stable
        # μ 值范围限制
        self.mu_min = 0.001
        self.mu_max = 1.0
        
        # TOPK 压缩比率动态调整参数
        # K值范围：[k_min, k_max]，k_min 表示最强压缩，k_max 表示最弱压缩
        self.k_min = 0.1  # 最强压缩：只保留 10% 的参数
        self.k_max = 1.0  # 最弱压缩：保留 100% 的参数（不压缩）
        
        # K值调整因子
        # 准确率提升时：尝试增加压缩强度（降低K值）
        self.k_factor_aggressive = 0.85  # K值减小 15%
        # 准确率下降时：减少压缩强度（提高K值）
        self.k_factor_conservative = 1.2  # K值增大 20%
        # 准确率稳定时：微调压缩强度
        self.k_factor_stable = 0.05  # K值微调 ±5%
        
        # 收敛速度容忍阈值
        # 如果准确率下降幅度小于此阈值，认为压缩是可以接受的
        self.convergence_tolerance = 0.005  # 0.5%
        
        # 记录训练时间
        self.Budget = []
        
        # 公共验证集准确率记录
        self.rs_public_acc = []


    def train(self):
        """
        FedProx-Beam 主训练流程
        
        训练逻辑：
        1. 遍历所有全局训练轮次
        2. 每轮执行以下步骤：
           a. 选择参与训练的客户端
           b. 发送模型到客户端（强客户端收 2 个，弱客户端收 1 个）
           c. 按评估间隔评估全局模型性能
           d. 客户端本地训练（使用近端正则化）
           e. 接收并聚合参数（构建 4 个候选分支）
           f. 评估候选模型并选择 Top-2
           g. 更新 μ 值和全局模型
        3. 训练结束后保存结果
        """
        # 遍历所有全局训练轮次
        for i in range(self.global_rounds + 1):
            s_t = time.time()  # 记录开始时间
            
            # 步骤 1: 选择本轮参与训练的客户端
            # 根据 join_ratio 随机选择一定比例的客户端
            self.selected_clients = self.select_clients()
            
            # 步骤 2: 发送全局模型到所有客户端
            # 强客户端接收 Top-2 模型，弱客户端接收 Top-1 模型
            self.send_models()
            
            # 步骤 3: 按评估间隔评估全局模型性能
            if i % self.eval_gap == 0:
                print(f"\n-------------第 {i} 轮-------------")
                print("\n评估全局模型")
                self.evaluate()
            
            # 步骤 4: 选中的客户端在本地进行训练
            # 每个客户端使用自己的本地数据，并应用近端正则化项
            for client in self.selected_clients:
                client.train()
            
            # 步骤 5: 接收客户端上传的模型参数并聚合
            # 这是 FedProx-Beam 的核心：构建 4 个候选分支并选择 Top-2
            self.receive_and_aggregate_parameters_beam()
            
            # 记录本轮训练时间
            self.Budget.append(time.time() - s_t)
            print('-' * 25, '时间消耗', '-' * 25, self.Budget[-1])
        
        # 训练结束，输出最佳性能
        print("\n训练完成。")
        print("\n最佳准确率:")
        print(max(self.rs_test_acc))
        print("\n平均时间消耗:")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))
        
        # 保存训练结果和全局模型
        self.save_results()
        self.save_global_model()
        
        # 如果有新客户端，进行微调评估
        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientProxBeam)
            print(f"\n-------------微调轮次-------------")
            print("\n评估新客户端")
            self.evaluate()


    def send_models(self):
        """
        发送全局模型到客户端
        
        发送策略：
        1. 按 beam_scores 排序，选择 Top-2 模型索引
        2. 强客户端：发送 Top-2 全局模型及其对应的 μ 值和 TOPK 压缩比率
        3. 弱客户端：发送 Top-1 全局模型及其对应的 μ 值和 TOPK 压缩比率
        
        方法说明：
        - 使用 client.set_parameters(models, mus, topk_ratios) 发送
        - 强客户端收到 2 个模型、 2 个 μ 值和 2 个 TOPK 压缩比率
        - 弱客户端收到 1 个模型、 1 个 μ 值和 1 个 TOPK 压缩比率
        """
        assert (len(self.clients) > 0)
        
        # 按累积 Beam 得分排序（降序）
        # sorted_indices[0] 是得分最高的模型索引
        # sorted_indices[1] 是得分第二高的模型索引
        sorted_indices = np.argsort(self.beam_scores)[::-1]
        
        # 打印排序信息
        print(f"\n[Beam Search 排序] 分支 {sorted_indices[0]} (得分：{self.beam_scores[sorted_indices[0]]:.6f}) > 分支 {sorted_indices[1]} (得分：{self.beam_scores[sorted_indices[1]]:.6f})")
        
        # 获取 Top-1 和 Top-2 的索引
        top_1_idx = sorted_indices[0]
        top_2_idx = sorted_indices[1]
        
        # 遍历所有客户端发送模型
        for client in self.clients:
            start_time = time.time()
            
            if client.num_models > 1:
                # 强客户端：发送 Top-2 全局模型及其对应的 μ 值和 TOPK 压缩比率
                # 深拷贝避免引用共享
                client.set_parameters(
                    [copy.deepcopy(self.global_models[top_1_idx]), 
                     copy.deepcopy(self.global_models[top_2_idx])],
                    [self.mus[top_1_idx], self.mus[top_2_idx]],
                    [self.topk_ratios[top_1_idx], self.topk_ratios[top_2_idx]]
                )
            else:
                # 弱客户端：只发送 Top-1 全局模型及其对应的 μ 值和 TOPK 压缩比率
                client.set_parameters(
                    copy.deepcopy(self.global_models[top_1_idx]),
                    self.mus[top_1_idx],
                    self.topk_ratios[top_1_idx]
                )
            
            # 记录发送时间开销
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)


    def receive_and_aggregate_parameters_beam(self):
        """
        接收并聚合参数（Beam Search 策略）
        
        这是 FedProx-Beam 的核心方法，实现基于 Beam Search 的参数聚合：
        
        聚合逻辑：
        1. 从客户端收集模型更新
        2. 构建 4 个候选分支：
           - 分支 00: 强客户端 M0 + 弱客户端 W0
           - 分支 01: 强客户端 M0 + 弱客户端 W1
           - 分支 10: 强客户端 M1 + 弱客户端 W0
           - 分支 11: 强客户端 M1 + 弱客户端 W1
        3. 使用简单平均聚合每个分支
        4. 评估 4 个候选模型
        5. 选择 Top-2 作为新的全局模型
        6. 更新 beam_scores 和 μ 值
        """
        assert (len(self.selected_clients) > 0)
        
        # 随机抽样活跃客户端（考虑客户端掉线率）
        active_clients = random.sample(
            self.selected_clients, 
            int((1 - self.client_drop_rate) * self.current_num_join_clients)
        )
        
        strong_updates = []  # 强客户端的模型更新列表
        weak_updates = []    # 弱客户端的模型更新列表
        
        # 处理每个活跃客户端的上传数据
        for client in active_clients:
            uploaded_data = client.get_upload_model()
            processed_models = []
            
            # 处理上传的数据
            if isinstance(uploaded_data, list):
                for m_data in uploaded_data:
                    processed_models.append(self._process_uploaded_model(m_data, client))
            else:
                processed_models.append(self._process_uploaded_model(uploaded_data, client))
            
            # 分类强客户端和弱客户端的更新
            # FedProxBeam: 强客户端返回 2 个训练后的模型，弱客户端返回 1 个
            if client.num_models > 1:
                # 强客户端：期望 2 个模型 [M0, M1]
                if len(processed_models) == 2:
                    strong_updates.append(processed_models)
            else:
                # 弱客户端：期望 1 个模型 [W0]
                if len(processed_models) == 1:
                    weak_updates.append(processed_models)
        
        # 使用简单平均聚合 4 个分支
        # FedProxBeam: 强客户端提供 2 个模型，弱客户端提供 1 个模型
        # 构建 4 个候选分支：
        #   分支 0: 所有客户端的模型（强客户端用 M0，弱客户端用 W0）- 保守策略
        #   分支 1: 所有客户端的模型（强客户端用 M1，弱客户端用 W0）- 激进策略
        #   分支 2: 只有强客户端的 M0 模型 - 探索策略 1
        #   分支 3: 只有强客户端的 M1 模型 - 探索策略 2
        candidates = []
        
        print("\n[Beam Search 聚合] 开始构建 4 个候选分支...")
        
        # 分支 0: 所有客户端的模型（强客户端用 M0，弱客户端用 W0）
        models_0 = [u[0] for u in strong_updates] + [u[0] for u in weak_updates]
        print(f"  分支 0 (保守策略): 聚合 {len(models_0)} 个模型 (强 M0{len(strong_updates)} + 弱 W0{len(weak_updates)})")
        candidates.append(self._aggregate_simple(models_0))
        
        # 分支 1: 所有客户端的模型（强客户端用 M1，弱客户端用 W0）
        models_1 = [u[1] for u in strong_updates] + [u[0] for u in weak_updates]
        print(f"  分支 1 (激进策略): 聚合 {len(models_1)} 个模型 (强 M1{len(strong_updates)} + 弱 W0{len(weak_updates)})")
        candidates.append(self._aggregate_simple(models_1))
        
        # 分支 2: 只有强客户端的 M0 模型（探索策略）
        if len(strong_updates) > 0:
            models_2 = [u[0] for u in strong_updates]
            print(f"  分支 2 (探索策略 1): 聚合 {len(models_2)} 个模型 (仅强 M0)")
            candidates.append(self._aggregate_simple(models_2))
        else:
            # 没有强客户端时使用分支 0 的模型
            candidates.append(copy.deepcopy(candidates[0]))
            print(f"  分支 2 (无强客户端): 复制分支 0")
        
        # 分支 3: 只有强客户端的 M1 模型（探索策略）
        if len(strong_updates) > 0:
            models_3 = [u[1] for u in strong_updates]
            print(f"  分支 3 (探索策略 2): 聚合 {len(models_3)} 个模型 (仅强 M1)")
            candidates.append(self._aggregate_simple(models_3))
        else:
            # 没有强客户端时使用分支 1 的模型
            candidates.append(copy.deepcopy(candidates[1]))
            print(f"  分支 3 (无强客户端): 复制分支 1")
        
        # 评估所有 4 个分支
        print("\n[Beam Search] 评估 4 个候选分支...")
        accs = self.evaluate_candidates(candidates)
        print(f"  准确率：{[f'{a:.4f}' for a in accs]}")
        
        # 记录最佳公共准确率
        if accs:
            self.rs_public_acc.append(max(accs))
        
        # 计算得分：Score_ij = Acc_ij * Prev_Norm_Score_i
        # 归一化上一轮的 beam_scores
        total_prev_score = sum(self.beam_scores)
        if total_prev_score > 0:
            norm_prev_scores = [s / total_prev_score for s in self.beam_scores]
        else:
            norm_prev_scores = [0.5, 0.5]
        
        raw_scores = []
        # 00, 01 来自父分支 0
        raw_scores.append(accs[0] * norm_prev_scores[0])
        raw_scores.append(accs[1] * norm_prev_scores[0])
        # 10, 11 来自父分支 1
        raw_scores.append(accs[2] * norm_prev_scores[1])
        raw_scores.append(accs[3] * norm_prev_scores[1])
        
        # 归一化得分用于当前轮次选择
        total_raw_score = sum(raw_scores)
        if total_raw_score > 0:
            scores = [s / total_raw_score for s in raw_scores]
        else:
            scores = [0.25] * 4
        
        print(f"  得分 (归一化): {[f'{s:.4f}' for s in scores]}")
        
        # 选择前 2 个分支
        # 如果得分相等，使用随机选择以避免总是选择 0 和 1
        if len(set(scores)) == 1:  # 所有得分相等
            top2_indices = np.random.choice([0, 1, 2, 3], 2, replace=False)
        else:
            top2_indices = np.argsort(scores)[::-1][:2]
        
        print(f"  选择的索引：{top2_indices}")
        
        # 更新全局模型和得分
        new_global_models = [candidates[i] for i in top2_indices]
        new_beam_scores = [scores[i] for i in top2_indices]
        
        # 更新 μ 值
        new_mus = self.update_mu_values(accs, top2_indices)
        
        # 更新 TOPK 压缩比率（K值）
        new_topk_ratios = self.update_topk_values(accs, top2_indices, new_mus)
        
        # 应用更新
        self.global_models = new_global_models
        self.beam_scores = new_beam_scores
        self.mus = new_mus
        self.topk_ratios = new_topk_ratios
        # 更新准确率用于下一轮比较
        self.beam_accuracies = [accs[i] for i in top2_indices]
        
        # 同步最佳模型用于其他用途
        best_of_top2 = 0 if scores[top2_indices[0]] > scores[top2_indices[1]] else 1
        self.global_model = copy.deepcopy(self.global_models[best_of_top2])
        
        print(f"  新的 μ 值：{[f'{m:.4f}' for m in self.mus]}")
        print(f"  新的 K 值：{[f'{k:.4f}' for k in self.topk_ratios]} (压缩率：{[(1-k)*100 for k in self.topk_ratios]}%)")


    def evaluate_candidates(self, candidates):
        """
        评估候选模型
        
        在公共验证集上评估 4 个候选模型的准确率
        
        参数:
            candidates: 候选模型列表（4 个模型）
        
        返回:
            [acc00, acc01, acc10, acc11]: 每个候选模型的准确率列表
        
        评估方法：
        1. 加载公共验证集
        2. 对每个候选模型进行前向传播
        3. 计算预测准确率
        """
        from torch.utils.data import DataLoader
        import torch
        import numpy as np
        
        # 加载公共验证集
        public_data_path = os.path.join('../dataset', self.dataset, 'public', 'public_data.npz')
        try:
            public_data = np.load(public_data_path, allow_pickle=True)
            public_images = torch.Tensor(public_data['x']).type(torch.float32)
            public_labels = torch.Tensor(public_data['y']).type(torch.int64)
            public_dataset = [(x, y) for x, y in zip(public_images, public_labels)]
            public_loader = DataLoader(public_dataset, self.batch_size, drop_last=False, shuffle=True)
        except Exception as e:
            print(f"加载公共验证集失败：{e}")
            return [0.0] * len(candidates)
        
        accs = []  # 存储每个模型的准确率
        
        # 遍历所有候选模型
        for model in candidates:
            model.eval()  # 设置为评估模式
            y_true, y_pred = [], []  # 存储真实标签和预测标签
            
            with torch.no_grad():  # 禁用梯度计算
                for x, y in public_loader:
                    # 处理输入数据
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    
                    # 前向传播
                    output = model(x)
                    pred = output.argmax(dim=1)  # 获取预测类别
                    
                    # 收集结果
                    y_true.extend(y.cpu().numpy())
                    y_pred.extend(pred.cpu().numpy())
            
            # 计算准确率
            accs.append(accuracy_score(y_true, y_pred))
        
        return accs


    def update_mu_values(self, accuracies, selected_indices):
        """
        更新 μ 值（近端正则化强度）
        
        根据模型性能趋势动态调整 μ 值：
        
        调整策略：
        1. 对每个选中的候选分支，计算趋势：trend = acc_current - acc_parent
        2. 根据趋势调整 μ 值：
           - trend > trend_up_threshold: 准确率提升，使用激进策略
             mu_new = mu_old * mu_factor_aggressive
           - trend < trend_down_threshold: 准确率下降，使用保守策略
             mu_new = mu_old * mu_factor_conservative
           - 否则：准确率稳定，使用稳定策略
             mu_new = mu_old * (1 ± mu_factor_stable)
        3. 限制 μ 值在 [mu_min, mu_max] 范围内
        
        参数:
            accuracies: 4 个候选分支的准确率列表 [acc00, acc01, acc10, acc11]
            selected_indices: 选中的 2 个候选分支索引
        
        返回:
            new_mus: 更新后的 2 个 μ 值列表
        """
        new_mus = []
        
        print("\n[μ 值更新] 动态调整近端正则化强度:")
        
        # 遍历选中的 2 个分支
        for i, idx in enumerate(selected_indices):
            # 确定父分支索引和准确率
            # 分支 0: 保守策略（所有客户端，强 M0 + 弱 W0）
            # 分支 1: 激进策略（所有客户端，强 M1 + 弱 W0）
            # 分支 2: 探索策略 1（仅强客户端 M0）
            # 分支 3: 探索策略 2（仅强客户端 M1）
            # 分支 0 和 2 使用 μ[0]，分支 1 和 3 使用 μ[1]
            if idx in [0, 2]:
                acc_parent = self.beam_accuracies[0]
                mu_old = self.mus[0]
                parent_str = "Global_0"
            else:
                acc_parent = self.beam_accuracies[1]
                mu_old = self.mus[1]
                parent_str = "Global_1"
            
            acc_current = accuracies[idx]
            trend = acc_current - acc_parent  # 计算趋势
            
            branch_names = ["保守策略", "激进策略", "探索策略 1", "探索策略 2"]
            branch_str = branch_names[idx]
            print(f"  获胜分支 {i+1}: {branch_str} (Acc: {acc_current:.4f}) <- 父分支 {parent_str} (Acc: {acc_parent:.4f})")
            print(f"    趋势：{trend:+.4f} (阈值：上升>{self.trend_up_threshold}, 下降<{self.trend_down_threshold})")
            
            # 根据趋势调整 μ 值
            if trend > self.trend_up_threshold:
                # 准确率提升：加大压缩（调整 μ）
                mu_new = mu_old * self.mu_factor_aggressive
                action = f"准确率提升 -> 激进调整 ({self.mu_factor_aggressive}x)"
            elif trend < self.trend_down_threshold:
                # 准确率下降：减少压缩（调整 μ）
                mu_new = mu_old * self.mu_factor_conservative
                action = f"准确率下降 -> 保守调整 ({self.mu_factor_conservative}x)"
            else:
                # 准确率稳定：双向微调
                # 随机选择增加或减少
                if random.random() < 0.5:
                    mu_new = mu_old * (1 + self.mu_factor_stable)
                else:
                    mu_new = mu_old * (1 - self.mu_factor_stable)
                action = f"准确率稳定 -> 双向微调 (+/- {self.mu_factor_stable*100}%)"
            
            # 限制 μ 值在 [mu_min, mu_max] 范围内
            mu_new = np.clip(mu_new, self.mu_min, self.mu_max)
            
            new_mus.append(mu_new)
            
            print(f"    动作：{action}")
            print(f"    旧 μ: {mu_old:.4f} -> 新 μ: {mu_new:.4f} (范围：[{self.mu_min}, {self.mu_max}])")
        
        return new_mus


    def update_topk_values(self, accuracies, selected_indices, new_mus):
        """
        动态调整 TOPK 压缩比率（K值）
        
        核心算法：
        1. 根据准确率变化趋势调整压缩强度
        2. 与μ值相互制衡，找到动态最优平衡点
        3. 考虑收敛速度和通信开销的权衡
        
        调整策略：
        - 准确率提升 > trend_up：尝试增加压缩强度（降低K值）
          * 如果μ值增大（模型更稳定），可以容忍更强的压缩
          * 如果μ值减小（模型更自由），需要保守压缩
        - 准确率下降 < trend_down：减少压缩强度（提高K值）
          * 如果下降幅度 < convergence_tolerance，认为压缩可接受
          * 否则需要减少压缩以确保收敛
        - 准确率稳定：微调压缩强度
          * 根据μ值变化方向调整K值
        
        参数:
            accuracies: 4 个候选分支的准确率列表
            selected_indices: 选中的 Top-2 分支索引
            new_mus: 新的μ值列表
        
        返回:
            new_topk_ratios: 新的 TOPK 压缩比率列表
        """
        new_topk_ratios = []
        
        print(f"\n[TOPK 压缩调整] 动态调整压缩强度...")
        
        # 遍历选中的 2 个分支
        for i, idx in enumerate(selected_indices):
            # 确定父分支索引和准确率
            if idx in [0, 2]:
                acc_parent = self.beam_accuracies[0]
                k_old = self.topk_ratios[0]
                mu_old = self.mus[0]
                mu_new = new_mus[i]
                parent_str = "Global_0"
            else:
                acc_parent = self.beam_accuracies[1]
                k_old = self.topk_ratios[1]
                mu_old = self.mus[1]
                mu_new = new_mus[i]
                parent_str = "Global_1"
            
            acc_current = accuracies[idx]
            trend = acc_current - acc_parent  # 计算趋势
            
            # 计算μ值变化
            mu_change = mu_new - mu_old
            mu_change_ratio = mu_change / mu_old if mu_old > 0 else 0
            
            branch_names = ["保守策略", "激进策略", "探索策略 1", "探索策略 2"]
            branch_str = branch_names[idx]
            
            print(f"  分支 {i+1}: {branch_str} (Acc: {acc_current:.4f})")
            print(f"    趋势：{trend:+.4f}, μ变化：{mu_change:+.4f} ({mu_change_ratio:+.2%})")
            
            # 根据趋势和μ值变化调整 K 值
            if trend > self.trend_up_threshold:
                # 准确率提升：尝试增加压缩强度（降低K值）
                # μ值增大 → 模型更稳定 → 可以容忍更强的压缩
                # μ值减小 → 模型更自由 → 需要保守压缩
                
                if mu_change > 0:
                    # μ值增大，模型更稳定，可以更激进地压缩
                    k_new = k_old * self.k_factor_aggressive * 0.95  # 额外减少 5%
                    action = f"准确率提升 + μ增大 -> 激进压缩 (K减小 {self.k_factor_aggressive*0.95:.2f}x)"
                else:
                    # μ值减小，模型更自由，需要保守压缩
                    k_new = k_old * self.k_factor_aggressive
                    action = f"准确率提升 + μ减小 -> 适度压缩 (K减小 {self.k_factor_aggressive:.2f}x)"
            
            elif trend < self.trend_down_threshold:
                # 准确率下降：需要判断是否可接受
                if abs(trend) < self.convergence_tolerance:
                    # 下降幅度在容忍范围内，认为压缩可接受
                    # 可以保持当前压缩强度，甚至尝试进一步压缩
                    if mu_change > 0:
                        # μ值增大，模型更稳定，可以尝试继续压缩
                        k_new = k_old * self.k_factor_aggressive
                        action = f"下降可容忍 + μ增大 -> 继续压缩 (K减小 {self.k_factor_aggressive:.2f}x)"
                    else:
                        # μ值减小，保持当前压缩强度
                        k_new = k_old
                        action = f"下降可容忍 + μ减小 -> 保持压缩 (K不变)"
                else:
                    # 下降幅度过大，需要减少压缩强度（提高K值）
                    # μ值增大 → 模型更稳定 → 可以适度减少压缩
                    # μ值减小 → 模型更自由 → 需要大幅减少压缩
                    
                    if mu_change > 0:
                        # μ值增大，适度减少压缩
                        k_new = k_old * self.k_factor_conservative * 0.95
                        action = f"下降过大 + μ增大 -> 适度减压 (K增大 {self.k_factor_conservative*0.95:.2f}x)"
                    else:
                        # μ值减小，大幅减少压缩
                        k_new = k_old * self.k_factor_conservative
                        action = f"下降过大 + μ减小 -> 大幅减压 (K增大 {self.k_factor_conservative:.2f}x)"
            
            else:
                # 准确率稳定：微调压缩强度
                # 根据μ值变化方向调整K值
                if mu_change > 0:
                    # μ值增大，模型更稳定，可以尝试增加压缩
                    k_new = k_old * (1 - self.k_factor_stable)
                    action = f"准确率稳定 + μ增大 -> 微增压缩 (K减小 {self.k_factor_stable*100}%)"
                elif mu_change < 0:
                    # μ值减小，模型更自由，需要微减压缩
                    k_new = k_old * (1 + self.k_factor_stable)
                    action = f"准确率稳定 + μ减小 -> 微减压缩 (K增大 {self.k_factor_stable*100}%)"
                else:
                    # μ值不变，随机微调
                    if random.random() < 0.5:
                        k_new = k_old * (1 - self.k_factor_stable)
                        action = f"准确率稳定 + μ不变 -> 随机微调 (K减小 {self.k_factor_stable*100}%)"
                    else:
                        k_new = k_old * (1 + self.k_factor_stable)
                        action = f"准确率稳定 + μ不变 -> 随机微调 (K增大 {self.k_factor_stable*100}%)"
            
            # 限制 K 值在 [k_min, k_max] 范围内
            k_new = np.clip(k_new, self.k_min, self.k_max)
            
            new_topk_ratios.append(k_new)
            
            print(f"    动作：{action}")
            print(f"    旧 K: {k_old:.4f} -> 新 K: {k_new:.4f} (压缩率：{(1-k_new)*100:.1f}%)")
        
        return new_topk_ratios


    def _aggregate_simple(self, models):
        """
        简单聚合模型参数
        
        使用简单平均的方法聚合多个模型的参数
        
        参数:
            models: 模型列表
        
        返回:
            聚合后的新模型
        
        聚合方法：
        1. 深拷贝第一个模型作为基础
        2. 清零所有参数
        3. 累加所有模型的参数
        4. 取平均值
        """
        if not models:  # 如果模型列表为空
            return copy.deepcopy(self.global_models[0])
        
        # 深拷贝第一个模型作为基础
        new_model = copy.deepcopy(models[0])
        
        # 清零所有参数
        for param in new_model.parameters():
            param.data.zero_()
        
        n = len(models)  # 模型数量
        
        # 累加所有模型的参数
        for model in models:
            for new_p, old_p in zip(new_model.parameters(), model.parameters()):
                new_p.data += old_p.data
        
        # 取平均值
        for param in new_model.parameters():
            param.data /= n
        
        return new_model


    def _process_uploaded_model(self, uploaded_data, client):
        """
        处理上传的模型
        
        检查上传的数据是否被压缩，如果是则解压缩
        
        参数:
            uploaded_data: 上传的数据
            client: 客户端对象
        
        返回:
            处理后的模型
        """
        # 检查是否为压缩数据
        if isinstance(uploaded_data, dict) and uploaded_data.get('_is_compressed', False):
            from utils.compression import decompress_from_communication
            try:
                # 解压缩数据
                return decompress_from_communication(uploaded_data, client.model, self.device)
            except Exception as e:
                # 解压缩失败，返回客户端模型
                print(f"解压缩失败：{e}")
                return client.model
        
        # 非压缩数据直接返回
        return uploaded_data


    def save_results(self):
        """
        保存训练结果到 H5 文件
        
        保存的指标包括：
        - rs_test_acc: 测试准确率
        - rs_test_auc: 测试 AUC
        - rs_train_loss: 训练损失
        - rs_public_acc: 公共验证集准确率
        """
        # 构建算法名称
        algo = self.dataset + "_" + self.algorithm
        result_path = self.args.result_dir
        
        # 创建结果目录（如果不存在）
        if not os.path.exists(result_path):
            os.makedirs(result_path, exist_ok=True)
        
        if len(self.rs_test_acc):  # 如果有测试准确率记录
            # 添加时间戳确保每次运行的文件名唯一
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            # 在文件名中包含初始 μ 和 topk 参数
            params_suffix = f"_mu{self.args.mu}_k{self.args.topk_ratio}"
            algo = algo + "_" + self.goal + "_" + str(self.times) + params_suffix + "_" + timestamp
            file_path = os.path.join(result_path, "{}.h5".format(algo))
            print("结果保存路径：" + file_path)
            
            # 保存数据到 H5 文件
            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                if len(self.rs_public_acc) > 0:
                    hf.create_dataset('rs_public_acc', data=self.rs_public_acc)
