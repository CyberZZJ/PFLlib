import time
import numpy as np
import random
import copy
import torch
import os
import h5py
from flcore.clients.clientbeam import clientBeam
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data
from sklearn.metrics import accuracy_score

class FedBeam(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # 选择慢速客户端
        self.set_slow_clients()
        
        # 使用 clientBeam 初始化客户端
        self.set_clients(clientBeam)

        # 客户端异质性分类
        # 1. 按数据量排序
        sorted_clients = sorted(self.clients, key=lambda c: c.train_samples, reverse=True)
        num_strong = int(len(self.clients) * 0.3)
        strong_ids = set(c.id for c in sorted_clients[:num_strong])
        
        # 2. 标记客户端并初始化贡献度
        self.contributions = {c.id: 1.0 / len(self.clients) for c in self.clients}
        for c in self.clients:
            if c.id in strong_ids:
                c.num_models = 2 # Beam width K=2
            else:
                c.num_models = 1
        
        print(f"\nFedBeam Initialized:")
        print(f"  Strong clients (30%): {len(strong_ids)} (Send 2 models)")
        print(f"  Weak clients (70%): {len(self.clients) - len(strong_ids)} (Send 1 model)")
        print(f"  Participation Ratio: {self.join_ratio} / {self.num_clients}")
        
        # Initialize two global models for the experiment
        self.global_models = [copy.deepcopy(args.model), copy.deepcopy(args.model)]
        self.beam_accuracies = [0.0, 0.0]
        self.beam_scores = [0.5, 0.5] # Cumulative Beam Search scores
        
        self.topk_ratios = [args.topk_ratio] * 4
        self.dynamic_topk_ratio = args.topk_ratio
        self.prev_best_acc = 0.0
        
        # Initialize new parameters
        self.result_dir = args.result_dir
        self.trend_up_threshold = args.trend_up_threshold
        self.trend_down_threshold = args.trend_down_threshold
        self.compression_factor_aggressive = args.compression_factor_aggressive
        self.compression_factor_conservative = args.compression_factor_conservative
        self.compression_factor_stable = args.compression_factor_stable

        self.Budget = []

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            
            # Send models: Strong gets top 2 models, Weak gets top 1 model
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------第 {i} 轮-------------")
                print("\n评估全局模型")
                # We can keep evaluation for logging, but logic is now in aggregation
                # if self.public_loader is not None:
                #      self.evaluate_public_multiple()
                self.evaluate()  

            for client in self.selected_clients:
                client.train()

            # Receive and aggregate into two new global models using Beam Search strategy
            self.receive_and_aggregate_parameters_beam()

            self.Budget.append(time.time() - s_t)
            print('-'*25, '时间消耗', '-'*25, self.Budget[-1])

        print("\n训练完成。")
        self.save_results()

    def send_models(self):
        assert (len(self.clients) > 0)
        
        # Sort global models by CUMULATIVE beam scores (descending)
        sorted_indices = np.argsort(self.beam_scores)[::-1]
        
        print(f"\n[Beam Search 排序] 分支 {sorted_indices[0]} (得分: {self.beam_scores[sorted_indices[0]]:.6f}) > 分支 {sorted_indices[1]} (得分: {self.beam_scores[sorted_indices[1]]:.6f})")
        
        top_1_idx = sorted_indices[0]
        top_2_idx = sorted_indices[1]

        for client in self.clients:
            start_time = time.time()
            
            if client.num_models > 1:
                # Strong client receives top 2 global models
                client.set_parameters([copy.deepcopy(self.global_models[top_1_idx]), 
                                     copy.deepcopy(self.global_models[top_2_idx])], self.topk_ratios)
            else:
                # Weak client receives the best global model
                client.set_parameters(copy.deepcopy(self.global_models[top_1_idx]), self.topk_ratios)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_and_aggregate_parameters_beam(self):
        """接收并聚合参数（Beam Search策略）
        
        这是FedBeam的核心方法，实现了基于Beam Search的参数聚合策略：
        1. 从客户端收集模型更新
        2. 构建4个候选分支
        3. 评估候选分支
        4. 计算分支得分
        5. 选择前2个最佳分支作为下一轮的父分支
        6. 根据趋势动态调整TopK压缩率
        7. 更新全局模型
        """
        assert (len(self.selected_clients) > 0)  # 确保已选择客户端

        # 随机抽样活跃客户端（考虑客户端掉线率）
        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.current_num_join_clients))

        strong_updates = []  # 强客户端的模型更新
        weak_updates = []    # 弱客户端的模型更新

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
            if client.num_models > 1:
                # 强客户端：期望4个模型 [M00, M01, M10, M11]
                if len(processed_models) == 4:
                    strong_updates.append(processed_models)
            else:
                # 弱客户端：期望2个模型 [W00, W01]
                if len(processed_models) == 2:
                    weak_updates.append(processed_models)

        # 使用简单平均聚合4个分支
        candidates = []
        
        print("\n[Beam Search 聚合] 开始构建 4 个候选分支...")
        
        # 分支 0_0: 强客户端(M00) + 弱客户端(W00)
        models_00 = [u[0] for u in strong_updates] + [u[0] for u in weak_updates]
        print(f"  分支 0_0 (基于 Global_0, 压缩 k0): 聚合 {len(models_00)} 个模型 (强{len(strong_updates)} + 弱{len(weak_updates)})")
        candidates.append(self._aggregate_simple(models_00))
        
        # 分支 0_1: 强客户端(M01) + 弱客户端(W01)
        models_01 = [u[1] for u in strong_updates] + [u[1] for u in weak_updates]
        print(f"  分支 0_1 (基于 Global_0, 压缩 k1): 聚合 {len(models_01)} 个模型 (强{len(strong_updates)} + 弱{len(weak_updates)})")
        candidates.append(self._aggregate_simple(models_01))
        
        # 分支 1_0: 强客户端(M10) + 弱客户端(W00)
        models_10 = [u[2] for u in strong_updates] + [u[0] for u in weak_updates]
        print(f"  分支 1_0 (基于 Global_1, 压缩 k0): 聚合 {len(models_10)} 个模型 (强{len(strong_updates)} + 弱{len(weak_updates)})")
        candidates.append(self._aggregate_simple(models_10))
        
        # 分支 1_1: 强客户端(M11) + 弱客户端(W01)
        models_11 = [u[3] for u in strong_updates] + [u[1] for u in weak_updates]
        print(f"  分支 1_1 (基于 Global_1, 压缩 k1): 聚合 {len(models_11)} 个模型 (强{len(strong_updates)} + 弱{len(weak_updates)})")
        candidates.append(self._aggregate_simple(models_11))
        
        # 评估所有4个分支
        print("\n[Beam Search] 评估 4 个候选分支...")
        accs = self.evaluate_candidates(candidates)
        print(f"  准确率: {[f'{a:.4f}' for a in accs]}")
        
        # 记录最佳公共准确率
        if accs:
            self.rs_public_acc.append(max(accs))
        
        # 计算得分: Score_ij = Acc_ij * Prev_Norm_Score_i
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
        
        # 选择前2个分支
        # 如果得分相等，使用随机选择以避免总是选择0和1
        if len(set(scores)) == 1:  # 所有得分相等
             top2_indices = np.random.choice([0, 1, 2, 3], 2, replace=False)
        else:
             top2_indices = np.argsort(scores)[::-1][:2]
              
        print(f"  选择的索引: {top2_indices}")
        
        # 更新全局模型和得分
        new_global_models = [candidates[i] for i in top2_indices]
        new_beam_scores = [scores[i] for i in top2_indices]
        
        # 根据趋势更新topk_ratios
        new_topk_ratios = []
        print("\n[Beam Search 更新] 动态调整 TopK 压缩率:")
        for i, idx in enumerate(top2_indices):
            # 确定父分支索引和准确率
            if idx in [0, 1]:
                acc_parent = self.beam_accuracies[0]
                k_used = self.topk_ratios[idx]  # 0 or 1
                parent_str = "Global_0"
            else:
                acc_parent = self.beam_accuracies[1]
                k_used = self.topk_ratios[idx]  # 2 or 3
                parent_str = "Global_1"
            
            acc_current = accs[idx]
            trend = acc_current - acc_parent  # 计算趋势
            
            branch_str = ["0_0", "0_1", "1_0", "1_1"][idx]
            print(f"  获胜分支 {i+1}: {branch_str} (Acc: {acc_current:.4f}) <- 父分支 {parent_str} (Acc: {acc_parent:.4f})")
            print(f"    趋势: {trend:+.4f} (阈值: 上升>{self.trend_up_threshold}, 下降<{self.trend_down_threshold})")
            
            # 根据趋势调整压缩率
            if trend > self.trend_up_threshold:
                # 精度提升，加大压缩
                pair = [k_used * self.compression_factor_aggressive, k_used]
                action = f"精度提升 -> 加大压缩 ({self.compression_factor_aggressive}x)"
            elif trend < self.trend_down_threshold:
                # 精度下降，减少压缩
                pair = [k_used * self.compression_factor_conservative, k_used]
                action = f"精度下降 -> 减少压缩 ({self.compression_factor_conservative}x)"
            else:
                # 精度稳定，双向微调
                pair = [k_used * (1 - self.compression_factor_stable), k_used * (1 + self.compression_factor_stable)]
                action = f"精度稳定 -> 双向微调 (+/- {self.compression_factor_stable*100}%)"
            
            # 确保压缩率在合理范围内
            pair = [np.clip(p, 0.05, 1.0) for p in pair]
            new_topk_ratios.extend(pair)
            print(f"    动作: {action}")
            print(f"    旧 K: {k_used:.4f} -> 新 K对: [{pair[0]:.4f}, {pair[1]:.4f}]")
            
        # 更新参数
        self.topk_ratios = new_topk_ratios
        self.global_models = new_global_models
        self.beam_scores = new_beam_scores
        # 更新准确率用于下一轮比较
        self.beam_accuracies = [accs[i] for i in top2_indices]
        
        # 同步最佳模型用于其他用途
        best_of_top2 = 0 if scores[top2_indices[0]] > scores[top2_indices[1]] else 1
        self.global_model = copy.deepcopy(self.global_models[best_of_top2])
        print(f"  新的压缩率: {[f'{r:.4f}' for r in self.topk_ratios]}")

    def _aggregate_simple(self, models):
        """简单聚合模型参数
        
        使用简单平均的方法聚合多个模型的参数。
        
        Args:
            models: 模型列表
            
        Returns:
            聚合后的新模型
        """
        if not models:  # 如果模型列表为空
            return copy.deepcopy(self.global_models[0])
            
        # 深拷贝第一个模型作为基础
        new_model = copy.deepcopy(models[0])
        # 清零参数
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

    def evaluate_candidates(self, candidates):
        """评估候选模型
        
        在公共数据集上评估多个候选模型的准确率。
        
        Args:
            candidates: 候选模型列表
            
        Returns:
            每个模型的准确率列表
        """
        if self.public_loader is None:  # 如果没有公共数据集加载器
            return [0.0] * len(candidates)
        
        accs = []  # 存储每个模型的准确率
        for model in candidates:
            model.eval()  # 设置为评估模式
            y_true, y_pred = [], []  # 存储真实标签和预测标签
            with torch.no_grad():  # 禁用梯度计算
                for x, y in self.public_loader:
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

    def _process_uploaded_model(self, uploaded_data, client):
        """处理上传的模型
        
        检查上传的数据是否被压缩，如果是则解压缩。
        
        Args:
            uploaded_data: 上传的数据
            client: 客户端对象
            
        Returns:
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
                return client.model
        # 非压缩数据直接返回
        return uploaded_data

    def evaluate_public_multiple(self):
        """评估多个全局模型
        
        在公共数据集上评估所有全局模型的表现。
        
        Returns:
            每个全局模型的准确率列表
        """
        if self.public_loader is None:  # 如果没有公共数据集加载器
            return [0.0, 0.0]

        print(f"\n[验证集评估] 各分支全局模型在公共数据集上的表现:")
        accs = []  # 存储每个模型的准确率
        for idx, gm in enumerate(self.global_models):
            gm.eval()  # 设置为评估模式
            y_true, y_pred = [], []  # 存储真实标签和预测标签
            with torch.no_grad():  # 禁用梯度计算
                for x, y in self.public_loader:
                    # 处理输入数据
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    
                    # 前向传播
                    output = gm(x)
                    pred = output.argmax(dim=1)  # 获取预测类别
                    
                    # 收集结果
                    y_true.extend(y.cpu().numpy())
                    y_pred.extend(pred.cpu().numpy())
            
            # 计算准确率
            acc = accuracy_score(y_true, y_pred)
            accs.append(acc)
            # 使用特殊格式使输出在日志中突出显示
            print(f"  >>> 分支 {idx} 全局模型验证集准确率: {acc:.4f}")
        return accs

    def save_results(self):
        """保存结果
        
        将训练结果保存到H5文件中，包括测试准确率、AUC、训练损失等指标。
        """
        # 构建算法名称
        algo = self.dataset + "_" + self.algorithm
        result_path = self.result_dir
        
        # 创建结果目录（如果不存在）
        if not os.path.exists(result_path):
            os.makedirs(result_path, exist_ok=True)

        if len(self.rs_test_acc):  # 如果有测试准确率记录
            # 添加压缩率到文件名
            compression_suffix = f"_topk({self.args.topk_ratio})" if self.enable_compression else ""
            
            # 添加时间戳确保每次运行的文件名唯一
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            # 在文件名中包含初始mu和topk参数
            params_suffix = f"_mu{self.args.mu}_k{self.args.topk_ratio}"
            algo = algo + "_" + self.goal + "_" + str(self.times) + params_suffix + "_" + timestamp
            file_path = os.path.join(result_path, "{}.h5".format(algo))
            print("结果保存路径: " + file_path)

            # 保存数据到H5文件
            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                if len(self.rs_public_acc) > 0:
                    hf.create_dataset('rs_public_acc', data=self.rs_public_acc)
                    hf.create_dataset('rs_public_prec', data=self.rs_public_prec)
                    hf.create_dataset('rs_public_rec', data=self.rs_public_rec)

