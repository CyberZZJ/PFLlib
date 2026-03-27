#!/usr/bin/env python
import os
import sys
import copy
import torch
import argparse
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.dataset_config import (
    get_dataset_config, 
    recommend_initial_params, 
    generate_param_grid,
    calculate_privacy_budget,
    validate_privacy_constraint,
    print_dataset_info
)
from utils.training_monitor import AdaptiveMonitor
from utils.report_generator import ReportGenerator

warnings.simplefilter("ignore")
torch.manual_seed(0)


class AdaptiveDPTuner:
    def __init__(self, args):
        self.args = args
        self.dataset_name = args.dataset
        self.algorithm = args.algorithm
        self.max_rounds = args.global_rounds
        self.patience = args.patience
        self.min_delta = args.min_delta
        self.max_param_combinations = args.max_param_combinations
        
        self.config = get_dataset_config(self.dataset_name)
        self.monitor = AdaptiveMonitor(
            patience=self.patience,
            min_delta=self.min_delta,
            max_rounds=self.max_rounds
        )
        
        self.results: List[Dict] = []
        self.current_params: Optional[Dict] = None
        self.param_combinations: List[Dict] = []
        self.current_param_idx: int = 0
        
        self.output_dir = os.path.join(
            os.path.dirname(__file__),
            "adaptive_results",
            f"{self.dataset_name}_{self.algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"\n{'#'*70}")
        print(f"自适应差分隐私调参系统初始化")
        print(f"{'#'*70}")
        print(f"数据集: {self.dataset_name}")
        print(f"算法: {self.algorithm}")
        print(f"最大轮数: {self.max_rounds}")
        print(f"耐心值: {self.patience}轮")
        print(f"最小提升阈值: {self.min_delta}")
        print(f"最大参数组合数: {self.max_param_combinations}")
        print(f"输出目录: {self.output_dir}")
        print(f"{'#'*70}\n")
        
        print_dataset_info(self.dataset_name)
    
    def generate_param_combinations(self) -> List[Dict]:
        print("\n[参数生成] 正在生成参数组合...")
        
        param_grid = generate_param_grid(
            self.dataset_name, 
            self.algorithm, 
            num_samples=3
        )
        
        recommended = recommend_initial_params(self.dataset_name, self.algorithm)
        param_grid.insert(0, recommended)
        
        valid_combinations = []
        for params in param_grid:
            actual_epsilon = calculate_privacy_budget(
                params['clip_norm'],
                params['noise_multiplier'],
                params['delta']
            )
            
            if validate_privacy_constraint(actual_epsilon, self.config.epsilon_range[1]):
                params['actual_epsilon'] = actual_epsilon
                valid_combinations.append(params)
        
        self.param_combinations = valid_combinations[:self.max_param_combinations]
        
        print(f"[参数生成] 生成了 {len(self.param_combinations)} 组有效参数组合")
        for i, params in enumerate(self.param_combinations[:5]):
            print(f"  组合{i+1}: ε={params['epsilon']:.2f}, μ={params['mu']:.3f}, "
                  f"C={params['clip_norm']:.2f}, σ={params['noise_multiplier']:.2f}")
        
        return self.param_combinations
    
    def update_args_with_params(self, params: Dict) -> None:
        self.args.mu = params.get('mu', 0.1)
        self.args.dp_clip_norm = params.get('clip_norm', 1.0)
        self.args.dp_noise_multiplier = params.get('noise_multiplier', 1.0)
        self.args.dp_epsilon = params.get('epsilon', 5.0)
        self.args.dp_delta = params.get('delta', 1e-5)
        
        if self.algorithm == "DPProxNative":
            self.args.lambda_privacy = params.get('lambda_privacy', 0.05)
            self.args.supplementary_noise_scale = params.get('supplementary_noise_scale', 0.001)
        
        print(f"\n[参数更新] 当前参数:")
        print(f"  μ = {self.args.mu}")
        print(f"  C (clip_norm) = {self.args.dp_clip_norm}")
        print(f"  σ (noise_multiplier) = {self.args.dp_noise_multiplier}")
        print(f"  ε = {self.args.dp_epsilon}")
        print(f"  δ = {self.args.dp_delta}")
    
    def run_single_training(self, params: Dict, param_idx: int) -> Dict:
        print(f"\n{'='*70}")
        print(f"[训练开始] 参数组合 #{param_idx + 1}/{len(self.param_combinations)}")
        print(f"{'='*70}")
        
        self.update_args_with_params(params)
        
        session_id = self.monitor.start_session(
            dataset_name=self.dataset_name,
            algorithm=self.algorithm,
            params=params
        )
        
        server = self._create_server()
        
        best_accuracy = 0.0
        best_round = 0
        no_improve_count = 0
        privacy_budget = calculate_privacy_budget(
            params['clip_norm'],
            params['noise_multiplier'],
            params['delta']
        )
        total_rounds = 0
        
        try:
            for round_num in range(1, self.max_rounds + 1):
                server.current_round = round_num - 1
                
                server.train_one_round()
                
                accuracy, auc = server.evaluate_public()
                loss = 1.0 - accuracy
                
                result = self.monitor.record_round(
                    round_num=round_num,
                    accuracy=accuracy,
                    loss=loss,
                    privacy_budget=privacy_budget
                )
                
                total_rounds = round_num
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_round = round_num
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                print(f"[轮次 {round_num:3d}] 准确率: {accuracy:.4f} | "
                      f"最佳: {best_accuracy:.4f} (轮次{best_round}) | "
                      f"无提升轮数: {no_improve_count}/{self.patience}")
                
                if result['should_terminate']:
                    print(f"\n[终止信号] {result['termination_reason']}")
                    break
                
                if no_improve_count >= self.patience:
                    print(f"\n[早停] 连续{self.patience}轮无提升，终止训练")
                    break
                
        except KeyboardInterrupt:
            print("\n[中断] 用户手动终止训练")
        except Exception as e:
            print(f"\n[错误] 训练过程出错: {str(e)}")
            import traceback
            traceback.print_exc()
        
        summary = self.monitor.end_session(session_id)
        
        training_result = {
            "param_idx": param_idx,
            "params": params,
            "best_accuracy": best_accuracy,
            "best_round": best_round,
            "total_rounds": total_rounds,
            "privacy_budget": privacy_budget,
            "session_id": session_id,
            "terminated_early": no_improve_count >= self.patience
        }
        
        self.results.append(training_result)
        
        self.monitor.update_best_overall(best_accuracy, params)
        
        self._save_intermediate_result(training_result)
        
        print(f"\n[训练结束] 最佳准确率: {best_accuracy:.4f} (第{best_round}轮)")
        
        return training_result
    
    def _create_server(self):
        from flcore.servers.serverdpprox import DPProx
        from flcore.servers.serverdpproxnative import DPProxNative
        from flcore.trainmodel.models import FedAvgCNN
        
        if "MNIST" in self.dataset_name:
            self.args.model = FedAvgCNN(in_features=1, num_classes=self.args.num_classes, dim=1024).to(self.args.device)
        elif "Cifar10" in self.dataset_name:
            self.args.model = FedAvgCNN(in_features=3, num_classes=self.args.num_classes, dim=1600).to(self.args.device)
        else:
            self.args.model = FedAvgCNN(in_features=1, num_classes=self.args.num_classes, dim=1024).to(self.args.device)
        
        if self.algorithm == "DPProx":
            return DPProx(self.args, 0)
        elif self.algorithm == "DPProxNative":
            return DPProxNative(self.args, 0)
        else:
            raise ValueError(f"不支持的算法: {self.algorithm}")
    
    def _save_intermediate_result(self, result: Dict) -> None:
        filepath = os.path.join(self.output_dir, "intermediate_results.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
    
    def run_adaptive_tuning(self) -> Dict:
        print("\n" + "#"*70)
        print("# 开始自适应调参")
        print("#"*70)
        
        self.generate_param_combinations()
        
        for idx, params in enumerate(self.param_combinations):
            print(f"\n{'*'*70}")
            print(f"参数组合进度: {idx + 1}/{len(self.param_combinations)}")
            print(f"{'*'*70}")
            
            result = self.run_single_training(params, idx)
            
            if result['best_accuracy'] >= self.args.target_accuracy:
                print(f"\n[达标] 准确率 {result['best_accuracy']:.4f} 达到目标 {self.args.target_accuracy}")
                break
        
        final_report = self.generate_final_report()
        
        return final_report
    
    def generate_final_report(self) -> Dict:
        print("\n" + "#"*70)
        print("# 生成最终报告")
        print("#"*70)
        
        if not self.results:
            print("警告: 没有训练结果")
            return {}
        
        best_result = max(self.results, key=lambda x: x['best_accuracy'])
        
        report = {
            "metadata": {
                "dataset": self.dataset_name,
                "algorithm": self.algorithm,
                "timestamp": datetime.now().isoformat(),
                "total_param_combinations": len(self.results),
            },
            "best_result": {
                "accuracy": best_result['best_accuracy'],
                "round": best_result['best_round'],
                "params": best_result['params'],
                "privacy_budget": best_result['privacy_budget'],
            },
            "all_results": [
                {
                    "param_idx": r['param_idx'],
                    "accuracy": r['best_accuracy'],
                    "round": r['best_round'],
                    "params": r['params'],
                    "privacy_budget": r['privacy_budget'],
                    "terminated_early": r['terminated_early'],
                }
                for r in sorted(self.results, key=lambda x: x['best_accuracy'], reverse=True)
            ],
            "dataset_config": {
                "epsilon_range": self.config.epsilon_range,
                "clip_norm_range": self.config.clip_norm_range,
                "noise_multiplier_range": self.config.noise_multiplier_range,
                "mu_range": self.config.mu_range,
            }
        }
        
        report_path = os.path.join(self.output_dir, "final_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n最终报告已保存: {report_path}")
        
        self._generate_visual_reports()
        
        self._print_summary(report)
        
        return report
    
    def _generate_visual_reports(self) -> None:
        try:
            report_gen = ReportGenerator(self.output_dir)
            
            config = {
                "epsilon_range": self.config.epsilon_range,
                "clip_norm_range": self.config.clip_norm_range,
                "noise_multiplier_range": self.config.noise_multiplier_range,
                "mu_range": self.config.mu_range,
            }
            
            report_gen.generate_full_report(self.results, self.dataset_name, 
                                           self.algorithm, config)
            report_gen.generate_accuracy_curve(self.results)
            report_gen.generate_privacy_analysis(self.results)
            report_gen.generate_json_report(self.results, self.dataset_name,
                                           self.algorithm, config)
            
            print("\n[报告生成] 可视化报告已生成")
        except Exception as e:
            print(f"\n[报告生成] 警告: 生成可视化报告时出错: {e}")
    
    def _print_summary(self, report: Dict) -> None:
        print("\n" + "="*70)
        print("自适应调参结果摘要")
        print("="*70)
        print(f"数据集: {report['metadata']['dataset']}")
        print(f"算法: {report['metadata']['algorithm']}")
        print(f"测试参数组合数: {report['metadata']['total_param_combinations']}")
        print("-"*70)
        print("最佳结果:")
        best = report['best_result']
        print(f"  准确率: {best['accuracy']:.4f}")
        print(f"  最佳轮次: {best['round']}")
        print(f"  隐私预算: {best['privacy_budget']:.4f}")
        print(f"  参数: μ={best['params']['mu']:.3f}, "
              f"C={best['params']['clip_norm']:.2f}, "
              f"σ={best['params']['noise_multiplier']:.2f}")
        print("-"*70)
        print("所有结果 (按准确率排序):")
        for i, r in enumerate(report['all_results'][:5], 1):
            print(f"  #{i}: 准确率={r['accuracy']:.4f}, "
                  f"ε={r['privacy_budget']:.2f}, "
                  f"早停={'是' if r['terminated_early'] else '否'}")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description="自适应差分隐私调参系统")
    
    parser.add_argument('-data', "--dataset", type=str, default="MNIST", 
                        help="数据集名称")
    parser.add_argument('-algo', "--algorithm", type=str, default="DPProx",
                        choices=["DPProx", "DPProxNative"],
                        help="DP算法")
    parser.add_argument('-gr', "--global_rounds", type=int, default=50,
                        help="每组参数的最大训练轮数")
    parser.add_argument('-patience', "--patience", type=int, default=3,
                        help="连续多少轮无提升时终止")
    parser.add_argument('-min_delta', "--min_delta", type=float, default=0.001,
                        help="最小提升阈值")
    parser.add_argument('-max_params', "--max_param_combinations", type=int, default=10,
                        help="最大参数组合数")
    parser.add_argument('-target_acc', "--target_accuracy", type=float, default=0.85,
                        help="目标准确率，达到后停止")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        help="运行设备")
    parser.add_argument('-did', "--device_id", type=str, default="0",
                        help="GPU设备ID")
    parser.add_argument('-ncl', "--num_classes", type=int, default=10,
                        help="类别数量")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="客户端数量")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10,
                        help="批大小")
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="本地学习率")
    parser.add_argument('-ls', "--local_epochs", type=int, default=1,
                        help="本地训练轮数")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="客户端参与比例")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="随机客户端参与比例")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="评估间隔")
    parser.add_argument('-fs', "--few_shot", type=int, default=0,
                        help="少样本学习参数")
    parser.add_argument('-alpha', "--alpha", type=float, default=0.1,
                        help="Dirichlet分布参数")
    
    parser.add_argument('-go', "--goal", type=str, default="test", help="实验目标")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False, help="学习率衰减")
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99, help="学习率衰减系数")
    parser.add_argument('-tc', "--top_cnt", type=int, default=100, help="自动停止参数")
    parser.add_argument('-pv', "--prev", type=int, default=0, help="之前的运行次数")
    parser.add_argument('-t', "--times", type=int, default=1, help="运行次数")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items', help="保存文件夹名称")
    parser.add_argument('-ab', "--auto_break", type=bool, default=False, help="自动停止")
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False, help="DLG评估")
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100, help="DLG评估间隔")
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2, help="每个客户端批次数")
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0, help="新客户端数量")
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0, help="新客户端微调轮数")
    parser.add_argument('-fd', "--feature_dim", type=int, default=512, help="特征维度")
    parser.add_argument('-vs', "--vocab_size", type=int, default=80, help="词汇表大小")
    parser.add_argument('-ml', "--max_len", type=int, default=200, help="最大序列长度")
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0, help="客户端掉线率")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0, help="慢客户端比例")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0, help="慢发送客户端比例")
    parser.add_argument('-ts', "--time_select", type=bool, default=False, help="时间选择")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000, help="时间阈值")
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda不可用，切换到cpu\n")
        args.device = "cpu"
    
    tuner = AdaptiveDPTuner(args)
    
    final_report = tuner.run_adaptive_tuning()
    
    print("\n自适应调参完成！")


if __name__ == "__main__":
    main()
