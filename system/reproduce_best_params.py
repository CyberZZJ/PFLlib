#!/usr/bin/env python
"""
复现最佳参数组合结果
参数配置:
- μ (mu) = 0.01
- C (clip_norm) = 0.2
- σ (noise_multiplier) = 0.5
- ε (epsilon) = 4.0
- 预期准确率: ~76.50%
- 隐私预算: ~1.94
"""
import os
import sys
import copy
import torch
import argparse
import time
import numpy as np
from datetime import datetime
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.simplefilter("ignore")
torch.manual_seed(0)


def run_fixed_params():
    parser = argparse.ArgumentParser(description="复现最佳参数组合结果")
    
    parser.add_argument('-data', "--dataset", type=str, default="MNIST")
    parser.add_argument('-algo', "--algorithm", type=str, default="DPProx",
                        choices=["DPProx", "DPProxNative"])
    parser.add_argument('-gr', "--global_rounds", type=int, default=10)
    parser.add_argument('-dev', "--device", type=str, default="cuda")
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-ncl', "--num_classes", type=int, default=10)
    parser.add_argument('-nc', "--num_clients", type=int, default=20)
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005)
    parser.add_argument('-ls', "--local_epochs", type=int, default=1)
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0)
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False)
    parser.add_argument('-eg', "--eval_gap", type=int, default=1)
    parser.add_argument('-fs', "--few_shot", type=int, default=0)
    
    parser.add_argument('-go', "--goal", type=str, default="test")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-tc', "--top_cnt", type=int, default=100)
    parser.add_argument('-pv', "--prev", type=int, default=0)
    parser.add_argument('-t', "--times", type=int, default=1)
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)
    parser.add_argument('-fd', "--feature_dim", type=int, default=512)
    parser.add_argument('-vs', "--vocab_size", type=int, default=80)
    parser.add_argument('-ml', "--max_len", type=int, default=200)
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0)
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0)
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0)
    parser.add_argument('-ts', "--time_select", type=bool, default=False)
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000)
    
    args = parser.parse_args()
    
    args.mu = 0.01
    args.dp_clip_norm = 0.2
    args.dp_noise_multiplier = 0.5
    args.dp_epsilon = 4.0
    args.dp_delta = 1e-5
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda不可用，切换到cpu\n")
        args.device = "cpu"
    
    print("="*70)
    print("复现最佳参数组合结果")
    print("="*70)
    print(f"数据集: {args.dataset}")
    print(f"算法: {args.algorithm}")
    print(f"训练轮数: {args.global_rounds}")
    print("-"*70)
    print("固定参数配置:")
    print(f"  μ (近端项系数) = {args.mu}")
    print(f"  C (裁剪范数) = {args.dp_clip_norm}")
    print(f"  σ (噪声乘数) = {args.dp_noise_multiplier}")
    print(f"  ε (隐私预算) = {args.dp_epsilon}")
    print(f"  δ (隐私参数) = {args.dp_delta}")
    print("-"*70)
    print("预期结果:")
    print(f"  准确率: ~76.50%")
    print(f"  实际隐私预算: ~1.94")
    print("="*70)
    
    from flcore.servers.serverdpprox import DPProx
    from flcore.servers.serverdpproxnative import DPProxNative
    from flcore.trainmodel.models import FedAvgCNN
    
    if "MNIST" in args.dataset:
        args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
    elif "Cifar10" in args.dataset:
        args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
    else:
        args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
    
    if args.algorithm == "DPProx":
        server = DPProx(args, 0)
    elif args.algorithm == "DPProxNative":
        server = DPProxNative(args, 0)
    else:
        raise ValueError(f"不支持的算法: {args.algorithm}")
    
    print(f"\n服务器初始化完成，共 {len(server.clients)} 个客户端")
    print("\n开始训练...\n")
    
    best_accuracy = 0.0
    best_round = 0
    accuracy_history = []
    
    start_time = time.time()
    
    for round_num in range(1, args.global_rounds + 1):
        server.current_round = round_num - 1
        
        server.train_one_round()
        
        accuracy, auc = server.evaluate_public()
        accuracy_history.append(accuracy)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_round = round_num
        
        print(f"[轮次 {round_num:3d}] 准确率: {accuracy:.4f} | AUC: {auc:.4f} | "
              f"最佳: {best_accuracy:.4f} (轮次{best_round})")
    
    total_time = time.time() - start_time
    
    actual_epsilon = args.dp_clip_norm * np.sqrt(2 * np.log(1.25 / args.dp_delta)) / args.dp_noise_multiplier
    
    print("\n" + "="*70)
    print("训练完成!")
    print("="*70)
    print(f"最终准确率: {accuracy_history[-1]:.4f}")
    print(f"最佳准确率: {best_accuracy:.4f} (第{best_round}轮)")
    print(f"实际隐私预算 ε: {actual_epsilon:.4f}")
    print(f"总训练时间: {total_time:.1f}秒")
    print("-"*70)
    print("准确率历史:")
    for i, acc in enumerate(accuracy_history, 1):
        marker = " <-- 最佳" if acc == best_accuracy else ""
        print(f"  轮次{i:2d}: {acc:.4f}{marker}")
    print("="*70)
    
    output_dir = os.path.join(os.path.dirname(__file__), "reproduction_results")
    os.makedirs(output_dir, exist_ok=True)
    
    result_file = os.path.join(output_dir, 
                               f"reproduction_{args.dataset}_{args.algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("复现最佳参数组合结果\n")
        f.write("="*70 + "\n\n")
        f.write(f"数据集: {args.dataset}\n")
        f.write(f"算法: {args.algorithm}\n")
        f.write(f"训练轮数: {args.global_rounds}\n\n")
        f.write("参数配置:\n")
        f.write(f"  μ = {args.mu}\n")
        f.write(f"  C = {args.dp_clip_norm}\n")
        f.write(f"  σ = {args.dp_noise_multiplier}\n")
        f.write(f"  ε = {args.dp_epsilon}\n")
        f.write(f"  δ = {args.dp_delta}\n\n")
        f.write("结果:\n")
        f.write(f"  最佳准确率: {best_accuracy:.4f}\n")
        f.write(f"  最佳轮次: {best_round}\n")
        f.write(f"  实际隐私预算: {actual_epsilon:.4f}\n")
        f.write(f"  训练时间: {total_time:.1f}秒\n\n")
        f.write("准确率历史:\n")
        for i, acc in enumerate(accuracy_history, 1):
            f.write(f"  轮次{i}: {acc:.4f}\n")
    
    print(f"\n结果已保存到: {result_file}")
    
    return {
        "best_accuracy": best_accuracy,
        "best_round": best_round,
        "actual_epsilon": actual_epsilon,
        "accuracy_history": accuracy_history
    }


if __name__ == "__main__":
    run_fixed_params()
