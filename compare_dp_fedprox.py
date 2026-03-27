#!/usr/bin/env python
import copy
import torch
import argparse
import os
import sys
import time
import warnings
import numpy as np
import torchvision
import logging

# Add system directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'system'))

# Set dataset directory path
dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset')
os.environ['DATASET_DIR'] = dataset_dir

from flcore.servers.serverprox import FedProx
from flcore.servers.serverdpprox import DPProx
from flcore.servers.serverdpproxnative import DPProxNative

from flcore.trainmodel.models import *

from flcore.trainmodel.bilstm import *
from flcore.trainmodel.resnet import *
from flcore.trainmodel.alexnet import *
from flcore.trainmodel.mobilenet_v2 import *
from flcore.trainmodel.transformer import *

from utils.result_utils import average_data
from utils.mem_utils import MemReporter

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)

def run_algorithm(args, algorithm, run_id):
    print(f"\n============= Running {algorithm} - Run: {run_id} ============")
    print("Creating server and clients ...")
    start = time.time()

    # Generate model
    model_str = args.model
    if model_str == "CNN":
        args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
    else:
        raise NotImplementedError("Only CNN model is supported for this comparison")

    print(args.model)

    # Create server based on algorithm
    if algorithm == "FedProx":
        server = FedProx(args, run_id)
    elif algorithm == "DPProx":
        server = DPProx(args, run_id)
    elif algorithm == "DPProxNative":
        server = DPProxNative(args, run_id)
    else:
        raise NotImplementedError

    # Train and get results
    server.train()
    training_time = time.time() - start

    # Get accuracy (assuming server has this attribute or method)
    accuracy = getattr(server, 'global_test_accuracy', 'N/A')
    
    # Get privacy budget (only for DP algorithms)
    privacy_budget = 'N/A'
    if algorithm in ["DPProx", "DPProxNative"]:
        if hasattr(server, 'epsilon'):
            privacy_budget = server.epsilon

    return training_time, accuracy, privacy_budget

def main():
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # Common parameters
    parser.add_argument('-go', "--goal", type=str, default="test", help="实验目标")
    parser.add_argument('-dev', "--device", type=str, default="cuda", choices=["cpu", "cuda"], help="运行设备")
    parser.add_argument('-did', "--device_id", type=str, default="0", help="GPU设备ID")
    parser.add_argument('-data', "--dataset", type=str, default="MNIST", help="数据集名称")
    parser.add_argument('-ncl', "--num_classes", type=int, default=10, help="类别数量")
    parser.add_argument('-m', "--model", type=str, default="CNN", help="模型类型")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10, help="批大小")
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005, help="本地学习率")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False, help="是否使用学习率衰减")
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99, help="学习率衰减系数")
    parser.add_argument('-gr', "--global_rounds", type=int, default=100, help="全局训练轮数")
    parser.add_argument('-tc', "--top_cnt", type=int, default=100, help="自动停止参数")
    parser.add_argument('-ls', "--local_epochs", type=int, default=1, help="每个本地周期的更新步数")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedProx", help="联邦学习算法")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0, help="每轮参与训练的客户端比例")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False, help="每轮随机选择客户端比例")
    parser.add_argument('-nc', "--num_clients", type=int, default=20, help="客户端总数")
    parser.add_argument('-pv', "--prev", type=int, default=0, help="之前的运行次数")
    parser.add_argument('-t', "--times", type=int, default=1, help="运行次数")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1, help="评估间隔轮数")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='compare_dp_fedprox', help="保存结果的文件夹名称")
    parser.add_argument('-ab', "--auto_break", type=bool, default=False, help="是否自动停止")
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False, help="是否进行dlg评估")
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100, help="dlg评估间隔")
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2, help="每个客户端的批次数")
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0, help="新增客户端数量")
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0, help="新客户端微调轮数")
    parser.add_argument('-fd', "--feature_dim", type=int, default=512, help="特征维度")
    parser.add_argument('-vs', "--vocab_size", type=int, default=80, help="词汇表大小")
    parser.add_argument('-ml', "--max_len", type=int, default=200, help="最大序列长度")
    parser.add_argument('-fs', "--few_shot", type=int, default=0, help="少样本学习参数")
    # 实际部署相关参数
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0, help="训练过程中掉线的客户端比例")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0, help="本地训练时慢客户端的比例")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0, help="发送全局模型时慢客户端的比例")
    parser.add_argument('-ts', "--time_select", type=bool, default=False, help="是否根据时间成本分组和选择客户端")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000, help="丢弃慢客户端的时间阈值")
    # pFedMe / PerAvg / FedProx / FedAMP / FedPHP / GPFL / FedCAC
    parser.add_argument('-bt', "--beta", type=float, default=0.0)
    parser.add_argument('-lam', "--lamda", type=float, default=1.0, help="Regularization weight")
    
    # Dirichlet distribution parameters
    parser.add_argument('-alpha', "--dirichlet_alpha", type=float, default=0.1, help="Dirichlet分布的alpha参数")
    
    # FedProx parameters
    parser.add_argument('-mu', "--mu", type=float, default=0.01, help="FedProx正则化参数")
    
    # DP parameters
    parser.add_argument('-dp_clip', '--dp_clip_norm', type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument('-dp_noise', '--dp_noise_multiplier', type=float, default=1.0, help="噪声乘数")
    parser.add_argument('-dp_eps', '--dp_epsilon', type=float, default=10.0, help="隐私预算")
    parser.add_argument('-dp_delta', '--dp_delta', type=float, default=1e-5, help="隐私参数")
    
    # DP-Prox-Native parameters
    parser.add_argument('-lambda_priv', '--lambda_privacy', type=float, default=0.05, help="隐私正则项系数")
    parser.add_argument('-sup_noise', '--use_supplementary_noise', action='store_true', help="是否使用补充噪声")
    parser.add_argument('-sup_noise_scale', '--supplementary_noise_scale', type=float, default=0.001, help="补充噪声尺度")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)
    for arg in vars(args):
        print(arg, '=', getattr(args, arg))
    print("=" * 50)

    # Algorithms to compare
    algorithms = ["FedProx", "DPProx", "DPProxNative"]
    results = {algo: {"times": [], "accuracies": [], "privacy_budgets": []} for algo in algorithms}

    # Run each algorithm multiple times
    for i in range(args.times):
        for algo in algorithms:
            training_time, accuracy, privacy_budget = run_algorithm(args, algo, i)
            results[algo]["times"].append(training_time)
            results[algo]["accuracies"].append(accuracy)
            results[algo]["privacy_budgets"].append(privacy_budget)

    # Calculate averages and print results
    print("\n" + "=" * 70)
    print("Comparison Results")
    print("=" * 70)
    print(f"{'Algorithm':<15} {'Avg Time (s)':<15} {'Avg Accuracy':<15} {'Avg Privacy Budget':<15}")
    print("-" * 70)

    for algo in algorithms:
        avg_time = np.average(results[algo]["times"])
        avg_accuracy = np.average(results[algo]["accuracies"]) if results[algo]["accuracies"][0] != 'N/A' else 'N/A'
        avg_privacy = np.average(results[algo]["privacy_budgets"]) if results[algo]["privacy_budgets"][0] != 'N/A' else 'N/A'
        
        print(f"{algo:<15} {avg_time:<15.2f} {avg_accuracy:<15} {avg_privacy:<15}")

    print("=" * 70)
    print(f"\nTotal experiment time: {round(time.time() - total_start, 2)}s.")
    print("All done!")

if __name__ == "__main__":
    main()
