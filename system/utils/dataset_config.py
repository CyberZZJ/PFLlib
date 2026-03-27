#!/usr/bin/env python
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import json


@dataclass
class DatasetConfig:
    dataset_name: str
    epsilon_range: Tuple[float, float]
    delta: float
    clip_norm_range: Tuple[float, float]
    noise_multiplier_range: Tuple[float, float]
    mu_range: Tuple[float, float]
    expected_accuracy_loss: float
    complexity_score: float
    sensitivity_score: float
    description: str


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "MNIST": DatasetConfig(
        dataset_name="MNIST",
        epsilon_range=(4.0, 5.0),
        delta=1e-5,
        clip_norm_range=(0.2, 1.0),
        noise_multiplier_range=(0.5, 1.5),
        mu_range=(0.01, 0.2),
        expected_accuracy_loss=0.0061,
        complexity_score=0.3,
        sensitivity_score=0.4,
        description="MNIST数据集对差分隐私容忍度较高，可使用较小隐私预算"
    ),
    "FashionMNIST": DatasetConfig(
        dataset_name="FashionMNIST",
        epsilon_range=(5.0, 10.0),
        delta=1e-5,
        clip_norm_range=(0.5, 2.0),
        noise_multiplier_range=(0.3, 1.0),
        mu_range=(0.05, 0.3),
        expected_accuracy_loss=0.0259,
        complexity_score=0.5,
        sensitivity_score=0.6,
        description="FashionMNIST对噪声更敏感，需要较大隐私预算"
    ),
    "Cifar10": DatasetConfig(
        dataset_name="Cifar10",
        epsilon_range=(8.0, 12.0),
        delta=1e-5,
        clip_norm_range=(0.5, 2.0),
        noise_multiplier_range=(0.3, 0.8),
        mu_range=(0.05, 0.3),
        expected_accuracy_loss=0.05,
        complexity_score=0.7,
        sensitivity_score=0.8,
        description="Cifar10复杂度较高，需要更多隐私预算"
    ),
}


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    if dataset_name not in DATASET_CONFIGS:
        print(f"警告: 数据集 {dataset_name} 未在配置中找到，使用MNIST作为默认配置")
        return DATASET_CONFIGS["MNIST"]
    return DATASET_CONFIGS[dataset_name]


def analyze_dataset_characteristics(dataset_name: str) -> Dict:
    config = get_dataset_config(dataset_name)
    return {
        "complexity": config.complexity_score,
        "sensitivity": config.sensitivity_score,
        "recommended_epsilon_range": config.epsilon_range,
        "recommended_clip_norm_range": config.clip_norm_range,
        "recommended_noise_range": config.noise_multiplier_range,
    }


def recommend_initial_params(dataset_name: str, algorithm: str = "DPProx") -> Dict:
    config = get_dataset_config(dataset_name)
    
    epsilon_min, epsilon_max = config.epsilon_range
    clip_min, clip_max = config.clip_norm_range
    noise_min, noise_max = config.noise_multiplier_range
    mu_min, mu_max = config.mu_range
    
    recommended = {
        "epsilon": (epsilon_min + epsilon_max) / 2,
        "delta": config.delta,
        "clip_norm": (clip_min + clip_max) / 2,
        "noise_multiplier": (noise_min + noise_max) / 2,
        "mu": (mu_min + mu_max) / 2,
    }
    
    if algorithm == "DPProxNative":
        recommended["lambda_privacy"] = 0.05
        recommended["supplementary_noise_scale"] = 0.001
    
    return recommended


def generate_param_grid(dataset_name: str, algorithm: str = "DPProx", 
                        num_samples: int = 5) -> List[Dict]:
    config = get_dataset_config(dataset_name)
    param_combinations = []
    
    epsilon_values = np.linspace(config.epsilon_range[0], config.epsilon_range[1], num_samples)
    clip_values = np.linspace(config.clip_norm_range[0], config.clip_norm_range[1], num_samples)
    noise_values = np.linspace(config.noise_multiplier_range[0], config.noise_multiplier_range[1], num_samples)
    mu_values = np.linspace(config.mu_range[0], config.mu_range[1], 3)
    
    for eps in epsilon_values:
        for clip in clip_values:
            for noise in noise_values:
                for mu in mu_values:
                    params = {
                        "epsilon": round(eps, 3),
                        "delta": config.delta,
                        "clip_norm": round(clip, 3),
                        "noise_multiplier": round(noise, 3),
                        "mu": round(mu, 3),
                    }
                    if algorithm == "DPProxNative":
                        params["lambda_privacy"] = 0.05
                        params["supplementary_noise_scale"] = 0.001
                    param_combinations.append(params)
    
    return param_combinations


def calculate_privacy_budget(clip_norm: float, noise_multiplier: float, 
                             delta: float = 1e-5) -> float:
    if noise_multiplier <= 0:
        return float('inf')
    
    epsilon = clip_norm * np.sqrt(2 * np.log(1.25 / delta)) / noise_multiplier
    return round(epsilon, 4)


def validate_privacy_constraint(epsilon: float, epsilon_max: float) -> bool:
    return epsilon <= epsilon_max


def get_param_search_priority(dataset_name: str) -> List[str]:
    config = get_dataset_config(dataset_name)
    
    if config.sensitivity_score > 0.5:
        return ["noise_multiplier", "clip_norm", "mu", "epsilon"]
    else:
        return ["epsilon", "mu", "clip_norm", "noise_multiplier"]


def export_config_to_json(dataset_name: str, filepath: str) -> None:
    config = get_dataset_config(dataset_name)
    config_dict = {
        "dataset_name": config.dataset_name,
        "epsilon_range": config.epsilon_range,
        "delta": config.delta,
        "clip_norm_range": config.clip_norm_range,
        "noise_multiplier_range": config.noise_multiplier_range,
        "mu_range": config.mu_range,
        "expected_accuracy_loss": config.expected_accuracy_loss,
        "complexity_score": config.complexity_score,
        "sensitivity_score": config.sensitivity_score,
        "description": config.description,
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    print(f"配置已导出到: {filepath}")


def print_dataset_info(dataset_name: str) -> None:
    config = get_dataset_config(dataset_name)
    print(f"\n{'='*60}")
    print(f"数据集: {config.dataset_name}")
    print(f"{'='*60}")
    print(f"描述: {config.description}")
    print(f"隐私预算范围 ε: {config.epsilon_range}")
    print(f"隐私参数 δ: {config.delta}")
    print(f"裁剪范数范围 C: {config.clip_norm_range}")
    print(f"噪声乘数范围 σ: {config.noise_multiplier_range}")
    print(f"近端项系数范围 μ: {config.mu_range}")
    print(f"预期准确率损失: {config.expected_accuracy_loss * 100:.2f}%")
    print(f"复杂度评分: {config.complexity_score}")
    print(f"敏感度评分: {config.sensitivity_score}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print_dataset_info("MNIST")
    print_dataset_info("FashionMNIST")
    
    recommended = recommend_initial_params("MNIST", "DPProx")
    print(f"MNIST推荐参数: {recommended}")
    
    grid = generate_param_grid("MNIST", "DPProx", num_samples=2)
    print(f"\n生成的参数网格 (共{len(grid)}组):")
    for i, params in enumerate(grid[:5]):
        print(f"  组合{i+1}: {params}")
