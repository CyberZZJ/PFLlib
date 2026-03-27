import torch
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# 导入噪声生成函数
from flcore.clients.clientdpprox import add_gaussian_noise

# 验证 DP-Prox 噪声分布
def validate_dpprox_noise():
    print("=== 验证 DP-Prox 噪声分布 ===")
    
    # 生成噪声样本
    num_samples = 10000
    noise_scale = 0.1
    clip_norm = 1.0
    
    # 创建一个测试张量
    test_update = torch.zeros(num_samples)
    
    # 生成带噪声的更新
    noisy_update = add_gaussian_noise(test_update, noise_scale, clip_norm)
    
    # 提取噪声部分
    noise = noisy_update.numpy()
    
    # 计算统计指标
    mean = np.mean(noise)
    std = np.std(noise)
    skewness = stats.skew(noise)
    kurtosis = stats.kurtosis(noise)
    
    # 正态性检验
    stat, p_value = stats.shapiro(noise[:5000])  # 限制样本数量以加速检验
    
    print(f"样本数量: {num_samples}")
    print(f"均值: {mean:.6f}")
    print(f"标准差: {std:.6f}")
    print(f"偏度: {skewness:.6f}")
    print(f"峰度: {kurtosis:.6f}")
    print(f"Shapiro-Wilk 检验统计量: {stat:.6f}")
    print(f"p值: {p_value:.6f}")
    print(f"是否符合正态分布: {'是' if p_value > 0.05 else '否'}")
    
    # 理论标准差
    theoretical_std = noise_scale * clip_norm
    print(f"理论标准差: {theoretical_std:.6f}")
    print(f"实际标准差与理论值的误差: {abs(std - theoretical_std):.6f}")
    
    return noise

# 验证 DP-Native-FedProx 噪声分布
def validate_dpnative_noise():
    print("\n=== 验证 DP-Native-FedProx 补充噪声分布 ===")
    
    # 生成噪声样本
    num_samples = 10000
    supplementary_noise_scale = 0.001
    
    # 生成噪声
    noise = torch.randn(num_samples) * supplementary_noise_scale
    noise = noise.numpy()
    
    # 计算统计指标
    mean = np.mean(noise)
    std = np.std(noise)
    skewness = stats.skew(noise)
    kurtosis = stats.kurtosis(noise)
    
    # 正态性检验
    stat, p_value = stats.shapiro(noise[:5000])
    
    print(f"样本数量: {num_samples}")
    print(f"均值: {mean:.6f}")
    print(f"标准差: {std:.6f}")
    print(f"偏度: {skewness:.6f}")
    print(f"峰度: {kurtosis:.6f}")
    print(f"Shapiro-Wilk 检验统计量: {stat:.6f}")
    print(f"p值: {p_value:.6f}")
    print(f"是否符合正态分布: {'是' if p_value > 0.05 else '否'}")
    
    # 理论标准差
    theoretical_std = supplementary_noise_scale
    print(f"理论标准差: {theoretical_std:.6f}")
    print(f"实际标准差与理论值的误差: {abs(std - theoretical_std):.6f}")
    
    return noise

# 验证差分隐私数学条件
def validate_dp_conditions():
    print("\n=== 验证差分隐私数学条件 ===")
    
    # 高斯机制的隐私预算计算公式
    # ε = C * sqrt(2 * ln(1.25/δ)) / σ
    
    C = 1.0  # 裁剪阈值
    δ = 1e-5  # 隐私参数
    σ = 0.1  # 噪声尺度
    
    # 计算理论隐私预算
    theoretical_epsilon = C * np.sqrt(2 * np.log(1.25 / δ)) / σ
    print(f"裁剪阈值 C: {C}")
    print(f"隐私参数 δ: {δ}")
    print(f"噪声尺度 σ: {σ}")
    print(f"理论隐私预算 ε: {theoretical_epsilon:.4f}")
    
    # 验证噪声尺度与隐私预算的关系
    print("\n噪声尺度与隐私预算关系:")
    for sigma in [0.05, 0.1, 0.2, 0.5, 1.0]:
        epsilon = C * np.sqrt(2 * np.log(1.25 / δ)) / sigma
        print(f"σ={sigma:.2f}: ε={epsilon:.4f}")

if __name__ == "__main__":
    print("差分隐私噪声分布验证")
    print("=" * 60)
    
    # 验证 DP-Prox 噪声
    dpprox_noise = validate_dpprox_noise()
    
    # 验证 DP-Native-FedProx 噪声
    dpnative_noise = validate_dpnative_noise()
    
    # 验证差分隐私数学条件
    validate_dp_conditions()
    
    print("\n" + "=" * 60)
    print("验证完成！")
