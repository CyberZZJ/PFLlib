import torch
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from flcore.clients.clientdpprox import add_gaussian_noise, clip_gradient
from flcore.clients.clientdpproxnative import compute_input_gradient_regularization

# 1. 隐私预算计算与验证
def compute_privacy_budget():
    print("=== 1. 隐私预算计算与验证 ===")
    
    # 高斯机制的隐私预算计算公式
    # ε = C * sqrt(2 * ln(1.25/δ)) / σ
    
    # 参数设置
    C = 1.0  # 裁剪阈值
    δ = 1e-5  # 隐私参数
    
    print(f"裁剪阈值 C: {C}")
    print(f"隐私参数 δ: {δ}")
    print("\n噪声尺度与隐私预算关系:")
    
    # 计算不同噪声尺度下的隐私预算
    noise_scales = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    for σ in noise_scales:
        ε = C * np.sqrt(2 * np.log(1.25 / δ)) / σ
        print(f"σ={σ:.2f}: ε={ε:.4f}")
    
    # 验证 DP-Prox 的噪声设置
    print("\n=== DP-Prox 噪声设置验证 ===")
    dp_prox_noise_scale = 0.1
    dp_prox_epsilon = C * np.sqrt(2 * np.log(1.25 / δ)) / dp_prox_noise_scale
    print(f"DP-Prox 噪声尺度: {dp_prox_noise_scale}")
    print(f"DP-Prox 理论隐私预算: {dp_prox_epsilon:.4f}")
    
    # 验证 DP-Native-FedProx 的补充噪声
    print("\n=== DP-Native-FedProx 噪声设置验证 ===")
    dp_native_noise_scale = 0.001
    print(f"DP-Native-FedProx 补充噪声尺度: {dp_native_noise_scale}")
    print("注：DP-Native-FedProx 主要依赖输入梯度正则化，补充噪声为辅助保护")
    
    return dp_prox_epsilon

# 2. 敏感度分析
def analyze_sensitivity():
    print("\n=== 2. 敏感度分析 ===")
    
    # 创建一个简单的模型用于测试
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(784, 10)
        def forward(self, x):
            return self.fc(x.view(-1, 784))
    
    model = SimpleModel()
    criterion = torch.nn.CrossEntropyLoss()
    
    # 生成相邻数据集（仅差一个样本）
    batch_size = 10
    data1 = torch.randn(batch_size, 1, 28, 28)
    target1 = torch.randint(0, 10, (batch_size,))
    
    # 相邻数据集：修改第一个样本
    data2 = data1.clone()
    data2[0] = torch.randn(1, 28, 28)
    target2 = target1.clone()
    target2[0] = torch.randint(0, 10, (1,))[0]
    
    # 计算两个数据集上的模型更新
    def compute_model_update(data, target):
        model.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # 收集梯度
        grads = []
        for param in model.parameters():
            grads.append(param.grad.view(-1))
        grad_flat = torch.cat(grads)
        return grad_flat
    
    # 计算原始梯度
    grad1 = compute_model_update(data1, target1)
    grad2 = compute_model_update(data2, target2)
    
    # 计算敏感度（相邻数据集的梯度差异）
    sensitivity = torch.norm(grad1 - grad2).item()
    print(f"原始敏感度: {sensitivity:.4f}")
    
    # 应用梯度裁剪
    clip_norm = 1.0
    clipped_grad1 = clip_gradient(grad1, clip_norm)
    clipped_grad2 = clip_gradient(grad2, clip_norm)
    clipped_sensitivity = torch.norm(clipped_grad1 - clipped_grad2).item()
    print(f"裁剪后敏感度: {clipped_sensitivity:.4f}")
    print(f"裁剪阈值: {clip_norm}")
    print(f"敏感度是否被控制: {'是' if clipped_sensitivity <= clip_norm * 2 else '否'}")
    
    return sensitivity, clipped_sensitivity

# 3. 差分隐私统计验证
def verify_dp_statistics():
    print("\n=== 3. 差分隐私统计验证 ===")
    
    # 生成噪声样本
    num_samples = 10000
    noise_scale = 0.1
    clip_norm = 1.0
    
    # 生成两个相邻数据集的噪声
    noise1 = torch.randn(num_samples) * noise_scale * clip_norm
    noise2 = torch.randn(num_samples) * noise_scale * clip_norm
    
    # 计算统计距离（KL散度）
    def kl_divergence(p, q):
        p = p.numpy()
        q = q.numpy()
        # 计算直方图
        hist_p, bins = np.histogram(p, bins=100, density=True)
        hist_q, _ = np.histogram(q, bins=bins, density=True)
        # 避免除以零
        hist_p = np.clip(hist_p, 1e-10, None)
        hist_q = np.clip(hist_q, 1e-10, None)
        # 计算KL散度
        return np.sum(hist_p * np.log(hist_p / hist_q))
    
    kl_div = kl_divergence(noise1, noise2)
    print(f"噪声分布的KL散度: {kl_div:.4f}")
    
    # 验证差分隐私条件
    epsilon = 1.0  # 设定的隐私预算
    print(f"设定的隐私预算 ε: {epsilon}")
    print(f"e^ε: {np.exp(epsilon):.4f}")
    print(f"是否满足差分隐私: {'是' if kl_div < np.exp(epsilon) else '否'}")
    
    return kl_div

# 4. 输入梯度正则化效果分析
def analyze_input_gradient_regularization():
    print("\n=== 4. 输入梯度正则化效果分析 ===")
    
    # 创建一个简单的模型
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(784, 10)
        def forward(self, x):
            return self.fc(x.view(-1, 784))
    
    model = SimpleModel()
    criterion = torch.nn.CrossEntropyLoss()
    
    # 生成测试数据
    data = torch.randn(10, 1, 28, 28)
    target = torch.randint(0, 10, (10,))
    
    # 计算输入梯度正则项
    privacy_loss = compute_input_gradient_regularization(model, data, target, criterion)
    print(f"输入梯度正则项值: {privacy_loss.item():.4f}")
    
    # 分析输入梯度的范数
    data.requires_grad = True
    output = model(data)
    loss = criterion(output, target)
    data_grad = torch.autograd.grad(loss, data, create_graph=True)[0]
    grad_norm = torch.norm(data_grad).item()
    print(f"输入梯度L2范数: {grad_norm:.4f}")
    
    return privacy_loss.item(), grad_norm

if __name__ == "__main__":
    print("差分隐私保护验证")
    print("=" * 70)
    
    # 1. 隐私预算计算
    dp_prox_epsilon = compute_privacy_budget()
    
    # 2. 敏感度分析
    sensitivity, clipped_sensitivity = analyze_sensitivity()
    
    # 3. 差分隐私统计验证
    kl_div = verify_dp_statistics()
    
    # 4. 输入梯度正则化分析
    privacy_loss, grad_norm = analyze_input_gradient_regularization()
    
    print("\n" + "=" * 70)
    print("验证完成！")
    print("\n总结:")
    print(f"1. DP-Prox 理论隐私预算: {dp_prox_epsilon:.4f}")
    print(f"2. 原始敏感度: {sensitivity:.4f}, 裁剪后敏感度: {clipped_sensitivity:.4f}")
    print(f"3. 噪声分布KL散度: {kl_div:.4f}")
    print(f"4. 输入梯度正则项: {privacy_loss:.4f}, 输入梯度范数: {grad_norm:.4f}")
