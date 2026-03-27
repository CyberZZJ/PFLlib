import torch
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from flcore.clients.clientdpprox import add_gaussian_noise, clip_gradient
from flcore.clients.clientdpproxnative import compute_input_gradient_regularization, compute_privacy_budget

# 验证改进后的 DP-Prox 噪声分布
def validate_improved_dpprox():
    print("=== 验证改进后的 DP-Prox 噪声分布 ===")
    
    # 生成噪声样本
    num_samples = 10000
    noise_scale = 1.0  # 增加的噪声尺度
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
    stat, p_value = stats.shapiro(noise[:5000])
    
    print(f"样本数量: {num_samples}")
    print(f"均值: {mean:.6f}")
    print(f"标准差: {std:.6f}")
    print(f"偏度: {skewness:.6f}")
    print(f"峰度: {kurtosis:.6f}")
    print(f"Shapiro-Wilk 检验统计量: {stat:.6f}")
    print(f"p值: {p_value:.6f}")
    print(f"是否符合正态分布: {'是' if p_value > 0.05 else '否'}")
    
    # 计算隐私预算
    epsilon = compute_privacy_budget(clip_norm, noise_scale)
    print(f"隐私预算 ε: {epsilon:.4f}")
    print(f"隐私保护强度: {'强' if epsilon < 10 else '中' if epsilon < 20 else '弱'}")
    
    return noise, epsilon

# 验证改进后的 DP-Native-FedProx
def validate_improved_dpnative():
    print("\n=== 验证改进后的 DP-Native-FedProx ===")
    
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
    
    # 计算补充噪声的隐私预算
    clip_norm = 1.0
    noise_scale = 1.0  # 使用与客户端一致的噪声尺度
    epsilon = compute_privacy_budget(clip_norm, noise_scale)
    print(f"补充噪声隐私预算 ε: {epsilon:.4f}")
    print(f"隐私保护强度: {'强' if epsilon < 10 else '中' if epsilon < 20 else '弱'}")
    
    # 验证梯度裁剪
    print("\n验证梯度裁剪:")
    model.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    
    # 计算裁剪前的梯度范数
    grad_norms = []
    for param in model.parameters():
        grad_norms.append(torch.norm(param.grad).item())
    total_norm = torch.norm(torch.cat([p.grad.view(-1) for p in model.parameters()])).item()
    print(f"裁剪前梯度总范数: {total_norm:.4f}")
    
    # 应用梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    
    # 计算裁剪后的梯度范数
    clipped_grad_norms = []
    for param in model.parameters():
        clipped_grad_norms.append(torch.norm(param.grad).item())
    clipped_total_norm = torch.norm(torch.cat([p.grad.view(-1) for p in model.parameters()])).item()
    print(f"裁剪后梯度总范数: {clipped_total_norm:.4f}")
    print(f"是否符合裁剪阈值: {'是' if clipped_total_norm <= clip_norm else '否'}")
    
    return privacy_loss.item(), grad_norm, epsilon

if __name__ == "__main__":
    print("差分隐私方案改进验证")
    print("=" * 70)
    
    # 验证改进后的 DP-Prox
    dpprox_noise, dpprox_epsilon = validate_improved_dpprox()
    
    # 验证改进后的 DP-Native-FedProx
    privacy_loss, grad_norm, dpnative_epsilon = validate_improved_dpnative()
    
    print("\n" + "=" * 70)
    print("改进验证完成！")
    print("\n改进效果总结:")
    print(f"1. DP-Prox 隐私预算: ε = {dpprox_epsilon:.4f}")
    print(f"2. DP-Native-FedProx 隐私预算: ε = {dpnative_epsilon:.4f}")
    print(f"3. DP-Native-FedProx 输入梯度范数: {grad_norm:.4f}")
    print(f"4. DP-Native-FedProx 输入梯度正则项: {privacy_loss:.4f}")
    
    # 评估改进效果
    print("\n改进评估:")
    if dpprox_epsilon < 10:
        print("✓ DP-Prox 隐私保护强度: 强")
    else:
        print("⚠ DP-Prox 隐私保护强度: 中/弱")
    
    if dpnative_epsilon < 10:
        print("✓ DP-Native-FedProx 隐私保护强度: 强")
    else:
        print("⚠ DP-Native-FedProx 隐私保护强度: 中/弱")
