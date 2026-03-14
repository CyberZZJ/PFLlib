import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_utils import read_client_data

# 加载公共验证集（第一个客户端的测试数据）
dataset = 'FashionMNIST'
client_id = 0
public_test_data = read_client_data(dataset, client_id, is_train=False, few_shot=0)

print(f"公共验证集大小: {len(public_test_data)}")
print(f"数据类型: {type(public_test_data)}")
print(f"第一个数据样本类型: {type(public_test_data[0])}")
print(f"第一个数据样本形状: 输入: {public_test_data[0][0].shape}, 标签: {public_test_data[0][1]}")
print(f"标签类型: {type(public_test_data[0][1])}")

# 统计标签分布
labels = [item[1].item() for item in public_test_data]
unique_labels = set(labels)
print(f"\n标签类别: {sorted(unique_labels)}")
print(f"标签数量: {len(unique_labels)}")

# 统计每个标签的样本数
label_counts = {}
for label in labels:
    if label not in label_counts:
        label_counts[label] = 0
    label_counts[label] += 1

print("\n每个标签的样本数:")
for label, count in sorted(label_counts.items()):
    print(f"标签 {label}: {count} 个样本")
