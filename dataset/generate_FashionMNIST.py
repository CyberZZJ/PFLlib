import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file


random.seed(1)
np.random.seed(1)
num_clients = 30
dir_path = "FashionMNIST/"
public_samples_per_class = 150  # 公共验证集每个类别的样本数


# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"
    public_path = dir_path + "public/"

    # 如果数据集已存在，直接返回
    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    # Get FashionMNIST data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    trainset = torchvision.datasets.FashionMNIST(
        root=dir_path+"rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(
        root=dir_path+"rawdata", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    # 创建公共验证集：每个类别取 150 个样本（从测试集中取）
    print(f"\nCreating public validation set with {public_samples_per_class} samples per class...")
    public_images = []
    public_labels = []
    public_indices = []  # 记录被选为公共验证集的样本索引
    
    # 只从测试集中选择公共验证集样本
    test_start_idx = len(trainset.data)
    
    for class_idx in range(num_classes):
        # 找到测试集中该类别的所有索引
        class_test_indices = np.where(dataset_label[test_start_idx:] == class_idx)[0] + test_start_idx
        
        # 随机选择 public_samples_per_class 个样本
        if len(class_test_indices) >= public_samples_per_class:
            selected_indices = np.random.choice(class_test_indices, size=public_samples_per_class, replace=False)
        else:
            # 如果测试集样本不足，从训练集补充
            remaining_from_train = public_samples_per_class - len(class_test_indices)
            train_class_indices = np.where(dataset_label[:test_start_idx] == class_idx)[0]
            selected_train = np.random.choice(train_class_indices, size=remaining_from_train, replace=False)
            selected_indices = np.concatenate([class_test_indices, selected_train])
        
        public_images.append(dataset_image[selected_indices])
        public_labels.append(dataset_label[selected_indices])
        public_indices.extend(selected_indices)
        print(f"Class {class_idx}: selected {len(selected_indices)} samples")
    
    public_images = np.concatenate(public_images, axis=0)
    public_labels = np.concatenate(public_labels, axis=0)
    
    # 打乱公共验证集
    shuffle_indices = np.random.permutation(len(public_labels))
    public_images = public_images[shuffle_indices]
    public_labels = public_labels[shuffle_indices]
    
    print(f"Public validation set created with {len(public_labels)} samples total")
    
    # 从原始数据集中移除公共验证集的样本
    public_indices = np.array(public_indices)
    all_indices = np.arange(len(dataset_label))
    remaining_indices = np.setdiff1d(all_indices, public_indices)
    
    dataset_image_remaining = dataset_image[remaining_indices]
    dataset_label_remaining = dataset_label[remaining_indices]
    print(f"Remaining dataset size: {len(dataset_label_remaining)} samples\n")

    X, y, statistic = separate_data((dataset_image_remaining, dataset_label_remaining), num_clients, num_classes, 
                                    niid, balance, partition, class_per_client=2)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition)
    
    # 保存公共验证集
    save_public_file(public_path, public_images, public_labels)


def save_public_file(public_path, images, labels):
    """保存公共验证集"""
    if not os.path.exists(public_path):
        os.makedirs(public_path)
    
    # 保存为单个 npz 文件
    public_file = public_path + "public_data.npz"
    np.savez_compressed(public_file, x=images, y=labels)
    print(f"\nPublic validation set saved to {public_file}")
    print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")
    
    # 统计标签分布
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\nPublic validation set label distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  Class {label}: {count} samples")


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_dataset(dir_path, num_clients, niid, balance, partition)
