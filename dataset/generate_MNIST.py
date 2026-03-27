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
num_clients = 20
dir_path = "MNIST/"


# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"
    public_path = dir_path + "public/"

    # Always create public dataset, even if other files exist
    # if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
    #     return

    # # FIX HTTP Error 403: Forbidden
    # from six.moves import urllib
    # opener = urllib.request.build_opener()
    # opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    # urllib.request.install_opener(opener)

    # Get MNIST data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    trainset = torchvision.datasets.MNIST(
        root=dir_path+"rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(
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

    # Create public dataset with 100 samples per class
    public_image = []
    public_label = []
    for i in range(num_classes):
        idx = np.where(dataset_label == i)[0]
        np.random.shuffle(idx)
        selected_idx = idx[:100]  # Take 100 samples per class
        public_image.extend(dataset_image[selected_idx])
        public_label.extend(dataset_label[selected_idx])
        # Remove selected samples from original dataset
        dataset_image = np.delete(dataset_image, selected_idx, axis=0)
        dataset_label = np.delete(dataset_label, selected_idx, axis=0)

    public_image = np.array(public_image)
    public_label = np.array(public_label)

    # Save public dataset
    if not os.path.exists(public_path):
        os.makedirs(public_path, exist_ok=True)
    public_data = {'x': public_image, 'y': public_label}
    public_file_path = os.path.join(public_path, 'public_data.npz')
    with open(public_file_path, 'wb') as f:
        np.savez_compressed(f, data=public_data)
    print(f"Public dataset created at {public_file_path} with {len(public_image)} samples (100 per class)")

    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition, class_per_client=2)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition)


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_dataset(dir_path, num_clients, niid, balance, partition)