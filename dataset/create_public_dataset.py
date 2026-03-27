import numpy as np
import os
import torch
import torchvision
import torchvision.transforms as transforms

# Set random seeds for reproducibility
torch.manual_seed(1)
np.random.seed(1)

# Directory paths
dir_path = os.path.dirname(os.path.abspath(__file__)) + "/MNIST/"
public_path = os.path.join(dir_path, "public/")

# Create public directory if it doesn't exist
if not os.path.exists(public_path):
    os.makedirs(public_path, exist_ok=True)
    print(f"Created directory: {public_path}")

# Get MNIST data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

trainset = torchvision.datasets.MNIST(
    root=os.path.join(dir_path, "rawdata"), train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(
    root=os.path.join(dir_path, "rawdata"), train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=len(trainset.data), shuffle=False)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=len(testset.data), shuffle=False)

# Extract data
for _, train_data in enumerate(trainloader, 0):
    trainset.data, trainset.targets = train_data
for _, test_data in enumerate(testloader, 0):
    testset.data, testset.targets = test_data

# Combine train and test sets
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
    print(f"Class {i}: selected {len(selected_idx)} samples")

public_image = np.array(public_image)
public_label = np.array(public_label)

# Save public dataset
public_data = {'x': public_image, 'y': public_label}
public_file_path = os.path.join(public_path, 'public_data.npz')
with open(public_file_path, 'wb') as f:
    np.savez_compressed(f, data=public_data)
print(f"Public dataset created at {public_file_path} with {len(public_image)} samples (100 per class)")
