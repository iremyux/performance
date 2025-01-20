import os
import tarfile
import requests
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import time
import psutil

# Download and prepare CIFAR-10 dataset
def download_and_extract_cifar10():
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    dataset_dir = "./cifar-10-batches-py"
    tar_path = "./cifar-10-python.tar.gz"

    if not os.path.exists(dataset_dir):
        print("Downloading CIFAR-10 dataset...")
        response = requests.get(url, stream=True)
        with open(tar_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        print("Extracting CIFAR-10 dataset...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=".")

    return dataset_dir

# Load CIFAR-10 data
def load_cifar10_data(dataset_dir, train=True):
    if train:
        batch_files = [f"{dataset_dir}/data_batch_{i}" for i in range(1, 6)]
    else:
        batch_files = [f"{dataset_dir}/test_batch"]

    data, labels = [], []
    for batch_file in batch_files:
        with open(batch_file, "rb") as f:
            batch = pickle.load(f, encoding="bytes")
            data.append(batch[b"data"])
            labels += batch[b"labels"]

    data = np.vstack(data).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    labels = np.array(labels, dtype=np.int64)
    return data, labels

# CIFAR-10 Dataset class
class CIFAR10Dataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = (self.data[idx] - 0.5) / 0.5  # Normalize to [-1, 1]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# ResNet50 implementation (simplified)
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Training setup
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Download and prepare data
    dataset_dir = download_and_extract_cifar10()
    train_data, train_labels = load_cifar10_data(dataset_dir, train=True)
    test_data, test_labels = load_cifar10_data(dataset_dir, train=False)

    # Subset the data
    train_data, train_labels = train_data[:2000], train_labels[:2000]
    test_data, test_labels = test_data[:500], test_labels[:500]

    # Dataset and DataLoader
    train_dataset = CIFAR10Dataset(train_data, train_labels)
    test_dataset = CIFAR10Dataset(test_data, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Model, loss, optimizer
    model = ResNet50(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Metrics collection
    training_times = []
    test_times = []
    train_latencies = []
    test_latencies = []
    cpu_usages = []

    # Training loop
    for epoch in range(5):
        print(f"Starting epoch {epoch+1}...")
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            batch_start = time.time()
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_latencies.append(time.time() - batch_start)
            cpu_usages.append(psutil.cpu_percent(interval=None))

        epoch_end = time.time()
        training_times.append(epoch_end - epoch_start)
        print(f"Epoch {epoch+1} completed. Loss: {running_loss/len(train_loader):.4f}. Training time: {epoch_end - epoch_start:.2f}s")

    # Evaluation
    print("Starting evaluation...")
    model.eval()
    correct = 0
    total = 0
    test_start = time.time()
    with torch.no_grad():
        for inputs, labels in test_loader:
            batch_start = time.time()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_latencies.append(time.time() - batch_start)
    test_end = time.time()
    test_times.append(test_end - test_start)

    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    print(f"Total Test Time: {test_end - test_start:.2f}s")

    # Metrics summary
    print("Training Times per Epoch (s):", training_times)
    print("Average Training Latency per Batch (s):", np.mean(train_latencies))
    print("Test Times (s):", test_times)
    print("Average Test Latency per Batch (s):", np.mean(test_latencies))
    print("CPU Usage (%):", np.mean(cpu_usages))

if __name__ == "__main__":
    train_model()
