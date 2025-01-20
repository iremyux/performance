import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import time
import psutil

# Monitor CPU usage
def get_cpu_usage():
    return psutil.cpu_percent(interval=None)

print("Starting PyTorch script...")

# 1. Prepare Data (Subset)
print("Preparing data...")
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet50 requires 224x224 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
print("Train dataset loaded.")

# Use a subset of the data (e.g., first 2000 samples)
train_dataset = torch.utils.data.Subset(train_dataset, range(2000))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
print("Train DataLoader created with a subset of 2000 samples.")

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_dataset = torch.utils.data.Subset(test_dataset, range(500))  # Use 500 samples for testing
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
print("Test DataLoader created with a subset of 500 samples.")

# 2. Load Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = torchvision.models.resnet50(pretrained=False, num_classes=10)
model = model.to(device)
print("ResNet50 model loaded and moved to device.")

# 3. Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
print("Loss function defined.")
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("Optimizer defined.")

# 4. Train Model
num_epochs = 5  # Reduced epochs for faster training
print("Starting training...")
start_time = time.time()
cpu_usages = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs} starting...")
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        cpu_usages.append(get_cpu_usage())

    print(f"Epoch {epoch + 1} complete. Loss: {running_loss / (i + 1):.4f}")

train_time = time.time() - start_time
print(f"Training complete. Total Training Time: {train_time:.2f} seconds")
print(f"Average CPU Usage During Training: {sum(cpu_usages) / len(cpu_usages):.2f}%")

# 5. Evaluate Model
print("Starting evaluation...")
model.eval()
correct = 0
total = 0
inference_times = []

with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        start_inference = time.time()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        inference_times.append(time.time() - start_inference)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total * 100
print(f"Evaluation complete. Test Accuracy: {accuracy:.2f}%")
print(f"Average Inference Latency per Batch: {sum(inference_times) / len(inference_times):.4f} seconds")
