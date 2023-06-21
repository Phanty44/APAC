import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision

from torchvision import transforms
from torch.utils.data import DataLoader

import visualization

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Define VAN architecture
class VAN(nn.Module):
    def __init__(self):
        super(VAN, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.classification = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            #nn.Softmax(dim=1)
        )

    def forward(self, x):
        attention_map = self.attention(x)
        attended_image = x * attention_map
        output = self.classification(attended_image)
        return output


# Load and preprocess the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
transform1 = transforms.Compose([
    transforms.RandomResizedCrop((28, 28)).to(device),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
transform2 = transforms.Compose([
    transforms.ElasticTransform(alpha=38.0, sigma=6.0).to(device),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
transform3 = transforms.Compose([
    transforms.GaussianBlur(kernel_size=3).to(device),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
trainset1 = torchvision.datasets.MNIST(root="./data", train=True, transform=transform1, download=True)
trainset2 = torchvision.datasets.MNIST(root="./data", train=True, transform=transform2, download=True)
trainset3 = torchvision.datasets.MNIST(root="./data", train=True, transform=transform3, download=True)

combined_trainset = data.ConcatDataset([train_dataset, trainset1, trainset2, trainset3])

test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)
testset1 = torchvision.datasets.MNIST(root="./data", train=False, transform=transform1, download=True)
testset2 = torchvision.datasets.MNIST(root="./data", train=False, transform=transform2, download=True)
testset3 = torchvision.datasets.MNIST(root="./data", train=False, transform=transform3, download=True)

combined_testset = data.ConcatDataset([train_dataset, testset1, testset2, testset3])

train_loader = DataLoader(combined_trainset, batch_size=128, shuffle=True)
test_loader = DataLoader(combined_testset, batch_size=128, shuffle=False)

# Create VAN model and move it to GPU if available
model = VAN().to(device)
print(model)
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
model.train().to(device)
for epoch in range(3):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    running_loss /= len(train_loader)
    print(f"Epoch {epoch + 1} - Training Loss: {running_loss}")

# Evaluation
model.eval()
correct = 0
total = 0
predicted_labels = []
true_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        predicted_labels.extend(predicted.cpu().numpy())  # Append predicted labels to the list
        true_labels.extend(labels.cpu().numpy())  # Append true labels to the list
print(f"Test Accuracy: {(100 * correct / total):.2f}%")
print("Confusion Matrix:")
visualization.visualize_confusion(true_labels, predicted_labels)
