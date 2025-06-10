import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),  # (32x32 → 30x30)
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3),  # (30x30 → 28x28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),       # (28x28 → 14x14)

            nn.Conv2d(32, 64, kernel_size=3),  # (14x14 → 12x12)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),  # (12x12 → 10x10)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)        # (10x10 → 5x5)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
writer = SummaryWriter('tensorboard_logs')

for epoch in range(15):
    model.train()
    total_loss = total_correct = 0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += (output.argmax(1) == labels).sum().item()
    acc = total_correct / len(trainset)
    writer.add_scalar('Loss/train', total_loss, epoch)
    writer.add_scalar('Accuracy/train', acc, epoch)

model.eval()
correct = 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        correct += (outputs.argmax(1) == labels).sum().item()
print("Test Accuracy:", correct / len(testset))