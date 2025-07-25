import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


transform = transforms.Compose([
    transforms.Resize((224, 224)),                             # Resize all images to resnet style, resizing needed cause images vary
    transforms.ToTensor(),                                     # Convert pil image to tensor
    transforms.Normalize([0.5, 0.5, 0.5],                      # Normalize RGB channels to (-1, 1)
                         [0.5, 0.5, 0.5])
])

train_data = datasets.ImageFolder(root='data/train', transform=transform)
test_data = datasets.ImageFolder(root='data/test', transform=transform)

trainLoader = DataLoader(train_data, batch_size=32, shuffle=True)
testLoader = DataLoader(test_data, batch_size=32, shuffle=False)

class FundusCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__() #kerrnals = extracting little info peanuts
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 3 channel (rgb) and 16 filterss
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # 224x224 → 112x112
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                           # 112x112 → 56x56
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                                # 64 channels * 56 * 56 = 200704
            nn.Linear(64 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, 2)                            # 2 output classes: healthy / unhealthy
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = FundusCNN() #init-ng it 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):  # (increase in testing, kept low for fast checks initially)
    running_loss = 0.0
    for images, labels in trainLoader:
        outputs = model(images)
        loss = criterion(outputs, labels) # 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch {epoch+1} complete. Loss: {running_loss/len(trainLoader):.4f}")

# test sets rn

correct = 0
total = 0

with torch.no_grad():
    for images, labels in testLoader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
