import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR‑10 for load and normalisation 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
test_loader  = DataLoader(testset,  batch_size=32, shuffle=False, num_workers=2)  # :contentReference[oaicite:1]{index=1}

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__() #definition time
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
        self.pool  = nn.MaxPool2d(2,2) 
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.fc1   = nn.Linear(32 * 5 * 5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # dynamcially flattening the tensor
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

if __name__ == "__main__":
    # Display a batch of images to confirm loading
    dataiter = iter(train_loader)
    images, labels = next(dataiter) #batch of images and labels
    grid = utils.make_grid(images)
    plt.imshow(grid.permute(1, 2, 0).numpy())
    print("Sample batch shape:", images.shape)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop with validation, keeping it low rn for testing later on 
    for epoch in range(5):
        model.train()
        running_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels) # read more abt cross entropy loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1} Train Loss: {running_loss/len(train_loader):.3f}")

        # Validation
        model.eval()
        total = correct = val_loss = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                _, preds = outputs.max(1) #indexing max log probab
                total += labels.size(0)
                correct += (preds == labels).sum().item() #predictions vs labels boss battle
        print(f"Epoch {epoch+1} Val Loss: {val_loss/len(test_loader):.3f} | Val Acc: {100*correct/total:.2f}%\n")
