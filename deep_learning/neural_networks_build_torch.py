import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Normalize image to [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download dataset
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Define Neural Networks
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)   # fully connected input => hidden
        self.fc2 = nn.Linear(128, 64)      # hidden => hidden
        self.fc3 = nn.Linear(64, 10)       # hidden => output (10 classes: 0–9)

    def forward(self, x):
        x = x.view(-1, 28*28)         # Flatten image (batch_size, 784)
        x = F.relu(self.fc1(x))       # Activation ReLU
        x = F.relu(self.fc2(x))
        x = self.fc3(x)               # Output (logits)
        return x

# Initialize model, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet().to(device)

criterion = nn.CrossEntropyLoss()        # Softmax + CE for classification
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()            # reset gradient
        output = model(data)             # forward
        loss = criterion(output, target) # loss
        loss.backward()                  # backward
        optimizer.step()                 # update weights
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():  # no gradient
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f"Accuracy on test set: {100 * correct / total:.2f}%")

# Save the model
torch.save(model.state_dict(), "mnist_model.pth")

# Reload the model
model = NeuralNet().to(device)
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()