# CNN components:
# Input Layer => input data (image).
# Convolutional Layers => extract features (edges, textures).
# Pooling Layers => dimensionality reduction, preserve important features.
# Flatten Layer => convert tensor into a flat vector.
# Dense (Fully Connected Layers) => learn complex relationships.
# Output Layer => classification (softmax).

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# Build CNN
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 classes for MNIST
])

# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(x_train, y_train, epochs=5,
                    validation_data=(x_test, y_test))

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Plot Loss and Accuracy
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Pytorch version

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. Prepare data
transform = transforms.Compose([
    transforms.ToTensor(),                # convert image to Tensor
    transforms.Normalize((0.5,), (0.5,))  # normalize to [-1, 1]
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 2. Build CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)   # 1 input channel (gray), 32 filters, kernel 3x3
        self.pool = nn.MaxPool2d(2, 2)     # MaxPooling 2x2
        self.conv2 = nn.Conv2d(32, 64, 3)  # 32 -> 64 filters
        self.fc1 = nn.Linear(64 * 5 * 5, 64)  # Flatten -> Dense 64
        self.fc2 = nn.Linear(64, 10)       # Output 10 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # conv1 -> relu -> pool
        x = self.pool(F.relu(self.conv2(x)))   # conv2 -> relu -> pool
        x = x.view(-1, 64 * 5 * 5)             # flatten
        x = F.relu(self.fc1(x))                # dense hidden
        x = self.fc2(x)                        # output
        return x

model = CNN()

# 3. Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training loop
train_losses, val_losses = [], []
for epoch in range(5):
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    train_losses.append(running_loss/len(train_loader))
    
    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            pred = outputs.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
    
    val_losses.append(val_loss/len(test_loader))
    accuracy = 100. * correct / len(test_dataset)
    print(f"Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Acc: {accuracy:.2f}%")

# 5. Visualization Loss
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
