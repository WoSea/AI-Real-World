import matplotlib.pylot as plt
from tourchvision import datasets, transforms
import numpy as np

# Load dataset
transform = transforms.ToTensor()
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

# Visualize some sample images
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i in range(10):
    image, label = train_dataset[i]
    axes[i//5, i%5].imshow(image.permute(1, 2, 0))
    axes[i//5, i%5].set_title(f"Label: {label}")
    axes[i//5, i%5].axis('off')
plt.show()

# Display pixel values for the first image
image, label = train_dataset[0]
print(f"Image shape: {image.shape}")
print("Pixel values for the first image:")
print(image)    

import tensorflow as tf

# Define a simple CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
tf.keras.utils.plot_model(model, to_file="cnn_model.png", show_shapes=True, show_layer_names=True)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

import torch.nn as nn

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x