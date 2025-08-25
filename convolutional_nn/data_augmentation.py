import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# 1. Load dataset (CIFAR-10 for illustration)
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 2. Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,        # random rotation
    width_shift_range=0.1,    # horizontal shift
    height_shift_range=0.1,   # vertical shift
    horizontal_flip=True,     # random horizontal flip
    zoom_range=0.1            # random zoom
)

datagen.fit(x_train)

# 3. Build CNN with Regularization
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', padding="same", 
                  kernel_regularizer=regularizers.l2(0.001),
                  input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),   # Dropout for regularization
    
    layers.Conv2D(64, (3,3), activation='relu', padding="same",
                  kernel_regularizer=regularizers.l2(0.001)),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')   # 10 classes
])

# 4. Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Train with Data Augmentation
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    validation_data=(x_test, y_test),
    epochs=10
)

# 6. Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")

# 7. Visualization
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel("Epochs"); plt.ylabel("Accuracy"); plt.legend()

plt.show()