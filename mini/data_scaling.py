from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

# Load Iris dataset
iris = load_iris()
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
y = iris.target

# Display dataset Information
print("Datset Info: ")
print(X.describe())

print("\n Target Classe: ", iris.target_names)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train k-NN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict and evaluate
y_pred = knn.predict(X_test)

print("\nAccuracy without scaling: ", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Scale features (Normalization)
scaler = MinMaxScaler() # [0, 1] / Using StandardScaler() - mean = 0, std = 1.
X_scaled = scaler.fit_transform(X)

# Split scaled data
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train k-NN classifier
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train_scaled)

# Predict and evaluate
y_pred_scaled = knn_scaled.predict(X_test_scaled)

print("\nAccuracy with Min-Max Scaling: ", accuracy_score(y_test_scaled, y_pred_scaled))
print("\nClassification Report:\n", classification_report(y_test_scaled, y_pred_scaled))
print("\nConfusion Matrix:\n", confusion_matrix(y_test_scaled, y_pred_scaled))
