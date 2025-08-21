from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

data = load_breast_cancer()
X, y = data.data, data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display dataset information
print("Dataset features:", data.feature_names)
print("Dataset target:", data.target_names)

# Train Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
print("Random Forest trained.")

# Predict with Random Forest model
y_pred = rf_model.predict(X_test)
print("Random Forest model predictions:", y_pred)

# Evaluate Random Forest model
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest model accuracy: {accuracy:.2f}")

print("\nRandom Forest model classification report:")
print(classification_report(y_test, y_pred))

# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'max_features': [None, 'sqrt', 'log2']
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best hyperparameters found:{grid_search.best_params_}")
print(f"Best Cross-Validation Accuracy:{grid_search.best_score_}")

