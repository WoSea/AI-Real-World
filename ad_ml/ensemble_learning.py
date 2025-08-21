from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

data =load_iris()
X, y = data.data, data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train individual models
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
print("Logistic Regression trained.")

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
print("Decision Tree trained.")

knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
print("K-Nearest Neighbors trained.")

# Create ensemble Classifier
ensemble_model = VotingClassifier(estimators=[
    ('logistic', log_model),
    ('decision_tree', dt_model),
    ('knn', knn_model)
], voting='hard')

# Train ensemble models
ensemble_model.fit(X_train, y_train)
print("Ensemble model trained.")

# Optimize individual models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    # "Voting Classifier": VotingClassifier(estimators=[
    #     ('logistic', log_model),
    #     ('decision_tree', dt_model),
    #     ('knn', knn_model)
    # ])
}

# for name, model in models.items():
#     model.fit(X_train, y_train)
#     print(f"{name} trained.")

# Predict with ensemble model
y_pred = ensemble_model.predict(X_test)
print("Ensemble model predictions:", y_pred)

# Evaluate ensemble model
print("\n Evaluate model accuracy:")
ensemble_accuracy = accuracy_score(y_test, y_pred)
print(f"Ensemble model accuracy: {ensemble_accuracy:.2f}")

# Evaluate individual models
print("\n Individual model accuracies:")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} accuracy: {accuracy:.2f}")