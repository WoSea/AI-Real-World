from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import pandas as pd

data = load_breast_cancer()
X,y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train set shape:", X_train.shape, y_train.shape)
print(f"Feature names: {data.feature_names}")
print(f"Class names: {data.target_names}")

# Train a RandomForestClassifier - Random Forest with default hyperparameters
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification report:")
print(classification_report(y_test, y_pred))
# print("ROC AUC score:")
# print(roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr'))

# Train Random Forest with adjusted hyperparameters
clf_tuned = RandomForestClassifier(n_estimators=4000, max_depth=5, random_state=42)
clf_tuned.fit(X_train, y_train)

# Make predictions
y_pred_tuned = clf_tuned.predict(X_test)

# Evaluate the model
print("Tuned Model Accuracy:", accuracy_score(y_test, y_pred_tuned))
print("Tuned Model Classification report:")
print(classification_report(y_test, y_pred_tuned))
# print("Tuned Model ROC AUC score:")