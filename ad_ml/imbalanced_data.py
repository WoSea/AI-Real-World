# pip install imblearn
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

path = kagglehub.dataset_download("jacklizhi/creditcard")

print("Path to dataset files:", path)
df = pd.read_csv(path + '/creditcard.csv')

print("Dataset information:")
print(df.info())
print("\n Class distribution:")
print(df['Class'].value_counts())

# Split dataset
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train model
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Classification report:")
print(classification_report(y_test, y_pred))
print("ROC AUC score:")
print(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Display new class distribution
print("New class distribution:")
print(pd.Series(y_resampled).value_counts())

# Train model on resampled data
rf_model_smote = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_model_smote.fit(X_resampled, y_resampled)

# Evaluate model
y_pred_smote = rf_model_smote.predict(X_test)
print("Classification report:")
print(classification_report(y_test, y_pred_smote))
print("ROC AUC score:")
print(roc_auc_score(y_test, rf_model_smote.predict_proba(X_test)[:, 1]))