import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import kagglehub

path = kagglehub.dataset_download("blastchar/telco-customer-churn")

df_telco = pd.read_csv(path + '/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Encode categorical variables
label_encoders = {}
for column in df_telco.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_telco[column] = le.fit_transform(df_telco[column])
    label_encoders[column] = le

# Define features and target
X = df_telco.drop('Churn', axis=1)
y = df_telco['Churn']

# Scale Features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
logistic_model = LogisticRegression(max_iter=200)
logistic_model.fit(X_train, y_train)

# Train KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Make predictions
y_pred = logistic_model.predict(X_test)
y_pred_knn = knn_model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Evaluate KNN model
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"Accuracy (KNN): {accuracy_knn}")
print("Classification Report (KNN):\n", classification_report(y_test, y_pred_knn))
print("Confusion Matrix (KNN):\n", confusion_matrix(y_test, y_pred_knn))


print(df_telco.info())
print(df_telco.describe())

sns.countplot(x='Churn', data=df_telco)
plt.title('Churn Distribution')
plt.show()

df_telco.fillna(df_telco.mean(), inplace=True)
