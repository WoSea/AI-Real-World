import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

path = "https://raw.githubusercontent.com/datasciencedojo/datasets/refs/heads/master/titanic.csv"
df = pd.read_csv(path)

# Display dataset
print(df.info())
# Review the first few row
print(df.head())
print("="*50)

def preprocess_data(df):
    # Fill numeric NaNs with median, categorical NaNs with mode
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df

def encode_categorical_variables(df, categorical_cols):
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df_encoded

def label_encoding(df, categorical_cols):
    # Apply label encoding to categorical
    for col in categorical_cols:
        if col in categorical_cols:
            le = LabelEncoder()
            df[col+'_encoded'] = le.fit_transform(df[col])
    return df

def frequency_encoding(df, categorical_cols):
    # Apply frequency encoding to categorical columns
    for col in categorical_cols:
        if col in df.columns:
            freq_col = f"{col}_freq"
            df[freq_col] = df[col].map(df[col].value_counts())
            print(df[[col, f"{col}_freq"]].head())
    return df

def train_and_evaluate_model(X, y):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train logistic regression model
    model = LogisticRegression(max_iter=2000, solver='liblinear')
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, model

# Cross-Validation & Hyperparameter Tuning
def train_with_cv_and_tuning(X, y):
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'saga'],
        'penalty': ['l1', 'l2'] # Lasso Regularization, Ridge Regularization
    }

    model = LogisticRegression(max_iter=2000)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print("Best Params:", best_params)
    print("Best CV Accuracy:", best_score)

    return best_model, best_score

def cross_validate_model(model, X, y, cv=5):
    # Perform k-fold cross-validation
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"Cross-validation scores (cv={cv}):", scores)
    print("Mean CV Accuracy:", scores.mean())
    return scores

if __name__ == "__main__":
    df = preprocess_data(df)
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['int64','object']).columns.tolist()
    print("Categorical Columns:", categorical_cols)

    # Encode categorical variables
    df_encoded = encode_categorical_variables(df, categorical_cols) # One hot encoding
    # Display the first few rows of the encoded DataFrame
    print(df_encoded.head())
    
    # Display summary of the encoded DataFrame
    print(df_encoded.info())

    # Apply label encoding
    df_label_encoded = label_encoding(df, categorical_cols)
    print(df_label_encoded.head())
    
    print(df_label_encoded[['Pclass','Pclass_encoded']].head())

    for col in categorical_cols:
        encoded_col = f"{col}_encoded"
        if encoded_col in df_label_encoded.columns:
            print(df_label_encoded[[col, encoded_col]].head())
    
    encoded_cols = [col for col in df_label_encoded.columns if col.endswith('_encoded')]
    print(df_label_encoded[encoded_cols].head())

    # Apply frequency encoding
    df_freq_encoded = frequency_encoding(df_label_encoded, categorical_cols)
    print(df_freq_encoded.info())

    # Train and evaluate model
    df_encoded_train = pd.get_dummies(df_freq_encoded, columns=['Sex','Embarked'], drop_first=True)
    print(df_encoded_train.head())
    print(df_encoded_train.info())

    X = df_encoded_train.drop(columns=["Survived", "PassengerId", "Name", "Ticket", "Cabin"])
    y = df_encoded_train["Survived"]

    accuracy, model = train_and_evaluate_model(X, y)
    print(f"Logistic Regression Accuracy: {accuracy:.4f}")

    # Save the trained model
    joblib.dump(model, "logistic_model.pkl")
    print("Model saved as logistic_model.pkl")

    # Train and evaluate model with cross-validation and hyperparameter tuning
    best_model, best_score = train_with_cv_and_tuning(X, y)
    joblib.dump(best_model, "logistic_model_tuned.pkl")
    print("Tuned model saved as logistic_model_tuned.pkl")
    
     # Cross-validation evaluation
    cross_validate_model(LogisticRegression(max_iter=2000, solver='liblinear'), X, y, cv=5)