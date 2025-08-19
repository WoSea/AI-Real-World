import pandas as pd

path = "https://raw.githubusercontent.com/datasciencedojo/datasets/refs/heads/master/titanic.csv"

df = pd.read_csv(path)

# Display dataset
print(df.info())
# Review the first few row
print(df.head())

# Seperate features
categorical_features = df.select_dtypes(include=['object']).columns
numerical_features = df.select_dtypes(include=["int64","float64"]).columns

print("Categorical Features: ", categorical_features)
print("Numerical Features: ", numerical_features)

# Display summary of categorical features
print("Categorical Features Summary:\n")
for col in categorical_features:
    print(f"{col}:\n{df[col].value_counts()}\n")

# Display Summary of numerical features
print("Numerical Features Summary:\n")
print(df[numerical_features].describe())