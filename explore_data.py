
from ucimlrepo import fetch_ucirepo
import pandas as pd

# Download the Student Performance dataset
dataset = fetch_ucirepo(id=320)

# Separate input features and target columns
X = dataset.data.features
y = dataset.data.targets

print("Dataset downloaded successfully!")

print("\nInput data shape:")
print(X.shape)

print("\nInput column names:")
print(X.columns.tolist())

print("\nFirst 5 rows of input data:")
print(X.head())

print("\nTarget column names:")
print(y.columns.tolist())

print("\nFirst 5 rows of target data:")
print(y.head())

print("\nMissing values in input data:")
print(X.isnull().sum())
