
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
from ucimlrepo import fetch_ucirepo

# Download the dataset
dataset = fetch_ucirepo(id=320)

# Get input features
X = dataset.data.features

# Get target columns
y_data = dataset.data.targets

# Select final grade as the target
y = y_data["G3"]

# Remove G1 and G2 if they exist in the input features
X = X.drop(columns=["G1", "G2"], errors="ignore")



print("Input shape:", X.shape)
print("Target shape:", y.shape)

# Step 3: Identify categorical and numerical columns
categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

print("Categorical columns:", categorical_columns)
print("Numerical columns:", numerical_columns)

# Step 4: Prepare numerical columns
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Step 5: Prepare categorical columns
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Step 6: Combine both preprocessing pipelines
preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

# Step 7: Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining input shape:", X_train.shape)
print("Testing input shape:", X_test.shape)
print("Training target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)

# Step 8: Learn preprocessing from training data only
X_train_processed = preprocessor.fit_transform(X_train)

# Step 9: Apply the same preprocessing to test data
X_test_processed = preprocessor.transform(X_test)

print("\nProcessed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)

# Step 10: Save the prepared data
import joblib

joblib.dump(preprocessor, "preprocessor.pkl")
joblib.dump(X_train_processed, "X_train.pkl")
joblib.dump(X_test_processed, "X_test.pkl")
joblib.dump(y_train, "y_train.pkl")
joblib.dump(y_test, "y_test.pkl")

print("\nPreprocessing completed successfully!")
