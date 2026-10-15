
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.ensemble import ExtraTreesRegressor

# 1. Load the preprocessed training and testing data
X_train = joblib.load("X_train.pkl")
X_test = joblib.load("X_test.pkl")
y_train = joblib.load("y_train.pkl")
y_test = joblib.load("y_test.pkl")

print("Dataset loaded successfully!")

# 2. Create the selected model
model = ExtraTreesRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

# 3. Define 5-fold cross-validation
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# 4. Evaluate the model on five different training/validation splits
scoring = {
    "mae": "neg_mean_absolute_error",
    "mse": "neg_mean_squared_error",
    "r2": "r2"
}

results = cross_validate(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring,
    n_jobs=-1
)

# 5. Convert negative error scores into positive values
mae_scores = -results["test_mae"]
rmse_scores = np.sqrt(-results["test_mse"])
r2_scores = results["test_r2"]

print("\n5-Fold Cross-Validation Results")
print("--------------------------------")

for i in range(5):
    print(
        f"Fold {i + 1}: "
        f"MAE = {mae_scores[i]:.3f}, "
        f"RMSE = {rmse_scores[i]:.3f}, "
        f"R2 = {r2_scores[i]:.3f}"
    )

print("\nAverage Cross-Validation Results")
print("--------------------------------")
print(f"Mean MAE:  {mae_scores.mean():.3f}")
print(f"Mean RMSE: {rmse_scores.mean():.3f}")
print(f"Mean R2:   {r2_scores.mean():.4f}")

print("\nVariation Across Folds")
print("-----------------------")
print(f"MAE standard deviation:  {mae_scores.std():.3f}")
print(f"RMSE standard deviation: {rmse_scores.std():.3f}")
print(f"R2 standard deviation:   {r2_scores.std():.3f}")

# 6. Train on the complete training set
model.fit(X_train, y_train)

# 7. Evaluate once on the held-out test set
predictions = model.predict(X_test)

test_mae = mean_absolute_error(y_test, predictions)
test_rmse = np.sqrt(mean_squared_error(y_test, predictions))
test_r2 = r2_score(y_test, predictions)

print("\nFinal Test Set Results")
print("----------------------")
print(f"MAE:  {test_mae:.3f}")
print(f"RMSE: {test_rmse:.3f}")
print(f"R2:   {test_r2:.4f}")

# 8. Save the validation results
summary = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2"],
    "CV_Mean": [
        mae_scores.mean(),
        rmse_scores.mean(),
        r2_scores.mean()
    ],
    "CV_Std": [
        mae_scores.std(),
        rmse_scores.std(),
        r2_scores.std()
    ],
    "Test_Set": [
        test_mae,
        test_rmse,
        test_r2
    ]
})

summary.to_csv("validation_results.csv", index=False)

print("\nSaved: validation_results.csv")
