
import joblib
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the prepared datasets
X_train = joblib.load("X_train.pkl")
X_test = joblib.load("X_test.pkl")
y_train = joblib.load("y_train.pkl")
y_test = joblib.load("y_test.pkl")

# Step 2: Create the candidate models
models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=200,
        random_state=42
    )
}

results = []
trained_models = {}

# Step 3: Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append((name, mae, rmse, r2))
    trained_models[name] = model

# Step 4: Display comparison
print("\nModel Comparison")
print("-" * 65)
print(f"{'Model':<22}{'MAE':>10}{'RMSE':>12}{'R2':>12}")

for name, mae, rmse, r2 in results:
    print(f"{name:<22}{mae:>10.2f}{rmse:>12.2f}{r2:>12.4f}")

# Step 5: Select the model with the highest R2
best_name, best_mae, best_rmse, best_r2 = max(
    results, key=lambda result: result[3]
)

print(f"\nBest model by test R2: {best_name}")
print(f"Test R2: {best_r2:.4f}")

# Step 6: Save the selected model
joblib.dump(trained_models[best_name], "best_student_model.pkl")

print("Saved as best_student_model.pkl")
