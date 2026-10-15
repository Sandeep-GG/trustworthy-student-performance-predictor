
import joblib
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the preprocessed data
X_train = joblib.load("X_train.pkl")
X_test = joblib.load("X_test.pkl")

y_train = joblib.load("y_train.pkl")
y_test = joblib.load("y_test.pkl")

print("Training data loaded successfully!")

# Step 2: Create the model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# Step 3: Train the model
model.fit(X_train, y_train)

print("Model training completed!")

# Step 4: Predict final grades for test students
y_pred = model.predict(X_test)

# Step 5: Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Results")
print("------------------------")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")

# Step 6: Compare actual and predicted grades
print("\nActual vs Predicted Grades")
print("--------------------------")

for actual, predicted in zip(
    y_test.iloc[:10], y_pred[:10]
):
    print(
        f"Actual: {actual:2d} | "
        f"Predicted: {predicted:.2f}"
    )

# Step 7: Save the trained model
joblib.dump(model, "student_model.pkl")

print("\nModel saved as student_model.pkl")
