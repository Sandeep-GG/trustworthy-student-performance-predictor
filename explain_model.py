
import joblib
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

# Step 1: Load the selected model and test data
model = joblib.load("best_student_model.pkl")
X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")

print("Selected model loaded successfully!")

# Step 2: Calculate the baseline R2 score
baseline_r2 = r2_score(
    y_test,
    model.predict(X_test)
)

print(f"Baseline R2: {baseline_r2:.4f}")

# Step 3: Calculate permutation importance
result = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="r2"
)

# Step 4: Get the feature names
preprocessor = joblib.load("preprocessor.pkl")
feature_names = preprocessor.get_feature_names_out()

# Step 5: Create a feature importance table
importance_table = pd.DataFrame({
    "Feature": feature_names,
    "Importance": result.importances_mean,
    "Std_Deviation": result.importances_std
})

importance_table = importance_table.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Features Affecting Model Performance")
print("-------------------------------------------")
print(importance_table.head(10).to_string(index=False))

# Step 6: Save the explanation
importance_table.to_csv(
    "permutation_importance.csv",
    index=False
)

print("\nSaved: permutation_importance.csv")
