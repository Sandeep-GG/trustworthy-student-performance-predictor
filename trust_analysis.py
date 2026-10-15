
import joblib
import numpy as np
import pandas as pd

# Step 1: Load the trained model and test data
model = joblib.load("student_model.pkl")
X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")

print("Model and test data loaded successfully!")

# Step 2: Get predictions from every decision tree
tree_predictions = np.array([
    tree.predict(X_test)
    for tree in model.estimators_
])

# Step 3: Calculate the average prediction
mean_predictions = tree_predictions.mean(axis=0)

# Step 4: Calculate disagreement between trees
uncertainty = tree_predictions.std(axis=0)

# Step 5: Create an approximate prediction interval
lower_bound = np.percentile(tree_predictions, 5, axis=0)
upper_bound = np.percentile(tree_predictions, 95, axis=0)

# Step 6: Display predictions and uncertainty
print("\nTrustworthiness Analysis")
print("------------------------")

for i in range(min(10, len(y_test))):
   spread = upper_bound[i] - lower_bound[i]

if uncertainty[i] <= 2:
    indicator = "Lower tree disagreement"
elif uncertainty[i] <= 4:
    indicator = "Moderate tree disagreement"
else:
    indicator = "Higher tree disagreement"

    print(f"\nStudent {i + 1}")
    print(f"Actual grade: {y_test.iloc[i]}")
    print(f"Predicted grade: {mean_predictions[i]:.2f}")
    print(
        f"Approximate interval: "
        f"{lower_bound[i]:.2f} to {upper_bound[i]:.2f}"
    )
    print(f"Tree disagreement: {uncertainty[i]:.2f}")
    print(f"Trust indicator: {indicator}")

# Step 7: Find important input features
feature_names = model.feature_names_in_ if hasattr(
    model, "feature_names_in_"
) else None

# Feature names belong to the preprocessed data
preprocessor = joblib.load("preprocessor.pkl")
feature_names = preprocessor.get_feature_names_out()

importance = model.feature_importances_

importance_table = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_table = importance_table.sort_values(
    by="Importance", ascending=False
)

print("\nTop 10 Important Features")
print("-------------------------")
print(importance_table.head(10).to_string(index=False))

# Step 8: Save the analysis results
results = pd.DataFrame({
    "Actual_Grade": y_test.to_numpy(),
    "Predicted_Grade": mean_predictions,
    "Tree_Disagreement": uncertainty,
    "Lower_Bound": lower_bound,
    "Upper_Bound": upper_bound
})

results.to_csv("trust_analysis_results.csv", index=False)
importance_table.to_csv("feature_importance.csv", index=False)

print("\nTrust analysis completed successfully!")
print("Saved: trust_analysis_results.csv")
print("Saved: feature_importance.csv")
