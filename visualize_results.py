
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# --------------------------------------------------
# 1. Load the trained model and test data
# --------------------------------------------------

model = joblib.load("best_student_model.pkl")
X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")

print("Model and test data loaded successfully!")

# --------------------------------------------------
# 2. Predict student final grades
# --------------------------------------------------

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("-----------------")
print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R2:   {r2:.4f}")

# Save actual and predicted grades
prediction_results = pd.DataFrame({
    "Actual_Grade": np.asarray(y_test),
    "Predicted_Grade": predictions,
    "Prediction_Error": np.asarray(y_test) - predictions
})

prediction_results.to_csv(
    "prediction_results.csv",
    index=False
)

print("\nSaved: prediction_results.csv")

# --------------------------------------------------
# 3. Graph 1: Actual vs Predicted Grades
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.7
)

# Ideal prediction line
minimum = min(np.min(y_test), np.min(predictions))
maximum = max(np.max(y_test), np.max(predictions))

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Final Grade")
plt.ylabel("Predicted Final Grade")
plt.title("Actual vs Predicted Student Grades")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.show()

# --------------------------------------------------
# 4. Graph 2: Prediction Error Distribution
# --------------------------------------------------

errors = np.asarray(y_test) - predictions

plt.figure(figsize=(8, 6))

plt.hist(
    errors,
    bins=15,
    edgecolor="black"
)

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel("Prediction Error (Actual - Predicted)")
plt.ylabel("Number of Students")
plt.title("Distribution of Prediction Errors")
plt.tight_layout()
plt.savefig("prediction_error_distribution.png")
plt.show()

# --------------------------------------------------
# 5. Graph 3: Feature Importance
# --------------------------------------------------

importance_df = pd.read_csv("permutation_importance.csv")

top_features = importance_df.head(10).sort_values(
    "Importance",
    ascending=True
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"],
    xerr=top_features["Std_Deviation"],
    capsize=3
)

plt.xlabel("Decrease in R2 After Shuffling")
plt.ylabel("Feature")
plt.title("Top 10 Features: Permutation Importance")
plt.tight_layout()
plt.savefig("feature_importance_graph.png")
plt.show()

# --------------------------------------------------
# 6. Graph 4: Tree Prediction Disagreement
# --------------------------------------------------

if hasattr(model, "estimators_"):
    tree_predictions = np.array([
        tree.predict(X_test)
        for tree in model.estimators_
    ])

    # Standard deviation across trees for each student
    disagreement = np.std(tree_predictions, axis=0)

    plt.figure(figsize=(8, 6))

    plt.hist(
        disagreement,
        bins=15,
        edgecolor="black"
    )

    plt.xlabel("Standard Deviation of Tree Predictions")
    plt.ylabel("Number of Students")
    plt.title("Model Prediction Disagreement")
    plt.tight_layout()
    plt.savefig("tree_prediction_disagreement.png")
    plt.show()

    print("\nTree Prediction Disagreement")
    print("----------------------------")
    print(f"Mean disagreement: {disagreement.mean():.3f}")
    print(f"Maximum disagreement: {disagreement.max():.3f}")

else:
    print(
        "\nTree disagreement graph skipped: "
        "the selected model does not expose individual trees."
    )

print("\nAll visualization tasks completed!")
print("Check your project folder for the PNG graphs.")
