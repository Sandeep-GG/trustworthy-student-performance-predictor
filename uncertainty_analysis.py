
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Load training data
X_train = joblib.load("X_train.pkl")
y_train = np.asarray(joblib.load("y_train.pkl"))

print("Data loaded successfully!")

# 2. Create five-fold cross-validation
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Store predictions and uncertainty for each validation row
oof_predictions = np.zeros(len(y_train))
tree_disagreement = np.zeros(len(y_train))

# 3. Train a separate model for each fold
for fold, (train_idx, val_idx) in enumerate(
    cv.split(X_train), start=1
):
    X_fold_train = X_train.iloc[train_idx] if hasattr(
        X_train, "iloc"
    ) else X_train[train_idx]

    X_fold_val = X_train.iloc[val_idx] if hasattr(
        X_train, "iloc"
    ) else X_train[val_idx]

    y_fold_train = y_train[train_idx]

    model = ExtraTreesRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_fold_train, y_fold_train)

    # 4. Predict students not used for this fold's training
    predictions = model.predict(X_fold_val)
    oof_predictions[val_idx] = predictions

    # 5. Measure disagreement between trees on validation students
    individual_tree_predictions = np.array([
        tree.predict(X_fold_val)
        for tree in model.estimators_
    ])

    tree_disagreement[val_idx] = np.std(
        individual_tree_predictions,
        axis=0
    )

    print(f"Completed fold {fold}/5")

# 6. Calculate actual errors
absolute_errors = np.abs(y_train - oof_predictions)

print("\nOut-of-fold results")
print("-------------------")
print(f"MAE:  {mean_absolute_error(y_train, oof_predictions):.3f}")
print(
    "RMSE: "
    f"{np.sqrt(mean_squared_error(y_train, oof_predictions)):.3f}"
)

print(
    f"Mean tree disagreement: {tree_disagreement.mean():.4f}"
)
print(
    f"Maximum tree disagreement: {tree_disagreement.max():.4f}"
)

# 7. Build the results table
results = pd.DataFrame({
    "Actual_Grade": y_train,
    "Predicted_Grade": oof_predictions,
    "Absolute_Error": absolute_errors,
    "Tree_Disagreement": tree_disagreement
})

# 8. Divide students into three groups
results["Uncertainty_Group"] = pd.qcut(
    results["Tree_Disagreement"].rank(method="first"),
    q=3,
    labels=["Lower", "Middle", "Higher"]
)

# 9. Compare actual errors across groups
summary = results.groupby(
    "Uncertainty_Group",
    observed=True
).agg(
    Student_Count=("Absolute_Error", "size"),
    Mean_Tree_Disagreement=("Tree_Disagreement", "mean"),
    Mean_Absolute_Error=("Absolute_Error", "mean"),
    Median_Absolute_Error=("Absolute_Error", "median")
).reset_index()

print("\nError by Tree-Disagreement Group")
print("--------------------------------")
print(summary.to_string(index=False))

# 10. Save the results
results.to_csv("uncertainty_predictions.csv", index=False)
summary.to_csv("uncertainty_summary.csv", index=False)

print("\nSaved uncertainty_predictions.csv")
print("Saved uncertainty_summary.csv")
