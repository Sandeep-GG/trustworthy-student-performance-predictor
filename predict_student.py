
import joblib
import numpy as np
import pandas as pd

# Step 1: Load the trained model and preprocessor
model = joblib.load("student_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

print("Student Performance Predictor")
print("-----------------------------")

# Step 2: Collect student information
student = {}

student["school"] = input("School (GP or MS): ").strip().upper()
student["sex"] = input("Sex (F or M): ").strip().upper()
student["age"] = int(input("Age: "))
student["address"] = input("Address (U or R): ").strip().upper()
student["famsize"] = input("Family size (GT3 or LE3): ").strip().upper()
student["Pstatus"] = input("Parents together (T or A): ").strip().upper()

student["Medu"] = int(input("Mother's education (0-4): "))
student["Fedu"] = int(input("Father's education (0-4): "))

student["Mjob"] = input(
    "Mother's job (teacher, health, services, at_home, other): "
).strip().lower()

student["Fjob"] = input(
    "Father's job (teacher, health, services, at_home, other): "
).strip().lower()

student["reason"] = input(
    "Reason for school choice (home, reputation, course, other): "
).strip().lower()

student["guardian"] = input(
    "Guardian (mother, father, other): "
).strip().lower()

student["traveltime"] = int(input("Travel time category (1-4): "))
student["studytime"] = int(input("Weekly study time category (1-4): "))
student["failures"] = int(input("Previous failures (0-3): "))

for column in [
    "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic"
]:
    student[column] = input(
        f"{column} (yes/no): "
    ).strip().lower()

student["famrel"] = int(input("Family relationship quality (1-5): "))
student["freetime"] = int(input("Free time (1-5): "))
student["goout"] = int(input("Going out with friends (1-5): "))
student["Dalc"] = int(input("Workday alcohol consumption (1-5): "))
student["Walc"] = int(input("Weekend alcohol consumption (1-5): "))
student["health"] = int(input("Health (1-5): "))
student["absences"] = int(input("Number of absences: "))

# Step 3: Convert input into a DataFrame
student_df = pd.DataFrame([student])

# Step 4: Apply the saved preprocessing
student_processed = preprocessor.transform(student_df)

# Step 5: Predict the final grade
prediction = model.predict(student_processed)[0]

# Step 6: Calculate disagreement between trees
tree_predictions = np.array([
    tree.predict(student_processed)[0]
    for tree in model.estimators_
])

uncertainty = tree_predictions.std()
lower = np.percentile(tree_predictions, 5)
upper = np.percentile(tree_predictions, 95)

if uncertainty <= 2:
    indicator = "Lower tree disagreement"
elif uncertainty <= 4:
    indicator = "Moderate tree disagreement"
else:
    indicator = "Higher tree disagreement"

# Step 7: Display results
print("\nPrediction Results")
print("------------------")
print(f"Predicted final grade: {prediction:.2f}/20")
print(f"Tree disagreement (standard deviation): {uncertainty:.2f}")
print(f"Tree prediction spread (5th-95th percentile): {lower:.2f} to {upper:.2f}")
print(f"Uncertainty indicator: {indicator}")

print("\nNote: This is an estimate, not a guaranteed result.")
print("Tree disagreement is not a calibrated confidence probability.")
