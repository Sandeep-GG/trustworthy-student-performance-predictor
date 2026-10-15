
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------------------------
# 1. Configure the application
# ------------------------------------------

st.set_page_config(
    page_title="Trustworthy Student Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Trustworthy Student Performance Predictor")

st.write(
    "Predict a student's final grade and examine "
    "the model's explanations and uncertainty indicators."
)

st.warning(
    "This model provides estimates, not guaranteed results. "
    "Do not use its predictions alone to make important "
    "decisions about a student."
)

# ------------------------------------------
# 2. Load saved model and preprocessor
# ------------------------------------------

@st.cache_resource
def load_files():
    model = joblib.load("best_student_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor

try:
    model, preprocessor = load_files()
except Exception as e:
    st.error(
        "Could not load the model files. Make sure "
        "best_student_model.pkl and preprocessor.pkl "
        "are in the same folder as app.py."
    )
    st.stop()

# ------------------------------------------
# 3. Student input form
# ------------------------------------------

st.header("Enter Student Details")

with st.form("student_form"):

    st.subheader("Personal Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        school = st.selectbox("School", ["GP", "MS"])
        sex = st.selectbox("Sex", ["F", "M"])
        age = st.number_input(
            "Age", min_value=15, max_value=22, value=16
        )

    with col2:
        address = st.selectbox("Address", ["U", "R"])
        famsize = st.selectbox("Family Size", ["GT3", "LE3"])
        Pstatus = st.selectbox(
            "Parents' Cohabitation Status", ["T", "A"]
        )

    with col3:
        Medu = st.selectbox("Mother's Education (0-4)", [0, 1, 2, 3, 4], index=2)
        Fedu = st.selectbox("Father's Education (0-4)", [0, 1, 2, 3, 4], index=2)
        guardian = st.selectbox(
            "Guardian", ["mother", "father", "other"]
        )

    st.subheader("Family and School")

    col1, col2, col3 = st.columns(3)

    with col1:
        Mjob = st.selectbox(
            "Mother's Job",
            ["teacher", "health", "services", "at_home", "other"]
        )
        Fjob = st.selectbox(
            "Father's Job",
            ["teacher", "health", "services", "at_home", "other"]
        )
        reason = st.selectbox(
            "Reason for School Choice",
            ["home", "reputation", "course", "other"]
        )

    with col2:
        traveltime = st.selectbox("Travel Time (1-4)", [1, 2, 3, 4])
        studytime = st.selectbox("Weekly Study Time (1-4)", [1, 2, 3, 4])
        failures = st.number_input(
            "Past Class Failures", min_value=0, max_value=4, value=0
        )

    with col3:
        schoolsup = st.selectbox("Extra School Support", ["yes", "no"])
        famsup = st.selectbox("Family Support", ["yes", "no"])
        paid = st.selectbox("Extra Paid Classes", ["yes", "no"])

    st.subheader("Lifestyle and Activities")

    col1, col2, col3 = st.columns(3)

    with col1:
        activities = st.selectbox("Extracurricular Activities", ["yes", "no"])
        nursery = st.selectbox("Attended Nursery School", ["yes", "no"])
        higher = st.selectbox("Wants Higher Education", ["yes", "no"])

    with col2:
        internet = st.selectbox("Internet Access", ["yes", "no"])
        romantic = st.selectbox("In a Romantic Relationship", ["yes", "no"])
        famrel = st.selectbox("Family Relationship Quality (1-5)", [1, 2, 3, 4, 5])

    with col3:
        freetime = st.selectbox("Free Time (1-5)", [1, 2, 3, 4, 5])
        goout = st.selectbox("Going Out (1-5)", [1, 2, 3, 4, 5])
        health = st.selectbox("Health (1-5)", [1, 2, 3, 4, 5])

    st.subheader("Daily Habits")

    col1, col2, col3 = st.columns(3)

    with col1:
        Dalc = st.selectbox("Workday Alcohol Consumption (1-5)", [1, 2, 3, 4, 5])

    with col2:
        Walc = st.selectbox("Weekend Alcohol Consumption (1-5)", [1, 2, 3, 4, 5])

    with col3:
        absences = st.number_input(
            "School Absences", min_value=0, max_value=100, value=0
        )

    submitted = st.form_submit_button("Predict Final Grade")

# ------------------------------------------
# 4. Make prediction
# ------------------------------------------

if submitted:

    student = {
        "school": school,
        "sex": sex,
        "age": age,
        "address": address,
        "famsize": famsize,
        "Pstatus": Pstatus,
        "Medu": Medu,
        "Fedu": Fedu,
        "Mjob": Mjob,
        "Fjob": Fjob,
        "reason": reason,
        "guardian": guardian,
        "traveltime": traveltime,
        "studytime": studytime,
        "failures": failures,
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": paid,
        "activities": activities,
        "nursery": nursery,
        "higher": higher,
        "internet": internet,
        "romantic": romantic,
        "famrel": famrel,
        "freetime": freetime,
        "goout": goout,
        "Dalc": Dalc,
        "Walc": Walc,
        "health": health,
        "absences": absences
    }

    student_df = pd.DataFrame([student])

    try:
        processed = preprocessor.transform(student_df)

        prediction = float(model.predict(processed)[0])

        st.header("Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Final Grade",
                f"{prediction:.2f} / 20"
            )

        # ------------------------------------------
        # 5. Tree disagreement
        # ------------------------------------------

        
        if hasattr(model, "estimators_"):
            tree_predictions = np.array([
                tree.predict(processed)[0]
                for tree in model.estimators_
            ])

            disagreement = float(np.std(tree_predictions))

            # Load cross-validation results
            validation_predictions = pd.read_csv(
                "uncertainty_predictions.csv"
            )
            validation_summary = pd.read_csv(
                "uncertainty_summary.csv"
            )

            # Calculate disagreement thresholds
            lower_cutoff, upper_cutoff = (
                validation_predictions["Tree_Disagreement"]
                .quantile([1 / 3, 2 / 3])
                .tolist()
            )

            # Classify the student's disagreement
            if disagreement <= lower_cutoff:
                group = "Lower"
            elif disagreement <= upper_cutoff:
                group = "Middle"
            else:
                group = "Higher"

            with col2:
                st.metric(
                    "Tree Disagreement",
                    f"{disagreement:.2f} grade points"
                )

            st.subheader("Prediction Reliability Check")
            st.write(f"Disagreement group: **{group}**")

            st.caption(
                f"Validation thresholds: Lower ≤ {lower_cutoff:.2f}, "
                f"Middle ≤ {upper_cutoff:.2f}, "
                f"Higher > {upper_cutoff:.2f}."
            )

            # Show the historical error for this group
            group_row = validation_summary[
                validation_summary["Uncertainty_Group"] == group
            ]

            if not group_row.empty:
                observed_mae = float(
                    group_row["Mean_Absolute_Error"].iloc[0]
                )

                st.write(
                    "Average absolute error observed for this group "
                    f"during cross-validation: **{observed_mae:.2f} "
                    "grade points**."
                )

            st.warning(
                "This is a historical group average, not a guarantee "
                "of the error for this individual student. Tree "
                "disagreement is not a calibrated confidence score."
            )


        # ------------------------------------------
        # 6. Explain important features
        # ------------------------------------------

        st.subheader("What Factors Does the Model Rely On?")

        importance_file = "permutation_importance.csv"

        try:
            importance_df = pd.read_csv(importance_file)
            top_features = importance_df.head(5).copy()

            top_features["Feature"] = (
                top_features["Feature"]
                .str.replace("numerical__", "", regex=False)
                .str.replace("categorical__", "", regex=False)
            )

            

            
            st.write("Top 5 important features")

            for _, row in top_features.iterrows():
                st.write(
                f"**{row['Feature']}** — "
                f"Importance: {row['Importance']:.4f}"
                )


            st.caption(
                "These are overall model-level importance values, "
                "not proof of why this individual student received "
                "this specific prediction."
            )

        except FileNotFoundError:
            st.info(
                "Run explain_model.py first to generate "
                "permutation_importance.csv."
            )

        st.info(
            "The model does not use G1 or G2 (earlier grades). "
            "It estimates G3 from the other recorded student features."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
