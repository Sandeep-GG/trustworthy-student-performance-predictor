# Trustworthy Student Performance Predictor

## Project Overview

This project predicts a student's final academic grade using student-related information and machine learning.

It also includes model evaluation, feature importance, and an uncertainty analysis to help users understand the limitations of individual predictions.

The project is built with Python, Scikit-learn, Pandas, NumPy, and Streamlit.

## Features

- Predicts final student grades on a scale of 0–20.
- Compares Linear Regression, Random Forest, and Extra Trees models.
- Evaluates performance using MAE, RMSE, and R².
- Uses cross-validation to evaluate model stability.
- Analyzes important features using permutation importance.
- Examines disagreement between decision trees as a warning signal.
- Provides an interactive Streamlit web application.
- Excludes earlier grades G1 and G2 from prediction.

## Dataset

The project uses the UCI Student Performance dataset.

Target variable: `G3` (final grade).

The model uses other available student attributes and excludes `G1` and `G2` to avoid relying on earlier grades.

## Model Performance

The selected Extra Trees model produced the following results on the project's test split:

| Metric | Result |
|---|---:|
| MAE | 2.108 |
| RMSE | 2.793 |
| R² | 0.2003 |

These are initial experimental results, not a guarantee of future performance. The test split was also used during model comparison, so an independent test set would be needed for an unbiased final evaluation.

## Trustworthiness and Limitations

- Tree disagreement is a warning signal, not a calibrated confidence score.
- Historical errors for groups with similar disagreement are not guarantees for individual predictions.
- Feature importance describes model behavior and does not establish causation.
- Predictions may be inaccurate and should not be used as the sole basis for educational decisions.
- The dataset may not represent students from every school, region, or background.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Joblib

## How to Run

1. Clone or download this repository.
2. Create and activate a Python virtual environment.
3. Install the dependencies from `requirements.txt`.
4. Ensure the trained model and preprocessing files are available.
5. Run the application:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

## Future Improvements

- Evaluate the model on a completely independent test set.
- Improve predictive performance through careful feature engineering and model tuning.
- Evaluate uncertainty estimates for calibration and error detection.
- Investigate fairness across relevant student groups.
- Add automated tests and reproducible training instructions.

## Disclaimer

This project is an educational machine learning project. Predictions are estimates and should not be treated as definitive judgments about a student's ability or future.
