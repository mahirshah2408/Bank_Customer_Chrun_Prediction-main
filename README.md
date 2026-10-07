# Bank Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-2F80ED)

A notebook-based machine learning project that predicts whether a bank customer is likely to leave. It covers data exploration, preprocessing, class balancing, training, and comparison across several classification algorithms.

## Project workflow

1. Load and explore the 10,000-row customer dataset.
2. Remove identifier columns: `RowNumber`, `CustomerId`, and `Surname`.
3. One-hot encode categorical features.
4. Separate the target column, `Exited`.
5. Balance the classes with SMOTE.
6. Split the data into 70% training and 30% testing sets.
7. Scale features with `StandardScaler`.
8. Train and compare seven classifiers.

## Models evaluated

- Logistic Regression
- Support Vector Classifier
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost

The notebook evaluates models with accuracy, precision, recall, and F1 score. In the saved notebook run, XGBoost produced the highest test accuracy at approximately **85.5%**, followed by Random Forest at approximately **84.4%**.

## Dataset

The included `Churn_Modelling.csv` dataset contains customer demographics, account information, activity indicators, and the binary `Exited` target.

Key predictive fields include credit score, geography, gender, age, tenure, balance, product count, credit-card ownership, active-member status, and estimated salary.

## Run locally

Install the notebook dependencies:

```bash
pip install pandas numpy seaborn scikit-learn imbalanced-learn xgboost jupyter
```

Start Jupyter and open `bank_churn_customer_final_project.ipynb`:

```bash
jupyter notebook
```

## Evaluation note

The current notebook applies SMOTE before the train/test split. For a stricter production evaluation, resampling should be applied only to the training data, ideally inside a cross-validation pipeline, to prevent information from the test distribution influencing training.
