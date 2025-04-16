# Machine Learning Assignment 1: Y4S1

# Credit Risk Assessment: Predicting Loan Default with Logistic Regression & CatBoost

This assignment implements supervised machine learning techniques to a real-world credit risk dataset to **predict loan defaults**. It was developed individually for the Software Engineering Y4S1 Machine Learning Module Assignment.

---

## 📁 Dataset

- **Source:** [Kaggle - Loan Default Prediction](https://www.kaggle.com/datasets/nikhil1e9/loan-default/data)
- **Size:** 255,347 rows × 18 columns
- **Target Variable:** `Default` (0 = No Default, 1 = Default)
- **Context:** Includes features like age, income, loan amount, credit score, employment history, and more.

---

## 🧪 Objective

To evaluate and compare two supervised learning algorithms:
1. **Logistic Regression with ElasticNet regularization**
2. **CatBoost Classifier** (gradient boosting model)

The goal is to build a model that accurately predicts whether a borrower is likely to default on a loan — with special emphasis on **recall** and **ROC AUC** due to class imbalance.

---

## 🔍 Methodology

- ✅ **Data Cleaning:** Outlier capping, duplicate removal
- ✅ **Encoding:** Categorical to numerical mapping
- ✅ **Feature Engineering:** Derived risk-related ratios and interaction terms
- ✅ **Feature Reduction:** Correlation-based multicollinearity check
- ✅ **Train/Test Split:** Stratified 70/30 split for balanced class distribution
- ✅ **Pipelines:** Built with `scikit-learn` using preprocessing + model tuning
- ✅ **Hyperparameter Tuning:** `GridSearchCV` with cross-validation
- ✅ **Evaluation Metrics:** Accuracy, Precision, Recall, F1 Score, ROC AUC, PR Curve


---

## 📊 Results Summary

| Metric         | Logistic Regression | CatBoost Classifier |
|----------------|---------------------|----------------------|
| Accuracy       | 0.685               | 0.656                |
| Precision      | 0.224               | 0.214                |
| Recall         | 0.696               | ✅ **0.734**          |
| F1 Score       | 0.339               | 0.331                |
| ROC AUC        | 0.755               | ✅ **0.759**          |
| Average Precision | 0.32            | ✅ **0.33**           |

---

## 📈 Visualizations Included

- Confusion Matrices
- ROC Curve Comparison
- Precision-Recall Curve
- Hyperparameter Tuning Plots (ElasticNet)
- Classification Reports

---

## 🏁 Conclusion

CatBoost outperformed Logistic Regression in recall, F1-score, and AUC — making it the preferred model for identifying loan defaulters. Logistic Regression remains useful for interpretability.

