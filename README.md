# 🧠 Stroke Risk Prediction Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)
![LightGBM](https://img.shields.io/badge/LightGBM-Best_Model-brightgreen)
![Models](https://img.shields.io/badge/Models_Tested-15-teal)
![AUC](https://img.shields.io/badge/Best_ROC--AUC-0.8446-success)

> **Team Nexus | Cohort 5 | Data Science Capstone | March 2026**  
> A team project applying machine learning to predict stroke risk from patient clinical data — 15 models trained, evaluated, and deployed.

---

##  Team Nexus

We are a team of data scientists passionate about using machine learning to solve real-world healthcare problems. This capstone project demonstrates an end-to-end data science pipeline — from raw patient data to a deployed predictive model — applied to one of the most critical challenges in public health.

---

## Overview

Stroke is one of the leading causes of death and long-term disability worldwide. This project builds a machine learning system to predict stroke risk from routine patient data, enabling early preventive intervention before a stroke occurs.

**The core challenge:** only 4.87% of patients experienced a stroke (249 of 5,109). Standard models exploit this imbalance to achieve high accuracy while catching almost no stroke patients. Every design decision in this project was built around solving that problem.

---

##  Dataset

**Source:** [Kaggle — fedesoriano (2021)](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)  
**Records:** 5,109 patients after cleaning | **Target:** stroke (0 = No Stroke, 1 = Stroke)

| Feature | Type | Description |
|---------|------|-------------|
| age | Integer | Patient age |
| gender | Categorical | Male / Female |
| hypertension | Binary | Has hypertension (0/1) |
| heart_disease | Binary | Has heart disease (0/1) |
| ever_married | Categorical | Yes / No |
| work_type | Categorical | Private, Self-employed, Govt, etc. |
| Residence_type | Categorical | Urban / Rural |
| avg_glucose_level | Float | Average blood glucose (mg/dL) |
| bmi | Float | Body Mass Index — 201 nulls imputed with median |
| smoking_status | Categorical | Formerly smoked, Never, Smokes, Unknown |

---

##  Results — All 15 Models

| Rank | Model | ROC-AUC | Recall | F1 Score | Accuracy |
|------|-------|---------|--------|----------|----------|
|  1 | **LightGBM** | **0.8446** | **72.0%** | **0.3038** | 83.86% |
| 2 | Logistic Regression | 0.8366 | 80.0% | 0.2299 | 73.78% |
| 3 | HistGradientBoosting | 0.8356 | 70.0% | 0.2734 | 81.80% |
| 4 | EasyEnsemble | 0.8355 | 84.0% | 0.1935 | 65.75% |
| 5 | SVC | 0.8278 | 78.0% | 0.1965 | 68.79% |
| 6 | Balanced Bagging | 0.8149 | 78.0% | 0.2549 | 77.69% |
| 7 | Balanced RF | 0.8142 | 84.0% | 0.2074 | 68.59% |
| 8 | Decision Tree | 0.7986 | 78.0% | 0.2261 | 73.87% |
| 9 | Naive Bayes | 0.7926 | 78.0% | 0.1526 | 57.63% |
| 10 | AdaBoost | 0.7883 | 84.0% | 0.2014 | 67.42% |
| 11 | Random Forest | 0.7783 | 10.0% | 0.1667 | 95.11% |
| 12 | SMOTETomek + RF | 0.7753 | 46.0% | 0.2018 | 82.19% |
| 13 | XGBoost | 0.7645 | 8.0% | 0.1127 | 93.84% |
| 14 | CatBoost | N/A* | 80.0% | 0.2402 | 75.24% |
| 15 | KNN | 0.6504 | 16.0% | 0.1250 | 89.04% |

>  **The Accuracy Trap:** Random Forest (95%) and XGBoost (94%) look strong on accuracy — but catch only **10%** and **8%** of actual stroke patients. This is why we optimised for ROC-AUC and Recall, not accuracy.  
> *\* CatBoost AUC not printed in notebook — to be confirmed on re-run.*

---

## Methodology

### Pipeline Architecture — Prevents Data Leakage
python
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('model', LGBMClassifier(is_unbalance=True, random_state=42, verbosity=-1))
])

grid = GridSearchCV(pipeline, param_grid,
                    cv=StratifiedKFold(n_splits=5),
                    scoring='f1', n_jobs=-1)


### Class Imbalance Strategies

| Strategy | Applied To |
|----------|-----------|
| SMOTE (inside Pipeline) | LR, RF, XGBoost, SVC, AdaBoost, KNN, Naive Bayes, CatBoost |
| SMOTETomek (inside Pipeline) | SMOTETomek + RF |
| Built-in balancing | EasyEnsemble, Balanced Bagging, Balanced RF |
| `class_weight='balanced'` | Decision Tree, HistGradientBoosting |
| `is_unbalance=True` | LightGBM |

---

##  Key Findings

- **Top 3 predictors:** BMI → Avg Glucose Level → Age (LightGBM feature importance)
- **Patients aged 65+** account for **63.9%** of all stroke cases (159 of 249)
- **Former smokers** had the highest stroke-to-population ratio of all smoking categories
- **Optimal deployment threshold:** 0.30–0.40 catches 82–84% of stroke patients vs 72% at default 0.50
- **Logistic Regression** has the highest Recall (80%) — best choice where maximum sensitivity is needed

---

##  Installation

bash
git clone https://github.com/team-nexus/stroke-risk-prediction
cd stroke-risk-prediction

pip install pandas numpy matplotlib seaborn scikit-learn xgboost \
            lightgbm catboost imbalanced-learn joblib


---

##  Usage

bash
# Run the full notebook
jupyter notebook CAPSTONE_COMBINED.ipynb


```python
# Load and use the saved model
import joblib

model = joblib.load('best_stroke_model.pkl')
probability = model.predict_proba(patient_data)[:, 1][0]
print(f"Stroke Risk: {probability:.2%}")


---

##  Ethics

- Model is a *screening tool only* — all final decisions must remain with a qualified clinician
- Dataset is *59% female* — model fairness must be audited across demographic groups before deployment
- Full *HIPAA / GDPR compliance* required for any real-world use
- Patients must be *informed* when AI contributes to their risk assessment
- *No high-risk score* should ever be used to deny care, insurance, or employment

---

##  References

- Fedesoriano (2021). Stroke Prediction Dataset. Kaggle.
- Chawla et al. (2002). SMOTE: Synthetic minority over-sampling technique. JAIR.
- Ke et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. NeurIPS.
- Pedregosa et al. (2011). Scikit-learn: Machine learning in Python. JMLR.

---

Team Nexus | Cohort 5 | Data Science Capstone | March 2026  
Best model saved: best_stroke_model.pkl (687.6 KB)
