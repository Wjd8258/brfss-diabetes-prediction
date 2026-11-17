# Predicting Self-Reported Diagnosed Diabetes in Adults from 2024 BRFSS Data

**Author:** Jiadi Wu

**Course:** DSCI 7000 (Capstone) — M.S. in Artificial Intelligence, Hawaii Pacific University

**Tools:** Python 3.12, scikit-learn 1.8.0, R 4.3.3 (survey 4.2.1)

## Overview

Can seven routine survey features identify adults who report a diabetes diagnosis? Using the CDC's 2024 Behavioral Risk Factor Surveillance System (457,670 records), this project compares three classifiers — logistic regression, random forest, and gradient boosting — on the same seven features: BMI category, age group, leisure-time physical activity, sex, race/ethnicity, education, and household income.

The analysis sample is 441,934 adults (65,809 reporting a diagnosis; unweighted prevalence 14.89%, design-weighted 12.98%). The sample is split 70/30 stratified by outcome; hyperparameters are tuned by five-fold cross-validation on average precision; thresholds are set by out-of-fold F1. Description and association analyses use the complex survey design (weights, strata, PSUs); the main prediction analysis is unweighted, with a weighted sensitivity analysis.

## Results (test set, n = 132,581)

| Model | AP (95% CI) | ROC-AUC | Recall | Precision | F1 | Brier |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic regression | 0.333 (0.327–0.338) | 0.761 | 0.572 | 0.314 | 0.405 | 0.113 |
| Random forest | 0.336 (0.330–0.341) | 0.763 | 0.584 | 0.312 | 0.407 | 0.112 |
| Gradient boosting | 0.337 (0.332–0.343) | 0.764 | 0.623 | 0.303 | 0.408 | 0.112 |
| Baseline: age + BMI | 0.275 (0.271–0.278) | 0.728 | 0.665 | 0.265 | 0.379 | 0.116 |
| Baseline: random | 0.149 | 0.500 | — | — | — | 0.127 |

The headline finding: **the models identify the label at a moderate level, and the complex algorithms add almost nothing.** Gradient boosting beats logistic regression by an absolute AP difference of only 0.004380 (95% interval 0.002800–0.005962), and under the weighted evaluation all pairwise difference intervals include 0.

Adjusted odds ratios (design-weighted): obesity vs. normal weight **3.51** (3.29–3.74), age 65+ vs. 18–44 **9.95** (9.28–10.66), no leisure-time activity vs. any **1.55** (1.48–1.62). Permutation importance ranks the features age > BMI category > leisure-time activity. The models must not be used for clinical diagnosis, causal inference, or prediction of future onset — the outcome is a self-reported diagnosis history in a cross-sectional survey.

## Repo layout

```
├── README.md     # this file
└── REPORT.md     # full report (English): literature review, methods, results, discussion, references, appendices
```

The analysis scripts referenced in the report (`models.py`, `audit_data.py`, etc.) and the extracted BRFSS subset are not included here — the report PDF is the deliverable for this project.
