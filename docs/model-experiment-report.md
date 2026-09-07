\# Fraud Detection Model Experiment Report



\## 1. Objective



Build a machine learning system capable of identifying potentially fraudulent

transactions using the IEEE-CIS Fraud Detection dataset.



The project uses a chronological train/validation/test split to better

represent a real-world scenario where future transactions must be predicted

using historical data.



\---



\## 2. Dataset



Dataset:



IEEE-CIS Fraud Detection



Final merged dataset:



\- Transactions: 590,540

\- Features after merging: 434

\- Target: `isFraud`

\- Fraud transactions: 20,663

\- Legitimate transactions: 569,877

\- Overall fraud rate: approximately 3.50%

\- Identity coverage: approximately 24.42%



The transaction and identity datasets were joined using `TransactionID`.



\---



\## 3. Data Splitting Strategy



A chronological split was used instead of a random split.



| Split | Rows | Fraud Rate |

|---|---:|---:|

| Train | 413,378 | 3.5169% |

| Validation | 88,581 | 3.4341% |

| Test | 88,581 | 3.4804% |



The split was performed using `TransactionDT`.



This helps reduce the risk of temporal leakage and provides a more realistic

estimate of future model performance.



\---



\## 4. Feature Preparation



The model excludes:



\- `TransactionID`

\- `isFraud`



The initial feature set contained 432 usable features.



Feature categories include:



\- Transaction features

\- Card features

\- C features

\- D features

\- M features

\- V features

\- Identity features



The dataset contains substantial missing data.



Missingness analysis showed that missingness itself can contain predictive

information. Therefore, the best model tested also included missingness

indicators.



\---



\# 5. Experiment 1 — Logistic Regression Baseline



\## Purpose



Establish a simple baseline before introducing a more powerful model.



A reduced set of transaction/card/domain features was used.



Class imbalance was handled using:



`class\_weight="balanced"`



\## Validation Results



| Metric | Result |

|---|---:|

| Precision | 0.0471 |

| Recall | 0.7275 |

| F1 | 0.0884 |

| ROC-AUC | 0.6845 |

| PR-AUC | 0.0941 |



\## Conclusion



The baseline achieved relatively high recall but very poor precision.



It generated a large number of false positives.



The baseline was therefore not suitable as the primary fraud detection model,

but it provided a useful reference point.



\---



\# 6. Experiment 2 — XGBoost



\## Purpose



Use a nonlinear gradient-boosted tree model capable of learning complex

relationships across the large feature set.



The model used:



\- 1,000 estimators

\- Maximum depth: 6

\- Learning rate: 0.05

\- Subsample: 0.8

\- Column subsampling: 0.8

\- `min\_child\_weight`: 5

\- Histogram tree construction

\- `scale\_pos\_weight` for class imbalance



\## Validation Results



| Metric | Result |

|---|---:|

| Precision | 0.5067 |

| Recall | 0.5713 |

| F1 | 0.5371 |

| ROC-AUC | 0.9206 |

| PR-AUC | 0.5815 |



\## Conclusion



XGBoost produced a major improvement over the logistic regression baseline.



Validation PR-AUC increased from:



`0.0941 → 0.5815`



This became the first strong model candidate.



\---



\# 7. Experiment 3 — Missingness-Aware XGBoost



\## Motivation



The dataset contains extensive missingness.



Missingness analysis showed that the number and pattern of missing V-features

were associated with substantially different fraud rates.



Instead of treating missing values only as values requiring imputation,

explicit missingness indicators were added.



The model therefore used:



\- Original features

\- Missingness indicators

\- Numeric median imputation

\- Categorical most-frequent imputation

\- Ordinal encoding for categorical features



Total model features after adding indicators:



864



\## Validation Results



| Metric | Original XGBoost | Missingness-Aware |

|---|---:|---:|

| Precision | 0.5067 | \*\*0.5185\*\* |

| Recall | 0.5713 | 0.5664 |

| F1 | 0.5371 | \*\*0.5414\*\* |

| ROC-AUC | 0.9206 | 0.9197 |

| PR-AUC | 0.5815 | \*\*0.5827\*\* |



\## Conclusion



Adding missingness indicators produced a small improvement.



The model reduced false positives while maintaining similar recall.



The missingness-aware model became the project champion.



\---



\# 8. Experiment 4 — Regularized XGBoost



\## Motivation



The missingness-aware model showed a significant difference between training

and validation performance.



Therefore, a more strongly regularized configuration was tested.



Changes included:



\- Maximum depth reduced from 6 to 4

\- `min\_child\_weight` increased to 10

\- Column subsampling reduced to 0.7

\- `gamma` increased

\- L1 regularization added

\- L2 regularization increased



\## Results



| Metric | Training | Validation |

|---|---:|---:|

| Precision | 0.2771 | 0.2897 |

| Recall | 0.8626 | 0.6788 |

| F1 | 0.4194 | 0.4061 |

| ROC-AUC | 0.9603 | 0.9099 |

| PR-AUC | 0.7243 | 0.5410 |



\## Conclusion



Regularization reduced the training/validation performance gap but also

reduced validation performance substantially.



Validation PR-AUC decreased:



`0.5827 → 0.5410`



The configuration was therefore rejected.



\---



\# 9. Threshold Optimization



The model's probability threshold was evaluated independently of PR-AUC.



The default threshold of 0.50 was compared with alternative thresholds.



\## Validation Results



| Threshold | Precision | Recall | F1 |

|---:|---:|---:|---:|

| 0.30 | 31.56% | 70.38% | 43.58% |

| 0.40 | 41.19% | 63.45% | 49.95% |

| 0.50 | 51.85% | 56.64% | 54.14% |

| 0.55 | 57.54% | 53.55% | 55.47% |

| \*\*0.60\*\* | \*\*62.90%\*\* | \*\*50.33%\*\* | \*\*55.92%\*\* |

| 0.70 | 72.22% | 44.18% | 54.82% |

| 0.80 | 80.43% | 38.23% | 51.83% |



The threshold of 0.60 produced the highest F1 among the tested thresholds.



For the project's default operating policy, threshold 0.60 was selected based

on validation data.



Threshold 0.50 can be used when higher recall is preferred, while higher

thresholds can be used when reducing false positives is more important.



\---



\# 10. Final Test Evaluation



The test set was not used during model selection.



After model and threshold selection, the champion model was evaluated on the

unseen chronological test set.



\## Threshold-Independent Metrics



| Metric | Test Result |

|---|---:|

| \*\*PR-AUC\*\* | \*\*0.5203\*\* |

| \*\*ROC-AUC\*\* | \*\*0.8989\*\* |



\## Test Performance



\### Threshold 0.50



\- Precision: 44.21%

\- Recall: 50.92%

\- F1: 47.33%

\- False Positive Rate: 2.3170%

\- True Positives: 1,570

\- False Positives: 1,981

\- False Negatives: 1,513

\- True Negatives: 83,517



\### Threshold 0.60



\- Precision: 55.22%

\- Recall: 45.83%

\- F1: 50.09%

\- False Positive Rate: 1.3404%

\- True Positives: 1,413

\- False Positives: 1,146

\- False Negatives: 1,670

\- True Negatives: 84,352



\### Threshold 0.70



\- Precision: 65.50%

\- Recall: 41.13%

\- F1: 50.53%

\- False Positive Rate: 0.7813%

\- True Positives: 1,268

\- False Positives: 668

\- False Negatives: 1,815

\- True Negatives: 84,830



\---



\# 11. Final Model Selection



\## Selected Model



Missingness-aware XGBoost.



Model:



`models/xgboost\_missing\_model.joblib`



Preprocessor:



`models/xgboost\_missing\_preprocessor.joblib`



\## Selected Validation Threshold



`0.60`



\## Final Unseen Test Performance



\- PR-AUC: \*\*0.5203\*\*

\- ROC-AUC: \*\*0.8989\*\*

\- Precision at 0.60: \*\*55.22%\*\*

\- Recall at 0.60: \*\*45.83%\*\*

\- F1 at 0.60: \*\*50.09%\*\*



\---



\# 12. Important Evaluation Principle



The test set was intentionally evaluated only after model selection.



The test results should not be used to repeatedly tune the model or threshold.



This preserves the role of the test set as an unbiased final evaluation.



Although threshold 0.70 produced a slightly higher F1 on this particular test

set, it was not selected after observing the test results.



The project's reported operating threshold therefore remains 0.60 because it was

selected using validation data before the test evaluation.



\---



\# 13. Key Findings



1\. XGBoost substantially outperformed the logistic regression baseline.



2\. Missingness contains useful predictive information in this dataset.



3\. Missingness indicators produced a modest but measurable improvement.



4\. Strong regularization reduced overfitting but hurt predictive performance.



5\. Threshold selection significantly changes the precision/recall tradeoff.



6\. The model generalizes to an unseen chronological test period.



7\. Test performance is lower than validation performance, demonstrating the

importance of evaluating on truly unseen data.



8\. PR-AUC is particularly useful for this problem because fraud is a minority

class.



\---



\# 14. Next Engineering Steps



The model development phase is complete.



The next phase will focus on production ML engineering:



1\. SHAP model explainability

2\. MLflow experiment tracking

3\. MLflow model registry

4\. FastAPI inference service

5\. PostgreSQL prediction/audit storage

6\. Redis for low-latency data

7\. Kafka transaction streaming

8\. PySpark streaming/feature processing

9\. Airflow orchestration

10\. Docker containerization

11\. CI/CD with GitHub Actions

12\. Kubernetes deployment

13\. Prometheus/Grafana monitoring

14\. Data and model drift monitoring

15\. Cloud deployment

16\. Infrastructure as Code with Terraform



The objective is to transform the validated ML model into a complete,

production-style fraud detection platform.

