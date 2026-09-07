\# Data Contract — IEEE-CIS Fraud Detection



\## 1. Dataset



Project dataset:



IEEE-CIS Fraud Detection



Primary files:



\- train\_transaction.csv

\- train\_identity.csv



Primary join key:



TransactionID



Target:



isFraud



\---



\## 2. Transaction Data



Expected transaction rows:



590,540



Expected columns:



394



Primary identifier:



TransactionID



Target column:



isFraud



\---



\## 3. Target Definition



isFraud is a binary classification target.



Allowed values:



0 = legitimate transaction

1 = fraudulent transaction



Expected class distribution:



0 = 569,877

1 = 20,663



Approximate fraud rate:



3.50%



\---



\## 4. Identity Data



Expected identity rows:



144,233



Expected columns:



41



Join key:



TransactionID



Identity information is not available for every transaction.



Expected identity coverage:



approximately 24.42%



Therefore:



Missing identity information must NOT automatically be interpreted as fraud.



\---



\## 5. Transaction ID Rules



TransactionID must:



\- exist

\- be unique in transaction data

\- be unique in identity data

\- contain no unexpected duplicate records



Observed validation:



Transaction IDs are unique.



\---



\## 6. Transaction Time



TransactionDT represents a relative transaction-time value.



Observed range:



Minimum: 86,400

Maximum: 15,811,131



TransactionDT will be used for temporal ordering.



Random splitting must be avoided when it would introduce future-information leakage.



\---



\## 7. Transaction Amount



TransactionAmt must:



\- exist

\- be numeric

\- be greater than zero



Observed:



Minimum: 0.251

Maximum: 31,937.391

Mean: 135.0272



No zero or negative transaction amounts were observed.



\---



\## 8. Missing Values



Missing values are expected in this dataset.



Missing values must NOT be removed blindly.



Feature-specific strategies will be evaluated, including:



\- numerical imputation

\- categorical imputation

\- missing-value indicators

\- models that naturally handle missing values



\---



\## 9. Data Leakage Rules



The following must be avoided:



\- using isFraud as an input feature

\- using future information to predict past transactions

\- fitting preprocessing transformations on validation/test data

\- calculating target-derived features using validation/test labels

\- allowing test information to influence model training



\---



\## 10. Join Rules



Transaction and identity data will be joined using:



TransactionID



The join must preserve transaction records.



A missing identity record does not mean the transaction is invalid.



\---



\## 11. Model Evaluation



Because the dataset is highly imbalanced, evaluation will include:



\- Precision

\- Recall

\- F1-score

\- ROC-AUC

\- PR-AUC

\- Confusion Matrix



Accuracy will not be treated as the primary metric.



\---



\## 12. Production Considerations



The final system should support:



\- reproducible preprocessing

\- versioned features

\- model versioning

\- batch inference

\- real-time inference

\- monitoring

\- data drift detection

\- model performance monitoring

\- explainability

