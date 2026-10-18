# Task 4: High-Imbalance Clinical Disease Predictor

Predict one-year risk after thoracic surgery from 16 clinical features, with a focus on **finding as many high-risk patients as possible (recall)** and reporting honestly how reliable the result is.

## Data
UCI Thoracic Surgery dataset: 470 patients, 17 columns, no missing values. Only 70 patients (14.9%) are high-risk (`Risk1Yr = T`).
Stratified 80/20 split: 376 training patients (56 high-risk) and 94 test patients (14 high-risk). The test set is used once, at the end.

## Method
- Pipeline: ordinal encoding, **SMOTENC** (oversampling for mixed numeric and categorical data), scaling and one-hot encoding, then the model. SMOTENC runs inside the pipeline, so it only ever sees training data, also inside cross-validation.
- Models tuned with `GridSearchCV` and **stratified 5-fold cross-validation**, chosen by recall: SVM, Random Forest, XGBoost.
- Extra checks: ROC-AUC and PR-AUC, a "flag everyone" baseline, a 95% confidence interval for recall, and threshold tuning on training data only.

## Results (94 test patients, 14 high-risk)
| Model | CV recall | Test recall | Test precision | Test F1 | Test ROC-AUC | Test PR-AUC |
|---|---|---|---|---|---|---|
| SVM | 0.823 | 0.857 | 0.156 | 0.264 | 0.633 | 0.236 |
| Random Forest | 0.395 | 0.643 | 0.225 | 0.333 | 0.601 | 0.192 |
| XGBoost | 0.395 | 0.571 | 0.222 | 0.320 | 0.634 | 0.245 |
| Baseline: flag everyone | - | 1.000 | 0.149 | 0.259 | 0.500 | 0.149 |

## What the results mean
- By the task rule (best CV recall) the SVM is chosen: it found 12 of 14 high-risk test patients (recall 0.86, 95% interval 0.60 to 0.96).
- That SVM flags 77 of 94 patients and its precision (0.156) is almost the same as the share of high-risk patients (0.149). It behaves almost like "flag everyone". Optimising recall alone favours this kind of model.
- Random Forest and XGBoost flag far fewer patients (about 40 and 36) with precision about 1.5 times the base rate. XGBoost has the best test PR-AUC and ROC-AUC.
- The signal in this dataset is weak: cross-validated ROC-AUC is only 0.52 to 0.54, test ROC-AUC 0.60 to 0.63.

## Limitations
470 patients, only 14 high-risk test patients, one train/test split, and large variation between folds. This is a learning project and not a clinical tool.

## Files
- `Task4_Clinical_Predictor.ipynb`: the full notebook with outputs (runs in Google Colab)
- `data/ThoracicSurgery.arff`: the dataset from the UCI repository
- `model_comparison_results.csv`: test results table

## Run it
Open the notebook in Google Colab, run the cells from top to bottom, and upload the ARFF file when asked (or use the download option in Step 2).
