# ML Internship

Three machine learning projects from my Machine Learning internship at Progree . Each project has its own folder with a notebook, a README with results, and the data or a link to it. All notebooks run in Google Colab.

**Author:** Zoya Ali, Computer Science graduate 

## Projects

| Project | What it does | Tools | Key result |
|---|---|---|---|
| [Task 2: Ridge regression](task2-ridge-regression) | Predicts car prices from 25 features, with imputation, scaling and one-hot encoding in one pipeline | scikit-learn, pandas | R² 0.925 and RMSE $3,024 on 41 test cars |
| [Task 3: CNN on MNIST](task3-cnn-mnist) | Classifies handwritten digits with a CNN, data augmentation and dropout | TensorFlow / Keras | 99.42% test accuracy on MNIST (58 wrong out of 10,000) |
| [Task 4: Clinical risk predictor](task4-clinical-predictor) | Finds high-risk thoracic surgery patients in imbalanced data (15% high-risk) with SMOTENC, SVM, Random Forest and XGBoost | scikit-learn, imbalanced-learn, XGBoost | SVM chosen by recall found 12 of 14 high-risk test patients, but flagged 77 of 94 patients; XGBoost has the best ROC-AUC (0.634) |

## What I focused on
- No data leakage: preprocessing, oversampling and tuning are done inside pipelines and cross-validation, and the test set is used once.
- Honest evaluation: baselines, confidence intervals and several metrics instead of one headline number. Task 4 shows why recall alone can pick a model that is almost the same as flagging every patient.

## Repository structure
```
progree-ml-internship/
  README.md
  task2-ridge-regression/
    Task2_Ridge_Regression.ipynb
    README.md
    task2_results.csv
    predicted_vs_real.png
  task3-cnn-mnist/
    Task3_CNN_MNIST.ipynb
    README.md
    training_curves.png
    confusion_matrix.png
  task4-clinical-predictor/
    Task4_Clinical_Predictor.ipynb
    README.md
    model_comparison_results.csv
    data/ThoracicSurgery.arff
```

## How to run
1. Open [Google Colab].
2. Choose File, then Upload notebook (or File, then Open notebook, then the GitHub tab), and select a notebook.
3. Run the cells from top to bottom. Task 4 asks for the file in its `data` folder.
