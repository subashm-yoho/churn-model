# Telco Churn Model

Branch: feature/churn-model

- churn_eda.ipynb: Task 1 (data checks, 20% hold-out split, 5 EDA charts, assumptions, metric)
- churn_baselines.ipynb: Task 2 (feature engineering, pipelines, baseline CV)
- results/baseline_results.csv: created after running churn_baselines.ipynb

Dataset: Telco Customer Churn. Put WA_Fn-UseC_-Telco-Customer-Churn.csv in the same folder if no internet.
## Task 3: XGBoost, Tuning and Comparison
- Best model: XGBoost (tuned), test ROC-AUC 0.846, F1 0.625
- Logistic Regression (tuned) 0.841, Random Forest 0.822
- Threshold 0.56 chosen to catch 75% of churners
- Top drivers: month-to-month contract, no online security, no tech support, fiber optic, electronic check payment

## Task 3: XGBoost, Tuning and Comparison
- xgboost_tuning_comparison.py: Task 3 (XGBoost pipeline, RandomizedSearchCV tuning, comparison, threshold, feature importance)
- outputs/: comparison_table.csv, roc_comparison.png, feature_importance.png, threshold_choice.png
- Best model: XGBoost (tuned), test ROC-AUC 0.846, F1 0.625
- Logistic Regression (tuned) 0.841, Random Forest 0.822
- Threshold 0.56 chosen to catch 75% of churners
- Top drivers: month-to-month contract, no online security, no tech support, fiber optic, electronic check payment 

## Task 1 - Concepts, Splits and Leakage

- Script: `splits_leakage.py` (output in `output.txt`, plot in `overfitting_plot.png`)
- Split: 60/20/20, stratified, random_state=42. Test set kept aside and not used.
- Overfitting: Decision Tree depth 1 to None. Train accuracy goes 0.73 to 1.00 but validation stays around 0.73-0.79, so deep trees overfit.
- CV: 5-fold LogisticRegression gives 0.81 +/- 0.01, a stable estimate.
- Leakage: scaling before the split leaks test statistics into training. Fix: split first, fit the scaler on train only. Accuracy was same here (0.8038) because the dataset is large, but the method is still wrong.
- Supervised: spam detection, house price prediction. Unsupervised: customer segmentation, anomaly detection.## Task 1 - Concepts, Splits and Leakage

- Script: `splits_leakage.py` (output in `output.txt`, plot in `overfitting_plot.png`)
- Split: 60/20/20, stratified, random_state=42. Test set kept aside and not used.
- Overfitting: Decision Tree depth 1 to None. Train accuracy goes 0.73 to 1.00 but validation stays around 0.73-0.79, so deep trees overfit.
- CV: 5-fold LogisticRegression gives 0.81 +/- 0.01, a stable estimate.
- Leakage: scaling before the split leaks test statistics into training. Fix: split first, fit the scaler on train only. Accuracy was same here (0.8038) because the dataset is large, but the method is still wrong.
- Supervised: spam detection, house price prediction. Unsupervised: customer segmentation, anomaly detection.

