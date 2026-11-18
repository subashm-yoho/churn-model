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
- Top drivers: month-to-month contract, no online security, no tech support, fiber optic, electronic check paymen