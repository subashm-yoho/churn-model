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

## Task 2 - Preprocessing, Features and Pipeline

- Script: `full_pipeline.py` (features in `features_demo.py`, saved model `churn_pipeline.joblib`)
- Missing values: TotalCharges had 11 blank values, converted to NaN. SimpleImputer (median for numeric, most frequent for categorical) and KNNImputer both tried. KNN fills using similar customers, median gives one value for all.
- Encoding: OneHotEncoder for unordered columns, OrdinalEncoder for Contract (Month-to-month < One year < Two year).
- New features: signup_year and signup_month (built from tenure, as Telco has no date column), charge_per_tenure (MonthlyCharges / (tenure + 1)), log_total_charges (log1p of skewed TotalCharges).
- Pipeline: ColumnTransformer inside a Pipeline, so imputing, scaling and encoding are re-fit on each CV fold. No preprocessing outside cross-validation.
- Result: Pipeline CV accuracy (KNN imputer) 0.7948 +/- 0.0097, validation accuracy 0.7913. Test set not used.
- joblib: pipeline saved and reloaded, then predicted 3 new raw rows: [1 0 0]. One row had missing TotalCharges and was handled by the pipeline.
 

 
## Task 3 - Models: Supervised and Unsupervised
- Files: classifier.py (4 classifiers, output in out3.txt), task3_regression.py (output in out3b.txt), kmeans.py, shared helpers in common.py.

- Classifiers (all inside pipelines, validation set)
Model	| Validation accuracy | Validation ROC-AUC
LogisticRegression |	0.8084 | 0.8421
DecisionTree (max_depth=5) |	0.7921  0.8162
RandomForest (300 trees) |	0.7850 |	0.8170
XGBoost (early stopping on validation) |	0.8105 |	0.8419

- RandomForest OOB score, XGBoost best iteration, the 5 predict_proba vs predict rows and the top 5 features of the forest and XGBoost are in out3.txt.
- predict_proba gives the churn probability; predict is 1 only when that probability is above 0.5.
- Regression (target = MonthlyCharges)
Model |	R2 |	MAE | RMSE
LinearRegression |	0.9989 |	0.780 |	1.033
Ridge (alpha=1) |	0.9989 |	0.780 |	1.034
Lasso (alpha=0.5) |	0.9916 |	2.258 |	2.807

- Telco has no loan amount, so MonthlyCharges is the numeric target. TotalCharges was left out of the features so the target does not leak.
- Total features after encoding: 41. Ridge keeps all 41 (0 zero coefficients). Lasso pushed 26 coefficients exactly to zero, so it also acts as feature selection.
- The big coefficients are the service columns (InternetService Fiber optic, StreamingTV, StreamingMovies, PhoneService), which makes sense because MonthlyCharges is the sum of service prices.
- K-Means customer segments (tenure, MonthlyCharges, TotalCharges)
- Plot: kmeans_elbow_silhouette.png (k = 2 to 8). The best k by silhouette score is 2.

- Segment 0: 1436 customers, long tenure (about 57 months), high monthly charges (about 89), churn rate 17%. Loyal, high-value customers.
- Segment 1: 2789 customers, short tenure (about 20 months), low monthly charges (about 52), churn rate 31%. Newer customers who leave more often.
- Churn was not used for clustering, only to describe the segments afterwards.
## Task 4 - Evaluation and Tuning
- Files: metrics.py (out4.txt), tuning.py (out4b.txt), final_test.py (out4c.txt).

- Metrics for every classifier (validation set, threshold 0.5)
Model |	Accuracy |	Precision |	Recall |	F1 |	ROC-AUC
LogisticRegression |	0.8084 |	0.6745 |	0.5374 |	0.5982 |	0.8421
DecisionTree |	0.7921 |	0.6709 |	0.4251 |	0.5205 |	0.8162
RandomForest |	0.7850 |	0.6320 |	0.4545 |	0.5288 |	0.8170
XGBoost |	0.8105 |	0.7019 |	0.4973 |	0.5822 |	0.8419

- Confusion matrices for each model are in out4.txt.

- Why accuracy misleads on imbalanced classes: about 73.5% of customers stay and 26.5% churn. A dummy model that always predicts "stay" would get 73.5% accuracy but find 0% of the churners. So accuracy of 0.80 looks good while recall is only 0.43 to 0.54. Recall, precision, F1 and ROC-AUC show how well churners are really found.

- ROC curves
- 3 models (LogisticRegression, RandomForest, XGBoost) in one chart with AUC: roc_curves.png.

- Threshold for at least 75% recall
- Best model on validation: LogisticRegression. The threshold 0.294 (default 0.5) gives recall 0.751, precision 0.534, F1 0.624 on the validation set. A lower threshold catches more churners (higher recall) but gives more false alarms (lower precision).

- Hyperparameter tuning (train set, 3-fold CV, scoring ROC-AUC)
Model	Search	CV ROC-AUC	Validation ROC-AUC
RandomForest	GridSearchCV (n_estimators, max_depth, min_samples_leaf)	0.8480	0.8368
XGBoost	RandomizedSearchCV (n_iter=15)	0.8511	0.8420

- Best parameters are printed in out4b.txt. Tuning improved RandomForest validation AUC from 0.8170 to 0.8368. XGBoost stayed about the same (0.8419 to 0.8420).

- Regression vs mean baseline (validation set)
Model |	MAE |	RMSE |	R2
Mean baseline |	26.840 |	30.582 |	-0.0004
LinearRegression |	0.780 |	1.033 |	0.9989
Ridge (alpha=1) |	0.780 |	1.034 |	0.9989
Lasso (alpha=0.5) |	2.258 |	2.807 |	0.9916

- Final evaluation on the test set (used once)
- The best model was chosen on the validation set only: LogisticRegression (ROC-AUC 0.8421, XGBoost 0.8420, tuned RandomForest 0.8368). The 0.294 threshold also came from the validation set. The test set was then used one time.

Threshold |	Accuracy |	Precision |	Recall |	F1
0.5 |	0.8056 |	0.6634 |	0.5428 |	0.5971
0.294 |	0.7573 |	0.5295 |	0.7674 |	0.6266

- At threshold 0.294 the model finds 287 of the 374 churners in the test set (recall 0.767), so the 75% recall target also holds on unseen data. The price is more false alarms (255 customers wrongly flagged).

- Notes
- Test set was kept aside until the very last step and used once.
- All preprocessing is inside pipelines, so it is fitted on training data only.
- Text columns are detected with dtype.kind because newer pandas stores strings as str instead of object.
