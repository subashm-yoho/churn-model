import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from common import ORDINAL_COLS, load_data, split_data

X, y_churn = load_data()
X_train, X_val, X_test, _, _, _ = split_data(X, y_churn) # same rows as before, test set aside

TARGET = "MonthlyCharges"
y_train = X_train[TARGET]
y_val = X_val[TARGET]
drop_cols = [TARGET, "TotalCharges"] # avoid leakage
Xtr = X_train.drop(columns=drop_cols)
Xva = X_val.drop(columns=drop_cols)

num_cols = ["SeniorCitizen", "tenure"]
nominal_cols = [c for c in Xtr.columns
if Xtr[c].dtype == "object" and c not in ORDINAL_COLS]


def make_reg_pipeline(model):
 pre = ColumnTransformer([
("num", Pipeline([("imp", SimpleImputer(strategy="median")),
("scale", StandardScaler())]), num_cols),
("ord", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
("enc", OrdinalEncoder(
categories=[["Month-to-month", "One year", "Two year"]]))]),
ORDINAL_COLS),
("nom", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
nominal_cols),
])
 return Pipeline([("preprocess", pre), ("model", model)])


models = {
"LinearRegression": LinearRegression(),
"Ridge (alpha=1)": Ridge(alpha=1.0),
"Lasso (alpha=0.5)": Lasso(alpha=0.5, max_iter=10000),
}

rows, coefs = [], {}
for name, model in models.items():
 pipe = make_reg_pipeline(model).fit(Xtr, y_train)
pred = pipe.predict(Xva)
rows.append({
"Model": name,
"Val R2": round(r2_score(y_val, pred), 4),
"Val MAE": round(mean_absolute_error(y_val, pred), 3),
"Val RMSE": round(float(np.sqrt(mean_squared_error(y_val, pred))), 3),
})
names = pipe.named_steps["preprocess"].get_feature_names_out()
coefs[name] = pd.Series(pipe.named_steps["model"].coef_, index=[n.split("__", 1)[-1] for n in names])

results = pd.DataFrame(rows)
print("Regression results (target = MonthlyCharges, validation set)")
print(results.to_string(index=False))

# ---------- Compare coefficients ----------
coef_df = pd.DataFrame(coefs)
first_col = coef_df.columns[0] # the LinearRegression column
top = coef_df.reindex(coef_df[first_col].abs().sort_values(ascending=False).index).head(10)
print("\nTop 10 coefficients (by size in LinearRegression):")
print(top.round(2))

print("\nTotal features:", len(coef_df))
for name in coef_df.columns:
 zeros = int((coef_df[name].abs() < 1e-8).sum())
print(f"{name:<20} zero coefficients: {zeros}")
print("\nRidge shrinks coefficients but keeps all of them; Lasso can push some exactly to zero")
print("(feature selection). Higher alpha = more shrinkage = more zeros.")

results.to_csv("task3_regression_results.csv", index=False)
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from common import ORDINAL_COLS, load_data, split_data

X, y_churn = load_data()
X_train, X_val, X_test, _, _, _ = split_data(X, y_churn) # same rows as before, test set aside

TARGET = "MonthlyCharges"
y_train = X_train[TARGET]
y_val = X_val[TARGET]
drop_cols = [TARGET, "TotalCharges"] # avoid leakage
Xtr = X_train.drop(columns=drop_cols)
Xva = X_val.drop(columns=drop_cols)

num_cols = ["SeniorCitizen", "tenure"]
nominal_cols = [c for c in Xtr.columns
if Xtr[c].dtype == "object" and c not in ORDINAL_COLS]


def make_reg_pipeline(model):
 pre = ColumnTransformer([
("num", Pipeline([("imp", SimpleImputer(strategy="median")),
("scale", StandardScaler())]), num_cols),
("ord", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
("enc", OrdinalEncoder(
categories=[["Month-to-month", "One year", "Two year"]]))]),
ORDINAL_COLS),
("nom", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
nominal_cols),
])
 return Pipeline([("preprocess", pre), ("model", model)])


models = {
"LinearRegression": LinearRegression(),
"Ridge (alpha=1)": Ridge(alpha=1.0),
"Lasso (alpha=0.5)": Lasso(alpha=0.5, max_iter=10000),
}

rows, coefs = [], {}
for name, model in models.items():
 pipe = make_reg_pipeline(model).fit(Xtr, y_train)
pred = pipe.predict(Xva)
rows.append({
"Model": name,
"Val R2": round(r2_score(y_val, pred), 4),
"Val MAE": round(mean_absolute_error(y_val, pred), 3),
"Val RMSE": round(float(np.sqrt(mean_squared_error(y_val, pred))), 3),
})
names = pipe.named_steps["preprocess"].get_feature_names_out()
coefs[name] = pd.Series(pipe.named_steps["model"].coef_, index=[n.split("__", 1)[-1] for n in names])

results = pd.DataFrame(rows)
print("Regression results (target = MonthlyCharges, validation set)")
print(results.to_string(index=False))

# ---------- Compare coefficients ----------
coef_df = pd.DataFrame(coefs)
first_col = coef_df.columns[0] # the LinearRegression column
top = coef_df.reindex(coef_df[first_col].abs().sort_values(ascending=False).index).head(10)
print("\nTop 10 coefficients (by size in LinearRegression):")
print(top.round(2))

print("\nTotal features:", len(coef_df))
for name in coef_df.columns:
 zeros = int((coef_df[name].abs() < 1e-8).sum())
print(f"{name:<20} zero coefficients: {zeros}")
print("\nRidge shrinks coefficients but keeps all of them; Lasso can push some exactly to zero")
print("(feature selection). Higher alpha = more shrinkage = more zeros.")

results.to_csv("task3_regression_results.csv", index=False)
