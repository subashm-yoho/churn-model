import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

CSV_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
RS = 42

df = pd.read_csv(CSV_PATH).drop(columns=["customerID"])
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
y = (df["Churn"] == "Yes").astype(int)
X = df.drop(columns=["Churn"])

X_temp, X_test, y_temp, y_test = train_test_split(
X, y, test_size=0.20, stratify=y, random_state=RS)
X_train, X_val, y_train, y_val = train_test_split(
X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=RS)

ordinal_cols = ["Contract"]
nominal_cols = [c for c in X.columns
if X[c].dtype == "object" and c not in ordinal_cols]
print("Ordinal column :", ordinal_cols)
print("Nominal columns:", len(nominal_cols), "->", nominal_cols)

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ohe.fit(X_train[nominal_cols])
ohe_train = ohe.transform(X_train[nominal_cols])
print("\nOneHot: before", X_train[nominal_cols].shape, "-> after", ohe_train.shape)
print("First 5 new columns:", list(ohe.get_feature_names_out()[:5]))

order = [["Month-to-month", "One year", "Two year"]]
ord_enc = OrdinalEncoder(categories=order)
ord_enc.fit(X_train[ordinal_cols])
ord_train = ord_enc.transform(X_train[ordinal_cols])

check = pd.DataFrame({
"Contract": X_train["Contract"].values[:6],
"encoded": ord_train[:6, 0],
})
print("\nOrdinalEncoder on Contract (first 6 rows):")
print(check)
print("\nWhy: OneHot for unordered labels (no fake order), Ordinal for Contract (order matters).")
