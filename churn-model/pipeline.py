import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer

CSV_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
RS = 42

df = pd.read_csv(CSV_PATH)
df = df.drop(columns=["customerID"])
# TotalCharges has blank strings (' '), convert to NaN so imputers can handle them
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print("Missing values per column (only non-zero):")
print(df.isna().sum()[df.isna().sum() > 0])

y = (df["Churn"] == "Yes").astype(int)
X = df.drop(columns=["Churn"])

X_temp, X_test, y_temp, y_test = train_test_split(
X, y, test_size=0.20, stratify=y, random_state=RS)
X_train, X_val, y_train, y_val = train_test_split(
X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=RS)
print("\nTrain", X_train.shape, "| Val", X_val.shape, "| Test", X_test.shape)

num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
cat_cols = [c for c in X.columns if c not in num_cols]

num_imp = SimpleImputer(strategy="median").fit(X_train[num_cols]) # fit on train only
med = pd.DataFrame(num_imp.transform(X_train[num_cols]), columns=num_cols, index=X_train.index)

cat_imp = SimpleImputer(strategy="most_frequent").fit(X_train[cat_cols])
print("\nCategorical missing in train:", int(X_train[cat_cols].isna().sum().sum()),
"(imputer ready, nothing to fill here)")

knn_imp = KNNImputer(n_neighbors=5).fit(X_train[num_cols])
knn = pd.DataFrame(knn_imp.transform(X_train[num_cols]), columns=num_cols, index=X_train.index)

missing_rows = X_train["TotalCharges"].isna()
print("\nRows with missing TotalCharges in train:", int(missing_rows.sum()))
compare = pd.DataFrame({
"tenure": X_train.loc[missing_rows, "tenure"],
"median_fill": med.loc[missing_rows, "TotalCharges"],
"knn_fill": knn.loc[missing_rows, "TotalCharges"],
})
print(compare)
print("\nMedian imputer fills every missing row with the same value.")
print("KNN imputer uses similar customers (tenure, MonthlyCharges), so values differ per row.")
