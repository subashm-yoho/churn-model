import numpy as np
import pandas as pd

CSV_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def add_features(df):
    df = df.copy()
    ref_date = pd.Timestamp("2020-01-01")
    signup_date = ref_date - pd.to_timedelta(df["tenure"] * 30, unit="D")
    df["signup_year"] = signup_date.dt.year
    df["signup_month"] = signup_date.dt.month
    df["charge_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)
    df["log_total_charges"] = np.log1p(df["TotalCharges"])
    return df


