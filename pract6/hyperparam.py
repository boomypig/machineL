import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def main():
    df = pd.read_csv("./telco_churn.csv")
    print(df.dtypes)
    TARGET_COLUMN = "Churn"
    y = df[TARGET_COLUMN]
    x_df = df.drop(columns = [TARGET_COLUMN])
    test_split = 0.15
    val_split = 0.15
    random_state = 119
    print(x_df["TotalCharges"].head())
    # changes is a str when it's supposed to be int
    x_df["TotalCharges"] = pd.to_numeric(x_df["TotalCharges"],errors="coerce")

    # print(f"x_df TotalCharges is now : \n {x_df["TotalCharges"].dtype}")

    # check for binary values
    changed_to_binary = []
    for col in x_df.columns:
        col_series = x_df[col].value_counts()
        if "Yes" in col_series.index and "No" in col_series.index:
            changed_to_binary.append(col)
            
    print(f"these columns have binary values: \n {changed_to_binary}")
            
main()