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
            x_df[col] = (x_df[col] == "Yes").astype(int)
            changed_to_binary.append(col)

            # print(x_df[col].value_counts())
    print(x_df["TotalCharges"].dtype)
    print(x_df["TotalCharges"].isna().sum())

    print(x_df.dtypes.value_counts())

    print(changed_to_binary)
    

    x_trainval, x_test, y_trainval, y_test = train_test_split(x_df,y,test_size=test_split, random_state=random_state)

    val_fraction = val_split/(1-test_split)
    x_train,x_val,y_train,y_val = train_test_split(x_trainval,y_trainval,test_size=val_fraction,random_state=random_state)            
    print(f"these columns have binary values: \n {changed_to_binary}")
    numeric_col = [c for c in x_train.columns if is_numeric_dtype(x_train[c])]
    cat_col = [c for c in x_train.columns if c not in numeric_col]

    numeric_pipeline = Pipeline(steps=[(
        ("imputer",SimpleImputer(strategy="median"))
        ("scaler",StandardScaler())
    )])

    cat_pipeline = Pipeline(steps=[
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer(transformers=[
        ("num_pipe", numeric_pipeline()),
        ("col_pipe", cat_pipeline())
    ]remainder="drop")


    clf = Pipeline([('pre', pre), ("model", LogisticRegression())])
    clf.fit(x_train,y_train)
    probs_val = clf.predict_proba(x_val)[:, 1]
    t= .455
    y_hat = (probs_val >= t).astype(int)
    print(clf.classes_)
    # print(prob[:5])
    # print("predict:" , y_hat[:5])
    # y_hat_t = (prob[:5,1] >= 0.3).astype(int)
    # print(y_hat_t)

main()