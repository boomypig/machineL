import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer

def main():
    df = pd.read_csv("./telco_churn.csv")


    TARGET_COLUMN = "Churn"
    y = df[TARGET_COLUMN]
    x_df = df.drop(columns = [TARGET_COLUMN])
    testing_split = 0.15
    validation_split = 0.15
    random_state = 119
    #notes 
    #we know that the column "TotalCharges" is the wrong dtype
    #also it has a few na's but it isn't caught bc its just an empty string
    #so it looks fine and not as an na

    x_df["TotalCharges"] = pd.to_numeric(x_df["TotalCharges"], errors="coerce")

    # this converts all yes no columns to 1 and 0
    changed_to_binary = []
    for col in x_df.columns:
        col_series = x_df[col].value_counts()
        if "Yes" in col_series.index and "No" in col_series.index:
            # print(x_df[col].value_counts())
            x_df[col] = (x_df[col] == "Yes").astype(int)
            changed_to_binary.append(col)

            # print(x_df[col].value_counts())
    print(x_df["TotalCharges"].dtype)
    print(x_df["TotalCharges"].isna().sum())

    print(x_df.dtypes.value_counts())

    print(changed_to_binary)
    

    x_trainval,x_test,y_trainval,y_test = train_test_split(x_df,y,test_size=testing_split,random_state=random_state)
    
    val_fraction = validation_split/(1-testing_split)

    x_train,x_val,y_train,y_val = train_test_split(x_trainval,y_trainval,test_size=val_fraction,random_state=random_state)

    #split into categorical and numerical 

    numeric_col = [c for c in x_train.columns if is_numeric_dtype(x_train[c])]
    cat_col = [c for c in x_train.columns if c not in numeric_col]

# print(f"numeric columns: \n {numeric_col}")
# print(f"categorial columns: \n {cat_col}")

    num_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
])

    cat_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",OneHotEncoder(handle_unknown="ignore"))
])

    pre = ColumnTransformer(
    transformers=[
        ("num", num_pipe, numeric_col),
        ("cat", cat_pipe, cat_col),
    ],
    remainder="drop"
)
# print(x_train)
    x_train_p = pre.fit_transform(x_train)
    x_val_p = pre.transform(x_val)
    x_test_p = pre.transform(x_test)            

    print(x_train_p.shape, x_val_p.shape, x_test_p.shape)

    print(y.value_counts())
    print(y.dtype)

# def checktypes(df):
#     # print(df.shape)
#     # print(df.head())
#     # print(df.dtypes)
#     # print(df.info())
#     for col in df.columns:
#         col_series = df[col].value_counts()
#         if len(col_series) == 2 and "Yes" in col_series:
#             print(df[col].head())
#             df[col] = (df[col] == "Yes").astype(int)
#             print(df[col].head())      
main()
