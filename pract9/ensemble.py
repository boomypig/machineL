import pandas as pd
import numpy as np
import matplotlib as plt
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

TESTING_SIZE = 0.15
RANDOM_SEED = 119

def split_data(x,y): 
    x_trainval,x_test,y_trainval,y_test = train_test_split(x,y,test_size=TESTING_SIZE, random_state=RANDOM_SEED, stratify=y)

    return x_trainval,x_test,y_trainval,y_test 

def pre_proc(x):
    num_col = [c for c in x.columns if is_numeric_dtype(x[c])]
    cat_col = [c for c in x.columns if c not in num_col]

    num_pipe= Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())])

    cat_pipe= Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    
    pre = ColumnTransformer(transformers=[
        ("num_pipe",num_pipe,num_col),
        ("cat_pipe",cat_pipe,cat_col)
    ],
    remainder="drop")
    return pre

def evaluate_model(model,x,y):
    y_pred = model.predict(x)


    metrics = {
        "Accuracy": accuracy_score(y,y_pred),
        "Recall": recall_score(y,y_pred),
        "Precision": precision_score(y,y_pred),
        "f1": f1_score(y,y_pred),
    }
    return metrics, y_pred 

def random_cv(params,pipe):
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=params,
        n_iter=40,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_SEED
    )
    return search


def main():
    df = pd.read_csv("./data/dataset.csv")
    print(df.head())

main()