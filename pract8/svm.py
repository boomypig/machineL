import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import loguniform, randint
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

RANDOM_SEED = 119
TESTING_SIZE = 0.15

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
        "cm": confusion_matrix(y,y_pred),
    }
    return metrics

def random_cv(params,pipe):
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=params,
        n_iter=40,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_SEED
    )
    return search



def main():
    df = pd.read_csv("./dataset_svm.csv")
    TARGET_COLUMN = "churned_next_month"
    DROP_COL = [TARGET_COLUMN, "customer_id"]

    y = df[TARGET_COLUMN]
    print("y before cleaned na's:", y.isna().sum())
    
    df = df.dropna(subset=[TARGET_COLUMN])

    y = df[TARGET_COLUMN]
    print("y after cleaned na's:", y.isna().sum())
    x = df.drop(columns=DROP_COL)

    x_trainval, x_test, y_trainval, y_test = split_data(x,y)

    pre = pre_proc(x_trainval)

    pipe = Pipeline(steps=[
        ("pre",pre),
        ("clf",SVC(
        class_weight="balanced"))
    ])

    C_values = np.logspace(-2, 3, 30)

    linear_params = {
        "clf__kernel": ["linear"],
        "clf__C": C_values,
    }

    kern_params = {
        "clf__C": C_values,
        "clf__kernel": ["rbf","poly"],
        "clf__gamma": loguniform(1e-4, 1e0),
        "clf__degree": randint(2, 6),
    }
    

    linear_search = random_cv(linear_params,pipe)

    linear_search.fit(x_trainval,y_trainval)

    linear_model = linear_search.best_estimator_

    linear_scores = pd.DataFrame(linear_search.cv_results_)
    print(linear_scores.head().sort_values(by="rank_test_score", ascending=True))

    print("linear search best params", linear_search.best_params_)
    print("linear best f1", linear_search.best_score_)


    kern_search = random_cv(kern_params,pipe)

    kern_search.fit(x_trainval,y_trainval)

    kern_model = kern_search.best_estimator_

    kern_scores = pd.DataFrame(kern_search.cv_results_)
    print(kern_scores.head().sort_values(by="rank_test_score", ascending=True))

    print("kerned serach best params", kern_search.best_params_)
    print("kern best f1", kern_search.best_score_)

    linear_metrics = evaluate_model(linear_model,x_test,y_test)
    kern_metrics = evaluate_model(kern_model,x_test,y_test)
    
    print("linear metrics: \n" , linear_metrics)
    print("kernal Metrics: \n", kern_metrics)

main()