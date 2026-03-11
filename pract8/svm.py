import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

RANDOM_SEED = 119
TESTING_SIZE = 0.15
VAL_SIZE = 0.15

def split_data(x,y):
    x_trainval,x_test,y_trainval,y_test = train_test_split(x,y,test_size=TESTING_SIZE, random_state=RANDOM_SEED, stratify=y)

    x_val_split = VAL_SIZE/(1.0-TESTING_SIZE)

    x_train,x_val,y_train,y_val = train_test_split(x_trainval,y_trainval,test_size=x_val_split,random_state=RANDOM_SEED,stratify=y_trainval)

    return x_train,x_val,x_trainval,x_test,y_train,y_val,y_trainval,y_test 

def pre_proc(x_train):
    num_col = [c for c in x_train.columns if is_numeric_dtype(x_train[c])]
    cat_col = [c for c in x_train.columns if c not in num_col]

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


def check_dtypes(df):
    columns = df.columns
    for col in columns:
        print(f"\nex of values in col: {col} \n {df[col].head()} \n")



def train_linear_svm(x_train,y_train,x_val,y_val,c_values):
    print("\n"+"=" * 60)
    print("training linear SVM model tested on val data")
    print("=" * 60)
    results = []
    models = {}

    for c in c_values:
        pre = pre_proc(x_train)
        # print(f"attempting with c: {c}")
        linear_svc_model = Pipeline(steps=[
            ("pre",pre),
            ("svm",LinearSVC(
                random_state=RANDOM_SEED,
                C=float(c),
                class_weight="balanced"
                ))
            ])
        linear_svc_model.fit(x_train,y_train)

        val_metrics = evaluate_model(linear_svc_model,x_val,y_val)

        row = {
            "model": "linear svc",
            "C": c,
            "Accuracy":val_metrics["Accuracy"],
            "Recall":val_metrics["Recall"],
            "Precision":val_metrics["Precision"],
            "f1":val_metrics["f1"],
        }
        models[c] = linear_svc_model

        results.append(row)
    
    compare_results_df = pd.DataFrame(results)
    best_metrics = selecting_best_c(compare_results_df)
    best_model = models[best_metrics["C"]]

    return best_model, best_metrics, compare_results_df



def evaluate_model(model,x,y):
    y_pred = model.predict(x)

    metrics = {
        "Accuracy": accuracy_score(y,y_pred),
        "Recall": recall_score(y,y_pred),
        "Precision": precision_score(y,y_pred),
        "f1": f1_score(y,y_pred),
    }
    return metrics



def selecting_best_c(results_df, tolerance = 0.01):
    best_f1 = results_df["f1"].max()
    contenders = results_df[results_df["f1"] >= best_f1 - tolerance]
    best_row = contenders.loc[contenders["C"].idxmin()]
    return best_row

def random_cv(params,model):
    search = RandomizedSearchCV(
        estimator=model,
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

    x_train, x_val, x_trainval, x_test, y_train , y_val, y_trainval, y_test = split_data(x,y)

    C_values = np.logspace(-3,3,13)

    linear_best_model, linear_best_metrics, linear_results_df = train_linear_svm(x_train,y_train,x_val,y_val,C_values)

    print(linear_results_df)

    print(linear_best_metrics)

    corr_with_target = x_train.select_dtypes(include="number").corrwith(y_train)
    corr_with_target.sort_values().plot(kind="barh")
    plt.title("Feature Correlation with Target")
    plt.savefig("correlations.png")
    plt.close()


    


    


main()