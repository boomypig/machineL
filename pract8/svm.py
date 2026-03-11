import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from sklearn.model_selection import train_test_split
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

    x_val_split = VAL_SIZE/(1-TESTING_SIZE)

    x_train,x_val,y_train,y_val = train_test_split(x_trainval,y_trainval,train_size=x_val_split,random_state=RANDOM_SEED,stratify=y_trainval)

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
    ])
    return pre


def check_dtypes(df):
    columns = df.columns
    for col in columns:
        print(f"\nex of values in col: {col} \n {df[col].head()} \n")



def train_linear_svm(x_train,y_train,x_val,y_val,c_values):
    print("\n"+"=" * 60)
    print("training linear SVM model")
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
                C=float(c)
                ))
            ])
        linear_svc_model.fit(x_train,y_train)

        y_pred, val_metrics = evaluate_model(linear_svc_model,x_val,y_val)

        row = {
            "model": "linear svc",
            "C": c,
            "Accuracy":val_metrics["Accuracy"],
            "Recall":val_metrics["Recall"],
            "Precision":val_metrics["Precision"],
            "f1":val_metrics["f1"],
            "y_pred": y_pred
        }
        models[c] = linear_svc_model

        results.append(row)
    
    compare_results_df = pd.DataFrame(results)
    best_metrics = selecting_best_c(compare_results_df)
    best_model = models[best_metrics["C"]]
    best_pred = best_metrics["y_pred"]

    return best_model, best_pred, best_metrics, compare_results_df



def evaluate_model(model,x,y,split_name="Validation"):
    y_pred = model.predict(x)

    metrics = {
        "Accuracy": accuracy_score(y,y_pred),
        "Recall": recall_score(y,y_pred),
        "Precision": precision_score(y,y_pred),
        "f1": f1_score(y,y_pred),
    }

    # print(f"{split_name.upper()} METRICS")
    # for metric_name, value in metrics.items():
    #     print(f"{metric_name:>10} : {value:4f}")
    # print("Confustion Matrix:")
    # print(confusion_matrix(y,y_pred))
    return y_pred,metrics



def selecting_best_c(results_df, tolerance = 0.01):
    best_f1 = results_df["f1"].max()
    contenders = results_df[results_df["f1"] >= best_f1 - tolerance]
    best_row = contenders.loc[contenders["C"].idxmin()]
    # best_metrics = {
    #         "model": best_row["model"],
    #         "C": best_row["C"],
    #         "Accuracy":best_row["Accuracy"],
    #         "Recall":best_row["Recall"],
    #         "Precision":best_row["Precision"],
    #         "f1":best_row["f1"],
    #         "y_pred": best_row["y_pred"]
    #     }
    return best_row



def main():
    df = pd.read_csv("./dataset_svm.csv")
    # check_dtypes(df)
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


    linear_best_model, linear_best_pred, linear_best_metrics, linear_results_df = train_linear_svm(x_train,y_train,x_val,y_val,C_values)
    drop_col = ["model","y_pred"]
    linear_results_df = linear_results_df.drop(columns=drop_col)
    print(linear_results_df)


    


main()