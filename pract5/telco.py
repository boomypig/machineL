import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score

import matplotlib.pyplot as plt



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
    remainder="drop")


    print(y.value_counts())
    print(y.dtype)


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

    print(confusion_matrix(y_val,y_hat))
    print(classification_report(y_val,y_hat))

    fpr, tpr, thresholds = roc_curve(y_val, probs_val)
    auc_score = roc_auc_score(y_val, probs_val)

    precision, recall, thresholds = precision_recall_curve(y_val, probs_val)
    ap_score = average_precision_score(y_val, probs_val)

    probs_test = clf.predict_proba(x_test)[:, 1]
    yhat_test = (probs_test >= t).astype(int)

    print("TEST SCORES \n ",confusion_matrix(y_test, yhat_test))
    print("TEST SCORES \n ",classification_report(y_test, yhat_test))

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")  # random baseline
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc curve", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(recall, precision, label=f"PR curve (AP = {ap_score:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("precision recall  curve", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(thresholds, precision[:-1], label="Precision")
    plt.plot(thresholds, recall[:-1], label="Recall")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision and Recall vs Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig("thresholds  curve", dpi=200)
    plt.close()



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
