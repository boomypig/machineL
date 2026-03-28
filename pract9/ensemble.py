import pandas as pd
import numpy as np
import matplotlib as plt
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
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
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_SEED
    )
    return search
def single_tree(x):
    pre = pre_proc(x)
    tree = Pipeline(steps=[
        ("pre",pre),
        ("clf",DecisionTreeClassifier(random_state=RANDOM_SEED))
    ])
    return tree

def random_forest(x):
    pre = pre_proc(x)
    forest = Pipeline(steps=[
        ("pre", pre),
        ("clf",RandomForestClassifier(bootstrap=True,oob_score=True,random_state=RANDOM_SEED))
    ])
    return forest

def ada_boost(model):
    ada_clf = AdaBoostClassifier(
        estimator=model)

def main():
    df = pd.read_csv("./data/dataset.csv")
    TARGET_COLUMN = "target"
    DROP_COLUMNS = [TARGET_COLUMN]
    y = df[TARGET_COLUMN]

    x = df.drop(columns=DROP_COLUMNS)

    x_trainval,x_test,y_trainval,y_test = split_data(x,y)

    simple_tree = single_tree(x_trainval)
    
    tree_params = {
        "clf__criterion":["gini","entropy"],
        "clf__max_depth":list(range(3,11)),
    }

    simple_tree_search = random_cv(tree_params,simple_tree)

    simple_tree_search.fit(x_trainval,y_trainval)

    best_simple_tree = simple_tree_search.best_estimator_

    simple_metrics, pred = evaluate_model(best_simple_tree,x_trainval,y_trainval)

    simple_tree_scores = pd.DataFrame(simple_tree_search.cv_results_)
    print(simple_tree_scores.head().sort_values(by="rank_test_score", ascending=True))

    print(simple_tree_search.best_params_)

    print(simple_metrics)

    print(confusion_matrix(y_trainval,pred))
    # ----------------------------------------------------------------------------------
    # Random forest
    # ----------------------------------------------------------------------------------
    forest = random_forest(x_trainval)

    forest_params = {
        "clf__max_depth":list(range(3,15)),
        "clf__criterion":["gini","entropy"],
        "clf__max_features": ["sqrt","log2","None"]
    }

    forest_search = random_cv(forest_params,forest)

    forest_search.fit(x_trainval,y_trainval)

    best_forest = forest_search.best_estimator_

    forest_metrics,forest_y_pred = evaluate_model(best_forest,x_trainval,y_trainval)
    print(forest_metrics)
    print(confusion_matrix(y_trainval,forest_y_pred))
    
    test_forest_metrics, test_forest_y_pred = evaluate_model(best_forest,x_test,y_test)

    print(test_forest_metrics)
    print(confusion_matrix(y_test,test_forest_y_pred))

    # ----------------------------------------------------------------------------------
    # Ada Boost forest
    # ----------------------------------------------------------------------------------


    
main()