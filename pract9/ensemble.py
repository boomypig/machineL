import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,BaggingClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.utils.class_weight import compute_sample_weight


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
    num_col = [i for i, c in enumerate(x.columns) if is_numeric_dtype(x[c])]
    cat_col = [i for i, c in enumerate(x.columns) if not is_numeric_dtype(x[c])]

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


def tune_threshold(model, x, y):
    y_prob = model.predict_proba(x)[:, 1]  # get probabilities for positive class
    
    precisions, recalls, thresholds = precision_recall_curve(y, y_prob)
    
    # find threshold that maximizes f1 (balance of precision and recall)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = f1_scores.argmax()
    best_threshold = thresholds[best_idx]
    
    print(f"Best threshold: {best_threshold:.3f}")
    print(f"Precision: {precisions[best_idx]:.3f}")
    print(f"Recall: {recalls[best_idx]:.3f}")
    print(f"F1: {f1_scores[best_idx]:.3f}")
    
    return best_threshold

def predict_with_threshold(model, x, threshold):
    y_prob = model.predict_proba(x)[:, 1]
    return (y_prob >= threshold).astype(int)

def random_cv(params,pipe):
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=params,
        n_iter=40,
        scoring="recall",
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
        ("clf",RandomForestClassifier(class_weight="balanced",bootstrap=True,oob_score=True,random_state=RANDOM_SEED))
    ])
    return forest

def ada_boost(model):
    ada_clf = AdaBoostClassifier(
        estimator=model,
        n_estimators=100,
        random_state=RANDOM_SEED,
        learning_rate=0.5,
        )
    return ada_clf

def bagg_clf(estimator):
    bagg_clf = BaggingClassifier(
        estimator=estimator,
        n_estimators=100,
        random_state=RANDOM_SEED,
        oob_score=True,
        n_jobs=-1
    )
    return bagg_clf
def gradient_clf(x):
    pre = pre_proc(x)
    gradient = Pipeline(steps=[
        ("pre", pre),
        ("clf",GradientBoostingClassifier(random_state=RANDOM_SEED))
    ])
    return gradient
def main():
    df = pd.read_csv("./data/dataset.csv")
    TARGET_COLUMN = "target"
    DROP_COLUMNS = [TARGET_COLUMN]
    y = df[TARGET_COLUMN]

    results = {}
    print(y.value_counts())
    print(y.value_counts(normalize=True)) 
    x = df.drop(columns=DROP_COLUMNS)
    
    x_trainval,x_test,y_trainval,y_test = split_data(x,y)

    simple_tree = single_tree(x_trainval)
    
    tree_params = {
        "clf__criterion":["gini","entropy"],
        "clf__max_depth":list(range(3,20)),
        "clf__class_weight":["balanced"]
    }
    
    simple_tree_search = random_cv(tree_params,simple_tree)

    simple_tree_search.fit(x_trainval,y_trainval)

    best_simple_tree = simple_tree_search.best_estimator_

    simple_metrics, pred = evaluate_model(best_simple_tree,x_test,y_test)
    # find best threshold on trainval, apply to test
    best_thresh = tune_threshold(best_simple_tree, x_trainval, y_trainval)

    # now evaluate with tuned threshold
    y_pred_tuned = predict_with_threshold(best_simple_tree, x_test, best_thresh)

    print(confusion_matrix(y_test, y_pred_tuned))
    print("Recall:", recall_score(y_test, y_pred_tuned))
    
    results["simple_tree"] = simple_metrics

    # simple_tree_scores = pd.DataFrame(simple_tree_search.cv_results_)
    # print(simple_tree_scores.head().sort_values(by="rank_test_score", ascending=True))

    print("simple tree params:\n",simple_tree_search.best_params_)

    print("simple CM: \n",confusion_matrix(y_test,pred))
    # ----------------------------------------------------------------------------------
    # Random forest
    # ----------------------------------------------------------------------------------
    forest = random_forest(x_trainval)

    forest_params = {
        "clf__max_depth":list(range(3,20)),
        "clf__criterion":["gini","entropy"],
        "clf__max_features": ["sqrt","log2",None]
    }

    forest_search = random_cv(forest_params,forest)

    forest_search.fit(x_trainval,y_trainval)

    best_forest = forest_search.best_estimator_

    forest_metrics,forest_y_pred = evaluate_model(best_forest,x_test,y_test)
    results["forest"] = forest_metrics

    print("Forest params: \n", forest_search.best_params_)

    print("Forest CM: \n",confusion_matrix(y_test,forest_y_pred))
    
    # ----------------------------------------------------------------------------------
    # bag forest
    # ----------------------------------------------------------------------------------

    # using best forest to bag 

    bagged_forest = bagg_clf(best_forest)

    bagged_forest.fit(x_trainval,y_trainval)


    bag_metrics, bag_y_pred = evaluate_model(bagged_forest,x_test,y_test)

    results["bagged_forest"] = bag_metrics

    print("OOB score:", best_forest.named_steps['clf'].oob_score_)
    print("Bagged Forest CM: \n", confusion_matrix(y_test,bag_y_pred))

    # ----------------------------------------------------------------------------------
    # Ada Boost tree
    # ----------------------------------------------------------------------------------
    
   # preprocess first
    pre = pre_proc(x_trainval)
    x_trainval_proc = pre.fit_transform(x_trainval)
    x_test_proc = pre.transform(x_test)

    # build ada pipeline without preprocessing
    stump = DecisionTreeClassifier(max_depth=1, random_state=RANDOM_SEED)


    weights = compute_sample_weight("balanced", y_trainval)


    ada = AdaBoostClassifier(
        estimator=stump,
        random_state=RANDOM_SEED
    )

    ada_params = {
        "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0],
        "n_estimators": [50, 100, 200, 300, 500],
    }

    ada_search = RandomizedSearchCV(
        estimator=ada,
        param_distributions=ada_params,
        n_iter=40,
        scoring="recall",
        cv=5,
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_SEED
    )

    ada_search.fit(x_trainval_proc, y_trainval, sample_weight=weights)

    best_ada = ada_search.best_estimator_

    print("Ada params:", ada_search.best_params_)

    ada_metrics, ada_pred = evaluate_model(best_ada, x_test_proc, y_test)
    results["ada"] = ada_metrics

    print("ada CM \n" , confusion_matrix(y_test, ada_pred))


    # ----------------------------------------------------------------------------------
    # Gradient Boost forest
    # ----------------------------------------------------------------------------------

    gradient = gradient_clf(x_trainval)

    weights = compute_sample_weight("balanced", y_trainval)

    gradient_params = {
    "clf__max_depth": list(range(2, 6)),
    "clf__n_estimators": [50, 100, 200],
    "clf__learning_rate": [0.01, 0.05, 0.1, 0.5],
    "clf__max_features": ["sqrt", "log2"],
    "clf__subsample": [0.6, 0.8, 1.0],
    }

    gradient_search = random_cv(gradient_params,gradient)


    gradient_search.fit(x_trainval, y_trainval, clf__sample_weight=weights)

    best_gradient = gradient_search.best_estimator_

    gradient_metrics, gradient_y_pred = evaluate_model(best_gradient,x_test,y_test)

    results["gradient"] = gradient_metrics

    print(gradient_search.best_params_)

    print("gradient CM \n ", confusion_matrix(y_test,gradient_y_pred))


    print(pd.DataFrame(results).T)
main()