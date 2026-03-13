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
from sklearn.decomposition import PCA
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
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_SEED
    )
    return search

def viz_pca(x_transformed,params,pipe):

    pca = PCA(n_components=2)
    x_2d = pca.fit_transform(x_transformed)

    print(f"variance explained: {pca.explained_variance_ratio_.sum():.2%}")

    viz_search = random_cv(params,pipe)

    return viz_search,x_2d,pca

def build_pipe(x):
    pre = pre_proc(x)

    pipe = Pipeline(steps=[
        ("pre",pre),
        ("clf", SVC())
    ])

    return pipe

def plot_decision_boundary(viz_svc, x_2d, y_trainval, pca):
    # meshgrid lives here
    # all three plots live here
    # savefig calls live here

    x_min, x_max = x_2d[:, 0].min() - 1, x_2d[:, 0].max() + 1
    y_min, y_max = x_2d[:, 1].min() - 1, x_2d[:, 1].max() + 1

    xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    Z = viz_svc.predict(grid).reshape(xx.shape)
    scores = viz_svc.decision_function(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))



    # Plot 1 — Decision Boundary

    ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    ax.contour(xx, yy, Z, colors="black", linewidths=1)

    ax.scatter(x_2d[:, 0], x_2d[:, 1],
            c=y_trainval, cmap="coolwarm",
            edgecolors="k", s=30)

    variance = pca.explained_variance_ratio_.sum()
    ax.set_title(f"Decision boundary (PCA {variance:.0%} variance explained)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    plt.savefig("decision_boundary.png")
    plt.close()

    # Plot 2 — Decision Score Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))

    heatmap = ax.contourf(xx, yy, scores, levels=20, cmap="RdBu_r", alpha=0.8)
    ax.contour(xx, yy, scores, levels=[0], colors="black", linewidths=2)

    ax.scatter(x_2d[:, 0], x_2d[:, 1],
            c=y_trainval, cmap="coolwarm",
            edgecolors="k", s=30)

    ax.set_title("Decision score heatmap (distance to boundary)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.colorbar(heatmap, ax=ax)

    plt.savefig("decision_scores.png")
    plt.close()

    # Plot 3 — Support Vectors

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")

    ax.scatter(x_2d[:, 0], x_2d[:, 1],
            c=y_trainval, cmap="coolwarm",
            edgecolors="k", s=30,
            label="data points")

    ax.scatter(viz_svc.named_steps["clf"].support_vectors_[:, 0],
           viz_svc.named_steps["clf"].support_vectors_[:, 1],
           s=120, facecolors="none",
           edgecolors="black", linewidths=2,
           label="support vectors")

    ax.set_title("Support vectors (PCA projection)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()

    plt.savefig("support_vectors.png")
    plt.close()


def main():

    # =======================================================
    # Preprocessing
    # =======================================================

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

    
    # =======================================================
    # Linear svm 
    # =======================================================

    linear_pipe = build_pipe(x_trainval)

    C_values = np.logspace(-2, 3, 30)

    linear_params = {
        "clf__kernel": ["linear"],
        "clf__C": C_values,
        "clf__class_weight": ["balanced"]
    }

    linear_search = random_cv(linear_params,linear_pipe)

    linear_search.fit(x_trainval,y_trainval)

    linear_model = linear_search.best_estimator_

    linear_scores = pd.DataFrame(linear_search.cv_results_)
    print(linear_scores.head().sort_values(by="rank_test_score", ascending=True))

    print("linear search best params", linear_search.best_params_)
    print("linear best f1", linear_search.best_score_)

    # =======================================================
    # Kerned svm 
    # =======================================================
    kern_pipe = build_pipe(x_trainval)

    kern_params = {
        "clf__C": np.logspace(-1, 4, 30),
        "clf__kernel": ["rbf","poly"],
        "clf__gamma": loguniform(1e-4, 1e0),
        "clf__degree": randint(2, 6),
        "clf__class_weight": [{0: 1, 1: 2},"balanced" ]
    }
    

    kern_search = random_cv(kern_params,kern_pipe)

    kern_search.fit(x_trainval,y_trainval)

    kern_model = kern_search.best_estimator_

    kern_scores = pd.DataFrame(kern_search.cv_results_)
    print(kern_scores.head().sort_values(by="rank_test_score", ascending=True))

    print("kerned search best params", kern_search.best_params_)
    print("kern best f1", kern_search.best_score_)

    linear_metrics = evaluate_model(linear_model,x_test,y_test)
    kern_metrics = evaluate_model(kern_model,x_test,y_test)
    
    print("linear metrics: \n" , linear_metrics)
    print("kernal Metrics: \n", kern_metrics)
    print(kern_metrics["cm"])

    # =======================================================
    # Plotting svm
    # =======================================================
    

    viz_pre = pre_proc(x_trainval)

    x_transformed = viz_pre.fit_transform(x_trainval)

    viz_pipeline = Pipeline(steps=([    
        ("clf", SVC())
    ]))

    viz_params = {
        "clf__class_weight" : ["balanced"],
        "clf__kernel": ["rbf", "linear"],
        "clf__gamma": loguniform(1e-4, 1e0),
        "clf__C": np.logspace(-1, 4, 30)
    }

    viz_search,x_2d,pca = viz_pca(x_transformed,viz_params,viz_pipeline)

    viz_search.fit(x_2d,y_trainval)

    viz_svc = viz_search.best_estimator_    

    print(viz_svc)

    plot_decision_boundary(viz_svc, x_2d, y_trainval, pca)

main()