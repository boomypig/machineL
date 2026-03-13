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

def one_plot_decision_boundary(viz_svc, x_2d, y_trainval, pca):
    x_min, x_max = x_2d[:, 0].min() - 1, x_2d[:, 0].max() + 1
    y_min, y_max = x_2d[:, 1].min() - 1, x_2d[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    Z = viz_svc.predict(grid).reshape(xx.shape)
    scores = viz_svc.decision_function(grid).reshape(xx.shape)
    variance = pca.explained_variance_ratio_.sum()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # plot 1 - decision boundary
    axes[0].contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    axes[0].contour(xx, yy, Z, colors="black", linewidths=1)
    axes[0].scatter(x_2d[:, 0], x_2d[:, 1], c=y_trainval, cmap="coolwarm", edgecolors="k", s=30)
    axes[0].set_title(f"Decision boundary\nPCA {variance:.0%} variance explained")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    # plot 2 - decision score heatmap
    heatmap = axes[1].contourf(xx, yy, scores, levels=20, cmap="RdBu_r", alpha=0.8)
    axes[1].contour(xx, yy, scores, levels=[0], colors="black", linewidths=2)
    axes[1].scatter(x_2d[:, 0], x_2d[:, 1], c=y_trainval, cmap="coolwarm", edgecolors="k", s=30)
    axes[1].set_title("Decision score heatmap")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    plt.colorbar(heatmap, ax=axes[1])

    # plot 3 - support vectors
    axes[2].contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    axes[2].scatter(x_2d[:, 0], x_2d[:, 1], c=y_trainval, cmap="coolwarm", edgecolors="k", s=30, label="data points")
    axes[2].scatter(viz_svc.named_steps["clf"].support_vectors_[:, 0],
                    viz_svc.named_steps["clf"].support_vectors_[:, 1],
                    s=120, facecolors="none", edgecolors="black", linewidths=2, label="support vectors")
    axes[2].set_title("Support vectors")
    axes[2].set_xlabel("PC1")
    axes[2].set_ylabel("PC2")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("svm_plots.png")
    plt.close()

def plot_linear_weights(linear_model):
    linear_svc = linear_model.named_steps["clf"]
    coefficients = linear_svc.coef_[0]
    
    preprocessor = linear_model.named_steps["pre"]
    num_features = preprocessor.transformers_[0][2]
    cat_features = preprocessor.transformers_[1][1].named_steps["onehot"].get_feature_names_out()
    feature_names = list(num_features) + list(cat_features)

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "weight": coefficients
    }).sort_values("weight")

    fig, ax = plt.subplots(figsize=(8, len(feature_names) * 0.4))
    colors = ["red" if w > 0 else "blue" for w in coef_df["weight"]]
    ax.barh(coef_df["feature"], coef_df["weight"], color=colors)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_title("Linear SVM feature weights")
    ax.set_xlabel("Weight (positive = predicts churn)")

    plt.tight_layout()
    plt.savefig("linear_weights.png")
    plt.close()

def final_test_evaluation(model, X_test, y_test, model_name):
    print("\n" + "=" * 60)
    print(f"FINAL TEST EVALUATION: {model_name}")
    print("=" * 60)

    test_metrics, y_test_pred = evaluate_model(model, X_test, y_test)
    return test_metrics, y_test_pred

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

    # # =======================================================
    # # Kerned svm 
    # # =======================================================
    # kern_pipe = build_pipe(x_trainval)

    # kern_params = {
    #     "clf__C": np.logspace(-1, 4, 30),
    #     "clf__kernel": ["rbf","poly"],
    #     "clf__gamma": loguniform(1e-4, 1e0),
    #     "clf__degree": randint(2, 6),
    #     "clf__class_weight": [{0: 1, 1: 2},"balanced" ]
    # }
    

    # kern_search = random_cv(kern_params,kern_pipe)

    # kern_search.fit(x_trainval,y_trainval)

    # kern_model = kern_search.best_estimator_

    # kern_scores = pd.DataFrame(kern_search.cv_results_)
    # print(kern_scores.head().sort_values(by="rank_test_score", ascending=True))

    # print("kerned search best params", kern_search.best_params_)
    # print("kern best f1", kern_search.best_score_)

    # linear_metrics,y_linear_pred = evaluate_model(linear_model,x_test,y_test)
    # kern_metrics,y_kernal_pred = evaluate_model(kern_model,x_test,y_test)
    
    # print("linear metrics: \n" , linear_metrics)
    # print(confusion_matrix(y_test,y_linear_pred))
    
    # linear_test_metrics, linear_y_pred = final_test_evaluation(linear_model,x_test,y_test,"linear test")
    # for metric_name, value in linear_test_metrics.items():
    #     print(f"{metric_name:>10}: {value:4f}")
    # print(linear_y_pred)


    # print("kernal Metrics: \n", kern_metrics)
    # print(confusion_matrix(y_test,y_kernal_pred))

    # kern_test_metrics, kern_y_pred = final_test_evaluation(kern_model,x_test,y_test,"Kernal test")
    # for metric_name, value in kern_test_metrics.items():
    #     print(f"{metric_name:>10}: {value:4f}")
    # print(kern_y_pred)

    # # =======================================================
    # # Plotting svm
    # # =======================================================
    

    # viz_pre = pre_proc(x_trainval)

    # x_transformed = viz_pre.fit_transform(x_trainval)

    # viz_pipeline = Pipeline(steps=([    
    #     ("clf", SVC())
    # ]))

    # viz_params = {
    #     "clf__class_weight" : ["balanced"],
    #     "clf__kernel": ["rbf", "linear"],
    #     "clf__gamma": loguniform(1e-4, 1e0),
    #     "clf__C": np.logspace(-1, 4, 30)
    # }

    # viz_search,x_2d,pca = viz_pca(x_transformed,viz_params,viz_pipeline)

    # viz_search.fit(x_2d,y_trainval)

    # viz_svc = viz_search.best_estimator_    

    # print(viz_svc)

    # plot_decision_boundary(viz_svc, x_2d, y_trainval, pca)
    # one_plot_decision_boundary(viz_svc,x_2d,y_trainval,pca)


    plot_linear_weights(linear_model)
main()