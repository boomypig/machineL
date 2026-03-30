import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

RANDOM_SEED = 119

DROP_COLUMNS = ["customer_id"]

NUM_FEATURES = [
    "age",
    "annual_income_k",
    "tenure_months",
    "monthly_orders",
    "avg_basket_usd",
    "discount_share",
    "app_sessions_per_month",
    "website_minutes_per_month",
    "support_tickets_6m",
    "returns_6m",
    "days_since_last_order",
    "delivery_distance_km",
    "satisfaction_score",
    "ad_exposure_score",
    "account_balance_points",
]

CAT_FEATURES = [
    "preferred_device",
    "region",
    "membership_tier",
    "primary_category",
]


def load_data(path):
    df = pd.read_csv(path)
    return df


def pre_proc():
    num_idx = list(range(len(NUM_FEATURES)))
    cat_idx = list(range(len(NUM_FEATURES), len(NUM_FEATURES) + len(CAT_FEATURES)))

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    pre = ColumnTransformer(transformers=[
        ("num_pipe", num_pipe, num_idx),
        ("cat_pipe", cat_pipe, cat_idx)
    ], remainder="drop")

    return pre


def run_pca(x_proc, n_components=None):
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    x_pca = pca.fit_transform(x_proc)
    return pca, x_pca


def plot_explained_variance(pca):
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    components = range(1, len(pca.explained_variance_ratio_) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(components, pca.explained_variance_ratio_, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance Ratio")
    axes[0].set_title("Per-Component Explained Variance")

    axes[1].plot(components, cumulative, marker="o", color="steelblue")
    axes[1].axhline(0.90, color="red", linestyle="--", label="90% threshold")
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("Cumulative Explained Variance")
    axes[1].set_title("Cumulative Explained Variance")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("pca_variance.png", dpi=150)
    plt.show()


def plot_pca_scatter(x_pca, color_by=None, label_name=""):
    fig, ax = plt.subplots(figsize=(8, 6))

    if color_by is not None:
        codes = pd.Categorical(color_by).codes
        scatter = ax.scatter(x_pca[:, 0], x_pca[:, 1], c=codes, cmap="tab10", alpha=0.7, s=30)
        plt.colorbar(scatter, ax=ax, label=label_name)
    else:
        ax.scatter(x_pca[:, 0], x_pca[:, 1], alpha=0.7, s=30, color="steelblue")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"PCA 2D Projection  {('– colored by ' + label_name) if label_name else ''}")
    plt.tight_layout()
    plt.savefig("pca_scatter.png", dpi=150)
    plt.show()


def run_kmeans_sweep(x_proc, k_values):
    results = []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(x_proc)
        inertia = km.inertia_
        sil = silhouette_score(x_proc, labels)
        results.append({"k": k, "inertia": inertia, "silhouette": sil, "model": km, "labels": labels})
        print(f"k={k}  |  inertia={inertia:.2f}  |  silhouette={sil:.4f}")
    return results


def plot_kmeans_metrics(sweep_results):
    ks = [r["k"] for r in sweep_results]
    inertias = [r["inertia"] for r in sweep_results]
    silhouettes = [r["silhouette"] for r in sweep_results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(ks, inertias, marker="o", color="steelblue")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Plot")

    axes[1].plot(ks, silhouettes, marker="o", color="coral")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Score by k")

    plt.tight_layout()
    plt.savefig("kmeans_metrics.png", dpi=150)
    plt.show()


def plot_clusters_pca(x_pca, labels, k):
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(x_pca[:, 0], x_pca[:, 1], c=labels, cmap="tab10", alpha=0.7, s=30)
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"k-Means Clusters (k={k}) in PCA Space")
    plt.tight_layout()
    plt.savefig(f"clusters_k{k}.png", dpi=150)
    plt.show()


def interpret_clusters(x, labels, k):
    x_copy = x[NUM_FEATURES].copy()
    x_copy["cluster"] = labels
    summary = x_copy.groupby("cluster")[NUM_FEATURES].mean().round(3)
    print(f"\nCluster Feature Means (k={k}):")
    print(summary.to_string())

    print(f"\nCluster Sizes (k={k}):")
    print(pd.Series(labels).value_counts().sort_index().to_string())

    cat_copy = x[CAT_FEATURES].copy()
    cat_copy["cluster"] = labels
    for col in CAT_FEATURES:
        print(f"\nTop value per cluster – {col}:")
        print(cat_copy.groupby("cluster")[col].agg(lambda s: s.value_counts().idxmax()).to_string())

    return summary


def main():
    df = load_data("./data/retail_customer_behavior_unsupervised.csv")

    print("Shape:", df.shape)
    print("\nMissing values:")
    print(df.isnull().sum())

    x = df.drop(columns=DROP_COLUMNS)
    x = x[NUM_FEATURES + CAT_FEATURES]

    print("\nFeatures included (numeric):", NUM_FEATURES)
    print("Features included (categorical):", CAT_FEATURES)
    print("Features excluded:", DROP_COLUMNS, "– identifier, carries no signal")

    # -------------------------------------------------------------------
    # Task 1 – Preprocessing
    # -------------------------------------------------------------------
    pre = pre_proc()
    x_proc = pre.fit_transform(x)
    print(f"\nProcessed shape: {x_proc.shape}")

    # -------------------------------------------------------------------
    # Task 2 – PCA
    # -------------------------------------------------------------------
    pca_full, _ = run_pca(x_proc)

    print("\nExplained variance ratio (first 10 components):")
    for i, v in enumerate(pca_full.explained_variance_ratio_[:10]):
        print(f"  PC{i+1}: {v:.4f}")

    n_90 = int(np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= 0.90)) + 1
    print(f"\nComponents needed to reach 90% variance: {n_90}")

    plot_explained_variance(pca_full)

    pca_2d, x_pca_2d = run_pca(x_proc, n_components=2)
    plot_pca_scatter(x_pca_2d, color_by=x["membership_tier"], label_name="membership_tier")

    # -------------------------------------------------------------------
    # Task 3 – k-Means sweep
    # -------------------------------------------------------------------
    K_VALUES = [2, 3, 4, 5, 6, 8]

    print("\nk-Means sweep:")
    sweep_results = run_kmeans_sweep(x_proc, K_VALUES)

    plot_kmeans_metrics(sweep_results)

    best_result = max(sweep_results, key=lambda r: r["silhouette"])
    best_k = best_result["k"]
    best_labels = best_result["labels"]

    print(f"\nChosen k={best_k}  (silhouette: {best_result['silhouette']:.4f})")

    sweep_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("model", "labels")} for r in sweep_results])
    print("\nSweep summary:")
    print(sweep_df.to_string(index=False))

    # -------------------------------------------------------------------
    # Task 4 – Cluster interpretation
    # -------------------------------------------------------------------
    plot_clusters_pca(x_pca_2d, best_labels, best_k)

    interpret_clusters(x, best_labels, best_k)

    # -------------------------------------------------------------------
    # Task 5 – Reflection
    # -------------------------------------------------------------------
    print("\n--- Reflection ---")
    print(
        "PCA showed how variance is distributed across the retail customer features and allowed "
        "us to project all customers into 2D to spot broad structure. k-Means then identified "
        "distinct customer segments — differing in spending, engagement, and tenure — without "
        "using any labels. In a supervised workflow these insights are directly useful: PCA "
        "components can replace correlated raw features to reduce noise, and cluster labels can "
        "be added as a categorical feature to help a classifier distinguish, for example, "
        "high-value loyal customers from low-engagement churners."
    )


main()