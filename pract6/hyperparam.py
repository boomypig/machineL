import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    roc_auc_score,
)

SEED = 119
TEST_SPLIT = 0.15
VAL_SPLIT = 0.15

PRIMARY_METRIC = "f1"  

def make_splits(X, y):
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=SEED, stratify=y
    )

    val_fraction = VAL_SPLIT / (1.0 - TEST_SPLIT)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_fraction, random_state=SEED, stratify=y_trainval
    )
    return X_train, X_val, X_trainval, X_test, y_train, y_val, y_trainval, y_test


def build_preprocessor(X_train):
    numeric_col = [c for c in X_train.columns if is_numeric_dtype(X_train[c])]
    cat_col = [c for c in X_train.columns if c not in numeric_col]

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(transformers=[
        ("num_pipe", numeric_pipeline, numeric_col),
        ("cat_pipe", cat_pipeline, cat_col),
    ])
    return pre


def eval_binary_classifier(y_true, probs, threshold=0.455):
    y_pred = (probs >= threshold).astype(int)

    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, probs),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "report": classification_report(y_true, y_pred, digits=4),
    }
    return out


def print_eval(name, results):
    print(f"\n=== {name} ===")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1:       {results['f1']:.4f}")
    print(f"ROC-AUC:  {results['roc_auc']:.4f}")
    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])
    print("\nClassification Report:")
    print(results["report"])


def main():
    # -----------------------------
    # Load + light cleaning
    # -----------------------------
    df = pd.read_csv("./telco_churn.csv")

    TARGET_COLUMN = "Churn"

    # Ensure target is 0/1
    if df[TARGET_COLUMN].dtype == object:
        y = (df[TARGET_COLUMN].astype(str).str.strip().str.lower() == "yes").astype(int)
    else:
        y = df[TARGET_COLUMN].astype(int)

    X = df.drop(columns=[TARGET_COLUMN]).copy()

    # Fix numeric that arrived as string
    if "TotalCharges" in X.columns:
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")

    # convert any Yes/No binary feature columns to 0/1
    for col in X.columns:
        vc = X[col].value_counts(dropna=False)
        if "Yes" in vc.index and "No" in vc.index:
            X[col] = (X[col] == "Yes").astype(int)

    # -----------------------------
    # Safe splits
    # -----------------------------
    X_train, X_val, X_trainval, X_test, y_train, y_val, y_trainval, y_test = make_splits(X, y)

    # -----------------------------
    # Pipeline + baseline (L2 Logistic)
    # -----------------------------
    pre = build_preprocessor(X_train)

    # Baseline model (L2 regularization is default;C=1.0 explicitly)
    baseline = Pipeline(steps=[
        ("pre", pre),
        ("model", LogisticRegression(
            l1_ratio=0,
            C=1.0,
            random_state=SEED
        ))
    ])

    baseline.fit(X_train, y_train)

    t = 0.455 
    probs_train = baseline.predict_proba(X_train)[:, 1]
    probs_val = baseline.predict_proba(X_val)[:, 1]

    train_res = eval_binary_classifier(y_train, probs_train, threshold=t)
    val_res = eval_binary_classifier(y_val, probs_val, threshold=t)

    print_eval("BASELINE — TRAIN", train_res)
    print_eval("BASELINE — VAL", val_res)

    # Quick baseline judgment signal (gap)
    gap = train_res[PRIMARY_METRIC] - val_res[PRIMARY_METRIC]
    print(f"\nBaseline {PRIMARY_METRIC} gap (train - val): {gap:.4f}")

    # -----------------------------
    # L2 regularization sweep (validation curve)
    # Complexity hyperparameter: C (inverse reg strength).
    # Smaller C => stronger regularization => simpler model.
    # -----------------------------
    C_values = np.logspace(-3, 3, 13)  # 0.001 ... 1000
    train_scores = []
    val_scores = []

    for C in C_values:
        model = Pipeline(steps=[
            ("pre", pre),
            ("model", LogisticRegression(
                l1_ratio=0,
                C=float(C),
                random_state=SEED
            ))
        ])
        model.fit(X_train, y_train)

        p_tr = model.predict_proba(X_train)[:, 1]
        p_va = model.predict_proba(X_val)[:, 1]

        tr = eval_binary_classifier(y_train, p_tr, threshold=t)[PRIMARY_METRIC]
        va = eval_binary_classifier(y_val, p_va, threshold=t)[PRIMARY_METRIC]

        train_scores.append(tr)
        val_scores.append(va)

    # Plot validation curve
    plt.figure()
    plt.semilogx(C_values, train_scores, marker="o", label=f"Train {PRIMARY_METRIC}")
    plt.semilogx(C_values, val_scores, marker="o", label=f"Val {PRIMARY_METRIC}")
    plt.xlabel("C (inverse L2 strength). Smaller C = more regularization.")
    plt.ylabel(PRIMARY_METRIC)
    plt.title("Validation Curve: L2 Logistic Regression")
    plt.legend()
    plt.savefig("validation curve", dpi=200)
    plt.close()

    # best C from the curve 
    best_idx = int(np.argmax(val_scores))
    chosen_C = float(C_values[best_idx])
    print(f"\nChosen C from validation curve (max val {PRIMARY_METRIC}): {chosen_C}")

    # -----------------------------
    # Small disciplined hyperparameter search (GridSearchCV)
    # -----------------------------
    pre2 = build_preprocessor(X_trainval)

    pipe = Pipeline(steps=[
        ("pre", pre2),
        ("model", LogisticRegression(
            l1_ratio=0,
            random_state=SEED
        ))
    ])

    # Small grid (scoped)
    param_grid = {
        "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
    }

    scoring = "f1" 
    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring=scoring,
        cv=5,
        n_jobs=-1,
        refit=True
    )
    grid.fit(X_trainval, y_trainval)

    print("\n=== GRID SEARCH RESULTS ===")
    print(f"Scoring: {scoring}")
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV score: {grid.best_score_:.4f}")

    # -----------------------------
    # training final model on TRAINVAL with best hyperparameters testing once.
    # -----------------------------
    final_model = grid.best_estimator_
    probs_test = final_model.predict_proba(X_test)[:, 1]
    test_res = eval_binary_classifier(y_test, probs_test, threshold=t)

    print_eval("FINAL (TUNED) — TEST", test_res)

    # Compare baseline vs tuned on VAL (baseline) and TEST (tuned)
    baseline_test_probs = baseline.predict_proba(X_test)[:, 1]
    baseline_test_res = eval_binary_classifier(y_test, baseline_test_probs, threshold=t)
    print_eval("BASELINE — TEST (evaluated only at end)", baseline_test_res)

    print("\n=== COMPARISON SUMMARY ===")
    print(f"Baseline TEST {PRIMARY_METRIC}: {baseline_test_res[PRIMARY_METRIC]:.4f}")
    print(f"Tuned    TEST {PRIMARY_METRIC}: {test_res[PRIMARY_METRIC]:.4f}")


if __name__ == "__main__":
    main()