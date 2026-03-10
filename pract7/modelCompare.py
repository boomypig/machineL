from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

def createSplits(matrices, messages):
    y = messages["label"].to_numpy(dtype=int)

    print("Label distribution:", np.bincount(y))

    with open("./data/split.json", "r") as f:
        split = json.load(f)

    train_ids = np.array(split["train_ids"], dtype=int)
    val_ids   = np.array(split["val_ids"], dtype=int)
    test_ids  = np.array(split["test_ids"], dtype=int)

    print("Train/Val/Test sizes:", len(train_ids), len(val_ids), len(test_ids))

    X = matrices

    X_train = X[train_ids]
    X_val   = X[val_ids]
    X_test  = X[test_ids]

    y_train = y[train_ids]
    y_val   = y[val_ids]
    y_test  = y[test_ids]

    print("Training feature matrix shape:", X_train.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test, train_ids, val_ids, test_ids


def evaluate_model(model,X,y,split_name="validation"):
    y_pred = model.predict(X)
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred), 
        "f1": f1_score(y, y_pred)
    }

    print(f"\n {split_name.upper()} METRICS")
    for metric_name, value in metrics.items():
        print(f"{metric_name:>10}: {value:4f}")
    print("confusion Matrix:")
    print(confusion_matrix(y, y_pred))
    return y_pred, metrics

def show_mistakes(messages,indices, y_true,y_pred, model_name, max_examples = 5):
    print(f"\n Representatives mistakes for {model_name}:")
    wrong_mask = y_true != y_pred
    wrong_positions = np.where(wrong_mask)[0]

    if len(wrong_positions) == 0:
        print("no mistakes were found")
        return
    # Try to find a text column automatically
    possible_text_cols = ["text", "message", "sms", "body", "content"]
    text_col = None
    for col in possible_text_cols:
        if col in messages.columns:
            text_col = col
            break

    for i, pos in enumerate(wrong_positions[:max_examples], start=1):
        original_idx = indices[pos]
        row = messages.iloc[original_idx]

        print(f"\nExample {i}")
        print(f"True label: {y_true[pos]}")
        print(f"Pred label : {y_pred[pos]}")

        if text_col is not None:
            print(f"Text: {row[text_col]}")
        else:
            print("Row preview:")
            print(row.to_dict())

def train_naive_bayes(X_train, y_train, X_val, y_val):
    print("\n" + "=" * 60)
    print("Training Multinomial Naive Bayes")
    print("=" * 60)
    nb_model = MultinomialNB()
    nb_model.fit(X_train,y_train)

    y_val_pred, val_metrics = evaluate_model(nb_model,X_val,y_val,split_name="validation")
    return nb_model, val_metrics, y_val_pred

def train_knn(X_train, y_train, X_val, y_val, k_values ):
    print("\n" + "=" * 60)
    print("Training Knn model")
    print("=" *60)

    results = []
    best_model = None
    best_metrics = None
    best_preds = None
    best_f1 = -1

    for k in k_values:
        print(f"attempting knn model with k:{k}")
        knn_model = Pipeline(steps=[
            ("scalar", MaxAbsScaler()), 
            ("knn", KNeighborsClassifier(n_neighbors=k, metric="cosine", algorithm="brute"))
        ])
        knn_model.fit(X_train, y_train)
        y_val_pred, val_metrics = evaluate_model(knn_model, X_val,y_val,split_name="validation")

        row = {
            "model": "kNN",
            "k": k,
            "metric": "cosine",
            "accuracy": val_metrics["accuracy"],
            "precision": val_metrics["precision"],
            "recall": val_metrics["recall"],
            "f1": val_metrics["f1"],
        }
        results.append(row)

        if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                best_model = knn_model
                best_metrics = val_metrics
                best_preds = y_val_pred

    return best_model, best_metrics, best_preds, results


def final_test_evaluation(model, X_test, y_test, winner_name):
    print("\n" + "=" * 60)
    print(f"FINAL TEST EVALUATION: {winner_name}")
    print("=" * 60)

    test_metrics, y_test_pred = evaluate_model(model, X_test, y_test, split_name="test")
    return test_metrics, y_test_pred


def main():
    DATA_DIR = Path("./data")

    # Load messages
    messages = pd.read_csv(DATA_DIR / "messages.csv")
    print("Messages shape:", messages.shape)
    print(messages.head())
    print(messages.columns)

    # Load sparse matrices
    X_counts = sparse.load_npz(DATA_DIR / "X_counts.npz")
    X_tfidf  = sparse.load_npz(DATA_DIR / "X_tfidf.npz")

    print("Counts matrix shape:", X_counts.shape)
    print("TF-IDF matrix shape:", X_tfidf.shape)

    # Shared fixed splits
    x_train_multi, x_val_multi, x_test_multi, y_train_multi, y_val_multi, y_test_multi, train_ids, val_ids, test_ids = createSplits(X_counts, messages)
    x_train_knn, x_val_knn, x_test_knn, y_train_knn, y_val_knn, y_test_knn, _, _, _ = createSplits(X_tfidf, messages)

    # --------------------------
    # Naive Bayes
    # --------------------------
    nb_model, nb_val_metrics, nb_val_pred = train_naive_bayes(
        x_train_multi, y_train_multi, x_val_multi, y_val_multi
    )

    # --------------------------
    # kNN (at least 3 k values)
    # --------------------------
    best_knn_model, best_knn_metrics, best_knn_val_pred, knn_rows = train_knn(
        x_train_knn, y_train_knn, x_val_knn, y_val_knn, k_values=(3, 5, 11)
    )

    # --------------------------
    # Comparison table
    # --------------------------
    comparison_rows = [
        {
            "model": "MultinomialNB",
            "k": "-",
            "metric": "-",
            "accuracy": nb_val_metrics["accuracy"],
            "precision": nb_val_metrics["precision"],
            "recall": nb_val_metrics["recall"],
            "f1": nb_val_metrics["f1"],
        }
    ] + knn_rows

    comparison_df = pd.DataFrame(comparison_rows)
    print("\n" + "=" * 60)
    print("VALIDATION COMPARISON TABLE")
    print("=" * 60)
    print(comparison_df.sort_values(by="f1", ascending=False).to_string(index=False))

    # --------------------------
    # Error analysis
    # --------------------------
    show_mistakes(messages, val_ids, y_val_multi, nb_val_pred, "MultinomialNB", max_examples=3)
    show_mistakes(messages, val_ids, y_val_knn, best_knn_val_pred, "Best kNN", max_examples=3)

    # --------------------------
    # Pick winner based on validation F1
    # --------------------------
    if nb_val_metrics["f1"] >= best_knn_metrics["f1"]:
        winner_name = "MultinomialNB"
        winner_model = nb_model
        winner_X_test = x_test_multi
        winner_y_test = y_test_multi
    else:
        winner_name = "kNN"
        winner_model = best_knn_model
        winner_X_test = x_test_knn
        winner_y_test = y_test_knn

    print(f"\nSelected winner based on validation F1: {winner_name}")

    # --------------------------
    # Final test evaluation
    # --------------------------
    final_test_evaluation(winner_model, winner_X_test, winner_y_test, winner_name)


main()