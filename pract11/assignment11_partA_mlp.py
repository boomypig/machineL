
import copy
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Configuration
# ============================================================
DATA_PATH = "./saas_customer_churn_mlp.csv"
TARGET = "churn_risk"
RANDOM_STATE = 119
TEST_SIZE = 0.15
VAL_SIZE = 0.15
BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Helper functions
# ============================================================
def print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def metric_dict(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def print_metrics(name: str, y_true, y_pred) -> dict:
    metrics = metric_dict(y_true, y_pred)
    print(f"\n{name}")
    for metric_name, value in metrics.items():
        print(f"{metric_name:>10}: {value:.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    return metrics


def evaluate_logits(y_true: np.ndarray, logits: np.ndarray, threshold: float = 0.5) -> tuple[dict, np.ndarray]:
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)
    return metric_dict(y_true, preds), preds


# ============================================================
# Data loading and splitting
# ============================================================
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def create_splits(df: pd.DataFrame):
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    val_fraction_of_trainval = VAL_SIZE / (1 - TEST_SIZE)

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_fraction_of_trainval,
        random_state=RANDOM_STATE,
        stratify=y_trainval,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# Preprocessing
# ============================================================
def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )
    return preprocessor


# ============================================================
# Baseline model
# ============================================================
def train_baseline(X_train, y_train, X_val, y_val):
    baseline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )
    baseline.fit(X_train, y_train)

    y_train_pred = baseline.predict(X_train)
    y_val_pred = baseline.predict(X_val)

    print_header("Baseline Model: Logistic Regression")
    train_metrics = print_metrics("TRAINING METRICS", y_train, y_train_pred)
    val_metrics = print_metrics("VALIDATION METRICS", y_val, y_val_pred)

    return baseline, train_metrics, val_metrics


# ============================================================
# PyTorch MLP
# ============================================================
class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(32,), dropout=0.1):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(1)


@dataclass
class TrainingHistory:
    train_loss: list
    val_loss: list
    val_f1: list
    val_accuracy: list


def make_tensor_dataloader(X_array, y_array, batch_size=BATCH_SIZE, shuffle=False):
    X_tensor = torch.tensor(X_array, dtype=torch.float32)
    y_tensor = torch.tensor(np.asarray(y_array), dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_mlp(
    X_train_processed,
    y_train,
    X_val_processed,
    y_val,
    hidden_dims=(32,),
    dropout=0.1,
    weight_decay=WEIGHT_DECAY,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
):
    train_loader = make_tensor_dataloader(X_train_processed, y_train, shuffle=True)
    val_loader = make_tensor_dataloader(X_val_processed, y_val, shuffle=False)

    model = MLPClassifier(
        input_dim=X_train_processed.shape[1],
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()

    history = TrainingHistory(train_loss=[], val_loss=[], val_f1=[], val_accuracy=[])
    best_state = None
    best_val_f1 = -1.0

    for epoch in range(1, epochs + 1):
        # ---------- training ----------
        model.train()
        running_train_loss = 0.0
        total_train_examples = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            batch_size = X_batch.size(0)
            running_train_loss += loss.item() * batch_size
            total_train_examples += batch_size

        avg_train_loss = running_train_loss / total_train_examples

        # ---------- validation ----------
        model.eval()
        running_val_loss = 0.0
        total_val_examples = 0
        all_val_logits = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                logits = model(X_batch)
                loss = criterion(logits, y_batch)

                batch_size = X_batch.size(0)
                running_val_loss += loss.item() * batch_size
                total_val_examples += batch_size
                all_val_logits.append(logits.cpu().numpy())

        avg_val_loss = running_val_loss / total_val_examples
        val_logits = np.concatenate(all_val_logits)
        val_metrics, _ = evaluate_logits(np.asarray(y_val), val_logits)

        history.train_loss.append(avg_train_loss)
        history.val_loss.append(avg_val_loss)
        history.val_f1.append(val_metrics["f1"])
        history.val_accuracy.append(val_metrics["accuracy"])

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = copy.deepcopy(model.state_dict())

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:>2}/{epochs} | "
                f"train_loss={avg_train_loss:.4f} | "
                f"val_loss={avg_val_loss:.4f} | "
                f"val_f1={val_metrics['f1']:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f}"
            )

    model.load_state_dict(best_state)
    return model, history


def predict_mlp(model, X_array):
    model.eval()
    X_tensor = torch.tensor(X_array, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits = model(X_tensor).cpu().numpy()
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    return logits, probs, preds


def plot_training_dynamics(history: TrainingHistory, out_path="mlp_training_dynamics.png"):
    epochs = np.arange(1, len(history.train_loss) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.train_loss, label="Train loss")
    plt.plot(epochs, history.val_loss, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MLP Training Dynamics: Loss vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.val_f1, label="Validation F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title("MLP Training Dynamics: Validation F1 vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig("mlp_validation_f1.png", dpi=200)
    plt.close()


# ============================================================
# Main workflow
# ============================================================
def main():
    print_header("Load dataset")
    df = load_data(DATA_PATH)
    print(df.head())
    print(f"\nShape: {df.shape}")
    print("\nMissing values by column:")
    print(df.isna().sum().sort_values(ascending=False))
    print("\nTarget distribution:")
    print(df[TARGET].value_counts(normalize=True).sort_index())

    X_train, X_val, X_test, y_train, y_val, y_test = create_splits(df)

    print_header("Split sizes")
    print(f"Train: {X_train.shape[0]}")
    print(f"Val:   {X_val.shape[0]}")
    print(f"Test:  {X_test.shape[0]}")

    # ---------------- baseline ----------------
    baseline_model, baseline_train_metrics, baseline_val_metrics = train_baseline(
        X_train, y_train, X_val, y_val
    )

    # ---------------- preprocessing for MLP ----------------
    print_header("Preprocessing for MLP")
    preprocessor = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    # convert sparse matrix to dense for PyTorch
    X_train_processed = X_train_processed.toarray() if hasattr(X_train_processed, "toarray") else X_train_processed
    X_val_processed = X_val_processed.toarray() if hasattr(X_val_processed, "toarray") else X_val_processed
    X_test_processed = X_test_processed.toarray() if hasattr(X_test_processed, "toarray") else X_test_processed

    print(f"Processed train shape: {X_train_processed.shape}")
    print(f"Processed val shape:   {X_val_processed.shape}")

    # ---------------- default small MLP ----------------
    print_header("MLP Model: Small Architecture")
    mlp_model, mlp_history = train_mlp(
        X_train_processed,
        y_train,
        X_val_processed,
        y_val,
        hidden_dims=(32,),
        dropout=0.1,
        weight_decay=1e-4,
        epochs=EPOCHS,
    )

    train_logits, _, train_preds = predict_mlp(mlp_model, X_train_processed)
    val_logits, _, val_preds = predict_mlp(mlp_model, X_val_processed)

    mlp_train_metrics = print_metrics("TRAINING METRICS", y_train, train_preds)
    mlp_val_metrics = print_metrics("VALIDATION METRICS", y_val, val_preds)

    plot_training_dynamics(mlp_history)

    # ---------------- controlled variation ----------------
    print_header("Controlled Variation: Increase Hidden Layer Size")
    varied_model, varied_history = train_mlp(
        X_train_processed,
        y_train,
        X_val_processed,
        y_val,
        hidden_dims=(64,),   # exact one change: hidden layer size 32 -> 64
        dropout=0.1,
        weight_decay=1e-4,
        epochs=EPOCHS,
    )

    _, _, varied_val_preds = predict_mlp(varied_model, X_val_processed)
    varied_val_metrics = print_metrics("VALIDATION METRICS (Hidden Size = 64)", y_val, varied_val_preds)

    # ---------------- model comparison ----------------
    print_header("Validation Comparison")
    print(f"Baseline Logistic Regression val F1: {baseline_val_metrics['f1']:.4f}")
    print(f"MLP (32)                    val F1: {mlp_val_metrics['f1']:.4f}")
    print(f"MLP (64)                    val F1: {varied_val_metrics['f1']:.4f}")

    candidates = [
        ("baseline", baseline_val_metrics["f1"]),
        ("mlp_32", mlp_val_metrics["f1"]),
        ("mlp_64", varied_val_metrics["f1"]),
    ]
    best_model_name = max(candidates, key=lambda x: x[1])[0]
    print(f"\nSelected final model based on validation F1: {best_model_name}")

    # ---------------- final test evaluation ----------------
    print_header("Final Test Evaluation")
    if best_model_name == "baseline":
        y_test_pred = baseline_model.predict(X_test)
        print_metrics("TEST METRICS - BASELINE", y_test, y_test_pred)

    elif best_model_name == "mlp_32":
        _, _, y_test_pred = predict_mlp(mlp_model, X_test_processed)
        print_metrics("TEST METRICS - MLP (32)", y_test, y_test_pred)

    else:
        _, _, y_test_pred = predict_mlp(varied_model, X_test_processed)
        print_metrics("TEST METRICS - MLP (64)", y_test, y_test_pred)

    print("\nSaved plots:")
    print("- mlp_training_dynamics.png")
    print("- mlp_validation_f1.png")


if __name__ == "__main__":
    main()
