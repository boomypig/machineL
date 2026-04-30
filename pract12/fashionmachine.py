#!/usr/bin/env python3
"""
Assignment 12 – Part A: Practical Deep Learning and Frameworks
Fashion-MNIST with PyTorch

What this script does:
- builds train/validation/test datasets and dataloaders
- prints batch shapes/dtypes
- defines a small CNN
- trains with explicit PyTorch loops
- logs train/validation loss and accuracy each epoch
- saves the best checkpoint by validation accuracy
- reloads the checkpoint and verifies matching validation results
- evaluates once on the test set
- saves a training plot
- prints package/version/device information for reproducibility

Run:
    python assignment12_partA_fashion_mnist.py
"""

from __future__ import annotations

import copy
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# ----------------------------
# Configuration
# ----------------------------
@dataclass
class Config:
    data_dir: str = "./fashion.csv"
    batch_size: int = 128
    num_workers: int = 0
    val_size: int = 10000
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    random_seed: int = 119
    checkpoint_path: str = "best_fashion_mnist_cnn.pt"
    plot_path: str = "fashion_mnist_training_plot.png"
    metrics_path: str = "fashion_mnist_metrics.json"
    device_preference: str = "auto"  # auto, cpu, cuda


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # These improve reproducibility. They can slightly reduce speed.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(preference: str = "auto") -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_run_metadata(cfg: Config, device: torch.device) -> None:
    metadata = {
        "config": asdict(cfg),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    with open("run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


# ----------------------------
# Data
# ----------------------------
def build_datasets(cfg: Config):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    full_train = datasets.FashionMNIST(
        root=cfg.data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.FashionMNIST(
        root=cfg.data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_size = len(full_train) - cfg.val_size
    val_size = cfg.val_size

    generator = torch.Generator().manual_seed(cfg.random_seed)
    train_dataset, val_dataset = random_split(
        full_train,
        [train_size, val_size],
        generator=generator,
    )
    return train_dataset, val_dataset, test_dataset


def build_dataloaders(cfg: Config):
    train_dataset, val_dataset, test_dataset = build_datasets(cfg)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def inspect_one_batch(loader: DataLoader) -> None:
    images, labels = next(iter(loader))
    print("=" * 70)
    print("Batch inspection")
    print("=" * 70)
    print(f"images shape: {tuple(images.shape)}")
    print(f"images dtype: {images.dtype}")
    print(f"labels shape: {tuple(labels.shape)}")
    print(f"labels dtype: {labels.dtype}")
    print(f"min/max image values: {images.min().item():.4f} / {images.max().item():.4f}")
    print()


# ----------------------------
# Model
# ----------------------------
class SmallFashionCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 28x28 -> 14x14

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 14x14 -> 7x7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# ----------------------------
# Metrics / Evaluation
# ----------------------------
def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


# ----------------------------
# Training
# ----------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_accuracy: float,
    cfg: Config,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_accuracy": best_val_accuracy,
        "config": asdict(cfg),
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> Dict:
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def plot_history(history: Dict[str, List[float]], save_path: str) -> None:
    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Fashion-MNIST: Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main() -> None:
    cfg = Config()
    set_seed(cfg.random_seed)
    device = get_device(cfg.device_preference)
    save_run_metadata(cfg, device)

    print("=" * 70)
    print("Deep Learning")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Torch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    print()

    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    inspect_one_batch(train_loader)

    model = SmallFashionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_val_accuracy = -1.0
    best_epoch = -1
    best_val_metrics = None

    print("=" * 70)
    print("Training")
    print("=" * 70)

    training_start = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.perf_counter()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        epoch_seconds = time.perf_counter() - epoch_start

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"time={epoch_seconds:.2f}s"
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            best_val_metrics = copy.deepcopy(val_metrics)
            save_checkpoint(
                path=cfg.checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_accuracy=best_val_accuracy,
                cfg=cfg,
            )

    total_training_seconds = time.perf_counter() - training_start
    print()
    print(f"Total training time: {total_training_seconds:.2f}s")
    print(f"Best checkpoint saved at epoch {best_epoch} with val_acc={best_val_accuracy:.4f}")
    print()

    plot_history(history, cfg.plot_path)

    print("=" * 70)
    print("Checkpoint reload verification")
    print("=" * 70)

    reloaded_model = SmallFashionCNN().to(device)
    reloaded_optimizer = torch.optim.Adam(
        reloaded_model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    checkpoint = load_checkpoint(
        path=cfg.checkpoint_path,
        model=reloaded_model,
        optimizer=reloaded_optimizer,
        map_location=device,
    )

    reloaded_val_metrics = evaluate(
        model=reloaded_model,
        loader=val_loader,
        criterion=criterion,
        device=device,
    )

    print(
        "Saved best validation accuracy: "
        f"{checkpoint['best_val_accuracy']:.4f}"
    )
    print(
        "Reloaded model validation accuracy: "
        f"{reloaded_val_metrics['accuracy']:.4f}"
    )
    print(
        "Reloaded model validation loss: "
        f"{reloaded_val_metrics['loss']:.4f}"
    )

    matches = np.isclose(
        checkpoint["best_val_accuracy"],
        reloaded_val_metrics["accuracy"],
        atol=1e-8,
    )
    print(f"Reload verification passed: {matches}")
    print()

    print("=" * 70)
    print("Final test evaluation")
    print("=" * 70)

    test_metrics = evaluate(
        model=reloaded_model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print(f"Validation accuracy (best checkpoint): {reloaded_val_metrics['accuracy']:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test loss: {test_metrics['loss']:.4f}")
    print()

    results = {
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "reloaded_val_metrics": reloaded_val_metrics,
        "test_metrics": test_metrics,
        "history": history,
        "training_time_seconds": total_training_seconds,
    }

    with open(cfg.metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Saved files:")
    print(f"- {cfg.checkpoint_path}")
    print(f"- {cfg.plot_path}")
    print(f"- {cfg.metrics_path}")
    print("- run_metadata.json")
    print()
if __name__ == "__main__":
    main()
