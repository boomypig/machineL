# Fashion-MNIST CNN Classifier

A PyTorch CNN that classifies Fashion-MNIST images into 10 clothing categories, achieving ~90.7% test accuracy.

## Model

A small CNN with two convolutional blocks (16 and 32 filters) followed by a fully connected classifier with dropout. Trained for 10 epochs using Adam optimizer.

## Setup

Requires Python 3.12+.

```bash
uv sync
```

## Usage

```bash
uv run python fashionmachine.py
```

The script downloads Fashion-MNIST automatically, trains the model, and saves:

- `best_fashion_mnist_cnn.pt` -- best model checkpoint
- `fashion_mnist_training_plot.png` -- loss curves
- `fashion_mnist_metrics.json` -- training history and final results

## Results

| Split      | Accuracy | Loss  |
|------------|----------|-------|
| Validation | 90.9%    | 0.245 |
| Test       | 90.7%    | 0.258 |
