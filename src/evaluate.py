from __future__ import annotations

from sklearn.metrics import cohen_kappa_score


def exact_accuracy(preds: list[float], labels: list[float]) -> float:
    correct = sum(1 for p, l in zip(preds, labels) if p == l)
    return correct / len(labels)


def mae(preds: list[float], labels: list[float]) -> float:
    return sum(abs(p - l) for p, l in zip(preds, labels)) / len(labels)


def qwk(preds: list[float], labels: list[float]) -> float:
    return cohen_kappa_score(labels, preds, weights="quadratic")


def compute_metrics(preds: list[float], labels: list[float]) -> dict[str, float]:
    return {
        "exact_accuracy": exact_accuracy(preds, labels),
        "mae": mae(preds, labels),
        "qwk": qwk(preds, labels),
    }
