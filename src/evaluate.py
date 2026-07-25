from __future__ import annotations

from sklearn.metrics import cohen_kappa_score


def _map_to_int(values: list[float]) -> tuple[list[int], dict[float, int]]:
    """Map continuous scores to integer indices for cohen_kappa_score."""
    all_vals = sorted(set(values))
    val_to_idx = {v: i for i, v in enumerate(all_vals)}
    return [val_to_idx[v] for v in values], val_to_idx


def exact_accuracy(preds: list[float], labels: list[float]) -> float:
    correct = sum(1 for p, l in zip(preds, labels) if p == l)
    return correct / len(labels)


def mae(preds: list[float], labels: list[float]) -> float:
    return sum(abs(p - l) for p, l in zip(preds, labels)) / len(labels)


def qwk(preds: list[float], labels: list[float]) -> float:
    """Quadratic Weighted Kappa."""
    all_vals = sorted(set(preds) | set(labels))
    val_to_idx = {v: i for i, v in enumerate(all_vals)}
    preds_int = [val_to_idx[p] for p in preds]
    labels_int = [val_to_idx[l] for l in labels]
    return cohen_kappa_score(labels_int, preds_int, weights="quadratic")


def tolerance_accuracy(preds: list[float], labels: list[float], tolerance: float) -> float:
    """Accuracy within tolerance: |pred - label| <= tolerance counts as correct."""
    correct = sum(1 for p, l in zip(preds, labels) if abs(p - l) <= tolerance)
    return correct / len(labels)


def tolerance_qwk(preds: list[float], labels: list[float], tolerance: float) -> float:
    """QWK after snapping predictions within tolerance to the label value."""
    adjusted = [l if abs(p - l) <= tolerance else p for p, l in zip(preds, labels)]
    all_vals = sorted(set(adjusted) | set(labels))
    val_to_idx = {v: i for i, v in enumerate(all_vals)}
    preds_int = [val_to_idx[a] for a in adjusted]
    labels_int = [val_to_idx[l] for l in labels]
    return cohen_kappa_score(labels_int, preds_int, weights="quadratic")


def compute_metrics(
    preds: list[float], labels: list[float], tolerance: float | None = None
) -> dict[str, float]:
    metrics = {
        "exact_accuracy": exact_accuracy(preds, labels),
        "mae": mae(preds, labels),
        "qwk": qwk(preds, labels),
    }
    if tolerance is not None:
        metrics[f"tolerance_acc({tolerance})"] = tolerance_accuracy(preds, labels, tolerance)
        metrics[f"tolerance_qwk({tolerance})"] = tolerance_qwk(preds, labels, tolerance)
    return metrics
