"""Post-hoc score calibration utilities for ordinal scoring models."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_linear_calibration(
    preds: list[float], labels: list[float]
) -> dict:
    """Fit y = a * x + b by least squares on (preds, labels)."""
    x = np.asarray(preds, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    a, b = np.polyfit(x, y, 1)
    return {"method": "linear", "a": float(a), "b": float(b)}


def fit_isotonic_calibration(
    preds: list[float], labels: list[float]
) -> dict:
    """Fit monotonic isotonic regression mapping preds -> labels."""
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(np.asarray(preds), np.asarray(labels))
    return {"method": "isotonic", "model": iso}


def fit_calibration(
    preds: list[float], labels: list[float], method: str = "linear"
) -> dict | None:
    """Fit calibration according to method: none, linear, or isotonic."""
    if method == "linear":
        return fit_linear_calibration(preds, labels)
    if method == "isotonic":
        return fit_isotonic_calibration(preds, labels)
    return None


def apply_calibration(
    preds: list[float], calibration: dict | None
) -> list[float]:
    """Apply calibration transform to raw predictions."""
    if calibration is None:
        return list(preds)
    method = calibration.get("method")
    if method == "linear":
        a = calibration["a"]
        b = calibration["b"]
        return [a * p + b for p in preds]
    if method == "isotonic":
        model = calibration["model"]
        return model.predict(np.asarray(preds)).tolist()
    return list(preds)


def snap_to_score_points(
    preds: list[float], score_points: list[float]
) -> list[float]:
    """Snap continuous predictions to the nearest score point."""
    pts = np.asarray(sorted(score_points), dtype=np.float64)
    arr = np.asarray(preds, dtype=np.float64)
    idx = np.argmin(np.abs(arr[:, None] - pts[None, :]), axis=1)
    return pts[idx].tolist()
