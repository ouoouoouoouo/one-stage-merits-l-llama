from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label_names: Sequence[str] | None = None,
) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
    }
    if label_names is not None:
        per_class = f1_score(
            y_true, y_pred, average=None, labels=list(range(len(label_names))), zero_division=0
        )
        for name, f in zip(label_names, per_class):
            out[f"f1/{name}"] = float(f)
    return out


def detailed_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label_names: Sequence[str] | None = None,
) -> str:
    target_names = list(label_names) if label_names is not None else None
    text = classification_report(
        y_true, y_pred, target_names=target_names, digits=4, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    return text + "\nConfusion matrix:\n" + np.array2string(cm)
