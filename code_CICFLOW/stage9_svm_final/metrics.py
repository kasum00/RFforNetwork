"""Explicit positive label, macro averaging and immutable prediction verification."""
import numpy as np
from .common import Blocked


def compute(y, prediction, labels):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                                 f1_score, precision_score, recall_score)
    y, prediction = np.asarray(y), np.asarray(prediction)
    benign, attack = labels["benign"], labels["attack"]
    if (y.ndim != 1 or y.shape != prediction.shape or not len(y) or benign == attack
            or not set(np.unique(y)) <= {benign, attack} or not set(np.unique(prediction)) <= {benign, attack}):
        raise Blocked("Invalid Test labels/prediction alignment")
    return dict(accuracy=float(accuracy_score(y, prediction)),
        precision_attack=float(precision_score(y, prediction, pos_label=attack, zero_division=0)),
        recall_attack=float(recall_score(y, prediction, pos_label=attack, zero_division=0)),
        f1_attack=float(f1_score(y, prediction, pos_label=attack, zero_division=0)),
        macro_f1=float(f1_score(y, prediction, labels=[benign, attack], average="macro", zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y, prediction)),
        confusion_matrix=confusion_matrix(y, prediction, labels=[benign, attack]).tolist(),
        confusion_matrix_labels=[benign, attack], positive_label=attack)
