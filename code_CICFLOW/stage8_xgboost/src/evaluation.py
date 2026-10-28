"""Metrics only; this module never opens Test or models."""
from .common import load_core

def evaluate(source, labels, prediction):
    result = load_core(source).metrics(labels, prediction)
    report = result["classification_report"]
    result.update(precision_benign=report["Benign_proxy"]["precision"],
                  f1_benign=report["Benign_proxy"]["f1-score"],
                  precision_attack_per_class=report["Attack_proxy"]["precision"],
                  recall_attack_per_class=report["Attack_proxy"]["recall"],
                  f1_attack_per_class=report["Attack_proxy"]["f1-score"])
    return result
