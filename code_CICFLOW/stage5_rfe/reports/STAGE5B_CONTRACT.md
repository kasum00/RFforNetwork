# Stage 5B Contract — Validation k-search and final evaluation

This document is a contract only. Stage 5B is not started by the Stage 5A
implementation.

2026-10-28 implementation update: the corrected
`notebooks/COMPACT_MI_RFE_CICFLOW_Colab_V2.ipynb` now calls the separate
`code_CICFLOW/colab/cicflow_reference_pipeline.py` research experiment runner.
That runner implements RF/SVM Validation k-search, frozen selection, final
Train+Validation scaler/classifier refit and one Test evaluation, with the
ColabPro experiment_results layout. The Stage 5A CLI remains Train-only.
No tests or training were executed by Codex. XGBoost ranking remains BLOCKED.

## Classifier hyperparameters confirmed by the user

The machine-readable classifier contract is
`D:\COMPACT-MI-RFE\code_CICFLOW\shared\classifier_contract.json`.

| Classifier | Confirmed parameters |
|---|---|
| Random Forest | `n_estimators=300`, **`max_depth=20`**, `min_samples_split=3` |
| SVM | `C=10.0`, `max_iter=20000` |
| XGBoost | `n_estimators=200`, `max_depth=10`, `min_child_weight=3` |

These parameters apply to the classifiers used for Validation k-search and
final refit. In the original RF script,
`CODE_ctu13/[[CTU13]9_ex2_rf.py`, `RF_PARAMS` explicitly contains
300 trees / depth 20 / split 3. Its separate `RFE_ESTIMATOR_PARAMS` dictionary
contains 100 trees / depth 10 / split 2. The SVM reference uses another RF
estimator with depth 15 for RFE. Those estimators determine feature rankings;
they are not the final classifiers.

The corrected Colab notebook includes the RF/SVM classifier implementation;
this does not assert that any classifier has already been fitted. Full RF
and compact-SVM dictionaries are reconciled with ciciomt24_training.py and
recorded with resolved estimator defaults in the new run identity. Remaining
XGBoost parameters/ranking still require reconciliation. Source/environment
binding and the contract hash must be frozen before
Stage 5B execution. Completed Stage 4/5A outputs must not be relabelled.

## Ranking inputs

For each configuration and classifier branch, Stage 5B consumes a locked Stage
5A ranking. It must verify the Stage 5A manifest and ranking digest before any
Validation value is read.

## Selection search

Search:

`k = 1, 2, ..., 100`

For each k, use the first k features from the locked Stage 5A ranking and fit
the explicitly contracted classifier on Train. Evaluate Accuracy on
Validation only. Select the k with maximum Validation Accuracy; on an exact
tie, select the smallest k.

No Test values may be read during k-search. No Test metric may influence k,
feature order, classifier hyperparameters, or branch selection.

## Final refit and Test

After k is frozen:

1. Refit the locked classifier on Train+Validation using exactly the selected
   k features.
2. Read Test once for final evaluation.
3. Produce Accuracy, Precision, Recall, F1, confusion matrix, runtime, and
   provenance/identity fields.

The final evaluation must be a separate, auditable run. It must not rewrite
Stage 4 or Stage 5A artifacts.

## XGBoost

XGBoost remains blocked until its estimator, version, parameters, ranking
semantics, and source compatibility are explicitly reconciled.

