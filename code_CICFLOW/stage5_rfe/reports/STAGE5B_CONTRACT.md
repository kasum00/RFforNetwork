# Stage 5B Contract — Validation k-search and final evaluation

This document is a contract only. Stage 5B is not started by the Stage 5A
implementation.

## Inputs

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

