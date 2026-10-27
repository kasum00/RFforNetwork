# Shared CICFLOW contracts

Shared utilities may be added here only when they are read-only compatible
with the existing dataset and Stage 1–3 artifacts. A shared utility must not
silently modify source data, labels, split membership, provenance or existing
results.

Stage 0 currently keeps its implementation self-contained in
`stage0_readiness/scientific_readiness.py` so that the readiness gate can be
reviewed independently.
