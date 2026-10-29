# A1 implementation handoff

Implementation written only. Static source review covered paths, imports, identity formation, stage/per-k checkpoint flow, rank ties, selection-only loader and CLI modes. No shell, Python, tests, preflight, Colab, training, benchmark, Git or package installation was executed. Runtime/syntax/import/FUSE integration and numerical parity remain unverified until the user runs the provided tests/notebook.

## New files only

All new files are under `code_CICFLOW/phuong_anA`:

| File | Purpose |
|---|---|
| `__init__.py`, `.gitignore`, `requirements-colab.txt` | package and explicit dependency profile |
| `contract.py` | A1 algorithm/source/parameter identity, exact isolated paths |
| `dataset.py` | manifest-driven config/count/label/split validation, Train/Val-only cached loader |
| `rfe.py` | step5 target1 elimination, removal-round importance, grouped rank/total order and resume checks |
| `store.py` | immutable per-stage/per-k deltas, SHA256/readback, parent chain, leases, restore/non-overwrite |
| `telemetry.py` | measured CPU/RAM/fit elapsed, periodic heartbeat, safe session deadline/stop, log deltas |
| `preflight.py` | source/Git/environment, manifests, resource/test evidence and run guards |
| `pipeline.py` | adapter calling unchanged A0 numerical functions, stage and100-k orchestration, selection verify |
| `results.py` | full Validation metrics/selection checks, result/report publication, no Test |
| `cli.py`, `controller.py` | isolated worker, native exit/status preservation, live stdout/stderr logging |
| `selftest.py`, `tests/__init__.py`, `tests/fixtures.py`, `tests/test_a1.py` | mock/tiny synthetic tests, source-bound evidence |
| `notebooks/PHUONG_ANA_A1_COLAB.ipynb` |8 cells, default preflight |
| `evidence/cicflow_split_evidence.json`, `evidence/README.md` | byte-exact small original CICFLOW split/proxy source snapshots and origin notes |
| `README.md`, `reports/A1_DESIGN.md`, `reports/A0_A1_COMPARISON.md`, `reports/COLAB_GUIDE.md`, this report | usage/contracts/comparison/limitations |

No A0/B source, raw dataset, existing result/checkpoint, original CTU script or reference notebook was edited. The evidence snapshots are copies of small original metadata/source bytes, not newly generated scientific artifacts. No large provenance CSV or Parquet is embedded.

## Invariants

380-feature order, cleaning/scaling, Train-only MI and RF-RFE, MI Top100/seed/defaults, original SVM-branch RF estimator, SVC RBF parameters, exhaustive k1..100, one predict per k, strict-greater Accuracy selection with smallest-k ties and source/data/checkpoint SHA guards are preserved. Test access and final refit are prohibited in A1's selection stage. Algorithm status is always EXPERIMENTAL_NON_OFFICIAL.

Only the scientific RFE step changes1→5, target stays1. Importance-based total-order tie handling is newly explicit because sklearn group ranks are no longer unique. Persistence/telemetry is independent: A1 checkpoints every stage/step/k instead of inheriting the old five-hour Drive scheduler.

## Tests written, not run

Coverage includes sklearn step5 grouped-rank parity with a tied fake estimator, importance/MI tie order, partial elimination resume, Train-only RFE, locked parameters/MI invariants, all100 SVC fits/predicts with fake models and real per-k checkpoint/restore, smaller-k ties, configurable session budget, actual fixture manifest row counts, forbidden Test I/O including footer/hash, Train/Val checksum errors, missing/invalid label/split evidence, upload interruption and incomplete-directory rejection, source/config/parameter/dataset/algorithm mismatches, duplicate/foreign/completed output protection, corruption, selection publication/verification and no model deserialization during verify. No real SVC is fit and no production dataset is used in tests.

## BLOCKED conditions / practical limits

- Missing7 original Stage3 sidecars, bad completion/schema/count/digest, foreign parent evidence, changed Train/Val SHA, unsupported proxy provenance or failed split/embargo evidence.
- Source/Git/library/default/scaler/ranking/parameters/config/run/algorithm identity mismatch, stale tests, reused A0/B checkpoint, gapped/corrupt checkpoint chain or different local artifacts.
- Existing foreign output, active/unacknowledged old lease, RAM/disk admission failure, telemetry/upload failure or partial conflicting publication. No automatic deletion, source repair or parameter reduction.
- Completed selection remains immutable. FUSE persistence is best effort; checkpoint readback is verified, internal cloud replication is not certified.
- An interrupted SVC cannot resume mid-fit; only completed k are retained. Time budget is a safe-boundary policy and may be exceeded by one long fit. Local cache needs rebuilding after VM loss.
- Historical split checks and proxy producer are bound, not re-audited against all raw flows. Capture/session/near-duplicate/ground-truth limitations remain. Test physical bytes are not verified because Test is never opened.
- Theoretical forest fits100→21 do not establish measured speedup or equal metrics. Actual A0 comparison fields remain null; final/Test remain unauthorized.
