# A1 scientific and persistence contract

Status: source written and reviewed statically, not executed. Algorithm identity is `MI_RFE_SVM_A1_STEP5_TARGET1`; all scientific outputs are `EXPERIMENTAL_NON_OFFICIAL` and `OFFICIAL_BLOCKED`.

## Preserved numerical functions

The adapter calls unchanged `code_CICFLOW/colab/cicflow_reference_pipeline.py` for cleaning/read_values, MI, saved MI, SVC factory/timed fit, metrics, best-k freezing and run lock. It calls unchanged `mi_rfe_svm_colab.runner.scaled_top`, `ranked_arrays`, `search`, `freeze`, plus local cache and k-history helpers. Those modules are hashed, never edited. The old A0 RFE and five-hour Drive policy are untouched. No phuong_anB source/checkpoint is accessed for training.

MI is the A0 reference-layout implementation: clean float64 numeric/nonfinite0 then float32; Train-only StandardScaler; one mutual_info_classif on the ordered380-feature matrix, seed42, dense auto discrete_features and default n_neighbors3; pandas descending ranking; Top100. This is not the older Stage4 per-feature seed42+index implementation. Preserving the A0 implementation does not prove numerical equivalence to a paper or to every historical MI run.

Top100 gets the same new Train-only scaler; Validation transform only. Float32 scaling arithmetic/layout is preserved; SVC gets the same C-order float64 prefix conversion as A0. RF-RFE keeps100 trees, depth15, balanced, seed42, n_jobs=-1 and the same library defaults. SVC keeps RBF/C10/gamma scale/probabilityFalse/cache10000/shrinkingTrue/max_iter20000/balanced/seed42. k=1..100 is exhaustive, sequential, one full Train fit and one full Validation prediction per k; metrics include Accuracy, attack precision/recall/F1, benign proxy recall, macro-F1, balanced accuracy, confusion matrix and fit/convergence information. Exact Accuracy ties retain the smaller k. No final refit/Test entry point exists.

## Exact A1 difference and ties

The new adapter implements sklearn RFE's step5/target1 elimination loop so that per-round importance and resume state can be captured. It clones the same RF estimator at each fit, squares importances, applies NumPy's default argsort, removes min(5,remaining-1), increments all eliminated ranks, and fits the final one-feature estimator. Tiny synthetic parity tests against installed sklearn RFE are provided but have not been run by Codex.

For100 input features,20 elimination rounds and21 total forest fits follow arithmetically. Native ranking has21 groups, not100 distinct ranks. A deterministic full Top-k order uses `(sklearn rank ascending, removal-round squared importance descending, original MI position ascending)`. The survivor is rank1; last batch may contain4 features. CSV preserves native rank, total-order position, MI position, removal round and importance separately. This policy is experimental and cannot recreate the step1 elimination sequence. Reproducibility is scoped to the identical input/source/environment; identical floating-point RF importance reductions across arbitrary machines are not guaranteed.

## Dataset evidence and isolation

Config discovery reads the actual Stage3 completed manifests in CICFLOWMETER_FEATURE_CSV. Rows come from those manifests and Parquet metadata; no CICIoMT2024 counts are imported. Required local sidecars: `.stage3_preflight.json`, `.stage3_complete.json`, `feature_schema.json`, `window_feature_schema.json`, `window_integrity.json`, `temporal_gap_audit.json`, `window_counts_by_segment.csv`.

The preflight rechecks the Stage3 binding digest, completion/integrity hashes, schema/count/gap hashes, original small V2 manifest hashes, parent guarded-export checks, role totals, segment source mapping and cross-role rank intervals. Train/Validation model files are fully SHA256-hashed, then their numeric schemas/order/counts are checked. Model matrices contain only380 statistical features; labels/provenance cannot become features. Existing frozen splits are never recreated.

The historical manifests attest row-membership uniqueness, embargo membership and exact cross-split duplicate checks. Capture/session boundaries, near duplicates, clock comparability and binary ground truth remain unverified. A1 binds those limitations and does not claim NO_LEAKAGE. Frozen producer source proves only the pre-existing folder-derived0/1 proxy policy; it is not executed. Missing/mismatched evidence blocks rather than guessing label semantics.

Test is never opened, hashed, read_table-loaded or predicted in selection. Its path/stat and declared manifest SHA are bound; that SHA is explicitly not reverified. Current physical Test content therefore cannot be certified by this selection-only task. Parent split attestations are not a new raw-flow audit.

## Durable stage/k checkpoints

Exact remote root: `MyDrive/experiment_results/phuong_anA/mi_rfe_svm/<CONFIG>/<RUN_ID>`. Local root appends the same namespace below `/content/mi_rfe_svm_5h`.

`run_manifest.json` binds algorithm, dataset/splits/fingerprint, config, run ID, source hashes/Git commit, environment/defaults, RF/SVM/scaler/MI contract and resource budget. Runtime session duration/margin is operational and recorded separately per attempt; it is configurable and is not inferred from the5h pathname.

Every stage, each completed RFE elimination and each completed k writes an immutable delta directory under `checkpoints/`. Files are copied into `.incomplete_*`, checked by size/SHA256, manifested with parent/sequence/identity/feature-order/scaler binding, marked COMMITTED, then promoted. `.incomplete` directories are preserved and ignored. The newest committed record is verified once after upload; previous immutable records are not re-read after every k. Full chains are checked on resume/verify. No raw arrays/Parquet/per-k fitted SVC models are uploaded.

Cache matrices and their checksummed manifests stay local. Root `cache_manifest.json` describes their identities and notes they must be rebuilt after runtime loss. Mutable root progress/heartbeat files are operational snapshots, not completion markers. Delta logs and immutable telemetry snapshots are included in checkpoints. Their local-completed and durable-completed k counts are distinct after an upload failure.

Resume requires exact identity. It restores verified deltas without replacing a different local artifact; rejects other algorithms/configs/splits/parameters; checks local MI/scaler/RFE/k chains; and re-commits a verified newer local prefix before starting another fit. Missing local caches are rebuilt once per split using the saved scaler; MI/RFE are not refit when their phase checkpoints are valid. An in-flight SVC must restart at its unfinished k. No k from another run is merged.

One local OS lock prevents duplicate workers in the same VM. Cross-runtime leases require explicit acknowledgement of the exact previous lease after that worker really stopped. Drive/FUSE is not a distributed transaction service: use one active runtime per run, and do not confirm takeover while the old worker is active. FUSE fsync/readback is best effort, not a claim about internal cloud replication.

## Heartbeat, budgets and completion

The resource sampler primes process CPU once and samples once per timed poll, avoiding consecutive measurements that falsely report0. The initial CPU field is null until a real interval exists. Progress is completed k/100; no invented percentage during MI/RFE. Fit elapsed time stops before prediction/checkpoint I/O. Process RSS, available RAM, measured CPU, elapsed session time, best-k-so-far and verified checkpoint status are reported.

At a configured deadline margin/user stop, the next safe boundary commits completed work and returns PAUSED. Long fits can exceed the deadline; there is no mid-fit SVC checkpoint or hard five-hour cutoff. Resource limits are sampled/boundary guards and cannot prevent every OS OOM. No automatic sampling, k skipping, cache/hyperparameter reduction or new experiment is permitted.

Selection completion requires all100 k, complete metric/count checks, frozen selection and a committed selection checkpoint. Telemetry is stopped before final publication. Result files are copied and verified; `results/SELECTION_COMPLETE.json` is last. It means Validation selection complete with Test0, not final model training. Completed output is protected; repeat verification/publication cannot replace a different file. A native exit0 alone is not scientific validation.
