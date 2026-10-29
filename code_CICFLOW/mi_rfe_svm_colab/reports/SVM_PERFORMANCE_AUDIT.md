# SVM performance audit — source inspection only

No Python, tests, preflight, training, Git or benchmark was executed. Line numbers below refer to the unchanged sources inspected before implementation. Runtime improvements are targets, not measurements.

| Question | Source evidence | Finding / action |
|---|---|---|
| Duplicate fit / predict at a k? | `huyen_code/ciciomt24_training.py:436–457`; `code_CICFLOW/colab/cicflow_reference_pipeline.py:694–725` | One fit and one Validation predict per k already. Do not claim removal of duplicate fits. |
| MI / RFE inside k loop? | reference `select_features:460–478`; CICFLOW `execute_run:1029–1037` | Already outside. Reuse their completed artifacts; no algorithm change. |
| Unnecessary scaling? | `colab/svm_resumable_training.py:391–413` | The legacy path computes scaled Train/Val, discards them, then transforms again for its memmap cache. The new adapter keeps the first transforms and persists a bound local cache. |
| Read Parquet each k? | CICFLOW `execute_run:1027–1037`; legacy `prepared_split:156–188` | Already loads outside the loop. New local checksum cache also avoids repeated reads in the same runtime. |
| Repeated subset construction? | reference `search_k:443–446`; CICFLOW `search_k:703–709` | Advanced indexing creates subsets at each k, followed by libsvm conversion to C-order float64. Reorder the already scaled float32 Top100 once, then explicitly create only the current C-order float64 prefix. |
| Model retained? | CICFLOW `search_k:711`; reference `search_k:436–457` | CICFLOW already deletes the fitted model. New code releases model, prediction and subsets before the next k. No per-k model serialization. |
| Drive writes each k? | legacy resumable `:439–461` calls `CheckpointStore.save` at459 and durable progress at461; `svm_resumable_store.py:save`; `DurableProgress:38–127` | Legacy persists many per-stage files and heartbeat/event updates at its durable root. New adapter uses local progress only; versioned Drive bundle at >=18000 seconds, after a completed fit/predict boundary. |
| RFE tie semantics? | CICFLOW `saved_rfe:648–690`; Stage5 `manual_rfe_step1:335–386` | Reuse squared importances, NumPy default argsort and the additional final one-feature forest fit. Separate SVM estimator, depth15. |
| Numeric compatibility? | reference `prepare_xy:163–184`, `scale_top100:373–378`, `fit_final_classifier:481–503` | Keep float64 cleaning -> float32; same indexing/layout before StandardScaler. Float64 conversion happens only after float32 scaling, at the SVC boundary. No manual gamma, copy=False scaler, sparse kernel or Gram matrix. Exact full-run equivalence remains untested. |
| Existing reuse? | CICFLOW `saved_mi:618–645`, `saved_rfe:648–690`; legacy store checksums/parents | Reuse new committed bundles and compatible CICFLOW reference-layout SVM MI/RFE artifacts, after checking identity, environment, source hashes, ordered names and checksums. Per-feature Stage4 MI is a different scientific variant and is rejected. Legacy CSV resumable bundles are reference/design material, not automatically imported as CICFLOW Parquet artifacts. |

Also inspected: `notebooks/COMPACT_MI_RFE_ColabPro.ipynb` (mount, exact source, dependencies, dataset validation, explicit run, publication and JSON summary purposes); `huyen_code/ciciomt24_training_memorylab.py:1145–1206` (isolated scaler/workspace experiments); SVM technical/scaling probes; legacy resumable execution/supervisor modules. Their sampling/probe policies are not used for scientific training.

The original SVM sources were additionally read: `CODE_ctu13/7_senario1_svm.py:100–115` and `CODE_ctu13/[[CTU13]8_experiment2_svm.py:114–162,180–213,234–247`. They support StandardScaler, a single matrix MI call, RF depth15 RFE target1/step1, one SVC fit/predict per k and strict-greater Accuracy selection. The original eagerly transforms Test; the CICFLOW adapter follows the user's stricter contract and defers all Test values until a durable claim after final refit. No original script was edited.

## Limits

RBF SVC remains CPU based and can dominate runtime. Full k=1..100 entails 100 independent SVC fits plus one final refit. A five-hour checkpoint is not an SVC mid-fit checkpoint: runtime loss can discard the current fit and all k results since the last committed Drive bundle. No measured speedup, memory ceiling guarantee, convergence guarantee or official reproduction is claimed. Folder/proxy labels and capture/session independence remain unresolved: `PROVISIONAL_RESEARCH`, `OFFICIAL_BLOCKED`.
