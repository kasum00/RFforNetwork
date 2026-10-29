# Five-hour checkpoint and resume design

## Boundaries and storage

`DRIVE_CHECKPOINT_INTERVAL_SECONDS=18000`, `LOCAL_PROGRESS_AFTER_EACH_K=True`, `DRIVE_CHECKPOINT_AFTER_EACH_K=False`.

There is no background Drive timer. The worker samples resources locally, completes one SVC fit and one Validation prediction, releases their temporary objects, and writes a local immutable k record. Only then does it check a monotonic timer. Before the interval is due, this check returns before any Drive stat/list/open. Once due, the worker snapshots state into a local ZIP, streams that ZIP to Drive, verifies byte size/SHA256, and publishes a new COMMITTED marker. The timer resets only after verified success.

If a fit takes 21000 seconds, no upload occurs at second18000. The next safe boundary records the overrun. This does not serialize an in-flight SVC or promise mid-fit recovery.

Additional explicit boundaries: completed MI; completed/reused RFE and its Train scaler; frozen selection; Cell13's completed final-refit save; final Test/result completion; requested checkpoint/stop; controlled exception. A model already present in a verified final-refit bundle is not uploaded again merely to begin Test. Root identity and session leases are written at session admission, not after every k. The durable Test claim is a necessary once-only exception to the interval.

## Layout

Local, isolated per run:

```text
/content/mi_rfe_svm_5h/mi_rfe_svm_FW256_FS8_run01/
  RUN.lock                    OS lock, released on process exit
  state/                      scientific JSON/CSV, k records, RFE steps, logs, completed models
  cache/                      raw/scaled/ranked npy arrays; NEVER included in Drive bundle
  bundles/                    staged ZIP files
  sync_receipt.json           latest verified remote ancestor
  console.log                 append-only local console stream
  CHECKPOINT_REQUESTED        optional local control flag
  STOP_REQUESTED              optional local control flag
```

Drive:

```text
MyDrive/experiment_results/mi_rfe_svm_FW256_FS8_run01/
  OWNER.json
  run_identity.json
  leases/                     immutable admission/release records, no heartbeat
  checkpoints/
    CP_000001_<uuid>.zip
    CP_000001_<uuid>.COMMITTED.json
    CP_000002_<uuid>.zip
    CP_000002_<uuid>.COMMITTED.json
  # Flat files below appear only at final publication:
  mi_feature_scores.csv, mi_top100.json, mi_state.json
  rfe_feature_ranking.csv, rfe_state.json
  top100_scaler.joblib, top100_scaler_state.json
  k_search_results.json, selection_frozen.json, features.json
  model.joblib, final_scaler.joblib, final_state.json
  TEST_CLAIM.json, test_result.json, result.json
  hyperparameters.json, preflight_audit.json
  events.jsonl, progress.json, resource_usage.json, console_snapshot.log
  artifact_manifest.json, RUN_COMPLETE.json
```

The other three configurations have their own folder with the same layout. Partial uploads are never committed and never overwritten or deleted automatically. All old bundles remain. FUSE fsync/readback is best effort; COMMITTED means this implementation verified its observed files, not a claim about Google's internal replication.

## Bundle identity

Bound fields: config/experiment/run name; exact Git commit and source file hashes; Python and installed library versions/defaults; Parquet file path/stat/footer/schema/order; small schema/integrity manifests and binary label histograms; all scientific parameters and preprocessing; resource budget; optional imported MI/RFE origin hashes.

Each bundle contains the identity, MI/RFE CSV/state, completed contiguous k list and metrics, frozen selection when available, log/resource snapshots and completed preprocessing/final models when available. No raw/scaled arrays, Parquet, dense Gram matrix or fitted model for each k is included. ZIP is streamed rather than held in memory. State files have their own size/SHA256 entries. Parent sequence, archive checksum, internal membership/path/size and identity are checked on restore. A corrupt newest COMMITTED bundle blocks; it does not silently roll back to an older ranking/selection.

The dataset fingerprint uses metadata/footer/stat and small sidecar hashes. It is explicitly **not a complete Parquet content hash** and cannot establish capture independence or detect a malicious content change preserving all bound metadata. Scientific status remains provisional.

## Resume

New runtime: verify source/environment/data binding, acknowledge the exact unreleased lease only after the prior runtime really stopped, restore the newest committed bundle into a new local state directory, then verify the MI/RFE and k dependency chain. Local progress in the same runtime can be newer than Drive if its identity, checksum chain and sync ancestor match. Gaps or a different ancestor block automatic merging. Never combine k records across runs or branches.

Committed k=1..61 means resume at62. A partial fit at62 is not a result and must start again. No completed k is refit. Local caches survive same-runtime resume; after a runtime reset Train/Validation are loaded again once and transformed using the saved Train Top100 scaler. MI/RFE are reused from their verified states.

If local receipt/state is inconsistent after a crash, preserve it and point LOCAL_ROOT at a fresh directory under `/content` to restore the newest committed Drive bundle. This intentionally abandons unproven newer local work; it does not delete it or merge it automatically.

## Stop, failure and Test once

Cell11 creates a local checkpoint or stop request. The worker waits until a safe boundary, verifies the Drive commit, releases its lease and reports PAUSED. A fit can take longer than the requested wait. A repeated `resume` after a deliberate stop requires `ACKNOWLEDGE_LOCAL_STOP=True`; this consumes only that owned local control flag.

RAM is sampled every0.5s locally. Limits block the next fit at a safe boundary; they cannot guarantee the OS will not kill a fit first. An OOM, disconnect or hard kill can lose all uncommitted k since the previous bundle. No cache-size/data/k reduction or automatic new experiment is permitted.

Final model/scaler and selection must be durable before Test. An exclusive, verified Drive `TEST_CLAIM.json` precedes the first Test value read. A saved Test result may be republished without model loading or predicting. A claim without a verified result is BLOCKED even if it might have failed before prediction: no automatic Test retry. Final publication is idempotent for byte-identical files; different or partially copied final root files block and require review. No completed output is replaced.

Same-runtime OS locks prevent duplicate workers. Drive leases provide a visible cross-runtime claim and require explicit takeover acknowledgement; mounted Drive is not a distributed transaction service. Use one active runtime per run. Never confirm takeover while an earlier worker remains active.

Technical PASS, a phase being saved, native exit0 and scientific validity are distinct fields. A running background queue is not complete. Final RUN_COMPLETE requires all100 k, frozen selection, final refit, exactly one verified Test result, the final committed bundle and a checked artifact manifest.
