# CICFLOW ColabPro layout rewrite — 2026-10-28

## Outcome

The existing `COMPACT_MI_RFE_CICFLOW_Colab_V2.ipynb` has been rewritten to the
reference notebook's twelve main cell purposes. It now launches one complete
RF or compact-SVM experiment per selected FW/FS configuration, including
Validation k-search and final evaluation. It publishes to
`/content/drive/MyDrive/experiment_results/<experiment>_<config>_<runNN>/`.

The completed directory contains all thirteen artifact names in the user's
screenshot, plus explicit identity, parameter, resource and completion metadata.
Incomplete state and per-elimination/per-k checkpoints remain under
`/content/drive/MyDrive/checkpoints/<run_name>.incomplete/`.

## Sources read

- `notebooks/COMPACT_MI_RFE_ColabPro.ipynb`: main cells 1–12; optional legacy
  embedded-patch/probe cells are not copied into the new notebook.
- `huyen_code/ciciomt24_training.py`: parameter dictionaries; prepare_xy;
  select_mi; scale_top100; rank_rfe; search_k; fit_final_classifier;
  evaluate_final; metrics; run_official event/publication order.
- `colab/checkpoint_manager.py`: reference completed artifact set and structure.
- `code_CICFLOW/stage4_mi/mi_selection.py`: actual CICFLOW schema helpers and
  old per-feature MI implementation, which remains untouched.
- `code_CICFLOW/stage5_rfe/rfe_selection.py`: independent RF-RFE branch
  contracts and manual step-one elimination implementation.
- `code_CICFLOW/shared/Invoke-CicflowLogged.ps1`: actual wrapper parameters.

## Scientific change and scope

The new experiment computes MI with Train StandardScaler and ONE
`mutual_info_classif(scaled, y, random_state=42)` call on all 380 features,
then uses pandas descending sorting as in the reference. Installed defaults
must be discrete_features=auto and n_neighbors=3; otherwise admission is
BLOCKED. There are no feature-specific seeds or MI feature batches.

This differs from the former Stage 4 per-feature seed42+index/no-external-
scaler variant. Scores and rankings may change. No old manifest, completed
run, or scientific parameter is rewritten to conceal this change. Old MI/RFE
artifacts are not adopted into a new experiment. Re-running the user-selected
experiment is intentional and must be explicitly triggered by the user.

Top100 scaling fits Train only. Independent RF-RFE branches use the source
100-tree depth10 RF estimator and 100-tree depth15 SVM-branch RF estimator.
The existing step-one elimination loop is reused on the scaled Train matrix,
including a final estimator fit on the remaining feature, matching the
target sklearn RFE execution structure. RFE checkpoints retain ordered Top100,
branch, source and MI bindings. Journal checks validate every support/rank
transition and checkpoint hash chain.

The RF CLASSIFIER uses 300 trees, depth20, min_samples_split3, sqrt, balanced,
seed42, n_jobs=-1. The compact-SVM classifier uses source C10/max_iter20000,
RBF/gamma=scale/probabilityFalse/cache10000MB/shrinkingTrue/balanced/seed42.
Resolved estimator defaults and MI signature are recorded with library versions.
XGBoost remains BLOCKED because its ranking contract has not been reconciled;
its confirmed classifier parameters are preserved in the shared contract.

K-search always covers 1..100. The first/smallest k wins an exact Validation
Accuracy tie. The selected feature set is frozen before final scaler/model
refit on Train+Validation. A durable Test intent is written only after final
model/scaler hashes and frozen binding validate. Test is opened and evaluated
once. An interrupted Test with no saved result cannot be repeated by resume.

## Files created/modified

Created:

- `code_CICFLOW/colab/cicflow_reference_pipeline.py`
- `code_CICFLOW/colab/test_cicflow_reference_pipeline.py`
- `notebooks/CICFLOW_COLABPRO_REWRITE_REPORT.md`

Modified:

- `notebooks/COMPACT_MI_RFE_CICFLOW_Colab_V2.ipynb`
- `notebooks/COMPACT_MI_RFE_COLAB_V2_README.md`
- `notebooks/HUONG_DAN_CHAY_CICFLOW_COLAB_V2.md`
- `code_CICFLOW/shared/classifier_contract.json` (implementation availability)
- `code_CICFLOW/stage5_rfe/reports/STAGE5B_CONTRACT.md` (implementation note)

Stage 0–4 source and outputs, the Stage 5A scientific source, five original
CTU-13 scripts, the ColabPro reference notebook, raw datasets and all old
completed runs/checkpoints have not been edited, removed, moved or relabelled.

## Execution and verification evidence

Codex performed direct source reads, text edits and notebook JSON inspection.
After the direct filesystem tool failed, the user explicitly authorized
read-only terminal file access. No Python, notebook cell, unit/regression test,
training, benchmark, Git or pip command was executed by Codex.

Synthetic tests were WRITTEN for one-call 380-column MI score/ranking parity,
preprocessing/order, classifier/RFE parameter separation, sklearn RFE ties and
resumed elimination, Train-only reads, MI/checkpoint identity/hash binding,
k-search continuation/ties, non-overwrite, resource failure, durable one-Test
guard, logged stdout/stderr/native exit/append and locked-log handling. A
synthetic lifecycle test exercises completed publication, read-only verify,
resume-without-Test and tamper detection. Fixture estimators/counts are small
and injected only inside tests; production parameters are not changed.

These tests have NOT been run by Codex. There is no new PASS claim. The user
must run Cell 5 and inspect its saved result/log before full admission. Even a
fixture PASS is not proof that actual CICFLOW scores equal another dataset,
another library version, or the CTU-13 paper's reported scores.

## Limits and recovery

All results remain PROVISIONAL_RESEARCH and official_reproduction=false.
Label/capture/session provenance and independent split validation are still
unresolved. Dataset identity covers path/stat/footer/schema and sidecar hashes;
it is explicitly not a full Parquet content digest. Library/source/run/git
identity must match exactly for resume. RAM estimates are admission heuristics,
not time forecasts or a guarantee against peak usage/OOM.

MI is checkpointed only after its one complete call; interrupted MI may need
to restart. RFE/k-search resume from completed journal entries, not within a
fit. Unsafe partial artifacts or interrupted final sealing are BLOCKED rather
than deleted/overwritten. Sealed checkpoints may be published without refitting
or Test evaluation. OS writer locks and staging-copy hash checks are provided;
mounted Google Drive FUSE atomicity cannot be guaranteed absolutely.

Adding source files changes legacy V2 aggregate source snapshots. Resuming an
old experiment still requires its exact original source snapshot; no identity
check is bypassed. The new notebook does not launch or migrate those old runs.

See `HUONG_DAN_CHAY_CICFLOW_COLAB_V2.md` for exact upload paths, full/resume/
verify instructions, four FW/FS selections and the logged Windows test command.
