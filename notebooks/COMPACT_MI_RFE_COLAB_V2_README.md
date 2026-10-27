# COMPACT-MI-RFE CICFLOW — Notebook Colab V2

## Files for this rewrite

| File | Purpose |
|---|---|
| `notebooks/COMPACT_MI_RFE_CICFLOW_Colab_V2.ipynb` | New 15-cell notebook; empty outputs, dry-run default |
| `code_CICFLOW/colab/colab_pipeline_v2.py` | Source import, test loading, preflight, fresh workers, logging, resume checks, verification |
| `code_CICFLOW/colab/test_colab_pipeline_v2.py` | Synthetic orchestration and interrupted-RFE regression tests |
| `notebooks/COMPACT_MI_RFE_COLAB_V2_README.md` | This guide and implementation notes |

Existing notebooks, `colab_pipeline.py`, Stage 4/5 scientific source, datasets,
completed runs and checkpoints are preserved. V2 does not use the old
orchestrator's `run_stage4_mi`/`run_stage5_rfe` functions: their `dry_run`
branches can fall through to a fit, and their resume flow does not safely skip
completed stages. V2 calls the existing scientific APIs only in guarded
`full`/`resume` workers.

These files were prepared by reading local source and writing code. No shell,
Python, test, preflight, MI/RFE, package installation or Git command was run as
part of this rewrite. All commands below are for the user to execute. No test
PASS or scientific compatibility is claimed.

## Direct mapping to the Colab Pro reference

Reference actually read:
`D:\COMPACT-MI-RFE\notebooks\COMPACT_MI_RFE_ColabPro.ipynb`, including the
source of its runtime/setup, dataset recovery, experiment forms, isolated
preflight/launcher, verification, result summary and optional probe cells.
The launcher `colab/experiment_launcher.py` and result-summary implementation
were also read to retain the actual behavior behind those cells.

V2 now follows the reference's code presentation: named `# CELL` blocks,
numbered prerequisite/payload/launch/result sections, explicit readiness
variables, validated experiment specs, native exit-code checks and saved
result verification. It retains the requested 15-cell CICFLOW layout by
separating the new MI and RFE stages.

| Reference cell ID / purpose | CICFLOW V2 adaptation |
|---|---|
| `stage6-cell-1` runtime | Cell 1 prints OS/architecture, logical/physical CPUs, actual versions, RAM/disk and CPU execution scope |
| `stage6-cell-2`, `stage6-cell-3` mounted sources | Cell 2 mounts Drive and validates the supplied source/dataset paths |
| `stage6-cell-4`, `stage71-cell-4a` source validation/cache | Cell 3 imports the existing Drive source explicitly, validates `__file__` and freezes SHA256 |
| `stage6-cell-3`, `stage6-cell-6` dataset inventory | Cell 4 checks the prepared CICFLOW Train Parquet directories |
| `stage6-cell-5` imports/tests/evidence | Cell 5 shows dependencies, actual test files/counts and `UNIT_TEST_EVIDENCE` bound to source |
| `stage6-cell-7` isolated dataset audit | Cell 6 launches the metadata worker and sets `DATASET_READY` from all returned results |
| `stage73a-review` scientific readiness | Cell 7 shows provisional scientific status and actual estimator contracts |
| `stage6-cell-8` experiment form + RunSpec | Cell 8 exposes EXPERIMENT/FW/FS forms, single/all selection and validated `RunSpec` objects |
| `stage6-cell-9` fresh preflight | Cell 9 prints each check, resource policy, exit code and report path |
| `stage6-cell-10` guarded isolated run | Cells 10/12 validate readiness, current selection, source and confirmation before MI/RFE workers |
| `stage6-cell-11` artifact publication verify | Cells 11/13 verify the saved MI/RFE outputs, hashes, identities and seals |
| `stage6-cell-12` result-only summary | Cell 14 reads saved worker report JSON and displays an optional pandas table |
| `stage6-cell-11`, `stage72-backup` recovery/output paths | Cell 15 writes a small-artifact inventory and prints persisted resume paths |

The notebook cell metadata also stores `reference_notebook` and
`reference_cells`, making this mapping visible in the notebook JSON.

The reference's C+ CSV transfer, old checkout paths, SVM timing probes and
final classifier/Test workflow belong to its original experiment. CICFLOW
continues to use the existing Train-only Stage 4/5 scientific functions and
the confirmed Drive layout. This is an adaptation of its notebook/launcher
structure, not a claim of identical MI scores or official reproduction.

## Actual Drive paths

```text
/content/drive/MyDrive/
├── code_CICFLOW/
│   ├── colab/
│   │   ├── colab_pipeline.py
│   │   ├── colab_pipeline_v2.py             # upload the new file
│   │   ├── test_colab_pipeline_v2.py        # upload the new test
│   │   └── tests/test_colab_pipeline.py
│   ├── stage4_mi/
│   │   ├── mi_selection.py
│   │   └── tests/test_mi_selection.py
│   ├── stage5_rfe/
│   │   ├── rfe_selection.py
│   │   └── tests/test_rfe_selection.py
│   └── logs/ALL_EXECUTION_RESULTS.log       # appended at runtime
├── CICFLOWMETER_FEATURE_CSV/
│   ├── FW256_FS8/<one supported Train Parquet>
│   ├── FW256_FS16/<one supported Train Parquet>
│   ├── FW512_FS8/<one supported Train Parquet>
│   └── FW512_FS16/<one supported Train Parquet>
└── COMPACT_MI_RFE_FRESH_RUNS/
    ├── RUN_MI_20261027T154149_761986Z/       # existing, preserved; not adopted by V2
    └── RUN_MI_<UTC timestamp>_<unique suffix>/
        ├── RUN_INFO.json
        ├── SOURCE_BINDING.json
        ├── EXECUTION_BINDING.json          # after full/resume preflight passes
        ├── stage4_mi/runs/<CONFIG>/<CONFIG_RUN_ID>/
        ├── stage5_rfe/runs/<CONFIG>/<CONFIG_RUN_ID>/<rf|svm>/
        ├── task_records/<CONFIG>/<mi|rf|svm>/
        ├── logs/
        └── reports/
```

The supported Train names are `train_windows.parquet`, `train.parquet`, and
`train_features.parquet`. Cell 4 requires exactly one of these in each config
to avoid silently choosing between two datasets. Keep actual schema/integrity
JSON sidecars with Train when available. Optional external JSON evidence paths
can be listed in `INPUT_EVIDENCE_FILES`. No Validation/Test Parquet is needed
for this notebook. Do not upload raw CSV or copy Parquet to the source tree.

The Drive file existence logs do **not** prove that source bytes match the
Windows workspace. Cell 3 imports the exact Drive file, records all relevant
Python/config file hashes, and freezes that snapshot. The previously reported
Drive digest `4bcdb79c...` is not treated as a verified hash for this V2 source.

## Run the notebook from the beginning

1. Upload the two new Python files to `MyDrive/code_CICFLOW/colab/`. Preserve
   the existing scientific modules and upload their tests if absent. Open
   `COMPACT_MI_RFE_CICFLOW_Colab_V2.ipynb` in Colab using Upload notebook.
2. Use a CPU runtime. High RAM can help, but the measured resource gates decide
   whether fitting may start. Python 3.13 is allowed; Cell 5 checks actual
   imports and versions in fresh processes. No old dependency pins are applied
   automatically. GPU acceleration is not assumed for sklearn.
3. Leave `MODE="dry_run"`, both confirmation booleans False, and
   `RUN_UNIT_TESTS=True` in Cell 1. Run Cells 1–3. Cell 2 mounts Drive if needed;
   Cell 3 resolves import conflicts, creates the experiment once, saves its
   Run IDs, and prints `RESULT_ROOT`.
4. Run Cells 4–9. Review the actual dependency report, test counts/status,
   metadata counts/schema/evidence, scientific status and preflight reasons.
   Missing tests, skipped tests and import errors block full/resume. Every
   report and stdout/stderr is saved on Drive as it is produced.
5. Cells 10–15 can be run in dry-run: MI/RFE fitting is skipped. Marker summaries
   say `NOT_STARTED`; this is not an execution success claim.

### Cell responsibilities

| Cell | Purpose |
|---:|---|
| 1 | Runtime, fixed Drive paths, mode, confirmation defaults |
| 2 | Mount and validate paths/files |
| 3 | Exact source import, source SHA256, initialize/read persisted experiment |
| 4 | Check all four directories and unambiguous Train file names |
| 5 | Fresh dependency probe and actual-file unit-test loader |
| 6 | Train footer/schema/row count and JSON evidence audit |
| 7 | Scientific readiness and actual RF estimator defaults |
| 8 | EXPERIMENT/FW/FS forms; single/all configuration selection; validated RunSpec |
| 9 | Resource, identity, tests, write access, output and confirmation gates |
| 10 | Fresh/resumed MI, one worker per configuration |
| 11 | Verify the new MI, including ranking and artifact hashes |
| 12 | Fresh/resumed independent RF-RFE branches bound to the verified new MI |
| 13 | Verify RFE and mapping to MI |
| 14 | Four-config marker and verification summary |
| 15 | Inventory, exact result/log/checkpoint paths and resume instructions |

## Start fresh MI/RFE

After reviewing the dry-run reports, use the same newly initialized V2
experiment in that session. Change the mode variable to `full` and explicitly
set the provisional confirmation in the notebook. Keep the reference flag
False until independent source reconciliation evidence exists:

```python
MODE = "full"
CONFIRM_PROVISIONAL_RESEARCH = True
REFERENCE_RECONCILIATION_CONFIRMED = False
CONFIRMATION = "RUN PROVISIONAL <exact experiment_id printed in Cell 3>"
```

Use the literal experiment ID printed by Cell 3 in `CONFIRMATION`. It is not
filled automatically. If editing Cell 1 resets evidence variables, rerun
Cells 2–9 before fitting. Cell 3 reuses this session's `ACTIVE_RESULT_ROOT`
instead of generating another ID. Starting in a new session with `full` and
no session root creates a completely fresh experiment.

Cell 8 follows the reference notebook's experiment forms and defaults to all
four configs and both branches. For one configuration and the RF branch, set:

```python
EXPERIMENT = "mi_rfe_rf"
CONFIG_SCOPE = "single"
FW = "256"
FS = 8
```

Use `EXPERIMENT="mi_rfe_svm"` for only the SVM branch's RF-RFE estimator,
or `EXPERIMENT="both"` for both branches. Use `CONFIG_SCOPE="all"` for all
four configurations. Run IDs are read from the stored experiment.

Run Cells 8–9 again after changing selection. Cell 8 invalidates the old
preflight, and the launch cells reject a selection/mode/source changed after
the successful preflight. Then run Cell 10 (MI), Cell 11
(MI verify), Cell 12 (RFE), and Cells 13–15 (verify/summary/inventory). A
configuration with failed MI verification is not passed to RFE. Other configs'
logs and completed artifacts are retained even if a worker fails.

Running `full` again against an existing task directory is blocked. Use resume
for unfinished tasks in the same experiment; use a new session/experiment for
a separate scientific rerun. No old Windows or pre-V2 MI result is imported.

## Scientific contract retained from actual source

- MI: Train only, 380 columns, current numeric/non-finite preprocessing and
  float32; no external StandardScaler; `mutual_info_classif` per feature with
  `discrete_features=False`, `n_neighbors=3`, seed `42 + original_index`.
- MI batch size stays the source default 20. This groups checkpoints, not a
  shared 380-column MI call. Scores sort descending, ties use original index.
- Both `rf` and `svm` RFE branches use `RandomForestClassifier`. The `svm`
  label is a downstream branch name, **not** SVM-RBF feature elimination.
- RF depth 10, SVM-branch RF depth 15; `n_estimators=100`, balanced classes,
  seed 42, and source `n_jobs=-1` are retained. RFE target 1, step 1, importance
  squaring and `np.argsort` behavior are retained. Unspecified estimator
  defaults are resolved and recorded at runtime, not silently changed.
- XGBoost-RFE and Stage 5B selection/refit/Test evaluation are outside this
  notebook. Test/Validation feature values are not read by MI/RFE.

Reference equivalence and capture/session independence remain unproven.
The metadata fingerprint binds paths, file size/mtime, footer and JSON hashes;
it is not a SHA256 of all Train bytes. Train label/value validity is checked
by existing preprocessing at load time; the metadata audit makes no claim to
have validated unseen rows. Optional sidecars are recorded; explicit technical
FAIL/BLOCKED evidence blocks continuation. No scientific leakage PASS is
fabricated when split evidence is incomplete.

## Resume after a disconnected runtime

In Cell 1 select:

```python
MODE = "resume"
EXISTING_RESULT_ROOT = "/content/drive/MyDrive/COMPACT_MI_RFE_FRESH_RUNS/<YOUR_V2_EXPERIMENT>"
CONFIRM_PROVISIONAL_RESEARCH = True
CONFIRMATION = "RUN PROVISIONAL <YOUR_V2_EXPERIMENT>"
```

Run Cells 2–9 to reload the stored config Run IDs and rerun source, library,
metadata and unit-test gates. Run Cells 10–15. MI that is complete within the
same experiment is hash/identity-verified and skipped; an unstarted task can
start within the same experiment. MI resumes from persisted feature batches;
RFE resumes from a structurally valid elimination checkpoint. Source,
Python/library versions, dataset identity, parameters, MI order and git
identifier must match the execution binding exactly.

Interrupted work since the last committed checkpoint may be recomputed.
Previously verified checkpoint steps are not deliberately replayed. `.tmp` or
`.writing` files, missing checkpoints, inconsistent state, and interrupted
final-artifact publication return `UNSUPPORTED` because the current scientific
source cannot reconcile them reliably. Do not delete such files or overwrite
the run to force continuation. The original source's Drive/FUSE filesystem
durability still needs runtime verification; V2 does not promise recovery from
all possible storage failures.

The existing `RUN_MI_20261027T154149_761986Z` is not automatically adopted: it
has no confirmed V2 binding. It remains preserved without claiming MI/RFE
completion. To rerun fresh with V2, leave `EXISTING_RESULT_ROOT` empty in full.

## Verify without fitting

Set `MODE="verify"` and the same `EXISTING_RESULT_ROOT`. Run Cells 1–9 and
11, 13–15; Cells 10/12 skip fitting in verify mode. Cell 11 validates Stage 4
content/identity and the V2 seal; Cell 13 validates the 100-to-1 RFE ranking,
mapping, branch identity and seal. A zero worker exit code without structured
evidence is not PASS. Merely finding a COMPLETE marker is not PASS.

## Results and logs to send back

Inside the new `RESULT_ROOT`, send `RUN_INFO.json`, `SOURCE_BINDING.json`,
`EXECUTION_BINDING.json`, relevant `reports/*.json`, and the per-task `logs/`.
For failures include the worker's stderr, exit code, source checkpoint,
BLOCKED/FAILED markers and saved task record. No Train Parquet is required.

Every worker tees stdout/stderr live into its own experiment log and appends
to `MyDrive/code_CICFLOW/logs/ALL_EXECUTION_RESULTS.log`. Start/end/runtime,
child command, exit code, program status, sampled process-tree RAM, artifact
paths and scientific status are separate fields. Output is saved as it arrives;
a killed runtime may leave START without END. That is not completion. Secret
redaction is best-effort; no tokens/passwords are required by these commands.
Logging failure blocks launch or stops the task with an error instead of
claiming results were saved. There is no automatic retry or resource-based
reduction of data/trees/features/step.

## User-run Windows checks with logging

These run synthetic tests/static notebook checks only. They do not fit the
CICFLOW dataset. Copy code only, without a `PS ...>` prompt. Use your existing
activated Python environment.

```powershell
Set-Location -LiteralPath 'D:\COMPACT-MI-RFE\code_CICFLOW'
$env:PYTHONPATH = 'D:\COMPACT-MI-RFE\code_CICFLOW'
$wrapper = 'D:\COMPACT-MI-RFE\code_CICFLOW\shared\Invoke-CicflowLogged.ps1'
$testCommand = @'
python -u -c "from pathlib import Path; import json; from colab.colab_pipeline_v2 import run_test_files, save_report; result=run_test_files(Path.cwd()); print(json.dumps(result, indent=2)); print(save_report(Path('colab/v2_local_checks'), 'unit_tests', result)); raise SystemExit(0 if result['status']=='PASS' else 2)"
'@
& $wrapper -Stage 'Colab V2' -Task 'Actual-file unit tests' -Experiment 'CICFLOW' -Configuration 'SYNTHETIC' -RunId 'V2_TESTS' -WorkingDirectory 'D:\COMPACT-MI-RFE\code_CICFLOW' -Command $testCommand -OutputDirectory 'D:\COMPACT-MI-RFE\code_CICFLOW\colab\v2_local_checks' -ScientificStatus NOT_ASSESSED
```

Validate notebook JSON and Python syntax without executing its cells:

```powershell
$syntaxCommand = @'
python -u -c "import json, ast; from pathlib import Path; path=Path('D:/COMPACT-MI-RFE/notebooks/COMPACT_MI_RFE_CICFLOW_Colab_V2.ipynb'); nb=json.loads(path.read_text(encoding='utf-8')); cells=[c for c in nb['cells'] if c['cell_type']=='code']; assert len(cells)==15; [ast.parse(''.join(c['source'])) for c in cells]; assert all(c['execution_count'] is None and c['outputs']==[] for c in cells); print('Notebook JSON and 15 code cell syntax checks PASS; no cell executed')"
'@
& $wrapper -Stage 'Colab V2' -Task 'Notebook static syntax' -Experiment 'CICFLOW' -Configuration 'NONE' -RunId 'V2_SYNTAX' -WorkingDirectory 'D:\COMPACT-MI-RFE\code_CICFLOW' -Command $syntaxCommand -OutputDirectory 'D:\COMPACT-MI-RFE\notebooks' -ScientificStatus NOT_ASSESSED
```

Print a comparable Windows source snapshot, then compare its `sha256` and
file list to Cell 3's saved `SOURCE_BINDING.json`:

```powershell
$hashCommand = @'
python -u -c "from pathlib import Path; import json; from colab.colab_pipeline_v2 import source_snapshot; print(json.dumps(source_snapshot(Path.cwd()), indent=2))"
'@
& $wrapper -Stage 'Colab V2' -Task 'Source snapshot' -Experiment 'CICFLOW' -Configuration 'NONE' -RunId 'V2_SOURCE_HASH' -WorkingDirectory 'D:\COMPACT-MI-RFE\code_CICFLOW' -Command $hashCommand -OutputDirectory 'D:\COMPACT-MI-RFE\code_CICFLOW\colab' -ScientificStatus NOT_ASSESSED
```

## Common errors

- **Wrong imported module:** upload `colab_pipeline_v2.py`, rerun Cell 3 and
  check both actual `__file__` paths. No fallback to `/content/RFforNetwork`
  is allowed. Existing `colab` namespaces are rebuilt from SOURCE_ROOT;
  `google.colab` is unaffected.
- **NO_TESTS_FOUND/SKIPPED:** upload actual `test_*.py` files into their source
  areas; no `tests/__init__.py` is required by the V2 loader. Set
  `RUN_UNIT_TESTS=True`. Read the file list/import traceback in the saved report.
- **Python 3.13 dependency failure:** inspect fresh-worker import errors and
  installed versions. Decide a compatible runtime/package set manually, then
  start a new experiment after changing versions. V2 installs nothing.
- **Source changed:** do not resume the locked run with different code. Compare
  hashes/upload the exact bound source or start a new experiment.
- **Metadata BLOCKED:** inspect schema/order/Train count/non-numeric columns and
  JSON evidence, without editing dataset or reducing its rows to obtain PASS.
- **RAM/disk BLOCKED:** request a runtime with enough measured resources. No
  sampling, tree reduction, parameter change or automatic retry is performed.
- **Output exists in full:** use resume/verify for that V2 experiment, or a new
  experiment for a fresh fit. Never remove old output to reuse its identity.
- **UNSUPPORTED resume:** preserve the partial files and send their diagnostics;
  the source's final-publication behavior does not support universal recovery.

## Verification still needed

The Drive versions, real footer/evidence contents, Python 3.13 imports,
existing synthetic tests, reference score parity, resource estimate adequacy
and actual Drive checkpoint behavior must be verified by user-run results.
The README and new notebook are an implementation handoff, not proof that
training has succeeded or that source is reference-equivalent.
