# KeyError at k=1 — fix and checkpoint-preserving handoff

No commands, Python, tests, preflight, Colab, training or Git were executed by Codex. Actual Drive/local checkpoint bytes are not available in this workspace. The reported heartbeat is evidence of failure, not proof of a committed k record.

## Exact cause and ordering

Unchanged `mi_rfe_svm_colab/runner.py:159–164` writes `k/k_001.json` locally via KProgress, selects the best row, emits `k_completed`, then invokes the Drive boundary callback. Its event contains `validation_accuracy` and `best_k`; it never emits `best_validation_accuracy`.

The faulty `phuong_anA/telemetry.py:100` (before this fix) indexed `details['best_validation_accuracy']`. The missing dictionary key was in **the event's keyword arguments**, not the Validation metrics row or a checkpoint. Assignment failed before updating best values, after completed_k was set, and before calling `boundary()`/commit_k. The worker's exception handler at `pipeline.py:175–182` can commit a failed_* checkpoint carrying logs/telemetry; it does not include the local k record. This explains local completed1, durable0, bestNone and SAVED failed_* without treating failure as successful k work. Exact files still require the read-only audit below.

The fix consumes the existing event's required measured validation_accuracy, checks sequence/finite accuracy/caller best_k, and updates best by strict greater, leaving exact ties at the earlier/smaller k. It validates before mutating completion. Missing scientific fields still fail; no get/default/fake Accuracy is used. Before any k, optional best fields remain None. Resume restores best only from the verified history. Durable progress is derived only from the verified ledger's contiguous k stages; failed_* is excluded.

## Changed files

- `phuong_anA/telemetry.py`: event contract, strict measured best update, restore_history and durable prefix display.
- `phuong_anA/pipeline.py`: use restore_history instead of assigning display state independently.
- `phuong_anA/recovery.py`: read-only checkpoint/source/MI/scaler/RFE/k-chain assessment, no Store/cache/model/dataset/lease writes or loading.
- `phuong_anA/tests/test_best_accuracy_recovery.py`: real search+Telemetry+Store regression with fake SVC; before/after commit failures, restore/skip, failed-stage exclusion, invalid source/data identities, corruption and selection-only entrypoints.
- This handoff document.

No A0 numerical functions, scientific parameters, source dataset, completed/failed checkpoints, manifests or run contents were edited. The new tests have not been run.

## Identity BLOCKED and migration plan

All these source/test files are part of the source snapshot. Therefore a run created before this fix cannot pass direct resume using the new source/commit. This is expected **BLOCKED**, not an error to remove or work around. Never edit the old manifest, assign its hash to the new code, monkeypatch the old worker, delete failed markers, or force durable k=1.

After the read-only assessment, two distinct recovery inputs may exist:

1. Verified COMMITTED `k_001` plus complete parent chain: a compatible continuation would start k2.
2. No verified COMMITTED k1: Drive-only continuation must recompute k1, reusing verified MI/scaler/RFE. A same-runtime newer local k1 can only be retained after local identity/hash/ancestry checks and a new verified commit. A failed_* record alone is never k1.

**Proposed migration, not implemented or performed here:** use a separately named new run (e.g. RUN_A1_01_FIX01), keep RUN_A1_01 entirely read-only, and first approve a migration report containing: original/new full identities and commit/source diffs; unchanged MI/preprocessing/RFE/SVC numerical file hashes and resolved parameters; equal current dataset fingerprint, split/feature order, environment and scaler/RFE/MI identities; verified old checkpoint ledger; exact selected prefix of completed Validation records. Rebind copies of MI/scaler/RFE/k records under the new identity, updating dependency hashes in order, while recording original artifact hashes and original parent checkpoint references in an immutable MIGRATION_MANIFEST. Never reuse a raw array cache across identities without rebinding/verification. Verify the new chain and metric/count/prefix equivalence before admitting another fit. If any of those checks cannot be demonstrated, migration remains BLOCKED; a fresh independent run requires an explicit user decision and must not start automatically. These steps are a plan only, not a license to overwrite the old run or treat numerical compatibility as proven.

## Colab cells

Use the fixed checkout as SOURCE_ROOT with a new exact approved commit. Keep the existing runtime/files intact until local audit is printed; do not disconnect/reset it to obtain new code. All cells below use fresh subprocesses so stale notebook imports cannot silently select the old implementation. Updating source must use a separate clean checkout; it must not modify the checkout/identity of an active old worker. These snippets assume the notebook setup already defines SOURCE_ROOT and request_for with the intended fixed checkout. No source reload changes a running worker.

### Read-only checkpoint assessment

This never instantiates Store or KProgress, creates directories or writes progress/leases. It does not access the dataset or deserialize joblib. It checks the saved identity and every committed artifact SHA, complete MI/scaler/A1 rank chain, complete metrics/counts/runtime, contiguous k hashes/parents and optional same-runtime local ancestry. Current source/dataset compatibility is checked separately by preflight, not inferred from a PASS here.

```python
import json, subprocess, sys
from pathlib import Path
CONFIG = 'FW256_FS8'
RUN_ID = 'RUN_A1_01'
DATASET_ROOT = DRIVE / 'CICFLOWMETER_FEATURE_CSV'
OUTPUT_ROOT = DRIVE / 'experiment_results'
LOCAL_ROOT = Path('/content/mi_rfe_svm_5h')
REMOTE = OUTPUT_ROOT / 'phuong_anA/mi_rfe_svm' / CONFIG / RUN_ID
LOCAL = LOCAL_ROOT / 'phuong_anA/mi_rfe_svm' / CONFIG / RUN_ID

def readonly_worker(code, payload):
    p = subprocess.run([sys.executable, '-B', '-u', '-c', code], cwd=str(SOURCE_ROOT),
                       input=json.dumps(payload), text=True, capture_output=True)
    if p.stderr: print(p.stderr)
    if p.returncode:
        print(p.stdout)
        raise RuntimeError(f'BLOCKED: read-only assessment exited {p.returncode}')
    return json.loads(p.stdout)

AUDIT = readonly_worker('''
import json, sys
from phuong_anA.recovery import inspect_run
p = json.load(sys.stdin)
print(json.dumps(inspect_run(p['remote'], local=p['local']), indent=2))
''', {'remote': str(REMOTE), 'local': str(LOCAL)})
print(json.dumps(AUDIT, indent=2))
```

Read `k1_committed_and_verified`, `next_fit_k_from_drive`, `local.pending_commit_k`, `local.next_fit_k_if_identity_matches`. No field in this report commits/authorizes a fit. If corruption/identity mismatch is raised, preserve files and report it. If the VM was lost, local status should say LOCAL_STATE_ABSENT.

### Synthetic tests (local logs only)

```python
import os, uuid
from datetime import datetime, timezone
from pathlib import Path
log_root = LOCAL_ROOT / 'phuong_anA/_unit_evidence'
log_root.mkdir(parents=True, exist_ok=True)
stamp = uuid.uuid4().hex
evidence = log_root / (stamp + '.json')
command = [sys.executable, '-B', '-m', 'phuong_anA.selftest',
           '--source', str(SOURCE_ROOT), '--evidence', str(evidence)]
started = datetime.now(timezone.utc).isoformat()
env = os.environ.copy(); env['PYTHONDONTWRITEBYTECODE'] = '1'
p = subprocess.Popen(command, cwd=str(SOURCE_ROOT), env=env, stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT, text=True, bufsize=1)
with (log_root / (stamp + '.log')).open('x') as log:
    for line in p.stdout:
        print(line, end='', flush=True); log.write(line); log.flush()
native_exit = p.wait()
(log_root / (stamp + '_native.json')).write_text(json.dumps({
    'command': command, 'start': started, 'end': datetime.now(timezone.utc).isoformat(),
    'native_exit_code': native_exit}, indent=2))
TEST_EVIDENCE = json.loads(evidence.read_text()) if evidence.exists() else {'status': 'FAIL'}
print('NATIVE EXIT:', native_exit, 'EVIDENCE:', TEST_EVIDENCE)
if native_exit or TEST_EVIDENCE['status'] != 'PASS':
    raise RuntimeError('Tests have not passed; do not resume')
```

### Fixed-source preflight against the unchanged old run

This is read-only (including Train/Validation hash/metadata checks); Test is not opened. With a changed commit/source, expect BLOCKED. It must not create a new run or migrate artifacts.

```python
REQ = request_for(CONFIG, 'resume')
REQ['run_id'] = RUN_ID
REQ['source'] = str(SOURCE_ROOT)
REQ['unit_tests'] = TEST_EVIDENCE
PLAN = readonly_worker('''
import json, sys
from phuong_anA.preflight import review
print(json.dumps(review(json.load(sys.stdin)), indent=2))
''', REQ)
saved = json.loads((REMOTE/'run_manifest.json').read_text())['identity']
print('STATUS:', PLAN['status'], 'REASONS:', PLAN['reasons'])
print('SOURCE MATCH:', saved['source'] == PLAN['identity']['source'])
print('GIT MATCH:', saved['git_commit'] == PLAN['identity']['git_commit'])
print('DATASET MATCH:', saved['dataset_fingerprint'] == PLAN['identity']['dataset_fingerprint'])
```

### Strictly guarded resume

**This will BLOCK for RUN_A1_01 using the fixed source.** It is supplied to show the required gates, not to bypass them. Use only after an explicitly implemented/verified migration or when every original identity field actually matches. Do not change RUN_ID automatically to conceal the mismatch. No training starts until all checks below pass and the human enters the exact run-specific phrase.

```python
if globals().get('JOB', {}).get('running'):
    raise RuntimeError('Worker is still active; do not launch another')
current_req = request_for(CONFIG, 'resume')
current_req.update(run_id=RUN_ID, source=str(SOURCE_ROOT), unit_tests=TEST_EVIDENCE)
PLAN = readonly_worker('''
import json, sys
from phuong_anA.preflight import review
print(json.dumps(review(json.load(sys.stdin))))
''', current_req)
if PLAN['status'] != 'PASS':
    raise RuntimeError('BLOCKED: ' + '; '.join(PLAN['reasons']))
AUDIT = readonly_worker('''
import json, sys
from phuong_anA.recovery import inspect_run
p = json.load(sys.stdin)
print(json.dumps(inspect_run(p['remote'], local=p['local'], expected_identity=p['identity'])))
''', {'remote': str(REMOTE), 'local': str(LOCAL), 'identity': PLAN['identity']})
exact = f'RESUME EXPERIMENTAL A1 {CONFIG}/{RUN_ID}'
if input(exact + '\n') != exact:
    raise RuntimeError('Confirmation differs')
current_req['confirmation'] = exact
start_queue([current_req])  # Existing Cell7 helper; it preserves native exit/status.
```

### Status (read-only)

```python
print('JOB:', {k: v for k, v in globals().get('JOB', {}).items()
               if k not in ('results', 'current_request')})
for p in (LOCAL/'heartbeat.json', REMOTE/'heartbeat.json', REMOTE/'progress.json'):
    if p.exists():
        row = json.loads(p.read_text())
        fields = ('utc', 'identity', 'status', 'stage', 'current_k', 'last_completed_k',
                  'durable_last_completed_k', 'best_k_so_far', 'best_validation_accuracy',
                  'checkpoint_status', 'last_checkpoint', 'test_values_read')
        print(str(p), {k: row[k] for k in fields if k in row})
```

Heartbeat fields may be stale historical observations. Only the read-only audit's validated k records establish durability; a failed checkpoint with status SAVED does not. Do not run selection verification/final refit/Test to clear this error.
