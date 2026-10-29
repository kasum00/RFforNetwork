# Giữ MI/RFE và k1 sau lỗi telemetry — migration sang run độc lập

Mã đã viết, **chưa chạy tests, plan, migration hoặc training**. Không suy ra PASS từ việc code đã tồn tại. Run cũ và backup không được ghi bởi tool này.

Đã có bằng chứng do người dùng chạy:

- Origin commit: `10e389bb5556b001e14c1cd650689d609801c5ba`.
- Origin run: `experiment_results/phuong_anA/mi_rfe_svm/FW256_FS8/RUN_A1_01`.
- MI/scaler/RFE có checkpoint Drive; `k_001` không có trên Drive; có failed checkpoint.
- Local k1 đã qua identity/hash/ranking/count/Accuracy check: Accuracy `0.93616714835761`.
- Recovery copy đã được người dùng xác minh tại `experiment_results/phuong_anA/recovery/FW256_FS8/RUN_A1_01/LOCAL_K001_bb73c060168a4f8aa1cd71e94cb25bb4`.

Tool sẽ tự kiểm tra lại các bằng chứng khi bạn chạy, gồm đầy đủ metrics/timing, không chỉ Accuracy. Nếu fail thì BLOCKED; không sửa manifest hoặc tự train lại.

## Phạm vi và điều kiện

Files mới: `phuong_anA/migration.py`, `phuong_anA/tests/test_migration.py`, hướng dẫn này. Không sửa thêm telemetry/pipeline/scientific/store/preflight để cho phép mismatch. Các source fix trước đó vẫn cần được push cùng commit mới.

Migration chỉ nhận đúng hai cặp SHA source trước/sau lỗi đã được kiểm tra trong phiên:

| File | Bản lỗi | Bản sửa |
|---|---|---|
| telemetry.py |15f56ab7046c313471968a220608ee539175f5cbde097128a50ccb0e7e95eb5c |d8837bfa45b8882d9964ca7d452cf30bbf24bf90a23f4722cba13f52c0b07107 |
| pipeline.py |1dc9f88e53b972cebafb279eebccaa7c7746478b7b283a06ae42149fc821d7e4 |df8317ad76d7e119382702ea3b2ac05d04b065313f0586f8ff4e9a040391c805 |

Các source có sẵn còn lại phải byte-identical. Chỉ chấp nhận thêm đúng module audit/migration và hai test file liên quan; không dùng wildcard cho source khoa học. MI/scaler/RFE/SVC implementations, full dataset/split/audit fingerprint, schema/feature order, library/defaults, resource budget và scientific contract phải giống nhau. Thiếu chứng cứ hoặc khác Python/library/platform/mtime dữ liệu cũng có thể BLOCKED; không tự nới validation.

Run đích được người dùng chọn rõ `RUN_A1_01_FIX01`. Origin và backup chỉ đọc. K1 không tự trở thành checkpoint của run cũ: nó được kiểm tra đầy đủ và tạo record có binding mới trong run mới, kèm immutable migration manifest chứa cả hai identity, source diff, origin ledger head và SHA artifact gốc. Bytes của MI CSV/Top100/scaler joblib/RFE CSV giữ nguyên; toàn bộ numerical/timing payload RFE và k1 giữ nguyên; chỉ identity/dependency references được tạo lại theo thứ tự.

Migration không deserialize joblib, không fit MI/scaler/RFE/SVM, không đọc Test, không final refit, không gọi tự động k2. Cache arrays cũ không được nhập lẫn. Khi resume sau đó, Train/Validation cache có thể cần dựng lại bằng scaler đã lưu; MI/RFE và k1 được bỏ qua khi checkpoint mới hợp lệ.

## Bước 1 — push đúng thư mục từ Windows

Bạn tự chạy, Codex không chạy:

```powershell
Set-Location -LiteralPath 'D:\COMPACT-MI-RFE'
git add -- code_CICFLOW/phuong_anA
git diff --cached --stat -- code_CICFLOW/phuong_anA
git commit --only -m "Add verified A1 k1 migration after telemetry fix" -- code_CICFLOW/phuong_anA
git push -u origin HEAD
git rev-parse HEAD
```

Copy SHA40 cuối. Không reset checkout đang dùng bởi worker cũ; không sửa run_manifest cũ. Giữ runtime hiện tại; các cell sau clone một checkout khác và chạy subprocess mới, không phải restart VM hoặc reload module trong worker cũ.

## Bước 2 — Colab: checkout riêng cho bản sửa/migration

Worker cũ phải đã dừng. DRIVE mount như cũ. Dán SHA mới vào FIX_COMMIT.

```python
from pathlib import Path
import json, os, re, subprocess, sys, uuid
from datetime import datetime, timezone

if globals().get('JOB', {}).get('running'):
    raise RuntimeError('Worker cũ vẫn chạy; chưa migration')

FIX_COMMIT = ''  # full SHA40 từ bước 1
if not re.fullmatch(r'[0-9a-fA-F]{40}', FIX_COMMIT):
    raise ValueError('Điền FIX_COMMIT đủ 40 ký tự')
FIX_COMMIT = FIX_COMMIT.lower()
FIX_REPO = Path('/content/CICFLOW_A1_MIGRATION_' + FIX_COMMIT[:12])
if not FIX_REPO.exists():
    subprocess.run(['git', 'clone', '--', 'https://github.com/kasum00/RFforNetwork.git', str(FIX_REPO)], check=True)
    subprocess.run(['git', '-C', str(FIX_REPO), 'checkout', '--detach', FIX_COMMIT], check=True)
def fixed_git(*args):
    return subprocess.run(['git', '-C', str(FIX_REPO), *args], check=True, text=True, capture_output=True).stdout.strip()
if fixed_git('rev-parse', 'HEAD') != FIX_COMMIT or fixed_git('status', '--porcelain'):
    raise RuntimeError('Checkout mới không sạch/không đúng SHA; không reset để ép chạy')

FIX_SOURCE = FIX_REPO / 'code_CICFLOW'
DRIVE = Path('/content/drive/MyDrive')
DATASET_ROOT = DRIVE / 'CICFLOWMETER_FEATURE_CSV'
OUTPUT_ROOT = DRIVE / 'experiment_results'
LOCAL_ROOT = Path('/content/mi_rfe_svm_5h')
ORIGIN = OUTPUT_ROOT / 'phuong_anA/mi_rfe_svm/FW256_FS8/RUN_A1_01'
RECOVERY = OUTPUT_ROOT / 'phuong_anA/recovery/FW256_FS8/RUN_A1_01/LOCAL_K001_bb73c060168a4f8aa1cd71e94cb25bb4'
NEW_RUN_ID = 'RUN_A1_01_FIX01'
print('FIXED SOURCE:', FIX_SOURCE)
print('ORIGIN READ ONLY:', ORIGIN)
print('RECOVERY READ ONLY:', RECOVERY)
print('NEW RUN:', NEW_RUN_ID)
```

## Bước 3 — synthetic tests trong interpreter mới

Không cài/đổi library để vượt migration gate; môi trường phải khớp run cũ. Test suite mới không fit SVC thật hoặc truy cập production dataset/Git. Tests migration dùng temporary fixture và fake estimator, chặn fit/load model, kiểm tra source drift, checksum, authorization, origin preservation và partial publication.

```python
SESSION_LOGS = LOCAL_ROOT / 'phuong_anA/_migration_sessions'
SESSION_LOGS.mkdir(parents=True, exist_ok=True)
stamp = uuid.uuid4().hex
evidence_path = SESSION_LOGS / (stamp + '_tests.json')
command = [sys.executable, '-B', '-u', '-m', 'phuong_anA.selftest',
           '--source', str(FIX_SOURCE), '--evidence', str(evidence_path)]
started = datetime.now(timezone.utc).isoformat()
env = os.environ.copy(); env['PYTHONDONTWRITEBYTECODE'] = '1'
p = subprocess.Popen(command, cwd=str(FIX_SOURCE), env=env, stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT, text=True, bufsize=1)
with (SESSION_LOGS / (stamp + '_tests.log')).open('x') as log:
    for line in p.stdout:
        print(line, end='', flush=True); log.write(line); log.flush()
native = p.wait()
(SESSION_LOGS / (stamp + '_tests_native.json')).write_text(json.dumps({
    'command': command, 'start': started, 'end': datetime.now(timezone.utc).isoformat(),
    'native_exit_code': native}, indent=2))
TESTS = json.loads(evidence_path.read_text()) if evidence_path.exists() else {'status': 'FAIL'}
print('EXIT:', native, 'TESTS:', TESTS)
if native or TESTS['status'] != 'PASS':
    raise RuntimeError('Tests chưa PASS; không plan/apply/resume')
```

## Bước 4 — helper lưu log và plan chỉ đọc

Logs helper chỉ ghi local session folder, không ghi vào origin hoặc backup. Plan rehash Train/Validation và source/checkpoint để xác minh, có thể mất thời gian I/O; không fit hoặc tạo run mới.

```python
def migration_call(action, payload):
    command = [sys.executable, '-B', '-u', '-m', 'phuong_anA.migration', action]
    stamp = uuid.uuid4().hex
    started = datetime.now(timezone.utc).isoformat()
    env = os.environ.copy(); env['PYTHONDONTWRITEBYTECODE'] = '1'
    p = subprocess.Popen(command, cwd=str(FIX_SOURCE), env=env, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    p.stdin.write(json.dumps(payload)); p.stdin.close()
    result = None
    with (SESSION_LOGS / (stamp + '_' + action + '.log')).open('x') as log:
        for line in p.stdout:
            log.write(line); log.flush()
            if line.startswith('__A1_MIGRATION_RESULT__'):
                if result is not None: raise RuntimeError('Duplicate worker result')
                result = json.loads(line[len('__A1_MIGRATION_RESULT__'):])
            else:
                print(line, end='', flush=True)
    native = p.wait()
    (SESSION_LOGS / (stamp + '_' + action + '_native.json')).write_text(json.dumps({
        'command': command, 'start': started, 'end': datetime.now(timezone.utc).isoformat(),
        'native_exit_code': native, 'reported_status': (result or {}).get('status')}, indent=2))
    print('EXIT:', native, 'STATUS:', (result or {}).get('status'))
    if native or result is None or result['status'] != 'PASS':
        raise RuntimeError('BLOCKED: ' + str(result))
    return result

old_identity = json.loads((ORIGIN / 'run_manifest.json').read_text())['identity']
MIG_REQUEST = {
    'source': str(FIX_SOURCE), 'git_commit': FIX_COMMIT,
    'dataset': str(DATASET_ROOT), 'output': str(OUTPUT_ROOT), 'local_root': str(LOCAL_ROOT),
    'config': 'FW256_FS8', 'run_id': NEW_RUN_ID, 'mode': 'preflight',
    'budget': old_identity['resource_budget'],
    'runtime': {'session_budget_seconds': 36000, 'stop_margin_seconds': 1800, 'heartbeat_seconds': 15},
    'unit_tests': TESTS, 'evidence_root': None,
}
payload = {'request': MIG_REQUEST, 'origin': str(ORIGIN), 'recovery': str(RECOVERY)}
MIG_PLAN = migration_call('plan', payload)
plan = MIG_PLAN['plan']
print('OLD:', plan['original_identity']['run_id'], plan['original_identity']['git_commit'])
print('NEW:', plan['new_identity']['run_id'], plan['new_identity']['git_commit'])
print('SOURCE REVIEW:', plan['source_compatibility'])
print('TRANSFER:', plan['transferred_k'], 'ACCURACY:', plan['validation_accuracy_k1'])
print('NEXT FIT:', plan['next_fit_k'])
print('TARGET:', plan['target'])
print('CONFIRMATION:', MIG_PLAN['authorization_phrase'])
```

Nếu source cũ có SHA khác hai cặp đã review, library/platform/dataset/mtime/budget khác, recovery thiếu metrics hoặc lease chưa release, plan sẽ BLOCKED. Gửi nguyên lỗi, không sửa hashes hoặc nới allowlist. Nếu run đích đã tồn tại, không xóa: chọn ID mới rõ ràng sau khi kiểm tra trạng thái. Plan không tự tạo random experiment.

## Bước 5 — apply có xác nhận, chỉ chuyển artifact

Chỉ chạy sau khi xem plan PASS. Apply tự kiểm tra lại toàn bộ plan và fingerprint; dữ liệu/source thay đổi từ lúc plan sẽ chặn xác nhận cũ. Không khởi chạy k2.

```python
exact = MIG_PLAN['authorization_phrase']
if input('Nhập đúng để CHỈ MIGRATE, chưa training:\n' + exact + '\n') != exact:
    raise RuntimeError('Confirmation differs')
MIG_RESULT = migration_call('apply', {
    **payload, 'plan_sha256': plan['plan_sha256'], 'authorization': exact,
})
print(json.dumps(MIG_RESULT, indent=2))
```

Kỳ vọng cần xác minh thực tế: statusPASS, phase_statusMIGRATION_VERIFIED, current_committed_k[1], next_fit_k2, original_accuracy_k1 giữ đúng0.93616714835761 và training_startedFalse. Đây là kỳ vọng, chưa phải kết quả Codex đã chạy.

Origin RUN_A1_01 và recovery giữ nguyên. Các failed checkpoints cũ không bị xóa/copy làm completed k. New run có MI/scaler/RFE/k1 checkpoint mới và MIGRATION_MANIFEST chứa parent references. Không sao chép array cache hay refit giai đoạn đã xong.

Migration sử dụng staging riêng `.incomplete_migration_*`. Run đích được reserve bằng MIGRATION_INTENT, không có run_manifest admission cho tới khi toàn bộ file/receipt được verify. run_manifest.json được publish cuối. Nếu apply bị gián đoạn trước đó, standard resume sẽ BLOCKED; không đổi marker hoặc chạy training để chữa. Staging và partial new target được giữ để review; không tự xóa hoặc ghi đè. FUSE readback/fsync là best effort, không bảo đảm nội bộ cloud replication.

## Bước 6 — verify chỉ đọc

```python
NEW_REMOTE = Path(plan['target'])
VERIFIED = migration_call('verify', {'remote': str(NEW_REMOTE)})
print(json.dumps(VERIFIED, indent=2))
```

Sau verify PASS, chưa có training mới. Thao tác resume chuẩn phải trỏ tới FIX_SOURCE, FIX_COMMIT, cùng MIG_REQUEST identity và NEW_RUN_ID; không trỏ về RUN_A1_01. Dưới đây là preflight + resume riêng, vẫn cần con người chủ động xác nhận.

## Bước 7 — preflight rồi resume từ k2

```python
# Preflight trong worker chuẩn, không fit.
RESUME_REQUEST = {**MIG_REQUEST, 'mode': 'preflight'}
pre_command = [sys.executable, '-B', '-u', '-m', 'phuong_anA.cli']
pre_started = datetime.now(timezone.utc).isoformat()
pre = subprocess.run(pre_command,
    cwd=str(FIX_SOURCE), input=json.dumps(RESUME_REQUEST), text=True, capture_output=True)
stamp = uuid.uuid4().hex
(SESSION_LOGS / (stamp + '_preflight.log')).write_text(pre.stdout + pre.stderr)
(SESSION_LOGS / (stamp + '_preflight_native.json')).write_text(json.dumps({
    'command': pre_command, 'start': pre_started, 'end': datetime.now(timezone.utc).isoformat(),
    'native_exit_code': pre.returncode}, indent=2))
print(pre.stderr)
matches = [line for line in pre.stdout.splitlines() if line.startswith('__A1_SELECTION_RESULT__')]
if pre.returncode or len(matches) != 1:
    print(pre.stdout); raise RuntimeError('Preflight failed; do not resume')
PREFLIGHT = json.loads(matches[0][len('__A1_SELECTION_RESULT__'):])
print('PREFLIGHT:', PREFLIGHT['status'], PREFLIGHT.get('reasons'))
if PREFLIGHT['status'] != 'PASS': raise RuntimeError('Preflight BLOCKED')
```

Sau đó dùng Cell7 helper `start_queue` hiện có. Helper launcher cũ được giữ nguyên hash trong phép review, nhận source/cwd từ request nên worker sẽ dùng FIX_SOURCE. Nếu runtime mới chưa có helper, chạy phần định nghĩa start_queue của notebook ở MODE=preflight trước; không chạy worker cũ.

```python
if globals().get('JOB', {}).get('running'):
    raise RuntimeError('Queue vẫn chạy')
RESUME_REQUEST = {**MIG_REQUEST, 'mode': 'resume'}
exact = 'RESUME EXPERIMENTAL A1 FW256_FS8/' + NEW_RUN_ID
if input(exact + '\n') != exact: raise RuntimeError('Confirmation differs')
RESUME_REQUEST['confirmation'] = exact
start_queue([RESUME_REQUEST])
```

Standard preflight/Store identity validation vẫn hoạt động. Code sẽ load full Train/Val một lần nếu cache mới chưa có, load scaler đã lưu, dùng lại MI/RFE/k1 rồi bắt đầufitk2. Không bảo đảm thời gian nhanh của bước dựng cache. Dừng an toàn/heartbeat như A1 bình thường. Không final refit/Test.

## Status đúng run mới

```python
NEW_LOCAL = Path(plan['local_target'])
print('JOB:', {k:v for k,v in globals().get('JOB', {}).items() if k not in ('results','current_request')})
for path in (NEW_LOCAL/'heartbeat.json', NEW_REMOTE/'heartbeat.json'):
    if path.exists():
        row = json.loads(path.read_text())
        fields = ('utc','status','stage','current_k','last_completed_k','durable_last_completed_k',
                  'best_k_so_far','best_validation_accuracy','checkpoint_status','last_checkpoint','test_values_read')
        print(str(path), {k:row[k] for k in fields if k in row})
```

Trước khi load/verify history trong worker, optional best fields có thểNone. Sau restore k1, best Accuracy phải lấy từ record đã chuyển; k1 vẫn durable1, current fit sẽ2. SAVED failed record không bao giờ làm tăng durable k. Không dùng output A1 thay official reproduction.
