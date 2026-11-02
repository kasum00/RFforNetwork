# Chạy Stage9 trên Colab từng bước

Stage9 chỉ final fit/Test, không chạy lại MI/RFE/k-search. Làm riêng A1 và B. Chưa chạy notebook hoặc tests bằng Codex.

## 1. Push file mới lên GitHub (bạn tự chạy)

```powershell
Set-Location -LiteralPath 'D:\COMPACT-MI-RFE'
git add -- code_CICFLOW/stage9_svm_final notebooks/STAGE9_SVM_A1_B_FINAL_EVALUATION_COLAB.ipynb
git diff --cached --stat -- code_CICFLOW/stage9_svm_final notebooks/STAGE9_SVM_A1_B_FINAL_EVALUATION_COLAB.ipynb
git commit --only -m "Add isolated Stage9 final SVM and one-time A1 B Test evaluation" -- code_CICFLOW/stage9_svm_final notebooks/STAGE9_SVM_A1_B_FINAL_EVALUATION_COLAB.ipynb
git push -u origin HEAD
git rev-parse HEAD
```

Copy SHA40 cuối cùng vào `APPROVED_COMMIT` ở Cell1. Source gốc của mỗi selection sẽ checkout theo **saved commit trong run**, không đổi commit trong artifact cũ. Remote cần còn các commit đó.

## 2. Mở notebook và cấu hình A1

Mở `STAGE9_SVM_A1_B_FINAL_EVALUATION_COLAB.ipynb` trong Colab. Đảm bảo worker selection cũ đã dừng trước khi đổi môi trường/session.

Cell1:

```python
APPROVED_COMMIT = 'SHA40_MOI_BAN_VUA_PUSH'  # thay bằng SHA thật, không giữ placeholder
METHOD = 'A1'
SELECTED_CONFIGS = ['FW256_FS8']  # thử từng config, hoặc list(CONFIGS) để chạy bốn tuần tự
MODE = 'preflight'
ENABLE_FINAL_FIT = False
ENABLE_TEST_EVALUATION = False
TEST_CONFIRMATION = ''
```

Các default A1 run ID là tên tham khảo từ lịch sử; **Cell6 phải xác nhận lại tên thực tế**. Không nhập best_k vào code. Không copy selection A1 sang B.

Drive cần giữ nguyên:

```text
MyDrive/CICFLOWMETER_FEATURE_CSV/<4 configs>/
MyDrive/experiment_results/phuong_anA/mi_rfe_svm/<config>/<run>/
MyDrive/experiment_results/phuong_anB_step2/mi_rfe_svm/<config>/<run>/
```

Giữ toàn bộ checkpoints/results/manifests của từng run. Không chỉ upload `selection.json`. Không cần copy dataset vào `/content` hoặc tạo lại sliding windows.

## 3. Setup và xác minh selection

| Cell | Bạn làm gì | Điều kiện tiếp tục |
|---|---|---|
| 1 | Chọn METHOD/config/SHA/run IDs | Chưa training |
| 2 | Mount Drive bình thường | Không force_remount |
| 3 | Checkout Stage9 SHA40 riêng | Clean exact commit |
| 4 | Kiểm tra profile thư viện | Không tự pip/restart |
| 5 | Kiểm tra SHA metadata nhỏ bốn config | Không mở Test Parquet |
| 6 | Xem tất cả A1/B run tìm được | Có completion chưa phải PASS |
| 7 | Checkout saved source; verify đúng A1 hoặc B | Full chain/source/env/dataset/scaler/order phải PASS |
| 8 | Chạy synthetic tests/logging | Native exit0 **và** status PASS |
| 9 | Preflight resource/source/unit evidence | Chưa fit |
| 10 | Xem từng READY/BLOCKED | Chỉ tiếp tục nếu cả configs đã chọn READY |

Nếu Cell6 tên run khác default: sửa `SELECTION_RUN_IDS` Cell1 rồi chạy setup lại khi không có worker. Nếu B chưa complete thì chờ B selection hoàn tất ở notebook B, không để Stage9 tự chạy selection.

Nếu thiếu/sai dependencies, chỉ sau khi không còn worker, bạn có thể **chủ động** chạy cell cài sau. Cell mặc định không làm gì; không cài nếu environment đã đúng:

```python
INSTALL_APPROVED_PROFILE = False  # chủ động đổi True nếu cần
if INSTALL_APPROVED_PROFILE:
    if globals().get('JOB9', {}).get('running') or globals().get('JOB', {}).get('running'):
        raise RuntimeError('Worker còn chạy; không cài thư viện.')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r',
        str(SOURCE_ROOT / 'stage9_svm_final/requirements-colab.txt')], check=True)
    raise RuntimeError('Cài xong. Tự restart session; tắt flag cài, chạy lại Cells1–10.')
```

Python patch version hoặc libraries không khớp selection vẫn BLOCKED. Không sửa `identity`, không sửa expected version để ép PASS; gửi log để đối chiếu môi trường đã lưu. OS/kernel lịch sử không bị giả lập.

## 4. Final fit riêng A1

Trong Cell11, đổi đúng hai dòng:

```python
ENABLE_FINAL_FIT = True
FINAL_ACTION = 'final-fit'
```

Chạy Cell11, nhập nguyên câu được in cho **từng config**. Câu có METHOD/config/Stage9 ID/selection ID, không dùng câu của run cũ. Sau đó chạy Cell12.

Cell12 chạy tuần tự những config đã xác nhận, không tự chuyển sang B. SVC.fit không có tiến độ nội bộ: heartbeat in stage/CPU/RAM/thời gian; tqdm đếm config đã hoàn tất verified. Không thấy % bên trong fit là đúng. Đừng chạy queue thứ hai khi `JOB9['running']` còn True.

Chỉ muốn xem tiếp monitor sau khi dừng hiển thị:

```python
monitor(JOB9)
```

Lệnh này không khởi động training. Dừng monitor không phải dừng worker. Không restart runtime khi worker đang fit/Test.

Khi worker đã dừng, chạy **Cell13**: từng model/scaler/final checkpoint phải verify PASS. Test chưa được đọc. Final model/scaler mới nằm trong namespace Stage9.

## 5. Test riêng A1 — chủ động xác nhận lần nữa

1. Chạy Cell14, đọc readiness/ledger từng config.
2. Claim ambiguous/interrupted hoặc BLOCKED: dừng, không retry.
3. Trong Cell15 đổi `ENABLE_TEST_EVALUATION = True`.
4. Chạy Cell15, nhập đúng câu `TEST_ONCE ... A1 ...` cho từng config. Không dùng câu final fit.
5. Chạy Cell16: claim durable/readback trước hash/footer/read Test và chỉ một `predict`.
6. Chạy Cell17 để verify saved y/prediction, metrics, matrix, hashes, ledger. Cell17 không mở Test dataset hay predict lại.

Nếu Test đã complete verified, Cell15 bỏ qua evaluation và bảo dùng Cell17. Claim tồn tại mà completion chưa verified luôn BLOCKED, kể cả native exit0 hoặc còn file prediction local.

## 6. Làm riêng B

Sau khi worker A1 đã kết thúc, Cell1 đặt:

```python
METHOD = 'B'
SELECTED_CONFIGS = list(CONFIGS)
```

Giữ flags final/Test False lúc setup. Chạy lại Cells1–10, kiểm tra B run IDs ở Cell6. Sau đó làm Cells11–13 cho final B, Cells14–17 cho Test B. Câu xác nhận chứa B; model/features/best_k/claim khác A1.

Không chọn lại best_k/phương án dựa vào bảng Test. A1 step5 và B step2/coarse-fine có thể cho ranking và k khác nhau.

## 7. Xem và xuất đủ tám dòng

Cell18 tổng hợp cả A1 và B. Metric chỉ hiện khi `VERIFIED_TEST`; PENDING/BLOCKED để trống. Cell19 tạo report version mới ở:

```text
MyDrive/experiment_results/stage9_svm_final/reports/<UTC_UUID>/
```

Có bảng A1, B, comparison riêng; CSV tổng hợp, JSON metric thô và Markdown reports. Xem `test_metrics.json` từng pair để đọc Precision/Recall/F1 attack/Macro-F1/Balanced Accuracy, matrix, features, counts, timings, model/selection SHA. CSV thể hiện phần trăm, không dùng Validation làm Test.

`final_fit_seconds` là SVC.fit đo bởi reference. `final_refit_elapsed_seconds` gồm final scaler + SVC fit + model/scaler serialization, không bao gồm load Train/Val hoặc copy Drive. `prediction_seconds` gồm final-scaler transform + SVC.predict, không gồm hash/load Test/metrics.

## 8. Reconnect/resume

Giữ METHOD/config/run IDs/SHA như cũ, mount và setup lại. Chạy unit tests mới cho session, preflight. Nếu final đã verified, đi Cell13–14; không training lại.

Final chưa complete và chưa claim Test: ở Cell11 dùng `FINAL_ACTION='resume-final'`, `ENABLE_FINAL_FIT=True`; nhập xác nhận rồi Cell12. Nếu lease chưa release, đọc [resume protocol](RESUME_PROTOCOL.md), không xác nhận worker đã dừng khi chưa chắc.

Không chạy lại Test sau interrupted claim. Không xóa claim/checkpoint để chạy cho được.

## 9. Unit tests local qua logging wrapper (bạn tự chạy)

Đây là synthetic tests, không thực nghiệm. Kết quả local không thay thế evidence Colab vì environment khác.

```powershell
Set-Location -LiteralPath 'D:\COMPACT-MI-RFE\code_CICFLOW'
$s9Evidence = Join-Path 'D:\COMPACT-MI-RFE\code_CICFLOW\logs' ('stage9_tests_' + [guid]::NewGuid().ToString('N') + '.json')
$s9Command = 'python -B -u -m stage9_svm_final.selftest --source "D:\COMPACT-MI-RFE\code_CICFLOW" --evidence "' + $s9Evidence + '"; exit $LASTEXITCODE'
& 'D:\COMPACT-MI-RFE\code_CICFLOW\shared\Invoke-CicflowLogged.ps1' -Stage '9' -Task 'synthetic-unit-tests' -Experiment 'STAGE9_SVM_FINAL' -Configuration 'SYNTHETIC' -RunId 'UNIT_STAGE9' -WorkingDirectory 'D:\COMPACT-MI-RFE\code_CICFLOW' -OutputDirectory 'D:\COMPACT-MI-RFE\code_CICFLOW\logs' -ScientificStatus 'NOT_ASSESSED' -Command $s9Command
Write-Host "NATIVE EXIT: $LASTEXITCODE"
Write-Host "EVIDENCE: $s9Evidence"
```

Discovery unittest tương đương (vẫn qua wrapper, không tạo evidence thay Cell8):

```powershell
Set-Location -LiteralPath 'D:\COMPACT-MI-RFE\code_CICFLOW'
& 'D:\COMPACT-MI-RFE\code_CICFLOW\shared\Invoke-CicflowLogged.ps1' -Stage '9' -Task 'unittest-discovery' -Experiment 'STAGE9_SVM_FINAL' -Configuration 'SYNTHETIC' -RunId 'UNIT_STAGE9' -WorkingDirectory 'D:\COMPACT-MI-RFE\code_CICFLOW' -ScientificStatus 'NOT_ASSESSED' -Command 'python -B -u -m unittest discover -s stage9_svm_final/tests -t . -p "test_*.py" -v; exit $LASTEXITCODE'
Write-Host "NATIVE EXIT: $LASTEXITCODE"
```

Log local: `D:\COMPACT-MI-RFE\code_CICFLOW\logs\ALL_EXECUTION_RESULTS.log`.
Log Colab: `MyDrive/code_CICFLOW/logs/ALL_EXECUTION_RESULTS.log` + per-task stdout/stderr/receipts trong namespace Stage9.

`exit $LASTEXITCODE` ở cuối chuỗi Command là lệnh của child PowerShell, giữ nguyên mã Python thay vì để child shell quy về mã lỗi tổng quát. Không sửa wrapper cũ.
