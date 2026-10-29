# Chạy A1 trên Colab — từng bước

Notebook: `code_CICFLOW/phuong_anA/notebooks/PHUONG_ANA_A1_COLAB.ipynb`. Codex chỉ viết/đọc source, chưa chạy notebook hoặc tests. Mọi lệnh dưới đây do bạn chủ động chạy.

## 1. Push source mới từ Windows

```powershell
Set-Location -LiteralPath 'D:\COMPACT-MI-RFE'
git add -- code_CICFLOW/phuong_anA
git diff --cached --stat -- code_CICFLOW/phuong_anA
git commit --only -m "Add experimental A1 step5 target1 Colab selection" -- code_CICFLOW/phuong_anA
git push -u origin HEAD
git rev-parse HEAD
```

Nếu đã push và không có thay đổi, lấy lại SHA bằng dòng cuối. Không add dataset/checkpoint/source A0 hoặc B. Source A1 vẫn cần các module A0 đã có trong repository vì nó gọi lại các hàm đó, không copy/ghi đè chúng.

## 2. Mở notebook và setup

Upload notebook A1 lên Colab và bắt đầu session khi worker cũ đã dừng. Chạy **Cell1** để mount MyDrive.

**Cell2** giữ nguyên:

```python
DATASET_ROOT = DRIVE / 'CICFLOWMETER_FEATURE_CSV'
OUTPUT_ROOT = DRIVE / 'experiment_results'
LOCAL_ROOT = Path('/content/mi_rfe_svm_5h')
SELECTED_CONFIGS = ['FW256_FS8']
RUN_ID = 'RUN_A1_01'
```

RUN_ID mới phải chưa tồn tại trong namespace A1/config. Để chọn nhiều cấu hình, dùng tên folder thật, ví dụ:

```python
SELECTED_CONFIGS = ['FW256_FS8', 'FW256_FS16', 'FW512_FS8', 'FW512_FS16']
```

Queue chạy từng config, không chạy song song SVM. Ngân sách mặc định của từng worker là36000s, margin1800s, heartbeat15s. Bạn có thể đặt thời gian trước khi bắt đầu phiên;0 là không đặt deadline. Đây không phải giới hạn5 giờ. Giữ nguyên cache_size và các hyperparameters khoa học.

**Cell3**: điền SHA40 từ Git vào APPROVED_COMMIT. Chạy để clone đúng source và xem environment. INSTALL_DEPENDENCIES mặc địnhFalse. Nếu tự bậtTrue, sau cài hãy restart session, chạy lại Cells1–3 và đặt lạiFalse. Không cài package khi worker đang chạy.

## 3. Manifest, tests và preflight

**Cell4**: xem catalog config hợp lệ, metadata, isolation và resource gate. Nó đọc/hash Train/Validation để xác minh SHA, có thể mất thời gian I/O; không fit và không mở Test. Output chỉ nằm dưới `experiment_results/phuong_anA/mi_rfe_svm/...`.

Các folder dataset cần nguyên các Parquet gốc và7 sidecar hiện có: `.stage3_preflight.json`, `.stage3_complete.json`, `feature_schema.json`, `window_feature_schema.json`, `window_integrity.json`, `temporal_gap_audit.json`, `window_counts_by_segment.csv`. Khi upload Drive nhớ cả file tên bắt đầu bằng dấu chấm. Thiếu file: upload bản gốc đúng config; không tự tạo lại manifest.

Parent split manifests nhỏ đã đóng gói dưới `phuong_anA/evidence`, đối chiếu SHA với chính Stage3 của config. Không cần upload source_provenance_manifest.csv lớn hoặc raw CSV. Nếu lineage khác, cung cấp EVIDENCE_ROOT chứa original basenames; mismatch sẽ BLOCKED.

**Cell5**: chạy synthetic unit tests. Cần unittest OK, native exit0 và TEST_EVIDENCE.status=PASS. Tests không fit SVC thật và không đọc production dataset. Không đổi expectation/skip để vượt lỗi.

**Cell6**: preflight đầy đủ sau tests. Cần PASS cho từng config. RAM/disk/library/source/manifest FAIL thì dừng; không giảm dữ liệu/k/số cây/cache để ép PASS.

## 4. Run mới — Cell7

Đổi dòng đầu Cell7:

```python
MODE = 'run'
```

Chạy Cell7, nhập chính xác câu được in, ví dụ:

```text
RUN EXPERIMENTAL A1 FW256_FS8/RUN_A1_01
```

Mỗi config có một câu xác nhận. Worker chạy background; cell xong không có nghĩa k-search xong. Dùng Cell8 theo dõi. Không chạy Cell7 lần nữa khi JOB.running=True.

MI dùng toàn bộ Train; RF-RFE A1 loại tối đa5 feature mỗi vòng; SVM fit đủ k1..100 trên toàn bộ Train/Validation. Sau mỗi k sẽ có checkpoint SAVED. Không phải chờ5 giờ để lưu Drive trong A1.

## 5. Theo dõi/dừng — Cell8

Giữ `ACTION='status'`, chạy lại Cell8 để đọc heartbeat local/Drive, current k, last completed k, durable k, CPU/RAM, best k và checkpoint.

Nếu muốn dừng an toàn:

```python
ACTION = 'stop'
```

Chạy Cell8 rồi chờ status PAUSED và checkpoint SAVED. REQUESTED LOCALLY chỉ là đã đặt cờ, chưa phải upload thành công. SVC đang fit cần đạt safe boundary; có thể vượt ngân sách phiên. Không hard-disconnect để mong giữ phần fit đang dở. Sau khi dừng, đặt ACTION vềstatus để tránh gửi lại cờ.

## 6. Resume

Cùng runtime: giữ source/dataset/config/RUN_ID/budget khoa học như cũ. Nếu đã graceful stop, đặt biến `ACKNOWLEDGE_LOCAL_STOP=True` trước chạy lại Cell7; không chạy lại Cell2 để reset các biến setup. Đổi:

```python
MODE = 'resume'
```

Nhập câu được in, ví dụ `RESUME EXPERIMENTAL A1 FW256_FS8/RUN_A1_01`.

Runtime mới: chạy Cells1–6 với đúng commit và environment cũ. Nếu thấy UNRELEASED LEASE, chỉ sau khi xác nhận worker cũ thực sự đã dừng, điền đúng ID:

```python
TAKEOVER['FW256_FS8'] = 'PREVIOUS RUNTIME STOPPED LEASE_<ID thực tế in ở preflight>'
```

Chạy lại Cell6, rồi Cell7 ởresume. Không dùng nguyên placeholder. Đã commit đến k61 thì k tiếp theo62; k đang fit dở phải fit lại. MI/RFE/scaler hợp lệ được tái dùng. Cache mảng bị mất thì đọc lại Train/Val một lần để dựng cache, không fit lại MI/RFE. Checkpoint khác A0/B, source, config, split, ranking hoặc parameters bị BLOCKED; không chỉnh manifest/hash để ép nối.

Nếu queue đã dừng giữa nhiều config: resume những config đã tạo run; dùng MODE=run cho những config chưa khởi tạo. Không tự trộn hai mode hoặc tạo run thay người dùng.

## 7. Kết quả và verify

Khi JOB.running=False, JOB.error=None và các config có phase_status=SELECTION_COMPLETE, đổi Cell7:

```python
MODE = 'verify'
```

Chạy Cell7 rồi đọc kết quả ở Cell8. Verify tái kiểm tra identity/dataset hash và checkpoint/result chain; không load SVC hoặc đọc Test. Results ở:

```text
MyDrive/experiment_results/phuong_anA/mi_rfe_svm/<CONFIG>/<RUN_ID>/results/
  selection.json
  k_search_results.json
  features.json
  rfe_feature_ranking.csv
  mi_top100.json
  selection_report.md
  A0_A1_COMPARISON.json
  artifact_manifest.json
  SELECTION_COMPLETE.json
```

Xem Validation metrics/best-k ởselection.json. Rank sklearn có nhiều feature đồng hạng; total_order_position mới là thứ tự Top-k. Không có final model hoặc Test metrics vì giai đoạn này chỉ selection. Chưa có dữ liệu để khẳng định A1 nhanh hơn bao nhiêu hoặc Accuracy bằng A0.

## 8. Unit tests Windows có aggregate logging

Bạn tự chạy trong environment có dependencies:

```powershell
$a1Evidence = 'D:\COMPACT-MI-RFE\code_CICFLOW\logs\a1_tests_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fffffff') + '.json'
$a1Command = 'python -B -m phuong_anA.selftest --source "D:\COMPACT-MI-RFE\code_CICFLOW" --evidence "' + $a1Evidence + '"; exit $LASTEXITCODE'
& 'D:\COMPACT-MI-RFE\code_CICFLOW\shared\Invoke-CicflowLogged.ps1' -Stage 'A1' -Task 'Synthetic tests' -Experiment 'phuong_anA mi_rfe_svm' -Configuration 'SYNTHETIC' -RunId 'A1_UNIT' -WorkingDirectory 'D:\COMPACT-MI-RFE\code_CICFLOW' -ScientificStatus 'NOT_ASSESSED' -Command $a1Command
```

Wrapper hiện có append vào `code_CICFLOW/logs/ALL_EXECUTION_RESULTS.log`; không được sửa/chạy bởi Codex trong nhiệm vụ này. Evidence Windows không thay evidence Colab; chạy Cell5 ở Colab.

CLI chạy qua controller: `launch(request_for(config,'preflight'))`; run/resume cần gán request.confirmation đúng chuỗi từ `confirmation(req)`. Native command là `python -B -u -m phuong_anA.cli` nhận JSON stdin hoặc `--request <local.json>`. Chỉ các mode metadata/preflight/run/resume/verify được nhận. Native0 là PASS kỹ thuật hoặc PAUSED;2 là BLOCKED/FAIL;130 là bị ngắt. Không suy ra scientific PASS từ exit0.
