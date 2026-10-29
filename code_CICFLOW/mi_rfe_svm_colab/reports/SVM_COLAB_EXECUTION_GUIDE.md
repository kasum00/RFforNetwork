# Chạy MI–RFE–SVM CICFLOW trên Colab

Mã mới chưa được Codex chạy/test/benchmark. Bạn tự chạy các bước dưới đây. Không dùng `Run all` để bật training: mặc định là preflight, final refit và Test có công tắc riêng.

## 1. Đưa source lên GitHub

Các lệnh sau để **bạn tự copy vào PowerShell**. Chỉ stage thư mục mới; không add dataset hoặc run cũ. Nếu Git báo không có thay đổi thì không tạo commit rỗng.

```powershell
Set-Location -LiteralPath 'D:\COMPACT-MI-RFE'
git add -- code_CICFLOW/mi_rfe_svm_colab
git diff --cached --stat -- code_CICFLOW/mi_rfe_svm_colab
git commit --only -m "Add CICFLOW SVM local progress and five-hour Drive checkpoints" -- code_CICFLOW/mi_rfe_svm_colab
git push -u origin HEAD
git rev-parse HEAD
```

Copy full commit40 ký tự vào `APPROVED_COMMIT` ở Cell2. Mở notebook trên GitHub hoặc upload notebook mới lên Colab:

`D:\COMPACT-MI-RFE\code_CICFLOW\mi_rfe_svm_colab\notebooks\MI_RFE_SVM_OPTIMIZED_COLAB.ipynb`

Source được clone theo đúng commit vào `/content`, không lấy ngẫu nhiên module Python cũ từ Drive. Repository phải có các source đang được tái sử dụng: `code_CICFLOW/colab/cicflow_reference_pipeline.py`, Stage4/5 helpers, `huyen_code/ciciomt24_training.py`, các legacy resumable sources, notebook ColabPro và CODE_ctu13. Không sửa hoặc push lại các source gốc chỉ để qua gate.

## 2. Dataset trên Drive

Giữ nguyên:

```text
MyDrive/CICFLOWMETER_FEATURE_CSV/
  FW256_FS8/
  FW256_FS16/
  FW512_FS8/
  FW512_FS16/
```

Mỗi folder cần các Parquet hiện có (`train_windows.parquet`, `val_windows.parquet`, `test_windows.parquet`) và JSON schema/integrity hiện có, gồm `feature_schema.json`, `window_feature_schema.json`, `window_integrity.json`. Không tự sinh lại sidecar/split để đạt PASS. Không upload cache numpy, Gram matrix hoặc 100 model SVC lên Drive.

Output mặc định:

```text
/content/drive/MyDrive/experiment_results/mi_rfe_svm_FW256_FS8_run01
/content/drive/MyDrive/experiment_results/mi_rfe_svm_FW256_FS16_run01
/content/drive/MyDrive/experiment_results/mi_rfe_svm_FW512_FS8_run01
/content/drive/MyDrive/experiment_results/mi_rfe_svm_FW512_FS16_run01
```

Run đã tồn tại khác identity/format: không ghi đè. Dùng `MODE='verify'` cho output reference-layout cũ đã complete; hoặc chọn `RUN_TAG='run02'` cho thí nghiệm mới. Có thể khai báo `REUSE_FROM[config]` trỏ tới MI/RFE-SVM reference-layout cũ để nhập artifact hợp lệ vào run mới. Các giới hạn import được giải thích bên dưới.

## 3. Trình tự các cell

| Cell | Thao tác |
|---|---|
| 1 | Mount Drive |
| 2 | Điền SHA, clone sạch đúng commit, kiểm tra đường dẫn import |
| 3 | Kiểm tra dependencies và chạy synthetic tests. Có import numpy/scikit-learn và forest fixture nhỏ, không fit SVC thật. Cài packages chỉ khi tự bật INSTALL_DEPENDENCIES; sau đó restart session. |
| 4 | Chọn cấu hình, run tag và paths |
| 5 | Xem scientific contract |
| 6 | Xem resource/source/dataset gate; BLOCKED thì xử lý nguyên nhân, không giảm cache/dữ liệu/k |
| 7 | Xem MI/RFE import, checkpoint gần nhất và lease chưa release |
| 8 | Chọn MODE và xác nhận local stop nếu có |
| 9 | Preflight/verify hoặc khởi chạy queue k-search run/resume; không final refit/Test |
| 10 | Xem local progress và checkpoint receipt; có thể chạy lại trong khi background worker hoạt động |
| 11 | Gửi yêu cầu checkpoint/stop local, chờ safe boundary |
| 12 | Xem selection đã khóa sau đủ100 k |
| 13 | Bật riêng final refit Train+Validation; chờ FINAL_REFIT_SAVED |
| 14 | Bật riêng Test một lần, chỉ sau Cell13 hoàn thành |
| 15 | Verify manifest/hash và đọc saved metrics; không load model hoặc đánh giá Test lại |

GPU T4 không tự tăng tốc sklearn SVC/RF. Resource estimate có tính cache SVC10000MiB và các bản sao MI/scaler. Không đủ RAM: BLOCKED, không tự đổi cache_size hoặc sampling.

## 4. Run mới

Ở Cell4, chọn riêng một config:

```python
SELECTED_CONFIGS = ['FW256_FS8']
RUN_TAG = 'run01'
```

Hoặc queue cả bốn trong cùng Colab, chạy tuần tự:

```python
SELECTED_CONFIGS = list(CONFIGS)
```

Chạy Cell4–7 rồi tại Cell8:

```python
MODE = 'run'
ACKNOWLEDGE_LOCAL_STOP = False
```

Chạy Cell9. Nhập đúng câu xác nhận riêng cho mỗi run đang chọn, ví dụ `RUN PROVISIONAL mi_rfe_svm_FW256_FS8_run01`. Không cần tự copy/paste stdout để lưu log. Worker hiển thị từng k và ghi local; bundle/log snapshot tự được commit theo chính sách5 giờ.

Cell9 trả về sớm vì queue chạy background. `JOB['running'] == True` nghĩa vẫn đang chạy. Cell10 hiển thị tiến độ; Cell12 chỉ xem selection sau khi queue đã xong. `SELECTION_FROZEN` chỉ xác nhận k-search đã hoàn thành, chưa phải kết quả Test cuối.

Nếu một config lỗi, queue dừng, không tự retry hoặc tự tạo thí nghiệm mới. Config đã có run thì resume; config chưa khởi tạo thì dùng run trong một queue mới. Không tự chuyển mode giữa run/resume cho từng config.

## 5. Checkpoint và graceful stop

Sau mỗi k: chỉ ghi local `/content/mi_rfe_svm_5h/<RUN_NAME>/state`. Drive chỉ được truy cập khi interval tới hạn hoặc một mốc đã nêu trong thiết kế.

Ở Cell11:

```python
ACTION = 'checkpoint'  # yêu cầu lưu tại safe boundary tiếp theo rồi tiếp tục
```

Hoặc:

```python
ACTION = 'stop'        # lưu tại safe boundary rồi dừng queue
```

Chờ `DRIVE CHECKPOINT COMMITTED` và trạng thái `PAUSED_AT_SAFE_CHECKPOINT` trước khi đóng runtime. Không coi thông báo `REQUESTED LOCALLY` là đã upload. Nếu fit kéo dài hơn5 giờ, nó vẫn phải hoàn thành hoặc bị ngắt trước khi xử lý yêu cầu. Không có checkpoint giữa SVC.fit.

## 6. Resume

Cùng runtime, giữ nguyên SOURCE, DATASET, OUTPUT, LOCAL_ROOT, RUN_TAG, BUDGET và REUSE_FROM. Cell8:

```python
MODE = 'resume'
ACKNOWLEDGE_LOCAL_STOP = True  # chỉ nếu resume sau yêu cầu stop của chính bạn
```

Chạy lại Cell9 với đúng các config đã có run. Local cache/scaler và k hoàn thành hợp lệ được dùng lại. Nếu Drive mới commit đến61 và không còn local tiến độ mới hơn, k tiếp theo là62. Fit đang dở không được resume.

Runtime mới: chạy Cell1–8 với đúng commit, environment và identity cũ. Nếu gate báo lease chưa release, trước hết bảo đảm runtime cũ thực sự dừng. Sau đó điền đúng ID Cell7 in:

```python
TAKEOVER_CONFIRMATIONS['FW256_FS8'] = 'PREVIOUS RUNTIME STOPPED LEASE_<ID chính xác>'
```

Không dùng nguyên placeholder. Chạy lại preflight, rồi Cell8–9. Checkpoint được restore từ bundle COMMITTED mới nhất. Corrupt checkpoint, mismatch source/library/data/budget hoặc gaps phải dừng để kiểm tra. Không xóa marker/hash để ép resume.

Nếu local state/receipt hỏng nhưng Drive vẫn có bundle hợp lệ, giữ nguyên thư mục đó và đổi `LOCAL_ROOT` sang thư mục mới dưới `/content`, ngoài `/content/drive`, rồi restore từ Drive. Các k chỉ có local nhưng chưa được chứng minh sẽ không được nhập lẫn.

## 7. Tái sử dụng MI/RFE

`REUSE_FROM[config]` chỉ chấp nhận `mi_rfe_svm` đúng config ở CICFLOW reference-layout, có `run_identity.json`, `mi_state.json`, `mi_top100.json`, `mi_feature_scores.csv`, `rfe_state.json`, `rfe_feature_ranking.csv`. Cả phase phải hoàn tất, checksum/order380→100→ranking1..100 đúng, source numerical hashes và môi trường hiện tại khớp. Run gốc và commit/hashes được ghi làm provenance; origin được đọc, không sửa.

Đây là import MI/RFE, không import các k từ run khác. Top100 scaler Train có thể phải fit một lần nếu origin không có scaler đã lưu. Không tính lại MI/RFE khi đã nhập hợp lệ. Stage4 seed42+feature_index là biến thể khác; ranking RF hoặc XGBoost cũng không được dùng thay SVM RFE. Legacy CSV resumable bundles chỉ dùng làm tài liệu thiết kế, không tự nhận làm CICFLOW Parquet checkpoint.

## 8. Final refit và Test

Khi Cell12 xác nhận frozen selection, bật `CONFIRM_FINAL_REFIT=True` ở Cell13 và nhập câu `FINALIZE PROVISIONAL <RUN_NAME>`. Chờ toàn bộ worker cần thiết báo `FINAL_REFIT_SAVED`; không bật Cell14 khi queue còn chạy. Scaler mới và SVC mới fit trên toàn bộ Train+Validation với đúng selected prefix.

Sau đó bật `CONFIRM_TEST_ONCE=True` ở Cell14. Code kiểm tra final artifacts, ghi/readback Test claim lên Drive, mới mở Test và predict đúng một lần. Nếu claim đã tồn tại mà không có saved result hợp lệ thì BLOCKED, không bấm chạy lại Test để chữa. Nếu kết quả đã lưu hợp lệ nhưng publication bị gián đoạn, finalize/resume chỉ kiểm tra và republish byte-identical artifacts, không predict lại.

Cuối cùng chạy Cell15. `PASS` ở verify là PASS kỹ thuật của artifact chain; khoa học vẫn `PROVISIONAL_RESEARCH / OFFICIAL_BLOCKED`. Kết quả cần xem: `result.json`, `test_result.json`, `k_search_results.json`, `selection_frozen.json`, feature lists/ranking, `model.joblib`, `final_scaler.joblib`. Accuracy/F1/confusion matrix nằm trong các JSON metrics, không suy ra từ tên folder.

## 9. Lệnh unit tests trên Windows, có logging wrapper

Bạn tự chạy trong môi trường có dependencies. Đây chỉ là mock/synthetic tests, không full SVM hoặc Parquet thật:

```powershell
$svmEvidence = 'D:\COMPACT-MI-RFE\code_CICFLOW\logs\svm_5h_tests_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fffffff') + '.json'
$svmTestCommand = 'python -B -m mi_rfe_svm_colab.selftest --source "D:\COMPACT-MI-RFE\code_CICFLOW" --evidence "' + $svmEvidence + '"; exit $LASTEXITCODE'
& 'D:\COMPACT-MI-RFE\code_CICFLOW\shared\Invoke-CicflowLogged.ps1' -Stage 'SVM 5-hour checkpoint' -Task 'Synthetic unit and regression tests' -Experiment 'mi_rfe_svm' -Configuration 'SYNTHETIC' -RunId 'SVM5H_UNIT' -WorkingDirectory 'D:\COMPACT-MI-RFE\code_CICFLOW' -ScientificStatus 'PROVISIONAL_RESEARCH' -OutputDirectory 'D:\COMPACT-MI-RFE\code_CICFLOW\logs' -Command $svmTestCommand
Write-Host "WRAPPER EXIT CODE: $LASTEXITCODE"
```

Wrapper hiện có append vào `D:\COMPACT-MI-RFE\code_CICFLOW\logs\ALL_EXECUTION_RESULTS.log`. Source wrapper được đọc để kiểm tra tham số; không được sửa hoặc chạy trong nhiệm vụ này. Evidence Windows không được coi là evidence cho environment Colab; chạy lại Cell3 ở Colab.

## 10. CLI Colab khi cần tự chạy ngoài notebook

Notebook là đường chạy khuyến nghị vì đã tạo request/evidence/confirmation, stream hai luồng output và giữ native exit code. CLI nhận JSON stdin hoặc file local:

```python
# Chạy sau các cell setup, cùng notebook. Không tự đổi tham số khoa học.
config = 'FW256_FS8'  # hoặc ba config còn lại
req = request_for(config, 'preflight')
result = launch(req)  # log local, native exit code và structured status đều được kiểm tra
```

Để dùng cùng helper với run/resume/finalize, tạo request đúng mode, đặt `req['confirmation'] = '<MODE viết hoa> PROVISIONAL <RUN_NAME>'`, và gửi qua `start_queue([req])`. Verify dùng `launch(request_for(config, 'verify'))`, không cần confirmation. Backend chỉ CPU, không có tự chuyển LinearSVC/SGD/GPU.

Native worker codes:0 khi trạng thái kỹ thuật được chấp nhận;2 khi FAIL/BLOCKED/exception;130 khi bị ngắt có kiểm soát. Exit0 có thể là selection đã khóa, final model đã lưu, hoặc PAUSED; phải đọc structured status/marker tương ứng. Log snapshot ở Drive chỉ cập nhật khi có bundle/mốc được xác nhận; output kể từ checkpoint cuối có thể mất khi runtime bị hard kill.

Cell3 lưu stdout/stderr tests đã merge, command, giờ bắt đầu/kết thúc và native exit vào file local có UUID dưới `/content`. Worker snapshot các log/evidence đó vào bundle ở boundary tiếp theo. Session logs của launcher cũng ở `/content/svm5h_notebook_sessions`; không được coi là đã durable trên Drive trước khi có bundle. Nếu lỗi cleanup sau final publication, code giữ nguyên completed artifacts, không append checkpoint mới; đọc native error rồi chạy verify để phân biệt lỗi tiến trình với kết quả đã xác minh.
