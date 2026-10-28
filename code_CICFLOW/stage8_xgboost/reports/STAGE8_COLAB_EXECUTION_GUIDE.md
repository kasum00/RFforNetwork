# Chạy XGBoost đủ 4 cấu hình trong một lần Colab

Notebook mới:
`D:\COMPACT-MI-RFE\code_CICFLOW\stage8_xgboost\notebooks\CICFLOW_STAGE8_XGBOOST_COLAB.ipynb`.

**Chạy setup Cell1–8. Tại Cell8 chọn MODE="full". Cell9 xác nhận ALL4 một lần,
sau đó tự chạy lần lượt FW256_FS8 → FW256_FS16 → FW512_FS8 → FW512_FS16.**
Không phải đổi FW/FS hay chạy lại notebook bốn lần. Không bấm bốn worker song
song; mỗi worker kết thúc mới bắt đầu cấu hình tiếp theo để giải phóng RAM/GPU.

Nhánh hiện tại là baseline_xgb, đủ380 feature. MI-RFE-XGB còn BLOCKED do policy
RFE chưa duyệt. Không dùng kết quả/RF-selected k từ mi_rfe_rf_*.

Setup dùng RUN_TAG mặc định ở Cell2 để lưu log preview đúng các thư mục run.
Nếu muốn run02 ngay từ đầu, hoặc baseline_xgb_*_run01 đã thuộc implementation
khác, đặt RUN_TAG="run02" ở Cell2 trước setup và giữ cùng tag tại Cell8.
Không sửa/xóa owner hoặc run cũ để ép dùng lại tên.

## 1. Đưa source lên Git và lấy SHA (bạn tự chạy)

PowerShell:

```powershell
Set-Location -LiteralPath 'D:\COMPACT-MI-RFE'
git add -- code_CICFLOW/stage8_xgboost
git diff --cached --stat -- code_CICFLOW/stage8_xgboost
git commit --only -m "Add CICFLOW Stage8 XGBoost four-config Colab queue" -- code_CICFLOW/stage8_xgboost
git push -u origin HEAD
git rev-parse HEAD
```

Không git add toàn repository/dataset/runs. Nếu source phụ thuộc
code_CICFLOW/colab/cicflow_reference_pipeline.py hoặc stage4_mi/stage5_rfe
chưa có trong Git, đưa đúng source đã dùng cho RF lên trước; Cell3 kiểm tra file
bắt buộc. Không upload lại Parquet/model/checkpoint RF.

Mở notebook mới trên GitHub bằng Colab, hoặc upload chính file ipynb mới.
Notebook clone source từ Git **đúng APPROVED_COMMIT**, không dùng source cũ
trên Drive. Vì vậy bạn không cần cập nhật file Python trên Drive cho notebook
Stage8 này. Notebook RF trước vẫn có cơ chế source khác; không nhầm hai notebook.

## 2. Chuẩn bị Drive

```text
/content/drive/MyDrive/CICFLOWMETER_FEATURE_CSV/
  FW256_FS8/
  FW256_FS16/
  FW512_FS8/
  FW512_FS16/
```

Mỗi folder cần Train/Val/Test Parquet đang có và các sidecar nhỏ đã tạo từ
Stage3: **window_integrity.json**, **window_feature_schema.json**; giữ thêm
feature_schema.json nếu đã có. Nếu Drive thiếu sidecar, copy đúng file nhỏ
tương ứng từ D:\COMPACT-MI-RFE\CICFLOWMETER_FEATURE_CSV\<CONFIG>\ lên.
Không tạo lại manifest/split/window hoặc copy thêm Parquet lớn.

Output root bắt buộc: /content/drive/MyDrive/experiment_results.
Không dùng MyDrive/COMPACT-MI-RFE/WINDOWS_CPLUS của notebook tham chiếu.

## 3. Thứ tự các cell

| Cell | Bạn làm gì | Có fit/Test không? |
|---|---|---|
|1|Mount Drive|Không|
|2|Xem versions/CPU/RAM; defaults preflight/CPU|Không|
|3|Dán APPROVED_COMMIT40 ký tự; clone/import sạch đúng SHA|Không|
|4|Kiểm tra paths và đủ bốn cấu hình/sidecars|Không|
|5|Metadata/schema/order/count/labels sidecar của cả4|Không đọc row values|
|6|Xem và kiểm tra BUDGET/version/backend cho cả4|Không model.fit|
|7|Chạy unit/mock tests; xem checkpoint state|Chỉ fixtures, không dataset training|
|8|Chọn MODE, EXPERIMENT, BACKEND, RUN_TAG|Không|
|9|Một lần: preflight/full/resume/verify ALL4|Fit chỉ khi full/resume|
|10|Đọc hướng dẫn resume|Không|
|11|Xem final-refit/feature state đã lưu|Không|
|12|Xem Test-once metrics/protocol đã lưu|Không predict lại|
|13|Verify artifacts và marker|Không load model/dataset|
|14|Bảng kết quả bốn cấu hình|Không|

XGBoost giữ200 cây/depth10/min_child_weight3; các tham số khác giữ source và
default của pinned version, được ghi trong configuration.json. Không tự thay
version/hyperparameter để gate PASS. CPU mặc định; CUDA phải chọn rõ và đủ
build/GPU/VRAM. Nếu thay backend/dependencies sau tests, chạy lại tests và
preflight; resume đòi đúng environment/backend ban đầu.

Profile nguồn: numpy2.2.6, pandas2.3.3, scipy1.15.3, sklearn1.7.2,
joblib1.5.3, psutil5.9.5, xgboost3.4.1; CUDA cần nvidia-ml-py13.580.82.
PyArrow phải có và version thực tế được bind. Notebook không tự pip install.
Nếu package/version không có hoặc không tương thích với runtime, dừng BLOCKED
và gửi output; không âm thầm thay version hoặc backend.

Nếu cần cài profile nguồn, sau Cell3 bạn có thể **chủ động thêm và chạy riêng**
cell sau, rồi restart runtime và chạy lại setup từ Cell1. Đây không phải bước
tự động của queue và Codex không chạy nó:

```python
import subprocess, sys
subprocess.run([
    sys.executable, "-m", "pip", "install", "-r",
    str(SOURCE_ROOT / "stage8_xgboost/requirements-colab-reference.txt"),
], check=True)
```

Nếu pip báo version không tồn tại/không hỗ trợ Python hiện tại, dừng và gửi lỗi;
không đổi sang version khác để ép PASS. Khả năng cài đủ profile này trên Colab
hiện tại chưa được Codex kiểm chứng.

Budget CICFLOW được hiển thị tại Cell6, khác budget C+44GiB/40GiB của nguồn:
start RAM8GiB + data estimate, runtime free RAM4GiB, RSS cap40GiB,
free disk4GiB, free VRAM4GiB khi CUDA, max12h/config. Đây là admission policy
cần bạn xem trước, không phải scientific parameters hay dự báo thời gian.
Không tự hạ budget, sampling hoặc giảm cây khi FAIL.

## 4. Dry preflight trước, sau đó một lần full ALL4

Cell8 mặc định:

```python
MODE = "preflight"
EXPERIMENT = "baseline_xgb"
BACKEND = "cpu"
RUN_TAG = "run01"
STOP_ON_ERROR = False
```

Chạy Cell8 rồi9 để xem preflight của đủ4 cấu hình. Nếu PASS và muốn train,
đổi **MODE="full"** trong Cell8, chạy lại8 rồi9. Cell9 hỏi đúng một câu xác
nhận ALL4, có backend/run tag/commit/budget digest. Copy/gõ đúng câu cell in,
không nhập chuỗi placeholder. Không cần chạy lại setup khi cùng session/source.

Cell9 thực hiện cho từng config:

```text
Fresh resource/identity admission
→ Train380 + scaler fit Train → XGB200 rounds
→ báo Validation fixed model (không tuning)
→ freeze380 feature (MI/RFE/k-search NOT_APPLICABLE)
→ scaler + XGB mới fit Train+Validation200 rounds
→ durable Test claim → Test evaluate một lần
→ verify hashes/identity → worker exit0 → RUN_COMPLETE tạo sau cùng
→ worker kế tiếp cho cấu hình tiếp theo
```

STOP_ON_ERROR=False: một cấu hình lỗi được ghi rõ, các cấu hình độc lập còn lại
vẫn được thử; không tự retry cấu hình lỗi. True: dừng ngay khi lỗi. Ctrl+C luôn
dừng queue. RUN_TAG run02 dùng cho experiment mới; không đổi tag khi resume.

## 5. Resume/verify cả batch

Sau runtime restart: chạy lại setup1–7 với cùng SHA/versions/backend/budget,
Cell8 chọn MODE="resume", RUN_TAG giống cũ, rồi Cell9 xác nhận RESUME ALL4.

- Đã hoàn tất: kiểm tra identity + verify rồi skip, không mở Test lại.
- Chưa bắt đầu: bắt đầu fresh trong phạm vi batch đã chọn.
- Fit dở: tiếp tục từ model+manifest10-round checkpoint đã verify. Chỉ giữ
  hai checkpoint mới nhất mỗi fit-stage theo source manager.
- Corrupt newest/orphan manifest/model: BLOCKED, không fallback ngầm.
- Test intent mơ hồ: BLOCKED; không đánh giá Test lần hai.
- Test đã hoàn tất và artifact đủ binding: chỉ phục hồi publication, không fit
  hoặc đọc dữ liệu lại.
- Partial model/scaler export không có final state: giữ nguyên để inspect,
  không tự xóa/ghi đè để ép resume.

Muốn chỉ verify: Cell8 MODE="verify", đúng run tag; Cell9 kiểm tra cả4, không
fit. Verify chỉ dùng metadata/hashes, không cần dự đoán Test hoặc load model.
Không tuyên bố CPU/CUDA hoặc checkpoint/single-call bitwise tương đương.

## 6. Output trên Drive

```text
/content/drive/MyDrive/experiment_results/
├── baseline_xgb_FW256_FS8_run01/
├── baseline_xgb_FW256_FS16_run01/
├── baseline_xgb_FW512_FS8_run01/
└── baseline_xgb_FW512_FS16_run01/
```

Mỗi run:

```text
run_manifest.json
configuration.json
preflight.json
feature_selection.json          # baseline: NOT_APPLICABLE, selected_features380
checkpoints/booster/             # protocol + validation_fit/final_refit UBJ manifests
checkpoints/final_refit_artifacts.json
models/final_model.ubj
models/final_scaler.joblib
models/booster_config.json
metrics/validation_metrics.json
metrics/test_metrics.json
metrics/confusion_matrix.csv
logs/execution.log               # append stdout+stderr, native exit, time/resource
logs/events.jsonl
logs/progress.json
reports/EXPERIMENT_REPORT.md
reports/resource_usage.json
reports/execution_*.json
reports/BATCH_*.json
RUN_VALIDATED.json
RUN_COMPLETE.json                # chỉ sau exit0 và verify đầy đủ
```

Mọi log/report Colab nằm trong run_directory; không có log Stage8 mới ở
repository root/MyDrive root. Folder có thể tồn tại từ preflight, nhưng chỉ
RUN_COMPLETE hợp lệ mới chứng minh hoàn tất kỹ thuật. Không tạo metric giả.
Logs ghi rõ nếu không lưu được; exit0 một mình không thành PASS.

Các mi_rfe_rf_* và notebook tham chiếu cũ được giữ nguyên, không migration.

## 7. Tests trên Windows nếu cần (bạn tự chạy)

Dùng Python environment đã có dependencies. Wrapper chung append log chính
thức của CICFLOW và bảo toàn native exit code:

```powershell
& 'D:\COMPACT-MI-RFE\code_CICFLOW\shared\Invoke-CicflowLogged.ps1' -Stage '8 XGBoost' -Task 'Synthetic tests' -Configuration 'ALL4_SYNTHETIC' -RunId 'STAGE8_UNIT' -WorkingDirectory 'D:\COMPACT-MI-RFE\code_CICFLOW' -Command 'python -m unittest discover -s stage8_xgboost/tests -t . -p test_*.py -v; exit $LASTEXITCODE'
```

Lệnh trên không được Codex chạy. Cell7 là cách chính để có bằng chứng tests
đúng source/environment Colab, tự lưu stdout/stderr/test counts.

## 8. Gửi lại sau khi chạy

Mỗi cấu hình gửi preflight.json, run_manifest.json, configuration.json,
metrics/validation_metrics.json, metrics/test_metrics.json nếu đã có,
reports/BATCH_*.json, reports/EXPERIMENT_REPORT.md và logs/execution.log.
Nếu lỗi thêm logs/progress.json, execution_*.json/FAILED_*.json và
checkpoints/booster/protocol_state.json. Không gửi Parquet/cache lớn hoặc
model.ubj trừ khi cần điều tra cụ thể.

Technical PASS/COMPLETE không đổi giới hạn khoa học:
PROVISIONAL_RESEARCH, OFFICIAL_BLOCKED, capture/session UNKNOWN, proxy labels.
Codex chỉ viết source/tests/docs, chưa chạy test/preflight/training mới.
