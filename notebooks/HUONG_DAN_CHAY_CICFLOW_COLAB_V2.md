# Hướng dẫn chạy CICFLOW Colab theo notebook mẫu

Dùng đúng file `COMPACT_MI_RFE_CICFLOW_Colab_V2.ipynb` **đã sửa 12 cell**. Bản này chạy toàn bộ experiment và lưu như ảnh của bạn; helper cũ colab_pipeline_v2.py không còn là entry point của notebook này.

## 1. Upload các file đã sửa lên Drive

Bạn đã có Stage 4/5 và dataset trên Drive. Cần upload mới hoặc thay đúng các file:

| File local | Đích Drive |
|---|---|
| code_CICFLOW/colab/cicflow_reference_pipeline.py | MyDrive/code_CICFLOW/colab/cicflow_reference_pipeline.py |
| code_CICFLOW/colab/test_cicflow_reference_pipeline.py | MyDrive/code_CICFLOW/colab/test_cicflow_reference_pipeline.py |
| notebooks/COMPACT_MI_RFE_CICFLOW_Colab_V2.ipynb | Mở bản mới bằng Colab hoặc upload notebook mới thay bản đang dùng |

Giữ các dependency source đã có: MyDrive/code_CICFLOW/stage4_mi/mi_selection.py và MyDrive/code_CICFLOW/stage5_rfe/rfe_selection.py (và package files mà chúng import). Không upload lại Parquet nếu đã đủ Train/Val/Test trong bốn cấu hình.

Không cần chuyển hoặc xóa COMPACT_MI_RFE_FRESH_RUNS. Code mới đọc dataset hiện có và tạo experiment mới. **Push Git không tự cập nhật source trên Drive**; Cell 4 phải thấy hai helper mới.

## 2. Chạy setup và dry-run

Chạy Cell 1–7 theo thứ tự. Cell 5 chủ động chạy unit/regression tests, không dùng dataset thật. Cell 6 kiểm tra metadata, chưa đọc row values. Nếu một bước FAIL/BLOCKED, xem log; không tiếp tục full.

Cell 8:

```python
EXPERIMENT = "mi_rfe_rf"
FW = "256"
FS = 8
RUN_NAME = ""             # tự chọn run01, run02... chưa tồn tại
MODE = "dry_run"
CONFIRMATION = ""
GIT_COMMIT = "NOT_CAPTURED"  # hoặc SHA thật của source bạn upload
```

Chạy Cell 8–9; đọc tên run, scientific contract và RAM/disk gate. Dry-run không chạy MI/RFE/k-search/final/Test. Cell 10 chỉ in skip; chưa có model hoặc kết quả Test là đúng.

Không tạo experiment ID dạng RUN_MI_... và không cần copy xác nhận từ experiment cũ.

## 3. Chuyển sang full trong cùng session

Sau dry-run, giữ nguyên tên run đã in ở Cell 8. Thêm một cell ngay trước Cell 9 nếu muốn chuyển mà không chạy setup lại:

```python
MODE = "full_train"
CONFIRMATION = "RUN PROVISIONAL " + SPEC.run_name
REQUEST.update(mode=MODE, confirmation=CONFIRMATION)
PREFLIGHT = None
print("Run:", SPEC.run_name)
print("Confirmation:", CONFIRMATION)
```

Chạy lại **Cell 9**, rồi **Cell 10 → 11 → 12**.

Hoặc sửa MODE trong Cell 8 thành full_train, điền đúng RUN_NAME đã chọn, rồi chạy Cell 8–12. Khi CONFIRMATION để trống, Cell 9 yêu cầu nhập đúng câu RUN PROVISIONAL <run_name thực tế> đã hiển thị. Không nhập nguyên dấu <...>.

Cell 10 chạy:
1. Train-only scaler + MI một call 380 feature, seed42.
2. Top100 + Train-only scaler + RF-RFE step1 riêng cho model.
3. k=1..100, classifier fit Train, score Validation Accuracy.
4. Freeze k/features; tie chọn k nhỏ hơn.
5. Final scaler + classifier fit Train+Validation.
6. Test được mở/predict một lần.
7. Seal manifest, copy/hash artifact và publish thư mục hoàn tất.

**RF classifier max_depth=20**, n_estimators300, split3. RFE RF depth10; SVM branch dùng RF-RFE depth15. Không nhầm các estimator này.

MI cũ per-feature không được dùng lại vì thuật toán khác mẫu. Full experiment sẽ lâu hơn Stage 4 MI: ngoài RFE còn có 100 classifier fits để chọn k. Kernel SVM với C10/max_iter20000/cache10000MB có thể rất lâu và cần nhiều RAM; resource gate không dự báo thời gian và không đảm bảo fit chắc chắn đủ RAM. Nếu FAIL/BLOCKED, không tự giảm dữ liệu, số cây, k grid hoặc RFE step.

## 4. Chạy đủ bốn cấu hình, riêng từng model

Sau khi run hiện tại kết thúc, lặp Cell 8–12; đổi FW/FS và để RUN_NAME trống cho experiment mới:

| CONFIG | FW | FS |
|---|---|---|
| FW256_FS8 | "256" | 8 |
| FW256_FS16 | "256" | 16 |
| FW512_FS8 | "512" | 8 |
| FW512_FS16 | "512" | 16 |

Chọn EXPERIMENT="mi_rfe_rf" cho RF hoặc EXPERIMENT="mi_rfe_svm" cho SVM. Mỗi model/config có thư mục riêng. Không tự động chạy cả tám experiment khi bấm một cell. XGBoost chưa có ranking contract nên không có lựa chọn full.

## 5. Kết quả nằm ở đâu?

```text
MyDrive/experiment_results/mi_rfe_rf_FW256_FS8_run01/
MyDrive/experiment_results/mi_rfe_svm_FW256_FS8_run01/
```

Tên runNN thực tế được in ở Cell 8. Các file giống ảnh:
artifact_manifest.json, events.jsonl, features.json, final_scaler.joblib,
k_search_results.json, mi_feature_scores.csv, mi_top100.json, model.joblib,
preflight_audit.json, progress.json, result.json, rfe_feature_ranking.csv,
selection_frozen.json. Có thêm identity, hyperparameters, resources, state và COMPLETE để verify.

Khi đang chạy hoặc bị ngắt:
`MyDrive/checkpoints/<run_name>.incomplete/`.
Đừng coi checkpoint là completed result chỉ vì đã có MI hoặc model file.

Log tổng hợp chính thức:
`MyDrive/code_CICFLOW/logs/ALL_EXECUTION_RESULTS.log`.

Log từng tác vụ:
`MyDrive/code_CICFLOW/logs/colab_reference/<run_name>/`.
Setup/tests/metadata chưa chọn experiment được ghi dưới setup/. Terminal vẫn hiển thị stdout/stderr. Log append, native exit code và program status ghi riêng; exit0 không đủ để kết luận scientific PASS.

Cell 11 phải verify PASS. Cell 12 tổng hợp JSON đã lưu, không đọc hoặc đánh giá Test lại. SVM convergence warning được lưu rõ; hoàn tất kỹ thuật không biến nó thành hội tụ hoặc official reproduction.

## 6. Resume sau ngắt session

Chạy lại Cell 1–7 bằng đúng source và environment cũ. Cell 8:

```python
EXPERIMENT = "mi_rfe_rf"
FW = "256"
FS = 8
RUN_NAME = "mi_rfe_rf_FW256_FS8_run01"  # đúng tên bạn đang tiếp tục
MODE = "resume"
CONFIRMATION = "RUN PROVISIONAL mi_rfe_rf_FW256_FS8_run01"
GIT_COMMIT = "NOT_CAPTURED"  # phải giống identity ban đầu
```

Chạy Cell 8–12. Resume không tạo Run ID mới. RFE/k-search dùng checkpoint đã lưu, fit đang dở bị mất. MI một call không có checkpoint giữa các feature; nếu chưa publish MI hoàn chỉnh thì có thể tính lại MI.

Nếu Test intent đã tồn tại nhưng chưa có test_result.json, hoặc có artifact partial/final sealing không đầy đủ: BLOCKED/UNSUPPORTED, không tự xóa để chạy tiếp. Giữ checkpoint và dùng run mới sau khi xem lỗi. Completed result được verify/skip; không chạy Test lần hai.

Để chỉ verify: nhập đúng Run Name và GIT_COMMIT như trên, MODE="verify"; Cell 9–12 không fit. Verify không bị chặn bởi RAM cần cho training.

## 7. Unit/regression tests trên Windows, nếu cần

Cell 5 là cách chính, đã tự lưu cả stdout/stderr và test counts. Muốn tự chạy trên Windows bằng wrapper chung:

```powershell
& 'D:\COMPACT-MI-RFE\code_CICFLOW\shared\Invoke-CicflowLogged.ps1' -Stage 'Colab reference' -Task 'Unit regression' -Configuration 'SYNTHETIC' -RunId 'REFERENCE_FIXTURE' -WorkingDirectory 'D:\COMPACT-MI-RFE\code_CICFLOW' -Command 'python -m unittest discover -s colab -p test_cicflow_reference_pipeline.py -v; exit $LASTEXITCODE'
```

Lệnh dùng đúng tham số của shared/Invoke-CicflowLogged.ps1 hiện có. Python phải trỏ tới environment đã cài dependencies. Codex chưa chạy lệnh/tests này.

## 8. Tình trạng khoa học

PROVISIONAL_RESEARCH. Đã viết regression so MI score/ranking của một call đủ380 với reference expression và RFE với sklearn trên fixture; **chưa có kết quả test do Codex chạy**. Fixture PASS khi bạn chạy không chứng minh đầy đủ score của dataset khác, software version khác hoặc official CTU-13 reproduction. Dataset label provenance và capture/session independence chưa được giải quyết bởi đổi notebook.
