# Stage 8 — Hướng dẫn chạy XGBoost trên Google Colab

Notebook: `notebooks/STAGE8_XGBOOST_COLAB.ipynb`

Codex chỉ viết source/notebook/tests. Các bước dưới đây do người dùng tự thực hiện. Không chạy nhiều cấu hình đồng thời và không dùng **Run all** để bỏ qua các gate.

## 1. Chuẩn bị Google Drive

Giữ dataset đã chuẩn bị, không tải hoặc tạo lại:

```text
MyDrive/COMPACT-MI-RFE/
├── CICIoMT24/
│   ├── WINDOWS_CPLUS/
│   └── CSV_SPLIT_CPLUS/
├── experiment_results/
├── environment/
├── approvals/
└── technical_probes/
```

Notebook chỉ đọc `WINDOWS_CPLUS` và manifest C+. Nếu dataset đang ở vị trí khác, sửa duy nhất `WINDOW_ROOT` tại Cell 5. Không copy lại dữ liệu nếu `window_run_summary.json`, manifest và validation hiện có đã hợp lệ.

## 2. Chạy notebook theo thứ tự

1. **Cell 1:** xem CPU/GPU/RAM/disk. GPU chỉ là inventory, không tự được chọn.
2. **Cell 2:** mount Drive.
3. **Cell 3:** nhập commit 40 ký tự đã review; notebook checkout detached và yêu cầu working tree sạch.
4. **Cell 4:** ghi SHA256 Stage 8 và training core.
5. **Cell 5:** trỏ đến dataset hiện có.
6. **Cell 6:** cài/check profile Stage 8, gồm `xgboost==3.4.1`. Nếu package thay đổi trong runtime, restart runtime rồi chạy lại từ Cell 1.
7. **Cell 7:** chạy unit tests và data validation. Đây là bước người dùng tự chạy; Codex chưa chạy.
8. **Cell 8:** chọn đúng một experiment:
   - `baseline_xgb`
   - `mi_rfe_xgb`
9. **Cell 9:** chọn một FW/FS và một `RUN_NAME` mới.
10. **Cell 10:** chạy dry preflight. Kết quả phải `PASS`, với `model_fit_calls=0` và `test_feature_rows_read=0`.
11. **Cell 11:** đọc budget, bật `I_ACCEPT_RESOURCE_BUDGET=True` và nhập đúng confirmation tại prompt tương tác. Token phiên chỉ được tạo trong RAM; **Run all** sẽ dừng ở prompt và không thể dùng một phrase đã lưu sẵn để tự mở training.
12. **Cell 12:** đặt `MODE='full_train'` rồi chạy riêng cell này. Không chạy lại nếu đã có output hoặc `.incomplete`.
13. **Cell 13:** chỉ verify artifacts; không load model để predict Test.
14. **Cell 14:** chỉ hiển thị metrics đã lưu trong `result.json`.

Mặc định:

```python
MODE = "dry_run"
BACKEND = "cpu"
I_ACCEPT_RESOURCE_BUDGET = False
CONFIRMATION = ""
```

Vì vậy mở notebook hoặc Run all với giá trị mặc định không thể bắt đầu training.

## 3. CPU và GPU

CPU là backend tham chiếu:

```python
tree_method = "hist"
device = "cpu"
```

CUDA chỉ được dùng khi người dùng đặt:

```python
BACKEND = "cuda"
```

Module đồng thời yêu cầu XGBoost build có `USE_CUDA`. GPU không thay đổi `n_estimators=200`, `max_depth=10` hoặc `min_child_weight=3`. Kết quả CPU và GPU không được tuyên bố bitwise-equivalent; backend được lưu trong mọi result.

## 4. Gate riêng cho MI-RFE + XGBoost

CTU-13 không có nhánh XGBoost độc lập. Vì vậy file:

```text
configs/stage8_xgboost_policy.BLOCKED.json
```

không thể mở official `mi_rfe_xgb`. Chính sách đề xuất là tái sử dụng đúng RF-RFE của nhánh MI-RFE+RF:

```python
n_estimators=100
max_depth=10
min_samples_split=2
max_features="sqrt"
class_weight="balanced"
random_state=42
n_jobs=-1
```

Sau khi người dùng phê duyệt riêng:

1. Copy template thành một file JSON mới trong Drive hoặc repository.
2. Giữ nguyên `schema`, `policy_id` và `rf_rfe_parameters`.
3. Đổi `status` thành `APPROVED`.
4. Đổi `approved` thành `true`.
5. Ghi reviewer và UTC nếu cần.
6. Trỏ `RFE_POLICY` ở Cell 9 tới file mới.

Không sửa template BLOCKED tại chỗ. Không dùng best-k RF/SVM cho XGBoost và không thay RF-RFE bằng XGBoost importance.

## 5. Khi run bị ngắt

- Kết quả đang ghi có hậu tố `.incomplete`.
- `FAILED.json` hoặc file `*.supervisor.json` cho biết nguyên nhân.
- Không tự retry.
- Không xóa `TEST_CLAIM.json`.
- Nếu Test đã claim, bắt buộc review thủ công; không chạy Test lại.
- Runner hiện ghi rõ `resume_supported=false`. Checkpoint chỉ là bằng chứng stage, không phải cam kết resume.

## 6. Artifact cần gửi để review

Trong:

```text
MyDrive/COMPACT-MI-RFE/experiment_results/<RUN_NAME>/
```

gửi tối thiểu:

- `preflight_audit.json`
- `run_identity.json`
- `progress.json`
- `events.jsonl`
- `selection_frozen.json`
- `hyperparameters.json`
- `booster_config.json`
- `result.json`
- `RUN_VALIDATED.json`
- `artifact_manifest.json`
- `TEST_CLAIM.json`

Với `mi_rfe_xgb`, gửi thêm:

- `mi_top100.json`
- `mi_feature_scores.csv`
- `rfe_feature_ranking.csv`
- `k_search_results.json`

Nếu lỗi, gửi:

- `FAILED.json`
- `<RUN_NAME>.supervisor.json`
- Toàn bộ output Cell 10–12.

Không cần gửi lại CSV window hoặc cache `.npy`.
