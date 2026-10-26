# Stage 8 — Audit hyperparameters và triển khai XGBoost cho Google Colab

Ngày triển khai source: 26/10/2026.

**Trạng thái: CODE/DOCS/TESTS WRITTEN — NOT EXECUTED.** Codex không chạy Python, unit tests, benchmark, training, Colab, package install, Git commit hoặc Git push. Không sửa dataset và official results RF/SVM cũ.

Theo trạng thái đầu vào do người dùng xác nhận, MI-RFE+Random Forest đã chạy 4/4 cấu hình. Stage 8 không chạy lại hoặc sửa các artifact đó; bảng audit chỉ xác minh contract source đang dùng.

## 1. Kết luận

- RF final đúng `300/20/3`.
- Baseline SVM và MI-RFE+SVM đúng `C=10.0`, `max_iter=20000`, đồng thời giữ khác biệt source về `probability` và `class_weight`.
- RF-RFE nhánh RF và nhánh SVM là hai estimator riêng, không bị đồng nhất với RF final.
- Repository trước Stage 8 chưa có official XGBoost path.
- Đã thêm `baseline_xgb` và `mi_rfe_xgb` bằng adapter riêng, không sửa đường chạy RF/SVM.
- XGBoost final khóa `n_estimators=200`, `max_depth=10`, `min_child_weight=3`.
- `mi_rfe_xgb` mặc định **BLOCKED** vì CTU-13 không có RF-RFE policy riêng. Code chỉ đề xuất tái sử dụng policy nhánh MI-RFE+RF sau phê duyệt JSON riêng.
- Notebook mặc định `MODE="dry_run"`; CPU là backend tham chiếu.

## 2. Audit siêu tham số

Các dòng của `ciciomt24_training.py` là vị trí trong snapshot audit; symbol là định danh bền vững nếu dòng dịch chuyển.

| Model | Vai trò | Tham số hiện tại | Yêu cầu | Kết quả | File/dòng hoặc symbol |
|---|---|---|---|---|---|
| SVM Baseline | Final 380 feature | C=10, max_iter=20000, RBF, gamma=scale, probability=True, không class_weight | C=10, max_iter=20000; giữ source extras | PASS | `huyen_code/ciciomt24_training.py:45-47`, `make_model` 332-337 |
| SVM compact | k-search/final MI-RFE+SVM | C=10, max_iter=20000, probability=False, balanced | C=10, max_iter=20000; không đồng nhất Baseline | PASS | `ciciomt24_training.py:48-50`, `make_model` |
| RF-RFE SVM | Ranking Top100 nhánh SVM | 100 cây, depth=15, balanced, n_jobs=-1, seed=42 | Giữ riêng CTU-13 | PASS | `ciciomt24_training.py:51-53`, `rank_rfe` 339-345 |
| RF-RFE RF | Ranking Top100 nhánh RF | 100 cây, depth=10, min_split=2, sqrt, balanced, seed=42 | Không dùng RF final 300/20/3 | PASS | `ciciomt24_training.py:54-56`, `rank_rfe` |
| RF final | k-search/final MI-RFE+RF | 300 cây, depth=20, min_split=3, sqrt, balanced | 300/20/3 | PASS | `ciciomt24_training.py:57-59`, `make_model` |
| XGBoost final | Baseline và compact | Trước Stage 8: chưa có | 200 cây, depth=10, min_child_weight=3 | FIXED/ADDED | `huyen_code/ciciomt24_xgboost.py::XGB_REQUIRED_PARAMS`, `make_xgb` |
| RF-RFE XGBoost | Ranking Top100 extension | CTU-13 không có policy | Không tự giả định; đề xuất reuse RF branch | BLOCKED_PENDING_APPROVAL | `ciciomt24_xgboost.py::load_rfe_policy`, `select_mi_rfe_xgb` |
| Colab legacy launcher | Điều phối RF/SVM cũ | Không tạo estimator; delegate core | Không làm đổi old paths | PASS | `colab/experiment_launcher.py:39-44,144-208` |
| Notebook Stage 8 | Điều phối XGBoost | Một experiment/config, dry-run mặc định | 14 bước, approval rõ ràng | WRITTEN, NOT RUN | `notebooks/STAGE8_XGBOOST_COLAB.ipynb`, Cell 1–14 |

`source_audit` trong training core tiếp tục đối chiếu AST ba script classifier CTU-13 với các constant trên. Năm script CTU-13 không bị sửa.

## 3. Mismatch và thay đổi

Không có mismatch đối với RF/SVM bắt buộc, vì vậy không sửa `RF_PARAMS`, `BASELINE_SVM_PARAMS`, `COMPACT_SVM_PARAMS`, `SVM_RFE_PARAMS` hoặc `RF_RFE_PARAMS`.

Khoảng trống duy nhất là repository chưa có official XGBoost runner. Stage 8 thêm module mới thay vì chèn XGBoost vào core RF/SVM:

```text
huyen_code/ciciomt24_xgboost.py
```

Final XGBoost effective policy:

```python
n_estimators=200
max_depth=10
min_child_weight=3
objective="binary:logistic"
eval_metric="logloss"
random_state=42
n_jobs=-1
tree_method="hist"
device="cpu" hoặc "cuda" do người dùng chọn
```

Không tự bật class weighting, `scale_pos_weight` khác 1, sampling hoặc early stopping. Mỗi run lưu `get_params()`, `get_xgb_params()`, booster config, backend và XGBoost version.

## 4. Pipeline XGBoost

Baseline:

```text
Train 380 → scaler fit Train → XGB Train → báo Validation, không tuning
→ freeze 380 → scaler + XGB mới fit Train+Validation
→ durable Test claim → load/evaluate Test đúng một lần
```

MI-RFE+XGBoost:

```text
Train 380 → MI Top100 Train-only → scaler Top100 Train-only
→ RF-RFE step=1 đầy đủ → XGB k=1..100 Train/Validation
→ Validation Accuracy cao nhất, hòa lấy k nhỏ
→ freeze ranking/k/features → final refit Train+Validation
→ durable Test claim → Test đúng một lần
```

Không lấy best-k RF/SVM và không dùng XGBoost importance thay RF-RFE.

## 5. Leakage và integrity guard

- Tái sử dụng C+ và `WINDOWS_CPLUS`; không tạo lại split/window.
- `WindowStore` xác minh schema 380 feature, inventory và provenance.
- Metadata/ID/provenance không vào X.
- MI/RFE chỉ fit Train; Validation chọn k.
- Final refit sau feature freeze.
- `TEST_CLAIM.json` được ghi trước lần đọc Test đầu tiên.
- Test evaluation count phải bằng 1.
- Run có Test claim nhưng chưa validated không tự retry.
- Output hoặc `.incomplete` có sẵn bị từ chối.
- Result ghi `SCENARIO_PROXY` và `capture_session_independence=NOT_ESTABLISHED`.

## 6. Resource và checkpoint

Supervisor theo dõi process-tree RSS, elapsed time, RAM available, disk và heartbeat. Vượt budget sẽ dừng worker, ghi `<RUN_NAME>.supervisor.json`, không giảm dữ liệu và không retry.

Checkpoint hiện chỉ là bằng chứng stage:

- `train_validation_loaded`
- `selection_frozen`
- `final_refit_completed`
- `TEST_CLAIM`
- `RUN_VALIDATED`

`resume_supported=false`; không tuyên bố resume XGBoost/RFE/k-search.

## 7. Tests đã viết, chưa chạy

`huyen_code/test_ciciomt24_xgboost.py` kiểm tra RF 300/20/3, SVM 10/20000, khác biệt hai SVM, tách riêng các RF-RFE, XGB 200/10/3, policy blocked, k1..100/tie nhỏ, final refit, Test-once, preflight không đọc Test, resource fail trước spawn, không overwrite, không sampling và old RF/SVM không bị ảnh hưởng.

Codex **không chạy** tests.

Lệnh PowerShell người dùng tự chạy:

```powershell
Set-Location 'D:\COMPACT-MI-RFE'
$env:PYTHONPATH = 'D:\COMPACT-MI-RFE;D:\COMPACT-MI-RFE\huyen_code'
python -m unittest -v huyen_code.test_ciciomt24_xgboost
```

Regression đề xuất:

```powershell
python -m unittest -v huyen_code.test_ciciomt24_training huyen_code.test_ciciomt24_xgboost colab.test_stage6_colab colab.test_stage73b_scientific_integrity
```

## 8. File tạo/sửa

Tạo mới:

- `D:\COMPACT-MI-RFE\huyen_code\ciciomt24_xgboost.py`
- `D:\COMPACT-MI-RFE\huyen_code\test_ciciomt24_xgboost.py`
- `D:\COMPACT-MI-RFE\huyen_code\STAGE8_REPORT.md`
- `D:\COMPACT-MI-RFE\huyen_code\STAGE8_COLAB_GUIDE.md`
- `D:\COMPACT-MI-RFE\notebooks\STAGE8_XGBOOST_COLAB.ipynb`
- `D:\COMPACT-MI-RFE\configs\stage8_xgboost.json`
- `D:\COMPACT-MI-RFE\configs\stage8_xgboost_policy.BLOCKED.json`
- `D:\COMPACT-MI-RFE\configs\stage8_xgboost_budget.BLOCKED.json`
- `D:\COMPACT-MI-RFE\requirements-stage8-colab.txt`

Chỉnh sửa:

- `D:\COMPACT-MI-RFE\README.md` — thêm Stage 8 và link tài liệu.

Không sửa CTU-13, dataset, split/window, checkpoint hoặc results cũ.

## 9. Git — người dùng tự thực hiện

```powershell
Set-Location 'D:\COMPACT-MI-RFE'
git add -- README.md requirements-stage8-colab.txt configs/stage8_xgboost.json configs/stage8_xgboost_policy.BLOCKED.json configs/stage8_xgboost_budget.BLOCKED.json huyen_code/ciciomt24_xgboost.py huyen_code/test_ciciomt24_xgboost.py huyen_code/STAGE8_REPORT.md huyen_code/STAGE8_COLAB_GUIDE.md notebooks/STAGE8_XGBOOST_COLAB.ipynb
git diff --cached --check
git status --short
git commit -m "Add guarded Stage 8 XGBoost Colab pipeline"
git push origin HEAD:main
```

Không dùng `git add .`.

## 10. Artifact cần gửi lại

Tối thiểu: `preflight_audit.json`, `progress.json`, `events.jsonl`, `selection_frozen.json`, `hyperparameters.json`, `booster_config.json`, `result.json`, `RUN_VALIDATED.json`, `artifact_manifest.json`, `TEST_CLAIM.json` và file `*.supervisor.json`.

Với MI-RFE-XGB gửi thêm `mi_top100.json`, `mi_feature_scores.csv`, `rfe_feature_ranking.csv`, `k_search_results.json`. Nếu lỗi gửi `FAILED.json` và output Cell 10–12. Không gửi CSV window/cache NPY.

## 11. Gate còn BLOCKED

1. Tests Stage 8 chưa chạy.
2. Chưa có clean Git commit chứa source mới.
3. Chưa kiểm chứng `xgboost==3.4.1` trên Colab thực.
4. Chưa chạy dataset/preflight bằng source mới.
5. Resource budget chưa được chấp nhận.
6. Chưa có exact confirmation.
7. `mi_rfe_xgb` chưa được duyệt RF-RFE policy.
8. GPU chưa được kiểm chứng; CPU là tham chiếu.
9. Chưa có official XGBoost result.

Ưu tiên kỹ thuật đầu tiên sau khi các gate PASS: `baseline_xgb/FW_512_FS_16`. Đây không phải lệnh tự chạy.

## 12. Tài liệu tham chiếu XGBoost

- https://xgboost.readthedocs.io/en/stable/install.html
- https://xgboost.readthedocs.io/en/stable/gpu/
- https://xgboost.readthedocs.io/en/stable/python/python_intro.html

CPU/GPU đều dùng `tree_method="hist"`; GPU dùng `device="cuda"`. Version được khóa để tránh default thay đổi.

## 13. Xác nhận phạm vi

Không chạy code/tests/training/benchmark, không cài package, không điều khiển Colab/Chrome, không commit/push, không sửa CTU-13, dataset, C+, window CSV hoặc kết quả cũ. Dừng sau source, tests, notebook, config và tài liệu.
