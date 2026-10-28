# CICFLOW Stage 8 — chạy 4 cấu hình XGBoost trong một session Colab

Notebook: [CICFLOW_STAGE8_XGBOOST_COLAB.ipynb](D:/COMPACT-MI-RFE/code_CICFLOW/stage8_xgboost/notebooks/CICFLOW_STAGE8_XGBOOST_COLAB.ipynb).

**Cell 8 đặt MODE="full"; Cell 9 xác nhận ALL4 một lần và chạy tuần tự cả bốn cấu hình.**
Mặc định MODE="preflight", BACKEND="cpu", RUN_TAG="run01".

Nhánh thực thi là **baseline_xgb**, đủ 380 feature. Source notebook tham chiếu
chặn MI-RFE-XGB và policy RF-RFE hiện có chưa được duyệt. Không lấy feature,
ranking, best-k, scaler hay model từ các thư mục mi_rfe_rf_*.

XGBoost classifier giữ đúng **200 cây, max_depth=10, min_child_weight=3**,
binary:logistic, logloss, seed42, n_jobs=-1, hist. Backend do người dùng chọn,
không tự fallback. Những tham số không được source gán giữ default của
XGBoost3.4.1 và được ghi từ native Booster configuration trước và sau fit.

```text
MyDrive/experiment_results/
├── baseline_xgb_FW256_FS8_run01/
├── baseline_xgb_FW256_FS16_run01/
├── baseline_xgb_FW512_FS8_run01/
└── baseline_xgb_FW512_FS16_run01/
```

Mỗi cấu hình có worker riêng, checkpoint mỗi10 round, Validation và final refit
đủ200 round, Test một lần/cấu hình. Không fit đồng thời bốn cấu hình, không gộp
dataset, không sampling. Default lỗi một cấu hình vẫn tiếp tục cấu hình khác;
STOP_ON_ERROR=True sẽ dừng queue khi gặp lỗi. Ctrl+C luôn dừng batch.

Để resume: cùng commit, source, environment, backend, budget và RUN_TAG;
Cell8 MODE="resume", rồi Cell9. Run hoàn tất được verify/skip, chưa bắt đầu
được bắt đầu, run dở dùng checkpoint khớp identity. Test đang mơ hồ sẽ BLOCKED.
MODE="verify" chỉ kiểm tra artifact, không fit hoặc đọc Test.

Scientific status luôn **PROVISIONAL_RESEARCH / OFFICIAL_BLOCKED**; capture/
session UNKNOWN, nhãn PROVISIONAL_FOLDER. Không có kết quả training mới được
khẳng định bởi việc viết source. Các tests mới **chưa được Codex chạy**.

Xem reports/STAGE8_COLAB_EXECUTION_GUIDE.md để push source, mở notebook,
chạy từng cell, kiểm tra output và gửi lại artifact. Xem
reports/XGBOOST_REFERENCE_AUDIT.md để hiểu vì sao dùng baseline.
