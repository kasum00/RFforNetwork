# Stage 7.3B — Cam kết toàn vẹn khoa học

## Baseline SVM

- X đúng 380 đặc trưng thống kê hành vi; metadata/provenance không vào X.
- NaN và ±Inf được thay bằng 0 trước khi fit; dữ liệu float32 trong lớp chuẩn bị.
- `StandardScaler` và SVC cuối fit trên Train+Validation.
- SVC: RBF, `C=10`, `gamma='scale'`, `probability=True`, `class_weight=None`, `shrinking=True`, `cache_size=10000`, `max_iter=20000`, seed 42.
- Không MI/RFE và không chọn tham số bằng Test.

## MI-RFE + SVM

- Mutual Information fit Train-only, lấy Top 100.
- Scaler Top100 fit Train, transform Validation.
- RF-RFE đầy đủ: `step=1`, `n_features_to_select=1`, RF 100 cây, depth 15, `class_weight='balanced'`, `n_jobs=-1`, seed 42.
- Duyệt đủ k=1..100; chọn Validation Accuracy cao nhất, hòa giữ k nhỏ hơn.
- Sau khi khóa feature/k, scaler và SVC cuối fit Train+Validation; MI/RFE không refit.
- SVC compact: RBF, `C=10`, `gamma='scale'`, `probability=False`, `class_weight='balanced'`, `shrinking=True`, `cache_size=10000`, `max_iter=20000`, seed 42.

## Quy tắc Test

`run_official` ghi `selection_frozen`, hoàn thành `final_refit_completed`, sau đó mới gọi `load_full(..., "test")`. Chỉ có một lời gọi `evaluate_final(fitted, test)`. Test chỉ báo cáo một lần; không được dùng để chọn feature, k, hyperparameter, cấu hình hay quyết định retry.

Artifact chính thức phải lưu Accuracy, Precision/Recall/F1 Attack, Macro-F1, Balanced Accuracy, confusion matrix `[Benign, Attack]`, classification report (bao gồm recall từng lớp), fit warnings, support-vector count, stage runtimes và peak RSS được lấy mẫu.

## Bảo vệ mới

- `ciciomt24_training.run_official` yêu cầu context Stage 7.3B cho cả hai SVM; `--confirm-full-training` đơn lẻ không còn đủ.
- CLI trực tiếp từ chối SVM và hướng sang supervisor.
- Random Forest không bị buộc dùng SVM authorization; thuật toán và đường chạy RF cũ được giữ.
- Approval gắn với source/data/runtime/budget cụ thể. File template trong repository cố ý `effective=false`.
- Output cũ, `.incomplete`, checkpoint hoặc publication cùng run name đều chặn run mới.

## Giới hạn khoa học

Nhãn Benign/Attack là nhãn thay thế theo scenario, không phải ground truth từng flow. Chia C+ loại flow embargo và ngăn source-row/window overlap theo thiết kế, nhưng không chứng minh capture hoặc session độc lập. Accuracy có thể bị chi phối bởi lớp Attack; luôn trình bày Macro-F1, Balanced Accuracy, recall và confusion matrix.
