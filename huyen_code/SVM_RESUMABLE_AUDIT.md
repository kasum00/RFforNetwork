# Audit trước khi triển khai MI-RFE SVM nhiều phiên

Ngày: 23/10/2026. Chỉ đọc source; chưa thực thi Python, tests, dataset audit hoặc training.

Workspace thực tế là `D:\COMPACT-MI-RFE`; đường dẫn `C:\Users\LAB_LEE\Desktop\RFforNetwork_GITHUB` trong prompt là vị trí triển khai thứ hai, dùng trong hướng dẫn lệnh.

## Hợp đồng khoa học đã đối chiếu

Đã đọc nguyên văn `CODE_ctu13/[[CTU13]8_experiment2_svm.py` và `[[CTU13]9_ex2_rf.py`. Script SVM tự ghi tên cũ `scenario2_mi_rfe_svm.py` trong docstring; không tìm thấy file riêng tên đó trong repository.

| Thành phần | CTU-13 MI-RFE SVM | CICIoMT2024 đang dùng | Quyết định |
|---|---|---|---|
| Chuẩn bị X | numeric float32, Inf/NaN→0; feature_names lấy trước numeric filter | whitelist đúng 380 tên, float32, Inf/NaN→0 | Giữ whitelist đã duyệt, không phục hồi lỗi tên cột. |
| MI | scaler fit Train; mutual_info_classif seed42; Top100 theo pandas sort | cùng phép tính trong select_mi | Gọi lại chính hàm này khi chưa có checkpoint. |
| Top100 | lấy từ raw X, scaler mới fit Train | scale_top100 | Giữ nguyên cả thứ tự lẫn scaler. |
| RF-RFE SVM | 100 cây, depth15, balanced, n_jobs=-1, seed42; sklearn RFE target1/step1 | rank_rfe(..., 'svm') | Không dùng custom RFE hoặc RF depth10. |
| RF-RFE của script RF | 100 cây, depth10, split2, sqrt | nhánh riêng RF_RFE_PARAMS | Chỉ đối chiếu, không dùng trong SVM hoặc sửa RF. |
| SVC | RBF C10 gamma scale probability=False balanced max_iter20000 shrinking=True cache10000 seed42 | COMPACT_SVM_PARAMS | Giữ nguyên. |
| k | 1..100; chỉ cập nhật best khi accuracy lớn hơn | search_k; hòa giữ k nhỏ | Checkpoint từng k chỉ thay điều phối. |
| Final | scaler mới và SVC fit Train+Val trên prefix ranking đã khóa | fit_final_classifier | Không fit lại MI/RFE. |
| Test | script gốc đọc/transform sớm nhưng không fit vào Test | official runner nạp muộn sau final fit | Giữ biện pháp nạp muộn đã duyệt. |
| Warnings/metrics | source suppress warnings, Accuracy/Precision/Recall/F1 | lưu warnings, thêm Macro-F1/Balanced Accuracy/report | Giữ báo cáo mở rộng, không đổi tiêu chí k. |

## Giới hạn hệ thống cũ

`CheckpointSink` lưu MI scores, ranking và từng k JSON nhưng không cung cấp loader resume hoặc đầy đủ scaler/parent fingerprints. `resume_supported=False`; `publish_completed_checkpoint` chỉ phục hồi publication khi đã run_validated. Không được tự coi checkpoint cũ là checkpoint schema mới.

`authorize_run` khóa identity/source/data/runtime và chặn output legacy đã tồn tại. Supervisor cũ không retry/resume. Phần mở rộng mới phải yêu cầu policy resume và session ID rõ ràng, giữ gate cũ cho Baseline/RF/legacy; không chuyển legacy .incomplete thành run resumable.

`SVC.fit`, MI hoặc sklearn RFE đang chạy không có checkpoint optimizer được hỗ trợ ở đây. Chỉ tái sử dụng stage đã commit đủ artifact và SHA256. Fit dở có thể phải chạy lại trong một phiên được phê duyệt mới; Test đã bắt đầu thì không được tự đánh giá lại.
