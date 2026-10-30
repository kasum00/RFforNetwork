# Phương án B — CICFLOW MI–RF-RFE step2–SVM

Dùng **đúng policy run_phuong_anB.py do người dùng dán**:

~~~python
COARSE = (10, 20, 30, 50, 75, 100)
RADIUS = 5
RFE_STEP = 2
~~~

RFE_STEP chỉ điều khiển số feature loại bỏ ở mỗi vòng RF-RFE. Coarse SVM luôn là
sáu mức k ở trên. Chọn coarse_best_k bằng Validation Accuracy cao nhất, hòa k nhỏ
nhất; fine xét k chưa thử trong [max(1,best-5),min(100,best+5)] theo thứ tự tăng dần.
Sau fine chọn best theo cùng tiêu chí. Không fit lại k có checkpoint hợp lệ cùng run.
Không còn policy thay thế hoặc yêu cầu người dùng chốt policy mới.

Dataset duy nhất: **CICFLOWMETER_FEATURE_CSV**, giữ nguyên splits/schema/labels/
preprocessing của A1. Không nhập dữ liệu CICIoMT2024 hoặc checkpoint A1.
Chỉ full Train/Validation selection; không final refit và không Test.
**EXPERIMENTAL_NON_OFFICIAL / OFFICIAL_BLOCKED**.

Notebook đầy đủ: notebooks/PHUONG_AN_B_MI_RFE_SVM_STEP2_COLAB.ipynb ở repository root.
Mặc định EXECUTE_QUEUE=False; source B phải push thành commit mới trước khi dùng Colab.

## Giữ nguyên phần khoa học

- MI một lần mutual_info_classif trên Train380, StandardScaler fit Train, seed42,
  default dense discrete_features=auto/n_neighbors=3, pandas ranking, Top100.
- Numeric float64 cleaning NaN/Inf→0 rồi float32, đúng thứ tự cột/hàng như A1.
- Scaler Train Top100 mới, Validation chỉ transform; layout float32 arithmetic gốc,
  C-order float64 ở SVC boundary.
- RF estimator RFE: n_estimators=100,max_depth=15,class_weight=balanced,
  random_state=42,n_jobs=-1; các default khác theo helper/environment đã khóa.
- SVM RBF: C=10,max_iter=20000,gamma=scale,balanced,shrinking=True,
  probability=False,cache_size=10000,random_state=42.
- Policy coarse/fine và tie-break từ B gốc. Thay đổi thuật toán so với B gốc chỉ
  là RF-RFE step1→step2; adapter dataset chuyển sang CICFLOW theo yêu cầu.

RFE step2 có thể đổi ranking so với step1; native rank groups giữ riêng, total order
trong nhóm dùng removal importance rồi MI position như A1. Không tuyên bố kết quả
ranking/Accuracy tương đương. RF Top100 step2 có 50 elimination + 1 final fit =51,
coarse SVM=6, fine=5 hoặc10 theo anchor, tổng SVM=11 hoặc16 nếu hoàn tất đúng lịch.
Đây là số fit theo thuật toán; không phải tốc độ benchmark hoặc bảo đảm optimum
của exhaustive k=1..100.

## Namespace, checkpoint và logging

~~~text
MyDrive/experiment_results/phuong_anB_step2/mi_rfe_svm/<CONFIG>/RUN_B2_01/
/content/mi_rfe_svm_b_step2/phuong_anB_step2/mi_rfe_svm/<CONFIG>/RUN_B2_01/
~~~

Algorithm: MI_RFE_SVM_B_STEP2_COARSE6_RADIUS5_V2. Format: phuong_anB_step2.selection.v2.
Identity bind dataset fingerprint, source/Git/environment, policy đủ six-grid/radius,
RFE step2, config/run ID và parameters. Checkpoint step1, A1 hoặc bản nháp grid cách2
không tương thích, bị BLOCKED; không migrate/overwrite ngầm.

Mỗi RF elimination và SVM candidate được lưu sau hoàn thành; local atomic record và
Drive COMMITTED/readback tách riêng. Candidate history là sparse ordinal/phase/k,
không yêu cầu fit toàn bộ k1..100. Fine plan chốt từ sáu coarse scores và không trôi
vùng tìm kiếm sau khi fine tìm thấy điểm tốt hơn. Resume không fit lại candidate
đã xác minh, interrupted fit chưa có record phải tính lại; lease/source mismatch
vẫn BLOCKED. Không copy Parquet hoặc arrays lên checkpoints.

Notebook dùng tqdm.auto và heartbeat có CPU/RAM/current k/best score/elapsed/ETA đo từ
công việc đã hoàn thành, checkpoint time; không có fake mid-fit progress. Queue
hỗ trợ một config hoặc cả bốn tuần tự; verify sau mỗi selection complete.

Log Colab append tại MyDrive/code_CICFLOW/logs/ALL_EXECUTION_RESULTS.log; giữ native
exit, reported status và scientific status riêng. Lỗi lưu log có cảnh báo.

## Hướng dẫn và kiểm thử

- reports/COLAB_GUIDE.md: push, từng cell, unit test có logging, preflight/run/resume/verify.
- reports/SCIENTIFIC_CONTRACT.md: đối chiếu source B gốc, dataset và điều chỉnh step2.
- reports/RESUME_PROTOCOL.md: SHA/identity/lease và interruption.
- reports/IMPLEMENTATION_REPORT.md: file mới/sửa, các giới hạn chưa được thực thi.

Tests đã viết cho pure-policy gốc, Train-only MI/scaler, sklearn step2 ranking,
no Test, dedup, pause/resume coarse/fine, identity/integrity, publication/non-overwrite,
monitor và native logging. Chưa chạy Python, tests, preflight, Git, pip, training
hoặc benchmark. Không tuyên bố PASS thực tế.
