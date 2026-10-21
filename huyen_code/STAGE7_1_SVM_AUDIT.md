# Stage 7.1 — SVM source audit & technical probe preparation

**Cập nhật 21/10/2026 theo log Colab người dùng gửi:** Cell 8 chạy unit test
Stage 7.1 **9/9 PASS**. Cell 9 preflight cho `baseline_svm/FW_256_FS_8`
**FAIL có chủ đích ở hai cổng SVM**: chưa có technical probe hợp lệ được chấp
nhận và chưa duyệt resource gate full SVM. Các kiểm tra RAM/disk, manifest và
source trong chính preflight đều PASS. **Chưa có output Cell 10/10A hoặc JSON
probe trong bằng chứng được gửi; chưa đủ cơ sở xin chạy full SVM Stage 7.2.**
Stage 6 từng có 20/20 PASS trong lượt trước; log lần này chỉ xác nhận
`unit_tests_current_code=PASS` qua preflight, không cung cấp toàn bộ output
Stage 6 tests. Không có benchmark hay training chính thức nào được chạy bởi
Codex. Chờ kết quả probe theo
[`SVM_TECHNICAL_PROBE.md`](SVM_TECHNICAL_PROBE.md).

## 1. Audit hai SVM tham chiếu

| Nội dung | Baseline — `7_senario1_svm.py` | MI-RFE-SVM — `[[CTU13]8_experiment2_svm.py` |
| --- | --- | --- |
| Feature | Toàn bộ cột numeric sau danh sách loại trừ; bản CICIoMT dùng whitelist đúng 380 | Scaler Train → MI Top100 Train → scaler Top100 Train → RF-RFE Train `step=1`, đích 1 feature → ranking |
| SVM | `SVC` RBF, C=10, gamma=`scale`, probability=True, cache_size=10000 MB, shrinking=True, max_iter=20000, random_state=42, **không đặt class_weight** | Cùng RBF/C/gamma/cache/shrinking/max_iter; probability=False, class_weight=`balanced`, random_state=42 ở constructor |
| RF trong RFE | Không có | 100 cây, max_depth=15, class_weight=`balanced`, n_jobs=-1, random_state=42. Không đặt min_samples_split/max_features riêng trong script này |
| Chọn k | Không có | Thử k=1..100 theo thứ tự tăng; cập nhật khi Validation Accuracy **lớn hơn** best cũ, hòa giữ k nhỏ nhất |
| Fit cuối | Scaler + SVM fit Train+Validation một lần, Test predict | Đóng băng feature/best_k; scaler + SVM mới fit Train+Validation trên cột đã khóa; **không fit lại MI/RFE**; Test predict một lần |

Tham số project C=10/max_iter=20000 khớp cả hai script. Không đồng nhất
`probability` và `class_weight`: so sánh Baseline/Compact của source không thể
quy toàn bộ chênh lệch riêng cho MI-RFE. Source Baseline chỉ hiển thị đầy đủ
bốn cấu hình; trong source MI-RFE-SVM, `CONFIGS` hiện chỉ bật FW512/FS8.
Runner CICIoMT cho phép chọn từng cấu hình mà vẫn giữ cùng phương pháp.

## 2. Đối chiếu source, runner và chống leakage

| Điểm kiểm tra | Hiện trạng sau Stage 7.1 |
| --- | --- |
| Train/Val/Test đã tách trước window | Thiết kế C+ và Stage 4.2 đã tạo sẵn; Stage 7.1 không sửa hay quét lại dữ liệu |
| X/y | `prepare_xy` trong runner chỉ lấy 380 feature hành vi theo thứ tự cố định; loại metadata/provenance, thay NaN/Inf bằng 0; y từ Label nhị phân proxy |
| Scaler/MI/RFE lúc chọn model | `select_mi`, `scale_top100`, `rank_rfe` fit Train; Validation chỉ được transform/predict để chọn k |
| Best k / refit | `search_k` strict `>`; `fit_final_classifier` dùng feature đã khóa, fit scaler/model trên Train+Validation, không fit lại MI/RFE |
| Test | `run_official` load Test sau `final_refit_completed`, gọi `evaluate_final` một lần; adapter Stage 6 checkpoint model/scaler trước mốc đọc Test |
| Probe Stage 6 cũ | `run_technical_svm_probe` trước đây gọi `benchmark_worker`, đọc cả Train và Validation, đồng thời đo RF. Không phù hợp yêu cầu Train-only Stage 7.1; **không sửa/xóa benchmark Stage 5 cũ** |
| Probe Stage 7.1 | Adapter nay gọi `ciciomt24_svm_probe.py`, chỉ `load_sample(..., "train", ...)`, split nội bộ từ Train; không đọc CSV Validation/Test |
| Full SVM gate | Preflight yêu cầu JSON probe đúng schema/config/experiment/code/window summary, đủ worker cho mọi sample size, SVC hội tụ và 0 Validation/Test rows. Dù probe PASS, `svm_full_resource_gate_reviewed=False` cố ý giữ full SVM **FAIL** cho đến quyết định Stage 7.2 riêng |

Sai khác/thận trọng với CTU-13 source:

1. Cả hai source SVM đọc Test sớm; MI-RFE source còn transform Test trước chọn
   feature. Test không được fit scaler nhưng bị lộ sớm về mặt điều phối. Runner
   CICIoMT hoãn đọc Test tới sau final refit.
2. MI-RFE-SVM source tạo `feature_names` trước khi lọc numeric, có thể có 388
   tên so với X 380 cột; runner dùng whitelist 380 và Stage 5 đã có unit test
   cho mismatch. Không dùng danh sách tên cột source gốc để gán cho X đã lọc.
3. Source dùng `warnings.filterwarnings("ignore")`; runner thu warning/fit_status
   cho SVM. `max_iter=20000` vẫn có thể chạm giới hạn hội tụ.
4. Source Baseline `class_weight=None/probability=True`, Compact
   `balanced/False`; không sửa để làm hai thí nghiệm giống nhau.
5. Nhãn là `SCENARIO_PROXY`, Benign cực hiếm. Split/window không chung source row
   theo audit trước, nhưng capture/session độc lập **chưa chứng minh**.

## 3. Technical probe mới và giới hạn khoa học

Tái sử dụng loader/feature matrix/MI/RFE/SVC của Stage 5 thay vì chép lại
thuật toán. Pool Train tối đa 8.192 window (cap Stage 5), mẫu 2.000/5.000/8.192,
seed 42. Nguồn pool là quota theo source + sparse prefix stride=8, rồi chọn
stratified bằng seed và tách internal Train-fit/holdout 80/20. Đây chỉ là đo
kỹ thuật; pool có thiên lệch vị trí và floor Benign nên **không** đại diện cho
phân phối full. Nhánh Compact chỉ loại ba feature trong RF-RFE để đo chi phí;
ranking chưa hoàn chỉnh, k=1/10/100 không cho best_k chính thức.

Mỗi kích thước chạy trong worker riêng với timeout, RSS/process-tree sampling,
ngưỡng RAM available, không ghi đè JSON cũ, không tiếp tục full. Output ghi
thời gian fit/predict, peak RSS, support vectors, fit_status/convergence và
warning; không ghi Accuracy như kết quả nghiên cứu. `PASS_TECHNICAL_ONLY` chỉ
chứng minh probe theo giới hạn đã chạy xong. Polling RSS không phải hard OS
limit; transient OOM vẫn là rủi ro.

Theo số window đã được kiểm định trước, Train mỗi config khoảng 674 nghìn
(FS16) hoặc 1,35 triệu (FS8). Với `SVC` RBF/libsvm, chi phí tăng phi tuyến;
Baseline còn `probability=True` và cache cap 10.000 MB (cap, không phải cấp
phát cố định). Không suy rộng ba điểm 2k–8k thành cam kết thời gian/RAM full.
Đã giữ nguyên C/max_iter/kernel/probability/class_weight của từng nhánh.

## 4. File thay đổi và kiểm định

- Tạo `huyen_code/ciciomt24_svm_probe.py`: supervisor + worker Train-only.
- Tạo `huyen_code/test_ciciomt24_svm_probe.py`: test cấu hình, sampling, role,
  resource guard, full-gate block; dùng mock, không fit mô hình thật.
- Sửa `colab/experiment_launcher.py`: route probe mới và chặn full SVM khi chưa
  có resource review Stage 7.2. Không đổi RF path.
- Tạo `huyen_code/SVM_TECHNICAL_PROBE.md`: lệnh PowerShell/Colab cho người dùng.
- Tạo báo cáo này.

**Log tests người dùng gửi:** Lượt đầu Stage 7.1 có 8 PASS/1 FAIL; Stage 6 có
20 PASS. FAIL ở `test_admission_fail_creates_incomplete_report_without_worker` là lỗi
mock: test patch `subprocess.Popen` để cấm khởi tạo worker, nhưng hàm
`training.environment()` gọi `platform.platform()` trên Windows và gián tiếp
dùng `Popen` để hỏi phiên bản hệ điều hành. Test nay mock riêng
`training.environment`; logic resource-admission của probe không thay đổi.
Người dùng đã chạy lại Stage 7.1: **9 tests, 0 failures (OK)**. Sau đó Cell 8
Colab cũng báo lại **9/9 PASS** và Cell 9 đã chạy preflight (chi tiết mục 6).
Trong bằng chứng hiện có vẫn **chưa có technical probe Cell 10/10A, SVM fit,
hash before-after hoặc Test evaluation**. Không có thay đổi được ghi nhận với
năm script CTU-13, runner Stage 5, split/window CSV hay model/result Stage 5/6
trong lần cập nhật báo cáo này. Static code review không thay thế kết quả
technical probe.

## 5. PASS/FAIL trước Stage 7.2

Stage 7.1 chỉ đủ điều kiện xin đánh giá Stage 7.2 khi: người dùng tự chạy unit
tests và gửi PASS; từng probe cần `status=PASS_TECHNICAL_ONLY`, đủ ba size,
converged, có fit/predict/RSS/support-vector measurements, 0 Validation/Test CSV
rows, không timeout/RSS cap/low-RAM; source/data không bị sửa. Nếu bất kỳ mục
nào FAIL hoặc output dở, **không** dùng probe để mở full training. Kể cả tất cả
PASS, Stage 7.2 vẫn cần người dùng xem tăng trưởng runtime/RSS, tỷ lệ support
vectors, tài nguyên runtime cụ thể và phê duyệt resource gate riêng. Không lấy
Test để đặt guard, chọn tham số hay cấu hình.

**Dừng sau Stage 7.1. Không có kết quả SVM chính thức.**

## 6. Bằng chứng Colab mới: Cell 8 unit tests và Cell 9 preflight

Nguồn: output do người dùng dán ngày 21/10/2026; **không phải** probe JSON
hay kết quả huấn luyện chính thức.

| Hạng mục | Giá trị quan sát | Diễn giải |
| --- | --- | --- |
| Cấu hình Cell 8 | `baseline_svm`, `FW_256_FS_8`, `MODE=dry_run`, `RUN_TECHNICAL_SVM_PROBE=True` | Đúng nhánh technical probe Train-only; Cell 8 chưa chạy probe |
| Tên run Cell 8 | `mi_rfe_rf_FW256_FS8_run03` | Tên gợi RF nhưng experiment thực tế là Baseline SVM; phải gắn artifact theo `experiment/config`, tránh ghi nhầm là RF |
| Unit tests Stage 7.1 | 9 tests, 0 failures, 0 skipped — **PASS** | Kiểm tra mock/guard, không phải đo runtime full SVM |
| Cell 9 preflight | **FAIL** | Full SVM vẫn bị khóa, đúng chính sách Stage 7.1 |
| `svm_technical_probe` | **FAIL** | Tại thời điểm Cell 9 chưa có JSON probe hợp lệ được chấp nhận; cần xem Cell 10/10A |
| `svm_full_resource_gate_reviewed` | **FAIL** | Cố ý giữ False cho đến quyết định Stage 7.2, kể cả khi probe sau đó PASS |
| Repository/Python/scientific profile/source/manifest/unit-test evidence | **PASS** từng mục | Chỉ phản ánh kiểm tra trong preflight hiện tại |
| Không có run local/Drive trùng; Drive mounted | **PASS** | Không có bằng chứng ghi đè run ở preflight |
| RAM admission | **PASS**: estimated 28.100.151.632 B, available 52.489.080.832 B, budget 44.615.718.707 B | Ước tính ~26,17 GiB so với ~48,89 GiB available; source tự cảnh báo RAM forest/support-vector chưa biết, **không** chứng minh full SVM khả thi |
| Disk admission | **PASS**: yêu cầu 8.589.934.592 B (8 GiB) | Chỉ là điều kiện preflight |
| An toàn tại Cell 9 | Test feature rows read = **0**; `model.fit` calls = **0** | Chưa fit mô hình, chưa đọc feature rows của Test trong preflight |

**Kết luận chuyển giao Stage 7.2:** Chưa đủ bằng chứng để bật full SVM.
Cần người dùng gửi output **Cell 10A hoặc file `*_svm_technical_probe.json`**
sau khi chạy Cell 10 với `MODE=dry_run`; kèm log lỗi Cell 10 nếu bị dừng.
Kiểm tra `status`, đủ ba cỡ mẫu, thời gian fit/predict, peak process-tree RSS,
support-vector count, convergence/warnings, stop reason, và
`validation_csv_rows_read=test_csv_rows_read=0`. Nếu muốn đánh giá cả
`mi_rfe_svm`, cần probe riêng đúng `experiment/config`; không suy từ
Baseline. `PASS_TECHNICAL_ONLY` (nếu có) vẫn **không tự mở** resource gate
full SVM và không được ngoại suy tuyến tính lên toàn bộ ~1,35 triệu Train
window FW256/FS8.
