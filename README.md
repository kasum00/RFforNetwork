# Nhật ký tiến độ nghiên cứu COMPACT-MI-RFE trên CICIoMT2024

## Windows local / máy ảo — tiếp tục checkpoint Colab

Bộ chạy MI-RFE SVM local: [cấu trúc thư mục và PowerShell](local/README_WINDOWS_LOCAL.md), [báo cáo triển khai và giới hạn](local/WINDOWS_LOCAL_REPORT.md). Giữ nguyên checkpoint scientific binding; có preflight, kiểm migration và approval Windows riêng. Source/tests mới chỉ được viết và review tĩnh, **chưa chạy**, không tự mở training. Chỉ tiếp tục FW512/FS16 của run Colab đã xác minh; không chạy lại từ k=1.

## Notebook SVM trên Google Colab — bốn cấu hình

MI-RFE SVM nhiều phiên: [notebook có Cell 13 start/resume](notebooks/CICIoMT24_MI_RFE_SVM_RESUMABLE.ipynb), [hướng dẫn và lệnh tự kiểm thử](huyen_code/SVM_RESUMABLE_GUIDE.md), [audit CTU-13](huyen_code/SVM_RESUMABLE_AUDIT.md), [báo cáo triển khai](huyen_code/SVM_RESUMABLE_REPORT.md). Mặc định tắt training; chỉ tái dùng MI/RFE/k/final đã commit và xác minh, không resume giữa một lần SVC.fit. Source mới cần commit/test evidence/approval mới; tests được viết nhưng Codex chưa chạy.

Bản tích hợp từ notebook FIXED_V3: [CICIoMT24_SVM_4_CONFIGS_COLAB.ipynb](notebooks/CICIoMT24_SVM_4_CONFIGS_COLAB.ipynb). Hỗ trợ `baseline_svm` và `mi_rfe_svm`; mỗi lượt chạy một cấu hình FW/FS. Source huấn luyện được khóa tại commit `c93eab87395edaf8fd141b29ee9914df066a2e7e`; các cờ chạy training và tạo approval mặc định tắt.

Xem [hướng dẫn đẩy GitHub và chạy Colab](notebooks/CICIoMT24_SVM_4_CONFIGS_COLAB_GUIDE.md). Notebook này đã được review source; chưa được thực thi hoặc xác nhận full SVM trên Colab trong lần tích hợp này. Các mục bên dưới là nhật ký các giai đoạn trước.

**Cập nhật: 20/10/2026**  
**Trạng thái:** Hoàn tất chia dữ liệu, xây dựng sliding windows đầy đủ và một thí nghiệm MI-RFE + Random Forest chính thức; các thí nghiệm còn lại chưa hoàn tất.

## 1. Mục tiêu

Công việc được triển khai riêng trên **CICIoMT2024**, dựa trên phương pháp trong bài *A Compact and Drift-Aware Framework for Network Intrusion Detection Using Temporal Aggregation and MI–RFE Feature Selection*.

## 2. Những công việc đã làm

### Giai đoạn 1 — Khảo sát dữ liệu

- Kiểm kê **71 file CSV** nguồn với **19.891.102 dòng flow**.
- Kiểm tra schema, số lượng đặc trưng và tình trạng nhãn.
- Xác định vấn đề quan trọng: trường nhãn gốc được kiểm tra chứa `No Label`; nhãn nhị phân sử dụng trong pipeline được suy ra theo kịch bản, ký hiệu **`SCENARIO_PROXY`**, chưa phải nhãn chính xác của từng flow.

### Giai đoạn 2 — Tách dữ liệu và kiểm soát rò rỉ (Stage 3.4)

- Triển khai chia dữ liệu trước khi tạo cửa sổ thời gian.
- Giữ **17.997.672 flow** cho Train/Validation/Test và đặt **1.893.430 flow** vào vùng embargo.
- Sau chia: Train **10.798.603**, Validation **3.599.535**, Test **3.599.534** flow.
- Xuất **93 source segments**; ghi nguồn CSV, chỉ số dòng gốc, split và nhãn proxy để truy vết.
- Kiểm tra không tái sử dụng cùng dòng nguồn qua các tập; vùng chặn tối thiểu 511 dòng tại các ranh giới áp dụng; không có exact duplicate qua split theo phép kiểm tra đã triển khai.
- **Chưa khẳng định capture/session độc lập**, do thiếu bằng chứng xác nhận.

### Giai đoạn 3 — Sinh sliding window (Stage 4.2)

- Tạo đủ bốn cấu hình `FW/FS = 256/8, 256/16, 512/8, 512/16`.
- Mỗi window được tạo trong đúng một source segment/split, không ghép xuyên ranh giới.
- Tính năm thống kê: `mean`, `std`, `median`, `skew`, `kurtosis` trên **76 đặc trưng hành vi**, sinh **380 đặc trưng window**.
- Tạo **6.737.006 window / 322 file CSV** và kiểm định readback, schema, chỉ số, nhãn và provenance.
- Ghi nhận NaN do phép thống kê trên đặc trưng hằng; xử lý trước khi đưa vào mô hình theo quy tắc đã kiểm chứng.

### Giai đoạn 4 — Cài đặt MI-RFE và Random Forest (Stage 5.1–5.2)

- Viết pipeline nạp dữ liệu theo split, tách 380 feature numeric ra khỏi metadata/ID/label.
- MI tính trên Train; chọn Top100 đặc trưng.
- RF-RFE xếp hạng 100 đặc trưng; đánh giá k từ 1 đến 100 trên Validation.
- Dùng Random Forest với `300 trees`, `max_depth=20`, `min_samples_split=3`, `class_weight=balanced`, `random_state=42`.
- Cố định k/feature được chọn; refit scaler và mô hình trên Train+Validation; chỉ sau đó mới đọc và đánh giá Test một lần.
- Kiểm định: **97/97 unit test PASS** trước khi chạy; **130/130 assertion hậu nghiệm PASS** sau chạy (không phải 130 unit test).

## 3. Kết quả chính thức đã được kiểm định

**MI-RFE + RF, FW512/FS16** (run hoàn thành tối 19/10/2026, xác minh 20/10/2026)

| Chỉ tiêu | Kết quả |
|---|---:|
| Train windows | 673.559 |
| Validation windows | 224.373 |
| Test windows | 224.287 |
| Đặc trưng đầu vào | 380 |
| Đặc trưng sau MI-RFE | **8** |
| Giảm số đặc trưng | **97,89%** |
| Validation Accuracy | **100%** |
| Test Accuracy | **99,99866%** |
| Test Macro-F1 | **99,83958%** |
| Test Balanced Accuracy | **99,68085%** |
| Test Recall Benign | **99,36170%** |
| Test Recall Attack | **100%** |
| Thời gian chạy toàn bộ | **~42 phút 9 giây** |
| Peak RAM process RSS được đo | **~9,99 GiB** |

**Confusion matrix Test** (theo nhãn proxy):

| Nhãn thật / Dự đoán | Benign | Attack |
|---|---:|---:|
| Benign | 467 | 3 |
| Attack | 0 | 223.817 |

8 feature đã chọn:

1. `Fwd Seg Size Avg_median`
2. `Fwd Pkt Len Mean_median`
3. `Fwd Pkt Len Max_median`
4. `Pkt Len Min_mean`
5. `Pkt Len Min_std`
6. `Bwd PSH Flags_kurtosis`
7. `Bwd PSH Flags_mean`
8. `Init Bwd Win Byts_std`

**Cảnh báo khi diễn giải:** Test chỉ có 470 window Benign, còn 223.817 window Attack. Accuracy rất cao nhưng không đủ chứng minh khả năng phát hiện trong môi trường thực tế. Nhãn là proxy theo scenario; chưa xác minh độc lập capture/session. Không thể khẳng định hoàn toàn *leakage-free*.

## 4. Công việc còn lại (Stage 5.3 trở đi)

- **Chưa chạy MI-RFE + RF FW256/FS16 theo protocol C+ chính thức**: preflight bị chặn vì RAM available khoảng 9,35 GiB, thấp hơn ngưỡng bảo thủ khoảng 12,01 GiB.
- Chưa có bảng so sánh **chính thức, cùng protocol C+** giữa các cấu hình FW/FS và giữa RF với các mô hình khác.
- Tiếp tục kiểm định nhãn, tính độc lập capture/session, độ phủ scenario và khả năng tái hiện thực nghiệm.
- Chuẩn hóa môi trường, lệnh chạy và tài liệu để người khác có thể tự tái hiện.

> Các kết quả thăm dò cũ dùng split, cỡ mẫu hoặc thiết lập khác cần để ở mục `exploratory / historical` riêng; không thay thế hoặc gộp với kết quả C+ chính thức.

