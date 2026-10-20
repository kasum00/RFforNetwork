# Báo cáo Giai đoạn 4.2 — Full Sliding Window CICIoMT2024

**Ngày hoàn tất:** 19/10/2026  
**Trạng thái:** PASS — toàn bộ output đã được đọc lại, kiểm định và promote sang thư mục chính thức.

## 1. Phạm vi và kết quả

- Input: `D:\COMPACT-MI-RFE\CICIoMT24\CSV_SPLIT_CPLUS`
- Output: `D:\COMPACT-MI-RFE\CICIoMT24\WINDOWS_CPLUS`
- Module: `D:\COMPACT-MI-RFE\huyen_code\ciciomt24_sliding_window.py`
- Đã xử lý đủ **93/93 source segments**; không nối segment, scenario hay split khi tạo window.
- Đã tạo đúng **6.737.006 windows** trong **322 CSV**. Thư mục `.incomplete` không còn tồn tại; output chỉ được promote sau khi full readback PASS.
- Giữ nguyên bốn cấu hình, 76 đặc trưng hành vi, năm thống kê NumPy/SciPy, cách gán nhãn existential rule và 380 đặc trưng window. Không huấn luyện mô hình, không fit scaler/MI/RFE.

## 2. Số window và nhãn proxy

`Benign/Attack` dưới đây là nhãn **SCENARIO_PROXY**, không phải ground truth từng flow. `Label` đầu ra là 0/1 theo nhãn proxy và existential rule đã duyệt.

| FW/FS | Split | Tổng window | Benign proxy | Attack proxy |
|---|---:|---:|---:|---:|
| 256/8 | Train | 1.348.317 | 3.182 | 1.345.135 |
| 256/8 | Val | 449.343 | 1.042 | 448.301 |
| 256/8 | Test | 449.251 | 1.003 | 448.248 |
| 256/16 | Train | 674.169 | 1.591 | 672.578 |
| 256/16 | Val | 224.677 | 521 | 224.156 |
| 256/16 | Test | 224.631 | 502 | 224.129 |
| 512/8 | Train | 1.347.101 | 3.150 | 1.343.951 |
| 512/8 | Val | 448.735 | 1.010 | 447.725 |
| 512/8 | Test | 448.563 | 939 | 447.624 |
| 512/16 | Train | 673.559 | 1.575 | 671.984 |
| 512/16 | Val | 224.373 | 505 | 223.868 |
| 512/16 | Test | 224.287 | 470 | 223.817 |
| **Tổng** | **mọi cấu hình** | **6.737.006** | **15.490** | **6.721.516** |

Tổng hàng cuối cộng các cấu hình khác nhau nên không phải số window duy nhất sau khi gộp FW/FS.

## 3. Kiểm định

| Kiểm tra | Kết quả | Bằng chứng |
|---|---|---|
| Manifest C+ và `validation.json` | PASS | 19.891.102 source flow; 93 segment |
| Số lượng từng FW/FS × split | PASS | Cả 12 ô khớp chính xác bảng số lượng đã duyệt |
| CSV output đọc lại, schema | PASS | 322/322 file; 394 cột/file: 380 feature + 14 metadata/label |
| Window ID / chỉ số | PASS | ID deterministic, duy nhất theo segment/cấu hình và chỉ số liên tục; không gap/reorder |
| Split, scenario, source segment | PASS | Kiểm tra metadata từng dòng; mỗi file thuộc một segment và một split |
| Truy vết source row | PASS | Có Source CSV/ID, Original Row Index Start/End và timestamp; chỉ số nằm trong segment manifest |
| Nhãn proxy | PASS | Train/Val/Test counts khớp theo Benign/Attack và nguồn nhãn |
| So sánh sinh NaN/Inf với readback | PASS | Các tổng khớp; Inf = 0 |
| Đối chiếu NumPy/SciPy | PASS | 228/228 mẫu; sai số hữu hạn lớn nhất `2.3377189677376e-06`, trong tolerance `rtol=1e-6`, `atol=1e-8`; NaN masks cũng được so sánh |
| Input C+ không đổi | PASS | SHA256 tree trước/sau trùng khớp |
| Năm script CTU-13 gốc không đổi | PASS | SHA256 khớp snapshot đã duyệt |
| Unit test | PASS | 6/6; `py_compile` PASS; `audit-manifest` PASS |
| Capture/session độc lập | **CHƯA ĐƯỢC CHỨNG MINH** | Chia theo block thời gian và segment không xác minh độc lập capture/session |

### NaN/Inf

Readback đếm được **626.545.486 ô NaN trong các feature**, **0 ô Inf**. NaN chủ yếu phát sinh khi skew/kurtosis áp dụng lên feature cột hằng, phù hợp với hành vi NumPy/SciPy được đối chiếu. Các file window được giữ nguyên; ở bước classifier phải xử lý như pipeline CTU-13 gốc (`fillna(0)`), không được đưa các ô NaN vào mô hình chưa xử lý.

## 4. Dung lượng, RAM và thời gian

- Kích thước CSV window: **21.036.370.082 byte** (xấp xỉ 19,59 GiB).
- Tổng kích thước các file trong output: **21.036.437.065 byte**.
- Peak RSS được lấy mẫu: **231.936.000 byte** (xấp xỉ 221,19 MiB).
- Lượt resume cuối cùng, gồm generation phần còn lại, full readback và hash check: **01:27:47**.
- Toàn chuỗi tính từ snapshot đầu tiên (10:55:45) đến khi summary PASS (21:07:26) kéo dài khoảng **10:11:41**, bao gồm các lần bị ngắt và resume. Do lần ngắt không tạo `FAILED.json`, tool không tích lũy chính xác runtime thực thi của các attempt trước; không dùng trường `elapsed_seconds_all_attempts` trong summary như tổng runtime chính xác.

Resume đã kiểm định lại prefix 14 segment ở lần đầu và 61 segment ở lần sau. CSV dở của segment đang chạy được chuyển vào archive sibling có thể khôi phục; không xóa input hoặc output cũ.

## 5. Ranh giới window và leakage

- Mỗi window chỉ được tạo trong một source segment CSV thuộc đúng một split. Vì membership Train/Val/Test đã được khóa ở cấp flow và không có source row thuộc nhiều split, không có **source row dùng chung giữa các tập** thông qua window generation.
- Các window *trong cùng một split* có thể chồng lấn source row theo sliding-window stride; đó là hành vi dự kiến. Cam kết ở đây là không tạo window vượt source segment/split và không tái sử dụng flow nguồn qua các split.
- Điều này **không đồng nghĩa** với không trùng bản ghi giữa capture, không có flow/session liên quan ở nhiều capture, hoặc capture độc lập. Capture/session independence vẫn chưa được chứng minh bằng metadata hiện có.
- Sáu scenario không có window trong Validation/Test theo các cấu hình: `ARP_Spoofing`, `MQTT-DoS-Publish_Flood`, `MQTT-Malformed_Data`, `Recon-Ping_Sweep`, `TCP_IP-DDoS-ICMP`, `TCP_IP-DoS-ICMP`. `Recon-Ping_Sweep` chỉ có 71 flow Train nên cũng không tạo được window với FW tối thiểu 256. Đây là giới hạn độ phủ dữ liệu, không phải lỗi generator.

## 6. Tương thích MI-RFE/classifier cho Giai đoạn 5

- X window thực tế có **380 feature**. 14 cột còn lại là `Window_ID`, `Window_Index`, `Num_Flows`, `Split`, `Scenario`, `Source_CSV`, `Source_CSV_ID`, `Original_Row_Index_Start`, `Original_Row_Index_End`, `Window_Start_Timestamp`, `Window_End_Timestamp`, `Proxy_Label`, `Label_Origin`, `Label`.
- Khi tạo X, phải loại metadata/ID/provenance; `Label` chỉ là y. Không đưa scenario proxy metadata hoặc source identifiers vào feature matrix.
- Hai script MI-RFE gốc có rủi ro danh sách `feature_names` được lấy trước khi lọc numeric: có thể giữ **388 tên** trong khi X thực tế chỉ có 380 cột. Giai đoạn 5 cần lấy feature names từ chính X sau cùng một bước lọc numeric (hoặc dùng danh sách 380 cột đã xác nhận). Chưa sửa script classifier trong giai đoạn này.
- NaN cần được xử lý bằng quy tắc tương thích CTU-13 (`fillna(0)`) trước classifier/MI/RFE. Không fit scaler, MI hay RFE trên Test; giữ nguyên mọi tham số thuật toán đã duyệt.

## 7. Cấu trúc và artifacts

```text
D:\COMPACT-MI-RFE\CICIoMT24\WINDOWS_CPLUS\
  FW_256_FS_8\{train,val,test}\{scenario}\..._windows.csv
  FW_256_FS_16\{train,val,test}\{scenario}\..._windows.csv
  FW_512_FS_8\{train,val,test}\{scenario}\..._windows.csv
  FW_512_FS_16\{train,val,test}\{scenario}\..._windows.csv
  window_run_summary.json
  reference_stat_samples.csv
  input_sha256_before.json / input_sha256_after.json
  ctu13_sha256_before.json / ctu13_sha256_after.json
```

`window_run_summary.json` là kết quả kiểm định máy đọc được đầy đủ. `reference_stat_samples.csv` lưu các mẫu đối chiếu phép thống kê.

## 8. Kết luận

Full sliding window C+ đã hoàn thành và kiểm định PASS trên toàn bộ **6.737.006 window**. Có thể chuyển sang Giai đoạn 5 để chuẩn bị feature matrix và chạy MI/RFE/classifier sau khi sửa đồng bộ danh sách `feature_names` và xử lý NaN. Kết quả không được mô tả là leakage-free tuyệt đối: độc lập capture/session chưa được xác nhận, và nhãn vẫn là scenario proxy.
