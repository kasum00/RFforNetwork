# Kết quả Giai đoạn 3 — bộ chia B2, unit test và smoke test

Đã triển khai một module hỗ trợ chia dữ liệu và audit:
D:\COMPACT-MI-RFE\huyen_code\ciciomt24_split.py

Unit test:
D:\COMPACT-MI-RFE\huyen_code\test_ciciomt24_split.py

Báo cáo smoke hiện hành:
D:\COMPACT-MI-RFE\huyen_code\runs\smoke_b2_stage3_02\report.json

Không chạy full dataset, không xuất CSV flow đã chia, không tạo sliding window,
không huấn luyện mô hình. SHA256 của cả năm file CODE_ctu13 trước/sau không đổi.

## Phạm vi đã kiểm thử

- 34 unit test: PASS, gồm fixture lỗi có chủ đích để kiểm tra khả năng từ chối.
- Smoke trên dữ liệu thật: 12 CSV, tối đa 3.000 dòng đầu/file, tổng 21.543 dòng.
- Manifest được ghi Parquet trước, sau đó đọc lại để kiểm định.
- Manifest có source CSV tuyệt đối, source ID, Original_Row_Index, timestamp gốc
  và timestamp parse, proxy label, Original_Label, split, embargo reason, sorted
  rank, Flow ID, khóa kết nối, Duration và khoảng hoạt động ước tính.
- Manifest_Scope=BOUNDED_PREFIX_ONLY; ranh giới vẫn PROVISIONAL.
- Nội dung cốt lõi sáu manifest nhóm của hai lượt smoke giống nhau.
- CSV gốc chỉ mở để đọc; size/mtime của 12 nguồn không đổi trước/sau smoke.
  Không đọc toàn bộ file lớn chỉ để băm SHA256 trong smoke.

## Số flow: mẫu thực đọc và thiết kế toàn bộ

| Tập | Manifest mẫu thực đọc | Toàn bộ dự kiến từ B2 đã duyệt |
|---|---:|---:|
| Train | 12.215 | 11.843.431 |
| Validation | 2.923 | 3.947.810 |
| Test | 4.216 | 3.947.817 |
| Embargo | 2.189 | 152.044 |
| Tổng | 21.543 | 19.891.102 |

Số toàn bộ là dự kiến Giai đoạn 2, chưa được đo lại bằng full scan ở Giai đoạn 3.
Giữ lại dự kiến 19.739.058 flow; tỷ lệ trên phần giữ lại:
59,99998075% / 19,99999189% / 20,00002736%.
Nếu sau review phải mở rộng embargo vì hoạt động flow/session, những số này sẽ
cần tính lại và trình duyệt; module chưa tự thực hiện thay đổi đó.

Prefix smoke không đại diện phân bố toàn bộ. Mẫu Benign gồm 3.000 Train, 0 Val,
3.000 Test do vị trí các prefix so với mốc B2; đây không phải phân bố toàn bộ B2.
Không dùng manifest smoke để báo cáo kết quả phân loại.

## PASS/FAIL và những điều chưa chứng minh

| Kiểm tra | Kết quả |
|---|---|
| Bảo toàn 21.543 dòng thực đọc, gồm embargo | PASS |
| Không trùng Source_CSV + Original_Row_Index giữa các tập | PASS |
| Đủ từng chỉ số dòng nguồn đã đọc, không thiếu/nhân đôi | PASS |
| Một bucket timestamp trong nhóm không bị gán sang hai tập | PASS |
| Các segment của cùng CSV không chồng lấn hoặc hở sorted rank | PASS |
| Phân bổ đúng các mốc B2 và sáu nhóm nhỏ Train-only | PASS |
| Original_Label giữ nguyên; nhãn proxy có nguồn gốc | PASS |
| Duration hợp lệ trong mẫu | PASS — 0 dòng không hợp lệ |
| Flow ID khớp 5-tuple hai chiều trong mẫu | PASS — 0 dòng bất nhất |
| Nội dung cốt lõi manifest mẫu tái lập | PASS |
| Fixture đầy đủ hai embargo, mỗi khoảng 511 flow | PASS trong unit test tổng hợp |
| Embargo >=511 tại mọi ranh giới toàn bộ dataset | CHƯA KIỂM TRA FULL |
| Bảo toàn 19.891.102 dòng toàn bộ | CHƯA KIỂM TRA FULL |
| Đơn vị Flow Duration được xác minh độc lập | CHƯA XÁC MINH |
| Capture/session độc lập | CHƯA CHỨNG MINH |
| An toàn ranh giới theo activity/session | CẦN REVIEW |

PASS cấu trúc không có nghĩa leakage-free.

## Rủi ro phát hiện trong dữ liệu thật

Theo giả định Duration tính bằng microsecond:

- 41 dòng nguồn riêng biệt thuộc Recon-VulScan có activity end ước tính đi qua
  mốc bắt đầu tập kế tiếp.
- 11 flow Train -> Validation; 30 flow Validation -> Test.
- Có 220 khóa kết nối riêng biệt tạo 294 ứng viên cặp tập.
- 281 ứng viên chỉ là khóa dùng lại, chưa thấy chồng min/max activity envelope.
- 13 ứng viên có thể chồng activity envelope.

5-tuple hoặc Flow ID dùng lại không xác minh session. Min/max envelope có thể
bao trùm thời gian idle giữa nhiều phiên khác nhau. Module lưu đúng trạng thái
ứng viên/rủi ro, không tạo Session_ID giả và không tự chuyển các flow này.

Ví dụ Train -> Validation:

- Source CSV: D:\COMPACT-MI-RFE\CICIoMT24\CSV\train\Recon-VulScan_train.pcap_Flow.csv
- Original_Row_Index: 77, zero-based, không tính header.
- Timestamp: 12/09/2023 22:30:26.
- Flow Duration: 110133846; nếu là microsecond, tương đương 110,133846 giây.
- Activity end ước tính: 22:32:16,133846.
- Validation bắt đầu: 22:32:11.
- Dòng thuộc Train theo thời điểm bắt đầu, nhưng thống kê flow có thể phản ánh
  hoạt động sau khi Validation đã bắt đầu. Vì vậy chỉ kiểm tra timestamp đầu
  và khoảng 511 dòng chưa đủ.

Ví dụ Validation -> Test:

- Source CSV: D:\COMPACT-MI-RFE\CICIoMT24\CSV\test\Recon-VulScan_test.pcap_Flow.csv
- Original_Row_Index: 47.
- Timestamp: 12/09/2023 22:33:26.
- Flow Duration: 112041874 theo giả định microsecond.
- Activity end ước tính: 22:35:18,041874.
- Test mới bắt đầu: 22:35:16.

Timestamp nguồn chỉ tới giây. Activity_End_Upper cộng thêm 1 giây để đánh dấu
nguy cơ do độ phân giải đồng hồ; không khôi phục được thời gian packet thật.

Chi tiết:

- D:\COMPACT-MI-RFE\huyen_code\runs\smoke_b2_stage3_02\activity_crossings.parquet
- D:\COMPACT-MI-RFE\huyen_code\runs\smoke_b2_stage3_02\activity_crossings_preview.csv
- D:\COMPACT-MI-RFE\huyen_code\runs\smoke_b2_stage3_02\session_candidates.parquet

## Ví dụ giải thích việc chia một CSV

Ví dụ rút gọn cho file Benign; timestamp đều ngày 14/09/2023.
Sau sort theo thời gian, chỉ số dòng gốc vẫn được giữ:

| Original_Row_Index | Timestamp | Source_Sorted_Rank | Split theo B2 |
|---:|---|---:|---|
| 1 | 11:00:00 | 0 | Train |
| 3 | 12:00:00 | 1 | Embargo |
| 2 | 13:00:00 | 2 | Validation |
| 0 | 19:00:00 | 3 | Test |

File gốc không bị sắp xếp lại hay ghi đè. Manifest ghi dòng gốc số 1 thuộc Train,
dòng số 2 thuộc Validation, v.v. Sorted rank dùng để mô tả các đoạn liên tục sau
sort, không phải khoảng raw row. Ví dụ bốn dòng chỉ minh họa ánh xạ; không đại
diện số dòng embargo thật. Unit test riêng đã tạo hai bucket bảo vệ đủ 511 dòng.

## Điều kiện trước bước tiếp theo

1. Xác minh đơn vị Duration từ công cụ/config xuất CSV, hoặc ghi rõ giả định.
2. Chốt chính sách cho activity crossing: có mở rộng vùng purge/embargo theo
   activity end, giữ nguyên toàn bucket timestamp và tính lại quota hay không.
3. Sau phê duyệt mới chạy full manifest audit để kiểm tra bảo toàn 19.891.102 dòng,
   toàn bộ ranh giới và các trường hợp rủi ro chưa có trong prefix.
4. Chỉ sau khi full manifest được kiểm định và duyệt mới triển khai/xuất CSV flow.

Module hiện chưa có chức năng xuất CSV flow; full manifest scan cần cờ
authorize-full-scan. Không có quá trình full nào đang chạy nền.
