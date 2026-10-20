# CICIoMT2024 — bộ chia B2 và audit, Giai đoạn 3

Chỉ có một module hỗ trợ: ciciomt24_split.py. test_ciciomt24_split.py là unit
test bằng unittest chuẩn của Python. Chưa sửa năm script CTU-13; không tạo
window, huấn luyện hay xuất CSV flow đã chia.

## Phạm vi và ranh giới

- Đọc nguồn CSV/train và CSV/test nhưng không ghi vào nguồn.
- Mapping kịch bản được liệt kê tường minh; kịch bản lạ bị từ chối.
- Original_Label giữ nguyên. Proxy_Label là nhãn kịch bản, không phải ground truth.
- Các mốc B2 ở Giai đoạn 2 được lưu tường minh, theo timestamp nguồn chưa có múi giờ.
- G01, G06, G07, G09, G12, G16 chỉ ở Train; cảnh báo độ phủ được lưu trong report.
- Sắp xếp ổn định theo nhóm, Timestamp, Source_CSV, Original_Row_Index.
- Source_Sorted_Rank và Group_Sorted_Rank khác chỉ số dòng CSV gốc.
- Giữ nguyên bucket timestamp; manifest gồm cả dòng embargo.
- Mọi ranh giới vẫn PROVISIONAL sau audit. Khi thấy activity/session nguy cơ,
  module không tự đổi ranh giới, nhãn hay xóa thêm dòng.

Thiết kế toàn bộ từ Giai đoạn 2, chưa được smoke test đo lại:

| Train | Validation | Test | Embargo |
|---:|---:|---:|---:|
| 11.843.431 | 3.947.810 | 3.947.817 | 152.044 |

Tổng 19.891.102; giữ lại 19.739.058. Tỷ lệ trên phần giữ lại:
59,99998075% / 19,99999189% / 20,00002736%.

## Duration, Flow ID và session

Mặc định microseconds là giả định có ghi rõ, chưa xác minh độc lập từ exporter.
Có thể khai báo đơn vị khác cùng mô tả bằng chứng. Một chuỗi bằng chứng người dùng
nhập không tự biến thành xác minh độc lập.

Activity_End_Estimate = Timestamp + Flow Duration theo đơn vị đã khai báo.
Activity_End_Upper cộng thêm 1 giây vì timestamp nguồn chỉ tới giây.
Duration lỗi/âm/không hữu hạn được giữ lại và đánh dấu unknown, không bỏ dòng.
Timestamp lỗi làm dừng lập manifest để không tự gán tập hoặc bỏ dòng.

Flow ID được đối chiếu với 5-tuple chuẩn hóa hai chiều. Các kiểm tra khác nhau:

1. Một flow có thời gian hoạt động ước tính vượt thời điểm bắt đầu tập kế tiếp.
2. Một khóa kết nối xuất hiện ở hai tập, có/không chồng min/max activity envelope.
3. Flow ID không parse được hoặc không khớp với 5-tuple.

5-tuple, Flow ID và min/max envelope đều không chứng minh cùng session.
Envelope có thể bao trùm khoảng idle giữa nhiều phiên. TCP sequence, packet trace,
capture ID thật và ngữ cảnh timeout/sessionization chưa được xác minh.
Không tuyên bố leakage-free.

## Chạy kiểm thử

Yêu cầu Python với pandas, numpy, pyarrow; không cần pytest hoặc thư viện ML.

PowerShell, unit test chỉ dùng fixture tổng hợp nhỏ:

~~~powershell
Set-Location 'D:\COMPACT-MI-RFE\huyen_code'
python -m unittest -v test_ciciomt24_split
~~~

Smoke đọc tối đa 3.000 dòng đầu mỗi file trong 12 CSV được chọn:
Benign, MQTT-DDoS-Connect_Flood, ARP_Spoofing, Recon-VulScan,
Recon-Ping_Sweep và MQTT-Malformed_Data, ở cả hai thư mục gốc.
Giới hạn cứng của smoke là 10.000 dòng/file. Nguồn ngắn hơn được đọc hết.

~~~powershell
python .\ciciomt24_split.py smoke --dataset-root 'D:\COMPACT-MI-RFE\CICIoMT24' --out 'D:\COMPACT-MI-RFE\huyen_code\runs\smoke_manual_01' --max-rows-per-file 3000 --duration-unit microseconds
~~~

Thư mục output phải mới/rỗng và nằm trong một thư mục con riêng của huyen_code.
Chế độ plan đọc toàn bộ bị chặn nếu thiếu cờ authorize-full-scan.
Chưa có chức năng xuất CSV flow: kiểm định full manifest và phê duyệt phải xảy ra
trước khi triển khai bước xuất đó. Không chạy chế độ plan ở Giai đoạn 3 hiện tại.

## Kết quả

- manifest/Gxx.parquet: manifest đầy đủ cho tất cả dòng thực đọc, gồm cả embargo.
- manifest_preview.csv: tối đa 100 dòng để xem nhanh.
- source_inventory.csv: nguồn, số dòng thực đọc, phạm vi prefix/full, size và mtime.
- segments.csv: các đoạn riêng từng CSV theo sorted rank, end exclusive.
- activity_crossings.parquet: mọi trường hợp flow có khả năng vượt ranh giới nếu có.
- activity_crossings_preview.csv: tối đa 100 ví dụ nếu có.
- session_candidates.parquet và session_candidates_preview.csv: khóa kết nối chung.
- report.json và report.md: PASS/FAIL, cảnh báo, số đếm và hạn chế.

Manifest được ghi trước, sau đó được đọc lại từ Parquet để kiểm tra bảo toàn,
identity, bucket, segment, mapping nhãn và ranh giới B2.

Báo cáo triển khai và ví dụ tiếng Việt: STAGE3_REPORT.md.
Lượt smoke hiện hành: runs/smoke_b2_stage3_02.
34 unit test đã PASS; smoke kiểm tra 21.543 dòng đã PASS cấu trúc, nhưng phát hiện
41 activity crossing có điều kiện theo giả định Duration=microseconds.

Smoke chỉ xác minh các dòng thực đọc. Nó không chứng minh bảo toàn 19.891.102
dòng, đủ 511 flow tại mọi ranh giới toàn bộ, hay không có session khác vượt biên.
Các kiểm tra đó được đánh dấu NOT_RUN/NOT_ESTABLISHED thay vì PASS.

Ví dụ sau sort, raw row 7 có timestamp sớm hơn raw row 2 thì raw row 7 đứng trước
trong manifest, nhưng Original_Row_Index vẫn là 7. Không dùng khoảng raw row
min/max để thay cho danh sách dòng thật sau sort.

## Giai đoạn 3.1: review có giới hạn, không áp dụng split mới

Báo cáo hiện hành: `STAGE31_REPORT.md`.
Kết quả audit: `runs/duration_review_stage31_03/review.json`.
48/48 unit test PASS. Đơn vị microsecond có bằng chứng nguồn chuẩn và mẫu thực,
nhưng chưa truy được build exporter cục bộ. 41 flow vượt mốc thời gian không
đồng nghĩa 41 session leakage; 9 flow tạo 10 cặp có thể chồng dưới bất định 1 giây.

Nhánh chỉ đọc lại smoke đã lưu và đúng prefix nguồn cũ, tối đa 12 file × 3.000 dòng:

~~~powershell
python .\ciciomt24_split.py review-smoke --input-run '.\runs\smoke_b2_stage3_02' --out '.\runs\duration_review_manual_01'
~~~

Đây là mô phỏng so sánh chi phí embargo, không ghi CSV flow hoặc thay manifest
B2 đã duyệt. Không có quyền chạy full scan trong nhánh review. Tỷ lệ toàn bộ
sau điều chỉnh và độc lập session vẫn chưa được xác lập.
