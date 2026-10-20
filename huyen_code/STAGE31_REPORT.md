# Giai đoạn 3.1 — Flow Duration và ranh giới B2

Ngày kiểm tra: 19/10/2026. Chỉ kiểm tra và mô phỏng trên 21.543 flow đã có trong smoke test. Không chạy full split, không ghi CSV dữ liệu phân tập, không tạo window, không huấn luyện và không sửa năm script CTU-13.

Kết quả hiện hành: `runs/duration_review_stage31_03/review.json`. Hai lượt `_01`, `_02` là lượt kiểm tra trung gian, không phải các lần chia dataset. Manifest B2 cũ và các ranh giới mặc định không bị thay đổi.

## 1. Kết luận để quyết định

1. **Microsecond được xác nhận trong mã nguồn CICFlowMeter và được dữ liệu mẫu củng cố rất mạnh.** Chưa xác minh được binary, commit, cấu hình hoặc quy trình đã xuất chính xác 71 CSV cục bộ. Vì vậy không gọi nguồn gốc exporter của toàn bộ dataset là đã xác minh.
2. **41 flow thực sự vượt mốc bắt đầu tập kế tiếp theo phép tính từ metadata.** Tuy nhiên, đó không đồng nghĩa 41 phiên kết nối bị chia đôi. Có 40 flow có cùng khóa kết nối trong tập kế tiếp. Không có cặp cùng khóa nào chồng thời gian theo end ước tính trực tiếp; 9 flow tạo 10 cặp có khả năng chạm/chồng khi xét bất định 1 giây.
3. **Nên bổ sung embargo theo thời gian hoạt động**, bên cạnh tối thiểu 511 flow và nguyên timestamp bucket. Chỉ mới mô phỏng, chưa áp dụng phương án mới vào split thật.
4. **Chưa chứng minh tỷ lệ 60:20:20 sau điều chỉnh trên toàn bộ dữ liệu.** Prefix smoke không đại diện phân bố toàn bộ. Không lấy 652 dòng loại thêm trên mẫu để trừ thẳng vào số đếm toàn bộ rồi công bố kết quả.
5. **Chưa đủ căn cứ cho full split.** Bước nên xin duyệt tiếp là audit metadata/manifest toàn bộ, không xuất CSV flow, để tính chi phí embargo và độ lệch tỷ lệ trước khi chốt chính sách.

## 2. Đơn vị và ý nghĩa Flow Duration

### Bằng chứng từ nguồn sơ cấp

- [PacketReader.java của CICFlowMeter](https://raw.githubusercontent.com/ahlashkari/CICFlowMeter/master/src/main/java/cic/cs/unb/ca/jnetpcap/PacketReader.java): đường đọc IPv4 lưu thời gian gói tin bằng microsecond.
- [BasicFlow.java của CICFlowMeter](https://raw.githubusercontent.com/ahlashkari/CICFlowMeter/master/src/main/java/cic/cs/unb/ca/jnetpcap/BasicFlow.java): duration là hiệu thời gian gói cuối và gói đầu. Công thức tốc độ đổi duration sang giây bằng hệ số một triệu. Timestamp xuất ra tới giây, khớp hình thức ngày/tháng/năm và AM/PM trong CSV mẫu.

Đây là mã nguồn chuẩn đang công khai, không phải bằng chứng rằng đúng build đó đã tạo CSV cục bộ. Không tìm thấy mã Java, JAR hay cấu hình exporter tương ứng trong phần dự án đã kiểm tra. 21.543 flow mẫu dùng địa chỉ dạng IPv4. Không suy rộng đường đọc này sang mọi exporter hoặc mọi loại địa chỉ.

### Đối chiếu trên dữ liệu thực

Đọc lại đúng các prefix đã lưu trong manifest: 12 CSV, tối đa 3.000 dòng/file, tổng 21.543 dòng nguồn riêng biệt. Không đọc toàn bộ các file lớn. Kiểm tra khớp Source_ID, Original_Row_Index, Flow ID, Timestamp, Duration và Label với manifest cũ.

| Kiểm tra | Số dòng phù hợp / số dòng đủ điều kiện |
|---|---:|
| Flow Pkts/s = tổng packet × 1.000.000 / duration | 21.412 / 21.412 |
| Flow Byts/s = tổng byte × 1.000.000 / duration | 15.746 / 15.746 |
| Fwd Pkts/s theo microsecond | 14.711 / 14.711 |
| Bwd Pkts/s theo microsecond | 21.412 / 21.412 |
| Flow IAT Mean × (tổng packet − 1) ≈ duration | 21.543 / 21.543 |
| Flow IAT Max, Fwd/Bwd IAT Tot, Active Max, Idle Max không vượt duration | 21.543 / 21.543 cho từng cột |

Sai số tương đối lớn nhất của các công thức tốc độ nhỏ hơn 1,2 × 10⁻¹⁴. Với packet/s, các giả thuyết giây, millisecond và nanosecond đều khớp 0/21.412 dòng ở cùng ngưỡng so sánh. Hệ số suy ra trung vị là 1.000.000.

Có 131 dòng duration bằng 0: giữ nguyên, không dùng làm mẫu số kiểm định tốc độ. Duration mẫu nằm trong [0; 119.999.999] microsecond. Không có giá trị duration âm/không hợp lệ trong manifest mẫu.

Các đặc trưng IAT củng cố tính nhất quán nội bộ. Riêng việc IAT và duration có cùng thang đo không tự chứng minh đơn vị vật lý; công thức packet/s và nguồn exporter là phần bằng chứng bổ sung cần thiết.

### Cách tính thời điểm kết thúc

```text
Flow_End_Time = Timestamp + Flow_Duration_us / 1.000.000 giây
Flow_End_Upper = Flow_End_Time + 1 giây
```

Ví dụ duration 110.133.846 là 110,133846 giây, không phải 110 triệu giây. Timestamp 12/09/2023 22:30:26 cho end ước tính 22:32:16.133846.

Timestamp CSV không có phần lẻ giây và không xác lập múi giờ. Với cách định dạng bỏ phần lẻ trong nguồn chuẩn, start thực nằm trong bucket một giây; end tính từ timestamp hiển thị cũng có bất định đó. `Flow_End_Upper` là biên bảo thủ để kiểm tra, không phải timestamp gói tin đo được. Nếu exporter cục bộ dùng quy ước làm tròn/đồng hồ khác, phải kiểm tra lại giả định này. Không dùng phép tính trên làm bằng chứng chính xác đến microsecond về thời gian tuyệt đối.

## 3. Phân tích đủ 41 trường hợp

`activity_crossings_preview.csv` được yêu cầu đọc chỉ chứa 10 ví dụ. Đã đối chiếu chúng với `activity_crossings.parquet`, tái tính từ manifest và phân tích đủ 41 dòng nguồn, không chỉ 10 dòng preview.

Chi tiết từng trường hợp, gồm source CSV, chỉ số dòng, split, start/end, mức vượt biên, Flow ID, 5-tuple, SYN/FIN/RST và kết luận: `runs/duration_review_stage31_03/crossing_details_all41.csv`.

| Ranh giới | Flow vượt mốc | Vượt mốc ước tính | Có khóa kết nối ở tập kế tiếp | Flow có chồng interval ước tính trực tiếp | Flow có thể chồng khi cộng bất định |
|---|---:|---|---:|---:|---:|
| Train → Validation | 11 | 1,396652–72,095455 giây | 10 | 0 | 2 |
| Validation → Test | 30 | 2,029018–26,004294 giây | 30 | 0 | 7 |
| Tổng | 41 | | 40 | 0 | 9 |

Tất cả thuộc nhóm Recon-VulScan. 11 flow Train bắt đầu từ 22:30:26 đến 22:32:09, kết thúc ước tính từ 22:32:12.396652 đến 22:33:23.095455. 30 flow Validation bắt đầu từ 22:33:26 đến 22:34:00, kết thúc từ 22:35:18.029018 đến 22:35:42.004294. Ngày của các mốc là 12/09/2023.

Có 56 cặp flow cùng khóa giữa 40 flow đó và các dòng của tập kế tiếp. 10 cặp nằm trong biên bất định, liên quan 9 flow trước ranh giới; một flow khớp hai dòng ở tập kế tiếp. Khoảng cách từ end ước tính của flow trước tới start hiển thị của flow sau trong 10 cặp là 0,359703–0,997696 giây.

### Ví dụ có thể truy vết

| Nguồn và dòng 0-based | Tập | Start | End ước tính | Ý nghĩa |
|---|---|---|---|---|
| Recon-VulScan_train, row 77 | Train | 22:30:26 | 22:32:16.133846 | Vượt mốc Validation 22:32:11 khoảng 5,13 giây; chưa thấy interval cùng khóa chồng lên trong mẫu |
| Recon-VulScan_train, row 88 | Train | 22:31:04 | 22:32:43.640297 | Cùng khóa với row 2440 của chính CSV đó nhưng row 2440 thuộc Validation, bắt đầu 22:32:44 |
| Recon-VulScan_test, row 49 | Validation | 22:33:43 | 22:35:42.003544 | Cùng khóa với row 627 của chính CSV đó nhưng row 627 thuộc Test, bắt đầu 22:35:43 |

Row 88 có Flow ID `192.168.137.140-192.168.137.216-6668-49974-6`, tương ứng TCP từ 192.168.137.140:6668 tới 192.168.137.216:49974. Gap ước tính tới row 2440 là 0,359703 giây. Sau khi cộng bất định một giây cho end, hai khoảng có thể chồng. Chưa thể xác định các gói của chúng là cùng phiên TCP.

Row 49 có Flow ID chiều ngược với cột Src/Dst hiện tại: Flow ID thể hiện 10.0.0.254:1883 và 10.0.0.3:44003, còn Src/Dst là 10.0.0.3:44003 → 10.0.0.254:1883. Khóa hai chiều nhận ra đây là cùng cặp endpoint, không tự kết luận cùng session.

### Diễn giải nguy cơ session

- 41 flow gồm 30 TCP, 9 UDP và 2 bản ghi Protocol=0. Hai khóa Protocol=0 không được coi là 5-tuple phiên TCP/UDP; cần xem riêng chất lượng/ý nghĩa metadata.
- Không flow nào trong 41 dòng có FIN/RST được ghi nhận. Một số có SYN, nhưng tổng số cờ không cung cấp thứ tự gói, sequence number hoặc chuỗi bắt tay/đóng kết nối.
- 9 flow tạo các cặp gần nhau trong biên bất định đều là TCP. Đây là các ứng viên cần ưu tiên kiểm tra phiên tiếp nối, không phải 9 session leakage đã được chứng minh.
- 31 flow có cùng khóa nhưng không có interval chồng quan sát được, cộng 1 flow chưa thấy khóa tương ứng, vẫn có thể vượt mốc thời gian chung của tập sau. Không thấy khóa trong prefix không chứng minh nó không tồn tại ở phần chưa đọc.
- Ngay cả hai flow không chồng interval cũng có thể là các phần của một phiên dài bị công cụ cắt theo timeout. [FlowGenerator.java](https://raw.githubusercontent.com/ahlashkari/CICFlowMeter/master/src/main/java/cic/cs/unb/ca/jnetpcap/FlowGenerator.java) có cơ chế kết thúc flow và mở bản ghi flow tiếp theo theo timeout cho cùng khóa. Chưa biết timeout/build đã dùng trên CSV này, nên đây là cơ chế có thể xảy ra, không phải kết luận nguyên nhân của 41 dòng.

Kết luận: **time overlap, cùng khóa kết nối và cùng session là ba mức bằng chứng khác nhau**. Cần PCAP, TCP sequence/handshake hoặc session/capture ID đáng tin cậy để kiểm tra mạnh hơn. Flow ID đang biểu diễn endpoint/protocol, không phải ID duy nhất của một phiên.

## 4. Đề xuất embargo thích ứng, chưa áp dụng

Giữ B2 làm cơ sở. Giữ nguyên sáu nhóm nhỏ Train-only, nhãn proxy, Original_Label và các định danh dòng nguồn.

Quy tắc đề nghị để xin duyệt:

1. Bắt đầu từ các block B2 đã duyệt, không shuffle dòng.
2. Giữ toàn bộ embargo cũ, với yêu cầu ít nhất 511 flow mỗi ranh giới trên quần thể đầy đủ. Mở rộng thêm theo timestamp nguyên bucket khi duration yêu cầu. Không lấy số flow 511 làm đại diện cho một khoảng thời gian cố định.
3. Với mỗi ranh giới, tính max end bảo thủ của tập trước. Yêu cầu end đó nhỏ hơn start tập sau theo mô hình thời gian đã nêu. Kiểm tra cả Train → Test, không chỉ hai cặp kề nhau.
4. So sánh chi phí lùi đuôi tập trước và đẩy đầu tập sau. Mọi dòng thay đổi chỉ được chuyển sang embargo có lý do, không chuyển sang một tập học khác.
5. Kiểm tra khóa kết nối và interval từng cặp trước/sau; ghi lại ứng viên cùng session, kể cả các phiên không chồng thời gian. Không xóa toàn bộ một 5-tuple lặp lại chỉ vì nó trùng.
6. Khi được duyệt audit metadata toàn bộ, chọn các bucket biên bằng mục tiêu đồng thời: an toàn interval, không mất các block cần đánh giá, tỷ lệ trên số flow giữ lại gần 60:20:20, và ít flow embargo. Nếu không thể đạt các điều kiện, trình bày chi phí/độ lệch và chờ lựa chọn. Không cố đạt tỷ lệ bằng random reallocation hoặc nhìn kết quả mô hình trên Test.

### Ba phương án đã thử riêng trên mẫu

| Phương án | Train | Validation | Test | Embargo | Embargo thêm | Flow vượt start tập sau còn quan sát được |
|---|---:|---:|---:|---:|---:|---:|
| B2 ban đầu | 12.215 | 2.923 | 4.216 | 2.189 | 0 | 41 |
| Đẩy đầu tập sau — right_prefix | 12.215 | 2.293 | 4.194 | 2.841 | 652 | 0 |
| Lùi đuôi tập trước — left_suffix | 11.223 | 2.853 | 4.216 | 3.251 | 1.062 | 0 |
| Chỉ loại bucket có flow vi phạm — bucket_purge | 11.725 | 2.869 | 4.216 | 2.733 | 544 | 0 |

Mỗi dòng trong cả bốn phương án vẫn được hạch toán đúng một lần: Train + Validation + Test + Embargo = 21.543. Không có dòng nguồn bị xóa khỏi sổ audit.

`bucket_purge` mất ít dòng nhất trong ba cách đã thử nhưng làm thủng các block thời gian giữ lại. Kiểm tra block liên tục FAIL, nên không khuyến nghị cho thiết kế B2 hiện tại. Segment vẫn có thể không chồng lấn, nhưng đó là tiêu chí khác với cả block của một split phải liên tục.

`right_prefix` mất ít flow hơn `left_suffix` trên mẫu và giữ các block liên tục. Tuy nhiên nó làm Validation mỏng đi đáng kể. Chỉ đề nghị làm một ứng viên để so sánh trên audit metadata toàn bộ, không tự chốt đây là phương án tốt nhất về tỷ lệ.

### Các mốc chỉ dành cho mô phỏng Recon-VulScan

| Ranh giới | Start cũ của tập sau | Ngưỡng start đề nghị theo end bảo thủ mẫu | Start đầu tiên thực sự có trong mẫu sau điều chỉnh | Flow thêm embargo trong mẫu |
|---|---|---|---|---:|
| Train → Validation | 22:32:11 | 22:33:25 | 22:33:25 | 630 |
| Validation → Test | 22:35:16 | 22:35:44 | 22:35:45 | 22 |

Không có dòng mẫu tại giây 22:35:44 nên start quan sát được là 22:35:45. Không suy ra bucket 22:35:44 trống trong toàn bộ dữ liệu. Không áp dụng các giờ này như ranh giới chính thức.

Ví dụ dễ báo cáo: trong một CSV đã sort theo thời gian, flow ở cuối Train có thể còn hoạt động khi những flow Validation đầu tiên bắt đầu. Ta không chuyển flow Train sang Validation. Ta giữ Train, đưa toàn bộ phần đầu Validation trước mốc an toàn vào embargo, giữ Original_Row_Index để truy vết. Cùng một giây phải có cùng split, dù các dòng nằm ở hai CSV gốc khác nhau trong cùng nhóm. Phương án ngược lại là đưa cả đuôi Train vào embargo; bảng trên lượng hóa chi phí của hai lựa chọn.

## 5. Tỷ lệ và độ phủ

Tất cả tỷ lệ dưới đây tính trên Train + Validation + Test, không gồm embargo.

| Phương án trên prefix smoke | Train | Validation | Test |
|---|---:|---:|---:|
| B2 ban đầu | 63,1136% | 15,1028% | 21,7836% |
| right_prefix | 65,3139% | 12,2607% | 22,4254% |
| left_suffix | 61,3547% | 15,5970% | 23,0483% |
| bucket_purge | 62,3339% | 15,2525% | 22,4136% |

Mẫu ban đầu đã không có tỷ lệ 60:20:20 vì lấy prefix từng CSV. Không thể dùng bảng này để tuyên bố chính sách toàn bộ đã đạt hoặc không thể đạt 60:20:20. Nó cho thấy right_prefix có chi phí về độ phủ Validation cần xem xét.

Trong phần Recon-VulScan của mẫu, số flow Validation giảm từ 702 xuống 72 khi dùng right_prefix; Train 2.007, Test từ 1.216 xuống 1.194. 72 dòng mẫu không đủ FW=256 hoặc 512. Đây là cảnh báo trên mẫu, chưa kết luận rằng Validation của toàn bộ kịch bản chỉ có 72 flow. Mẫu Benign ban đầu cũng chưa chứa Validation, nên smoke không dùng để đánh giá độ phủ nhãn toàn bộ.

Các số thiết kế toàn bộ của B2 cũ vẫn là Train 11.843.431, Validation 3.947.810, Test 3.947.817, Embargo 152.044. Tổng giữ lại 19.739.058, tương ứng xấp xỉ 60:20:20. **Đó là số kế thừa Giai đoạn 2, không phải kết quả đo lại Giai đoạn 3.1 và không phải số của phương án embargo mới.**

Số flow/tỷ lệ toàn bộ sau điều chỉnh hiện là **CHƯA ĐO**. Chưa có tiêu chí sai lệch tỷ lệ tối đa được phê duyệt. Nếu phát sinh xung đột giữa độc lập tốt hơn và tỷ lệ, cần trình bày các lựa chọn trước khi quyết định.

## 6. Kiểm thử và kiểm định leakage

Unit test: **48/48 PASS**, gồm 34 test cũ và 14 test bổ sung cho mô phỏng, định danh nguồn, bất định timestamp, bảo toàn dữ liệu, chặn population input và giới hạn prefix. Chạy bằng Python 3.10 với pandas/numpy/pyarrow sẵn có; không cài thêm thư viện và không dùng thư viện huấn luyện.

| Kiểm tra trên cùng 21.543 flow | B2 cũ | right_prefix | left_suffix | bucket_purge |
|---|---|---|---|---|
| Bảo toàn số dòng và identity nguồn duy nhất | PASS | PASS | PASS | PASS |
| Không cắt timestamp bucket trong nhóm | PASS | PASS | PASS | PASS |
| Segment không chồng lấn, hạch toán liên tục | PASS | PASS | PASS | PASS |
| Mỗi block Train/Val/Test liên tục trong nhóm | PASS | PASS | PASS | FAIL |
| Không chuyển một dòng sang tập học khác | PASS | PASS | PASS | PASS |
| Không có flow vượt start tập sau trong phần mẫu quan sát được | FAIL: 41 | PASS: 0 | PASS: 0 | PASS: 0 |
| Ít nhất 511 flow tại mọi ranh giới toàn bộ | CHƯA XÁC LẬP | CHƯA XÁC LẬP | CHƯA XÁC LẬP | CHƯA XÁC LẬP |
| Độc lập capture/session | CHƯA CHỨNG MINH | CHƯA CHỨNG MINH | CHƯA CHỨNG MINH | CHƯA CHỨNG MINH |

Đã kiểm tra từng cặp flow cùng khóa trên toàn bộ các nhóm trong mẫu, không chỉ 41 flow. B2 có 123.352 cặp cùng khóa cần xét, 0 cặp chồng interval ước tính trực tiếp và 10 cặp có thể chồng dưới bất định. right_prefix có 122.973 cặp cần xét, 0 cặp ở cả hai mức. Những con số này không phải số session độc lập.

Chỉ số min/max envelope giảm từ 13 xuống 4 ứng viên khi dùng right_prefix, nhưng không được đổi 4 thành 4 leakage. Các envelope còn lại thuộc khóa lặp lại qua nhiều kịch bản/ngày, trong đó có khóa DHCP broadcast và khóa Protocol=0. Kiểm tra từng cặp không thấy interval chồng trên mẫu. Đây là ví dụ cho việc min/max envelope có thể nối qua thời gian idle dài.

Source CSV giữ nguyên size/mtime so với inventory cũ và sau mọi lần đọc có giới hạn. Đây là kiểm tra stat, không phải SHA256 lại hàng GB dữ liệu nguồn. SHA256 của năm script CTU-13 khớp baseline đã ghi nhận. Không chạy lại full fingerprint scan; việc không có source identity trùng không thay thế kiểm tra các bản ghi nội dung trùng nhưng có identity khác.

Không có model, MI/RFE, scaler hoặc sliding window được chạy trong nhánh review. Chưa có window để kiểm định vượt segment/split; điều kiện phải được thực thi ở giai đoạn tạo window sau khi duyệt. Metadata như source path, split, timestamp, Original_Row_Index và proxy-label origin chỉ phục vụ audit, không được đưa vào đặc trưng học. MI, RFE và scaler sau này chỉ fit trên Train; Validation để lựa chọn cấu hình, Test không tham gia chọn ranh giới/đặc trưng/tham số.

## 7. Thay đổi hỗ trợ và cách tái lập

Chỉ mở rộng module hỗ trợ hiện có `ciciomt24_split.py` bằng nhánh **review-smoke** và thêm test vào `test_ciciomt24_split.py`. Không tạo script riêng cho từng kịch bản. `run_pipeline`, B2_CUTOFFS, EXPECTED_B2 và đường chia mặc định giữ nguyên. Các thay đổi membership mô phỏng chỉ có trong bộ nhớ và ledger audit riêng, không ghi đè manifest B2.

Các file của lượt review hiện hành:

- `review.json`: số đếm, đơn vị, giả định, kiểm định và so sánh máy đọc được.
- `crossing_details_all41.csv`: đủ 41 dòng với start/end, overrun, Flow ID, 5-tuple và kết luận có giới hạn.
- `matching_flow_pairs.csv`: 56 cặp cùng khóa, khoảng cách thời gian và cờ TCP của flow sau.
- `comparison.csv`: so sánh số flow, tỷ lệ trên flow giữ lại và crossing của bốn cách.
- `proposed_boundaries_SAMPLE_ONLY.csv`: mốc giả lập và chi phí, không phải cấu hình đã phê duyệt.
- `experimental_changes_SAMPLE_ONLY.parquet`: dòng nào được đưa thêm vào embargo trong mỗi mô phỏng.

Không có train.csv/val.csv/test.csv đầu ra. Các CSV kể trên là bảng audit nhỏ, không phải dataset đã chia.

Lệnh PowerShell tái lập, dùng thư mục output mới chưa có nội dung:

```powershell
Set-Location 'D:\COMPACT-MI-RFE\huyen_code'
python -m unittest -v test_ciciomt24_split
python .\ciciomt24_split.py review-smoke --input-run '.\runs\smoke_b2_stage3_02' --out '.\runs\duration_review_manual_01'
```

Nhánh review chỉ nhận smoke prefix, tối đa 12 nguồn và 3.000 dòng/nguồn, kiểm tra số dòng Parquet trước khi nạp, và không nhận cờ cho full scan. Không chạy chế độ plan hoặc full split trong giai đoạn này.

## 8. Cách trình bày với giảng viên

“Em chia theo các block thời gian trong từng nhóm kịch bản, giữ nguyên bucket timestamp và truy vết từng dòng gốc. Embargo 511 flow giúp tạo khoảng cách theo số dòng nhưng không kiểm soát được flow dài. Vì vậy em bổ sung kiểm tra duration theo microsecond, có đối chiếu công cụ và công thức tốc độ. Trên smoke test, 41 flow vượt mốc bắt đầu tập kế tiếp; chưa có bằng chứng chắc chắn cùng session bị chia đôi. Khi mô phỏng mở rộng embargo, các khoảng hoạt động vượt mốc quan sát được giảm về 0, đổi lại phải bỏ thêm dữ liệu và có thể giảm độ phủ Validation. Em chưa gọi cách chia này là session-disjoint hay leakage-free. Cần audit metadata toàn bộ để báo cáo tỷ lệ thực và chi phí trước khi xin chia dữ liệu chính thức.”

Điểm chờ duyệt: chính sách embargo thích ứng và bước audit metadata/manifest toàn bộ. Chưa xin coi 652 là số flow embargo bổ sung của toàn dataset và chưa có cơ sở chốt full split.
