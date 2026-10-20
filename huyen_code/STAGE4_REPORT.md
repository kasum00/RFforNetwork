# Giai đoạn 4 — Adapt sliding window CTU-13 cho CICIoMT2024

Ngày thực hiện: 2026-10-19  
Trạng thái: **đã triển khai module và smoke test; chưa chạy full window generation**.

## Phạm vi đã làm

- Tạo `ciciomt24_sliding_window.py` và `test_ciciomt24_sliding_window.py` trong `huyen_code`.
- Không sửa cutoff C+, không ghi vào CSV split, không sửa năm script CTU-13, không tạo sliding window toàn bộ, không fit scaler/MI/RFE và không huấn luyện.
- Script gốc được giữ nguyên. SHA256 hiện tại:

| Script CTU-13 | SHA256 |
|---|---|
| `2_labeling.py` | `EA8C3468609A99C69E9F2B5AE8B86B7B7B9B55F5C00147884B0AE4D3808D2305` |
| `3_sliding_windown_and_aggregate_features.py` | `9DF4BFD2BE6AD72DAA943C51265352CFDB92740B1C8429B03C098A4D0662562A` |
| `7_senario1_svm.py` | `E44BEF8B6A6F35D4160C06DB30691510601DDD1FD31BDB12F681E8831DFA3F0F` |
| `[[CTU13]8_experiment2_svm.py` | `AADF2D829DD91C67620124AB033E78636C7A9D63C0CCAE73BE799916F88B6664` |
| `[[CTU13]9_ex2_rf.py` | `6DB8448FC1FD5CD49416B7CE37AC5DA6E71E6DC1554557589D73F73C936E8CB4` |

## Quy tắc kế thừa từ script gốc

Window start dùng đúng `range(0, n - FW + 1, FS)`, riêng bên trong từng CSV segment. Bốn cấu hình là FW/FS = 256/8, 256/16, 512/8, 512/16. Mỗi feature dùng `np.nan_to_num(nan=0, posinf=0, neginf=0)`, `np.mean`, `np.std(ddof=0)`, `np.median`, `scipy.stats.skew` mặc định biased và `scipy.stats.kurtosis` mặc định Fisher/biased. Nhãn cửa sổ theo existential rule: có ít nhất một flow `Proxy_Label=Attack` thì `Label=1`, nếu không thì `Label=0`.

Đối chiếu schema split cho thấy có 79 trường CICFlowMeter dạng số (76 trường hành vi + Src Port, Dst Port, Protocol). Ba script phân loại CTU-13 hiện tại loại các thống kê của Src Port/Dst Port/Protocol khỏi feature matrix; module vì thế bỏ ba trường này trước khi tổng hợp. Feature matrix có **76 × 5 = 380** cột số. Không đưa Flow ID, IP, Timestamp, Label gốc, Source_CSV, Source_CSV_ID, Original_Row_Index, Split hoặc metadata scenario/provenance vào 380 feature.

Output smoke có 394 cột: 14 metadata/nhãn + 380 feature. `Label` gốc trong CSV flow vẫn là `No Label`; chỉ window output có `Label` nhị phân và `Label_Origin=SCENARIO_PROXY`. Source_CSV_ID và original row index được ghi kèm tiền tố chữ (`sid...`, `row:...`) để tránh bị pandas nhận nhầm là feature số. Trong smoke, phép đọc ngược xác nhận cột số đúng là 380 feature cộng `Window_Index`, `Num_Flows`, `Label`; hai metadata số đầu đã nằm trong danh sách loại bỏ của classifier gốc.

Mỗi output CSV window tương ứng đúng một input segment, split và source CSV. Không nối các segment trước khi tạo window. Engine full (chưa chạy) đọc một input segment theo chunk, giữ tối đa 511 flow làm carry cho FW=512, tạo cả bốn cấu hình trong cùng lượt đọc và ghi batch; không giữ `all_windows` trong RAM. Window_ID chứa config, split, source id và index cục bộ.

## Window count dự kiến từ manifest C+

Tính bằng công thức gốc trên từng segment manifest; audit xác nhận đúng bảng đã được duyệt.

| Cấu hình | Train | Validation | Test | Tổng |
|---|---:|---:|---:|---:|
| FW256 / FS8 | 1,348,317 | 449,343 | 449,251 | 2,246,911 |
| FW256 / FS16 | 674,169 | 224,677 | 224,631 | 1,123,477 |
| FW512 / FS8 | 1,347,101 | 448,735 | 448,563 | 2,244,399 |
| FW512 / FS16 | 673,559 | 224,373 | 224,287 | 1,122,219 |
| **Tổng** | **4,043,146** | **1,347,128** | **1,346,732** | **6,737,006** |

Sáu scenario nhỏ được giữ trong Train theo thiết kế nhưng không có window ở Validation/Test: ARP_Spoofing, MQTT-DoS-Publish_Flood, MQTT-Malformed_Data, Recon-Ping_Sweep, TCP_IP-DDoS-ICMP và TCP_IP-DoS-ICMP. Recon-Ping_Sweep chỉ có 114 flow Train nên cũng không đủ FW=256. Đây là độ phủ theo thiết kế B2/C+, không phải lỗi do windowing.

## Kiểm thử và smoke test

- `py_compile`: PASS.
- Unit test: **6/6 PASS**; gồm đối chiếu trực tiếp thống kê NumPy/SciPy với reference của script gốc (NaN/Inf, cột hằng), công thức số window, existential label, chunk carry, đọc file/schema/provenance theo streaming.
- Metadata audit: **PASS**, 93 segment không rỗng, 19,891,102 source rows theo manifest; số window dự kiến khớp toàn bộ 12 split/config. Các safety checks được đọc từ `validation.json` Stage 3.4 và đều PASS.
- Smoke lấy 2.048 dòng đầu từ mỗi một segment Train: Benign `sid022` (25.707 dòng trong manifest) và Attack `sid023`/MQTT-DDoS-Connect_Flood (1.750.338 dòng). Không quét hết hai file này. Cả bốn cấu hình cho mỗi lớp đều chạy và đọc ngược được; mỗi config tạo 225/113/193/97 window tương ứng.
- Smoke readback: schema 394 cột; nhãn Benign=0/Attack=1; Label_Origin, Split, Scenario, source id và row provenance hợp lệ; Window_ID không trùng trong config; không có metadata số ngoài ba cột đã nêu; window output không ghép source segment. **Không có full manifest row readback trong Stage 4 vì full windowing chưa chạy.**
- Formal smoke artifacts: `D:\COMPACT-MI-RFE\huyen_code\runs\stage4_smoke_final`; 8 CSV smoke + `smoke_summary.json`, khoảng 5.9 MB.

## Thời gian, RAM và dung lượng

- Formal smoke: 1.256 window, khoảng 3.9 giây; tốc độ biểu kiến khoảng 323 window/giây. Ngoại suy thô tuyến tính cho 6.737.006 window là khoảng **5.8 giờ**. Đây không phải benchmark full; thời gian thực tế phụ thuộc tốc độ đọc/ghi, segment, CPU và độ rộng chuỗi số.
- Peak RSS đo bằng sampling tiến trình trong smoke: khoảng **242 MiB** với prefix 2.048 dòng. Full default đọc chunk 50.000 dòng và batch 64 window; phần ma trận số đầu vào/carry và batch thống kê được giữ hữu hạn. Full RSS chưa đo; nên dành tối thiểu 1 GiB RAM trống và theo dõi khi chạy.
- Dự phóng kích thước thực nghiệm từ hai scenario smoke: khoảng **26.6 GB** CSV; thêm 25% dự phòng khoảng **33.3 GB**. Đây là ước lượng, không phải giới hạn trên vì Attack mới được lấy mẫu từ một scenario.
- Lệnh `run` có disk preflight bảo thủ: 11.000 byte/window + 20% dự phòng, tương đương khoảng **88.9 tỷ byte** cho manifest hiện tại. Khi audit, ổ D còn khoảng **120.1 GiB**; hiện đủ ngưỡng, nhưng cần kiểm tra lại ngay trước full run.

## Rủi ro và giới hạn

- `SCENARIO_PROXY` là nhãn proxy cấp scenario, không phải ground truth của từng flow.
- Window overlap bên trong cùng split/segment là chủ ý theo FS; chưa có full window output để kiểm tra thực nghiệm. Tuy vậy, mỗi window được sinh độc lập trong đúng một source CSV/split nên engine không tạo window bắc qua split hoặc segment.
- Stage 3.4 đã xác nhận source row không dùng chung giữa split và activity crossing bằng 0 theo Duration + 1 giây. Điều đó không chứng minh capture/session độc lập. Stage 3.4 còn 914.650 khóa 5-tuple dùng chung giữa split và 166 session candidates theo heuristic; trùng 5-tuple không tự nó xác nhận cùng session.
- Smoke chỉ xác minh mẫu nhỏ và đường đi streaming trên fixture; chưa chạy hết 93 segment, chưa đo dung lượng/tốc độ/RAM thực tế của full output.

## Lệnh full sau khi được phê duyệt

Lệnh dưới đây chỉ là hướng dẫn; **chưa được chạy trong Giai đoạn 4 hiện tại**. Output root chưa tồn tại và module từ chối ghi đè nếu đã có.

```powershell
python 'D:\COMPACT-MI-RFE\huyen_code\ciciomt24_sliding_window.py' run `
  --input-root 'D:\COMPACT-MI-RFE\CICIoMT24\CSV_SPLIT_CPLUS' `
  --output-root 'D:\COMPACT-MI-RFE\CICIoMT24\WINDOWS_CPLUS' `
  --chunksize 50000 `
  --batch-windows 64 `
  --confirm-full
```

Lệnh này chỉ tạo window; không chạy scaler, MI/RFE hoặc classifier. Chỉ chạy sau khi người dùng duyệt full generation.
