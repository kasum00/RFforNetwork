# Giai đoạn 3.4 — Full split CICIoMT2024 theo C+

## Kết quả

Đã xuất và kiểm định toàn bộ **17,997,672** flow được giữ lại từ **19,891,102** dòng nguồn. Thiết kế cutoff C+ được đọc từ file đã phê duyệt; không tối ưu lại cutoff.

| Role | Flow | Tỷ lệ trên toàn bộ nguồn | Tỷ lệ trên phần giữ lại |
|---|---:|---:|---:|
| train | 10,798,603 | 54.288611% | 59.999999% |
| val | 3,599,535 | 18.096207% | 20.000003% |
| test | 3,599,534 | 18.096202% | 19.999998% |
| embargo | 1,893,430 | 9.518980% | — |

Train/Validation/Test trên 17,997,672 flow giữ lại đạt xấp xỉ **60:20:20**. Embargo không được ghi ra các tập và không bị xóa khỏi CSV gốc.

| Thư mục/nhóm output | Dung lượng (bytes) |
|---|---:|
| CSV flow train | 7,141,887,375 |
| CSV flow val | 2,387,457,164 |
| CSV flow test | 2,374,799,690 |
| CSV flow Train/Validation/Test | 11,904,144,229 |
| Toàn bộ thư mục sau publish (100 file, gồm manifest và validation) | 11,904,228,678 |

## Nhãn scenario proxy

`Label` gốc trong CSV vẫn là `No Label`. Cột `Proxy_Label` được thêm riêng và `Proxy_Label_Source=SCENARIO_PROXY`; đây là nhãn theo kịch bản, không phải ground truth từng flow.

| Split | Benign proxy | Attack proxy | Tổng |
|---|---:|---:|---:|
| embargo | 1,022 | 1,892,408 | 1,893,430 |
| test | 8,526 | 3,591,008 | 3,599,534 |
| train | 25,707 | 10,772,896 | 10,798,603 |
| val | 8,587 | 3,590,948 | 3,599,535 |

### Flow theo scenario

| Scenario | Proxy | Train | Validation | Test | Embargo |
|---|---|---:|---:|---:|---:|
| ARP_Spoofing | Attack | 1,617 | 0 | 0 | 0 |
| Benign | Benign | 25,707 | 8,587 | 8,526 | 1,022 |
| MQTT-DDoS-Connect_Flood | Attack | 1,750,338 | 812,976 | 826,423 | 777,633 |
| MQTT-DDoS-Publish_Flood | Attack | 39,839 | 4,002 | 18,639 | 31,770 |
| MQTT-DoS-Connect_Flood | Attack | 69,542 | 8,812 | 28,471 | 36,328 |
| MQTT-DoS-Publish_Flood | Attack | 1,052 | 0 | 0 | 0 |
| MQTT-Malformed_Data | Attack | 2,477 | 0 | 0 | 0 |
| Recon-OS_Scan | Attack | 35,526 | 2,016 | 17,170 | 30,612 |
| Recon-Ping_Sweep | Attack | 114 | 0 | 0 | 0 |
| Recon-Port_Scan | Attack | 212,683 | 92,503 | 90,096 | 90,519 |
| Recon-VulScan | Attack | 2,095 | 1,277 | 1,216 | 3,990 |
| TCP_IP-DDoS-ICMP | Attack | 3,852 | 0 | 0 | 0 |
| TCP_IP-DDoS-SYN | Attack | 2,960,578 | 811,890 | 1,048,576 | 338,236 |
| TCP_IP-DDoS-TCP | Attack | 3,291,938 | 1,163,890 | 732,869 | 354,316 |
| TCP_IP-DDoS-UDP | Attack | 130,570 | 39,038 | 44,930 | 13,293 |
| TCP_IP-DoS-ICMP | Attack | 3,018 | 0 | 0 | 0 |
| TCP_IP-DoS-SYN | Attack | 1,303,977 | 378,565 | 439,346 | 81,236 |
| TCP_IP-DoS-TCP | Attack | 918,183 | 262,471 | 327,959 | 131,233 |
| TCP_IP-DoS-UDP | Attack | 45,497 | 13,508 | 15,313 | 3,242 |

## Cấu trúc và truy vết

- Thư mục đầu ra: `D:\COMPACT-MI-RFE\CICIoMT24\CSV_SPLIT_CPLUS` (train/val/test, phân thư mục theo scenario).
- Mỗi CSV là một segment của đúng một source CSV và một split; từng segment được sort ổn định theo `(Timestamp, Source_CSV_ID, Original_Row_Index)`.
- Mỗi flow xuất ra có `Source_CSV`, `Source_CSV_ID`, `Original_Row_Index`, `Split`, `Proxy_Label`, `Proxy_Label_Source`; `split_manifest.csv` còn ghi cả số flow embargo theo source.
- File kiểm định: `validation.json`; mapping nhãn: `label_mapping.csv`; cutoff snapshot: `Cplus_cutoffs_used.csv`; nguồn: `source_manifest.csv`.
- Không nối các CSV/segment với nhau khi tạo window sau này; xử lý từng file độc lập.

## Kiểm định toàn bộ

| Kiểm tra | Kết quả |
|---|---|
| `source_rows_total_19891102` | **PASS** |
| `cplus_role_counts_exact` | **PASS** |
| `source_row_unique_across_output_splits` | **PASS** |
| `timestamp_buckets_not_cut` | **PASS** |
| `source_segments_nonoverlap` | **PASS** |
| `embargo_at_least_511_each_guard` | **PASS** |
| `activity_crossing_duration_plus_1s` | **PASS** |
| `no_exact_duplicate_across_splits` | **PASS** |
| `scenario_and_proxy_counts` | **PASS** |
| `csv_readback_schema_sort_and_raw_fields` | **PASS** |
| `source_csv_and_original_row_traceability` | **PASS** |
| `raw_source_sha256_unchanged` | **PASS** |
| `no_embargo_rows_exported` | **PASS** |
| `future_windows_must_stay_within_each_source_file` | **PASS_BY_SOURCE_SPLIT_SEGMENTS** |
| `capture_session_independence` | **NOT_ESTABLISHED** |
| `actual_sliding_window_overlap` | **NOT_CREATED_NOT_TESTED** |

Activity crossing theo `Timestamp + Flow Duration + 1 giây`: **0**. Exact duplicate record đi qua nhiều split: **0 nhóm**. Output gồm **93 file CSV dữ liệu**, **11,904,144,229 bytes**; thời gian chạy khoảng **531.3 giây**.

SHA256 toàn bộ 71 CSV nguồn đã khớp index trước khi chạy và khớp lại sau khi xuất; size/mtime cũng không đổi. SHA256 của cả năm script CTU-13 cũng khớp snapshot Giai đoạn 3.2. Không sửa CSV gốc hoặc năm script CTU-13.

## Leakage còn lại và giới hạn

Audit C+ ghi nhận 914,650 khóa 5-tuple dùng chung qua các split và 166 session candidate theo heuristic 1 giây (không đồng nghĩa session đã xác nhận). Capture/session độc lập chưa được chứng minh. Timestamp có độ phân giải giây và guard dùng Duration + 1 giây; điều đó không chứng minh độc lập capture.
Không tạo sliding window trong giai đoạn này, nên overlap giữa các window chưa được đo. Việc không dùng chung source row được bảo đảm bởi membership duy nhất và segment tách theo source/split; ở bước tiếp theo phải tạo window độc lập trong từng file, không nối qua file, split hoặc capture.

## Lệnh PowerShell kiểm tra manifest

```powershell
$root = 'D:\COMPACT-MI-RFE\CICIoMT24\CSV_SPLIT_CPLUS'
$m = Import-Csv (Join-Path $root 'split_manifest.csv')
$m | Where-Object Split -ne 'embargo' | Group-Object Split | ForEach-Object {
  [pscustomobject]@{ Split=$_.Name; Flows=(($_.Group | Measure-Object -Property Flow_Count -Sum).Sum) }
} | Sort-Object Split | Format-Table -AutoSize
(Get-Content (Join-Path $root 'validation.json') -Raw | ConvertFrom-Json).checks
Get-ChildItem $root -Recurse -File -Filter '*.csv' | Measure-Object Length -Sum
```

## Có thể chuyển bước không?

Có thể bắt đầu **thiết kế/triển khai sliding-window riêng theo từng file segment**, sau khi xác nhận proxy label phù hợp mục tiêu nghiên cứu. Chưa thể tuyên bố leakage-free hoàn toàn hoặc capture/session-independent; không chạy feature selection hay mô hình trong Stage 3.4.
