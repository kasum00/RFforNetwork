# STAGE 5.4 — RESOURCE PREFLIGHT MI-RFE + RF, FW512/FS8

Ngày kiểm tra: 20/10/2026. Số liệu tài nguyên đo lúc 07:05:35, múi giờ +07:00 (UTC 00:05:35). Đây là snapshot, không phải tài nguyên được đặt trước cho lần chạy sau.

## Kết luận

**ADMISSION_FAIL với runner và policy hiện tại. KHÔNG KHỞI CHẠY HUẤN LUYỆN.**

Guard bảo thủ yêu cầu 23,993641 GiB RAM khả dụng; lúc đo chỉ có 9,327602 GiB, thiếu 14,666039 GiB. Ngay cả toàn bộ RAM Windows dùng được là 23,730339 GiB cũng thấp hơn guard khoảng 0,263302 GiB. Vì vậy chỉ đóng ứng dụng không đủ để đạt policy này trên mức RAM usable hiện tại; Stage5.4 không đóng ứng dụng hay thay đổi hệ thống.

Đây là kết luận admission theo policy, không phải bằng chứng peak FS8 chính xác là 23,99 GiB hoặc chương trình chắc chắn không thể chạy nếu paging. Runner tối ưu chưa triển khai/đo: INSUFFICIENT_EVIDENCE. Không tự hạ guard hoặc thay thiết kế để chuyển thành PASS.

Kiểm tra metadata dữ liệu PASS trong phạm vi manifest/header/file stat và audit đã có. Không có kết quả MI/RFE/RF hoặc Test mới.

## 1. Trạng thái hai thí nghiệm đã hoàn thành

| Chỉ tiêu | Stage5.2: FW512/FS16 | Stage5.3: FW256/FS16 |
|---|---:|---:|
| Trạng thái chính thức | PASS | PASS |
| Train / Validation / Test | 673.559 / 224.373 / 224.287 | 674.169 / 224.677 / 224.631 |
| best_k, chọn bằng Validation | 8 | 18 |
| MI, giây | 1.312,974 | 1.341,728 |
| RF-RFE, giây | 292,334 | 273,536 |
| k-search fit + Validation, giây | 833,077 | 770,131 |
| Final refit, giây | 8,647 | 10,598 |
| Tổng, giây | 2.529,148 | 2.476,996 |
| Peak RSS bảo thủ, GiB | 9,993862 | 10,006531 |

Run đã đối chiếu result.json, preflight_audit.json và log:

- D:\COMPACT-MI-RFE\huyen_code\runs\stage52_rf_FW512_FS16_vscode_20261019_222901
- D:\COMPACT-MI-RFE\huyen_code\runs\stage53_mi_rfe_rf_FW256_FS16

Hash 12 artifact mỗi run khớp bộ kiểm định đã có. Không ghi đè hai run, không nạp model để predict và không đánh giá Test thêm lần nữa.

### Peak theo từng giai đoạn đã quan sát

| Giai đoạn/loại đo, GiB | Stage5.2 | Stage5.3 |
|---|---:|---:|
| MI (bao gồm scaler đầu vào) | 9,993862 | 9,987415 |
| RF-RFE fit | 2,947968 | 2,916950 |
| Lớn nhất các k-search fit | 2,941319 | 3,027924 |
| Final refit fit | 2,521587 | 2,557449 |
| Whole-run tracker | 9,948452 | 10,006531 |
| Giá trị bảo thủ dùng cho guard | 9,993862 | 10,006531 |

Scaler chưa có peak riêng. Tracker whole-run và tracker từng fit lấy mẫu độc lập nên whole-run có thể thấp hơn đỉnh đo ở MI; dùng giá trị lớn nhất, không chọn con số thấp hơn. Chu kỳ lấy mẫu khoảng 50 ms, có tính RSS con theo cách đo của runner; vẫn có thể bỏ sót đỉnh ngắn.

Loader chỉ có các snapshot sự kiện: Train khoảng 1,11 GiB, Val 1,42–1,45 GiB, Test 1,74–1,75 GiB ở hai run cũ. Snapshot refit/Test evaluation không phải continuous peak; số thô nằm trong memory_estimation.json. Không đồng nhất những snapshot này với peak độc lập của từng phase.

## 2. Kiểm tra dataset FW512/FS8

Input: D:\COMPACT-MI-RFE\CICIoMT24\WINDOWS_CPLUS\FW_512_FS_8

| Split | Window | Benign proxy | Attack proxy | File | CSV, byte |
|---|---:|---:|---:|---:|---:|
| Train | 1.347.101 | 3.150 | 1.343.951 | 35 | 4.133.525.156 |
| Validation | 448.735 | 1.010 | 447.725 | 19 | 1.464.249.591 |
| Test | 448.563 | 939 | 447.624 | 21 | 1.502.972.702 |
| Tổng | 2.244.399 | 5.099 | 2.239.300 | 75 | 7.100.747.449 |

Các con số khớp manifest, summary và full readback Stage4.2. Không đếm lại hàng triệu dòng ở Stage5.4. Với FW512, 75/93 source segments có window; không nối các segment ngắn để tăng mẫu.

Kiểm tra hiện tại:

- PASS: tồn tại đầy đủ 75 file mục tiêu; từng header có chính xác 394 cột = 380 feature thống kê hành vi + 14 metadata/label, đúng canonical feature list.
- PASS: tất cả 322 CSV window của bốn cấu hình khớp size/mtime với cả preflight Stage5.2 và Stage5.3.
- PASS: số window và nhãn proxy của cả ba tập khớp số dự kiến.
- PASS: năm script CTU-13 giữ nguyên SHA256; code triển khai không sửa.
- Không parse dòng dữ liệu CSV; chỉ đọc header, manifest, thống kê sẵn có và file stat. Đối với Test, không đọc giá trị feature, không fit/predict/evaluate.

Size/mtime bằng nhau không thay thế một lần hash toàn bộ nội dung CSV mới. Bằng chứng kiểm định đầy đủ nội dung, Window_ID, provenance, không nối source/split và đối chiếu NumPy/SciPy được tái sử dụng từ Stage4.2, không tuyên bố chạy lại ở đây.

14 cột không được vào X: Window_ID, Split, Scenario, Source_CSV, Source_CSV_ID, Original_Row_Index_Start, Original_Row_Index_End, Window_Start_Timestamp, Window_End_Timestamp, Window_Index, Num_Flows, Proxy_Label, Label_Origin, Label. X chỉ gồm 380 cột canonical; y là Label.

Theo full readback cũ cho riêng FW512/FS8: 206.439.420 giá trị NaN feature, Inf=0. NaN chủ yếu do skew/kurtosis trên cột hằng, không phải số dòng hỏng. Giữ quy tắc fillna(0) tương thích source và bắt buộc kiểm tra finite trước fit nếu được phép chạy sau này; Stage5.4 không sửa CSV hay xử lý lại các mảng.

Lưu ý audit trước: lỗi danh sách feature_names lấy trước lọc numeric thuộc script MI-RFE-SVM; script RF gốc đã lấy list(X.columns). Bản triển khai dùng whitelist 380 feature, không đưa 14 cột provenance vào X.

## 3. Tài nguyên Windows tại thời điểm đo

GiB = byte / 1.073.741.824. RAM resident, dung lượng DIMM và commit không phải cùng khái niệm.

| Đại lượng | Kết quả |
|---|---:|
| Tổng RAM vật lý Windows dùng được | 23,730339 GiB |
| RAM vật lý khả dụng | 9,327602 GiB |
| Commit limit | 42,730339 GiB |
| Commit total | 21,807045 GiB |
| Available commit | 20,923294 GiB |
| File D:\pagefile.sys phát hiện được | 19 GiB |
| Ổ C còn trống | 34,149567 GiB |
| Ổ D còn trống | 92,493118 GiB |

Installed physical DIMM capacity chưa xác minh trực tiếp được: GetPhysicallyInstalledSystemMemory trả lỗi 5 (Access denied); CIM Win32_PhysicalMemory cũng không khả dụng. Do đó 23,730339 GiB là total usable, không tự gọi là dung lượng DIMM đã xác nhận.

Commit lấy từ GetPerformanceInfo và GlobalMemoryStatusEx trong ngữ cảnh công cụ; hai phép đo khớp. Commit là cam kết bộ nhớ có backing, không phải lượng RAM trống. Thông số Windows tính theo page được đổi sang byte theo PageSize. [Windows PERFORMANCE_INFORMATION](https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-performance_information)

D:\pagefile.sys có kích thước 20.401.094.656 byte qua file stat. Không xác định được usage hiện tại: psutil swap probe lỗi performance counter, CIM Win32_PageFileUsage bị giới hạn truy cập. C:\pagefile.sys không tìm thấy trong view hiện tại; không suy ra toàn hệ thống không có pagefile khác. Không sửa pagefile.

### Giới hạn quan sát process

API hiện chỉ thấy 4 process trong phạm vi sandbox/job của công cụ, không phải danh sách đầy đủ ứng dụng trên host:

| Process nhìn thấy | PID | RSS |
|---|---:|---:|
| python.exe — chính probe metadata | 17308 | 136,785 MiB |
| pwsh.exe — shell probe | 31660 | 78,902 MiB |
| conhost.exe | 33556 | 7,258 MiB |
| System Idle Process | 0 | 8 KiB |

Vì giới hạn này, không thể phân bổ toàn bộ RAM đang dùng cho các ứng dụng lớn trên host và không chứng minh được không có Python training khác ngoài view. Không tự kết thúc process nào. Trường private_bytes trong JSON không được hiểu là physical RSS.

### Môi trường

Python 3.10.0 64-bit; NumPy 1.26.4; pandas 2.3.3; SciPy 1.15.3; scikit-learn 1.7.2; joblib 1.5.3. Windows kernel 10.0.26200; CPU báo 20 core vật lý, 28 logical; effective n_jobs(-1)=28. Giữ nguyên cấu hình n_jobs, không tự giảm song song.

## 4. Ước tính RAM: phân biệt ba loại bằng chứng

### 4.1. Payload mảng tính chính xác từ shape/dtype

Đây là số byte dữ liệu, không phải tổng RSS và không bảo đảm tất cả cùng resident.

| Thành phần | Payload, GiB |
|---|---:|
| Train X, 1.347.101 x 380 float32 | 1,906970 |
| Train y int8 | 0,001255 |
| Validation X, 448.735 x 380 float32 | 0,635233 |
| Validation y int8 | 0,000418 |
| Test X float32, chỉ tải sau refit nếu được duyệt | 0,634991 |
| X float32 của cả ba tập | 3,177194 |
| scaled Train float32 dùng trước MI | 1,906970 |
| Một Train work/temp float64 đầy đủ | 3,813940 |
| Một mask bool Train x 380 | 0,476743 |
| Top100 Train float32 | 0,501834 |
| Top100 Validation float32 | 0,167167 |
| Raw refit Train+Val với k=100 float32 | 0,669001 |
| Raw + scaled refit với k=100 | 1,338002 |
| Chunk 8.192 x 380 float64, chưa gồm DataFrame/copy khác | 0,023193 |

Ba vector scaler mean/var/scale khoảng 9.120 byte, rất nhỏ; không được nhầm model scaler nhỏ với chi phí fit/transform.

Refit full Train+Val có 1.795.836 dòng. Nếu k=8 hoặc 18 thì raw selected matrix tương ứng khoảng 0,053520 hoặc 0,120420 GiB; đây chỉ là minh họa theo k cũ, KHÔNG dự đoán k của FS8. K=1..100 vẫn phải được thử nếu có phê duyệt chạy.

### 4.2. Điểm tạo peak và bộ nhớ chưa định lượng được

Source sklearn cài thực tế cho thấy:

- StandardScaler copy=True materialize output; thống kê variance có thể dùng float64 temporary và bool mask full-shape.
- MI nhận scaled float32 rồi tạo float64 work; advanced-column indexing, nanstd/abs và nhiễu random còn tạo temporary lớn. Đây là lý do memmap X/y không khống chế toàn bộ peak.
- MI mặc định copy=True, n_neighbors=3, n_jobs=None; dữ liệu dense được coi liên tục. RF n_jobs=-1 không có nghĩa MI cũng chạy 28 feature đồng thời.
- RF-RFE fit 100-tree estimator trên các subset cột; k-search fit 300-tree RF mới cho từng k. Tree builder/weights/bootstrap/labels có scratch riêng.
- Với 28 worker, riêng minh họa sample weights float64 + bootstrap indices int32 + bincount int64 có thể khoảng 0,702568 GiB ở Train hoặc 0,936601 GiB ở refit nếu cùng tồn tại; chưa bao gồm nodes, labels, builder, predict buffers hoặc allocator. Không phải toàn bộ X bị copy 28 lần chỉ vì có 28 threads.
- Số node thực của RF FS8, RSS file pages, overlap giữa temporary, bộ nhớ NN/KDTree, allocator và process khác chưa xác định.

[StandardScaler 1.7](https://scikit-learn.org/1.7/modules/generated/sklearn.preprocessing.StandardScaler.html) và [MI 1.7](https://scikit-learn.org/1.7/modules/generated/sklearn.feature_selection.mutual_info_classif.html) mô tả các tùy chọn; allocation cụ thể đã đối chiếu code sklearn 1.7.2 đang cài, lưu path/line/SHA256 trong memory_estimation.json.

Ví dụ tính payload: raw Train+Val memmap pages + scaled Train float32 + bốn mảng Train float64 tương đương 19,704935 GiB. Đây chỉ là một kịch bản vòng đời để giải thích khả năng phình RAM, KHÔNG khẳng định bốn temporary luôn cùng sống, không phải cận trên hoặc peak đã đo. Không cộng tất cả mảng ở mọi phase với nhau.

### 4.3. Guard bảo thủ, không thay thế bằng suy đoán tuyến tính

Với từng run tham chiếu:

    r = max(Ntrain_mới/Ntrain_cũ, Nval_mới/Nval_cũ, Ntổng_mới/Ntổng_cũ)
    scaled_peak = ceil(peak_RSS_bảo_thủ_đã_đo * r)
    headroom = max(2 GiB, ceil(20% * scaled_peak))
    required_available = scaled_peak + headroom

Lấy required_available lớn nhất của hai run. Không tự đổi policy để dùng cận mảng thấp hơn.

| Run tham chiếu | r | Scaled peak, byte | Headroom, byte | Cần khả dụng, byte |
|---|---:|---:|---:|---:|
| Stage5.2 | 1,999974761 | 21.461.384.716 | 4.292.276.944 | 25.753.661.660 |
| Stage5.3 | 1,998165149 | 21.469.146.750 | 4.293.829.350 | 25.762.976.100 |

Guard cuối: 25.762.976.100 byte = 23,993641 GiB, gồm khoảng 19,994701 GiB scaled peak và 3,998940 GiB headroom. Thiếu 15.747.539.300 byte so với available lúc đo; guard còn cao hơn total usable 282.718.564 byte.

Đây là kịch bản an toàn theo quan sát và hệ số policy, KHÔNG phải khẳng định quan hệ peak tuyến tính chính xác, cận trên toán học hoặc đo full FS8. Nếu cần estimate chặt hơn, phải có phê duyệt triển khai/benchmark riêng; chưa có bằng chứng để thay guard.

## 5. Disk và so sánh FS8/FS16

CSV FW512/FS8 đã tồn tại: 6,613086 GiB, không sao chép hoặc tái tạo.

Nếu sau này chạy, cache memmap mới cho X/y của ba tập có payload 3.413.730.879 byte = 3,179285 GiB. Kiểm tra disk tối thiểu của runner tương đương 1,5 x tổng X float32 = 4,765792 GiB; ổ D còn 92,493118 GiB nên PASS cho mức sàn này.

Đây không phải cận trên toàn bộ disk: còn artifact, forest, log và workspace nếu được đề xuất riêng. Workspace scaled32 + MI64 có thể thêm khoảng 5,72 GiB nếu thực sự triển khai; Stage5.4 không tạo cache hoặc workspace lớn. Pagefile hiện hữu đã phản ánh trong dung lượng trống, không cộng thêm lần nữa.

FW512/FS8 gần gấp đôi số dòng FW512/FS16; số feature vẫn 380, Top100 và k-grid không đổi. Có thể ước lượng thời gian có điều kiện bằng scale tuyến tính đến NlogN từ hai run trước:

| Hạng mục | Khoảng minh họa, phút |
|---|---:|
| MI | 43,8–47,0 |
| RF-RFE | 9,1–10,3 |
| k-search fit + Validation | 25,6–29,2 |
| Tổng runner | 82,5–88,7 |

Khoảng này không phải benchmark FS8, không phải confidence interval hoặc cam kết thời gian. Giả định tài nguyên tương tự, không paging nặng; paging, contention, cấu trúc cây/neighbor khác hoặc OOM có thể khiến lâu hơn rất nhiều hay không hoàn thành. Không sử dụng thời gian dự báo để biện minh khởi chạy khi guard FAIL.

## 6. Khả năng chạy trên máy hiện tại và tối ưu đề xuất

Runner nguyên trạng: ADMISSION_FAIL theo physical-available guard. Available commit 20,923294 GiB và pagefile 19 GiB không chứng minh đủ RAM resident; paging có thể rất chậm và commit vẫn có giới hạn.

Đề xuất chỉ ở mức phân tích:

| Hướng | Phạm vi tiết kiệm có thể có | Đánh giá |
|---|---|---|
| Xóa model k cũ sớm, vòng đời buffer rõ | Tránh một forest cũ; không giải quyết peak MI | Có thể giữ phương pháp, cần kiểm định |
| Hoãn load Validation tới sau MI | Tối đa payload raw Val khoảng 0,635 GiB cùng buffer | File-backed RSS không đảm bảo giảm tương ứng |
| Scale in-place trên Top100 copy riêng | Khoảng 0,502 GiB Train hoặc 0,167 GiB Val ở phase tương ứng | Không được sửa raw memmap |
| Ranking workspace, prefix views | Tránh một số bản sao per-k | Cần kiểm tra layout và sklearn có copy lại không |
| Scratch memmap riêng | Chuyển payload ra file, không xóa mọi temporary | Không bảo đảm giảm RSS; tăng I/O/disk |
| Refactor temporary MI, incremental fit | Có tiềm năng lớn hơn | Có thể đổi số học/RNG; chưa coi tương đương |

Phân tích đầy đủ RAM, thời gian, tái lập, khả năng đổi kết quả, cách kiểm định và nhu cầu sửa code cho từng phương án tại:

[memory_optimization_options.md](D:/COMPACT-MI-RFE/huyen_code/runs/stage54_preflight/memory_optimization_options.md)

Không triển khai phương án nào. Không sampling, giảm Top100, tăng RFE step, giảm cây/k-grid, đổi n_jobs hoặc seed. gc.collect không phải bảo đảm RSS được trả ngay cho Windows.

## 7. Data leakage: bốn điều kiện phải tách biệt

Tái sử dụng audit Stage3.4 và Stage4.2, không chia lại dữ liệu:

| Điều kiện | Bằng chứng hiện có | Không chứng minh được điều gì |
|---|---|---|
| Không chung source row giữa split | PASS Stage3.4; window chỉ từ một source segment theo Stage4.2 | Không chứng minh session/capture độc lập |
| Không exact duplicate giữa split | 0 nhóm trùng qua split theo audit flow | Không loại trừ gần-trùng, dấu vết capture/thiết bị |
| Không giao cắt khoảng hoạt động | 0 crossing theo Duration + 1 giây và guard đã thiết kế | Không chứng minh hai flow không thuộc cùng session dài |
| Capture/session independence | NOT_ESTABLISHED | Không được suy ra từ ba hàng trước |

Dữ liệu C+ giữ 17.997.672 flow, embargo 1.893.430 trên tổng 19.891.102. Embargo bảo vệ timestamp bucket và activity interval theo metadata, không chứng nhận capture độc lập.

Source row của một window chỉ nằm trong một source segment. Window chồng lấn trong cùng split là chủ ý của FW/FS; vấn đề cần tránh là dùng chung source row qua split. Không đồng nhất hai loại overlap.

### Những rủi ro còn tồn tại

- Audit có 914.650 khóa 5-tuple xuất hiện ở nhiều split. Các cặp Train/Val 612.345, Train/Test 541.053, Val/Test 723.220 không cộng lại thành số unique vì các tập giao nhau. Trùng tuple không tự chứng minh chung session.
- Heuristic theo nhóm scenario ghi nhận 166 session-continuity candidates ở Val→Test; trong đó 19 TCP candidates thiếu dấu FIN/RST theo quy tắc kiểm tra. Train→Val và Train→Test không có candidates theo heuristic đó. Đây là nghi vấn, không phải 166 session leakage đã xác nhận. Số shared_group_keys trong heuristic có scope khác thống kê unique 5-tuple, không đồng nhất hai phép đếm.
- Không overlap activity không loại trừ session có khoảng nghỉ, tuple tái sử dụng hoặc capture kéo dài. Timestamp chỉ có độ phân giải giây; guard +1 giây là giả định bảo thủ đã duyệt.
- Sáu scenario không có window trong Validation/Test: ARP_Spoofing, MQTT-DoS-Publish_Flood, MQTT-Malformed_Data, Recon-Ping_Sweep, TCP_IP-DDoS-ICMP, TCP_IP-DoS-ICMP. Độ phủ đánh giá không đồng đều.
- Recon-Ping_Sweep có 114 flow Train từ hai source 43+71; không phải 71 flow tổng. Cả hai source đều quá ngắn cho FW256/512 nên không có window. Không nối nguồn để bù.
- Benign chỉ khoảng 0,2338% Train, 0,2251% Validation, 0,2093% Test. Dự đoán Attack mọi mẫu cũng đạt Accuracy rất cao; vẫn giữ Accuracy chọn k theo source, nhưng kết quả chính thức phải kèm minority Precision/Recall/F1, Macro-F1, Balanced Accuracy và confusion matrix.
- Label là SCENARIO_PROXY, không phải ground truth từng flow. Loại metadata khỏi X không bảo đảm feature hành vi không chứa dấu vết thiết bị/capture. Scenario grouping không tương đương capture đã xác minh.

Kiểm tra độc lập trong tương lai, chỉ đề xuất: lấy PCAP/capture manifest chính thức và provenance thiết bị; đối chiếu Flow ID, bidirectional tuple, interval, protocol flags/SYN/FIN/RST và sequence nếu có; phân loại capture đã xác minh vs suy đoán; đánh giá capture/device-disjoint như thí nghiệm riêng được duyệt. Không tự gán lại nhãn hoặc thay split C+ hiện tại.

## 8. Cấu hình giữ nguyên nếu được duyệt giai đoạn sau

- MI: mutual_info_classif, random_state=42, Top100, fit Train only; không sửa các mặc định source đang dùng.
- RF-RFE: n_estimators=100, max_depth=10, min_samples_split=2, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1, step=1, n_features_to_select=1.
- RF cho mỗi k và final: n_estimators=300, max_depth=20, min_samples_split=3, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1.
- k=1..100; chọn Validation Accuracy lớn nhất, hòa chọn k nhỏ nhất.
- Freeze k và feature; refit scaler/RF trên Train+Validation theo source. Chỉ sau final refit mới đọc Test và đánh giá một lần.
- Không tự đổi tham số, số mẫu, thứ tự thống kê window hoặc split để vượt admission.

## 9. Điều kiện trước Stage5.5

1. Người dùng phê duyệt rõ nhiệm vụ tiếp theo; Stage5.4 không tự chuyển giai đoạn.
2. Nếu giữ runner: preflight mới phải có RAM khả dụng đạt guard 25.762.976.100 byte cùng đủ disk/commit; kết quả snapshot hôm nay không giữ chỗ tài nguyên. Mức total usable hiện tại không đủ guard này.
3. Nếu muốn tối ưu: phê duyệt riêng việc sửa bản triển khai và kiểm định tương đương Train/Validation, sau đó đo lại peak. Chưa có số đo thì INSUFFICIENT_EVIDENCE cho runner tối ưu, không tự giảm guard hoặc chạy full.
4. Bổ sung quan sát process/host nếu có quyền để loại rủi ro tranh chấp tài nguyên; không kết luận không có training khác chỉ từ view hiện nay.
5. Giữ nguyên full counts, parameters, Test-deferred và provenance. Không sửa dữ liệu hoặc năm script gốc. Chỉ chạy duy nhất cấu hình được duyệt, không SVM hoặc FW256/FS8.

## 10. Kết quả kiểm tra và artifact

| Kiểm tra | Kết quả |
|---|---|
| Stage5.2 / Stage5.3 chính thức và counts | PASS |
| Manifest/schema/counts FW512/FS8 | PASS_METADATA_ONLY |
| Header 75 CSV; đúng 380 + 14 cột | PASS |
| Window size/mtime 322 file so với hai preflight cũ | PASS |
| Năm source CTU-13 SHA256 và artifact run cũ | PASS |
| Disk theo mức sàn runner | PASS_MINIMUM_DISK_ONLY |
| Physical-RAM admission hiện tại | FAIL |
| Installed DIMM / đầy đủ host process / pagefile usage | UNVERIFIED_ACCESS_LIMIT |
| Actual peak FS8 / runner tối ưu | NOT_MEASURED |
| Capture/session independence | NOT_ESTABLISHED |
| Stage5.4 có training, scaler/MI/RFE/RF fit, Test evaluation | KHÔNG |
| Stage5.4 sửa code, dữ liệu, tham số hoặc hệ thống | KHÔNG |

Không chạy lại unit/benchmark có fit ở Stage5.4. 97/97 unit tests là bằng chứng từ giai đoạn trước, không trình bày thành lượt PASS mới.

Năm SHA256 gốc:

| Script trong CODE_ctu13 | SHA256 |
|---|---|
| 2_labeling.py | EA8C3468609A99C69E9F2B5AE8B86B7B7B9B55F5C00147884B0AE4D3808D2305 |
| 3_sliding_windown_and_aggregate_features.py | 9DF4BFD2BE6AD72DAA943C51265352CFDB92740B1C8429B03C098A4D0662562A |
| 7_senario1_svm.py | E44BEF8B6A6F35D4160C06DB30691510601DDD1FD31BDB12F681E8831DFA3F0F |
| [[CTU13]8_experiment2_svm.py | AADF2D829DD91C67620124AB033E78636C7A9D63C0CCAE73BE799916F88B6664 |
| [[CTU13]9_ex2_rf.py | 6DB8448FC1FD5CD49416B7CE37AC5DA6E71E6DC1554557589D73F73C936E8CB4 |

ciciomt24_training.py SHA256: 5E0FD78E8F306F65B38C5AF811FE8615C3AD6C278357DEB3C190963D608CAD5C.

Artifact tại D:\COMPACT-MI-RFE\huyen_code\runs\stage54_preflight:

- resource_check.json: timestamp đo, RAM/commit/disk/môi trường, process nhìn thấy và giới hạn đo.
- memory_estimation.json: số byte từng mảng, peak/timing run cũ, công thức guard, tham số và source thư viện.
- memory_optimization_options.md: đề xuất chưa triển khai, phân loại tương đương/sai khác và kế hoạch kiểm định.
- data_integrity_review.json: manifest counts, inventory 75 file, snapshot stat 322 file, hashes và giới hạn leakage. mtime_ns lưu dạng chuỗi để giữ chính xác nanosecond; không phải manifest hàng triệu dòng.

Lệnh PowerShell chỉ đọc báo cáo, KHÔNG chạy training:

    Get-Content -LiteralPath 'D:\COMPACT-MI-RFE\huyen_code\STAGE54_PREFLIGHT_REPORT.md' -Raw
    Get-Content -LiteralPath 'D:\COMPACT-MI-RFE\huyen_code\runs\stage54_preflight\resource_check.json' -Raw | ConvertFrom-Json | Select-Object checked_at_local, admission_status, physical_available_bytes, admission_required_available_bytes
    Get-Content -LiteralPath 'D:\COMPACT-MI-RFE\huyen_code\runs\stage54_preflight\memory_estimation.json' -Raw | ConvertFrom-Json | Select-Object config, admission_status, conservative_required_available_bytes, training_launched, optimizations_implemented

Dừng sau preflight. Chờ phê duyệt; không gọi training runner hoặc tạo kết quả Test.
