<<<<<<< HEAD
# STAGE 5.5 — Tối ưu RAM MI-RFE + RF (báo cáo đang cập nhật)

Trạng thái: IN_PROGRESS — benchmark kỹ thuật đang chạy tuần tự. Đây chưa phải kết quả chính thức FW512/FS8.

## Phạm vi

Bản runner riêng: D:\COMPACT-MI-RFE\huyen_code\ciciomt24_training_memorylab.py. Runner chính thức có SHA256 5E0FD78E8F306F65B38C5AF811FE8615C3AD6C278357DEB3C190963D608CAD5C trước thử nghiệm. Năm script CTU-13 và dataset không sửa.

Chỉ dùng FW512/FS16, mẫu cố định 8.192 Train (19 Benign proxy) và 2.048 Validation (5 Benign proxy). Test CSV không đọc dữ liệu, không predict/evaluate. Cả MI Top100, RF-RFE step1 và 100 giá trị k được giữ trong benchmark. Artifacts ở D:\COMPACT-MI-RFE\huyen_code\runs\stage55_memorylab.

## Số liệu đã có

| Chế độ | Peak RSS MiB | MI giây | RFE giây | k-search giây | Tổng giây | Best k | Tương đương |
|---|---:|---:|---:|---:|---:|---:|---|
| Baseline gốc | 239,301 | 16,532 | 41,512 | 103,088 | 162,272 | 1 | Tham chiếu |
| E1 | 239,602 | đang tổng hợp | đang tổng hợp | đang tổng hợp | 148,0 | 1 | MI/rank/100 confusion khớp sơ bộ |

Các số peak chỉ từ sample. Chênh RSS baseline–E1 dưới 1 MiB không chứng minh tiết kiệm. E2, E3, E4 và combined chưa có kết quả tại thời điểm ghi phần nháp.

## Kết luận admission hiện thời

Runner nguyên gốc FW512/FS8 vẫn ADMISSION_FAIL theo guard Stage5.4: cần 23,993641 GiB available, tổng RAM usable khoảng 23,730339 GiB. Optimized runner: INSUFFICIENT_EVIDENCE cho tới khi có đo và đánh giá đại diện; benchmark nhỏ không tự cấp PASS.

Báo cáo này sẽ được cập nhật theo artifacts hoàn tất. Nếu benchmark gián đoạn, trạng thái và artifacts dở vẫn được giữ, không trình bày như kết quả chính thức.
=======
# STAGE 5.5 — Tối ưu RAM MI-RFE + Random Forest

**Trạng thái: HOÀN THÀNH BENCHMARK KỸ THUẬT; optimized runner admission = INSUFFICIENT_EVIDENCE.**

Sáu chế độ trên cùng mẫu Train/Validation FW512/FS16 đã chạy tuần tự, mỗi chế độ đủ MI Top100 → RF-RFE step=1 → k=1..100 → refit Train+Validation. Năm chế độ E1, E2, E3, E4 và kết hợp đều tương đương baseline trên mẫu theo các kiểm định bên dưới. Không có huấn luyện chính thức FW512/FS8 hoặc Test evaluation mới.

## 1. Source, runner và ranh giới chạy

Runner chính thức D:\COMPACT-MI-RFE\huyen_code\ciciomt24_training.py giữ SHA256:

    5E0FD78E8F306F65B38C5AF811FE8615C3AD6C278357DEB3C190963D608CAD5C

Bản thử nghiệm D:\COMPACT-MI-RFE\huyen_code\ciciomt24_training_memorylab.py có SHA256:

    DFEEA643529D093AC845D67095739381463A59FDAD8774A4D51E52CD1243AA89

Bản memorylab được tách từ runner hiện hành. Diff thống nhất giữa hai file được lưu trong optimization_diff.patch và đã xác minh khớp chính xác nội dung hiện tại. CLI memorylab chỉ có benchmark và worker benchmark; lời gọi run_official trong module này bị chặn. Baseline gọi trực tiếp các hàm chọn đặc trưng, fit và refit của module chính thức; các chế độ E1–E4 dùng các nhánh riêng bật/tắt theo flag. Mọi chế độ vẫn dùng cùng tham số RF-RFE 100 cây, RF 300 cây, random_state=42, n_jobs=-1, MI Top100, RFE step=1 và toàn bộ k-grid 1..100.

Thư mục artifact kỹ thuật: D:\COMPACT-MI-RFE\huyen_code\runs\stage55_memorylab. status.json ghi COMPLETE_TECHNICAL_ONLY; tất cả file kết quả đều ghi TECHNICAL_BENCHMARK_NOT_OFFICIAL. Tổng thời gian từ trạng thái khởi tạo đến hoàn thành khoảng 15 phút, gồm chuẩn bị mẫu và sáu worker. Không ghi đè run Stage5.2/5.3.

## 2. Mẫu cố định và phạm vi dữ liệu

Dùng duy nhất FW_512_FS_16. Mẫu lấy một lần qua WindowStore.load_sample với quota theo source và stride cố định, lưu thành hai cache độc lập:

| Tập | Window giữ | Benign proxy | Attack proxy | Dòng CSV đã parse lúc chuẩn bị mẫu | Thời gian chuẩn bị |
|---|---:|---:|---:|---:|---:|
| Train | 8.192 | 19 | 8.173 | 65.501 | 9,039 giây |
| Validation | 2.048 | 5 | 2.043 | 16.370 | 3,063 giây |
| Test | 0 | 0 | 0 | 0 | 0 |

Mỗi worker dùng lại đúng hai cache với hash X/y và thứ tự Window_ID đã lưu trong sample_manifest.json. Đây là sampling cho benchmark kỹ thuật, không phải giảm mẫu cho thí nghiệm chính thức. Best k trên mẫu là 1 và feature là Pkt Len Min_mean; kết quả chính thức Stage5.2 full FW512/FS16 có best k=8. Mẫu rất nhỏ và chỉ có 19/5 Benign, nên khác biệt này không dùng để sửa pipeline.

Không đọc hàng dữ liệu, feature hoặc nhãn Test trong benchmark. WindowStore khởi tạo inventory và kiểm tra stat/manifest cả bốn cấu hình, bao gồm metadata đường dẫn Test; đó không phải nạp Test CSV. Không mở Test CSV để fit, chọn k hay đánh giá. Loader sample chỉ nhận role train/val; worker chỉ nạp train_sample.npz và val_sample.npz.

## 3. Vòng đời RAM và điểm tạo copy

Phân tích chi tiết shape, dtype, memmap, copy/view, tham chiếu và source thư viện ở [mi_memory_analysis.md](D:/COMPACT-MI-RFE/huyen_code/runs/stage55_memorylab/mi_memory_analysis.md).

Điểm chính:

1. Full loader của runner chính thức tạo memmap X/y float32/int8 theo split, nhưng chuẩn bị CSV chunk còn giữ DataFrame, float64 và float32 tạm. Memmap không ngăn các estimator materialize dữ liệu.
2. StandardScaler đầu tiên tạo scaled Train float32 full-size khi copy=True; thống kê mean/variance có mask và temporary. Peak scaler riêng chưa có số đo trên full runs.
3. MI của sklearn 1.7.2 tạo X float64, copy do advanced indexing của continuous mask, temporary abs/scale và full-shape RNG jitter. Một X64 FW512/FS8 có payload 3,813940 GiB; bộ nhớ neighbor/allocator còn chưa có cận full. MI là peak đã quan sát ở hai run FS16 trước.
4. Top100 advanced indexing tạo bản riêng; scaler mặc định thêm output. RFE liên tục dùng X[:, features] copy. K-search baseline dùng X[:, indices] copy, đồng thời model k trước còn tham chiếu khi model k sau bắt đầu fit.
5. Final refit tạo raw Train+Validation Nxk rồi bản scaled Nxk; k đã khóa bằng Validation. Memorylab chỉ refit mẫu này và không dùng Test.

## 4. Những tối ưu đã triển khai

| Flag | Tác động thực hiện | Kiểm tra an toàn trong memorylab |
|---|---|---|
| E1 | Xóa model k cũ sau khi ghi Validation metrics, trước vòng k tiếp theo | Giữ nguyên thứ tự fit, estimator, seed, k-grid; không xóa model refit |
| E2 | Nạp cache Validation sau khi MI hoàn tất | MI chỉ có Train; thứ tự/nhãn Validation từ cùng cache; so MI scores và RNG |
| E3 | StandardScaler(copy=False) trên bản Top100 riêng | Bản Train/Val writable, không share với raw X, float32; scaler thật sự tái sử dụng bản đó; raw SHA không đổi |
| E4 | Sắp Top100 một lần theo ranking RFE, dùng workspace[:, :k] | So dtype, thứ tự và giá trị với X[:, ranking[:k]] cho tất cả 100 k; validate_data của sklearn giữ view ở k=1,50,100 trên sample |
| E1+E2+E3+E4 | Bật cả bốn trong cùng worker | Cùng kiểm định từng nhánh và đối chiếu baseline |

E4 chỉ chứng minh tầng validate_data của forest giữ view float32 ở ba k được dò; tree builder bên trong có thể cấp bản sao khác. Workspace ban đầu cần một lần copy, và lợi ích peak RSS không được suy ra chỉ từ np.shares_memory.

## 5. Benchmark RSS và thời gian

Các worker chạy nối tiếp, 50 ms một mẫu RSS của process và child theo cùng cách đo. MiB = byte / 2^20. Peak bảo thủ là số lớn hơn giữa tracker toàn worker và từng phase, vì hai sampler có thể bắt được thời điểm khác nhau.

| Chế độ | Peak bảo thủ MiB | MI peak | Top100 scale peak | RFE peak | k-search peak | Refit peak | Tổng giây |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 239,301 | 239,301 | 161,703 | 178,031 | 180,734 | 175,680 | 162,272 |
| E1 | 239,602 | 239,602 | 161,887 | 178,527 | 181,859 | 175,711 | 148,044 |
| E2 | 238,988 | 236,047 | 161,586 | 177,957 | 181,172 | 175,078 | 143,950 |
| E3 | 255,590 | 239,594 | 161,922 | 178,418 | 181,879 | 176,961 | 135,576 |
| E4 | 263,137 | 263,137 | 161,641 | 179,688 | 178,988 | 179,039 | 122,175 |
| Kết hợp | 236,793 | 236,793 | 161,930 | 178,473 | 180,027 | 180,000 | 159,614 |

Thời gian theo phase (giây):

| Chế độ | Load cache Train+Val | MI | RF-RFE | k-search đủ 100 k | Refit |
|---|---:|---:|---:|---:|---:|
| Baseline | 0,041 | 16,532 | 41,512 | 103,088 | 0,969 |
| E1 | 0,031 | 16,607 | 41,321 | 89,088 | 0,895 |
| E2 | 0,032 | 15,744 | 41,359 | 85,836 | 0,902 |
| E3 | 0,029 | 14,081 | 38,205 | 82,314 | 0,871 |
| E4 | 0,035 | 15,978 | 32,842 | 72,380 | 0,854 |
| Kết hợp | 0,034 | 15,690 | 41,606 | 101,217 | 0,990 |

E2 giảm MI phase peak khoảng 3,254 MiB khi hoãn cache Validation, nhưng peak toàn worker chỉ giảm khoảng 0,313 MiB. Kết hợp thấp hơn baseline khoảng 2,508 MiB (1,05%) trong một lượt đo. E1 không cho thấy giảm peak; E3/E4 có peak bảo thủ cao hơn trong một lượt. Peak MI của E4 cao hơn baseline dù E4 chỉ bắt đầu sau MI: đây là dấu hiệu điều kiện chạy, allocator hoặc sampling thay đổi, không thể quy nguyên nhân đó cho workspace E4. Ở E3, peak toàn worker vượt mọi phase fit tách riêng; thao tác sau fit như hashing raw sample cũng có thể tạo mảng ngắn hạn.

Tổng thời gian E4 nhanh hơn baseline trong lượt này, nhưng cả MI/RFE trước khi E4 có thể tác động cũng thay đổi thời gian. E3 giữ nguyên k-search mà k-search vẫn nhanh hơn baseline hơn 20 giây. Do đó đây chưa phải chứng cứ định lượng ổn định về speedup; chưa có lượt lặp để tách cache và nhiễu đo. Không lấy 2,508 MiB của sample để ngoại suy tuyến tính sang 1,347 triệu Train FS8.

### Commit, RAM available, page faults và disk

| Chế độ | Peak private commit process, MiB | RAM available trước → sau worker, GiB | Process page faults tăng |
|---|---:|---:|---:|
| Baseline | 1.680,414 | 5,815 → 4,821 | 1.140.500 |
| E1 | 1.680,465 | 4,787 → 4,175 | 1.139.422 |
| E2 | 1.680,098 | 4,150 → 4,663 | 1.084.471 |
| E3 | 1.704,562 | 4,702 → 4,824 | 1.093.665 |
| E4 | 1.703,977 | 5,358 → 5,844 | 1.141.328 |
| Kết hợp | 1.677,641 | 5,789 → 5,026 | 1.066.531 |

Private bytes là proxy commit riêng của process, không phải system commit peak. Bộ đếm num_page_faults gồm cả fault mềm; không suy ra số lượt đọc pagefile. RAM available chịu ảnh hưởng các ứng dụng ngoài benchmark, nên chênh trước/sau không quy trực tiếp cho tối ưu. Không đo được hard page faults hoặc commit toàn hệ thống theo phase.

Hai cache mẫu chiếm 15.576.020 byte, khoảng 14,85 MiB; tổng thư mục lúc tổng hợp khoảng 15,52 MiB. Không tạo full X/y memmap hoặc CSV window mới. optimization_benchmarks.json lưu số đo chi tiết từng phase và từng worker.

## 6. Kiểm định tương đương

equivalence_report.json ghi PASS cho cả năm chế độ so baseline. Cụ thể:

- Raw X/y trước và sau từng worker có cùng SHA256; cùng 8.192/2.048 dòng và cùng thứ tự feature/cột, Window_ID mẫu cố định.
- Scaler MI, scaler Top100 và scaler refit có mean/var/scale khớp bitwise sau JSON float roundtrip. Scaled Top100 Train/Val cũng có cùng SHA256.
- Cả 380 MI scores khớp bitwise; sai khác tuyệt đối và tương đối lớn nhất đều 0. Top100 có đúng cùng thứ tự. Có tám nhóm MI score bằng nhau, tổng 53 feature trong các nhóm đó; thứ tự vẫn khớp. Feature ở ranh Top100 có score không đồng hạng tại chính ranh giới (tie count=1).
- RFE gồm 100 forest fits, 100 rank và thứ tự ranking hoàn toàn giống. Prefix feature giống nhau ở mọi k.
- Mỗi chế độ thử đủ k=1..100. Accuracy và confusion matrix ở từng k giống baseline; số k khác biệt bằng 0. Best k=1 trong cả sáu chế độ, cùng frozen feature và luật hòa k nhỏ nhất.
- RNG toàn cục không bị MI tiêu thụ trong từng worker; MI dùng random_state=42 cố định. E3 cô lập bản copy; E4 kiểm tra 100 workspace prefix.
- Refit scaler trên cùng Train+Validation sample, không đọc/evaluate Test. Không có warning RF fit trong 100 k ở sáu lượt.

Kiểm định này áp dụng cho mẫu kỹ thuật trên môi trường đã ghi, không chứng minh bitwise cho toàn bộ FW512/FS8, không chứng minh mọi tree allocation giống nhau và không thay thế đánh giá Test chính thức. Không dùng tolerance rộng để bỏ qua sai khác: các phần số học nêu trên khớp bitwise; confusion/Accuracy khớp chính xác.

## 7. MI bottleneck và hướng E/N1, N2

MI vẫn là peak lớn nhất của baseline và chế độ kết hợp trên mẫu, cũng là peak bảo thủ đã đo ở Stage5.2/5.3 full FS16 khoảng 10 GiB. E1, E3, E4 không can thiệp các temporary float64 của MI. E2 chỉ giảm thời gian Validation cùng tồn tại trước MI.

E/N1 đề xuất scratch memmap riêng cho scaled float32 và MI float64, giữ raw bất biến và thứ tự arithmetic/cast; payload scratch lý thuyết khoảng 5,721 GiB disk ở FS8. File-backed không đồng nghĩa giảm RSS, có thể tăng I/O/page faults. N2 xem xét xử lý temporary MI theo block; muốn tương đương phải bảo toàn dòng RNG của full N x 380, thứ tự cột, thống kê floating-point và neighbor estimation. Gọi MI từng cột với cùng seed 42 không bảo toàn chuỗi nhiễu nguồn. Cả E/N1 và N2 chưa triển khai, chưa đo, chưa đủ điều kiện áp dụng. Thiết kế chi tiết và phương pháp so MI scores nằm trong mi_memory_analysis.md.

## 8. Khả năng chạy FW512/FS8 và quyết định admission

Runner nguyên gốc vẫn dùng guard Stage5.4: cần 25.762.976.100 byte, tức 23,993641 GiB available. Stage5.4 đo tổng RAM Windows dùng được khoảng 23,730339 GiB; guard đó không đổi. Không dùng kết quả mẫu để hạ headroom hoặc đổi tham số.

**OPTIMIZED_RUNNER_ADMISSION=INSUFFICIENT_EVIDENCE.** E1–E4 chưa chứng minh đủ giảm peak MI full. Kết quả sample kết hợp tiết kiệm 2,508 MiB theo peak bảo thủ trong một lượt; khi E4 chạy riêng peak lại cao hơn baseline và RAM available toàn host biến động rõ. Các tối ưu ở phase sau MI không giải quyết được phần float64/temporary quyết định peak. Không có cơ sở kết luận full FW512/FS8 chạy an toàn trên máy hiện tại, hoặc đặt guard mới.

Muốn xét admission tiếp phải có benchmark quy mô đại diện cho MI trên Train/Validation với monitoring RSS/commit/page faults, lặp lại để tách biến động cache/host, đối chiếu tương đương chi tiết; mọi thay đổi vào MI E/N1/N2 cần phê duyệt riêng. Không đọc Test để chọn tối ưu. Sau đó mới xây dựng policy headroom minh bạch. Không tự chuyển memorylab thành runner chính thức.

## 9. Bảo toàn dữ liệu và giới hạn khoa học

Sau benchmark đã kiểm tra:

| Kiểm tra | Kết quả |
|---|---|
| SHA256 runner chính thức | PASS, giữ nguyên |
| SHA256 năm script CTU-13 | PASS |
| 322 CSV window, size và mtime_ns so với manifest đầu thử nghiệm | PASS |
| SHA256 24 artifact metadata/model/scaler của Stage5.2/5.3 | PASS |
| SHA256 hai cache sample | PASS |
| Sáu chế độ, mỗi chế độ đủ 100 k | PASS |
| Raw sample X/y không đổi | PASS |
| Test data rows/predictions trong memorylab | 0 / 0 |
| Full FW512/FS8 training | KHÔNG |
| Capture/session độc lập | CHƯA ĐƯỢC CHỨNG MINH |

Size/mtime của window là kiểm tra metadata, không phải hash mới của toàn bộ hàng triệu dòng. Nhãn tiếp tục là SCENARIO_PROXY, không phải ground truth từng flow. C+ ngăn chung source row/window qua split theo audit đã duyệt; vẫn còn shared 5-tuple và session-continuity candidates, và scenario/capture có thể mang dấu vết thiết bị. Benchmark này không thay split hay nhãn.

## 10. Artifact và lệnh xem kết quả

- [baseline_profile.json](D:/COMPACT-MI-RFE/huyen_code/runs/stage55_memorylab/baseline_profile.json): pipeline/mẫu/MI/RFE/100 k/refit và resource baseline.
- [optimization_benchmarks.json](D:/COMPACT-MI-RFE/huyen_code/runs/stage55_memorylab/optimization_benchmarks.json): bảng số đo sáu chế độ.
- [equivalence_report.json](D:/COMPACT-MI-RFE/huyen_code/runs/stage55_memorylab/equivalence_report.json): kiểm định số học và Validation.
- [mi_memory_analysis.md](D:/COMPACT-MI-RFE/huyen_code/runs/stage55_memorylab/mi_memory_analysis.md): vòng đời mảng và thiết kế MI.
- [optimization_diff.patch](D:/COMPACT-MI-RFE/huyen_code/runs/stage55_memorylab/optimization_diff.patch): toàn bộ khác biệt so runner chính thức.
- [resource_admission_recommendation.json](D:/COMPACT-MI-RFE/huyen_code/runs/stage55_memorylab/resource_admission_recommendation.json): giữ guard gốc và optimized admission.
- sample_manifest.json, train_sample.npz, val_sample.npz, E1/E2/E3/E4/E1_E2_E3_E4.json và status.json: cache kỹ thuật và kết quả từng worker.

Lệnh PowerShell chỉ đọc:

    $lab = 'D:\COMPACT-MI-RFE\huyen_code\runs\stage55_memorylab'
    Get-Content -LiteralPath (Join-Path $lab 'status.json') -Raw
    Get-Content -LiteralPath (Join-Path $lab 'optimization_benchmarks.json') -Raw | ConvertFrom-Json | Select-Object status,all_100_k_each_mode
    Get-Content -LiteralPath (Join-Path $lab 'equivalence_report.json') -Raw | ConvertFrom-Json | Select-Object status,sample_rows
    Get-Content -LiteralPath (Join-Path $lab 'resource_admission_recommendation.json') -Raw

Stage5.5 dừng ở benchmark và báo cáo. Bước tiếp theo là quyết định riêng của người dùng về nghiên cứu MI hoặc tài nguyên; không có full training, FW256/FS8, SVM hay Test refit/evaluation mới.
>>>>>>> e07ffe7 (Update COMPACT-MI-RFE experiments)
