# Giai đoạn 5.1 — Chuẩn bị COMPACT-MI-RFE trên CICIoMT2024

Ngày: 19/10/2026. Trạng thái: **PASS ở mức audit, unit test và benchmark giới hạn. Chưa chạy thí nghiệm chính thức.**

## 1. Kết luận để quyết định

Đã triển khai `ciciomt24_training.py` và `test_ciciomt24_training.py` trong `D:\COMPACT-MI-RFE\huyen_code`. Bản triển khai dùng đúng 380 feature hành vi, giữ riêng hai RF-RFE, chọn k bằng Validation Accuracy và refit cuối đúng logic source. **25/25 test mới và 69/69 test hồi quy PASS.** Bốn benchmark giới hạn đều PASS; không đọc hàng dữ liệu Test, không lưu model thực nghiệm chính thức.

Chưa thể phê chuẩn việc chạy đủ ba thí nghiệm trên máy hiện tại chỉ dựa vào benchmark nhỏ. SVM RBF toàn bộ có rủi ro lớn về thời gian và RAM. Bộ kiểm tra RAM bảo thủ yêu cầu xem xét tài nguyên trước cả hai nhánh SVM; RF FS=16 có triển vọng hơn, nhưng RF FS=8 cũng thiếu headroom tại thời điểm kiểm tra. Không tự giảm dữ liệu, đổi tham số hoặc thay thuật toán.

Khuyến nghị: giữ nguyên thiết kế khoa học, quyết định tài nguyên chạy trước. Sau khi được duyệt, mới chạy tuần tự SVM Baseline → MI-RFE-SVM → MI-RFE-RF, mỗi thí nghiệm đủ bốn FW/FS. **Dừng ở Giai đoạn 5.1, chờ phê duyệt.**

## 2. Những sửa đổi cần thiết trong bản triển khai

Không sửa năm script `D:\COMPACT-MI-RFE\CODE_ctu13`, CSV flow, cutoff C+, CSV window hay công thức thống kê.

| Thành phần | Thực hiện |
|---|---|
| Input | Đọc danh sách segment và manifest C+; đối chiếu summary Stage 4.2 PASS, 322 CSV và số lượng của cả 12 ô FW/FS × split |
| X | Whitelist cố định 380 tên `feature_stat` từ generator đã kiểm định; thứ tự tên trùng thứ tự ma trận, không dựa vào việc metadata có numeric hay không |
| y | Chỉ lấy `Label` 0=Benign proxy, 1=Attack proxy; kiểm tra nhãn nhị phân |
| Metadata | Loại toàn bộ 14 cột metadata/label khỏi X, bao gồm Window_Index, Num_Flows, Scenario, Source_CSV và provenance |
| NaN/Inf | Như cả ba source classifier: `replace([+Inf,-Inf], NaN).fillna(0)`, sau đó float32; kiểm tra finite trước fit. Overflow do cast cũng bị từ chối |
| Loader | Chunking, xác minh header 394 cột, Window_ID/Index liên tục, split/source/scenario/proxy metadata và endpoint source-row nằm trong segment |
| Chạy đầy đủ trong tương lai | Có đường đọc tất cả window vào float32/int8 memmap, không sampling; mới kiểm thử bằng fixture tổng cộng hai hàng giả lập, chưa dùng trên full dataset |
| MI/RFE/scaler | Guard vai trò Train/Validation; MI và RFE không nhận Test trong pipeline lựa chọn; khóa feature/k trước lần đọc Test đầu tiên |
| Convergence | Ghi warning, SVC fit_status, n_iter và support-vector count, không che convergence warning như source. Không đổi luật chọn k để ưu tiên candidate khác |
| Kết quả | Accuracy, Precision/Recall/F1 Attack, Macro-F1, Balanced Accuracy, confusion matrix cố định thứ tự [Benign, Attack], report từng lớp và runtime |
| An toàn | Không ghi đè run cũ; benchmark có giới hạn mẫu/thời gian; full runner mặc định từ chối nếu thiếu xác nhận, kiểm tra RAM/disk và dùng trạng thái `.incomplete` |

Miền tham số classifier không được mở rộng. Không thêm XGBoost hoặc thuật toán khác. Hai script MI gốc chỉ kích hoạt FW512/FS8 trong danh sách cấu hình; bản chuyển đổi cho phép đủ bốn cấu hình theo yêu cầu, không đổi estimator/hyperparameter.

### Đính chính lỗi feature_names của báo cáo Stage 4.2

Audit và test tái hiện trực tiếp hàm chuẩn bị X của source, không import module gốc có tác dụng phụ:

| Source | Shape X trên fixture schema CICIoMT | Số feature_names trả về |
|---|---:|---:|
| `[[CTU13]8_experiment2_svm.py` | 2 × 380 | **388 — mismatch** |
| `[[CTU13]9_ex2_rf.py` | 2 × 380 | **380 — đã đúng** |
| Bản triển khai mới | 2 × 380 | **380 — đúng và whitelist** |

Script MI-RFE-SVM trả về `feature_cols` trước khi lọc numeric, ở dòng 78–82. Script RF đã trả về `list(X.columns)` ở dòng 94–97; baseline cũng lấy tên từ X sau lọc. Vì vậy nhận định trong Stage 4.2 rằng *cả hai* script MI có lỗi này là không chính xác. Báo cáo cũ được giữ nguyên để bảo toàn lịch sử; đính chính tại đây. Dù RF không có mismatch, vẫn cần whitelist để metadata numeric không thể lọt vào X khi schema thay đổi.

## 3. Đối chiếu tham số với source gốc

### Ba classifier

| Tham số | SVM Baseline — script 7 | MI-RFE-SVM — script 8 | MI-RFE-RF — script 9 |
|---|---|---|---|
| Estimator | SVC | SVC | RandomForestClassifier |
| C / kernel / gamma | 10.0 / rbf / scale | 10.0 / rbf / scale | — |
| max_iter | 20000 | 20000 | — |
| probability | **True** | **False** | — |
| cache_size / shrinking | 10000 MB / True | 10000 MB / True | — |
| class_weight | **None (mặc định)** | **balanced** | **balanced** |
| n_estimators / max_depth | — | — | 300 / 20 |
| min_samples_split / max_features | — | — | 3 / sqrt |
| random_state | 42 | 42 | 42 |
| n_jobs | SVC không có tham số này | SVC không có tham số này | -1 |

### Hai RF-RFE — không đồng nhất tham số

| Tham số | RFE của MI-RFE-SVM | RFE của MI-RFE-RF |
|---|---|---|
| n_estimators | 100 | 100 |
| max_depth | **15** | **10** |
| min_samples_split | Không truyền; mặc định 2 | Truyền 2 |
| max_features | Không truyền; mặc định sqrt trong sklearn đang dùng | Truyền sqrt |
| class_weight / random_state / n_jobs | balanced / 42 / -1 | balanced / 42 / -1 |
| RFE | n_features_to_select=1, step=1 | n_features_to_select=1, step=1 |

`mutual_info_classif(..., random_state=42)`, Top 100. Giữ các mặc định khác của thư viện: dữ liệu dense liên tục, n_neighbors=3, copy=True, không tự song song hóa MI. Một lượt RFE hoàn chỉnh từ 100 xuống 1 thực hiện 100 forest fit, gồm fit cuối. Mỗi nhánh thử đủ k=1..100 với một ranking RFE cố định, không fit lại RFE cho từng k.

Chọn k chỉ khi Accuracy mới **lớn hơn** tốt nhất trước đó. Accuracy hòa giữ k nhỏ hơn do duyệt tăng dần; không dùng F1 hoặc thời gian để phá hòa. Test unit đã xác nhận trường hợp Accuracy bằng nhau nhưng Attack F1 khác nhau.

## 4. Luồng scaler, lựa chọn và refit — ranh giới leakage

1. Làm sạch bằng phép thay hằng số 0, không học mean/median từ Val hoặc Test.
2. MI: StandardScaler fit trên **Train 380 cột**, MI fit trên Train đã scale; lấy Top 100 tên theo cách sort pandas của source.
3. Lấy lại **dữ liệu thô đã làm sạch** của Top 100; StandardScaler mới fit Train Top 100, transform Validation. Không tiếp tục dùng scaler MI 380 cột.
4. RF-RFE chỉ fit trên Train Top 100 đã scale. Ranking cố định; classifier từng k fit Train, Accuracy tính trên Validation.
5. Khóa k và danh sách feature. Không rerun MI/RFE trên Train+Validation, không dùng Test để đổi k/cấu hình.
6. Refit cuối: lấy feature đã khóa từ Train+Validation thô, fit **scaler mới trên Train+Validation**, fit classifier mới, sau đó transform và đánh giá Test một lần cho cấu hình đã khóa. Đây là refit có chủ đích đúng source, khác với fit scaler Train+Val khi còn đang lựa chọn k.

Riêng baseline chính thức: không chọn k; ghép Train+Val trước, fit scaler và SVC trên phần ghép, đánh giá Test. Benchmark kỹ thuật dùng Train nhỏ → Validation chỉ để đo tốc độ, **không phải** thay thế quy trình baseline chính thức.

Việc báo cáo số lượng/nhãn Test trong tài liệu này tái sử dụng thống kê manifest/summary của Stage 4.2, không dùng giá trị feature hoặc dự đoán Test để lựa chọn mô hình.

## 5. Unit test và benchmark giới hạn

### Test

| Nhóm | Kết quả |
|---|---|
| Test mới `test_ciciomt24_training.py` | **25/25 PASS**, lượt cuối 0,375 giây |
| Hồi quy splitter + sliding window | **69/69 PASS**, 4,170 giây; chỉ fixture nhỏ, không chạy lại full split/window |
| Tổng | **94/94 PASS** |
| py_compile hai file mới | PASS |
| SHA256 năm script source trước/sau benchmark | PASS, khớp snapshot đã duyệt |
| 322 CSV window size/mtime_ns trước/sau | PASS |

Các test kiểm tra 380 feature/names và loại numeric metadata; NaN/Inf/overflow/nhãn sai; tham số đúng source; lỗi 388/380; MI/scaler Train-only; từ chối Test ở selection/benchmark; hai RF-RFE riêng và step=1; Accuracy tie; convergence warning; feature khóa và scaler Train+Val khi refit; baseline đủ 380 cột; sampler tái lập/giữ nguồn; metadata và window ID; fixture streaming giữ đủ hàng; thiếu RAM và thiếu xác nhận full thì dừng.

Đây không phải chứng minh full model sẽ hội tụ hoặc đủ RAM. Trong Stage 5.1 không đọc lại toàn bộ 21 GB CSV để hash/kiểm định feature: tái sử dụng full validation của Stage 4.2, xác minh inventory/counts và stat snapshot, rồi kiểm tra nội dung những chunk benchmark thực sự đọc. **Stat snapshot không tương đương hash toàn bộ nội dung.**

### Thiết kế benchmark

- Run dùng để kết luận: `D:\COMPACT-MI-RFE\huyen_code\runs\stage51_pretraining_v2`.
- Mỗi FW/FS: lấy 8.192 Train và 2.048 Val theo quota tỷ lệ từng source, vị trí thưa trong prefix có giới hạn stride=8. Mẫu 4.096 là subset xác định của mẫu Train 8.192, giữ mọi source đã có khi đủ quota. Không random undersampling, không cân bằng lại lớp.
- FW256 có 32 source/17 scenario trong mẫu Train; FW512 có 30 source/16 scenario. Cả hai kích thước giữ cùng số nguồn/scenario trong mỗi cấu hình.
- 4.096 Train gồm 10 Benign + 4.086 Attack; 8.192 gồm 19 + 8.173; Val gồm 5 + 2.043 cho mỗi cấu hình. Chỉ vài Benign nên không diễn giải Accuracy/F1 benchmark thành bằng chứng chất lượng mô hình.
- Mỗi mẫu: baseline SVM đủ tham số source, MI đủ 380 → Top100, mỗi RF-RFE chạy ba bước loại bỏ (target=97, **4 forest fit**), classifier thử k=1,10,100 cho hai nhánh.
- Partial RFE còn 97 feature đồng hạng 1, nên kết quả k ở đây chỉ kiểm tra kỹ thuật/timing; `official_best_k=null`. Chưa có ranking hoàn chỉnh hay k chính thức.
- Giới hạn 240 giây/worker/config. Bốn worker tuần tự; cùng một mẫu dùng lại cho các phép benchmark, không đọc data Test.
- Đã parse **327.442 hàng CSV** trên bốn cấu hình, giữ 40.960 hàng mẫu Train+Val (mẫu 4.096 là subset, không cộng lần nữa). **Test CSV rows read=0.**

Run `stage51_pretraining` trước đó được giữ lại để không ghi đè lịch sử. Sampler subset 4.096 ban đầu không giữ đủ nguồn; không dùng run đó để kết luận/extrapolate. Bảng dưới dùng hoàn toàn run v2 đã sửa sampler và kiểm định metadata.

### Số đo thực tế — giây, không phải kết quả cuối

| FW/FS | Load Train / Val | Baseline fit 4.096 / 8.192 | MI 8.192 | RFE-SVM 4 fit | RFE-RF 4 fit | Worker tổng | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 256/8 | 4,412 / 1,452 | 1,704 / 7,087 | 8,851 | 0,897 | 0,863 | 35,750 | 285,36 |
| 256/16 | 4,379 / 1,426 | 1,875 / 7,252 | 8,977 | 1,049 | 0,987 | 36,823 | 285,42 |
| 512/8 | 4,369 / 1,520 | 1,799 / 7,288 | 9,109 | 0,923 | 0,977 | 35,788 | 287,98 |
| 512/16 | 4,139 / 1,344 | 1,759 / 9,261 | 9,037 | 0,911 | 0,924 | 38,586 | 282,99 |

RFE và bảng classifier dưới dùng mẫu 8.192. Load gồm parse/provenance validation/chuẩn bị X cho mẫu giữ lại; không phải thời gian load full. Worker tổng gồm cả hai kích thước mẫu, predict và ghi metadata. Tổng bốn worker **146,946 giây**, không bao gồm mọi overhead khởi động tiến trình cha. Peak lấy mẫu RSS mỗi 50 ms, gồm tiến trình con nếu có; không phải hard upper bound.

| FW/FS | Compact SVM fit k=1 / 10 / 100 | RF classifier fit k=1 / 10 / 100 |
|---|---:|---:|
| 256/8 | 0,031 / 0,074 / 0,204 | 0,470 / 0,477 / 0,491 |
| 256/16 | 0,036 / 0,096 / 0,327 | 0,595 / 0,575 / 0,497 |
| 512/8 | 0,018 / 0,058 / 0,192 | 0,463 / 0,479 / 0,497 |
| 512/16 | 0,028 / 0,066 / 0,219 | 0,524 / 0,594 / 0,582 |

Mọi SVC benchmark có fit_status=0; không có model-fit warning. Đã thay **3.884.324 ô NaN** trong mẫu giữ lại; Inf=0 và sau làm sạch X finite. Có Pandas FutureWarning khi `fillna(False)` ghi cờ membership trong file sample_ids; đây là cảnh báo tương thích tương lai của metadata benchmark, không phải cảnh báo fit, không sửa CSV window và không đổi X/y.

Artifacts v2 gồm audit.json, bốn worker JSON, bốn sample_ids.csv và benchmark_summary.json; tổng khoảng **4.112.313 byte**. Không lưu model. File JSON lưu thêm metric Validation kỹ thuật và thông tin convergence; không dùng như bảng thực nghiệm chính thức.

## 6. Ước tính thời gian và RAM cho dữ liệu đầy đủ

### Môi trường đo

Python 3.10.0; NumPy 1.26.4; Pandas 2.3.3; SciPy 1.15.3; scikit-learn 1.7.2. Windows 10 kernel 26200; CPU Intel64 Family 6 Model 183, 20 core vật lý/28 logical CPU. RAM tổng **23,73 GiB**, available lúc bắt đầu v2 **10,70 GiB**; một checkpoint khi các ứng dụng khác hoạt động chỉ còn khoảng 9,44 GiB available. Ổ D còn khoảng **96,95 GiB** sau benchmark. Đây là môi trường hiện có, chưa xác nhận bằng môi trường/package-lock của thí nghiệm CTU-13 gốc.

Source/doc scikit-learn đang cài xác nhận SVC/libsvm dùng đầu vào float64; chi phí fit tăng ít nhất bậc hai theo số mẫu trong mô tả chung; `probability=True` còn thực hiện calibration cross-validation 5-fold nội bộ. Đây không phải bảo đảm độ phức tạp chính xác trên bộ dữ liệu này, nhất là khi max_iter=20000 có thể chạm giới hạn trước hội tụ.

### Thời gian — các kịch bản ngoại suy, không phải cam kết

Đơn vị **giờ**. MI và RF ngoại suy tuyến tính đến NlogN từ mẫu 8.192; RFE ngoại suy từ 4 lên 100 forest fit, thêm hệ số heuristic 0,4–1,5 để phản ánh số feature giảm/độ bất định. Các hệ số không phải khoảng tin cậy thống kê. Dự báo parse/validate/clean tất cả ba tập khoảng 2,4–5,1 phút/cấu hình, **chưa gồm ghi memmap**, biến động disk/cache hoặc model.

| FW/FS | MI | RFE-SVM đầy đủ | RFE-RF đầy đủ | RF search 100 k | RF refit k=100 minh họa |
|---|---:|---:|---:|---:|---:|
| 256/8 | 0,405–0,634 | 0,410–2,410 | 0,395–2,318 | 2,192–3,433 | 0,030–0,048 |
| 256/16 | 0,205–0,306 | 0,240–1,339 | 0,226–1,260 | 1,271–1,893 | 0,015–0,023 |
| 512/8 | 0,416–0,652 | 0,422–2,478 | 0,446–2,621 | 2,192–3,433 | 0,030–0,048 |
| 512/16 | 0,206–0,307 | 0,208–1,162 | 0,211–1,178 | 1,294–1,927 | 0,018–0,027 |

RF pipeline có thể ở quy mô **vài giờ mỗi cấu hình** trong các giả định này, nhưng mẫu rất ít Benign và nhiều feature gần hằng có thể làm forest mẫu nông hơn full; chưa chứng nhận thời gian/RAM full. Search RF dùng trung bình ba k mẫu để minh họa 100 k, không khẳng định mọi k có cùng chi phí. Refit thực tế dùng k được khóa, chưa biết k đó.

Đối với SVM, bảng sau chỉ công khai phép nhân N² và N³ để nhìn thấy rủi ro khi tăng số mẫu 82–165 lần, hoặc 110–219 lần ở fit Train+Val. **Hai đầu không phải lower/upper bound runtime**, N³ chỉ là kịch bản minh họa nhạy cảm. Không được trích bảng này như dự báo full đã được xác minh.

| FW/FS | Baseline cuối Train+Val: N² / N³ | Compact SVM search 100 k: N² / N³ | Compact SVM refit k=100: N² / N³ |
|---|---:|---:|---:|
| 256/8 | 94,8 / 20.802,1 | 77,7 / 12.783,3 | 2,7 / 599,9 |
| 256/16 | 24,3 / 2.661,0 | 28,7 / 2.362,5 | 1,1 / 119,9 |
| 512/8 | 97,3 / 21.327,5 | 67,2 / 11.053,3 | 2,6 / 563,0 |
| 512/16 | 30,9 / 3.387,9 | 19,5 / 1.607,4 | 0,7 / 80,0 |

Ngay kịch bản N² đã ở quy mô ngày cho baseline FS8 và search compact SVM. Không có căn cứ hứa hoàn thành trong một đêm. Full có thể có tỷ lệ support vector khác, độ khó khác hoặc không hội tụ trong 20.000 iteration; khi đó ngoại suy từ các fit mẫu đã hội tụ không còn đáng tin. Không tăng max_iter hoặc tắt probability để che vấn đề. Thời gian của MI và RFE phải cộng thêm vào nhánh compact SVM.

### RAM — kích thước mảng và admission rule bảo thủ

Đơn vị **GiB**. Ba cột cuối là scenario kiểm tra tài nguyên trước full runner, **không phải peak RAM đã đo**. Chúng tính dự phòng cache cap/copy/support-vector buffers ở SVM và headroom ở RF; memmap cũng không làm toàn bộ thuật toán out-of-core. Cache cap 10000 MB ≈9,766 GiB là giới hạn có thể dùng, không được cấp phát cố định ngay lúc bắt đầu.

| FW/FS | Raw float32 cả 3 tập | Scenario mảng MI | Admission Baseline SVM | Admission MI-RFE-SVM | Admission MI-RFE-RF |
|---|---:|---:|---:|---:|---:|
| 256/8 | 3,181 | 8,271 | 26,170 | 16,795 | 9,271 |
| 256/16 | 1,590 | 4,135 | 18,218 | 13,530 | 5,135 |
| 512/8 | 3,177 | 8,263 | 26,154 | 16,788 | 9,263 |
| 512/16 | 1,589 | 4,132 | 18,210 | 13,527 | 5,132 |

Admission budget =85% RAM available, không phải 85% RAM lắp đặt. Budget lúc bắt đầu v2 khoảng 9,10 GiB; checkpoint sau khoảng 8,02 GiB. Cả SVM và RF FS8 bị từ chối ở mức này; RF FS16 qua được ngưỡng mảng/headroom nhưng vẫn chưa phải chứng minh đủ RAM cho mọi forest. `resource_plan` còn lưu scenario mảng baseline đơn giản hơn; full runner dùng rule bảo thủ hơn ở bảng này, có thêm dự phòng copy/support vectors. Không dùng các scenario như bộ đo peak.

Tree-buffer tuyến tính từ mẫu cho RFE chỉ khoảng 4,8–9,1 MiB/forest và classifier RF cuối khoảng 17,9–34,4 MiB, **không đáng tin như upper bound** vì cây full có thể nhiều node hơn rất nhiều. Đã lưu node count/buffer thực tế của mẫu trong JSON để truy vết, không cộng một con số forest RAM giả chính xác vào bảng.

Với RF: một nhánh RFE hoàn chỉnh tạo 10.000 cây qua 100 fit; search classifier + refit là 30.300 cây qua 101 fit, nhưng các model được thay tuần tự, không giữ tất cả cây đồng thời. Với SVM: libsvm float64 riêng đầu vào Train+Val baseline khoảng 5,09 GiB ở FS8 hoặc 2,54 GiB ở FS16, ngoài các bản float32 và bộ nhớ model/cache.

### Lựa chọn trước khi chạy chính thức

1. **Ưu tiên giữ đúng source:** dùng máy/server RAM nhiều hơn và lịch chạy CPU dài; 64 GiB RAM là mức tài nguyên nên cân nhắc để có dư địa, không phải bảo đảm hiệu năng/hội tụ. Chạy tuần tự, giữ nguyên mọi mẫu và tham số. Nếu cần một benchmark trung gian lớn hơn để giảm bất định, phải được phê duyệt riêng, chưa chạy ở đây.
2. Giữ máy hiện tại: đóng ứng dụng để tăng RAM available rồi xem xét RF, nhất là FS16, nhưng cần được duyệt thứ tự/nguồn lực trước; không tự bỏ hai thí nghiệm SVM khỏi kế hoạch nghiên cứu.
3. Giảm cache_size, tắt probability, giảm số mẫu chính thức hoặc đổi estimator sẽ khác source/thiết kế. Chỉ xem xét nếu người dùng phê duyệt một thí nghiệm sửa đổi riêng. **Chưa thực hiện bất kỳ thay đổi nào.**

## 7. Mất cân bằng nhãn và đính chính Recon-Ping_Sweep

Thống kê dưới lấy từ kết quả full window đã kiểm định, **không ngoại suy từ benchmark**. Tất cả là **SCENARIO_PROXY**, không phải ground truth từng flow.

| FW/FS | Train Benign / Attack | Validation Benign / Attack | Test Benign / Attack |
|---|---:|---:|---:|
| 256/8 | 3.182 / 1.345.135 | 1.042 / 448.301 | 1.003 / 448.248 |
| 256/16 | 1.591 / 672.578 | 521 / 224.156 | 502 / 224.129 |
| 512/8 | 3.150 / 1.343.951 | 1.010 / 447.725 | 939 / 447.624 |
| 512/16 | 1.575 / 671.984 | 505 / 223.868 | 470 / 223.817 |

Benign chỉ khoảng 0,209–0,236%. Một classifier luôn đoán Attack đã có Accuracy khoảng 99,76–99,79%, dù Recall Benign=0 và Balanced Accuracy=0,5. Vì vậy không dùng Accuracy cao đơn lẻ để tuyên bố mô hình tốt. Vẫn giữ Accuracy làm tiêu chí chọn k theo source; xuất thêm Precision, Recall, F1, Macro-F1, Balanced Accuracy, confusion matrix và runtime, đặc biệt đọc kết quả lớp Benign trong classification report.

Recon-Ping_Sweep theo `split_manifest.csv` và validation C+:

| Nguồn gốc | Source ID | C+ split | Flow |
|---|---|---|---:|
| `CSV\test\Recon-Ping_Sweep_test.pcap_Flow.csv` | sid008 | Train | 43 |
| `CSV\train\Recon-Ping_Sweep_train.pcap_Flow.csv` | sid029 | Train | 71 |
| Tổng scenario | Hai source segment riêng | Train | **114** |

Stage 3.4 ghi 114 là đúng. Stage 4.2 ghi 71 là nhầm một segment thành tổng scenario. Không thất thoát 43 flow; tên thư mục train/test *gốc* không phải membership C+ hiện tại. Mỗi segment 43 và 71 đều <256 nên **0 window cho tất cả bốn cấu hình**. Không nối chúng lại để tạo window; số lượng 6.737.006 đã kiểm định không cần sửa.

Sáu scenario không có window Validation/Test vẫn là ARP_Spoofing, MQTT-DoS-Publish_Flood, MQTT-Malformed_Data, Recon-Ping_Sweep, TCP_IP-DDoS-ICMP và TCP_IP-DoS-ICMP. Không sửa chia tập để ép đủ scenario ở giai đoạn này.

## 8. Những giới hạn khoa học còn lại

- Chia trước window và tách source segment bảo vệ việc không dùng chung **source row** qua Train/Val/Test. Window trong cùng tập vẫn chồng lấn mạnh và không phải mẫu IID độc lập; số window lớn không đồng nghĩa cùng số quan sát độc lập.
- Audit C+ trước đây: không exact duplicate record qua split và không activity crossing theo Duration+1 giây. Đây là các kết quả khác nhau, không phải chứng minh capture/session độc lập.
- Vẫn có 914.650 khóa 5-tuple dùng chung giữa split và 166 session candidate theo heuristic của audit trước; chưa xác nhận session thật. Không audit lại bằng benchmark nhỏ, không gọi các candidate là leakage đã chứng minh hoặc đã giải quyết.
- Scenario không đồng nghĩa capture độc lập. Signature của thiết bị, capture hoặc môi trường có thể nằm trong các feature hành vi dù metadata bị loại.
- Scenario proxy labels không chứng minh từng flow trong một kịch bản attack thật sự độc hại. Kết quả cuối phải nêu rõ nguồn nhãn này và giới hạn khái quát hóa.
- Hai SVM source không giống hoàn toàn: baseline class_weight=None/probability=True, compact balanced/False. Vì giữ nguyên source, so sánh là **so sánh pipeline gốc**, không thể quy mọi khác biệt Accuracy/thời gian riêng cho MI-RFE.
- Benign rất hiếm, một số scenario vắng ở held-out. Validation Accuracy có thể ưu tiên lớp Attack; giữ tiêu chí đó không có nghĩa bỏ qua minority metrics.
- Package version source gốc chưa có lock xác nhận. Tie sorting MI/RFE và giá trị thư viện có thể khác môi trường nghiên cứu gốc; bản chạy này cố định seed và ghi version, không tự đổi phương pháp sort.
- Benchmark là quota/prefix thưa xác định, không phải mẫu ngẫu nhiên đại diện toàn bộ lịch sử; adjacent sampled windows vẫn có thể liên quan nhau. Các ngoại suy tài nguyên có độ bất định lớn.

## 9. Kế hoạch chính thức sau phê duyệt và lệnh kiểm tra

Kế hoạch cố định: script 7 (SVM Baseline), rồi script 8 (MI-RFE-SVM), rồi script 9 (MI-RFE-RF). Trong mỗi thí nghiệm, lần lượt FW256/FS8, FW256/FS16, FW512/FS8, FW512/FS16; không chạy đồng thời nhiều fit lớn. Mọi Train/Val/Test window chính thức được giữ, không dùng mẫu benchmark thay thế. Khóa pipeline/config và quy tắc chọn k trước khi đánh giá Test; không xem kết quả Test để chọn lại feature hoặc estimator.

Nếu full SVM không hội tụ hoặc thiếu tài nguyên: lưu trạng thái/warning, không gọi là kết quả thực nghiệm hợp lệ đã hội tụ; dừng để xin quyết định, không tự tăng max_iter, giảm dữ liệu hay đổi cutoff. Kết quả runner có thể ghi `COMPLETED_WITH_CONVERGENCE_WARNING`; người đọc phải phân biệt với `PASS` hội tụ.

Lệnh PowerShell an toàn để kiểm tra lại code và artifacts (không full training):

```powershell
Set-Location 'D:\COMPACT-MI-RFE\huyen_code'
python -m unittest -v test_ciciomt24_training
python -m unittest -v test_ciciomt24_split test_ciciomt24_sliding_window
python -m py_compile ciciomt24_training.py test_ciciomt24_training.py
$bench = Get-Content -LiteralPath '.\runs\stage51_pretraining_v2\benchmark_summary.json' -Raw | ConvertFrom-Json
$bench | Select-Object status,technical_benchmark_only,models_saved,full_training_executed,original_sha256_before_after_match,window_size_mtime_before_after_match
python ciciomt24_training.py --help
```

Benchmark chỉ chạy lại nếu thật sự cần, dùng **thư mục run mới**; không ghi đè v2. Lệnh `train` của module là đường tương lai có cờ `--confirm-full-training`, chưa được thực thi ở Giai đoạn 5.1.

Artifact chính:

- `D:\COMPACT-MI-RFE\huyen_code\ciciomt24_training.py`
- `D:\COMPACT-MI-RFE\huyen_code\test_ciciomt24_training.py`
- `D:\COMPACT-MI-RFE\huyen_code\runs\stage51_pretraining_v2\audit.json`
- `D:\COMPACT-MI-RFE\huyen_code\runs\stage51_pretraining_v2\benchmark_summary.json`

SHA256 code benchmark/triển khai tại thời điểm báo cáo:

```text
ciciomt24_training.py
ED5DBB3B64E34794FC673D0E24AD39F78C495F215320882ACC6BA35716CF5B8B
test_ciciomt24_training.py
4CD92D7457CED2C1C3E3F3472CB9A43EAE3186C619DCFD3F56193320C55A331C
```

Năm SHA256 source gốc trước/sau nằm trong audit.json và khớp snapshot của các giai đoạn trước. Stage 5.1 không sửa chúng. Không tạo window, không chạy full MI/RFE/SVM/RF và không thay đổi dataset. **Đã hoàn tất chuẩn bị và benchmark giới hạn; cần phê duyệt tài nguyên/kế hoạch trước khi chạy chính thức.**
