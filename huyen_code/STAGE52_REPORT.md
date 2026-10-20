# Giai đoạn 5.2 — Kết quả chính thức MI-RFE + Random Forest FW512/FS16

**Trạng thái: PASS — đã xác minh artifacts ngày 20/10/2026.**

Run chính thức: D:\COMPACT-MI-RFE\huyen_code\runs\stage52_rf_FW512_FS16_vscode_20261019_222901

Run do người dùng chạy trong VS Code, từ khoảng 22:29:03 đến 23:11:12 ngày 19/10/2026 (Asia/Saigon). Tổng elapsed trong result.json là **2529,1477476 giây, khoảng 42 phút 9 giây**.

Đã thực hiện 130/130 assertion kiểm tra artifacts, trong đó 100 assertion đối chiếu confusion matrix/Accuracy từng k trên Validation. Đây là kiểm định hậu nghiệm artifacts, không phải 130 unit test và không huấn luyện lại. 97/97 unit test trước chạy được tái sử dụng làm bằng chứng kiểm thử code. Không predict/evaluate Test thêm lần nào trong quá trình xác minh.

## 1. Số mẫu thực tế và nguồn nhãn

| Split | Window thực tế | Benign proxy | Attack proxy |
|---|---:|---:|---:|
| Train | 673.559 | 1.575 | 671.984 |
| Validation | 224.373 | 505 | 223.868 |
| Test | 224.287 | 470 | 223.817 |
| Train + Validation dùng refit | 897.932 | 2.080 | 895.852 |

Input: D:\COMPACT-MI-RFE\CICIoMT24\WINDOWS_CPLUS\FW_512_FS_16

Sáu ma trận X/y đã được kiểm tra shape, dtype và số nhãn: X có đúng 380 cột float32; y là int8 nhị phân. Counts khớp full manifest và result.json; không dùng mẫu benchmark, sampling hoặc undersampling. Log full loader xác nhận X finite sau thay Inf thành NaN rồi fillna(0). Metadata, Scenario, Split, Window_ID, Source_CSV và provenance không thuộc X.

Nhãn là **SCENARIO_PROXY**: 0=Benign, 1=Attack theo scenario, không phải ground truth từng flow.

## 2. Pipeline và tham số đã xác minh

- StandardScaler đầu tiên và mutual_info_classif chỉ fit Train; random_state=42, Top 100 trong 380 feature.
- Sau MI, lấy lại raw-cleaned Top100 và fit scaler mới trên Train; transform Validation.
- RF-RFE: n_estimators=100, max_depth=10, min_samples_split=2, max_features=sqrt, class_weight=balanced, random_state=42, n_jobs=-1. RFE step=1, target=1; ranking đầy đủ 100 xuống 1, tương ứng 100 forest fit.
- RF classifier: n_estimators=300, max_depth=20, min_samples_split=3, max_features=sqrt, class_weight=balanced, random_state=42, n_jobs=-1.
- Ranking cố định, đủ k=1..100; chọn Accuracy Validation lớn nhất, hòa giữ k nhỏ.
- Khóa feature/k; không fit MI/RFE lại trên Train+Validation.
- Scaler mới và classifier mới được fit Train+Validation; chỉ sau khi refit hoàn tất mới mở CSV Test và evaluate một lần.

Tham số model joblib thực tế khớp source. Đã kiểm tra 300 cây, classes=[0,1], n_features_in_=8. Scaler có n_features_in_=8 và n_samples_seen_=897932; mean/variance được đối chiếu bằng cách cộng thống kê cột đã khóa từ ma trận Train+Validation, không gọi fit và không dùng Test.

SHA256 code triển khai khớp preflight của run, nên log được đối chiếu với đúng bản code đã chạy:

    7631093FDC76F64CC74F5C7C1E595E11237536E08F8DBA5FD7DC05494B60BCC6

Không sửa code triển khai hoặc năm script CTU-13 trong lần xác minh này.

## 3. Lựa chọn k và tám feature

**Best k=8, Best Validation Accuracy=1.0.** Có nhiều k cùng đạt 1.0; k=8 là nhỏ nhất. Đã xác minh cả danh sách k_search_results.json và 100 event k_completed liên tục từ 1 tới 100. Mỗi confusion matrix Validation có 505 Benign và 223868 Attack; Accuracy khớp tổng đường chéo chia tổng mẫu.

| RFE rank | Feature được khóa |
|---:|---|
| 1 | Fwd Seg Size Avg_median |
| 2 | Fwd Pkt Len Mean_median |
| 3 | Fwd Pkt Len Max_median |
| 4 | Pkt Len Min_mean |
| 5 | Pkt Len Min_std |
| 6 | Bwd PSH Flags_kurtosis |
| 7 | Bwd PSH Flags_mean |
| 8 | Init Bwd Win Byts_std |

Danh sách trên trùng giữa selection_frozen.json, features.json, result.json và prefix tám tên của rfe_feature_ranking.csv. MI có đúng 380 tên canonical, scores hữu hạn/không âm/sort giảm; mi_top100.json là đúng 100 tên đầu. RFE chứa đúng các tên Top100 và ranking duy nhất 1..100.

Giảm 380 xuống 8 feature: loại 372 cột, tương đương **97,8947368421%**. Từ Top100 xuống 8 giảm 92%. Đây không đồng nghĩa giảm cùng tỷ lệ thời gian toàn pipeline: MI vẫn tính trên 380 feature và RFE/search vẫn phải chạy đầy đủ.

## 4. Thứ tự thời gian kiểm soát Test

Giờ địa phương Asia/Saigon ngày 19/10/2026:

| Sự kiện trong events.jsonl | Thời điểm |
|---|---|
| MI bắt đầu trên Train | 22:29:59,796 |
| MI hoàn tất | 22:51:53,015 |
| RF-RFE bắt đầu | 22:51:55,022 |
| RF-RFE hoàn tất | 22:56:47,393 |
| Khóa k=8 và feature | 23:10:48,104 |
| Refit Train+Validation bắt đầu | 23:10:48,114 |
| Refit hoàn tất | 23:10:57,382 |
| Lần đầu load Test | 23:10:57,397 |
| Test evaluation bắt đầu | 23:11:11,859 |
| Test evaluation hoàn tất | 23:11:12,382 |
| Run validated | 23:11:12,547 |

Trật tự khóa feature → refit hoàn tất → đọc Test đã PASS. Chỉ có một event bắt đầu/kết thúc Test evaluation; result.json ghi test_evaluation_calls=1 và test_rows_evaluated=224287. Không có bước MI/RFE lại sau khi ghép Train+Validation.

Bằng chứng dựa trên code có SHA256 khớp, nhật ký và artifacts của runner. Nó không phải chứng minh độc lập capture/session, cũng không phải cơ chế kiểm chứng chống sửa log từ bên ngoài.

## 5. Kết quả Test chính thức

| Chỉ số | Giá trị [0,1] | Phần trăm |
|---|---:|---:|
| Accuracy | 0,9999866242804978 | 99,9986624280% |
| Precision Attack | 0,9999865963720848 | 99,9986596372% |
| Recall Attack | 1,0 | 100% |
| F1 Attack | 0,9999932981411277 | 99,9993298141% |
| Macro-F1 | 0,9983957952818765 | 99,8395795282% |
| Balanced Accuracy | 0,9968085106382979 | 99,6808510638% |
| Precision Benign | 1,0 | 100% |
| Recall Benign | 0,9936170212765958 | 99,3617021277% |
| F1 Benign | 0,9967982924226254 | 99,6798292423% |

Confusion matrix có hàng là nhãn thật proxy, cột là dự đoán:

| True / Predicted | Benign | Attack |
|---|---:|---:|
| Benign proxy | 467 | 3 |
| Attack proxy | 0 | 223.817 |

Tổng 224.287 mẫu Test, sai ba Benign thành Attack và không bỏ sót Attack theo nhãn proxy. Không diễn giải kết quả này thành khả năng phát hiện attack thực địa có ground truth từng flow.

## 6. Thời gian và RAM thực tế

| Công đoạn | Thời gian đo |
|---|---:|
| Load + kiểm định Train | 41,844 giây theo event |
| Load + kiểm định Validation | 14,270 giây theo event |
| MI | 1312,973972 giây (21 phút 52,974 giây) |
| RF-RFE đầy đủ | 292,333995 giây (4 phút 52,334 giây) |
| Tổng fit classifier của 100 k | 788,279670 giây |
| Tổng predict + metrics Validation của 100 k | 44,797099 giây |
| Fit + predict/metrics của k-search | 833,076769 giây |
| K-search wall, từ RFE xong tới khóa feature | Khoảng 840,701 giây, gồm logging/điều phối |
| Refit RF cuối, riêng fit | 8,646837 giây |
| Giai đoạn refit gồm chuẩn bị/scaling | Khoảng 9,278 giây theo event |
| Load + kiểm định Test | 14,443 giây theo event |
| Predict + metrics Test | 0,430662 giây |
| Tổng run theo result.json | **2529,1477476 giây** |

Peak RSS toàn-run được bộ lấy mẫu riêng ghi là 10.682.068.992 byte, khoảng 9,948452 GiB. Bộ lấy mẫu pha MI ghi peak cao hơn: **10.730.827.776 byte, khoảng 9,993862 GiB**. Hai sampler 50 ms có thời điểm quan sát khác nhau; không ép hai con số bằng nhau. Khi kiểm tra tài nguyên lần sau dùng giá trị lớn hơn một cách bảo thủ, khoảng 10 GiB. RSS này không phải mức commit tổng, không bao gồm mọi bộ nhớ của các ứng dụng khác, và không phải hard upper bound.

Peak RFE khoảng 2,948 GiB; peak refit RF khoảng 2,522 GiB. Không có warning được ghi trong fit RF-RFE, 100 fit classifier hoặc refit cuối. RF không có tiêu chí hội tụ kiểu max_iter/fit_status của SVM; không được gọi đó là kiểm chứng hội tụ số học như SVM. Bộ đo MI không thu một danh sách warning riêng, nên không tuyên bố đã chứng minh không có mọi warning MI.

Tổng kích thước artifacts, gồm sáu memmap ma trận: **1.708.574.240 byte**. Model 1.229.769 byte; final_scaler 807 byte.

## 7. Kiểm định artifacts và tính nguyên vẹn

| Nhóm kiểm tra | Kết quả |
|---|---|
| 12 artifacts metadata/model/scaler bắt buộc và sáu file ma trận có mặt/đọc được | PASS |
| Đúng experiment, config, số mẫu; không sampling | PASS |
| MI 380 tên và Top100 chính xác | PASS |
| Ranking 100 feature duy nhất 1..100 | PASS |
| Đủ 100 k, Accuracy và luật phá hòa | PASS |
| Danh sách tám feature khớp mọi artifact | PASS |
| Model params, số cây, số feature và classes | PASS |
| Scaler số mẫu, số feature, mean/variance Train+Validation | PASS |
| Trật tự khóa/refit/đọc Test, Test evaluation đúng một lần | PASS |
| Metrics, confusion matrix và elapsed khớp giá trị yêu cầu | PASS |
| Số nhãn trong cả ba y memmap | PASS |
| SHA256 năm script CTU-13 khớp preflight và snapshot được duyệt | PASS |
| SHA256 code triển khai khớp preflight run | PASS |
| Snapshot size/mtime_ns của 322 CSV window khớp preflight | PASS |
| Capture/session độc lập | CHƯA ĐƯỢC CHỨNG MINH |

Không đọc lại 71 CSV flow gốc hoặc toàn bộ CSV window để tính hash. Snapshot size/mtime không tương đương SHA256 toàn nội dung; nguyên tắc không ghi dataset được bảo toàn vì thao tác xác minh chỉ mở dataset/memmap để đọc. Kiểm định full window trước đây của Stage 4.2 được tái sử dụng, không giả định đã thực hiện lại ở đây.

Các file bắt buộc đã kiểm tra: preflight_audit.json, progress.json, events.jsonl, mi_feature_scores.csv, mi_top100.json, rfe_feature_ranking.csv, k_search_results.json, selection_frozen.json, features.json, final_scaler.joblib, model.joblib và result.json; cùng train/val/test X/y .npy trong matrices.

Các SHA256 artifact được lưu riêng tại D:\COMPACT-MI-RFE\huyen_code\runs\stage53_preflight\stage52_artifact_validation.json. Không thay đổi result.json, model/scaler, features, logs hoặc ma trận của run đã hoàn tất. Run vốn đã là PASS; báo cáo này xác nhận và cập nhật lại trạng thái báo cáo, không viết lại lịch sử run.

## 8. Lịch sử lượt bị dừng

Lượt D:\COMPACT-MI-RFE\huyen_code\runs\stage52_mi_rfe_rf_FW512_FS16.incomplete trước đó bị dừng theo yêu cầu người dùng khi còn ở MI, khoảng 900 giây. Nó không có kết quả chính thức và không được dùng thay thế run VS Code. Giữ nguyên toàn bộ artifacts dở, không xóa, không resume.

Heartbeat theo dõi cũ đã PAUSED; không bật lại. Bản nháp trước của báo cáo này mô tả lượt bị dừng. Báo cáo đã được hoàn thiện theo yêu cầu Giai đoạn 5.3 bằng kết quả run VS Code, không phải tự chạy lại FW512/FS16.

## 9. Giới hạn và cách trình bày với giảng viên

- Đã chia flow trước khi tạo window và không dùng chung source row qua split; trong cùng tập các window vẫn chồng lấn mạnh, không phải các quan sát IID độc lập.
- Nhãn SCENARIO_PROXY và độ phủ scenario hạn chế phải được công bố. Test chỉ có 470 Benign so với 223817 Attack, nên Accuracy cao đơn lẻ không đủ; cần báo Recall Benign, Macro-F1 và Balanced Accuracy.
- Các kết quả không có exact duplicate/activity crossing của audit trước khác với capture/session independence. Metadata vẫn chưa chứng minh capture/session độc lập; không khẳng định hoàn toàn leakage-free.
- Lựa chọn qua 100 k trên một Validation có thể tối ưu hóa riêng tập này. Giữ đúng source nhưng không biến Validation Accuracy=1.0 thành bằng chứng tổng quát hóa ngoài dataset.
- Không dùng Test để chọn k hoặc sửa classifier. Thí nghiệm tiếp theo FW256/FS16 là cấu hình người dùng chỉ định, không phải tối ưu tham số theo Test của run này.
- 380→8 giảm số cột của classifier cuối; thời gian toàn pipeline vẫn bao gồm MI380, RFE100 và 100 classifier fit.

Có thể trình bày: “Tôi tái hiện pipeline MI Top100 → RF-RFE step1 → chọn k bằng Validation Accuracy → refit trên Train+Validation. Run FW512/FS16 chọn tám feature, Test sai ba mẫu Benign theo scenario proxy. Đã kiểm soát ranh giới flow/window và vai trò Train/Val/Test, nhưng chưa chứng minh độc lập capture/session.”

## 10. Lệnh PowerShell kiểm tra, không huấn luyện/predict lại

    $stage52Run = 'D:\COMPACT-MI-RFE\huyen_code\runs\stage52_rf_FW512_FS16_vscode_20261019_222901'
    Get-Content -LiteralPath (Join-Path $stage52Run 'result.json') -Raw
    Get-Content -LiteralPath (Join-Path $stage52Run 'selection_frozen.json') -Raw
    $stage52K = Get-Content -LiteralPath (Join-Path $stage52Run 'k_search_results.json') -Raw | ConvertFrom-Json
    $stage52K.Count
    $stage52K | Sort-Object @{Expression='accuracy';Descending=$true}, @{Expression='k';Ascending=$true} | Select-Object -First 1 k,accuracy
    Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $stage52Run 'model.joblib'), (Join-Path $stage52Run 'final_scaler.joblib')

Kết luận: **Giai đoạn 5.2 đã hoàn tất và artifacts được xác minh PASS. Không huấn luyện lại FW512/FS16.**
