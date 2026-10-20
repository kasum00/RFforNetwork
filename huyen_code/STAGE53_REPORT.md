# Giai đoạn 5.3 — Kết quả chính thức MI-RFE + RF FW256/FS16

**Trạng thái: PASS — run đã hoàn tất và artifacts được xác minh.**

Ngày xác minh: 20/10/2026, múi giờ Asia/Saigon.

- Run mới: D:\COMPACT-MI-RFE\huyen_code\runs\stage53_mi_rfe_rf_FW256_FS16
- Run đối chiếu: D:\COMPACT-MI-RFE\huyen_code\runs\stage52_rf_FW512_FS16_vscode_20261019_222901
- Người dùng tự chạy trong VS Code. Run FW256 bắt đầu khoảng 00:33:46 và hoàn tất 01:15:03 ngày 20/10/2026.
- Chỉ thực hiện MI-RFE + Random Forest FW256/FS16. Không chạy SVM/FS8 hoặc huấn luyện lại FW512/FS16.
- Báo cáo được tạo bằng kiểm tra artifacts sau chạy; không fit, predict hoặc evaluate Test thêm lần nào.

## 1. Tóm tắt kết quả

FW256/FS16 chọn **18 feature**, Validation Accuracy cao nhất **1.0**, Test Accuracy **99,9768509244%**, Macro-F1 **97,2631080047%**, Balanced Accuracy **94,8207171315%**, Recall Benign **89,6414342629%**. Tổng thời gian **2476,9956208 giây, khoảng 41 phút 17 giây**; peak RSS lấy số lớn hơn của các sampler khoảng **10,006531 GiB**.

Confusion matrix [Benign, Attack] = [[450,52],[0,224129]]: nhầm 52 Benign thành Attack, không bỏ sót Attack theo nhãn proxy. Không dùng Accuracy cao để che mất mức Recall Benign thấp hơn cấu hình FW512.

Đã thực hiện **132/132 assertion hậu nghiệm PASS**, gồm 100 assertion kiểm tra số mẫu/Accuracy Validation từng k. Đây là kiểm tra artifacts, không phải 132 unit test mới. Tái sử dụng bộ 97/97 unit test đã PASS trước khi chạy.

Audit máy đọc được: D:\COMPACT-MI-RFE\huyen_code\runs\stage53_preflight\stage53_artifact_validation.json

## 2. Hoàn thiện Stage 5.2 và bảo toàn kết quả

[STAGE52_REPORT.md](D:/COMPACT-MI-RFE/huyen_code/STAGE52_REPORT.md) đã được hoàn thiện ở bước trước, xác minh 130/130 assertion cho run FW512/FS16, k=8 và Test Accuracy 0.9999866242804978.

Trong lần xuất báo cáo này, SHA256 của 12 artifacts metadata/model/scaler của Stage52 tiếp tục khớp bản audit trước. Không sửa result.json, model, scaler, feature list, logs hoặc ma trận của Stage52. Không huấn luyện lại và không evaluate Test của Stage52 thêm lần nào.

Lượt Stage52 bị dừng trước đó trong thư mục stage52_mi_rfe_rf_FW512_FS16.incomplete vẫn được giữ nguyên, không dùng làm kết quả chính thức và không resume.

## 3. Dữ liệu chính thức FW256/FS16

Input: D:\COMPACT-MI-RFE\CICIoMT24\WINDOWS_CPLUS\FW_256_FS_16

| Split | Window thực tế | Benign proxy | Attack proxy | Source segment có window |
|---|---:|---:|---:|---:|
| Train | 674.169 | 1.591 | 672.578 | 45 |
| Validation | 224.677 | 521 | 224.156 | 19 |
| Test | 224.631 | 502 | 224.129 | 22 |
| Train + Validation dùng refit | 898.846 | 2.112 | 896.734 | — |

Sử dụng toàn bộ dữ liệu theo manifest, không sampling hoặc undersampling. Đã kiểm tra shape và dtype của sáu file ma trận: X là float32 với đúng 380 cột; y là int8 nhị phân và counts khớp bảng.

X chỉ gồm 76 đặc trưng hành vi × năm thống kê = 380 feature canonical. Metadata, Scenario, Split, Window_ID, Source_CSV và provenance không đưa vào X. Nhãn y là 0=Benign proxy, 1=Attack proxy.

Full loader ghi nhận cả ba split có X finite sau phép thay Inf thành NaN rồi fillna(0), tương thích source. Không sửa CSV window. Nhãn vẫn là **Label_Origin=SCENARIO_PROXY**, không phải ground truth từng flow.

## 4. Pipeline và tham số giữ nguyên

| Bước | Thiết kế đã kiểm tra |
|---|---|
| MI | StandardScaler fit Train 380 cột; mutual_info_classif fit Train, random_state=42; lấy Top100 |
| Chuẩn bị RFE | Lấy lại raw-cleaned Top100, scaler mới fit Train; chỉ transform Validation |
| RF-RFE | 100 cây, max_depth=10, min_samples_split=2, max_features=sqrt, class_weight=balanced, random_state=42, n_jobs=-1 |
| RFE | step=1, n_features_to_select=1; ranking đầy đủ 100 xuống 1, gồm 100 forest fit |
| RF cho từng k | 300 cây, max_depth=20, min_samples_split=3, max_features=sqrt, class_weight=balanced, random_state=42, n_jobs=-1 |
| Chọn k | Ranking cố định; thử đủ k=1..100; Accuracy Validation lớn nhất, hòa giữ k nhỏ hơn |
| Refit | Khóa feature/k, scaler và classifier mới fit Train+Validation; không fit MI/RFE lại |
| Test | Chỉ load sau refit hoàn tất; đánh giá một lần |

Model joblib thật có 300 cây, 18 input, classes [0,1] và mọi tham số RF trên khớp source. Scaler có n_features_in_=18, n_samples_seen_=898846. Mean/variance của scaler đã được đối chiếu bằng tổng thống kê các cột đã khóa từ Train+Validation, không gọi fit lại và không dùng Test trong phép đối chiếu này.

Hai run dùng cùng môi trường: Python 3.10.0, NumPy 1.26.4, Pandas 2.3.3, SciPy 1.15.3, scikit-learn 1.7.2; seed=42.

### Khác biệt code giữa hai run

Chỉ bổ sung ba dòng hiển thị Validation Accuracy hiện tại và fit_seconds khi một k hoàn tất. Đã kiểm chứng bằng byte: loại đúng ba dòng logging này khỏi code Stage53 thì SHA256 bằng chính SHA256 implementation trong preflight Stage52. Không đổi thuật toán, dữ liệu, seed, scaler, MI, RFE, RF hoặc luật chọn k.

- SHA256 implementation Stage52: 7631093FDC76F64CC74F5C7C1E595E11237536E08F8DBA5FD7DC05494B60BCC6
- SHA256 implementation Stage53: 5E0FD78E8F306F65B38C5AF811FE8615C3AD6C278357DEB3C190963D608CAD5C

Năm script CTU-13 gốc không đổi; các SHA256 khớp preflight và snapshot được duyệt.

## 5. Best k và 18 feature được chọn

MI có đủ 380 tên canonical, scores hữu hạn/không âm và sort giảm. mi_top100.json khớp 100 tên đầu. RFE chứa đúng Top100 với rank duy nhất từ 1 đến 100.

K-search JSON và events đều đủ k=1..100. Accuracy Validation tối đa là 1.0 tại k=18..100; vì hòa giữ k nhỏ nên **best_k=18**. Không dùng F1 hoặc Test để phá hòa, không điều chỉnh k sau khi xem Test.

Danh sách dưới trùng giữa selection_frozen.json, features.json, result.json và 18 tên đầu của RFE ranking:

| RFE rank | Selected feature |
|---:|---|
| 1 | Pkt Len Min_mean |
| 2 | Pkt Len Min_std |
| 3 | Fwd Pkt Len Min_std |
| 4 | Fwd Pkt Len Max_mean |
| 5 | Fwd Seg Size Avg_mean |
| 6 | Fwd Pkt Len Mean_mean |
| 7 | Fwd Pkt Len Max_median |
| 8 | Active Std_std |
| 9 | Fwd Pkt Len Std_mean |
| 10 | Fwd Pkt Len Mean_median |
| 11 | Pkt Size Avg_std |
| 12 | PSH Flag Cnt_std |
| 13 | Fwd Seg Size Avg_median |
| 14 | Init Bwd Win Byts_std |
| 15 | Init Bwd Win Byts_median |
| 16 | PSH Flag Cnt_mean |
| 17 | Bwd PSH Flags_mean |
| 18 | Bwd IAT Tot_std |

Giảm 380 xuống 18: loại 362 cột, mức giảm **95,2631578947%**. Từ Top100 xuống 18 giảm 82%. FW512/FS16 giảm 380 xuống 8, tương đương 97,8947368421%. Hai danh sách có bảy feature chung; không dùng sự khác nhau này để sửa pipeline.

Mức giảm cột của classifier cuối không phải mức giảm tương ứng thời gian toàn pipeline: MI vẫn tính trên 380 cột, RFE/search vẫn chạy đầy đủ.

## 6. Khóa lựa chọn, refit rồi mới đọc Test

Tất cả mốc dưới là giờ Asia/Saigon ngày 20/10/2026:

| Event | Thời điểm |
|---|---|
| Run bắt đầu | 00:33:46,053 |
| Load Train xong | 00:34:26,811 |
| Load Validation xong | 00:34:41,275 |
| MI bắt đầu | 00:34:41,520 |
| MI hoàn tất | 00:57:03,524 |
| RFE bắt đầu | 00:57:05,536 |
| RFE hoàn tất | 01:01:39,127 |
| Khóa k=18 và features | 01:14:36,835 |
| Refit Train+Validation bắt đầu | 01:14:36,844 |
| Refit hoàn tất | 01:14:48,222 |
| Lần đầu load Test | 01:14:48,231 |
| Test evaluation bắt đầu | 01:15:02,331 |
| Test evaluation hoàn tất | 01:15:02,892 |
| Run validated | 01:15:03,045 |

Đã xác minh trật tự khóa → refit xong → load Test. Chỉ một event bắt đầu/kết thúc Test evaluation và test_evaluation_calls=1. Các kiểm tra hậu nghiệm không chạy predict hoặc metrics evaluator trên Test nữa; chỉ đối chiếu số học từ confusion matrix đã lưu.

## 7. Kết quả Test FW256/FS16

| Chỉ số | Giá trị [0,1] | Phần trăm |
|---|---:|---:|
| Accuracy | 0,9997685092440491 | 99,9768509244% |
| Precision Attack | 0,9997680445711278 | 99,9768044571% |
| Recall Attack | 1,0 | 100% |
| F1 Attack | 0,9998840088331735 | 99,9884008833% |
| Precision Benign | 1,0 | 100% |
| Recall Benign | 0,896414342629482 | 89,6414342629% |
| F1 Benign | 0,9453781512605042 | 94,5378151261% |
| Macro-F1 | 0,9726310800468388 | 97,2631080047% |
| Balanced Accuracy | 0,9482071713147411 | 94,8207171315% |

Confusion matrix có hàng là nhãn thật proxy, cột là dự đoán:

| True / Predicted | Benign | Attack |
|---|---:|---:|
| Benign proxy | 450 | 52 |
| Attack proxy | 0 | 224.129 |

Tổng Test 224631 mẫu. Benign chỉ có 502 mẫu; tỷ lệ nhầm Benign thành Attack là 52/502 ≈10,358566%. Recall Attack=100% ở đây là theo scenario proxy, không phải chứng minh phát hiện mọi attack có ground truth ngoài thực tế.

## 8. Bảng so sánh chính thức hai cấu hình

Cả hai đều là MI-RFE+RF, cùng split C+ nền tảng và FS=16. Số window/nhãn Test khác nhau do FW khác nhau; đây là so sánh mô tả theo thiết kế, không phải hai dự đoán trên một tập window giống hệt nhau.

| Đại lượng | FW256/FS16 — Stage53 | FW512/FS16 — Stage52 |
|---|---:|---:|
| Train windows | 674.169 | 673.559 |
| Validation windows | 224.677 | 224.373 |
| Test windows | 224.631 | 224.287 |
| Benign Test | 502 | 470 |
| Selected features | 18 | 8 |
| Giảm so với 380 feature | 95,263158% | 97,894737% |
| Best Validation Accuracy | 100% | 100% |
| Test Accuracy | 99,976851% | 99,998662% |
| Precision Attack | 99,976804% | 99,998660% |
| Recall Attack | 100% | 100% |
| F1 Attack | 99,988401% | 99,999330% |
| Macro-F1 | 97,263108% | 99,839580% |
| Balanced Accuracy | 94,820717% | 99,680851% |
| Precision Benign | 100% | 100% |
| Recall Benign | **89,641434%** | **99,361702%** |
| F1 Benign | 94,537815% | 99,679829% |
| Benign bị nhầm thành Attack | 52/502 | 3/470 |
| Attack bị nhầm thành Benign | 0/224129 | 0/223817 |
| Tổng elapsed | 2476,995621 s | 2529,147748 s |
| Tổng elapsed, làm tròn giây | 41 phút 17 giây | 42 phút 9 giây |
| Peak RSS bảo thủ: max các sampler | 10,006531 GiB | 9,993862 GiB |

Theo các lượt chạy này, FW512/FS16 có ít feature hơn và các metric Test tốt hơn, đặc biệt lớp Benign. Chênh lệch FW256 trừ FW512:

- Accuracy: -0,0218115 điểm phần trăm.
- Macro-F1: -2,5764715 điểm phần trăm.
- Balanced Accuracy: -4,8601339 điểm phần trăm.
- Recall Benign: **-9,7202679 điểm phần trăm**.

Chênh lệch Accuracy nhìn rất nhỏ nhưng che khuất khác biệt lớn ở lớp Benign. Không kết luận thống kê rằng một FW luôn tốt hơn chỉ từ một seed/holdout có window phụ thuộc nhau. Không dùng bảng Test này để chọn lại k, đổi class_weight hoặc chỉnh classifier. Tiêu chí lựa chọn của source vẫn là Validation Accuracy.

FW256 hoàn tất nhanh hơn khoảng 52,152 giây trong hai run quan sát được. Chưa đo lặp nhiều lần nên không tuyên bố tăng tốc có ý nghĩa hay tổng quát cho mọi môi trường.

## 9. Thời gian và RAM theo công đoạn

| Công đoạn | FW256/FS16 | FW512/FS16 |
|---|---:|---:|
| Load/validate Train, theo event | 40,750 s | 41,844 s |
| Load/validate Validation, theo event | 14,454 s | 14,270 s |
| MI | 1341,727681 s | 1312,973972 s |
| RF-RFE đầy đủ | 273,535643 s | 292,333995 s |
| Tổng fit RF của 100 k | 725,721849 s | 788,279670 s |
| Predict + metrics Validation của 100 k | 44,409252 s | 44,797099 s |
| K-search wall, gồm overhead/logging | 777,707850 s | 840,701132 s |
| Refit RF, riêng fit | 10,597612 s | 8,646837 s |
| Giai đoạn refit gồm chuẩn bị/scaling | 11,378438 s | 9,278285 s |
| Load/validate Test | 14,092299 s | 14,443014 s |
| Test predict + metrics | 0,431821 s | 0,430662 s |
| Tổng run | 2476,995621 s | 2529,147748 s |
| Peak whole-run sampler | 10,006531 GiB | 9,948452 GiB |
| Peak sampler MI | 9,987415 GiB | 9,993862 GiB |

Các thời gian fit riêng không bao gồm toàn bộ I/O/chuẩn bị dữ liệu/metrics/logging nên không phải mọi hàng cộng lại bằng tổng run. Hai sampler RAM có thời điểm lấy mẫu 50 ms khác nhau; dùng max khi so sánh hoặc kiểm tra tài nguyên. Không gọi RSS là commit tổng hoặc hard upper bound RAM.

Không có warning được ghi trong RF-RFE, 100 RF classifier fit hoặc refit cuối của cả hai run. RF không có hội tụ kiểu max_iter của SVM. MI measurement không lưu danh sách warning riêng; không suy diễn từ đó rằng mọi warning MI đều đã được kiểm tra độc lập.

Artifacts FW256, gồm sáu memmap: **1.711.409.885 byte**. Model 2.143.049 byte; final_scaler 1047 byte. Chỉ đọc model/scaler để kiểm tra params/state trong bước xuất báo cáo; không dự đoán lại.

## 10. Bằng chứng resource admission đã PASS

Lần đầu Stage53 bị chặn tại preflight khi available khoảng 9,35 GiB. Resource check FAIL cũ được giữ nguyên tại runs\stage53_preflight\resource_check.json; đó là lịch sử checkpoint, không phải trạng thái run đã hoàn tất.

Trước lần chạy trong VS Code, guard mới được tính lại theo đúng peak quan sát Stage52 và headroom đã chốt:

| Đại lượng | Byte | GiB |
|---|---:|---:|
| Peak tham chiếu, lấy max Stage52 | 10.730.827.776 | 9,993862 |
| Peak scale theo số hàng target/reference | 10.745.366.833 | 10,007403 |
| Headroom max(2 GiB, 20% peak scale) | 2.149.073.367 | 2,001481 |
| Required available | 12.894.440.200 | **12,008883** |
| Available trong guard trước run | 13.303.476.224 | **12,389828** |
| Dư so với required | 409.036.024 | 0,380944 |
| Available trong preflight của runner | 13.295.570.944 | 12,382465 |

**Guard PASS.** Công thức dùng max tỷ lệ Train, Validation và tổng ba tập của FW256/FW512, rồi cộng headroom. Không dựa riêng vào admission floor khoảng 5,14 GiB của Stage 5.1, không hạ ngưỡng hoặc giảm mẫu để chạy.

Bản ghi guard độc lập:
D:\COMPACT-MI-RFE\huyen_code\runs\stage53_preflight\admission_1792431225858410500.json

Runner ghi disk free trước chạy khoảng 94,087 GiB, đủ cho ma trận/artifacts/dự phòng. Snapshot tài nguyên trước đó đã ghi nhận pagefile D khoảng 19 GiB và giới hạn đo pagefile usage; pagefile không được dùng để bỏ qua guard RAM vật lý. Không thay đổi pagefile trong quy trình này.

Peak thực tế FW256 khoảng 10,007 GiB, vẫn dưới available trước chạy. Đây chỉ là kết quả quan sát một run, không chứng minh mọi lần chạy tương lai có cùng peak.

## 11. Kiểm định đầy đủ sau chạy

| Nhóm kiểm tra | Kết quả |
|---|---|
| 12 artifacts metadata/model/scaler và sáu memmap đọc được | PASS |
| Đúng một experiment MI-RFE+RF, đúng FW256/FS16 | PASS |
| Counts và labels Train/Val/Test khớp manifest | PASS |
| Canonical X380, MI380 hữu hạn, Top100 chính xác | PASS |
| RFE100 xuống 1, đủ rank, params source | PASS theo code/hash/log |
| K-search đủ k=1..100, mỗi k có counts/Accuracy Validation đúng | PASS |
| Best k=18, Validation Accuracy=1.0 và luật hòa k nhỏ | PASS |
| Frozen features khớp prefix RFE và mọi artifact | PASS |
| RF cuối 300 cây, 18 input, params/class đúng | PASS |
| Scaler 898846 mẫu và mean/variance Train+Validation đúng | PASS |
| Khóa feature/refit xong trước lần đầu đọc Test | PASS |
| Test evaluation đúng một lần; metrics nhất quán từ confusion matrix | PASS |
| X finite theo log full loader cả ba split | PASS |
| Guard peak thực tế cộng headroom đã PASS trước chạy | PASS |
| SHA256 implementation khớp preflight Stage53 | PASS |
| Năm SHA256 source CTU-13 khớp snapshot | PASS |
| 322 CSV window size/mtime khớp preflight | PASS |
| 12 SHA256 artifacts Stage52 không thay đổi | PASS |
| Capture/session independence | CHƯA ĐƯỢC CHỨNG MINH |

Đã kiểm tra stats/labels từ memmap đã lưu, không quét lại 71 CSV flow hoặc toàn bộ CSV window. Size/mtime snapshot không tương đương hash nội dung toàn dataset. Các kết luận về pipeline dựa trên code/hash/log/artifacts nhất quán, không phải bằng chứng chống mọi khả năng sửa dữ liệu hoặc log từ bên ngoài.

Không thay đổi result.json, artifacts model/scaler/log của hai run trong quá trình làm báo cáo. Audit mới lưu bên ngoài run, trong stage53_preflight.

## 12. Giới hạn khoa học và cách trình bày

- Tất cả nhãn là SCENARIO_PROXY, không phải ground truth từng flow. Các run có Recall Attack=100% theo proxy không chứng minh phát hiện mọi tấn công thực tế.
- C+ chia flow trước window và không dùng chung source row qua split; windows trong cùng tập vẫn chồng lấn mạnh. Số window không bằng số quan sát IID độc lập.
- Không source-row overlap, không exact duplicate, không activity crossing và capture/session độc lập là bốn điều kiện khác nhau. Metadata hiện có chưa chứng minh capture/session independence; không tuyên bố dữ liệu hoàn toàn leakage-free.
- Loại metadata khỏi X giúp hạn chế rò rỉ trực tiếp, nhưng signature môi trường/capture/thiết bị vẫn có thể nằm trong feature hành vi.
- Benign hiếm. Accuracy gần 100% không đủ; phải trình bày Recall Benign, Macro-F1, Balanced Accuracy và confusion matrix.
- Một số scenario thiếu window held-out theo thiết kế; không sửa chia tập để tăng độ phủ hoặc tăng điểm.
- Hai cấu hình có tập window và số Benign khác nhau, chỉ một seed=42; chưa có kiểm định lặp/độc lập capture để tuyên bố ưu thế thống kê.
- Đã thử 100 k trên cùng Validation; giữ đúng source nhưng phải công bố khả năng thích nghi riêng với tập Validation. Không dùng Test để sửa lựa chọn.
- Tỷ lệ flow C+ 60:20:20 không bắt buộc tỷ lệ window đúng tuyệt đối 60:20:20, vì chỉ tạo window đủ độ dài trong từng source segment.

Có thể báo cáo với giảng viên: “Tôi đã tái hiện hai cấu hình MI-RFE+RF với cùng tham số source. FW256/FS16 chọn 18 feature, FW512/FS16 chọn tám feature. Cả hai Validation Accuracy đạt 1.0, nhưng Test Recall Benign khác đáng kể, 89,64% so với 99,36%. Tôi báo cáo đầy đủ sự khác biệt thay vì chỉ dựa vào Accuracy; các kết luận vẫn giới hạn bởi scenario proxy và chưa xác nhận capture/session độc lập.”

## 13. Đề xuất thứ tự thực nghiệm tiếp theo — chưa thực hiện

Sau khi được phê duyệt riêng và resource admission PASS:

1. MI-RFE+RF FW512/FS8 để đối chiếu FS khi giữ FW512, rồi MI-RFE+RF FW256/FS8 để hoàn thiện bốn cấu hình RF.
2. Trước mỗi FS8 phải tính lại guard từ peak thực tế và số mẫu gần gấp đôi, không dùng lại ngưỡng 12 GiB cho FS16. Ngoại suy theo guard hiện tại có thể yêu cầu xấp xỉ 24 GiB available, lớn hơn tổng RAM khoảng 23,73 GiB của máy; cần xem xét tài nguyên phù hợp trước, không tự giảm mẫu.
3. SVM Baseline và MI-RFE-SVM chỉ sau quyết định nguồn lực/khả thi riêng, giữ tham số source; không tự đổi RBF thành thuật toán khác hoặc tắt probability của baseline.
4. Thứ tự trên phục vụ hoàn thiện thiết kế và đối chiếu FW/FS, không phải chọn hyperparameter mới theo Test.

**Không chạy thêm thí nghiệm trong lượt xuất báo cáo này.**

## 14. Artifacts và lệnh kiểm tra không huấn luyện lại

- Báo cáo này: D:\COMPACT-MI-RFE\huyen_code\STAGE53_REPORT.md
- Báo cáo đối chiếu: D:\COMPACT-MI-RFE\huyen_code\STAGE52_REPORT.md
- Result chính thức FW256: D:\COMPACT-MI-RFE\huyen_code\runs\stage53_mi_rfe_rf_FW256_FS16\result.json
- Audit mới: D:\COMPACT-MI-RFE\huyen_code\runs\stage53_preflight\stage53_artifact_validation.json
- Result chính thức FW512: D:\COMPACT-MI-RFE\huyen_code\runs\stage52_rf_FW512_FS16_vscode_20261019_222901\result.json

PowerShell:

    $stage53Run = 'D:\COMPACT-MI-RFE\huyen_code\runs\stage53_mi_rfe_rf_FW256_FS16'
    $stage52Run = 'D:\COMPACT-MI-RFE\huyen_code\runs\stage52_rf_FW512_FS16_vscode_20261019_222901'
    Get-Content -LiteralPath (Join-Path $stage53Run 'result.json') -Raw
    Get-Content -LiteralPath (Join-Path $stage53Run 'selection_frozen.json') -Raw
    $stage53K = Get-Content -LiteralPath (Join-Path $stage53Run 'k_search_results.json') -Raw | ConvertFrom-Json
    $stage53K.Count
    $stage53K | Sort-Object @{Expression='accuracy';Descending=$true}, @{Expression='k';Ascending=$true} | Select-Object -First 1 k,accuracy
    Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $stage53Run 'model.joblib'), (Join-Path $stage53Run 'final_scaler.joblib')
    Get-Content -LiteralPath 'D:\COMPACT-MI-RFE\huyen_code\runs\stage53_preflight\stage53_artifact_validation.json' -Raw

Không chạy lại model.predict hoặc Test evaluation khi chỉ cần xem kết quả.

**Kết luận: Giai đoạn 5.3 đã hoàn tất và kiểm định PASS; đã có bảng so sánh chính thức FW256/FS16 với FW512/FS16. Dừng sau báo cáo, không chạy SVM hoặc FS8.**
