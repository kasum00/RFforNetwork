# STAGE 7.2 — Bounded SVM scaling probe, FW_256_FS_8

**Cập nhật 22/10/2026.** **Thu thập bằng chứng kỹ thuật: HOÀN TẤT cho cả Baseline SVM và MI-RFE + SVM** ở 16.000/32.000/64.000 window Train. **Đánh giá tài nguyên cho full SVM: REVIEW_REQUIRED; full training chưa được cấp phép.** Đây không phải Accuracy/Validation/Test chính thức và không chọn best_k. Codex chỉ đọc JSON và log unit test do người dùng cung cấp, không chạy lại probe hay mô hình.

## 1. Artifact dùng trong báo cáo

- Baseline: `E:\baseline_svm_FW256_FS8_scaling02.json` và ba worker `.n16000/.n32000/.n64000.json`.
- MI-RFE + SVM: `E:\mi_rfe_svm_FW256_FS8_scaling02.json` và ba worker `.n16000/.n32000/.n64000.json`.
- File cũ `E:\mi_rfe_svm_FW256_FS8_scaling01.json` **bị loại khỏi phân tích nhánh MI-RFE**: nội dung của nó là `experiment=baseline_svm` với 380 feature, không có MI/RFE. Không đổi tên hay ghi đè artifact này.
- Ba JSON Stage 7.1 (2k/5k/8.192) chỉ làm bối cảnh, không thay thế số liệu Stage 7.2.

Log người dùng gửi: Stage 7.2 **14/14**, Stage 7.1 **9/9**, Colab Stage 6 **20/20** unit tests PASS; tổng **43/43**. Đây là output người dùng, không phải tests Codex tự chạy. Warning ZIP duplicate trong mock test Stage 6 không làm suite FAIL.

## 2. Kiểm tra tính nhất quán và an toàn

| Kiểm tra | Kết quả / giới hạn |
| --- | --- |
| Identity | **PASS theo JSON:** cả hai parent là `stage7.2.train_only.svm_scaling.v1`, đúng experiment, `FW_256_FS_8`, seed 42 và sizes 16k/32k/64k. |
| Parent ↔ worker | **PASS:** với từng size, mọi trường có trong worker JSON khớp run tương ứng trong parent; worker return code 0, `supervisor_stop_reason=null`. |
| Sample/fit/holdout | **PASS theo JSON:** actual=requested; fit/holdout 80/20 (12.800/3.200, 25.600/6.400, 51.200/12.800). Hai nhánh có cùng SHA256 chuỗi Window_ID, cùng Benign/Attack và cùng fit/holdout class counts tại từng size: so sánh kỹ thuật trên cùng mẫu. |
| Nhãn/schema | **PASS ở mức code path + JSON:** 380 đặc trưng hành vi, nhãn `SCENARIO_PROXY`. Sample không thay thế flow-level ground truth. Code path kiểm tra Window_ID không lặp; JSON chỉ giữ hash ID, không đủ dựng lại từng ID độc lập. |
| MI/RFE | **PASS cho phạm vi probe:** MI Top100 và partial RF-RFE trên internal Train-fit; mỗi size có đúng 4 forest fits, `complete_ranking=false`, SVC k=1/10/100. Đây không phải ranking đầy đủ hoặc tìm k chính thức. |
| Hội tụ/cảnh báo | **PASS theo JSON:** cả 3 Baseline SVC và cả 9 MI-RFE-SVM SVC có `fit_status=0`, `converged=true`, `warnings=[]`; partial RFE cũng không warning. |
| Resource guard | **PASS trong phạm vi đã chạy:** không timeout/stop, 1.200 giây và cap tree RSS 12 GiB/worker chưa bị chạm. RSS được poll, không phải giới hạn bộ nhớ cứng của OS. |
| Không dùng Validation/Test | JSON của cả hai nhánh ghi `validation_csv_rows_read=0`, `test_csv_rows_read=0`, `scientific_result=false`, `sampled_training=true`. Đây là bằng chứng cấu trúc/log và tests mock, không phải trace I/O hệ thống độc lập. Không báo Accuracy từ internal holdout. |
| Full-run gate | **BLOCKED theo JSON:** `resource_gate=REVIEW_REQUIRED`, `full_training_authorized=false`, `official_best_k=null` ở cả hai nhánh. |

**Provenance:** Hai parent cùng hash training/scaling code, window summary và C+ manifest, nên so sánh nội bộ hai nhánh dùng cùng snapshot Colab. Năm script CTU-13 và window summary khớp byte với bản Windows; scaling module và manifest khớp sau chuẩn hóa CRLF→LF. Riêng `ciciomt24_training.py` Colab ghi SHA256 `AECACFAC4FA85D5B11FB8E754C83AEEA0F85D7FA0BB4CAA60FDCA2F78D6D4F1F`, trong khi Windows hiện tại là `5E0FD78E8F306F65B38C5AF811FE8615C3AD6C278357DEB3C190963D608CAD5C` cả raw lẫn LF-normalized. **Chưa thể giải thích khác biệt này khi không có file source/commit từ Colab.** Số liệu JSON có tính nhất quán nội bộ nhưng tái lập với checkout Windows hiện tại cần đối chiếu thêm.

## 3. Runtime và phân bố mẫu

Colab: Python 3.13.15, NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2; 4 CPU vật lý/8 logic, RAM tổng 50,99 GiB, RAM available đầu run khoảng 48,8 GiB. Full Train FW256/FS8 theo manifest: **1.348.317 window = 3.182 Benign + 1.345.135 Attack**. Mẫu 64k chỉ chiếm **4,75%** Train. Class-aware sampling không thay thế window, giữ tỷ lệ gần tự nhiên; `minority_floor_applied=false` ở cả ba size.

| Size Train sample | Benign / Attack | Source / scenario | Fit / internal holdout |
| ---: | ---: | ---: | ---: |
| 16.000 | 38 / 15.962 | 36 / 18 | 12.800 / 3.200 |
| 32.000 | 76 / 31.924 | 37 / 18 | 25.600 / 6.400 |
| 64.000 | 151 / 63.849 | 41 / 18 | 51.200 / 12.800 |

Mẫu theo lớp này chỉ phục vụ phép đo kỹ thuật; không chứng minh đại diện đầy đủ theo thời gian, capture hoặc session. Internal holdout lấy từ Train, **không phải Validation chính thức**.

## 4. Thời gian và RAM quan sát

Đơn vị RSS là **MiB**. Peak tree RSS là mẫu poll của supervisor; peak RSS theo phép đo bên trong worker có thể nhỉnh hơn do tần suất/thời điểm lấy mẫu khác nhau.

| Nhánh | Sample | Load s | MI s | Partial RFE s | SVC fit s | Peak tree RSS MiB | Elapsed toàn run |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Baseline | 16k | 97,15 | — | — | 380 feature: **12,55** | 370,4 | |
| Baseline | 32k | 107,03 | — | — | 380 feature: **46,41** | 530,1 | **515,28 s** cho cả ba worker |
| Baseline | 64k | 105,93 | — | — | 380 feature: **131,90** | 894,4 | |
| MI-RFE-SVM | 16k | 101,86 | 19,97 | 1,31 | k1/k10/k100: **0,050 / 0,074 / 0,195** | 440,6 | |
| MI-RFE-SVM | 32k | 104,52 | 42,30 | 1,59 | **1,650 / 0,154 / 0,617** | 647,1 | **485,05 s** cho cả ba worker |
| MI-RFE-SVM | 64k | 106,09 | 88,43 | 2,07 | **4,070 / 0,477 / 2,110** | 1.066,5 | |

Mỗi size chạy một worker riêng và đọc Train lại; elapsed toàn run gồm load, MI, RFE, fit, predict và overhead. MI peak RSS tại 16k/32k/64k: **459,0 / 650,7 / 1.132,1 MiB**; partial RFE peak: **323,5 / 416,1 / 596,9 MiB**. Baseline 64k có fit peak RSS **913,5 MiB**; MI-RFE-SVM 64k có process peak **1.138,6 MiB**. Các số này không phải RAM full SVM.

## 5. Support vectors và tăng trưởng

| Nhánh / k | SV 16k | SV 32k | SV 64k | Tỷ lệ SV/fit rows ở 64k |
| --- | ---: | ---: | ---: | ---: |
| Baseline / 380 | 556 | 856 | 1.209 | 2,36% |
| MI-RFE-SVM / k=1 | 54 | 2.322 | 2.456 | 4,80% |
| MI-RFE-SVM / k=10 | 119 | 75 | 127 | 0,25% |
| MI-RFE-SVM / k=100 | 220 | 331 | 494 | 0,96% |

Baseline 16k→32k fit/RSS/SV count tăng **3,70×/1,43×/1,54×**; 32k→64k tăng **2,84×/1,69×/1,41×**. MI time tăng **2,12×** rồi **2,09×**; peak tree RSS tăng **1,47×** rồi **1,65×**. SVC k=1 tăng bất thường từ **54 lên 2.322 SV** khi 16k→32k. Ranking partial RFE được tính lại theo từng sample size nên k=1 không nhất thiết là cùng feature giữa các size; không diễn giải biến động này như một đường scaling cố định, cũng không tự coi là lỗi mô hình.

SVC theo MI-RFE chạy nhanh hơn Baseline trong mẫu kỹ thuật này, nhưng `partial_rfe.forest_fits=4` và chỉ thử ba k. Pipeline MI-RFE-SVM chính thức cần ranking đầy đủ (RFE step=1) và chọn trong k=1..100 bằng Validation rồi refit; thời gian/memory có thể khác lớn. **Không dùng probe để chọn feature, best_k hay cấu hình nghiên cứu.**

## 6. Kết luận và điều kiện bước sau

**Stage 7.2 hoàn thành ở mức thu thập và so sánh bounded technical probes cho hai nhánh.** Điều này chỉ chứng minh chúng chạy được đến 64k dưới guard và trên runtime đã quan sát; **không chứng minh full SVM khả thi** với ~1,35 triệu Train window. RBF/libsvm có thể tăng phi tuyến; cache SVC 10.000 MiB là cap cấu hình, không phải cấp phát cố định. RSS polling có thể bỏ sót spike/OOM. Benign cực hiếm, nhãn là `SCENARIO_PROXY`; capture/session independence chưa chứng minh.

Giữ `svm_full_resource_gate_reviewed=False` và không bắt đầu Stage 7.3/full training chỉ từ báo cáo này. Trước quyết định riêng, cần (1) đối chiếu bản `ciciomt24_training.py` hoặc commit thực tế trên Colab với Windows, (2) đánh giá ngân sách thời gian/RAM cho ranking đầy đủ, 100 giá trị k và refit, (3) chấp nhận rủi ro full SVM hoặc chọn phương án nghiên cứu khác bằng quyết định của người dùng. Không giảm dataset/đổi thuật toán hay đọc Test để ra quyết định khi chưa được phê duyệt.
