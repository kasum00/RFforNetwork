# Stage 7.3A — Báo cáo chuẩn bị huấn luyện SVM chính thức

**Ngày lập:** 22/10/2026 (Asia/Saigon)  
**Trạng thái:** Hoàn tất phần chuẩn bị/audit; **full SVM vẫn `BLOCKED_STAGE73A`**.  
**Phạm vi:** CICIoMT2024 `WINDOWS_CPLUS`, hai nhánh Baseline SVM và MI-RFE + SVM. Không có official SVM training hay Test metric ở giai đoạn này.

## 1. Nguồn bằng chứng và mức xác minh

- Source cục bộ đã audit: `huyen_code/ciciomt24_training.py`, `ciciomt24_svm_probe.py`, `ciciomt24_svm_scaling_probe.py`, `colab/experiment_launcher.py`, `checkpoint_manager.py`, `dataset_validator.py`, notebook và các test liên quan. Chi tiết tại [STAGE7_3A_SVM_FULL_AUDIT.md](STAGE7_3A_SVM_FULL_AUDIT.md).
- Artifact Stage 7.2 được dùng để đánh giá kỹ thuật có giới hạn: `baseline_svm_FW256_FS8_scaling02.json` và `mi_rfe_svm_FW256_FS8_scaling02.json`. Không dùng `mi_rfe_svm_FW256_FS8_scaling01.json` vì sai identity. Probe không phải official result.
- Log Colab do người dùng gửi: trước khi sửa test, `Ran 52 tests ... FAILED (failures=1, skipped=3)` do assertion tĩnh sai chỗ tạo `test_evaluation_calls`. Đã sửa **test**, không sửa mã huấn luyện. Log đầy đủ của lần chạy lại sau sửa **chưa được gửi**; preflight mới báo `unit_tests_current_code=PASS`, tức evidence trong runtime được chấp nhận, nhưng không suy ra số test PASS chính xác.
- Log preflight Colab do người dùng gửi cho `baseline_svm/FW_512_FS_16`, run `baseline_svm_FW512_FS16_review02`. Codex **không tự chạy** test, audit dataset, probe, Colab hoặc training để tạo báo cáo này. Các kết quả PASS/FAIL bên dưới là từ log người dùng, không phải kiểm chứng độc lập trên Colab.

## 2. Audit khoa học và chống rò rỉ

| Nội dung | Baseline SVM | MI-RFE + SVM |
| --- | --- | --- |
| Đầu vào | Đúng 380 thống kê hành vi; metadata/provenance không vào X | MI Top100 từ 380 feature, Train-only |
| SVC | RBF, C=10, gamma=`scale`, probability=True, class_weight=None, shrinking=True, cache_size=10000 MB, max_iter=20000, seed42 | RBF, C=10, gamma=`scale`, probability=False, class_weight=`balanced`, shrinking=True, cache_size=10000 MB, max_iter=20000, seed42 |
| Feature selection | Không MI/RFE | RF-RFE 100 cây, depth15, balanced, `step=1`, target1; k=1..100 chọn bằng Validation Accuracy, hòa chọn k nhỏ |
| Final fit/Test | Fit scaler/SVC trên Train+Validation; chỉ sau final fit mới đọc Test | Khóa MI/RFE ranking và k, fit lại scaler/SVC trên Train+Validation, không fit lại MI/RFE; sau đó mới đọc Test một lần |

`prepare_xy` whitelist 380 cột và thay NaN/±Inf bằng 0 theo bản thích nghi. Khác biệt với CTU-13 là dataset/nhãn `SCENARIO_PROXY`, sửa mismatch tên feature numeric và nạp Test muộn; không tự thay thuật toán hoặc tham số. Sliding windows/split C+ đã có từ các giai đoạn trước, không tạo lại ở Stage 7.3A. **Không có flow/window overlap** theo kiểm định trước đó không đồng nghĩa **capture/session độc lập**; điều này vẫn chưa được chứng minh.

## 3. Source và dataset identity trong preflight

| Trường | Giá trị trong log Colab |
| --- | --- |
| Experiment/config/run | `baseline_svm` / `FW_512_FS_16` / `baseline_svm_FW512_FS16_review02` |
| Git commit, checkout | `17378a6574a45fee031a8a5a352a8f836d5ae1f6`; `clean_git_checkout=PASS` trên worktree được kiểm tra |
| Training SHA256 | `aecacfac4fa85d5b11fb8e754c83aeea0f85d7fa0bb4caa60fdca2f78d6d4f1f` |
| Sliding-window SHA256 | `555f51d5c70b4ab6c3677928ccfea33b4fbec9ca96af5faa574151730eecc40a` |
| C+ manifest SHA256 | `ba7e1b3315673e1fa4f4c36f30984836f84179122aa7efe8875f9a6cc75adcd9` |
| C+ validation SHA256 | `1c67873a76321ef50afa537c7b48624113ce1097e413de524e6bb6e0b1e38f70` |
| Window summary SHA256 | `4b0b26a2ad22b9de8ed4798dffb9f60d5fc5e730a8b05abd5d397d3e2d80e02f` |
| Dataset fingerprint SHA256 | `ad92f182e4db8906bb62a2cbfbc29aaf67d57491d98788ebe81dae5f43897927` |

Training hash khớp artifact Stage 7.2 sau lần đồng bộ Windows/Colab; năm script CTU-13 đã được source check chấp nhận (`approved_original_scripts=PASS`). `cross_platform_source_contract=PASS` xác nhận ROOT tương đối và hai `PureWindowsPath(...).stem` theo guard hiện tại. Đây là identity cụ thể của log này; thay code, commit, manifest, window summary hoặc run name thì phải audit/duyệt lại. Fingerprint của `dataset_validator` gồm manifest + summary + đường dẫn/kích thước CSV, **không** phải full-content SHA256 của 322 window CSV. Checkout `/content/RFforNetwork` cũ từng có sửa đổi cục bộ; `clean_git_checkout=PASS` chỉ áp dụng cho checkout/worktree mà preflight đã kiểm tra, không chứng minh checkout cũ sạch.

## 4. Kết quả preflight Stage 7.3A

| Gate | Kết quả | Ý nghĩa |
| --- | --- | --- |
| Repository, Python/profile khoa học, unit evidence, năm script gốc, manifest/schema | PASS | Cấu trúc và metadata phù hợp theo kiểm tra hiện có. |
| Không trùng local/Drive run, Drive mounted | PASS | Tên run `review02` chưa thấy output trùng tại thời điểm kiểm tra. |
| RAM và local disk admission | PASS | Đủ theo **scenario admission**, không phải chứng minh full peak/runtime. |
| Portable source contract, Git checkout sạch | PASS | Đã vượt các kiểm tra source cục bộ của worktree hiện tại. |
| `svm_technical_probe` | FAIL | Ô Stage 7.3A không truyền một artifact Stage 7.1 hợp lệ gắn đúng `baseline_svm/FW_512_FS_16` và source/window hash. Stage 7.2 FW256/FS8 không thay thế. Không có bằng chứng ở log rằng SVM fit thất bại. |
| `svm_full_resource_gate_reviewed` | FAIL | Cố ý đóng; probe PASS cũng không tự mở full. |
| `source_and_dataset_identity_approved` | FAIL | Chưa truyền identity được người dùng phê duyệt; **không** có nghĩa hash quan sát tự nó mismatch. |
| `per_run_budget_approval_bound` | FAIL | Chưa có phê duyệt time/RSS gắn đúng experiment/config/run_name/commit/snapshot. |
| `stage73b_implementation_and_authorization` | FAIL | Cố định chặn ở Stage 7.3A; Stage 7.3B chưa triển khai/duyệt. |

**Kết quả tổng:** `STATUS=BLOCKED_STAGE73A`, `model_fit_calls=0`, `test_feature_rows_read=0`. Không có model chính thức, Test Accuracy hoặc confusion matrix. Không sửa thủ công các cờ FAIL thành PASS.

## 5. Tài nguyên và mức độ chắc chắn

Phiên Colab trong log: Python 3.13.15, 4 CPU vật lý/8 logical, RAM tổng 54.750.404.608 B (~50,99 GiB), available ~52.066.844.672 B (~48,49 GiB), đĩa local trống 164.993.728.512 B (~153,66 GiB). Scenario RAM 19.552.686.992 B (~18,21 GiB) dưới admission budget 44.256.817.971 B (~41,22 GiB), nên RAM guard PASS. `cache_size=10000` là trần kernel cache, không phải mức RAM chắc chắn cấp phát; support-vector/forest memory và runtime full vẫn UNKNOWN. GPU Tesla T4 có mặt nhưng không tăng tốc pipeline scikit-learn này. `selected_scientific_profile=PASS`, trong khi `scientific_versions_match_stage5=False`: không tuyên bố tái lập bitwise với Stage 5.

| Cấu hình | Train | Validation | Test | Final fit Train+Val | Đánh giá full Baseline / MI-RFE+SVM |
| --- | ---: | ---: | ---: | ---: | --- |
| FW_512_FS_16 | 673.559 | 224.373 | 224.287 | 897.932 | UNKNOWN runtime/peak RAM cho cả hai nhánh |
| FW_256_FS_16 | 674.169 | 224.677 | 224.631 | 898.846 | UNKNOWN runtime/peak RAM cho cả hai nhánh |
| FW_512_FS_8 | 1.347.101 | 448.735 | 448.563 | 1.795.836 | UNKNOWN runtime/peak RAM cho cả hai nhánh |
| FW_256_FS_8 | 1.348.317 | 449.343 | 449.251 | 1.797.660 | UNKNOWN runtime/peak RAM cho cả hai nhánh |

Đây là **8 combination** (2 nhánh × 4 FW/FS), không phải 8 run đã thực hiện. Stage 7.2 probe tối đa 64k Train windows: Baseline fit 51.200 dòng trong 131,90 s, peak process-tree RSS 894,4 MiB; MI-RFE+SVM chỉ chạy MI/partial RFE và k=1/10/100, peak 1.066,5 MiB. Không ngoại suy tuyến tính lên ~0,9–1,8 triệu dòng; chi phí RF-RFE đầy đủ và 100 SVC validation fits chưa đo. Bảng chi tiết tại [STAGE7_3A_RESOURCE_BUDGET.md](STAGE7_3A_RESOURCE_BUDGET.md).

## 6. Checkpoint, giới hạn và quyết định

Checkpoint hiện ghi completed-stage evidence; `resume_supported=False`. Chỉ được tái publish sau `run_validated` với đúng một lần Test evaluation trong event log; không resume một `SVC.fit`, RFE hay k-search bị ngắt, không tự đọc Test lần hai. Full SVM chưa có worker isolation/timeout hard cap được kiểm thử. Đường CLI `ciciomt24_training.py train --confirm-full-training` có thể bypass Colab gate, nên **không dùng** cho SVM ở Stage 7.3A và phải đóng/kiểm soát ở Stage 7.3B. RSS polling không bảo đảm ngăn OOM.

**Đề xuất thứ tự nếu sau này được duyệt:** Baseline SVM `FW_512_FS_16` đầu tiên vì ít dòng final fit nhất và không có MI/RFE/100 k-search; đây không phải lệnh chạy. Trước Stage 7.3B cần (1) log unit tests đầy đủ sau bản sửa, (2) xác định có cần/gắn probe kỹ thuật đúng experiment/config hay cập nhật quy tắc gate bằng quyết định khoa học tường minh, (3) người dùng chốt ngân sách tối đa elapsed/RSS và safety margin cho đúng một run, (4) triển khai/kiểm thử authorization thống nhất mọi entry point, worker/timeout và artifact schema, (5) đối chiếu lại commit và dataset snapshot. Nếu không đạt, giữ `BLOCKED_STAGE73A`.

**Dừng tại đây.** Không huấn luyện SVM, không MI/RFE full, không đọc Test để chọn mô hình. Hướng dẫn tự chạy/đồng bộ tại [STAGE7_3A_COLAB_GUIDE.md](STAGE7_3A_COLAB_GUIDE.md).
