# Stage 7.3A — SVM official source audit

**Trạng thái BLOCKED_STAGE73A / REVIEW_REQUIRED.** Đây là kiểm tra source và thiết kế guard, không phải một lần chạy Python, unit test, probe hay training. Log 43/43 PASS do người dùng cung cấp là cho Stage 6/7.1/7.2 **trước** thay đổi Stage 7.3A; không được gọi là test PASS của mã mới.

## Source consistency

Đã đối chiếu `huyen_code/ciciomt24_training.py`, `ciciomt24_svm_probe.py`, `ciciomt24_svm_scaling_probe.py`, `colab/experiment_launcher.py`, `checkpoint_manager.py`, `dataset_validator.py`, notebook, các test Stage 5/6/7.1/7.2 và báo cáo Stage 7.2. Windows source hiện có `ROOT = Path(__file__).resolve().parents[1]` và đúng hai chỗ `PureWindowsPath(segment.output_relative_path).stem`. Module guard mới kiểm tra lại, xuất SHA256 source chuẩn hóa CRLF→LF; chỉ khác biệt newline mới được coi không đổi ngữ nghĩa.

**Source consistency chưa PASS.** HEAD Windows quan sát trước sửa là `1258abd523f63dace823daa16eb35fc58e28c6fe`, không phải commit Stage 7.3A được người dùng duyệt. Chưa có commit/push mới, checkout Colab sạch và đối chiếu SHA hai máy. Lệch training hash Windows/Colab trong Stage 7.2 đã được đồng bộ: Windows hiện tại có SHA256 `AECACFAC4FA85D5B11FB8E754C83AEEA0F85D7FA0BB4CAA60FDCA2F78D6D4F1F`, bằng training hash trong hai `scaling02.json`. Năm script CTU-13 cũng khớp byte với hash trong artifact Stage 7.2. Đây chưa chứng minh toàn bộ checkout Colab hiện hành khớp; sliding/adapter/commit mới vẫn cần đối chiếu. `mi_rfe_svm_FW256_FS8_scaling01.json` sai experiment identity nên loại; dùng bản `scaling02` cho cả hai nhánh làm bằng chứng kỹ thuật có giới hạn.

SHA256 raw Windows đọc-only ở lần audit này: `ciciomt24_sliding_window.py` = `555F51D5C70B4AB6C3677928CCFEA33B4FBEC9CA96AF5FAA574151730EECC40A`; năm CTU-13 lần lượt `2_labeling.py` = `EA8C3468609A99C69E9F2B5AE8B86B7B7B9B55F5C00147884B0AE4D3808D2305`, `3_sliding_windown_and_aggregate_features.py` = `9DF4BFD2BE6AD72DAA943C51265352CFDB92740B1C8429B03C098A4D0662562A`, `7_senario1_svm.py` = `E44BEF8B6A6F35D4160C06DB30691510601DDD1FD31BDB12F681E8831DFA3F0F`, `[[CTU13]8_experiment2_svm.py` = `AADF2D829DD91C67620124AB033E78636C7A9D63C0CCAE73BE799916F88B6664`, `[[CTU13]9_ex2_rf.py` = `6DB8448FC1FD5CD49416B7CE37AC5DA6E71E6DC1554557589D73F73C936E8CB4`. Hash các file Stage7.3A mới phải tính lại sau commit; không dùng hash tạm trước khi chỉnh sửa làm identity chính thức.

Trước Stage 7.3B cần ghi commit sạch, SHA raw và LF-normalized của source huấn luyện/window/adapter/checkpoint/guard/notebook, hash năm CTU-13 script, `split_manifest.csv`, `validation.json`, `window_run_summary.json`, dataset fingerprint và scope. `validate_dataset` fingerprint manifest+summary+322 đường dẫn/kích thước CSV, không full-content hash CSV; không được đồng nhất với Stage 4.2 full readback. Thay source sau test/phê duyệt làm mất hiệu lực bằng chứng.

Notebook có Cell 4A legacy nhúng support files, có thể ghi đè checkout. Cell này nay fail-closed khi thấy module Stage 7.3A; người dùng phải bỏ qua Cell 4A và không dùng Run all. Cell 5 chấp nhận checkout Git mới. Nếu Cell 5 thực sự sửa ROOT hoặc phiên bản thư viện, kiểm tra lại Git/hash trước bất cứ quyết định nào.

## Scientific method audit

| Bước | Baseline SVM | MI-RFE + SVM | Mã hiện tại |
| --- | --- | --- | --- |
| X | 380 feature hành vi | 380→MI Top100 | `prepare_xy` whitelist, metadata/provenance không vào X, NaN/±Inf→0. |
| Chọn feature | Không MI/RFE | MI/scaler trên Train; Top100 scaler fit Train, transform Val | role guard chặn Test trong selection. |
| RFE | Không áp dụng | RF 100 cây, depth15, balanced, n_jobs=-1, seed42, step1, target1 | 100 forest fits; partial RFE probe không thay thế. |
| k | Không áp dụng | k=1..100, Validation Accuracy; hòa chọn k nhỏ | `search_k` duyệt tăng và chỉ cập nhật khi `>`. |
| SVC | RBF, C10, gamma scale, probability=True, class_weight=None | RBF, C10, gamma scale, probability=False, class_weight=balanced | Cả hai shrinking=True, cache_size=10000 MB, max_iter=20000, seed42; tham số tách biệt. |
| Cuối | scaler/SVC fit Train+Val trên 380 feature | khóa ranking/k rồi scaler/SVC fit Train+Val trên feature khóa | Không fit lại MI/RFE; official runner đọc Test sau final refit, đánh giá một lần. |

Đối chiếu `CODE_ctu13/7_senario1_svm.py` và `CODE_ctu13/[[CTU13]8_experiment2_svm.py`; `source_audit` kiểm tra tham số và hash nguồn. Khác biệt đã chủ ý: CICIoMT2024/C+ và nhãn `SCENARIO_PROXY`; sửa mismatch `feature_names`/numeric X; Test được nạp muộn hơn source gốc; streaming/memmap để phù hợp RAM. Đây là thay đổi dữ liệu/điều phối, không thay SVM/MI/RFE. Helper `refit_final` nhận Test sớm nhưng official `run_official` không dùng helper đó. Nếu Colab source có thay đổi khoa học khác: dừng, không sửa để ép PASS.

Nhãn proxy rất mất cân bằng; capture/session independence chưa được chứng minh. Accuracy không đủ: phải báo precision/recall/F1, macro-F1, balanced accuracy, confusion matrix và recall từng lớp. Không dùng Test để chọn đặc trưng, k hoặc cấu hình.

## Resource gate và recovery

`experiment_launcher.preflight` kiểm tra source gốc, unit evidence, manifest/schema, run trùng, RAM/disk và probe; SVM luôn `svm_full_resource_gate_reviewed=False`, nên `run_one` chặn. `PASS_TECHNICAL_ONLY` không mở full. Module `colab/svm_official_readiness.py` mới kiểm tra source path, Git clean, identity ràng buộc experiment/config/run_name/commit/source+dataset hash và approval về time/RSS, nhưng **luôn** trả `BLOCKED_STAGE73A` và không huấn luyện. Chưa có approval thực tế hoặc Stage7.3B authorization/worker được kiểm chứng.

**Giới hạn guard hiện tại:** người dùng có thể gọi trực tiếp CLI `ciciomt24_training.py train --confirm-full-training`, bỏ qua Colab `run_one`/readiness gate. Không dùng CLI đó cho SVM ở Stage7.3A. Trước Stage7.3B cần đóng đường bypass này hoặc có cơ chế full-run authorization duy nhất cho mọi entry point, rồi test lại; chưa thể tuyên bố mọi cách khởi chạy đều bị khóa bằng code.

`official_memory_preflight` là admission scenario 85% RAM available; 10 GB kernel cache là *cap*, không phải cấp phát cố định; forest/support-vector buffers còn bất định. Full SVM chưa có worker isolation hoặc timeout hard cap; RSS polling chỉ quan sát, không chứng minh chống OOM. Nếu thiếu bất cứ gate nào, giữ BLOCKED.

`CheckpointSink` lưu mốc và SHA256 trên Drive; `resume_supported=False`. Không thể resume giữa SVC.fit, RFE hoặc k-search. `.incomplete` không phải kết quả. `publish_completed_checkpoint` chỉ tái publish khi `run_validated` và log có đúng một Test start/completed sau feature freeze/final refit; không fit hay đọc Test lại. Nếu lỗi trước `run_validated`, không tự resume hoặc đánh giá Test lần hai. Drive FUSE atomicity chưa chứng minh; phải kiểm manifest bản sao. Không upload memmap nhiều GB.

Giới hạn quan trọng: nếu người dùng **tự** chạy lại dưới tên run mới sau một lần Test đã thực sự được đánh giá nhưng trước `run_validated`, code không thể chứng minh Test toàn dự án chỉ được xem một lần. Cần kiểm tra `events.jsonl`/`.incomplete` và quyết định nghiên cứu thủ công; tuyệt đối không tự retry để có điểm Test đẹp hơn.

Artifact hiện có `result.json`, preflight, event/progress, selection_frozen, k_search_results, MI/RFE metadata, model/scaler joblib, environment, artifact_manifest. Metrics trong `result.json` và `classification_report` bao gồm recall hai lớp, fit warnings/support vectors, elapsed/peak sampled, Test call count, proxy/capture flags. Còn thiếu/phân tán: commit Git khóa, hyperparameters/scaler metadata đầy đủ dưới JSON thống nhất, RSS/timeout budget duyệt, stage runtime theo bảng, clean-checkout proof và full CSV-content hashes. Cần hoàn thiện trong Stage7.3B trước official run, không sửa RF results cũ.

## Tests

Đã viết nhưng **chưa chạy** `colab/test_svm_official_readiness.py`: identity mismatch, CRLF/LF, portable path, approval/probe không unlock, RAM FAIL, hai bộ SVM params, Train-only, tie nhỏ nhất, final refit→Test và checkpoint incomplete không đánh giá lại. Test Stage5/6/7.1/7.2 sẵn có kiểm schema, run trùng, timeout/RSS của **probe** và orchestration mock. Chưa có test worker isolation full SVM vì worker này chưa tồn tại; đây là blocker. Lệnh tự chạy ở `STAGE7_3A_COLAB_GUIDE.md`.
