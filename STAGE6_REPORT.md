# Stage 6 — Triển khai cục bộ notebook Google Colab Pro

> Bản này ghi trạng thái triển khai ban đầu. Lỗi Cell 5/6 và hướng dẫn dataset folder đã được cập nhật trong `STAGE6_DATASET_FIX_REPORT.md`; ưu tiên báo cáo fix khi chạy lại notebook.

Ngày 20/10/2026. **Trạng thái: ĐÃ TRIỂN KHAI VÀ KIỂM THỬ CỤC BỘ; CHƯA XÁC MINH COLAB/DRIVE; KHÔNG HUẤN LUYỆN CHÍNH THỨC.**

## 1. Căn cứ và repository

Đã đọc lại STAGE51_REPORT.md, STAGE52_REPORT.md, STAGE53_REPORT.md, STAGE54_PREFLIGHT_REPORT.md, STAGE55_MEMORY_OPTIMIZATION_REPORT.md, code `ciciomt24_training.py`/`ciciomt24_sliding_window.py`, manifest C+, và kế hoạch Stage 6 do người dùng cung cấp. Hai run RF FS16 đã hoàn tất: FW512/FS16 k=8, FW256/FS16 k=18; FS8 chưa chạy chính thức. Stage5.4 guard FW512/FS8 là **25.762.976.100 byte RAM available** và Stage5.5 chưa đủ bằng chứng để nới guard; adapter giữ nguyên guard đó.

Repository cục bộ `D:\COMPACT-MI-RFE` có Git origin `https://github.com/kasum00/RFforNetwork.git`. `git ls-tree -r HEAD --name-only` tại thời điểm triển khai trả về **chỉ `README.md`**. Code nghiên cứu hiện diện trên ổ D nhưng chưa nằm trong commit cục bộ. Không truy cập GitHub web, không commit/push; không suy đoán nội dung remote. Colab clone sẽ dừng nếu GitHub chưa được người dùng tự cập nhật source nhỏ theo `COLAB_SETUP.md`.

## 2. File tạo mới

| Đường dẫn Windows đầy đủ | Vai trò |
|---|---|
| `D:\COMPACT-MI-RFE\notebooks\COMPACT_MI_RFE_ColabPro.ipynb` | 12 cell runtime → Drive → dataset → clone → dependencies/tests → chuẩn bị/validate → lựa chọn → preflight → full run có xác nhận → lưu/kiểm tra → summary |
| `D:\COMPACT-MI-RFE\colab\__init__.py` | Package cục bộ |
| `D:\COMPACT-MI-RFE\colab\setup_colab.py` | Runtime, dependency và repo inventory |
| `D:\COMPACT-MI-RFE\colab\drive_manager.py` | Kiểm tra folder ID chia sẻ qua quyền đọc do người dùng cấp trong Colab, stream ZIP, kiểm tra quyền ghi output MyDrive |
| `D:\COMPACT-MI-RFE\colab\dataset_validator.py` | ZIP guard/CRC, extract staging, metadata C+, 322-file inventory/schema và fingerprint phạm vi rõ ràng |
| `D:\COMPACT-MI-RFE\colab\checkpoint_manager.py` | Lưu artifact hoàn thành theo stage/từng k, kiểm SHA256, xuất kết quả hoàn chỉnh; chỉ promote checkpoint đã `run_validated` |
| `D:\COMPACT-MI-RFE\colab\experiment_launcher.py` | Portable one-config wrapper, resource admission, SVM probe kỹ thuật, gọi runner Stage 5 và checkpoint trước Test |
| `D:\COMPACT-MI-RFE\colab\results_summary.py` | Đọc `result.json` hoàn chỉnh; không dự đoán lại Test |
| `D:\COMPACT-MI-RFE\colab\test_stage6_colab.py` | Unit/smoke test cục bộ, gồm mock đường final-refit → Test |
| `D:\COMPACT-MI-RFE\configs\colab_experiments.json` | Danh mục thí nghiệm, tham số và mặc định dry run |
| `D:\COMPACT-MI-RFE\configs\cplus_manifest\split_manifest.csv` | Bản sao metadata C+ nhỏ, 284 record; Source_CSV Windows ở đây **chỉ là provenance**, không phải đường dẫn file mở trong Colab |
| `D:\COMPACT-MI-RFE\configs\cplus_manifest\validation.json` | Bản sao validation C+ nhỏ |
| `D:\COMPACT-MI-RFE\requirements-colab.txt` | Pin phiên bản khoa học Stage 5; Python 3.10–3.12 |
| `D:\COMPACT-MI-RFE\COLAB_SETUP.md` | Hướng dẫn người dùng và điều kiện dừng |
| `D:\COMPACT-MI-RFE\.gitignore` | Chặn dataset/runs/cache lớn khỏi Git |
| `D:\COMPACT-MI-RFE\STAGE6_REPORT.md` | Báo cáo này |

Không chỉnh sửa file source Stage 5 hoặc năm script CTU-13, không sửa CSV/window/split, không ghi đè run 5.2/5.3. Bản sao metadata C+ có khác biệt line ending/blank line cuối ở mức byte, nhưng test so sánh toàn bộ CSV record và nội dung JSON **bằng nhau về ngữ nghĩa** với nguồn; hash fingerprint Colab sẽ dựa trên bản metadata portable cụ thể.

## 3. Luồng khoa học và checkpoint

Adapter **không sao chép/đổi** các hàm MI, scaler, RF-RFE, SVM, RF, luật chọn k hoặc evaluator của `huyen_code/ciciomt24_training.py`. Nó tạm ràng buộc `WindowStore` sang manifest Linux và bọc `RunProgress` để đồng bộ mốc hoàn thành lên Drive. Khi `fit_final_classifier` của source trả về model đã fit trên Train+Validation, adapter serialize đúng model/scaler đó rồi hoàn tất checkpoint `final_refit_completed` **trước** `store.load_full(...,"test",...)` của runner source. Test được đánh giá một lần theo source. Mọi hàm nguồn được khôi phục sau run dù thành công/lỗi.

Chỉ ghi checkpoint sau MI scores/Top100, RFE ranking, từng `k_completed`, `selection_frozen`, `final_refit_completed`, và `run_validated`. Mỗi record có experiment/config/seed, fingerprint dataset, code hash, version thư viện, stage, completed k, timestamp. File tạm được copy/kiểm SHA256 rồi rename trên Drive mount. Tính nguyên tử của Drive FUSE chưa kiểm định trên Colab. **Không resume fit từ checkpoint dở**; chỉ kết quả đã `run_validated` đủ artifact mới được `publish_completed_checkpoint` mà không gọi fit/Test lại. Matrix memmap không được sao chép lên Drive (quá lớn); model/log/metrics và môi trường được giữ.

RF FS8 dùng đầy đủ Train/Validation/Test C+ và mọi k=1..100, không sampling; MI Top100 Train-only; RF-RFE 100 cây, depth10, split2, sqrt, balanced, seed42, step1; RF classifier 300 cây, depth20, split3, sqrt, balanced, seed42. SVM baseline và MI-RFE-SVM giữ bộ tham số riêng từ source; SVM full chỉ được xét sau bounded technical probe được người dùng bật và RAM admission PASS. Hai RF FS16 đã hoàn tất bị chặn không rerun. Không dùng Test để chọn feature/k hoặc để sửa guard.

## 4. Kết quả kiểm thử cục bộ

| Kiểm tra | Kết quả |
|---|---|
| Notebook JSON nbformat4, đúng 12 code cell, tất cả cell parse Python AST | PASS |
| Default `dry_run`; Cell 10 chỉ full khi `MODE=full_train`, preflight PASS, confirmation chính xác | PASS kiểm tra code/test; chưa chạy trên Colab |
| `python -m compileall -q D:\COMPACT-MI-RFE\colab` | PASS |
| Unit test Stage 6 (`python -m unittest -q colab.test_stage6_colab`) | **11/11 PASS** |
| Unit test nguồn (`python -m unittest -q test_ciciomt24_training test_ciciomt24_split test_ciciomt24_sliding_window`) | **97/97 PASS** |
| Manifest C+ portable và bản gốc: toàn bộ CSV rows/JSON semantic equality | PASS |
| Kiểm định metadata/schema dataset cục bộ `WINDOWS_CPLUS` | PASS: 322 CSV, 380 feature, 394 cột, đủ 12 ô expected, Test feature rows đọc = 0 |
| ZIP traversal/duplicate path unit fixture | PASS từ chối ZIP giả độc hại; **chưa có ZIP Drive thực để kiểm CRC** |
| Stage5.4 guard FW512/FS8 tính lại | PASS đúng 25.762.976.100 byte |
| Preflight dry no `timed_fit`/MI fit (mock cấm), Test rows = 0 | PASS; status FAIL cục bộ vì Drive chưa mount |
| Mock orchestration: model/scaler checkpoint xong trước Test event, source symbols được khôi phục, completed checkpoint có thể promote | PASS fixture, **không fit mô hình thật** |
| SHA256 runner Stage 5 hiện tại | `5E0FD78E8F306F65B38C5AF811FE8615C3AD6C278357DEB3C190963D608CAD5C`, khớp báo cáo 5.3–5.5 |
| Stage5.2/5.3 `result.json`, `model.joblib`, `final_scaler.joblib` đối chiếu SHA256 đã lưu trong audit Stage5.3 | 6/6 PASS; không sửa run cũ |

Local Python 3.10.0, NumPy 1.26.4, pandas 2.3.3, SciPy 1.15.3, scikit-learn 1.7.2, joblib 1.5.3 đã import được. Đây **không phải** kết quả runtime Colab.

## 5. Chưa xác minh / giới hạn

- Không mở/điều khiển Chrome, Colab hay Google Drive theo yêu cầu mới. Chưa kiểm tra quyền Drive folder, tên ZIP, cấu trúc ZIP thật, mount/FUSE, runtime Python/RAM/disk/GPU hoặc package conflicts trên Colab.
- Không chạy full model, không fit MI/RFE/RF/SVM chính thức và không evaluate Test. Adapter chỉ được mock-regression và unit test cục bộ; tính tương đương số học full trên Linux/Colab chưa chứng minh. Nếu package/Python khác Stage 5 hoặc unit tests FAIL thì notebook dừng.
- Dataset validator preflight tái sử dụng full readback Stage 4.2, so metadata/sizes/header hiện tại; không hash lại toàn bộ 21GB CSV window. ZIP CRC được kiểm khi người dùng chạy Cell 6 thật. Không tuyên bố fingerprint metadata bằng hash nội dung toàn dataset.
- Chưa chứng minh capture/session độc lập. Window cùng split chồng lấn; nhãn SCENARIO_PROXY, Benign hiếm, một số scenario không có window Validation/Test. Bốn điều kiện: không chung source row, không exact duplicate, không activity crossing, capture/session độc lập **không tương đương**.
- Nếu Colab reset giữa một fit hoặc trước `run_validated`, checkpoint chỉ là bằng chứng tiến độ, không thể resume và không phải kết quả khoa học hợp lệ. Drive mount atomic rename/throughput cần kiểm chứng thực tế.
- Kết quả FS16 cũ trên Drive/GitHub chưa kiểm tra. Nếu thiếu, Cell 12 báo `PREVIOUS_ARTIFACTS_MISSING`; không chạy lại hay tạo giả.

## 6. Bước thủ công của người dùng

1. Tự commit/push **chỉ các file source nhỏ** liệt kê trong `COLAB_SETUP.md` lên GitHub, xem danh sách staged trước; không push dataset/run cũ.
2. Mở notebook từ GitHub bằng URL trong `COLAB_SETUP.md` hoặc upload file `.ipynb` cục bộ, kết nối runtime Colab CPU đủ RAM/disk thực tế.
3. Tự cấp quyền mount Drive và, nếu cần, quyền Drive API đọc folder chia sẻ. Chỉ định ZIP ID nếu nhiều ZIP hoặc đường dẫn shortcut/extracted folder.
4. Chạy Cell 1–9 mặc định dry-run; chỉ khi mọi guard PASS mới chủ động chọn một experiment, nhập xác nhận và chạy Cell 10. Không sử dụng Test hoặc Stage5 Test metrics để chọn lại tham số.

**Dừng Stage 6 tại đây.** Không có tuyên bố notebook đã chạy được trên Colab thật trước khi người dùng xác minh các bước trên.
