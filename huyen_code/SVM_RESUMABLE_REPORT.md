# Báo cáo triển khai MI-RFE SVM nhiều phiên

Ngày 23/10/2026. **LOCAL WINDOWS UNIT TESTS PASS — 47/47 (USER-RUN); COLAB/FULL TRAINING NOT VERIFIED.** Log người dùng chạy lại sau bản sửa ghi `Ran 47 tests in 13.286s`, `OK`, không lỗi hoặc skip. Log trước bản sửa có 44 tests, 34 OK và 10 ERROR. Codex chỉ đọc/sửa source, viết tests và hướng dẫn; không tự chạy Python, scripts, tests, benchmark, dataset audit, notebook hoặc training; chưa commit/push; không tạo approval hiệu lực.

## Đối chiếu CTU-13

Xem `SVM_RESUMABLE_AUDIT.md`: nhánh SVM giữ RFE RF100/depth15/balanced/n_jobs=-1/seed42/step1/target1, không đồng nhất với RF depth10 của script thứ hai. Giữ MI Top100 Train-only; scaler Train-only ở selection; full k1..100, Validation Accuracy và tie k nhỏ; final scaler/SVC Train+Validation; Test sau final fit. Cả 380 behavioral features, Inf/NaN→0, full counts và proxy/capture flags được giữ.

Khác biệt đã có từ pipeline CICIoMT2024 được giữ: sửa mismatch feature_names, metadata whitelist, Test nạp muộn, bổ sung metric/warning. Không tự đổi thuật toán để tiết kiệm RAM.

## Các giai đoạn mới

| Giai đoạn | Cách thực hiện | Khả năng resume |
|---|---|---|
| A — Data | canonical WindowStore.load_full cho Train/Val; local memmap với signature/checksum | dùng lại local cache đã xác minh; runtime mới đọc lại CSV, không tải memmap Drive |
| B — MI | gọi select_mi, scale_top100; lưu scores/feature order/MI scaler/Top100 scaler | reuse bundle MI hoàn tất |
| C — RFE | gọi rank_rfe(...,'svm',target=1), đúng sklearn RFE | reuse full ranking hoàn tất; không resume từng bước bên trong RFE |
| D — k | gọi search_k(...,k_values=(k,)) trên prefix ranking, save ngay sau metric validation | reuse k1..N, bắt đầu kN+1; fit dở phải lặp lại nếu phiên mới được duyệt |
| E — Final | gọi fit_final_classifier trên toàn Train+Val với feature khóa, lưu model/scaler trước Test | reuse final model hoàn tất; không resume optimizer SVC |
| F — Test | exclusive durable claim trước loader Test, evaluate_final một lần, commit result | không tự retry claim chưa có result; chỉ publication recovery khi final_test hoàn tất |

Mỗi bundle ràng buộc SHA identity toàn run, feature order, scientific contract, seed, split hashes và SHA của parent bundle. Ghi vào directory tạm, commit manifest+rename, kiểm lại toàn bộ SHA khi đọc. Không dùng tồn tại file làm bằng chứng thành công.

## Authorization và supervisor

Giữ `authorize_run`, sealed context, base preflight và legacy guards nguyên vẹn. Mode mới yêu cầu thêm `execution_mode=stage73b.mi_rfe_svm.resume-policy.v1`, policy hoàn tất-stage, module hashes và session ID/budget riêng. Old approval/probe PASS không tự mở mode này. Same run resume phải khớp commit/source/data/features/environment scientific identity; source đổi phải tạo approval mới và không load checkpoint khác binding.

Worker được xác minh parent PID/create time và token session lock. Supervisor giám sát đúng worker tree, elapsed/RSS/RAM/disk, dừng con trong phạm vi đã thấy; không retry, không giảm mẫu. Dùng lại hàm guard/terminate cũ, không sửa supervisor RF/Baseline. Khóa concurrent session; tiếp quản lock của Colab đã chết cần xác nhận termination trong approval mới và heartbeat cũ hết khoảng bảo vệ.

Budget mặc định đề xuất 36.000 giây, không bắt đầu công việc độc lập mới ở 300 giây cuối; heartbeat 60 giây. Hard elapsed có thể hủy fit dở. RSS polling không phải hard memory guarantee và không có cam kết full fit vừa phiên Colab.

## Logging và kết quả

Progress/event/heartbeat liên tục trên Drive với UTC fit start/finish, stage/k, completed count, session elapsed, total observed worker time, stage elapsed, PID/status, process-tree RSS/available RAM/CPU, checkpoint gần nhất và lỗi/stop reason. Có output flush trực tiếp ở Colab, không chỉ một dòng STARTING.

Result giữ metrics Accuracy/Precision/Recall/F1/Macro-F1/Balanced Accuracy/confusion matrix/report từng lớp, fit convergence/warnings/support vectors, runtime từng stage và full counts. Final model/scaler/selection/result có artifact manifest SHA; không upload cache matrices.

## Unit tests — người dùng chạy trên Windows: 47/47 PASS

Ba suite mới ban đầu có 32 test methods; sau bản sửa Windows có thêm 3 test hồi quy (35 methods). Cộng 12 test Stage 7.3B, log chạy lại do người dùng gửi xác nhận tổng 47 tests đều OK trong 13.286 giây. Ba test hồi quy fsync cũng PASS. Đây là kết quả unit test với fixture nhỏ/mocks, không phải huấn luyện hoặc đánh giá full dataset.

- Checkpoint save/load MI/scaler và RFE dependency; corruption/identity/config/feature-order rejection.
- Immutable k, prefix, contiguous k history; interrupted writes không false-complete.
- Resume sau k12 gọi duy nhất k13 rồi safe pause k14, không gọi MI/RFE lại.
- Hyperparameters/bốn cấu hình, canonical metrics và smallest-k tie.
- Tiny StandardScaler reload/transform comparison; không fit SVM thật trong tests.
- Deadline margin, heartbeat persistence, late Test claim/read ordering, claimed Test blocks mọi loader/fit.
- Legacy/missing approval, session/budget/module mismatch, sealed context và Baseline/RF rejection.
- Supervisor timeout/RSS/RAM/disk/exception paths bằng mocks, worker thất bại và lock confirmation.
- Publication recovery source không có fit/Test.

Nguồn bằng chứng: `C:\Users\admin\.codex\attachments\f80fe2d4-15ee-4bd8-ac79-a83f6f677161\Pasted text.txt`. Các event `test_evaluation_completed`, `completed_result_reused`, `worker_failed` và `session_paused_at_checkpoint` xuất hiện bên trong unit test mô phỏng, không phải kết quả thí nghiệm thật. `worker_failed` do Test đã claim là nhánh chặn được test cố ý kích hoạt và kết thúc `ok`.

Chưa có bằng chứng chạy trên Colab/Drive, full-data resource admission hoặc tương đương số học end-to-end của full experiment. Log được cung cấp không tự ràng buộc commit/SHA đang checkout; evidence đúng commit/module hashes trên Colab vẫn bắt buộc theo guard hiện có.

## Danh sách file

Tạo mới:

1. `D:\COMPACT-MI-RFE\colab\svm_resumable_store.py`
2. `D:\COMPACT-MI-RFE\colab\svm_resumable_authorization.py`
3. `D:\COMPACT-MI-RFE\colab\svm_resumable_training.py`
4. `D:\COMPACT-MI-RFE\colab\svm_resumable_execution.py`
5. `D:\COMPACT-MI-RFE\colab\test_svm_resumable_store.py`
6. `D:\COMPACT-MI-RFE\colab\test_svm_resumable_training.py`
7. `D:\COMPACT-MI-RFE\colab\test_svm_resumable_execution.py`
8. `D:\COMPACT-MI-RFE\notebooks\CICIoMT24_MI_RFE_SVM_RESUMABLE.ipynb`
9. `D:\COMPACT-MI-RFE\huyen_code\SVM_RESUMABLE_AUDIT.md`
10. `D:\COMPACT-MI-RFE\huyen_code\SVM_RESUMABLE_REPORT.md`
11. `D:\COMPACT-MI-RFE\huyen_code\SVM_RESUMABLE_GUIDE.md`

Chỉnh sửa: `D:\COMPACT-MI-RFE\README.md` để dẫn đến notebook/guide mới. Không sửa năm script CTU-13, `ciciomt24_training.py`, source split/window, legacy RF/Baseline runner, datasets, kết quả hoặc approval cũ.

## Giới hạn cần báo cáo

SVC final fit, một k SVC fit hoặc toàn MI/RFE đang dở không có optimizer checkpoint; một fit quá 10 giờ vẫn có thể không hoàn thành. Ghi Drive/FUSE và lock across runtimes là best effort, không phải transaction của database/distributed lock; luôn cần SHA verification, xác nhận phiên cũ đã chết và tránh hai runtime chạy đồng thời. Kernel Colab chết có thể không kịp ghi session_result; heartbeat/checkpoint cuối là bằng chứng còn lại. Claim Test chưa result bắt buộc manual review dù final model đã lưu.

Environment identity dùng lại quy tắc Stage 7.3B (gồm profile Python/packages/platform/CPU); runtime mới khác identity sẽ bị chặn, không tự đổi contract để load checkpoint. Dataset fingerprint gốc không phải full-content CSV hash; kiểm nội dung SSD được notebook thực hiện riêng. Nhãn SCENARIO_PROXY và capture/session independence NOT_ESTABLISHED vẫn là giới hạn khoa học.

Chỉ triển khai và dừng; chưa đủ bằng chứng để gọi full experiment thành công hoặc execution gate đã được duyệt.

## Bổ sung sau review source

Đã siết kiểm tra publication để yêu cầu đầy đủ artifact và so SHA với checkpoint nguồn; verify toàn bộ MI/RFE/100 k/final chain trước Test claim; dùng chung deadline supervisor/worker để thời gian reauthorization không làm lệch safety margin; serialize thao tác claim/takeover bằng OS file lock và kiểm lại token trước archive. Bộ unit test được người dùng chạy lại đã PASS như trên; Drive/FUSE cross-runtime lock vẫn là giới hạn cần kiểm chứng thực tế, không coi là distributed lock bảo đảm.

## Sửa lỗi Windows từ log người dùng

Cả 10 ERROR trong log đều dừng tại `CheckpointStore.save()` khi gọi `os.fsync()` trên descriptor mở bằng `rb`: Windows báo `OSError: [Errno 9] Bad file descriptor`. Đây là lỗi lưu checkpoint, không phải bằng chứng lỗi thuật toán MI/RFE/SVM.

Đã thay thao tác này bằng helper `sync_file()`: mở artifact hiện có bằng `r+b` (có quyền ghi nhưng không truncate), flush rồi fsync, sau đó mới tính SHA256 và commit manifest như trước. Không bỏ qua lỗi fsync; lỗi vẫn dừng commit và giữ directory `.incomplete-*` để kiểm tra. Các phép đọc SHA256 vẫn dùng `rb`.

Thêm ba test hồi quy: kiểm tra mode/flush trước fsync, bytes và checksum không đổi, sync thất bại không tạo checkpoint COMPLETE và giữ artifact dở. Người dùng đã chạy lại, cả ba test PASS; Codex không tự thực thi. Thuật toán, hyperparameters, late Test và authorization guards không thay đổi. Module SHA thay đổi nên approval/evidence cũ không được tự tái sử dụng cho source mới.

Lệnh người dùng đã chạy lại thành công (47 tests):

```powershell
Set-Location 'D:\COMPACT-MI-RFE'
$env:PYTHONPATH = 'D:\COMPACT-MI-RFE;D:\COMPACT-MI-RFE\huyen_code'
python -m unittest -v colab.test_svm_resumable_store colab.test_svm_resumable_training colab.test_svm_resumable_execution colab.test_stage73b_authorization colab.test_stage73b_scientific_integrity
```

Trạng thái sau sửa: **WINDOWS UNIT TESTS PASS**. Có thể chuẩn bị đồng bộ source/notebook lên GitHub và chạy các bước kiểm tra Colab theo guide; không tự mở training. Chưa kiểm chứng checkpoint trên Drive/FUSE, tài nguyên full SVM, approval theo commit mới hoặc full experiment thành công.
