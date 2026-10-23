# WINDOWS LOCAL MI-RFE SVM — báo cáo triển khai

Ngày 23/10/2026 (Asia/Saigon). **IMPLEMENTED / STATIC REVIEW ONLY / TESTS NOT EXECUTED / FULL RESUME NOT VERIFIED.**

Codex chỉ đọc source/cấu hình, chỉnh source bằng apply_patch, viết tests và tài liệu. Không chạy Python, tests, scripts dataset, benchmark, pip, Colab/browser, preflight, MI/RFE/SVM; không copy dataset hoặc checkpoint thật; không commit/push. Việc dùng công cụ apply_patch để ghi source không phải chạy pipeline nghiên cứu.

## Phase 1 — inspect và thiết kế

Đã đọc các source training/sliding-window, checkpoint store, engine/auth/supervisor resumable, auth/supervisor Stage7.3B, experiment_launcher/setup/dataset validator, notebook và tests liên quan. Source thực tế nằm D:\COMPACT-MI-RFE; đường dẫn LAB_LEE trong prompt không tồn tại trên máy đang chỉnh. Dataset cũ WINDOWS_CPLUS và CSV_SPLIT_CPLUS có mặt, chỉ kiểm thư mục; không audit hay quét dữ liệu.

Các ràng buộc cũ:

- authorize_run/preflight Colab yêu cầu Drive mount và identity môi trường gồm Linux/platform/CPU. Approval Colab không hợp lệ mặc định trên Windows.
- binding checkpoint gồm toàn bộ identity và bốn module hashes. Thay hash/manifest để lách kiểm tra bị cấm.
- Fingerprint Colab dùng đường dẫn tương đối với dấu / và size, không full CSV hash.
- Engine canonical đã có Train-only MI/Top100 scaling, sklearn RF-RFE và k search, final refit + Test claim.
- Mỗi run cần MI/RFE đã hoàn thành. Cache memmap mới trên VM có thể tái tạo bằng loader, không fit lại MI/RFE.
- CLI/scientific file gốc và các notebook cũ giữ giao thức mặc định của chúng. Windows dùng entry riêng, không giả tạo seal Colab.

## Phase 2–4 — hạ tầng, resume, authorization

### Danh tính và migration

Origin run bị khóa: experiment mi_rfe_svm, FW_512_FS_16, mi_rfe_svm_FW512_FS16_resumable01, origin Git 7718c74c44070600ea876a3a1c678632ccb184dd. Đây là identity được hỗ trợ, không phải k cuối đã xác minh.

Scanner không tạo run, không load pickle, không đọc dataset. Xác minh toàn bộ completed artifact SHA/size, parent hashes, MI feature order/indices, full RFE ranking, contiguous k1..N và prefix/full row counts; k đầu chưa hoàn tất là N+1. Không có completed k → BLOCK, không từ k1. Corrupt hoặc missing manifest ở tên completed → BLOCK; .incomplete-* giữ nguyên và không tính complete.

Origin manifest/binding byte không sửa. Local execution envelope lưu riêng commit triển khai, module hashes, physical paths, Windows environment, approval/budgets. Source khoa học phải trùng normalized bytes và Git blob origin; CTU-13 giữ raw-byte hashes. Với engine dùng chung, verifier so sánh AST toàn bộ thuật toán sau đúng ba adapter: manifest location, progress factory, deadline factory. Kiểm tra tên hàm duy nhất/signature/decorator; toàn bộ helper/import/class cũ vẫn phải khớp. Không vá hash để ép tương thích.

Fingerprint được dựng lại theo đúng logical POSIX inventory gốc, dù file vật lý ở ổ Windows khác. Bắt buộc manifest SSD checksum export tin cậy từ Colab. Preflight kiểm toàn bộ byte Train/Val; worker kiểm lại trước sử dụng. Test chỉ stat trước final; sau final-model commit và single Test claim mới hash nội dung và load Test. Checksum manifest không phải chứng thư chống nguồn giả mạo; chỉ dùng export của chính người dùng.

Import optional: destination vắng mặt và disjoint, copy đủ file/hidden/empty directories, so SHA nguồn/copy trước promote; không overwrite hay đổi nguồn. Người dùng đã cài folder trên VM thì bỏ bước import.

### Supervisor và quyền chạy

Local approval mới dùng schema mi_rfe_svm.windows_local.authorization.v1, không kế thừa quyền Colab. Ràng buộc run/source/snapshot checkpoint/dataset/profile/physical paths/session/budget/evidence; yêu cầu đúng câu xác nhận. Source đổi cần commit/test evidence/approval mới.

OS byte lock giữ suốt phiên, active_session token, PID/create time, worker PID/create time, hostname, heartbeat và manual-confirmed takeover. Lock cũ chỉ được archive theo giao thức sau xác nhận/stale heartbeat; không tự xóa lock cũ. Local worker cũ còn sống thì chặn takeover.

Windows Job Object có JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE; worker đợi handoff record trước authorize/fit. VM không cho assign Job → BLOCK. Job handle đóng hoặc supervisor bị kill thì tree được Windows kết thúc; không để training không giám sát. psutil giám sát elapsed/RSS/RAM/disk và xác thực PID/create time, không kill theo tên process, không dùng POSIX signals.

Resource budget local cố ý null ở các giá trị RAM/disk cần người dùng duyệt. Không tự giữ 42 GiB hoặc tự hạ giới hạn. Giữ 10 giờ mặc định, margin 300s; session dài hơn phải xác nhận riêng. Không auto retry/new session.

Log UTF-8 unbuffered inherited terminal, event ngay lúc fit/checkpoint, durable heartbeat 60s, console heartbeat 300s, RSS/CPU/fit time/accuracy/F1/checkpoint SHA. request-stop chỉ ngăn work unit tiếp theo; SVC.fit dở không có optimizer checkpoint.

## Scientific invariants giữ nguyên

- Dùng toàn bộ Train 673559 / Val 224373 / Test 224287; final fit 897932.
- X đúng 380 cột canonical, không metadata; Inf/NaN→0, float32 theo scientific core.
- Nhãn SCENARIO_PROXY; Train/Val Benign 1575/505; không đổi labeling/split/FW/FS.
- MI Train-only Top100, giữ feature order và saved scalers; không refit MI/RFE sau khi resume.
- SVM-RFE riêng: RF100, depth15, balanced, n_jobs=-1, seed42, RFE target1/step1. Không dùng depth10 của nhánh RF.
- SVC C10, gamma scale, RBF, probability=False, cache_size10000, shrinking=True, max_iter20000, balanced, seed42.
- k1..100, prefix ranking cố định, Validation Accuracy strict greater, tie giữ k nhỏ.
- Sau selection mới final scaler/SVC Train+Val; Test chỉ một lần sau final, claim dở không auto retry.
- Không sửa CODE_ctu13 hoặc scientific functions trong ciciomt24_training.py/ciciomt24_sliding_window.py.

## Phase 5 — tests được viết, NOT EXECUTED

Hai suite local có 45 test methods (đếm bằng đọc source, KHÔNG thực thi). Các nhóm chính:

1. Copy/import không overwrite, giữ bytes/hidden file/empty directory.
2. Manifest/binding/checksum, source-origin commit, feature order/scientific contract.
3. Logical POSIX/Windows inventory tạo cùng fingerprint, metadata khác bị khác fingerprint.
4. MI/RFE reuse, full counts, k đầu chưa xong, partial k giữ nguyên, corrupt/gap bị chặn.
5. Final model/selection reuse và single-Test claim block.
6. Budget null/thiếu RAM/disk, pinned Python/OS/packages.
7. Sealed local context, không dùng legacy approval, approval bị đổi.
8. Byte lock, stale/live/orphan process takeover.
9. PID reuse không kill, scope worker tree, resource stop không retry.
10. Job handle assignment/close và assignment failure block.
11. Deadline/manual safe stop, heartbeat persistence và console throttling.
12. Shared-engine scientific calls và final→claim→Test ordering.
13. Worker rehash Train/Val, Job handoff trước authorize; Test hashing yêu cầu claim/final.
14. AST adapter verification, duplicate definition và decorator rejection.

User-invoked tests command sẽ chạy hai suite local cùng năm suite legacy/resumable được chọn, dự kiến 92 methods nếu inventory không thay đổi. Không suy ra 92 PASS. Kết quả 47/47 trước đây thuộc source trước refactor, không tái sử dụng như evidence source mới.

Đã thực hiện static review chính và review source độc lập; đã chỉnh các điểm Job custody/handoff publish nguyên tử, AST uniqueness/signature, Train/Val rehash, line ending protection, PID identity và ghi nhận lỗi cleanup/AccessDenied thay vì để mất báo cáo gốc. Chưa parse/import bằng Python, chưa chứng minh test hoặc runtime chạy được.

Kiểm tra không thực thi Python đã làm: đọc/parse hai JSON template bằng PowerShell thành công; đếm 45 test local + 47 test legacy bằng tìm chuỗi khai báo; đọc SHA256 năm file CTU-13 đều khớp snapshot được bảo tồn trong source. Rà soát độc lập cuối không thấy blocker runtime/fixture rõ ràng còn lại, nhưng không thay thế việc chạy tests hoặc kiểm checkpoint thật.

## Danh sách file

Tạo mới:

1. D:\COMPACT-MI-RFE\local\__init__.py
2. D:\COMPACT-MI-RFE\local\paths.py
3. D:\COMPACT-MI-RFE\local\checkpoint_migration.py
4. D:\COMPACT-MI-RFE\local\windows_preflight.py
5. D:\COMPACT-MI-RFE\local\local_authorization.py
6. D:\COMPACT-MI-RFE\local\local_supervisor.py
7. D:\COMPACT-MI-RFE\local\windows_job.py
8. D:\COMPACT-MI-RFE\local\progress_console.py
9. D:\COMPACT-MI-RFE\local\run_mi_rfe_svm.py
10. D:\COMPACT-MI-RFE\local\requirements-windows.txt
11. D:\COMPACT-MI-RFE\local\windows_paths.example.json
12. D:\COMPACT-MI-RFE\local\local_budget.BLOCKED.example.json
13. D:\COMPACT-MI-RFE\local\test_windows_local.py
14. D:\COMPACT-MI-RFE\local\test_local_supervisor.py
15. D:\COMPACT-MI-RFE\local\README_WINDOWS_LOCAL.md
16. D:\COMPACT-MI-RFE\local\WINDOWS_LOCAL_REPORT.md
17. D:\COMPACT-MI-RFE\.gitattributes

Chỉnh sửa:

- D:\COMPACT-MI-RFE\colab\svm_resumable_training.py: public Colab wrapper giữ validation, thêm Windows entry có local seal; tách shared body, ba I/O factory/location adapters. Không thay fit/predict/selection statements.
- D:\COMPACT-MI-RFE\colab\test_svm_resumable_training.py: test soi late Test chuyển sang shared body, không bỏ assertion.
- D:\COMPACT-MI-RFE\.gitignore: bỏ qua checkpoint/control/cache/approval/probe/environment local.
- D:\COMPACT-MI-RFE\README.md: dẫn đến hướng dẫn Windows mới.

.gitattributes bảo vệ byte của CTU-13 và manifest tracked. Không tự renormalize hoặc checkout lại các file đó.

## Chưa được xác minh / nguyên nhân BLOCK có thể gặp

- Chưa có quyền/bằng chứng đọc blob Git origin trong lượt static review này; verifier trên clone VM sẽ đối chiếu lại. Không tự assert source-origin equivalence PASS.
- Không có checkpoint thật của VM trong lượt này: chưa biết kN, source/binding/schema/profile có khớp, artifact có thiếu/corrupt không.
- Session01 preflight/authorization và approval/probe/checksum manifest export có thể thiếu. Thiếu thì báo nguyên nhân cụ thể, không đổi identity.
- Chưa đo RAM/SSD/CPU VM, chưa xét budget của người dùng, chưa xác minh wheel hoặc Job/nested-job support/NTFS fsync/locking.
- Linux→Windows không cam kết bitwise numerical equivalence dù cùng package versions; approval yêu cầu chấp thuận và phải ghi trong báo cáo thực nghiệm.
- Nếu dữ liệu có thể bị người khác sửa trong lúc chạy, hash-before-use không thay thế immutable filesystem snapshot. Giữ dataset read-only; không chạy hai bản run ở hai máy.
- Worker resource sampling không bảo đảm mọi OOM; final fit hoặc một k SVC có thể vượt 10 giờ.
- Pickle/joblib chỉ dùng từ nguồn tin cậy. SHA không chứng minh nguồn độc hại an toàn.
- Test claim chưa final_test chặn tự chạy lại, kể cả Test chưa predict do checksum fail.
- SCENARIO_PROXY, class imbalance và capture/session independence NOT_ESTABLISHED vẫn giữ nguyên.

Hướng dẫn cấu trúc/thao tác thủ công/lệnh PowerShell đầy đủ ở D:\COMPACT-MI-RFE\local\README_WINDOWS_LOCAL.md. Dừng sau triển khai; chờ người dùng tự chạy tests/inspect/preflight và gửi output.

