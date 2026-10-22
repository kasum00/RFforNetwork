# Stage 7.3B — Controlled official SVM training implementation

## Kết luận

Đã triển khai source cục bộ theo cơ chế **fail-closed**, nhưng **chưa chạy code, test, audit, probe hoặc training**. Full SVM hiện vẫn **BLOCKED**. Repository không chứa approval hợp lệ; notebook mới mặc định khóa và cell khởi chạy sẽ ném lỗi nếu chưa bật hai xác nhận cùng exact phrase.

## 1–6. Authorization và entry point

1. Thêm schema authorization riêng cho một experiment/config/run/commit/data/runtime.
2. Ràng buộc training/sliding/manifest/validation/window summary SHA, dataset fingerprint+scope và hash năm CTU-13.
3. Ràng buộc elapsed, process-tree RSS, RAM start/runtime reserve, disk và polling interval.
4. Bắt buộc exact technical probe hoặc policy thay thế được người dùng chấp nhận rõ ràng; probe không tự cấp quyền.
5. Đóng bypass: `run_official` yêu cầu sealed authorization context cho SVM; CLI SVM trực tiếp bị từ chối dù có `--confirm-full-training`.
6. RF không yêu cầu SVM authorization, giữ đường điều phối cũ.

## 7–11. Worker, resource guard và checkpoint

7. Worker nội bộ revalidate approval/identity rồi mới gọi orchestration hiện có.
8. Supervisor theo dõi đúng PID/cây con, elapsed, RSS hiện tại/peak, host available RAM, disk và stage heartbeat.
9. Khi vượt ngưỡng, chỉ dừng cây worker; không retry/resume/sampling/đổi tham số.
10. `.incomplete` không phải kết quả. Nếu Test đã bắt đầu nhưng chưa validated, bắt buộc manual review và cấm tự retry.
11. Checkpoint/publish bổ sung `authorization.json`, `hyperparameters.json`, `feature_schema.json`; recovery chỉ publish checkpoint đã `run_validated`, không Test lại.

## 12–15. Toàn vẹn khoa học và output

12. Baseline giữ 380 feature, scaler+SVC Train+Val và SVC RBF C10/probability=True/class_weight=None.
13. Compact giữ MI Train-only Top100, RF-RFE step1/100 cây/depth15, k1..100 chọn Validation Accuracy và SVC probability=False/balanced.
14. Test vẫn được nạp sau feature freeze/final refit và đánh giá đúng một lần.
15. Result bổ sung hyperparameter contract, schema hash, stage timings, support-vector count, warnings; metrics cũ vẫn gồm Accuracy, Precision, Recall, F1, Macro-F1, Balanced Accuracy, confusion matrix và classification report.

## 16–18. Notebook, test và tài liệu

16. Tạo notebook 12 cell `STAGE7_3B_OFFICIAL_SVM.ipynb`; không dùng legacy Cell 4A, không `Run all`, launch mặc định disabled.
17. Viết ba bộ test mock/static cho authorization+bypass/RF regression, supervisor+checkpoint và scientific ordering. **Chưa chạy**, nên trạng thái là `PENDING USER TEST`, không phải PASS.
18. Tạo resource policy, Colab guide, scientific-integrity document và invalid approval template.

## File tạo mới

- `D:\COMPACT-MI-RFE\colab\svm_run_authorization.py`
- `D:\COMPACT-MI-RFE\colab\svm_official_worker.py`
- `D:\COMPACT-MI-RFE\colab\svm_official_supervisor.py`
- `D:\COMPACT-MI-RFE\colab\test_stage73b_authorization.py`
- `D:\COMPACT-MI-RFE\colab\test_stage73b_supervisor.py`
- `D:\COMPACT-MI-RFE\colab\test_stage73b_scientific_integrity.py`
- `D:\COMPACT-MI-RFE\notebooks\STAGE7_3B_OFFICIAL_SVM.ipynb`
- `D:\COMPACT-MI-RFE\configs\stage73b\authorization_template.BLOCKED.json`
- `D:\COMPACT-MI-RFE\huyen_code\STAGE7_3B_REPORT.md`
- `D:\COMPACT-MI-RFE\huyen_code\STAGE7_3B_RESOURCE_POLICY.md`
- `D:\COMPACT-MI-RFE\huyen_code\STAGE7_3B_COLAB_GUIDE.md`
- `D:\COMPACT-MI-RFE\huyen_code\STAGE7_3B_SCIENTIFIC_INTEGRITY.md`

## File chỉnh sửa

- `D:\COMPACT-MI-RFE\huyen_code\ciciomt24_training.py`
- `D:\COMPACT-MI-RFE\colab\experiment_launcher.py`
- `D:\COMPACT-MI-RFE\colab\checkpoint_manager.py`

Không sửa năm script CTU-13, window CSV, Stage 5 result hoặc dữ liệu gốc.

## Blocker trước khi có thể chạy Baseline FW512/FS16

- Mã Stage 7.3B mới chưa được người dùng chạy unit/regression test.
- Chưa có commit sạch và đối chiếu hash Windows/Colab cho source mới.
- Chưa có unit-test evidence gắn với đúng training SHA.
- Chưa có approval hợp lệ và resource budget do người dùng chốt.
- Chưa có exact-config probe; nếu dùng evidence FW256/FS8 thay thế, người dùng phải duyệt rõ non-equivalence.
- Chưa chạy authorization review trên runtime Colab hiện tại.

## Audit tác động Stage 6 / Random Forest

Nhánh `mi_rfe_rf` không yêu cầu Stage 7.3B authorization context và CLI RF vẫn chuyển đến `run_official` như trước. Các hằng `RF_RFE_PARAMS`/`RF_PARAMS`, MI Top100, RFE step=1, k1..100, tie rule và final refit không bị sửa. Checkpoint chỉ bắt buộc ba artifact Stage 7.3B mới đối với `baseline_svm`/`mi_rfe_svm`, nên recovery/publish RF cũ không bị ép có `authorization.json`. Test mock `test_rf_path_does_not_require_svm_authorization` đã được viết nhưng chưa chạy. Không sửa hoặc ghi đè bất kỳ Stage 5 result nào.

## PASS/FAIL trước official execution

PASS yêu cầu đồng thời: checkout sạch; identity/hash/source/data/runtime đúng approval; test evidence hiện hành; dataset/source preflight; technical evidence policy; budget và admission; không có run/result/checkpoint/environment trùng tên; authorization review báo `PASS_STAGE73B_AUTHORIZED` với 0 fit và 0 Test feature rows. Bất kỳ FAIL/UNKNOWN nào đều giữ BLOCKED. Ngay cả khi tất cả PASS, người dùng vẫn phải có quyết định riêng và exact confirmation trước Cell 10; code không tự chuyển trạng thái.

## Đánh giá rủi ro cấu hình đầu tiên

`baseline_svm/FW_512_FS_16` là lựa chọn SVM rủi ro thấp hơn nhánh compact vì không chạy MI/RFE/k-search, nhưng final fit vẫn có 897.932 window × 380 feature và `probability=True`. Thời gian và peak RAM full vẫn UNKNOWN; Stage 7.2 64k không thể ngoại suy tuyến tính. Supervisor giảm rủi ro lan sang tiến trình khác và tạo bằng chứng dừng, nhưng polling không bảo đảm tránh Colab OOM hoặc session eviction.

## Trạng thái chuyển bước

Chưa được chạy Cell 8 hoặc official SVM. Bước tiếp theo là người dùng tự chạy các lệnh test trong `STAGE7_3B_COLAB_GUIDE.md`, commit/push phạm vi Stage 7.3B, clone checkout sạch trên Colab và gửi log/hash để review. Chỉ sau một quyết định phê duyệt riêng mới tạo approval hiệu lực và chạy supervisor.
