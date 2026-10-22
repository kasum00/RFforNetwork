# Stage 7.3B — Chính sách tài nguyên cho official SVM

## Trạng thái

**BLOCKED theo mặc định.** Stage 7.2 chỉ là probe giới hạn trên `FW_256_FS_8`; nó không chứng minh thời gian hoặc RAM của `baseline_svm/FW_512_FS_16`. Phần mềm không tự suy rộng probe và không tự sinh approval hợp lệ.

Mỗi lần chạy phải có một approval JSON riêng, khóa cứng các trường: experiment, config, run name, Git commit, SHA256 của training/sliding/manifest/validation/window summary, dataset fingerprint và scope, environment profile, hash tập năm script CTU-13. Chỉ một thay đổi cũng vô hiệu hóa approval.

## Budget bắt buộc

| Trường | Ý nghĩa | Hành vi khi vi phạm |
| --- | --- | --- |
| `max_elapsed_seconds` | Thời gian tối đa của worker | Dừng đúng cây worker, không retry. |
| `max_tree_rss_bytes` | RSS tối đa của worker và toàn bộ con | Dừng đúng cây worker, giữ `.incomplete`. |
| `min_start_available_ram_bytes` | RAM available tối thiểu trước khi khởi chạy | Không tạo worker. |
| `min_runtime_available_ram_bytes` | RAM host phải còn trong khi chạy | Dừng worker nếu thấp hơn. |
| `min_free_disk_bytes` | Dung lượng local tối thiểu trước và trong run | Không chạy hoặc dừng worker. |
| `poll_interval_seconds` | Chu kỳ đo 1–60 giây | Chỉ là polling, không bảo đảm bắt mọi spike/OOM. |

Điều kiện bổ sung: `max_tree_rss_bytes + min_runtime_available_ram_bytes <= available RAM` tại thời điểm admission. Các ngưỡng phải do người dùng duyệt; code không thay bằng số probe hoặc tự hạ dataset.

## Bằng chứng kỹ thuật

Approval phải chọn đúng một policy:

- `EXACT_CONFIG_PROBE`: evidence thuộc đúng experiment/config hiện chạy.
- `EXPLICIT_APPROVED_REPLACEMENT`: chỉ dùng khi người dùng chủ động chấp nhận probe khác config/nguồn bằng chứng thay thế, ghi lý do tối thiểu và xác nhận hai cấu hình không tương đương.

`PASS_TECHNICAL_ONLY` không tự mở full training. SHA và identity của file evidence phải khớp approval. Probe `FW_256_FS_8` hiện có không tự cấp phép cho `FW_512_FS_16`.

## Supervisor và trạng thái lỗi

Supervisor tạo đúng một worker và giám sát PID/cây con của worker. Khi vi phạm budget, nó terminate rồi kill các PID trong cây đã thu thập; không dùng lệnh giết theo tên, không ảnh hưởng tiến trình ngoài cây. Không retry, không resume, không đổi tham số, không giảm mẫu.

Nếu Test đã bắt đầu nhưng chưa có `run_validated`, báo cáo đặt `manual_review_required` và `do_not_retry_if_test_started_without_validation=true`. Không chạy lại Test dưới tên khác để tìm kết quả đẹp hơn. Checkpoint chỉ được publish lại khi đã `run_validated`, có đúng một Test start/completed và đúng thứ tự freeze → final refit → Test → validated.

## Rủi ro còn lại

- RSS polling không phải hard kernel/cgroup limit; spike giữa hai lần đo hoặc Colab OOM vẫn có thể xảy ra.
- RBF SVC có độ phức tạp phụ thuộc số support vector và dữ liệu, không ngoại suy tuyến tính từ 64k.
- `cache_size=10000` là trần cache, không phải RSS cố định.
- Colab có thể thu hồi phiên; fit SVC/RFE đang chạy không resume được.
- GPU T4 không tăng tốc pipeline scikit-learn này.
- Nhãn là `SCENARIO_PROXY`; capture/session independence chưa được chứng minh.
