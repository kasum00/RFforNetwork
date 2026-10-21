# Stage 7.3A — Full SVM resource feasibility

**Quyết định hiện tại: REVIEW_REQUIRED, full SVM BLOCKED.** Không có số đo full SVM; không suy rộng tuyến tính probe 64k thành dự đoán giờ/RAM. Số dòng dưới đây là manifest được duyệt, không phải kết quả training mới.

| Experiment | Cấu hình | Train | Val | Test | Final fit Train+Val | Full compute / peak RAM | Gate |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| Baseline SVM | FW_512_FS_16 | 673.559 | 224.373 | 224.287 | 897.932 | UNKNOWN / UNKNOWN | REVIEW_REQUIRED |
| Baseline SVM | FW_256_FS_16 | 674.169 | 224.677 | 224.631 | 898.846 | UNKNOWN / UNKNOWN | REVIEW_REQUIRED |
| Baseline SVM | FW_512_FS_8 | 1.347.101 | 448.735 | 448.563 | 1.795.836 | UNKNOWN / UNKNOWN | REVIEW_REQUIRED |
| Baseline SVM | FW_256_FS_8 | 1.348.317 | 449.343 | 449.251 | 1.797.660 | UNKNOWN / UNKNOWN | REVIEW_REQUIRED |
| MI-RFE + SVM | FW_512_FS_16 | 673.559 | 224.373 | 224.287 | 897.932 | UNKNOWN / UNKNOWN | REVIEW_REQUIRED |
| MI-RFE + SVM | FW_256_FS_16 | 674.169 | 224.677 | 224.631 | 898.846 | UNKNOWN / UNKNOWN | REVIEW_REQUIRED |
| MI-RFE + SVM | FW_512_FS_8 | 1.347.101 | 448.735 | 448.563 | 1.795.836 | UNKNOWN / UNKNOWN | REVIEW_REQUIRED |
| MI-RFE + SVM | FW_256_FS_8 | 1.348.317 | 449.343 | 449.251 | 1.797.660 | UNKNOWN / UNKNOWN | REVIEW_REQUIRED |

## Quan sát thực, không phải ước lượng full

Stage 7.2 chỉ đo Train-only 16k/32k/64k, tối đa 51.200 dòng fit sau internal split. 64k Baseline 380 feature: `SVC.fit` 131,90 s, peak process-tree RSS 894,4 MiB, 1.209 support vectors, fit_status=0. MI-RFE-SVM: MI 88,43 s; **partial** RFE 4 forest fits/2,07 s; k=1/10/100 SVC 4,070/0,477/2,110 s; peak tree RSS 1.066,5 MiB, fit_status=0. Sample 64k chỉ ~4,75% Train FW256/FS8. Support-vector fraction và class composition có thể đổi mạnh ở full. Không dùng bất kỳ probe accuracy làm kết quả chính thức.

Runtime Colab khi probe: khoảng 4 CPU vật lý, RAM tổng 50,99 GiB, available trước run khoảng 48,8 GiB. Đây **không** phải hạn mức được bảo đảm cho phiên sau. Probe có cap 12 GiB và timeout 1.200 s/worker; cap/timeout đó **không** tự áp dụng cho full. CPU budget chính thức, max process-tree RSS, max elapsed, disk reserve và thời lượng phiên Colab chưa được người dùng chốt; trạng thái `UNKNOWN`.

Baseline `probability=True` tăng chi phí fitting so với `probability=False`; libsvm RBF có kernel cache cap 10.000 MB, không phải lượng RAM luôn cấp phát. Nhu cầu support vectors và kernel operations có thể tăng phi tuyến theo N. Full matrix Train+Val của Baseline 380 float32 có lower bound riêng ~1,27 GiB ở FS16/~2,54 GiB ở FS8, chưa kể source arrays, float64 libsvm copies, cache, Test, Python/process overhead hoặc model. `official_memory_preflight` áp scenario RAM phải nhỏ hơn 85% RAM available lúc chạy và kiểm tra disk tối thiểu max(1,5× raw float32 all splits, 8 GiB); **đó chỉ là admission floor**, không phải upper bound peak hay bằng chứng runtime vừa phiên.

MI-RFE-SVM thêm MI Top100 trên toàn Train, RF-RFE hoàn chỉnh `step=1` (~100 forest fits), 100 SVC validation fits k1..100, rồi final SVC fit Train+Val. RFE forests chạy `n_jobs=-1`; CPU và tree buffers phụ thuộc dữ liệu. Probe partial RFE 4 fits hoàn toàn không đo được chi phí 100 fits. k-search có thể kéo dài nhiều giờ hoặc vượt một phiên Colab; chi phí restart từ đầu rất cao vì hiện **không resume** giữa RFE/fit/k-search. FS8 có ~2× số window FS16, không được giả định chính xác 2× thời gian/RAM.

| Hạng mục | Giá trị hiện có | Độ chắc chắn | Quyết định |
| --- | --- | --- | --- |
| Peak RSS 64k probe | 894,4 MiB Baseline; 1.066,5 MiB MI-RFE-SVM | Cao cho đúng probe/runtime đó | Không dự báo full. |
| Full compute time, từng nhánh/config | UNKNOWN | Không đủ căn cứ | Chưa duyệt session budget. |
| Full peak process-tree RSS | UNKNOWN | Không đủ căn cứ | Cần runtime admission và safety margin riêng. |
| Colab phiên mới CPU/RAM/đĩa | UNKNOWN trước khi user chạy preflight | Thay đổi theo phiên | Không reuse số cũ như quota bảo đảm. |
| `cache_size=10000` | 10.000 MB cap | Cao về tham số, không phải allocated RSS | Giữ nguyên khoa học. |
| Disk/artifact/Drive | ≥8 GiB local admission; artifact thực tế UNKNOWN | Chỉ là floor | Cần dự phòng và verify SHA; không upload memmap. |
| Worker isolation, timeout, hard RSS | Chưa có cho official SVM | Cao qua source audit | Full run BLOCKED đến khi thiết kế/test Stage7.3B. |

## Phương án theo mức rủi ro

Ưu tiên đánh giá **Baseline SVM FW_512_FS_16** đầu tiên: final fit 897.932 dòng, ít hơn các cấu hình còn lại và không có MI/RFE/100 k-search. FW_256_FS_16 chỉ lớn hơn ít; thứ tự này không phải chọn mô hình theo Test. Sau khi có budget và run chính thức hợp lệ mới cân nhắc Baseline FS8, rồi MI-RFE-SVM FS16/FS8; 100-fit k-search khiến nhánh compact rủi ro phiên cao hơn. Đây chỉ là **đề xuất thứ tự**, không cấp phép chạy hay đổi cấu hình khoa học.

Ngân sách Stage7.3B phải được người dùng nhập rõ `max_elapsed_seconds`, `max_tree_rss_bytes`, RAM safety margin, disk free threshold, experiment/config/run_name, Git commit, training/sliding hashes, manifest/window fingerprint; trước đó mọi resource gate là REVIEW_REQUIRED. Nếu người dùng không thể chấp nhận thời gian/RAM hoặc phiên Colab không đạt admission, dừng; không tự giảm mẫu hay đổi sang linear/approximate SVM. Polling RSS có thể bỏ lỡ spike; dù có supervisor, vẫn không đảm bảo tránh OOM. Cần báo run incomplete và giữ artifacts nếu dừng, không dùng Test để điều chỉnh quyết định.
