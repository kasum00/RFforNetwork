# B step2 — triển khai hoàn chỉnh theo run_phuong_anB.py người dùng cung cấp

## Kết quả

Notebook độc lập: notebooks/PHUONG_AN_B_MI_RFE_SVM_STEP2_COLAB.ipynb, 14 code cells,
outputs rỗng, EXECUTE_QUEUE=False và RFE_STEP=2. Source hỗ trợ tại
code_CICFLOW/phuong_anB_step2. Không ghi đè source/notebook/checkpoint A1.

Policy cuối cùng khớp source được người dùng dán:
COARSE=(10,20,30,50,75,100), RADIUS=5. Coarse best theo Validation Accuracy cao nhất,
hòa k nhỏ nhất; fine tăng dần trong coarse_best±5 giới hạn1..100 và bỏ tried k.
Best sau fine dùng cùng tie-break. Không dùng coarse grid cách2; đó không phải ý
nghĩa của RFE_STEP=2. Các policy draft tự đề xuất đã bị loại khỏi production.

## Đối chiếu input và dataset

Đã đọc E:/PHUONG_ANA_A1_COLAB_CLEAN_V2 (1).ipynb, source A1/A0/core mà notebook
sử dụng và bảy metadata sidecar của đủ bốn config CICFLOW. Full run_phuong_anB.py
được người dùng dán trong chat; không có local file được cung cấp để hash toàn bộ.
Evidence original_policy_excerpt.py giữ nguyên constants/pure fine_grid/choose_best,
không phải full original runner. implementation_review.json ghi provenance trung thực.

Dataset root bắt buộc MyDrive/CICFLOWMETER_FEATURE_CSV. Đã khóa nguyên bytes/hashes
28 sidecar nhỏ và declared model SHA vào dataset_reference.json. Không tạo/sửa CSV,
Parquet, split, window, schema hoặc label; không tải/copy dữ liệu lớn. Actual
Train/Validation bytes/footer vẫn phải được user preflight xác minh. Test không mở,
không hash/footer/read_table; adapter chỉ stat và dùng declared metadata như A1.

FW512_FS8 path alias Windows được xử lý bằng original frozen snapshot bytes sau
SHA/size/mapping check, qua evidence_root được hỗ trợ, không sửa sidecar/fingerprint.

## Thay đổi thuật toán và phần hạ tầng

| Hạng mục | B gốc người dùng dán | B CICFLOW step2 |
|---|---|---|
| Coarse SVM k | 10,20,30,50,75,100 | Giữ nguyên |
| Fine radius | 5 | Giữ nguyên |
| Coarse/final tie-break | max Validation Accuracy, min k | Giữ nguyên |
| Không refit tried k | checkpoint hợp lệ cùng binding | Giữ nguyên |
| RF-RFE elimination step | 1 | 2, thay đổi thuật toán được yêu cầu |
| MI/scaler/RF estimator/SVC params | canonical helpers | Giữ helper numerical của A1 CICFLOW đã khóa SHA |
| Dataset | CICIoMT2024 | CICFLOWMETER_FEATURE_CSV theo yêu cầu, splits/schema A1 nguyên trạng |
| Supported config | original driver subset | đủ bốn cấu hình CICFLOW |
| Source/ranking reuse | original store nếu tương thích | namespace B riêng; không reuse A1/step1 ranking |
| Final refit/Test | Không | Không có entrypoint |

RFE step2 giữ native sklearn rank semantics; grouped ties có deterministic total
order theo rule A1 (rank, removal importance, MI position). Rule được bind contract;
không khẳng định ranking/Accuracy tương đương step1. Full Top100 step2 có51 RF fits.
Coarse SVM=6, fine=5 hoặc10, completed unique SVM=11 hoặc16. Counts là lý thuyết lịch
công việc, không phải speedup đo thực tế. Không guarantee exhaustive1..100 optimum.

## Identity và checkpoint

Algorithm MI_RFE_SVM_B_STEP2_COARSE6_RADIUS5_V2, FORMAT phuong_anB_step2.selection.v2;
policy B.step2.coarse6.radius5.v2. Source/Git, environment, dataset fingerprint,
scientific params, RFE step2 và six-grid/radius5, config/run ID đều bind vào identity.
Checkpoint A1/step1 hoặc B draft policy bị BLOCKED. Không migrate/overwrite để né gate.

Delta store fork từ A1 giữ SHA/readback/parent sequence, exclusive artifacts,
COMMITTED marker và incomplete staging. Mỗi RF elimination, MI/scaler stage và SVM
candidate hoàn thành có checkpoint. Candidate chain mới theo ordinal/k/phase,
không contiguous1..100. Local verified pending prefix được commit trước fit tiếp;
failed checkpoint không tăng durable candidate count.

Fine plan được COMMITTED sau đủ six coarse candidates và trước fine đầu tiên,
bind coarse results digest/anchor/radius/list; verify kiểm tra cả thứ tự commit.
Resume giữa fine không đổi anchor theo best fine tạm thời. SVC fit bị ngắt chưa có
completed record phải tính lại; không có mid-fit checkpoint. Mismatch/partial
publication không xác minh được giữ nguyên và có thể BLOCKED để review.

## Progress, logging, thống kê

Notebook có tqdm.auto theo config/RFE_COARSE/COARSE/FINE, total/completed durable,
current k, best k/Validation Accuracy, elapsed, ETA từ unit time đã đo, RAM/CPU và
checkpoint timestamp. Heartbeat vẫn chạy khi SVM fit lâu; không đoán mid-fit %.
Waiting snapshot luôn đủ time/path keys; không lặp lỗi monitor A1.

Mọi worker/tests dùng controller append stdout/stderr gộp có nhãn và request/command,
start/end/runtime/native exit/program/scientific status vào:
MyDrive/code_CICFLOW/logs/ALL_EXECUTION_RESULTS.log. Có local raw log/native receipt;
worker logs/resources bind trong checkpoints. Log failure có cảnh báo, không đổi
native exit hoặc khẳng định đã lưu đủ.

Statistics lưu RF/SVM completed fits, recorded attempts, MI/RF/coarse/fine durations,
recorded selection-session wall time và peak sampled RSS. Work-complete receipt khóa
các prior session receipts để resume publication không làm trôi frozen thời gian.
Không dựng lại thời gian/fit attempts của hard interruption chưa persist; không benchmark.

## Kiểm thử đã viết — chưa thực thi

52 test methods dùng fake SVC, tied forest/nhỏ synthetic RF và fixture Parquet nhỏ:
- Original pure-policy regression: mọi coarse anchor, inclusive radius/boundaries,
  tie-break, no duplicate, max16 schedules; cùng fine_grid/choose_best từ source dán.
- One-call Train MI/seed/scaler/layout, fixed params, full feature mapping.
- RFE step2 so với sklearn (tied và RF nhỏ), partial elimination resume.
- Sparse k order/phase, final tie, interrupted fit, before/after commit failures,
  pause/resume trong coarse và fine, fine anchor freeze.
- No Test/final; fixed CICFLOW sidecar/evidence alias, source/dataset/policy/step identity.
- Integrity/duplicate/non-overwrite, publication/verify, log native/status separation,
  log I/O failure và monitor waiting/safe-stop behavior.

Tests không bỏ/skip/xfail; không đổi expectation để che lỗi production. Selftest guard
chặn real SVC.fit và production Parquet access. Chưa chạy để khẳng định tests PASS.

## Kiểm chứng chỉ bằng đọc/ghi file

Notebook JSON đọc lại được, outputs trống. Source A1/core helper và dataset sidecar
được so lại với snapshots để kiểm tra không thay đổi; không mở Parquet. Không chạy
Python/AST/compile/unittest/selftest/pip/Git/preflight/Colab/training/benchmark/Test.
Danh sách check đọc file cuối được trình bày trong bàn giao; không gọi là pipeline PASS.

## File mới trong nhiệm vụ B

- code_CICFLOW/phuong_anB_step2/.gitignore
- code_CICFLOW/phuong_anB_step2/README.md
- code_CICFLOW/phuong_anB_step2/__init__.py
- code_CICFLOW/phuong_anB_step2/cli.py
- code_CICFLOW/phuong_anB_step2/contract.py
- code_CICFLOW/phuong_anB_step2/controller.py
- code_CICFLOW/phuong_anB_step2/dataset.py
- code_CICFLOW/phuong_anB_step2/environment.py
- code_CICFLOW/phuong_anB_step2/evidence/__init__.py
- code_CICFLOW/phuong_anB_step2/evidence/approved_helpers.json
- code_CICFLOW/phuong_anB_step2/evidence/cicflow_split_evidence.json
- code_CICFLOW/phuong_anB_step2/evidence/dataset_reference.json
- code_CICFLOW/phuong_anB_step2/evidence/implementation_review.json
- code_CICFLOW/phuong_anB_step2/evidence/original_policy_excerpt.py
- code_CICFLOW/phuong_anB_step2/history.py
- code_CICFLOW/phuong_anB_step2/io_utils.py
- code_CICFLOW/phuong_anB_step2/notebook_helpers.py
- code_CICFLOW/phuong_anB_step2/pipeline.py
- code_CICFLOW/phuong_anB_step2/policy.py
- code_CICFLOW/phuong_anB_step2/preflight.py
- code_CICFLOW/phuong_anB_step2/reports/COLAB_GUIDE.md
- code_CICFLOW/phuong_anB_step2/reports/IMPLEMENTATION_REPORT.md
- code_CICFLOW/phuong_anB_step2/reports/RESUME_PROTOCOL.md
- code_CICFLOW/phuong_anB_step2/reports/SCIENTIFIC_CONTRACT.md
- code_CICFLOW/phuong_anB_step2/requirements-colab.txt
- code_CICFLOW/phuong_anB_step2/results.py
- code_CICFLOW/phuong_anB_step2/rfe.py
- code_CICFLOW/phuong_anB_step2/search.py
- code_CICFLOW/phuong_anB_step2/selftest.py
- code_CICFLOW/phuong_anB_step2/store.py
- code_CICFLOW/phuong_anB_step2/telemetry.py
- code_CICFLOW/phuong_anB_step2/tests/__init__.py
- code_CICFLOW/phuong_anB_step2/tests/fixtures.py
- code_CICFLOW/phuong_anB_step2/tests/helpers.py
- code_CICFLOW/phuong_anB_step2/tests/test_data_store.py
- code_CICFLOW/phuong_anB_step2/tests/test_policy_rfe.py
- code_CICFLOW/phuong_anB_step2/tests/test_publication_ui.py
- code_CICFLOW/phuong_anB_step2/tests/test_scaler_evidence.py
- code_CICFLOW/phuong_anB_step2/tests/test_search_resume.py
- notebooks/PHUONG_AN_B_MI_RFE_SVM_STEP2_COLAB.ipynb

Trong lượt chỉnh theo source gốc: sửa policy/contract/history/dataset/store/results/
pipeline/notebook và các tests/docs liên quan; thêm original_policy_excerpt.py và
metadata provenance. A1, CTU-13 gốc, datasets, old runs và shared wrapper không sửa.
Lệnh push/unit tests có logging và các cell preflight/run/resume/verify ở COLAB_GUIDE.md.
