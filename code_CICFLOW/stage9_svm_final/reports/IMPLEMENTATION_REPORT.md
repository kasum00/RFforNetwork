# Stage9 implementation handoff

Ngày: 2026-11-02. Trạng thái bàn giao: SOURCE WRITTEN — RUNTIME NOT VERIFIED.

## Kết quả triển khai

- Notebook mới có 19 code cells và cell IDs, không có saved output, mặc định preflight; final/Test đều False.
- Namespace A1 và B độc lập cho bốn cấu hình; archived-source verifier riêng, không selection fitting.
- Core final scaler/SVC được gọi trực tiếp sau freeze: full Train+Validation, giữ scientific parameters/defaults/order.
- Exclusive durable Test claim/readback trước mọi Test byte; completed chỉ verify, interrupted/ambiguous không retry.
- Checkpoint/model/scaler/selection SHA, worker leases, local publication resume, heartbeat đo thực, native exit và append logging.
- Test metrics thô và phần trăm, prediction-based recomputation, confusion matrix, eight-row reports, bảng A1/B/comparison riêng.

## File mới — không sửa file cũ

`notebooks/STAGE9_SVM_A1_B_FINAL_EVALUATION_COLAB.ipynb`

- `code_CICFLOW/stage9_svm_final/.gitignore`
- `code_CICFLOW/stage9_svm_final/README.md`
- `code_CICFLOW/stage9_svm_final/__init__.py`
- `code_CICFLOW/stage9_svm_final/cli.py`
- `code_CICFLOW/stage9_svm_final/common.py`
- `code_CICFLOW/stage9_svm_final/controller.py`
- `code_CICFLOW/stage9_svm_final/data.py`
- `code_CICFLOW/stage9_svm_final/environment.py`
- `code_CICFLOW/stage9_svm_final/metrics.py`
- `code_CICFLOW/stage9_svm_final/notebook_helpers.py`
- `code_CICFLOW/stage9_svm_final/pipeline.py`
- `code_CICFLOW/stage9_svm_final/preflight.py`
- `code_CICFLOW/stage9_svm_final/reference_helpers.json`
- `code_CICFLOW/stage9_svm_final/reports.py`
- `code_CICFLOW/stage9_svm_final/reports/COLAB_GUIDE.md`
- `code_CICFLOW/stage9_svm_final/reports/IMPLEMENTATION_REPORT.md`
- `code_CICFLOW/stage9_svm_final/reports/RESUME_PROTOCOL.md`
- `code_CICFLOW/stage9_svm_final/reports/SOURCE_AND_CHECKPOINT_AUDIT.md`
- `code_CICFLOW/stage9_svm_final/requirements-colab.txt`
- `code_CICFLOW/stage9_svm_final/selection.py`
- `code_CICFLOW/stage9_svm_final/selection_worker.py`
- `code_CICFLOW/stage9_svm_final/selftest.py`
- `code_CICFLOW/stage9_svm_final/store.py`
- `code_CICFLOW/stage9_svm_final/telemetry.py`
- `code_CICFLOW/stage9_svm_final/tests/__init__.py`
- `code_CICFLOW/stage9_svm_final/tests/fixtures.py`
- `code_CICFLOW/stage9_svm_final/tests/test_admission_reports.py`
- `code_CICFLOW/stage9_svm_final/tests/test_final_and_test.py`
- `code_CICFLOW/stage9_svm_final/tests/test_logging.py`

## Synthetic tests đã viết, chưa thực thi

| File | Số test methods đã viết | Kết quả thực thi |
|---|---:|---|
| tests/test_admission_reports.py | 18 | NOT_RUN |
| tests/test_final_and_test.py | 23 | NOT_RUN |
| tests/test_logging.py | 2 | NOT_RUN |

Tổng: 43 test methods. Không skip/xfail. Fixture Parquet nhỏ, final scaler thật trên fixture; costly SVC.fit được thay bằng synthetic fitted artifact để kiểm tra flow. Selftest chặn real SVC.fit và production Parquet reads. Các mock git/subprocess trong tests chỉ là fixture, Codex không gọi chúng.

Coverage: discovery/completion khác verification; selected features count/order; method/config/source/identity mismatches; source byte gate trước import; original verifier error không được promote; label producer; resource gate; current-source test evidence; full Train+Val scaler/column order; no Test trước claim/authorization; interrupted claim; single predict; completed verify; publication failure; final-only resume; partial non-overwrite; model/data SHA; metric positive label/macro; missing Test results blank; versioned reports; stdout/stderr/native exit và log failure.

## Điều đã kiểm tra bằng đọc file

- JSON notebook đọc lại được; đúng 19 code cells, execution_count=null, outputs=[]. Đây không phải chạy notebook.
- Đối chiếu SHA 68 file .py/.json/.ipynb A1/B/A0 đã đọc ở đầu nhiệm vụ: không thay đổi.
- Tất cả thao tác ghi của nhiệm vụ chỉ nhắm package Stage9 mới và notebook Stage9 mới. Dataset, Stage0–4, checkpoint, kết quả cũ và notebook selection không sửa/xóa.

## Chưa kiểm chứng / blocker còn lại

- Không chạy Python, syntax compilation, unittest, preflight, Git, pip, Colab, training, benchmark hoặc evaluation. Không có runtime PASS.
- Không mở Drive/checkpoints thực tế; source review không chứng minh tám selection usable. Cell6–7 phải discovery/full verify từng run.
- Chưa biết SHA40 commit mới cho Stage9; người dùng tự push rồi điền SHA. Không dùng commit cũ thiếu package này.
- B run IDs trong notebook là đề xuất cần đối chiếu, không phải bằng chứng hoàn thành. Không hardcode các best_k tham khảo.
- Original selection commits phải còn trên GitHub và checkout nguyên vẹn; libraries/Python không khớp sẽ BLOCKED.
- RAM/disk là estimate; không có dự báo runtime hoặc benchmark. SVC không có mid-fit resume.
- Runtime/source/dataset/identity thay đổi, foreign/partial artifacts hoặc unknown Test claim đều được chặn; không tự migration.
- Phiên bản này không tự recovery publication cho Test đã claim. Local prediction/checkpoint còn lại được bảo tồn để thiết kế recovery riêng khi cần.
- Proxy labels/capture limitations còn nguyên; Test không tạo official reproduction hay reference-equivalence.

## Cách bạn thực thi

Đọc `reports/COLAB_GUIDE.md`: lệnh Git, unittest qua shared PowerShell wrapper và toàn bộ các bước Cell1–19.

Lần đầu: push file mới → điền SHA40 → chọn METHOD/config/run IDs → Cells1–10. Chỉ sau READY mới chủ động cho phép Cell11–12 final fit; verify Cell13; Test readiness Cell14; xác nhận riêng Cell15–16; verify/export Cell17–19.

Xem `reports/SOURCE_AND_CHECKPOINT_AUDIT.md` về từng run và `reports/RESUME_PROTOCOL.md` về interrupted final/Test.

Bổ sung rà soát cuối: chặn archived/current core helper khác byte; phát hiện kernel còn giữ phiên bản thư viện cũ dù metadata đã đổi. Không tự restart hoặc sửa gate.
