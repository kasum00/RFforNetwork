# Stage 9 — Final SVM A1/B và Test một lần

Notebook: `notebooks/STAGE9_SVM_A1_B_FINAL_EVALUATION_COLAB.ipynb`.

Dataset duy nhất: `/content/drive/MyDrive/CICFLOWMETER_FEATURE_CSV`.
Trạng thái khoa học vẫn `EXPERIMENTAL_NON_OFFICIAL / OFFICIAL_BLOCKED`.

Chỉ Stage9 được tạo mới. Không sửa A1, B, A0, Stage0–4, notebook selection, dataset hoặc checkpoint cũ.
Không MI, RF-RFE, k-search, sampling hay chọn lại k trong worker Stage9.

## Hợp đồng đã đối chiếu

- A1: RF-RFE step5, exhaustive k=1..100; B: step2, coarse `(10,20,30,50,75,100)`, fine ±5 quanh một coarse winner. Ranking và best_k có thể khác nhau.
- Mỗi phương án dùng **chính** bộ verify/ledger của source tại saved commit. Stage9 gọi chúng trong process riêng; không gọi `pipeline.run` của selection.
- Kiểm tra đủ SHA/sequence/parent/source/scaler/selection publication, full history theo policy. A1 còn yêu cầu mỗi k đã COMMITTED trước selection; B verifier kiểm tra durable coarse/fine plan.
- Đọc best_k/features theo checkpoint, giữ nguyên thứ tự; các best_k từng thấy trong notebook không được dùng làm fallback.
- `colab/cicflow_reference_pipeline.py:743` quy định final scaler **mới** fit Train+Validation trên các cột đã freeze, C-order float32, Train trước Validation. Stage9 gọi trực tiếp hàm này. Scaler Top100 Train-only vẫn được xác minh như bằng chứng selection và không dùng thay final scaler.
- SVC giữ `C=10`, RBF, `max_iter=20000`, `gamma="scale"`, balanced, shrinking, probability=False, cache_size=10000 MiB, seed42 và toàn bộ defaults đã resolve từ selection source.
- Train/Validation dùng nguyên split, nguyên hàng và cùng cleaning float64/nonfinite→0/float32. Không fit scaler trên Test.
- Nhãn Attack xác nhận qua manifest/producer SHA. Mapping hiện tại là Attack_folder_proxy=1, Benign_folder_proxy=0; chưa phải ground truth độc lập. Hàm metric truyền pos_label rõ ràng và tính macro riêng.

## Output

```text
MyDrive/experiment_results/stage9_svm_final/
  A1/<CONFIG>/
    identity_manifest.json
    final_model/
      model.joblib
      final_scaler.joblib
      final_state.json
      selection_binding.json
      resource_usage.json
      final_checkpoint.json
    FINAL_COMPLETE.json
    TEST_CLAIM.json
    test_ledger.json
    test_metrics.json
    confusion_matrix.csv
    test_predictions.npz
    test_checkpoint.json
    TEST_COMPLETE.json
    leases/
    heartbeat.json
  B/<CONFIG>/ ... bộ artifact độc lập tương tự ...
  logs/<METHOD>/<CONFIG>/<task-uuid>.log + .receipt.json + .tests.json
  reports/<UTC_UUID>/
    SVM_A1_TEST_RESULTS.csv
    SVM_B_TEST_RESULTS.csv
    SVM_A1_B_COMPARISON.csv
    SVM_A1_B_FINAL_RESULTS.csv
    SVM_A1_B_FINAL_RESULTS.json
    STAGE9_VERIFICATION_REPORT.md
    STAGE9_EXECUTION_SUMMARY.md
```

Mỗi `(method, config)` có một namespace Test. Run identity được khóa trong thư mục này, không có sub-run mới để lách claim. Identity khác thì BLOCKED, không ghi đè.
Reports dùng thư mục version mới mỗi lần export. Kết quả thiếu/không verified để trống, PENDING/BLOCKED; không điền 0. JSON giữ metric thô 0–1, CSV/Markdown nhân 100 ở bước trình bày, không làm tròn trước tính toán.

## CLI (chỉ bạn chủ động thực thi)

`python -B -u -m stage9_svm_final.cli --request <request.json>` nhận một request JSON như `request_for()` trong notebook.
Modes: `verify-selection`, `preflight`, `final-fit`, `resume-final`, `verify-final`, `test-readiness`, `test`, `verify-test`.

Trong Colab, luôn dùng `launch(request)` hoặc `safe_launch(request)` để có stdout/stderr trực tiếp và log tự động.
Final fit và Test có hai flag, hai câu xác nhận khác nhau cho từng method/config/run. Không có mode final+Test tự động.

Log tổng hợp chính thức: `/content/drive/MyDrive/code_CICFLOW/logs/ALL_EXECUTION_RESULTS.log`.
Controller append start/end, command, request đã che secrets, stdout/stderr, native exit, reported status, runtime, heartbeat/resource và artifact paths; log lỗi có cảnh báo, giữ native exit. Không suy PASS từ exit0.

## Hướng dẫn và giới hạn

- [Chạy từng cell](reports/COLAB_GUIDE.md).
- [Scientific/source review và tình trạng checkpoint](reports/SOURCE_AND_CHECKPOINT_AUDIT.md).
- [Resume và claim bị gián đoạn](reports/RESUME_PROTOCOL.md).
- [Danh sách bàn giao/chưa kiểm chứng](reports/IMPLEMENTATION_REPORT.md).

Không có Test claim thì final fit bị ngắt có thể tính lại chính final fit; không có mid-SVC resume. Local final checkpoint hoàn chỉnh được dùng tiếp để publication, không fit lại. Partial/corrupt remote bundle không được ghi đè. Nếu runtime mất khi bundle remote còn thiếu thì BLOCKED để kiểm tra thủ công.
Test đã claim mà chưa có completion verified thì **không retry**, kể cả còn prediction/checkpoint local. Hãy giữ chúng để xử lý publication/recovery riêng có kiểm chứng; phiên bản này không tự recovery claim.

Drive FUSE readback/SHA không chứng minh Google đã replicate cloud tức thì. Lỗi ghi claim có thể để lại partial file: vẫn chặn Test. Lease stale chỉ được takeover bằng câu xác nhận sau khi biết worker cũ thực sự dừng.

Codex chưa chạy Python, tests, Git, preflight, training hoặc Test; chưa có runtime PASS và chưa xác minh trực tiếp các run trên Drive.
