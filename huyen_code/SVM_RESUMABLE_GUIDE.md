# MI-RFE SVM — tiếp tục nhiều phiên Colab

Notebook: `notebooks/CICIoMT24_MI_RFE_SVM_RESUMABLE.ipynb`. Cell **13** của notebook này gọi supervisor mới, chỉ hỗ trợ MI-RFE SVM. Các notebook/runner cũ vẫn giữ giao thức không resume; không chuyển checkpoint legacy sang schema mới.

## 1. Lệnh kiểm thử để người dùng tự chạy

Đường dẫn Windows trong prompt (chỉ dùng khi đã chép source mới vào repository đó):

```powershell
Set-Location 'C:\Users\LAB_LEE\Desktop\RFforNetwork_GITHUB'
$env:PYTHONPATH = 'C:\Users\LAB_LEE\Desktop\RFforNetwork_GITHUB;C:\Users\LAB_LEE\Desktop\RFforNetwork_GITHUB\huyen_code'
python -m unittest -v colab.test_svm_resumable_store colab.test_svm_resumable_training colab.test_svm_resumable_execution colab.test_stage73b_authorization colab.test_stage73b_scientific_integrity
```

Workspace thực tế nơi Codex đã thêm file:

```powershell
Set-Location 'D:\COMPACT-MI-RFE'
$env:PYTHONPATH = 'D:\COMPACT-MI-RFE;D:\COMPACT-MI-RFE\huyen_code'
python -m unittest -v colab.test_svm_resumable_store colab.test_svm_resumable_training colab.test_svm_resumable_execution colab.test_stage73b_authorization colab.test_stage73b_scientific_integrity
```

Codex **chưa thực thi** bất kỳ lệnh test/Python nào. Người dùng đã gửi log chạy lệnh trên tại Windows sau bản sửa fsync: **47/47 tests PASS trong 13.286 giây**, không lỗi hoặc skip. Đây là unit test với fixture/mocks, không phải kết quả full training. Vẫn phải tạo/xác minh evidence đúng commit và module hashes trên Colab; log Windows không tự mở execution gate.

## 2. Git: chỉ code và tài liệu mới

Dùng đúng một workspace. Các lệnh dưới đây dùng `D:\COMPACT-MI-RFE`; nếu làm trên máy LAB_LEE, thay dòng Set-Location bằng đường dẫn trên máy đó.

```powershell
Set-Location 'D:\COMPACT-MI-RFE'
git status --short
git diff -- README.md
git add -- colab/svm_resumable_store.py colab/svm_resumable_authorization.py colab/svm_resumable_training.py colab/svm_resumable_execution.py colab/test_svm_resumable_store.py colab/test_svm_resumable_training.py colab/test_svm_resumable_execution.py notebooks/CICIoMT24_MI_RFE_SVM_RESUMABLE.ipynb huyen_code/SVM_RESUMABLE_AUDIT.md huyen_code/SVM_RESUMABLE_REPORT.md huyen_code/SVM_RESUMABLE_GUIDE.md README.md
git diff --cached --name-only
git diff --cached --stat
git diff --cached -- colab notebooks/CICIoMT24_MI_RFE_SVM_RESUMABLE.ipynb huyen_code/SVM_RESUMABLE_AUDIT.md huyen_code/SVM_RESUMABLE_REPORT.md huyen_code/SVM_RESUMABLE_GUIDE.md README.md
```

Sau khi tự review và test:

```powershell
Set-Location 'D:\COMPACT-MI-RFE'
git commit -m "Add authorized completed-stage resume for MI-RFE SVM"
git push origin HEAD:main
git rev-parse HEAD
```

Không dùng `git add .`. Không thêm dataset, models, Drive approvals, cache hoặc run artifacts. Codex không stage, commit hoặc push.

## 3. Mở notebook mới

Sau khi push, mở:

[CICIoMT24_MI_RFE_SVM_RESUMABLE.ipynb trên Colab](https://colab.research.google.com/github/kasum00/RFforNetwork/blob/main/notebooks/CICIoMT24_MI_RFE_SVM_RESUMABLE.ipynb)

Dán **full commit mới** vừa có từ `git rev-parse HEAD` vào `EXPECTED_COMMIT` tại Cell 1. Không dùng commit c93eab cũ: nó chưa có bốn module resumable. Checkout mới dùng `/content/RFforNetwork_svm_resumable`; checkout đã tồn tại phải sạch và đúng commit. Dùng runtime sạch để tránh cache module Python từ checkout khác.

## 4. Phiên đầu tiên

Các biến mặc định:

```python
EXPERIMENT = 'mi_rfe_svm'
CONFIG = 'FW_512_FS_16'
RUN_ID = 'resumable01'
SESSION_ID = 'session01'
RESUME = True
SESSION_BUDGET_SECONDS = 36000
HEARTBEAT_INTERVAL_SECONDS = 60
STOP_STARTING_MARGIN_SECONDS = 300
ENABLE_OFFICIAL_TRAINING = False
```

Chạy từng cell, không Run all:

1. Cell 1–6: chọn đúng source, mount Drive, kiểm profile, kiểm inventory và copy/xác minh SHA256 SSD theo V3. Không tạo lại windows.
2. Cell 7: lấy identity mới. Dataset fingerprint vẫn có scope manifest+summary+relative paths/sizes; kiểm nội dung SSD bằng SHA256 là bước riêng ở Cell 6, không đồng nhất hai phép kiểm.
3. Cell 8: chủ động bật `RUN_RESUMABLE_UNIT_TESTS=True` nếu cần tự chạy test và tạo evidence. Evidence phải chứa tên ba suite mới, commit mới và hash bốn module; evidence Stage7.3B cũ không đủ. Sau khi thành công có thể để False để tái dùng evidence đã xác minh.
4. Cell 8A/9: probe Train-only đã hoàn tất, đúng config và đúng training/summary hash. Không lấy partial RFE/k của probe làm ranking/best_k chính thức. Nếu training hash không đổi, file probe hiện có có thể hợp lệ sau đối chiếu; approval cũ luôn không hợp lệ cho mode mới.
5. Cell 10: tạo draft vô hiệu. Các ngưỡng 28 GiB RSS/42 GiB start/10 GiB reserve/30 GiB disk là đề xuất vận hành cần người dùng xem xét; không phải dự đoán full fit. `max_elapsed_seconds` phải bằng session budget được duyệt.
6. Cell 11: chỉ khi đồng ý thì bật thao tác tạo approval mới và nhập đủ ba xác nhận. Phải chấp thuận rõ completed-stage resume/repeat-only-uncommitted/single-Test. Approval được đặt tên `<RUN_NAME>.<SESSION_ID>.json`, không dùng lại approval legacy.
7. Cell 12: authorize read-only; phải trả context hợp lệ, chưa fit/Test.
8. Cell 13: chỉ sau quyết định của bạn, bật `ENABLE_OFFICIAL_TRAINING=True` trong Cell 1 hoặc đặt biến này trước Cell 13. Chạy đúng một session. Worker cập nhật stage/k/elapsed/CPU/RSS và log Drive mỗi 60 giây, thêm event ngay mỗi khi fit/checkpoint kết thúc.
9. Cell 14: xem trạng thái. `PAUSED_AT_SAFE_CHECKPOINT` chưa phải kết quả khoa học cuối cùng.

## 5. Phiên tiếp theo

Giữ **nguyên RUN_ID/RUN_NAME/config/source/dataset/scientific profile**. Chỉ đổi `SESSION_ID='session02'` và tạo một approval mới cho phiên đó. `RESUME=True` vẫn giữ.

Ví dụ phiên 1 commit k1..12 rồi bị ngắt ở k13: khi phiên 2 được duyệt, hệ thống kiểm SHA256/dependency của MI, RFE, k1..12, dựng lại memmap nếu runtime mới, rồi bắt đầu k13. Không gọi lại MI/RFE hay fit k1..12.

Nếu local SSD/memmap còn cùng identity và checksum, hệ thống dùng lại. Nếu runtime mới mất `/content`, Train/Validation được đọc lại bằng loader canonical; MI/RFE và các k hoàn tất vẫn dùng artifact nhỏ trên Drive. Không upload ma trận nhiều GiB lên Drive.

Nếu final fit đã commit thì phiên tiếp theo dùng model/scaler đó và không đọc lại Train/Validation chỉ để refit. Nếu Test chưa được claim, có thể chuyển tới Test. Nếu final Test đã commit nhưng publication lỗi, session mới chỉ xác minh/publish lại artifact, không fit/Test lần nữa.

### Phiên Colab bị thu hồi khi còn lock

Khóa `active_session.json` chặn concurrent sessions. Khi phiên cũ chết hẳn, trong draft của **phiên mới** ghi:

```python
draft['interrupted_session_takeover'] = {
    'previous_session_id': 'session01',
    'previous_session_terminated_confirmed': True,
}
```

Thực hiện trước tạo approval Cell 11, sau khi đã xác nhận runtime cũ không còn chạy. Supervisor vẫn từ chối khi heartbeat cũ còn mới (dưới max(180 giây, 3×heartbeat interval)). Lock cũ được chuyển sang archive, không xóa checkpoint/log. Không xác nhận termination chỉ vì trình duyệt mất kết nối: worker Colab có thể vẫn còn chạy.

## 6. Dừng theo ngân sách 10 giờ

Không bắt đầu MI/RFE/k/final/Test mới khi thời gian còn lại không quá 300 giây. Fit đã hoàn tất được commit trước khi nghỉ. Supervisor vẫn giữ hard elapsed/RSS/RAM/disk guard; polling có thể trễ một chu kỳ và không ngăn chắc spike/OOM.

Nếu một SVC.fit kéo dài quá 10 giờ, supervisor có thể hủy fit đó. Các k hoàn tất trước đó được giữ; **fit đang dở không resume được**. Final SVC fit cũng phải hoàn tất trong một phiên; không thể chia optimizer của final fit qua nhiều phiên.

Không có auto retry hoặc auto mở phiên kế tiếp. Mỗi phiên mới cần session ID/approval riêng. JSON progress có `elapsed_seconds` của phiên và `total_elapsed_seconds` là tổng thời gian worker đã quan sát (không gồm thời gian chờ giữa phiên; một phiên chết đột ngột có thể thiếu phần sau heartbeat cuối).

## 7. Test chỉ một lần

Trước lần đọc Test đầu tiên, tạo độc quyền `results/test_evaluation_claim.json`, ràng buộc final checkpoint hash. Nếu session chết sau claim mà chưa có `final_test` hợp lệ, mọi automatic resume Test bị chặn, kể cả chưa biết prediction đã hoàn tất hay chưa. Đây là cách xử lý bảo thủ để tránh đo Test hai lần. Final model vẫn được giữ; không refit nó để vượt lỗi Test.

Khi đã có `final_test` hợp lệ, có thể phục hồi publication sau lỗi copy bằng session mới; không gọi evaluate_final nữa. Mỗi lần copy/publish đều xác minh checksum. Không dùng Test để quyết định đổi k, feature hoặc tham số.

## 8. Cấu trúc output

```text
COMPACT-MI-RFE/svm_resumable/mi_rfe_svm/FW_512_FS_16/<RUN_NAME>/
  run_manifest.json
  active_session.json                         # chỉ khi có phiên active
  progress.json
  heartbeat.jsonl
  events.jsonl
  sessions/
    session01/
      authorization.json
      preflight.json
      supervisor_progress.json
      worker_result.json                     # nếu worker kết thúc có kiểm soát
      session_result.json
  checkpoints/
    data_prepared/{checkpoint.json,data.json}
    mi/{checkpoint.json,data.json,mi.joblib,top100_scaler.joblib,mi_feature_scores.csv}
    rfe/{checkpoint.json,data.json,rfe_feature_ranking.csv}
    k_search/k_001/{checkpoint.json,data.json}
    ...
    k_search/k_100/{checkpoint.json,data.json}
    selection/{checkpoint.json,data.json}
    final/{checkpoint.json,data.json,final_model.joblib,final_scaler.joblib}
    final_test/{checkpoint.json,data.json,result.json}
  results/
    test_evaluation_claim.json
    completed/
      result.json
      final_test_metrics.json
      best_k.json
      final_model.joblib
      final_scaler.joblib
      run_manifest.json
      artifact_manifest.json
```

Thư mục `.incomplete-<uuid>` được giữ làm bằng chứng ghi dở, không được load như checkpoint. Một directory chính thức thiếu/sai manifest hoặc checksum sẽ dừng thay vì âm thầm recompute hoặc overwrite. File joblib chỉ được load sau khi xác minh checkpoint của run đã duyệt trên Drive tin cậy của bạn.

## 9. Bằng chứng cần gửi lại

Log test, commit mới, Cell 12, progress và session_result, run_manifest, final_test result và artifact_manifest nếu hoàn tất. Nếu lỗi, gửi last error/stop reason, session_result, progress và tình trạng Test claim. Không gửi dataset và không chạy Test lại để lấy log.
