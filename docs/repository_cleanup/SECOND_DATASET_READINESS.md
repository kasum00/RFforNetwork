# Readiness cho dataset thứ hai

Ngày đánh giá: 2026-10-26. Kết luận này là static architecture review, chưa phải readiness PASS.

## Kết luận

Repository có mức sẵn sàng **một phần** cho dataset thứ hai:

- Execution/checkpoint/authorization pattern có thể tái sử dụng ở mức trung bình.
- Model-selection semantics có thể tái sử dụng nếu giữ nguyên contract Train → Validation → final refit → Test.
- Preprocessing, label policy, split C+, window segmentation và feature contract hiện vẫn gắn với CICIoMT2024.
- Chưa nên copy một training script mới cho dataset thứ hai; cần một adapter/config contract trước.

## Thành phần có thể tái sử dụng

| Thành phần | Mức tái sử dụng | Điều kiện |
|---|---|---|
| `colab/checkpoint_manager.py`, `svm_resumable_store.py`, execution/supervisor | Cao về cơ chế | Dataset fingerprint, source hash và artifact schema phải là input rõ ràng; không dùng checkpoint CICIoMT24 cho dataset mới. |
| `local/` control plane | Trung bình | Tách identity/path binding khỏi tên file CICIoMT24; giữ fail-closed authorization. |
| MI/RF-RFE/SVM/RF orchestration trong `huyen_code/ciciomt24_training.py` | Trung bình | Đưa feature columns, label mapping, split names và paths vào adapter; bảo toàn hyperparameter/semantics đã phê duyệt. |
| Metrics/evaluation/reporting | Trung bình–cao | Xác định class order, imbalance policy và late-Test guard cho dataset mới. |
| `code_CICFLOW/paper_repro/` | Trung bình | Dùng như package/runner reference, không coi là core đã tách hoàn toàn. |
| `CODE_ctu13/` | Thấp về runtime, cao về reference | Giữ làm baseline/sha reference; không dùng path CTU-13 làm dataset adapter. |

## Thành phần phải xử lý riêng

| Thành phần | Ràng buộc hiện tại |
|---|---|
| `huyen_code/ciciomt24_split.py` | Default root `D:\COMPACT-MI-RFE\CICIoMT24`, scenario mapping, C+ chronology/duration/embargo và manifest schema riêng. |
| `huyen_code/ciciomt24_sliding_window.py` | Default `CSV_SPLIT_CPLUS`, source segment semantics, FW/FS và window provenance riêng. |
| Feature extraction/training | 76 flow features × 5 statistics = 380 columns; column whitelist và NaN/Inf policy cần được định nghĩa lại. |
| Labels | `SCENARIO_PROXY` theo scenario; dataset mới cần authoritative label policy, class order và audit riêng. |
| `colab/dataset_validator.py`/`drive_manager.py` | Tên `CICIoMT24`, `WINDOWS_CPLUS`, `CSV_SPLIT_CPLUS` và Drive layout hard-code. |
| `local/windows_preflight.py`/`checkpoint_migration.py` | Kiểm tra path/hash các module CICIoMT24 và Stage 7.3B. |
| Notebook/guide | Nhiều cell clone/import path và `/content/...` gắn dataset hiện tại. |

## Hard-code và dependency cần tháo trước dataset thứ hai

1. Dataset root và names trong `huyen_code/`, `colab/`, `local/`.
2. Manifest schema, fingerprint, source segment IDs và window summary.
3. `SCENARIO_PROXY`, mapping Benign/Attack, class imbalance và class order.
4. Feature list 76 cột, 5 statistics và 380-column contract.
5. C+ split/embargo/activity-crossing policy; không tái dùng cutoff CICIoMT24.
6. Source hash contract gồm 5 script CTU-13 và code CICIoMT24.
7. Checkpoint identity: dataset ID, config, run name, commit, source hashes và artifact root.

## Lộ trình đề xuất

1. Tạo `datasets/<dataset_id>/adapter.py`, `schema.json`, `label_policy.md`, `split_policy.md` và `README.md`; chỉ chứa metadata/code, không chứa dataset nặng.
2. Tạo interface trả về manifest, segment, feature contract và fingerprint; giữ implementation CICIoMT24 hiện tại làm adapter cũ.
3. Đưa MI/RFE/classifier/evaluation vào API nhận dữ liệu đã chuẩn hóa, không tự biết tên dataset.
4. Tạo config experiment mới với dataset ID và feature contract; không sửa config CICIoMT24.
5. Chạy static checks + synthetic/unit tests trên adapter mới; sau đó mới chạy audit/preflight dataset thật.
6. Chỉ tạo checkpoint mới với run name/dataset fingerprint mới; không resume chéo dataset.

## Điều kiện để tuyên bố ready

- Source/adapter hash và config identity cố định.
- Schema/label mapping được audit đầy đủ.
- Split trước window, embargo và activity-crossing policy có evidence.
- Feature count/column whitelist và NaN/Inf handling được khóa.
- Test được nạp muộn; validation dùng để chọn k/cấu hình; final refit trước Test.
- Checkpoint/resume và artifact paths được kiểm thử trên dữ liệu fixture nhỏ.

Hiện chưa đạt các điều kiện trên cho dataset thứ hai; trạng thái là `PARTIAL / PENDING ADAPTER DESIGN`, không phải `PASS`.
