# Dependency map — COMPACT-MI-RFE

Ngày audit: 2026-10-26. Đây là mapping tĩnh; chưa có xác minh runtime sau di chuyển vì đợt này không di chuyển source/runtime.

## Luồng module hiện tại

```text
CODE_ctu13 (source đối chiếu/hash)
        │
        ├── ciciomt24_split.py ──> C+ manifest / segments / validation
        │                              │
        ├── ciciomt24_sliding_window.py ──> WINDOWS_CPLUS / window summary
        │                              │
        └── ciciomt24_training.py <── colab adapters / local continuation
                                      │
                                      ├── MI → RF-RFE → SVM/RF
                                      ├── checkpoint + authorization
                                      └── late Test evaluation

code_CICFLOW/stage*.py, *.ps1 ──> STAGE15..29 evidence/artifacts
code_CICFLOW/paper_repro ────────> four-config paper reproduction package
notebooks ───────────────────────> Colab entrypoints, import colab + huyen_code
```

## Python import/path contracts

| Consumer | Dependency/path contract | Rủi ro nếu di chuyển |
|---|---|---|
| `huyen_code/test_ciciomt24_svm_probe.py`, `test_ciciomt24_svm_scaling_probe.py` | Import `colab.experiment_launcher`; thêm repository root vào `sys.path` | Đổi tên `colab/` hoặc chạy từ root khác làm hỏng import. |
| `local/local_authorization.py`, `local/progress_console.py` | Import `colab.svm_resumable_*` và module tương đối trong `local/` | Tách `local/` khỏi `colab/` mà không có shim sẽ hỏng authorization/checkpoint. |
| `local/checkpoint_migration.py`, `local/windows_preflight.py` | Kiểm tra đúng chuỗi path như `huyen_code/ciciomt24_training.py`, `huyen_code/ciciomt24_sliding_window.py`, nhóm `colab/*.py` | Di chuyển làm mismatch hash/path và chặn resume. |
| `huyen_code/ciciomt24_split.py` | Default `D:\COMPACT-MI-RFE\CICIoMT24`; output mặc định trong `huyen_code/runs/`; tham chiếu `CODE_ctu13/` | Đổi root/code path làm thay đổi default và khả năng reproduce. |
| `huyen_code/ciciomt24_sliding_window.py` | Default `CICIoMT24/CSV_SPLIT_CPLUS`; tham chiếu `CODE_ctu13/3_sliding_windown_and_aggregate_features.py` | Đổi tên manifest/window/source gốc có thể làm fail scientific guard. |
| `huyen_code/ciciomt24_training.py` | Repo root theo `Path(__file__).resolve().parents[1]`; load `WINDOWS_CPLUS`, `CSV_SPLIT_CPLUS`, feature contract và checkpoint | Đây là code chính thức; refactor chỉ sau hash/test/approval. |
| `colab/dataset_validator.py` | Linux path `/content/datasets/CICIoMT24/WINDOWS_CPLUS`; chỉ nạp `WINDOWS_CPLUS`, không tải `CSV_SPLIT_CPLUS` | Đổi tên dataset hoặc folder Drive làm preflight fail hoặc tải nhầm artifact. |
| `colab/drive_manager.py` | Drive root `/content/drive/MyDrive/COMPACT-MI-RFE`; folder names `WINDOWS_CPLUS`, `CSV_SPLIT_CPLUS` | Đổi layout Drive có thể phá resumable transfer và folder discovery. |
| `code_CICFLOW/stage3_v2_windows.py` | Reference `CODE_ctu13/3_sliding_windown_and_aggregate_features.py` | Không đổi CTU-13 hoặc reference path. |
| `CODE_ctu13/*.py` | Có absolute path Linux của dataset CTU-13 gốc | Đây là legacy/reference, không dùng làm code đa dataset. |

## Config, manifest và notebook contracts

| Thành phần | Tham chiếu quan sát | Trạng thái |
|---|---|---|
| `configs/cplus_manifest/split_manifest.csv` | Được ignore exception và được notebook/validator dùng làm manifest portable | Bảo vệ nội dung và path tương đối. |
| `configs/cplus_manifest/validation.json` | Scientific validation/fingerprint companion | Bảo vệ; không tái sinh. |
| `configs/stage73b/authorization_template.BLOCKED.json` | Template approval cho Stage 7.3B | Giữ tên/schema; không biến BLOCKED thành approval. |
| `configs/colab_experiments.json` | Config experiment Colab | Chỉ sửa khi chứng minh không đổi hyperparameter/identity. |
| `notebooks/*.ipynb` | Dùng `colab/`, `huyen_code/`, `configs/`, `requirements-colab.txt` và một số `/content/...` | Đổi path cần sửa đồng thời notebook + guide + test static. |
| `code_CICFLOW/paper_repro/PAPER_REPRO_SPEC.json` | Bốn cấu hình paper reproduction, checkpoint/runner/validator của subtree | Giữ subtree nguyên khối cho đến khi có adapter mới. |

## Checkpoint/resume contract

Các guide và module hiện mô tả các stage `data_prepared`, `mi`, `rfe`, `k_search/k_001..k_100`, `selection`, `final`, `final_test`. Checkpoint có thể chứa đường dẫn tuyệt đối hoặc relative tới `WINDOWS_CPLUS`, manifest, source hash và run root. Vì vậy:

- Không đổi tên hoặc di chuyển `huyen_code/runs/`, `CICIoMT24/WINDOWS_CPLUS/`, `CSV_SPLIT_CPLUS/`, approval, model hoặc checkpoint.
- Không gom `STAGE29_*` vào một folder duy nhất; V1/V2/V2_R2 có gate và provenance riêng.
- Không đổi `huyen_code`/`colab` trước khi có compatibility layer và kiểm tra identity.
- `.venv_stage3/` không phải checkpoint; chỉ được ignore, không xóa.

## Colab/GitHub contracts

Notebook `STAGE7_3B_OFFICIAL_SVM.ipynb` và các guide yêu cầu clone/checkout source cụ thể, hash source, hash manifest/window summary và hash 5 script CTU-13. `colab/svm_official_readiness.py`, `svm_run_authorization.py`, `local/windows_preflight.py` đều có guard tương ứng. Vì vậy mọi migration source phải cập nhật và tái duyệt identity; không sửa hash để ép PASS.

## Safe migration boundary

Trong đợt hiện tại chỉ tạo docs/index và ignore environment. Các thay đổi sau cần một đợt riêng và validation của người dùng:

1. Tách `src/common` khỏi `huyen_code`/`colab`.
2. Đổi tên package hoặc Stage directory.
3. Di chuyển report đang được notebook/guide tham chiếu.
4. Chuẩn hóa dataset root thành `datasets/<id>`.
5. Đưa artifact vào `artifacts/` hoặc đổi checkpoint root.

