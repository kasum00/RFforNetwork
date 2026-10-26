# Repository audit — COMPACT-MI-RFE

Ngày audit: 2026-10-26  
Phạm vi: kiểm kê tĩnh repository `D:\COMPACT-MI-RFE`, không đọc toàn bộ dữ liệu lớn.

## Kết luận điều hành

Repository hiện có một pipeline CICIoMT2024 C+ đang được bảo vệ bởi nhiều lớp evidence, manifest, hash và checkpoint. Cấu trúc hiện tại chưa tách hẳn `source`, `dataset-specific code`, `research evidence` và `generated artifacts`, nhưng việc tách bằng cách đổi tên hoặc di chuyển hàng loạt sẽ có rủi ro vì code và notebook đang dùng đường dẫn cố định.

Đợt audit này chọn chiến lược an toàn:

- Giữ nguyên mọi đường dẫn dữ liệu, manifest, window, model, checkpoint và Stage evidence.
- Không coi các Stage cũ là rác chỉ vì tên hoặc số Stage.
- Tạo tài liệu điều hướng và kế hoạch migration có thể hoàn tác.
- Chỉ coi `.venv_stage3/` là môi trường cục bộ; thêm rule ignore riêng, không xóa.

## Cơ sở kiểm kê

Đã thực hiện kiểm tra tĩnh bằng `Get-ChildItem`, `git status`, `git diff --stat/name-status`, `git ls-files`, `git check-ignore` và `rg`. Không chạy Python, unit test, smoke test, preflight khoa học, dataset scan, benchmark, training hoặc full pipeline.

Worktree đã dirty trước phần cleanup: các thay đổi ở `README.md`, nhóm module `colab/`, nhóm tài liệu Stage 7.3B và `huyen_code/ciciomt24_training.py` được giữ nguyên, không gộp vào cleanup.

## Bản đồ thư mục hiện tại

Số liệu dưới đây là inventory theo file/size, không phải readback hay xác minh nội dung khoa học.

| Đường dẫn | Inventory quan sát | Vai trò phân loại | Quyết định |
|---|---:|---|---|
| `CODE_ctu13/` | 5 file, khoảng 37 KiB | D — mã CTU-13 gốc | Bảo vệ tuyệt đối; nhiều report và guard dùng SHA256 của 5 file này. |
| `code_CICFLOW/` | 47 file trong cây hiện tại, gồm script Stage 2.x–4 và `paper_repro/` | B — dataset-specific/gate code; `paper_repro/` là nhánh tái hiện bốn cấu hình | Giữ nguyên; phân loại riêng trong dependency map. |
| `huyen_code/` | 326 file, khoảng 5.56 GiB, phần lớn ở `runs/` | A/B/C/E — core khoa học CICIoMT2024, test, report và run artifacts | Không di chuyển; code và report có path/hash phụ thuộc. |
| `colab/` | 37 file quan sát, gồm adapter, authorization, supervisor, checkpoint và test | A một phần, B/E một phần — execution framework còn ràng buộc CICIoMT2024 | Không đổi path trong đợt này. |
| `local/` | 16 file | A/B — Windows continuation wrapper, nhưng kiểm tra hash/path của source hiện tại | Giữ nguyên; cần compatibility shim trước khi tách dùng chung. |
| `configs/` | 4 file, gồm manifest C+ được version-control và template authorization | C — scientific configuration/evidence | Không đổi nội dung manifest hoặc fingerprint. |
| `notebooks/` | 5 notebook/guide | B/E — entrypoint Colab cho pipeline hiện tại | Giữ nguyên các path `colab/`, `huyen_code/`, `configs/`. |
| `CICIoMT24/` | 579 file, khoảng 51.87 GiB | E — CSV nguồn, split C+, windows và partial artifacts | Bảo vệ tuyệt đối; không đọc/sửa/di chuyển. |
| `CICFLOWMETER_FEATURE_CSV/` | 4,047 file, khoảng 134.71 GiB | E — raw/processed feature data; vai trò từng nhánh chưa được suy luận từ tên | Không động vào; nội dung lớn chỉ inventory theo metadata. |
| `.venv_stage3/` | 11,711 file, khoảng 0.39 GiB | F — local Python environment | Không xóa; đã thêm ignore `.venv_stage*/`. |

`.git/` được quan sát qua Git metadata nhưng không chỉnh sửa. Không dùng `git reset`, `clean`, `checkout`, commit, push hoặc pull.

## Stage evidence và artifact

| Thư mục | Inventory | Mục đích thực tế đọc từ report/tên file | Phân loại |
|---|---:|---|---|
| `STAGE1_AUDIT/` | 9 file, khoảng 0.05 GiB | Full audit CICFlowMeter, schema, inventory, leakage risk và checkpoint SQLite | C/E — evidence quan trọng |
| `STAGE1_AUDIT_SMOKE/` | 7 file | Smoke audit giới hạn, khác phạm vi full audit | C/E — không trùng vai trò với Stage 1 full |
| `STAGE15_REVIEW/` | 6 file | Corrected audit và split feasibility | C — evidence |
| `STAGE21_SUBSET_PLAN/` | 5 file | Kế hoạch representative subset | C — planning evidence |
| `STAGE22_REVIEW/` | 4 file | Review sampling/protocol và leakage | C — review evidence |
| `STAGE23_PREFLIGHT/` | 6 file | C+ temporal preflight, chưa phải split | C — gate evidence |
| `STAGE24_CHRONOLOGY/` | 270 file, khoảng 0.03 GiB | Chronology/duration index, SQLite và checkpoint theo source | C/E — resume evidence |
| `STAGE25_ORDERED_INDEX/` | 410 file, khoảng 0.27 GiB | Ordered index, binary/CSV/JSON source artifacts | C/E — input của downstream gate |
| `STAGE26_GROUP_MAPPING/` | 6 file | Scenario/group mapping và policy review | C — policy evidence |
| `STAGE27_POLICY/` | 5 file | Approval/readiness policy, template reviewer | C — approval evidence |
| `STAGE28_CPLUS_PLAN/` | 75 file, khoảng 0.07 GiB | C+ dry-run, cutoffs, bucket summary, checkpoint | C/E — không thay manifest chính thức |
| `STAGE29_MATERIALIZATION/` | 867 file, khoảng 10.99 GiB | V1 guarded materialization, membership/provenance/checkpoint | C/E — không xóa để “dọn” |
| `STAGE29_MATERIALIZATION_V2/` | 651 file, khoảng 9.04 GiB | V2 allocation preflight; report ghi `BLOCKED` | C/E — giữ làm evidence của nhánh blocked |
| `STAGE29_MATERIALIZATION_V2_R2/` | 638 file, khoảng 7.86 GiB | V2 R2 preflight và guarded materialization; report ghi `PASS_FOR_GUARDED_MATERIALIZATION` | C/E — evidence/artifact chính của nhánh R2 |
| `STAGE3_WINDOWS_V2/` | 933 file, khoảng 3.96 GiB | Window artifacts và metadata theo FW/FS | E/C — không đổi path |
| `STAGE4_COLAB_PACKAGES/` | 17 file, khoảng 0.64 GiB | Package/metadata cho Colab | E/C — notebook/package dependency |

Các STAGE 2.4, 2.5, 2.9 và Stage 3/4 có file SQLite, binary, Parquet, CSV và checkpoint. Không có bằng chứng an toàn để hợp nhất V1/V2/R2 hoặc xóa nhánh blocked.

## Phân loại A–F

### A — Core reusable code

Các thành phần có thể tái sử dụng một phần cho dataset khác:

- `colab/checkpoint_manager.py`, `colab/svm_resumable_store.py`, `colab/svm_resumable_execution.py` và các module authorization/supervisor.
- `local/` ở vai trò execution/control plane.
- `code_CICFLOW/paper_repro/` ở vai trò runner, validator, checkpoint và notebook package cho bốn cấu hình.
- Một số utility trong `huyen_code/ciciomt24_training.py`, nhưng file hiện gắn chặt với feature contract và dataset manifest nên chưa được coi là core thuần.

### B — Dataset-specific code

- `huyen_code/ciciomt24_split.py`, `ciciomt24_sliding_window.py`, các runner/probe có tiền tố `ciciomt24_`.
- `code_CICFLOW/stage*.py`, `stage*.ps1` cho chronology, C+, allocation, materialization và Colab package.
- `colab/dataset_validator.py`, các notebook và config gắn `CICIoMT24`, `WINDOWS_CPLUS`, `CSV_SPLIT_CPLUS`.

### C — Research evidence

- `STAGE*` reports, manifest, JSON validation, hash/index/checkpoint và các report ở `huyen_code/`.
- `configs/cplus_manifest/` và các identity/source hash được code/approval tham chiếu.
- `README.md` và các guide công bố giới hạn khoa học; không coi chúng là output runtime.

### D — Legacy code

- Toàn bộ `CODE_ctu13/` là source đối chiếu gốc, không phải code cần refactor.
- Các Stage script cũ hoặc nhánh thử nghiệm chỉ được gắn legacy sau khi đối chiếu reference/hash; hiện chưa di chuyển.

### E — Generated artifacts

- `CICIoMT24/`, `CICFLOWMETER_FEATURE_CSV/`, `huyen_code/runs/`, `STAGE29_*`, `STAGE3_WINDOWS_V2/`, `STAGE4_COLAB_PACKAGES/` và các checkpoint/partial directory.
- Một số artifact có thể là kết quả chính thức, một số là probe hoặc blocked attempt; tên thư mục chưa đủ để phân biệt.

### F — Temporary/duplicate candidates

- `.venv_stage3/`, `__pycache__/` và notebook cache là generated local files; chỉ `.venv_stage*` được ignore trong đợt này, không xóa file.
- `STAGE1_AUDIT` và `STAGE1_AUDIT_SMOKE` không phải duplicate vì một bên full, một bên smoke.
- `STAGE29_MATERIALIZATION`, `V2`, `V2_R2` không được coi là duplicate: report cho thấy trạng thái/gate khác nhau.
- Đã tìm thấy một cặp artifact JSON byte-identical trong phạm vi file nhỏ được hash: `huyen_code/runs/duration_review_stage31_02/review.json` và `huyen_code/runs/duration_review_stage31_03/review.json`, SHA256 `FC2DA71E29D3037EA2EDE6DBA0E9269E0C72EA838E267C69067F00FC6ABBC786`. Đây chỉ là duplicate candidate; chưa xóa/merge vì tên run và provenance có thể vẫn cần giữ.
- `code_CICFLOW/paper_repro/colab/paper_repro_checkpoint.py` và `paper_repro_runner.py` là import shim trỏ về module cha, không phải bản sao độc lập.
- `huyen_code/stage15_review.ps1`, `stage15_review_fixed.ps1`, `stage15_review_v2.ps1` có khác biệt nội dung/lineage; chưa đủ cơ sở gộp.
- Không xác nhận file khoa học nào là duplicate chỉ dựa trên tên hoặc cùng phần mở rộng.

Phép hash duplicate chỉ áp dụng cho file source/document nhỏ dưới 20 MB trong `CODE_ctu13/`, `code_CICFLOW/`, `colab/`, `huyen_code/`, `local/`, `notebooks/` và `configs/`; không quét hoặc băm lại dataset/artifact lớn.

## Những mục tuyệt đối không động vào

`CICIoMT24/CSV`, `CICIoMT24/CSV_SPLIT_CPLUS`, `CICIoMT24/WINDOWS_CPLUS`, dataset materialized, manifest/fingerprint, model/checkpoint, official artifacts, `CODE_ctu13/` và mọi đường dẫn đang được checkpoint/resume hoặc source-hash guard tham chiếu.

## UNVERIFIED

- Không đọc toàn bộ nội dung CSV/Parquet/SQLite lớn và không băm lại dataset.
- Chưa chạy import/runtime, unit test, preflight hoặc notebook.
- Chưa chứng minh toàn bộ references động trong notebook/Colab/Drive.
- Chưa chứng minh dataset thứ hai tương thích với feature contract 380 cột.
- Chưa phân biệt chắc chắn từng nhánh con trong `CICFLOWMETER_FEATURE_CSV/`; chỉ inventory metadata.
