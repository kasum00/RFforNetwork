# Stage index — COMPACT-MI-RFE

Mục đích của file này là giúp điều hướng; nó không thay thế report hoặc manifest gốc.

| Nhóm | Thư mục | Nội dung | Trạng thái/ghi chú |
|---|---|---|---|
| 1 | `STAGE1_AUDIT/` | Full CICFlowMeter audit, schema, inventory, leakage risk | Evidence full; giữ nguyên |
| 1 | `STAGE1_AUDIT_SMOKE/` | Smoke audit giới hạn | Evidence smoke; không thay full |
| 1.5 | `STAGE15_REVIEW/` | Corrected audit và split feasibility | Review evidence |
| 2.1 | `STAGE21_SUBSET_PLAN/` | Representative subset plan | Planning evidence |
| 2.2 | `STAGE22_REVIEW/` | Sampling/protocol/leakage review | Review evidence |
| 2.3 | `STAGE23_PREFLIGHT/` | C+ temporal preflight | Gate trước split |
| 2.4 | `STAGE24_CHRONOLOGY/` | Chronology/duration/index checkpoint | Input/evidence |
| 2.5 | `STAGE25_ORDERED_INDEX/` | Chronological replay index | Input/evidence |
| 2.6 | `STAGE26_GROUP_MAPPING/` | Scenario/group mapping | Policy evidence |
| 2.7 | `STAGE27_POLICY/` | C+ policy/readiness/approval template | Policy evidence; approval không tự suy ra |
| 2.8 | `STAGE28_CPLUS_PLAN/` | C+ dry-run/cutoffs/bucket evidence | Không thay manifest chính thức |
| 2.9 V1 | `STAGE29_MATERIALIZATION/` | Guarded materialization V1 | Giữ provenance/checkpoints |
| 2.9 V2 | `STAGE29_MATERIALIZATION_V2/` | V2 allocation preflight | Report ghi `BLOCKED` |
| 2.9 V2 R2 | `STAGE29_MATERIALIZATION_V2_R2/` | R2 preflight + guarded materialization | Report ghi `PASS_FOR_GUARDED_MATERIALIZATION` |
| 3 | `STAGE3_WINDOWS_V2/` | Window artifacts/metadata | Generated artifact; không đổi |
| 4 | `STAGE4_COLAB_PACKAGES/` | Colab package windows/metadata | Generated package; không đổi |
| 6–7 | `STAGE6_REPORT.md`, `huyen_code/STAGE7_*` | Colab/SVM audit, probe, authorization and resumable docs | Liên kết từ README/source; không gom path trong batch này |

## Quy tắc đọc Stage

- Đọc report trước artifact.
- Phân biệt `PASS`, `BLOCKED`, `PENDING_RESEARCHER_APPROVAL`, `NOT_RUN` và `UNVERIFIED`.
- Không dùng artifact/probe làm kết quả chính thức nếu report không claim official result.
- Không xóa Stage blocked hoặc smoke; chúng là provenance của quyết định.
- Không đổi tên Stage directory khi còn checkpoint/path/reference chưa được chứng minh.
