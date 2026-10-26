# Cleanup plan — COMPACT-MI-RFE

## Nguyên tắc

Migration phải tăng dần, có thể hoàn tác, giữ backward compatibility và không làm thay đổi scientific configuration. Mọi bước liên quan đến source hash, manifest, checkpoint, model hoặc artifact chính thức phải có validation riêng trước khi coi là hoàn tất.

## Hiện trạng

```text
COMPACT-MI-RFE/
├── CODE_ctu13/                  # legacy/reference, hash-protected
├── code_CICFLOW/                # Stage 2.x–4 + paper_repro subtree
├── huyen_code/                  # CICIoMT24 split/window/training/tests/reports/runs
├── colab/                       # Colab adapter, authorization, resumable execution
├── local/                       # Windows local continuation/control plane
├── configs/                     # C+ manifest, validation, experiment/approval templates
├── notebooks/                   # Colab entrypoints
├── CICIoMT24/                   # protected data/split/windows/artifacts
├── CICFLOWMETER_FEATURE_CSV/     # large feature data, role not fully verified
├── STAGE*/                      # evidence and generated intermediate/final artifacts
└── root docs/configs
```

## Kiến trúc mục tiêu

Đây là target architecture, chưa phải lệnh di chuyển:

```text
COMPACT-MI-RFE/
├── src/
│   ├── common/                  # logging, hashing, schemas, checkpoint primitives
│   ├── preprocessing/           # dataset adapters, split/window interfaces
│   ├── feature_selection/      # MI/RFE contracts, dataset-agnostic APIs
│   ├── models/                  # SVM/RF wrappers, no dataset paths
│   ├── evaluation/              # metrics/reporting, late-Test guard
│   └── pipelines/               # orchestration with explicit config
├── datasets/
│   ├── ciciomt2024/             # adapter/config/readme only; data stays external
│   └── README.md
├── experiments/                 # small, versioned experiment definitions
├── configs/                     # manifests and scientific configs
├── notebooks/                   # thin entrypoints
├── docs/
│   ├── pipeline/
│   ├── audits/
│   ├── experiments/
│   ├── repository_cleanup/
│   └── archive/
├── artifacts/                   # documented external/local artifact roots
├── legacy/                      # only after hash/reference audit
└── README.md
```

## Các bước migration

| Bước | Nội dung | Quyền thực hiện hiện tại | Rủi ro |
|---|---|---|---|
| 0 | Freeze identity: commit/hash/source manifest/checkpoint root; ghi worktree state | Chưa thực hiện trong cleanup vì worktree đang dirty | Cao nếu dùng nhầm snapshot |
| 1 | Tạo audit, dependency map, stage index, readiness và report | Đã thực hiện | Thấp |
| 2 | Ignore environment local `.venv_stage*`, giữ file tại chỗ | Đã thực hiện | Thấp; không thay runtime |
| 3 | Thêm compatibility wrappers để `src/` gọi implementation cũ | Chờ phê duyệt + tests | Cao; có thể đổi import/hash |
| 4 | Tách dataset adapter khỏi model/selection/evaluation | Chờ phê duyệt + kiểm thử synthetic/static | Cao; dễ đổi feature contract |
| 5 | Đưa reports vào `docs/` theo từng batch, cập nhật mọi link trước mỗi batch | Chưa thực hiện | Trung bình; path evidence có thể đổi |
| 6 | Chuẩn hóa artifact root bằng config mới, không di chuyển checkpoint hiện có | Chờ phê duyệt riêng | Rất cao; ảnh hưởng resume/fingerprint |
| 7 | Đưa CTU-13 vào `legacy/` chỉ bằng shim giữ path/hash | Chờ phê duyệt riêng | Rất cao; guard hiện dùng path cũ |

## Batch được phép trong đợt này

- Tài liệu audit và index tại `docs/repository_cleanup/`.
- Liên kết điều hướng trong `README.md`.
- Ignore riêng cho `.venv_stage*`.

Không có code/runtime/data/checkpoint move trong batch này.

## Thứ tự kiểm tra cho batch tương lai

1. `rg` toàn repository để lập danh sách reference cũ/mới.
2. Kiểm tra source hash, manifest/fingerprint, checkpoint binding và notebook package list.
3. Tạo compatibility shim giữ path/API cũ.
4. Chạy static checks và unit tests do người dùng chủ động cho phép.
5. Chỉ sau khi evidence mới khớp mới đổi default/config.
6. Giữ alias hoặc migration note trong ít nhất một chu kỳ thí nghiệm.

