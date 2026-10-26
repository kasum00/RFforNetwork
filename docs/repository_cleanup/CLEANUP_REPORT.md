# Cleanup report — COMPACT-MI-RFE

Ngày: 2026-10-26  
Phạm vi: cleanup an toàn, không thay đổi scientific pipeline.

## Đã thực hiện

1. Kiểm kê root, thư mục Stage, code, configs, notebooks, data/artifact root và Git worktree.
2. Đọc report/headings của các Stage audit, C+, chronology, ordered index, policy và materialization.
3. Truy vết tĩnh các import, `Path(__file__)`, `sys.path`, path dataset, manifest, checkpoint và Colab references.
4. Tạo bộ tài liệu điều hướng:
   - `docs/repository_cleanup/REPOSITORY_AUDIT.md`
   - `docs/repository_cleanup/DEPENDENCY_MAP.md`
   - `docs/repository_cleanup/CLEANUP_PLAN.md`
   - `docs/repository_cleanup/CLEANUP_REPORT.md`
   - `docs/repository_cleanup/SECOND_DATASET_READINESS.md`
   - `docs/repository_cleanup/STAGE_INDEX.md`
5. Cập nhật `README.md` bằng liên kết tới bộ audit này.
6. Thêm `.venv_stage*/` vào `.gitignore` để tránh add nhầm môi trường local; không xóa môi trường và không untrack file nào.

## Không thực hiện

- Không xóa, đổi tên hoặc di chuyển file/thư mục.
- Không sửa CSV, Parquet, manifest, fingerprint, model, checkpoint, run artifact hoặc CTU-13.
- Không refactor `huyen_code/`, `colab/`, `local/` hoặc `code_CICFLOW/`.
- Không thay đổi thuật toán MI/RFE/SVM/RF, split C+, embargo, window contract, hyperparameter hoặc kết quả.
- Không chạy Python, test, preflight, scan dữ liệu, benchmark, training, Git mutation, package install, commit/push/pull.

## Trạng thái worktree

Các thay đổi trước cleanup được giữ nguyên. Tại thời điểm audit, worktree có các file modified trong `README.md`, nhóm `colab/`, nhóm report/guide Stage 7.3B và `huyen_code/ciciomt24_training.py`, cùng nhiều file untracked Stage/code. Cleanup không reset, stash, clean hoặc overwrite chúng.

## Kiểm tra tĩnh đã dùng

- Inventory root và subtree với `Get-ChildItem`.
- `git status --short`, `git diff --stat`, `git diff --name-status`, `git ls-files`.
- `rg` cho imports, paths, `sys.path`, manifest, checkpoint, Colab và Stage references.
- `git check-ignore` cho dataset roots, manifest và environment.
- Sau khi patch: kiểm tra tồn tại tài liệu, diff scope và `git diff --check`.

Đây chỉ là kiểm tra tĩnh. Không được diễn giải là `PASS` cho pipeline hoặc runtime.

## Chờ phê duyệt/validation

- Có di chuyển report Stage vào `docs/` hay giữ nguyên path lịch sử.
- Có tạo `src/` compatibility layer và tách dataset adapter hay không.
- Có chuẩn hóa artifact/checkpoint root hay không.
- Có thay đổi `.gitignore` cho các Stage artifact lớn hay không; hiện chưa thay đổi vì chúng có thể là evidence.
- Unit test/import/runtime validation sau bất kỳ path migration nào.

