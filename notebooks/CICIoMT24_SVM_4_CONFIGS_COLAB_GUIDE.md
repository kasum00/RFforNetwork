# CICIoMT2024 SVM — bốn cấu hình trên Colab

Notebook dùng: [CICIoMT24_SVM_4_CONFIGS_COLAB.ipynb](CICIoMT24_SVM_4_CONFIGS_COLAB.ipynb).

Được tích hợp từ `E:\CICIoMT24_SVM_4_CONFIGS_COLAB_FIXED_V3.ipynb`. Giữ nguyên logic V3 về kiểm SHA256 SSD, chọn probe dự phòng, tính dung lượng khi tái sử dụng, approval và hậu kiểm supervisor. Không thay đổi thuật toán/siêu tham số, dữ liệu hay source huấn luyện.

Thay đổi phục vụ tích hợp: chọn source commit đã duyệt trong checkout **vừa clone mới**, dùng đường dẫn runtime riêng `/content/RFforNetwork_svm4_c93eab87395e`, cập nhật ghi chú notebook. Checkout đã tồn tại không được tự pull/reset/switch. Nếu checkout cũ lệch commit hoặc có sửa đổi, notebook dừng.

## 1. Đẩy đúng ba file lên GitHub từ Windows

Chạy trong PowerShell của VS Code:

```powershell
Set-Location 'D:\COMPACT-MI-RFE'
git status --short
git add -- README.md notebooks/CICIoMT24_SVM_4_CONFIGS_COLAB.ipynb notebooks/CICIoMT24_SVM_4_CONFIGS_COLAB_GUIDE.md
git diff --cached --name-only
git diff --cached --stat
```

Kiểm tra danh sách staged chỉ chứa ba file trên rồi chạy:

```powershell
Set-Location 'D:\COMPACT-MI-RFE'
git commit -m "Add reviewed CICIoMT24 four-config SVM Colab notebook"
git push origin HEAD:main
```

Không dùng `git add .`: repository có các thư mục dữ liệu/audit chưa theo dõi. Lần tích hợp này cũng phát hiện marker merge conflict đã có sẵn trong `.gitignore`; file đó chưa được sửa trong phạm vi tích hợp notebook, vì vậy không dựa riêng vào `.gitignore` để tránh upload dữ liệu. Không thêm dataset, approvals có hiệu lực, log, model, run output hoặc `STAGE73B_GITHUB_INSTRUCTIONS.md` chưa theo dõi vào commit này.

Codex chưa stage/commit/push và chưa chạy bất kỳ test hay training nào.

## 2. Mở Colab sau khi push

[Mở notebook SVM trên Colab](https://colab.research.google.com/github/kasum00/RFforNetwork/blob/main/notebooks/CICIoMT24_SVM_4_CONFIGS_COLAB.ipynb).

Nếu muốn lưu thay đổi riêng, chọn **File → Save a copy in Drive**. Dùng phiên runtime sạch để tránh tái sử dụng module Python đã import từ checkout khác. Chạy từng cell theo tiêu đề, không dùng Run all.

Notebook nằm trên `main`, nhưng source thực hiện huấn luyện được pin tại:

```text
c93eab87395edaf8fd141b29ee9914df066a2e7e
```

Đây là chủ ý: commit bổ sung notebook/hướng dẫn mới không làm đổi danh tính source huấn luyện và unit evidence đã chuẩn bị. Không thay `EXPECTED_COMMIT` thành commit mới chỉ vì vừa push notebook. Nếu thay source khoa học hoặc các module điều phối, cần kiểm tra lại source/tests/evidence và approval cho commit đó.

## 3. Dữ liệu và evidence trên Drive

Theo các giá trị mặc định Cell 1:

```text
/content/drive/MyDrive/WINDOWS_CPLUS/
    window_run_summary.json
    FW_256_FS_8/
    FW_256_FS_16/
    FW_512_FS_8/
    FW_512_FS_16/

/content/drive/MyDrive/CSV_SPLIT_CPLUS/
    split_manifest.csv
    validation.json

/content/drive/MyDrive/COMPACT-MI-RFE/
    environment/stage73b_unit_tests_c93eab87395e.json
    technical_probes/<probe JSON>
```

`CSV_SPLIT_CPLUS` ở đây chỉ cần hai file metadata, không cần chép CSV flow gốc. Có thể chỉnh biến đường dẫn ở Cell 1 nếu Drive của bạn đặt folder ở chỗ khác.

Cell 8 kiểm unit evidence có sẵn, không tự tạo evidence PASS. Nếu thiếu hoặc hash không khớp, dùng Cell 5 của notebook Stage 7.3B trong đúng checkout `c93eab…` để tự chạy các suite và tạo evidence, hoặc thực hiện lại quy trình test/evidence đã thống nhất. Không điền tay `passed=True` để vượt kiểm tra. Probe phải thuộc đúng nhánh/config/source/summary; probe PASS không tự duyệt full run.

## 4. Thứ tự chạy thủ công

1. Cell 1: chọn `EXPERIMENT`, `CONFIG`, `RUN_ID` và kiểm tra đường dẫn. Mặc định là `baseline_svm / FW_512_FS_16`.
2. Cell 2–4: mount Drive, xác minh checkout đã pin và scientific profile. Nếu chủ động cài lại thư viện, restart runtime rồi chạy lại từ đầu.
3. Cell 5–6: kiểm inventory/disk, copy SSD hoặc xác minh mọi SHA256 khi tái sử dụng. Hash/copy đọc byte file để kiểm toàn vẹn; không dùng nội dung Test để chọn mô hình.
4. Cell 7–8: thu run identity và kiểm unit-test evidence.
5. Cell 8A: bỏ qua nếu đã có probe hợp lệ; chỉ bật khi cần tự thực hiện Train-only probe cho cấu hình hiện chọn.
6. Cell 9: chọn exact-config evidence hợp lệ.
7. Cell 10: xem bản draft PENDING và ngân sách đề xuất. Các giá trị 4 giờ/28 GiB không phải dự báo full SVM chắc chắn hoàn thành.
8. Cell 11: chỉ khi bạn đã xem xét và đồng ý run cụ thể, chủ động bật tạo approval, nhập đủ hai câu xác nhận. Notebook mặc định tắt bước này.
9. Cell 12: kiểm authorization read-only. Chỉ tiếp tục khi hàm trả context hợp lệ.
10. Cell 13: chỉ khi bạn quyết định bắt đầu official run, bật cả `START_OFFICIAL_SVM` và `I_ACCEPT_SINGLE_FINAL_TEST_EVALUATION`. Một lần chỉ chạy một experiment/config; supervisor kiểm lại approval trước khi tạo worker.
11. Cell 14: đọc metadata, hash artifact và báo cáo supervisor để xác nhận kết quả. Không dự đoán Test lần nữa.

Muốn thực hiện cả hai nhánh trên bốn cấu hình sẽ có tám official run, mỗi run có identity/approval riêng. Nếu có `.incomplete`, không đổi tên run để tự retry trước khi kiểm tra sự kiện Test và trạng thái validation. Lỗi copy/publish không cho phép đánh giá Test lại.

## 5. Kết quả và bằng chứng cần giữ

```text
COMPACT-MI-RFE/experiment_results/<RUN_NAME>/
COMPACT-MI-RFE/checkpoints/<RUN_NAME>.incomplete/
COMPACT-MI-RFE/reports/<RUN_NAME>_supervisor.json
COMPACT-MI-RFE/reports/<RUN_NAME>_supervisor_events.jsonl
COMPACT-MI-RFE/environment/<RUN_NAME>.json
```

Gửi output Cell 12/14 và `result.json`, `artifact_manifest.json`, supervisor JSON, `preflight_audit.json`, `authorization.json`, `events.jsonl` khi cần kiểm định. Nếu lỗi, giữ `.incomplete`, log và `FAILED.json` nếu có; không tự resume fit/Test.

## 6. Trạng thái kiểm tra của lần tích hợp

- Đã đọc JSON notebook, so sánh source với V3 và đối chiếu giao diện supervisor/authorization của repository.
- Các cờ cài package, technical probe, tạo approval và full training vẫn mặc định `False`.
- Chưa thực thi cell Python, unit test, probe, full SVM hoặc Google Colab/Drive.
- Chưa xác nhận runtime Colab thực tế, dung lượng/RAM hoặc evidence/data trên tài khoản Drive.
- Nhãn là `SCENARIO_PROXY`; capture/session independence vẫn `NOT_ESTABLISHED`.
