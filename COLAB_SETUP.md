# Chạy COMPACT-MI-RFE trên Google Colab Pro

Stage 6 chỉ chuẩn bị hệ thống. **Không có thí nghiệm chính thức nào được chạy trong Stage 6.** Notebook mặc định `dry_run` và không tải hàng dữ liệu Test để chọn mô hình.

## 1. Việc cần làm trước khi mở notebook

Checkout Git cục bộ tại `D:\COMPACT-MI-RFE` hiện có `origin=https://github.com/kasum00/RFforNetwork.git`, nhưng commit cục bộ đang theo dõi **chỉ `README.md`**. Các thư mục `huyen_code` và `CODE_ctu13` có trên máy nhưng chưa được bảo đảm có trong GitHub. Vì Stage 6 không được phép commit/push, người dùng cần tự đưa các file **source nhỏ** sau lên repository trước khi Colab clone:

- Năm script trong `CODE_ctu13/` (không sửa nội dung).
- `huyen_code/ciciomt24_training.py`, `huyen_code/ciciomt24_sliding_window.py`, `huyen_code/ciciomt24_split.py` và ba file `test_ciciomt24_*.py`.
- `colab/`, `configs/`, `notebooks/`, `requirements-colab.txt`, `COLAB_SETUP.md`, `STAGE6_REPORT.md`, `.gitignore`.

**Không** đưa `CICIoMT24/`, `huyen_code/runs/`, model cũ, ZIP, file CSV nhiều triệu dòng hay thông tin tài khoản lên GitHub. `.gitignore` mới chặn các đường dẫn dữ liệu lớn, nhưng hãy xem `git status`/danh sách staged trước khi tự commit. Notebook sẽ dừng tại Cell 4 nếu clone thiếu file, không tiếp tục một pipeline nửa vời.

Ví dụ lệnh PowerShell **để bạn tự thực hiện**, sau khi đã xem nội dung sẽ công khai:

```powershell
Set-Location 'D:\COMPACT-MI-RFE'
git add -- .gitignore CODE_ctu13 colab configs notebooks requirements-colab.txt COLAB_SETUP.md STAGE6_REPORT.md
git add -- huyen_code/ciciomt24_training.py huyen_code/ciciomt24_sliding_window.py huyen_code/ciciomt24_split.py huyen_code/test_ciciomt24_training.py huyen_code/test_ciciomt24_split.py huyen_code/test_ciciomt24_sliding_window.py
git diff --cached --name-only
git diff --cached --stat
# Chỉ khi danh sách staged đúng và bạn muốn công bố:
git commit -m 'Prepare COMPACT-MI-RFE Colab Stage 6'
git push origin main
```

Sau khi các file có mặt trên nhánh `main`, mở trực tiếp:

<https://colab.research.google.com/github/kasum00/RFforNetwork/blob/main/notebooks/COMPACT_MI_RFE_ColabPro.ipynb>

Hoặc Colab → File → Upload notebook và chọn `D:\COMPACT-MI-RFE\notebooks\COMPACT_MI_RFE_ColabPro.ipynb`; dù upload notebook, Cell 4 vẫn cần clone repository chứa source đầy đủ.

## 2. Chạy tuần tự 12 cell

1. Cell 1 hiển thị OS, Python, CPU, RAM available, disk, GPU probe và package versions **thực tế**. CPU/RAM lớn hoặc GPU không được giả định từ gói Colab Pro.
2. Cell 2 mount Drive; bạn tự phê duyệt màn hình Google.
3. Cell 3: nếu ZIP hoặc thư mục `WINDOWS_CPLUS` đã có qua shortcut trong MyDrive, điền `DRIVE_DATASET_PATH` là đường dẫn mounted. Nếu không, để trống để Cell 6 kiểm tra folder ID chia sẻ `1Ixy5iJdeavkEbFJ3cy6LjkQ8h300OAxh` bằng Drive API sau khi bạn đồng ý quyền đọc. Không giả định folder chia sẻ nằm trong MyDrive. Nếu có nhiều ZIP, điền `DRIVE_ARCHIVE_ID`; nếu chỉ có một ZIP, nó được chọn sau khi liệt kê.
4. Cell 4 clone vào `/content/RFforNetwork`. Checkout đã tồn tại sẽ **không** được pull/reset, để tránh mất local edits.
5. Cell 5 kiểm tra Python 3.10–3.12, cài phiên bản khoa học của Stage 5 chỉ khi thiếu/khác. Nếu vừa cài lại thư viện khoa học, cell chủ động dừng: restart runtime rồi chạy lại từ Cell 1 để không trộn phiên bản đã nạp và vừa cài. Sau đó cell chạy `pip check` và unit test. Nếu Colab đổi Python ngoài khoảng được chuẩn bị hoặc `pip check`/test FAIL, dừng; không tự downgrade Python.
6. Cell 6 kiểm tra tên/cấu trúc ZIP, dung lượng, Zip Slip, duplicate path và CRC; hoặc tái dùng bản local đã hợp lệ. Dataset được chuẩn bị ở `/content/datasets/CICIoMT24/WINDOWS_CPLUS` bằng thư mục `.incomplete`, kiểm định xong mới promote. Metadata C+ nhỏ được đặt ở thư mục sibling `CSV_SPLIT_CPLUS` từ `configs/cplus_manifest`; **không** sao chép toàn bộ flow CSV. ZIP/shortcut gốc trên Drive không bị xóa/di chuyển. Nếu `.incomplete` còn sót, dừng để kiểm tra, không merge/ghi đè tự động.
7. Cell 7 đối chiếu 322 file, 6.737.006 window theo 12 ô, 380 feature + 14 metadata/label, proxy label. Preflight chỉ kiểm tra Train/Validation headers; hàng Test và Test header được giữ đến sau refit. Full readback Stage 4.2 được tái sử dụng, không tuyên bố quét lại toàn bộ CSV ở Colab preflight.
8. Cell 8 chọn một thí nghiệm và một FW/FS. Mặc định `mi_rfe_rf`, `FW_512_FS_8`, `dry_run`.
9. Cell 9 kiểm tra lại manifest/schema, source SHA256, unit-test evidence, phiên bản package, RAM available, disk local, Drive mount và tên run chưa tồn tại. RF FS8 dùng đúng guard Stage 5.4: peak FS16 đo được × tỷ lệ hàng lớn nhất, cộng `max(2 GiB, 20%)`; FW512/FS8 cần **25.762.976.100 byte RAM available** theo dữ liệu hiện tại, không phải cam kết peak thật. Nếu FAIL, không chạy full.
10. Cell 10 chỉ chạy full khi bạn đổi `MODE=full_train`, Cell 9 PASS và nhập chính xác `RUN <experiment> <config> <run_name>` vào `CONFIRMATION`. Sau đó Cell 10 gọi đúng **một** run. Hai RF FS16 cũ luôn bị chặn không chạy lại trong notebook này. SVM cần benchmark kỹ thuật giới hạn (bật `RUN_TECHNICAL_SVM_PROBE=True`) và admission PASS; sau benchmark phải **xem kết quả, tắt toggle, chạy lại Cell 9 rồi mới chủ động chạy full**. Benchmark không phải kết quả khoa học chính thức.
11. Cell 11 xác minh bản công bố trên Drive. Nếu một run **đã hoàn thành `run_validated`** nhưng Colab ngắt đúng lúc công bố, bật `RECOVER_COMPLETED_CHECKPOINT=True` để chỉ promote checkpoint đủ artifacts; **không fit/predict/evaluate Test lại**. Checkpoint đang dở không thể resume.
12. Cell 12 chỉ đọc `result.json` hoàn chỉnh để lập bảng Accuracy, Macro-F1, Balanced Accuracy, Precision/Recall/F1 từng lớp, số feature, thời gian, peak RAM và confusion matrix. Không gọi model hoặc Test data.

## 3. Quy tắc khoa học và bảo vệ kết quả

`ciciomt24_training.py` và năm script CTU-13 **không bị sửa**. Adapter Colab tái sử dụng chính các hàm `select_features`, `fit_final_classifier`, `evaluate_final` và runner Stage 5. Nó chỉ truyền đường dẫn manifest Linux, chụp checkpoint sau MI/RFE/từng k/feature freeze/refit, và serialize model đã refit **trước khi runner bắt đầu đọc Test**. MI/RFE và scaler lựa chọn chỉ fit Train; k=1..100 chọn Validation Accuracy lớn nhất, hòa giữ k nhỏ; scaler/model cuối fit Train+Validation với feature đã khóa; Test chỉ evaluate một lần.

Stage 5.2 FW512/FS16 (k=8) và Stage 5.3 FW256/FS16 (k=18) đã hoàn thành trên Windows. Nếu result/model cũ chưa có trong Drive/GitHub, Cell 12 ghi `PREVIOUS_ARTIFACTS_MISSING`; notebook **không tạo giả hoặc chạy lại** hai kết quả đó. Có thể chép artifacts cũ lên Drive thủ công nếu cần bảng so sánh, nhưng đừng sửa `result.json` gốc.

Checkpoint được ghi vào `MyDrive/COMPACT-MI-RFE/checkpoints/<run>.incomplete`: MI scores/Top100, RFE ranking, Validation metrics từng k, frozen selection, model+scaler sau refit, và final result/log sau Test. File được ghi qua staging và kiểm SHA256 cho bản sao; tính nguyên tử của rename trên Drive mount chưa được xác minh thực tế. **Không có resume fit/rerun từ k giữa chừng**; reset runtime trước `run_validated` giữ bằng chứng tiến độ nhưng run không hợp lệ, cần quyết định mới và tên run mới. Bản hoàn chỉnh được publish ở `experiment_results/<run>`; `environment/<run>.json` lưu môi trường. Matrix memmap nhiều GiB nằm ở runtime local, không copy sang Drive; có thể tái tạo từ dataset qua một run mới, nhưng đó là thí nghiệm mới, không phải resume.

Nhãn là `SCENARIO_PROXY`, Benign hiếm, window trong một split chồng lấn và độc lập capture/session chưa được chứng minh. Không chỉ trích Accuracy; cần trình bày Recall Benign, Macro-F1, Balanced Accuracy và confusion matrix.

## 4. Nếu một bước FAIL

- Thiếu file ở Cell 4: kiểm tra source đã thực sự được bạn commit/push lên GitHub. Không thay URL hay tự tải code khác.
- Không thấy Drive folder: bạn cấp quyền đọc cho đúng tài khoản trong notebook, hoặc thêm shortcut folder chia sẻ vào MyDrive rồi điền đường dẫn mounted. Folder này không cần quyền ghi; output ở MyDrive cá nhân.
- Thiếu disk: không giải nén; dùng runtime có đủ disk. Không xóa ZIP gốc, không giảm dataset.
- RAM guard FAIL: dừng, chọn runtime có RAM available cao hơn. GPU không giúp scikit-learn CPU hiện tại.
- Checkpoint `.incomplete`: kiểm tra `checkpoint.json`, `events.jsonl`, `FAILED.json` local nếu còn. Chỉ checkpoint `run_validated` được phép promote không training; checkpoint MI/RFE/k dở chỉ phục vụ audit.

Không tự chạy full training, không commit/push và không thao tác tài khoản Google trong Stage 6 tại máy cục bộ.
