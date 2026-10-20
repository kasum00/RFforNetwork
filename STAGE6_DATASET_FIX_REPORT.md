# FIX Stage 6 — Google Colab và dataset folder trên Drive

Ngày 20/10/2026. **Trạng thái: đã sửa và kiểm thử cục bộ; chưa chạy Cell 1–9 trên Google Colab thật. Không có full training.**

## Nguyên nhân hai lỗi

1. Cell 6 cũ liệt kê folder ID nhưng chỉ lọc ZIP. Folder người dùng mô tả có `WINDOWS_CPLUS/` và `CSV_SPLIT_CPLUS/`, nên nhánh cũ không chọn được dataset. Không có yêu cầu nén lại hoặc upload lại.
2. Cell 5 cũ chặn mọi Python ngoài 3.10–3.12 trước khi xét thư viện. NumPy 1.26.4 của Stage 5 không có wheel CPython 3.13, trong khi NumPy 2.2.6 có. Cell nay chọn profile theo Python và chặn training nếu cài đặt/`pip check`/unit tests thất bại. Sự khác biệt NumPy 2.x là rủi ro tái lập số học, không phải thay đổi thuật toán. Tham khảo [NumPy 1.26.4](https://pypi.org/project/numpy/1.26.4/), [NumPy 2.2.6](https://pypi.org/project/numpy/2.2.6/), [SciPy 1.15.3](https://pypi.org/project/scipy/1.15.3/), [scikit-learn 1.7.2](https://pypi.org/project/scikit-learn/1.7.2/) và [pandas 2.3.3](https://pypi.org/project/pandas/2.3.3/). Việc có wheel không thay cho kiểm thử Colab thực tế.

## File đã sửa

| File Windows | Thay đổi |
|---|---|
| `D:\COMPACT-MI-RFE\notebooks\COMPACT_MI_RFE_ColabPro.ipynb` | Cell 3 hướng dẫn folder; Cell 5 profile Python 3.13 + restart/test; Cell 6 ưu tiên folder API/mounted; Cell 7 in `DATASET_READY`; Cell 9 chặn khi dataset chưa PASS |
| `D:\COMPACT-MI-RFE\colab\drive_manager.py` | OAuth Colab, phân trang, folder MIME, shortcut, kiểm quyền, liệt kê đệ quy và tải từng file |
| `D:\COMPACT-MI-RFE\colab\dataset_validator.py` | manifest portable fallback; tải/copy vào staging từng file, MD5/size, disk guard, tiếp tục file đã xác minh, validation trước promote |
| `D:\COMPACT-MI-RFE\colab\setup_colab.py` | Profile 3.10–3.12 và 3.13; thông tin profile và Stage 5 exact |
| `D:\COMPACT-MI-RFE\colab\experiment_launcher.py` | Preflight dùng profile đã chọn và manifest portable, không đòi flow CSV local |
| `D:\COMPACT-MI-RFE\requirements-colab.txt` | Pin NumPy theo marker Python; giữ các pin khoa học còn lại |
| `D:\COMPACT-MI-RFE\colab\test_stage6_colab.py` | Regression tests folder, API, phân trang, shortcut, quyền, thiếu file/metadata, staging, disk, resume và Python profile |
| `D:\COMPACT-MI-RFE\COLAB_SETUP.md` | Hướng dẫn chạy folder đã giải nén và restart Runtime |
| `D:\COMPACT-MI-RFE\STAGE6_REPORT.md` | Gắn ghi chú dẫn chiếu báo cáo fix này |

Không sửa năm script CTU-13, runner Stage 5, CSV nguồn, C+ split, window output, model/result Stage 5.2 hoặc 5.3; không push GitHub.

## Luồng dữ liệu sau sửa

Với `DRIVE_DATASET_PATH=""`, Cell 6 xác thực bằng tài khoản của người dùng, gọi Drive API cho folder ID `1Ixy5iJdeavkEbFJ3cy6LjkQ8h300OAxh`, lấy **mọi trang**, giải shortcut hợp lệ và tìm folder tên chính xác `WINDOWS_CPLUS` với MIME `application/vnd.google-apps.folder`. `CSV_SPLIT_CPLUS` chỉ được ghi nhận là có/không; **không tải flow CSV** từ đó. Manifest C+ nhỏ ở `configs/cplus_manifest` là đầu vào provenance cho `WindowStore`; X vẫn chỉ gồm 380 feature window, y từ `Label` SCENARIO_PROXY. Nếu `DRIVE_DATASET_PATH` trỏ folder đã mount, Cell 6 dùng trực tiếp path đó và không gọi API. ZIP vẫn được giữ làm fallback tùy chọn, không dùng cho trường hợp hiện tại.

Drive files được tải một lần vào `/content/datasets/CICIoMT24.incomplete/WINDOWS_CPLUS` theo từng file; so kích thước và MD5 Drive, log số file đã xử lý, chỉ promote sang **`/content/datasets/CICIoMT24/WINDOWS_CPLUS`** sau khi kiểm định metadata/schema đạt PASS. Kế hoạch tải lưu trong `.incomplete`; khi chạy lại, file hoàn chỉnh được hash/xác minh rồi bỏ qua, chỉ tải file thiếu. File `.partial` không được tự xóa/ghi đè; người dùng phải xem nếu transfer hỏng giữa một file. Trước tải kiểm disk còn đủ cho số byte còn thiếu cộng dự trữ 8 GiB; báo lượng thiếu nếu FAIL. Không di chuyển/xóa dữ liệu Drive.

Cell 6–7 chỉ in `DATASET_READY=True` sau khi thấy 4 cấu hình, 322 CSV, manifest-derived counts đủ 12 ô, 380 feature và header Train/Validation hợp lệ. Test feature rows không được đọc tại preflight. Đây **không phải** quét lại nội dung 6.737.006 hàng hay chứng minh hash toàn bộ 21 GB; kiểm định full readback Stage 4.2 cũ được tái sử dụng.

## Kiểm thử thực tế tại máy cục bộ

| Kiểm tra | Kết quả |
|---|---|
| `python -m unittest -q colab.test_stage6_colab` | **20/20 PASS**; fixture tách khỏi dataset nghiên cứu |
| `python -m unittest -q test_ciciomt24_training test_ciciomt24_split test_ciciomt24_sliding_window` tại `huyen_code` | **97/97 PASS** |
| `python -m compileall -q colab` | PASS |
| Notebook nbformat4 JSON, 12 code cell parse Python AST | PASS; chưa chạy `nbformat.validate` vì package `nbformat` không có trong Python local |
| Validator trên dataset cục bộ `D:\COMPACT-MI-RFE\CICIoMT24\WINDOWS_CPLUS` | `PASS_METADATA_SCHEMA`: 4 config, 322 CSV, 394 cột, 380 feature; Test feature rows đọc = 0; full CSV content reverify = False |
| Folder Drive thật/Colab Cell 1–9 | **CHƯA CHẠY, CHƯA XÁC MINH**: không có OAuth/Drive mount/runtime Colab trong môi trường này |
| Python 3.13 thực tế | **CHƯA CHẠY**: wheel có trên PyPI; unit test profile chạy trên Python local 3.10, không chứng minh `pip check` hoặc source tests trên Colab 3.13 |

Không tải 21 GB, không fit MI/RFE/RF/SVM chính thức và không đọc Test để chọn model. Kết quả fixture/mock không phải kết quả thực nghiệm.

## Người dùng chạy lại tối thiểu

1. Tự kiểm tra diff và commit/push các file nguồn nhỏ đã sửa lên GitHub; **không** đưa dataset/run/model lên GitHub. Repo đang có local source chưa được theo dõi đầy đủ, nên clone thiếu file sẽ bị Cell 4 chặn. Nếu Colab còn clone cũ, bắt đầu runtime mới hoặc kiểm tra checkout thủ công; notebook không tự pull/reset.
2. Mở notebook `notebooks/COMPACT_MI_RFE_ColabPro.ipynb`, chạy Cell 1–4. Cell 3 **để trống** `DRIVE_DATASET_PATH` và `DRIVE_ARCHIVE_ID`; không nén lại dataset.
3. Chạy Cell 5. Nếu cell cài package khoa học và dừng, chọn **Runtime → Restart runtime**, chạy lại từ Cell 1. Chỉ đi tiếp khi `pip check` và cả hai bộ unit test PASS.
4. Chạy Cell 6; cấp quyền đọc Drive cho đúng tài khoản khi Colab hỏi. Chờ log từng file và `DATASET_READY=True`; kiểm đường dẫn `/content/datasets/CICIoMT24/WINDOWS_CPLUS`.
5. Chạy Cell 7–9. Gửi output `PREPARED`, `DATA_AUDIT`, `PREFLIGHT` (không gửi token/OAuth) nếu một guard FAIL. **Giữ `MODE="dry_run"`; chưa chạy Cell 10/full training.**

Chưa có bằng chứng trên Colab cho disk/RAM thực tế, quyền folder, MD5 transfer, thời gian tải hoặc tính tương đương số học Python 3.13. Chờ người dùng xác nhận preflight Cell 1–9 thành công trước khi xin phép huấn luyện.
