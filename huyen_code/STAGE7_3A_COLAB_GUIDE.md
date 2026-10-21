# Stage 7.3A — các bước người dùng tự chạy (không full SVM)

**Mục tiêu:** đồng bộ đúng một checkout, chạy unit tests và preflight chỉ đọc, gửi bằng chứng để quyết định Stage7.3B. Stage7.3A không có nút mở full SVM. Không bấm `Run all`, không chạy Cell 10, không chạy training CLI, không tạo probe mới nếu chỉ muốn kiểm tra readiness.

## 1. PowerShell Windows — tự chạy tests và kiểm diff

Mở PowerShell, thực hiện từng lệnh. Codex **chưa chạy** các lệnh này:

```powershell
Set-Location 'D:\COMPACT-MI-RFE'
$env:PYTHONPATH = 'D:\COMPACT-MI-RFE;D:\COMPACT-MI-RFE\huyen_code'
python -m unittest -v colab.test_svm_official_readiness colab.test_stage6_colab test_ciciomt24_svm_probe test_ciciomt24_svm_scaling_probe
python -m unittest discover -s 'D:\COMPACT-MI-RFE\huyen_code' -p 'test_ciciomt24_training.py' -v
git -C 'D:\COMPACT-MI-RFE' status --short
git -C 'D:\COMPACT-MI-RFE' diff --check
```

Nếu bất kỳ test FAIL hoặc import lỗi: dừng; gửi toàn bộ traceback, không bật full SVM. Test trên mock/synthetic **không** chứng minh full runtime khả thi. Kiểm diff notebook đặc biệt để chắc không stage outputs cũ ngoài ý muốn.

## 2. Git — chỉ sau khi test và diff được xem xét

Các lệnh dưới đây dành cho **bạn tự thực hiện**, không phải Codex. Chỉ stage code/tài liệu đúng danh sách; không `git add .`, không add dataset, Windows runs hoặc Stage5 results:

```powershell
git -C 'D:\COMPACT-MI-RFE' diff -- 'huyen_code/ciciomt24_training.py' 'huyen_code/ciciomt24_sliding_window.py' 'notebooks/COMPACT_MI_RFE_ColabPro.ipynb'
git -C 'D:\COMPACT-MI-RFE' add -- 'huyen_code/ciciomt24_training.py' 'huyen_code/ciciomt24_sliding_window.py' 'colab/svm_official_readiness.py' 'colab/test_svm_official_readiness.py' 'notebooks/COMPACT_MI_RFE_ColabPro.ipynb' 'huyen_code/STAGE7_3A_SVM_FULL_AUDIT.md' 'huyen_code/STAGE7_3A_RESOURCE_BUDGET.md' 'huyen_code/STAGE7_3A_COLAB_GUIDE.md'
git -C 'D:\COMPACT-MI-RFE' diff --cached --name-only
git -C 'D:\COMPACT-MI-RFE' diff --cached --check
git -C 'D:\COMPACT-MI-RFE' diff --cached --stat
git -C 'D:\COMPACT-MI-RFE' commit -m 'Prepare fail-closed Stage 7.3A SVM readiness review'
git -C 'D:\COMPACT-MI-RFE' push origin HEAD
git -C 'D:\COMPACT-MI-RFE' rev-parse HEAD
git -C 'D:\COMPACT-MI-RFE' status --porcelain
```

Nếu staged list chứa dataset/checkpoints/results hoặc notebook chứa nội dung bất ngờ: **không commit**; xem xét trước. Nếu trước đó hai file cross-platform đã commit thì `git add` chúng không tạo diff mới. Ghi lại SHA commit sau push và bảo đảm checkout sạch; đừng lấy HEAD cũ `1258abd...` làm commit Stage7.3A.

## 3. Colab — checkout an toàn

Trong runtime Colab Pro, ưu tiên phiên mới để clone sạch. Nếu `/content/RFforNetwork` đã có checkout, kiểm tra trước bằng một ô riêng:

```python
!git -C /content/RFforNetwork status --porcelain
!git -C /content/RFforNetwork rev-parse HEAD
```

Nếu có dòng dirty, **dừng**, không reset/stash/pull chồng lên dữ liệu người dùng. Nếu checkout sạch nhưng không đúng commit mới, tự fetch rồi checkout đúng SHA đã ghi (thay placeholder bằng SHA thật, không chạy nguyên placeholder):

```python
!git -C /content/RFforNetwork fetch origin
!git -C /content/RFforNetwork switch --detach <SHA_COMMIT_DA_DUYET>
!git -C /content/RFforNetwork status --porcelain
!git -C /content/RFforNetwork rev-parse HEAD
```

Nếu chưa có checkout, dùng Cell 4 của notebook để clone rồi đối chiếu HEAD; Cell 4 **không pull** một checkout sẵn có. Không xóa dataset local đang dùng. Không chạy Cell 4A legacy trên source mới; nó được chặn để không ghi đè. Chạy Cell 5 theo thứ tự để kiểm phiên bản khoa học và test Stage5/6; Cell 5 có thể cài thư viện khi thiếu và yêu cầu restart runtime, nên phải xác minh lại Git/hash sau đó. Nếu Cell 5 thay ROOT thì phải dừng source consistency.

Tiếp theo chạy Cell 6/7 để `DATASET_READY=True` trên `WINDOWS_CPLUS` hiện có; không transfer lại nếu bản cũ đã xác minh. Cell 8 chọn duy nhất `baseline_svm`, `FW=512`, `FS=16`, `RUN_NAME` mới (ví dụ `baseline_svm_FW512_FS16_review01`), `MODE='dry_run'`. Đây là cấu hình gợi ý cho **review**, chưa phê duyệt official run. Giữ confirmation full-training trống/không dùng. Cell 9 preflight cũ có thể báo FAIL do thiếu technical probe đúng identity và `svm_full_resource_gate_reviewed=False`; đó là trạng thái an toàn, không sửa thành PASS.

Ở cuối notebook có `stage73a-tests`: đổi duy nhất `STAGE73A_RUN_TESTS = True`, chạy **ô đó** rồi xem traceback/PASS. Sau đó ở `stage73a-review`, đổi `STAGE73A_RUN_PREFLIGHT = True`, chạy **ô đó**. Kết quả phải là `BLOCKED_STAGE73A`, `model_fit_calls=0`, `test_feature_rows_read=0`. Nó kiểm metadata/header, RAM/disk và source; không đọc Test feature rows hay train. Không dùng Cell 10, Stage7.2 RUN hay technical probe cell trong quy trình này. Nếu muốn review nhánh khác/config khác, tạo `RUN_NAME` mới ở Cell 8 rồi chạy lại review; không dùng approval của run cũ.

## 4. Những gate phải xem, không được bỏ qua

- Git commit Windows/Colab và working tree sạch; source raw/LF-normalized hash phù hợp, năm CTU13 script không đổi.
- Feature schema 380, manifest/snapshot và số Train/Val/Test đúng cấu hình.
- Unit tests mới PASS trên chính commit/runtime, source hash trong evidence khớp.
- Không có run hoặc `.incomplete` trùng tên trên local/Drive.
- RAM/disk admission PASS trên phiên hiện tại nhưng **chưa đủ**: full time/peak memory vẫn UNKNOWN.
- Resource feasibility `REVIEW_REQUIRED` cho cả tám experiment/config; cần budget max elapsed/RSS, safety margin và người dùng duyệt riêng.
- `stage73b_implementation_and_authorization=False` là đúng cho Stage7.3A. Không chuyển sang true bằng sửa notebook hoặc `svm_full_resource_gate_reviewed=True`.

## 5. Gửi lại để quyết định Stage7.3B

Gửi (có thể dán text/screenshot hoặc JSON nhỏ, không cần gửi dataset/Test CSV): log test Windows và Colab gồm số PASS/FAIL; SHA commit Windows/Colab + `git status --porcelain`; output Cell 7 (counts/schema/fingerprint); output Cell 9 và `stage73a-review` (từng check, RAM guard, disk required, identity hashes, runtime CPU/RAM/versions); hai `scaling02.json` đã được audit hoặc đường dẫn của chúng; ngân sách thời gian và RSS tối đa bạn chấp nhận cho **một** experiment/config/run_name; dung lượng local/Drive trống và dự kiến thời lượng phiên Colab. Nếu có mismatch/FAIL, gửi traceback đầy đủ. Không gửi Test predictions để chọn mô hình.

Chỉ sau khi chúng ta xem đủ bằng chứng và bạn cấp phép cụ thể mới thiết kế/kiểm chứng Stage7.3B. Stage7.3A không cấp quyền full SVM, không tạo official Test metric, không chứng minh capture/session độc lập.
