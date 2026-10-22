# Stage 7.3B — Hướng dẫn Colab (chưa cấp phép chạy)

## 1. Đồng bộ source

Không dùng notebook cũ và không dùng `Run all`. Commit/push các file Stage 7.3B từ Windows, sau đó clone/switch đúng commit sạch trên Colab. Commit `0461d2ba...` trong Stage 7.3A **không tự động là commit Stage 7.3B** vì source đã thay đổi.

Tạo checkout Colab mới, không ghi đè checkout cũ (thay chuỗi commit bằng full SHA vừa push):

```python
from pathlib import Path
import subprocess

APPROVED_STAGE73B_COMMIT = "PASTE_FULL_STAGE73B_COMMIT_HERE"
REPO = Path(f"/content/RFforNetwork_stage73b_{APPROVED_STAGE73B_COMMIT[:8]}")
if len(APPROVED_STAGE73B_COMMIT) != 40 or not all(c in "0123456789abcdef" for c in APPROVED_STAGE73B_COMMIT.lower()):
    raise ValueError("Dán full 40-character Git commit trước")
if REPO.exists():
    raise FileExistsError(f"Không ghi đè checkout cũ: {REPO}")
subprocess.run(["git", "clone", "--", "https://github.com/kasum00/RFforNetwork.git", str(REPO)], check=True)
subprocess.run(["git", "-C", str(REPO), "switch", "--detach", APPROVED_STAGE73B_COMMIT], check=True)
status = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], text=True, capture_output=True, check=True).stdout
if status.strip():
    raise RuntimeError("Checkout mới không sạch")
print(REPO, APPROVED_STAGE73B_COMMIT)
```

Sau đó sửa biến `REPO` ở Cell 1 notebook thành đúng thư mục mới này. Không chạy Cell 4A legacy và không copy file upload đè lên checkout.

PowerShell để người dùng tự kiểm thử trước commit:

```powershell
Set-Location 'D:\COMPACT-MI-RFE'
$env:PYTHONPATH = 'D:\COMPACT-MI-RFE;D:\COMPACT-MI-RFE\huyen_code'
python -m unittest -v colab.test_stage73b_authorization colab.test_stage73b_supervisor colab.test_stage73b_scientific_integrity
python -m unittest -v colab.test_svm_official_readiness colab.test_stage6_colab
```

Lệnh Git có phạm vi (xem diff trước khi commit):

```powershell
Set-Location 'D:\COMPACT-MI-RFE'
git status --short
git diff -- huyen_code/ciciomt24_training.py colab/experiment_launcher.py colab/checkpoint_manager.py colab/svm_run_authorization.py colab/svm_official_worker.py colab/svm_official_supervisor.py colab/test_stage73b_authorization.py colab/test_stage73b_supervisor.py colab/test_stage73b_scientific_integrity.py notebooks/STAGE7_3B_OFFICIAL_SVM.ipynb configs/stage73b/authorization_template.BLOCKED.json huyen_code/STAGE7_3B_REPORT.md huyen_code/STAGE7_3B_RESOURCE_POLICY.md huyen_code/STAGE7_3B_COLAB_GUIDE.md huyen_code/STAGE7_3B_SCIENTIFIC_INTEGRITY.md
git add -- huyen_code/ciciomt24_training.py colab/experiment_launcher.py colab/checkpoint_manager.py colab/svm_run_authorization.py colab/svm_official_worker.py colab/svm_official_supervisor.py colab/test_stage73b_authorization.py colab/test_stage73b_supervisor.py colab/test_stage73b_scientific_integrity.py notebooks/STAGE7_3B_OFFICIAL_SVM.ipynb configs/stage73b/authorization_template.BLOCKED.json huyen_code/STAGE7_3B_REPORT.md huyen_code/STAGE7_3B_RESOURCE_POLICY.md huyen_code/STAGE7_3B_COLAB_GUIDE.md huyen_code/STAGE7_3B_SCIENTIFIC_INTEGRITY.md
git commit -m "Implement fail-closed Stage 7.3B SVM supervisor"
git push origin HEAD:main
```

Codex chưa chạy các lệnh trên và không tuyên bố test PASS.

## 2. Notebook 12 cell

Mở `notebooks/STAGE7_3B_OFFICIAL_SVM.ipynb`, chạy từng cell:

1. Cell 1: xem Python/CPU/RAM/disk; chỉnh `REPO` sang checkout mới.
2. Cell 2: mount Drive khi chủ động bật.
3. Cell 3: kiểm commit, clean checkout và source SHA.
4. Cell 4: kiểm dataset/manifest/header; không tạo lại window và không đọc Test feature rows.
5. Cell 5: chạy test trong process riêng, lưu log/evidence gắn đúng training SHA.
6. Cell 6: chọn đúng một experiment/config/run name.
7. Cell 7: thu identity/readiness; vẫn BLOCKED và không fit.
8. Cell 8: đọc probe JSON, hiển thị khác biệt config và ngân sách chưa duyệt.
9. Cell 9: kiểm approval read-only sau khi người dùng tự duyệt; không fit/Test.
10. Cell 10: execution; **không chạy trong Stage 7.3B**.
11. Cell 11: chỉ verify artifact sau khi một run được phê duyệt và hoàn thành.
12. Cell 12: chỉ đọc `result.json`, không load model hoặc dự đoán Test.

Lưu ý số cell đúng của notebook mới: Cell 10 là execution và **tuyệt đối chưa chạy** trong Stage 7.3B; Cell 11 là verify artifact; Cell 12 chỉ đọc `result.json`. Không có Cell 4A.

## 3. Approval

Không sửa trực tiếp file template trong Git thành approval dùng chung. Tạo một bản riêng trên Drive cho đúng run. Điền identity do checkout/dataset hiện tại sinh, budget do người dùng chốt, SHA evidence và một trong hai technical policy. Exact phrase có dạng:

```text
AUTHORIZE STAGE7.3B baseline_svm FW_512_FS_16 <RUN_NAME> <FULL_GIT_COMMIT>
```

Approval mất hiệu lực nếu commit/source/dataset/runtime profile/run name/budget/evidence thay đổi. Một approval không dùng cho run khác.

## Tiêu chí PASS trước khi cân nhắc Cell 10

- Checkout Git sạch, commit và toàn bộ identity/hash khớp approval.
- Các test mới và regression suite do người dùng chạy đều PASS; evidence gắn đúng training SHA.
- Dataset readiness, schema/manifest và source CTU-13 PASS mà không đọc Test feature rows.
- Không tồn tại local/Drive result, `.incomplete`, checkpoint hoặc environment JSON cùng run name.
- Technical evidence SHA/identity PASS; khác config chỉ được chấp nhận bằng policy thay thế tường minh.
- Approval có hiệu lực cho đúng run và mọi budget là số nguyên dương; RAM/disk admission PASS.
- Cell 9 trả `PASS_STAGE73B_AUTHORIZED`, `model_fit_calls=0`, `test_feature_rows_read=0`.

Chỉ một điều kiện FAIL hoặc UNKNOWN thì giữ BLOCKED. PASS các điều kiện trên vẫn chỉ tạo điều kiện để người dùng ra quyết định riêng; nó không tự bật Cell 10.

## 4. Bằng chứng cần gửi lại

- Toàn bộ log unit test, commit và `git status --porcelain`.
- Output authorization review (không chứa Test rows).
- `reports/<run>_supervisor.json` và `_supervisor_events.jsonl`.
- `environment/<run>.json`.
- Trong `experiment_results/<run>`: `result.json`, `artifact_manifest.json`, `preflight_audit.json`, `authorization.json`, `hyperparameters.json`, `feature_schema.json`, `selection_frozen.json`, `events.jsonl`, `progress.json`; với compact thêm MI/RFE/k artifacts.
- Nếu lỗi: `FAILED.json`, checkpoint state và supervisor report; không tự chạy lại.

## 5. Recovery đã validation

Chỉ khi checkpoint có `stage=run_validated`, log đúng một Test start/completed và đúng thứ tự, người dùng mới xem xét lệnh publication recovery. Hàm `publish_completed_checkpoint` không fit và không đánh giá Test lại. Nếu Test đã bắt đầu mà chưa `run_validated`: dừng và xin quyết định khoa học, không retry.
