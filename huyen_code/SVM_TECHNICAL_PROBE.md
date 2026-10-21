# Stage 7.1 — Hướng dẫn SVM technical probe (chỉ Train)

Probe này **không phải thí nghiệm chính thức** và không tạo Accuracy để báo cáo
khoa học. Nó chỉ đo chi phí của SVM RBF trên mẫu tăng dần. Không đọc hàng CSV
Validation/Test; nhãn vẫn là `SCENARIO_PROXY`.

## Thiết kế và giới hạn

- Một lệnh chỉ chạy một `experiment` (`baseline_svm` hoặc `mi_rfe_svm`) và một
  FW/FS. Không tự chạy full training.
- Tái sử dụng `WindowStore.load_sample` của Stage 5: pool tối đa **8.192** window
  Train, lấy bằng quota theo source và sparse prefix stride=8. Đây không phải
  mẫu ngẫu nhiên đại diện cho toàn bộ phân phối; giới hạn 8.192 được giữ để
  không nới âm thầm cap cũ. Ba kích thước mặc định: 2.000, 5.000, 8.192.
- Từ pool Train, chọn mẫu theo proxy class với seed 42; mỗi kích thước tách
  internal fit/holdout 80/20. Khi Benign quá ít, áp dụng sàn hai mẫu để cả fit
  và holdout có hai lớp. JSON ghi seed, số Benign/Attack, số source, hash ID mẫu.
- Baseline dùng đúng SVC source: RBF, C=10, gamma=scale, probability=True,
  cache_size=10000 MB, shrinking=True, max_iter=20000, class_weight=None.
- MI-RFE-SVM technical probe fit scaler/MI chỉ trên internal Train-fit, MI Top100,
  RF-RFE theo tham số source nhưng chỉ loại ba feature (`target=97`, bốn forest
  fit), rồi đo SVC với k=1,10,100. Ranking này **chưa hoàn chỉnh**, không có
  `best_k` chính thức, không được dùng để chọn feature hoặc k cho full run.
- Mỗi size là một worker riêng. Supervisor chỉ giám sát/terminate process tree
  do chính probe tạo, không tác động tiến trình khác. Default: timeout 600 giây
  mỗi size, RSS tree cap 8 GiB, yêu cầu RAM available ít nhất cap + 2 GiB; dừng
  nếu RAM host còn dưới 1 GiB. Cap được kiểm tra bằng polling 0,25 giây, **không
  phải hard limit của OS** và không loại trừ OOM tức thời.
- JSON ghi fit time, predict time, RSS peak đo trong fit/predict và RSS peak
  process tree quan sát, `support_vectors`, `fit_status`, `converged`, warnings,
  versions/source hash. Không ngoại suy tuyến tính thành cam kết full SVM.
- `PASS_TECHNICAL_ONLY` có nghĩa mọi worker hoàn tất và các SVC fit báo hội tụ;
  **không** đồng nghĩa resource gate full đã PASS. Full SVM vẫn bị preflight
  chặn tại Stage 7.1 cho tới quyết định Stage 7.2 riêng.

## PowerShell: bạn tự chạy (Codex không thực thi)

Chỉ chạy khi máy không còn tiến trình nặng khác và có đủ RAM available. Đừng
thay `--experiment` sang RF hoặc nâng sample vượt 8.192.

```powershell
Set-Location 'D:\COMPACT-MI-RFE'
$env:PYTHONPATH = 'D:\COMPACT-MI-RFE;D:\COMPACT-MI-RFE\huyen_code'
python -m unittest discover -s 'D:\COMPACT-MI-RFE\huyen_code' -p 'test_ciciomt24_svm_probe.py' -v
python -m unittest discover -s 'D:\COMPACT-MI-RFE\colab' -p 'test_stage6_colab.py' -v
```

Ví dụ chạy **một** probe Baseline, FW512/FS16:

```powershell
python 'D:\COMPACT-MI-RFE\huyen_code\ciciomt24_svm_probe.py' probe `
  --repo 'D:\COMPACT-MI-RFE' `
  --windows 'D:\COMPACT-MI-RFE\CICIoMT24\WINDOWS_CPLUS' `
  --manifest-root 'D:\COMPACT-MI-RFE\CICIoMT24\CSV_SPLIT_CPLUS' `
  --config FW_512_FS_16 --experiment baseline_svm `
  --output 'D:\COMPACT-MI-RFE\huyen_code\runs\stage71_baseline_FW512_FS16_probe.json' `
  --sizes 2000 5000 8192 --seed 42 --timeout-per-size 600 --max-rss-gib 8
```

Lệnh riêng cho MI-RFE-SVM (chỉ chạy **sau** probe trên nếu muốn đo cả hai):

```powershell
python 'D:\COMPACT-MI-RFE\huyen_code\ciciomt24_svm_probe.py' probe `
  --repo 'D:\COMPACT-MI-RFE' `
  --windows 'D:\COMPACT-MI-RFE\CICIoMT24\WINDOWS_CPLUS' `
  --manifest-root 'D:\COMPACT-MI-RFE\CICIoMT24\CSV_SPLIT_CPLUS' `
  --config FW_512_FS_16 --experiment mi_rfe_svm `
  --output 'D:\COMPACT-MI-RFE\huyen_code\runs\stage71_mi_rfe_svm_FW512_FS16_probe.json' `
  --sizes 2000 5000 8192 --seed 42 --timeout-per-size 600 --max-rss-gib 8
```

Không chạy hai probe song song. Probe từ chối ghi đè output/worker artifact cũ.
Nếu một worker timeout, thiếu RAM, không hội tụ hoặc lỗi, giữ JSON làm bằng
chứng kỹ thuật; không dùng làm kết quả chính thức và không tự resume.

## Google Colab: cell để bạn tự chạy sau khi chuẩn bị dataset

**Colab là tùy chọn ở Stage 7.1.** 9/9 unit test trên Windows chỉ xác nhận
logic kiểm soát; chưa đo thời gian/RAM. Bạn có thể chạy probe ngay trên Windows
bằng lệnh PowerShell phía trên, không cần Colab. Nếu dùng Colab, notebook sẽ
clone GitHub vào `/content/RFforNetwork`: các file vừa sửa trên ổ D **không tự
xuất hiện** ở đó. Trước khi chạy probe, bạn phải tự đồng bộ tối thiểu
`huyen_code/ciciomt24_svm_probe.py` và bản mới của
`colab/experiment_launcher.py` vào checkout Colab. Không dùng bản launcher cũ
vì technical probe Stage 6 cũ đọc cả Validation.

Nếu đã đồng bộ code và dữ liệu, cách đơn giản nhất là dùng chính notebook
`notebooks/COMPACT_MI_RFE_ColabPro.ipynb`:

1. Chạy các cell chuẩn bị 1–7; xác nhận `DATASET_READY=True` và đúng đường dẫn
   `WINDOW_ROOT`. Chạy unit tests Stage 7.1 trong Colab nếu mã mới đã đồng bộ.
2. Cell 8: chọn `EXPERIMENT="baseline_svm"` **hoặc** `"mi_rfe_svm"`, FW/FS
   mong muốn, `RUN_NAME` mới; giữ `MODE="dry_run"`, `CONFIRMATION=""`, bật
   `RUN_TECHNICAL_SVM_PROBE=True`.
3. Cell 9 chỉ kiểm tra preflight; với SVM ở Stage 7.1, `PREFLIGHT=FAIL` là
   **dự kiến** vì full SVM chưa được duyệt. Không sửa gate để ép PASS.
4. Cell 10 chạy **chỉ probe** khi `MODE="dry_run"`; đọc JSON tại
   `/content/stage6_runs/<RUN_NAME>_svm_technical_probe.json`. Không bật
   `full_train`. Nếu chạy lại, chọn `RUN_NAME`/output mới; probe không ghi đè.

Chỉ chạy một thí nghiệm một lần. Probe mặc định ba cỡ mẫu Train 2.000/5.000/8.192,
không nạp dữ liệu Test và không tạo kết quả Accuracy chính thức.

Chạy Cell 1–8 của notebook Stage 6 trước, để `MODE="dry_run"`, và chắc chắn
checkout GitHub đã có file `huyen_code/ciciomt24_svm_probe.py` mới. Cell dưới
đây chỉ gọi probe, **không** gọi `run_one` hay Test:

```python
from pathlib import Path
from colab.experiment_launcher import RunSpec, run_technical_svm_probe

repo = Path('/content/RFforNetwork')
windows = Path('/content/datasets/CICIoMT24/WINDOWS_CPLUS')
spec = RunSpec('baseline_svm', 'FW_512_FS_16', 'stage71_baseline_FW512_FS16_probe')
output = Path('/content/runs/stage71_baseline_FW512_FS16_probe.json')
probe = run_technical_svm_probe(repo, windows, spec, output)
print(probe['status'], probe['validation_csv_rows_read'], probe['test_csv_rows_read'])
print(output)
```

Cell riêng cho MI-RFE-SVM (không chạy cùng lúc với cell trên):

```python
from pathlib import Path
from colab.experiment_launcher import RunSpec, run_technical_svm_probe

repo = Path('/content/RFforNetwork')
windows = Path('/content/datasets/CICIoMT24/WINDOWS_CPLUS')
spec = RunSpec('mi_rfe_svm', 'FW_512_FS_16', 'stage71_mi_rfe_svm_FW512_FS16_probe')
output = Path('/content/runs/stage71_mi_rfe_svm_FW512_FS16_probe.json')
probe = run_technical_svm_probe(repo, windows, spec, output)
print(probe['status'], probe['validation_csv_rows_read'], probe['test_csv_rows_read'])
print(output)
```

Sau khi xem JSON,
người dùng có thể tự lưu file JSON nhỏ lên Drive. Đừng bật `full_train` trong
Stage 7.1: `svm_full_resource_gate_reviewed=False` vẫn chặn toàn bộ full SVM.

## Đọc JSON và quyết định bước sau

`status=PASS_TECHNICAL_ONLY`, `validation_csv_rows_read=0`,
`test_csv_rows_read=0`, `full_training_authorized=false` là bằng chứng probe
đúng giới hạn, **không** phải bằng chứng full run khả thi. So sánh thời gian fit,
predict, RSS, support-vector count/fraction, `fit_status`, warnings giữa các
size; đánh dấu `REVIEW_REQUIRED` nếu tăng vọt, không hội tụ hoặc cap bị chạm.
SVM RBF có thể tăng phi tuyến theo số mẫu, nên không cam kết giờ/RAM full từ ba
điểm nhỏ. Stage 7.2 cần bạn duyệt policy resource và cấu hình trước khi mở khóa
full training; Test luôn chỉ được đọc sau final refit của thí nghiệm chính thức.
