# Stage 7.2 — Bounded SVM scaling probe (chỉ Train)

**Trạng thái cập nhật 22/10/2026:** Đây là hướng dẫn triển khai ban đầu. Người
dùng đã chạy tests và cả hai nhánh Baseline/MI-RFE-SVM scaling probe
16k/32k/64k. Xem kết quả và giới hạn tại
[STAGE7_2_REPORT.md](STAGE7_2_REPORT.md). File run MI-RFE `scaling01`
trước đó sai identity; báo cáo dùng run `scaling02` đúng nhánh. Codex không
tự chạy Python, unit test, probe, dataset audit hoặc training. Full SVM vẫn
bị khóa, resource gate còn `REVIEW_REQUIRED`.

## Audit Stage 7.1 và thay đổi có chủ đích

- `WindowStore.load_sample` trong `ciciomt24_training.py` chặn Train ở 8.192
  window; `ProbeLimits` Stage 7.1 cũng chặn 8.192. Không sửa hai giới hạn này.
- Stage 7.1 lấy quota theo source, chỉ quét sparse prefix (stride=8), rồi chọn
  stratified với seed 42 và internal fit/holdout 80/20. Đây là mẫu kỹ thuật,
  không đại diện chắc cho full Train; mẫu 8.192 quan sát có 19 Benign.
- `run_worker` Stage 7.1 chỉ gọi `load_sample(config, "train", ...)`.
  `select_mi` và scaler Top100 fit internal Train-fit; Baseline scaler cũng fit
  internal Train-fit. Nhãn `val` của `PreparedSplit` internal holdout chỉ phục
  vụ API, không phải CSV Validation thật.
- Supervisor Stage 7.1 chạy worker tuần tự, poll RSS process tree/timeout và
  chỉ dừng process tree do chính nó tạo. RSS polling không phải hard OS limit.
- JSON Stage 7.1 từ chối output có sẵn và có status dở khi lỗi. Full SVM gate
  vẫn False ngay cả sau `PASS_TECHNICAL_ONLY`.
- Stage 7.2 là module **riêng** `ciciomt24_svm_scaling_probe.py`; Stage 7.1
  probe, benchmark/results cũ và tham số estimator không đổi.

## Lấy mẫu 16k/32k/64k

Chỉ chạy khi có `--enable-scaling-probe`. `ScalingLimits` chỉ chấp nhận tập
con tăng dần của 16.000/32.000/64.000, hard cap 64.000; không rơi về 8.192.
Loader dựa trên manifest C+ để đếm đầy đủ Train theo proxy class, dùng
permutation riêng từng class với seed cố định, lấy không thay thế. Prefix
permutation tạo mẫu lồng nhau qua các cỡ mẫu. Chỉ CSV role `train` được đọc;
từng chunk được kiểm tra schema/provenance, rồi mới giữ các dòng đã chọn.
Không cần nạp toàn bộ ma trận 380 chiều vào RAM, nhưng một lượt sample có thể
quét phần lớn Train CSV nên thời gian đọc đáng kể. Mỗi worker đọc lại Train;
không có memmap trung gian nhiều GB và không chuyển memmap lên Drive.

Quota Benign gần tỷ lệ toàn Train, nhưng có floor **hai Benign** để internal
fit/holdout đều có hai lớp. Floor, số Benign/Attack toàn Train, số thực tế
trong sample/fit/holdout, số source/scenario và SHA256 chuỗi Window_ID đều
được ghi. Đây là **class-aware technical sampling**, không phải phân phối ngẫu
nhiên nguyên bản của Train và không dùng để tính Accuracy khoa học. Việc lấy
mẫu toàn cục khác Stage 7.1 sparse prefix; do đó không so sánh trực tiếp thời
gian load hoặc tỷ lệ class của hai probe như cùng một thiết kế. Không sửa hay
diễn giải lại kết quả Stage 7.1 đã hoàn thành.

Baseline giữ SVC source `C=10`, RBF, `gamma=scale`, `probability=True`,
`class_weight=None`, `max_iter=20000`, `shrinking=True`, `cache_size=10000`.
MI-RFE-SVM giữ SVC source `probability=False`, `class_weight=balanced`, MI
Top100 fit internal Train-fit, partial RF-RFE `target=97` (bốn forest fits),
k thử kỹ thuật 1/10/100. Partial ranking **không** là ranking đầy đủ, không
chọn best_k chính thức. `Test` và CSV Validation không được mở.

## Guard và artifact

Mặc định 1200 giây và RSS tree cap 12 GiB mỗi worker; chỉ đổi qua CLI/cell
trước khi chạy. RAM available phải lớn hơn cap cộng headroom `max(2 GiB,
20% available)` trước **mỗi** worker. Supervisor dừng worker của chính probe
nếu timeout/RSS vượt cap hoặc available host quá thấp; không tiếp tục size lớn
hơn sau FAIL/nonconvergence/integrity mismatch. Chạy một probe tại một thời
điểm; lock host cấm hai probe dùng cùng dataset đồng thời. Không auto-resume.
Polling 0,25 giây có thể bỏ sót spike/OOM, không bảo đảm full SVM an toàn.

Mỗi worker có JSON riêng `.n16000.json`, `.n32000.json`, `.n64000.json`;
supervisor ghi JSON tổng. Không ghi đè output cũ. Worker lỗi vẫn để lại JSON
`WORKER_FAILED`/partial evidence. `PASS_TECHNICAL_ONLY` chỉ được gán khi đủ
measurement/label/provenance safety. `resource_gate` luôn `REVIEW_REQUIRED`,
`full_training_authorized=false`, `official_best_k=null`. Cần xem cả JSON
worker và tổng nếu kết quả dở; không xóa artifact lỗi.

## Bạn tự chạy unit tests trên Windows (Codex không chạy)

```powershell
Set-Location 'D:\COMPACT-MI-RFE'
$env:PYTHONPATH = 'D:\COMPACT-MI-RFE;D:\COMPACT-MI-RFE\huyen_code'
python -m unittest discover -s 'D:\COMPACT-MI-RFE\huyen_code' -p 'test_ciciomt24_svm_scaling_probe.py' -v
python -m unittest discover -s 'D:\COMPACT-MI-RFE\huyen_code' -p 'test_ciciomt24_svm_probe.py' -v
python -m unittest discover -s 'D:\COMPACT-MI-RFE\colab' -p 'test_stage6_colab.py' -v
```

**Chỉ sau khi tests PASS**, đồng bộ GitHub thủ công. Vì Codex không đọc được
Git status ở workspace này, trước khi commit hãy tự xem toàn bộ thay đổi:

```powershell
git -C 'D:\COMPACT-MI-RFE' status --short
git -C 'D:\COMPACT-MI-RFE' diff --check
git -C 'D:\COMPACT-MI-RFE' add -- 'notebooks/COMPACT_MI_RFE_ColabPro.ipynb' 'huyen_code/ciciomt24_svm_scaling_probe.py' 'huyen_code/test_ciciomt24_svm_scaling_probe.py' 'huyen_code/STAGE7_2_SCALING_PROBE.md' 'huyen_code/STAGE7_2_RESOURCE_REVIEW.md'
git -C 'D:\COMPACT-MI-RFE' diff --cached --stat
git -C 'D:\COMPACT-MI-RFE' commit -m 'Prepare bounded Stage 7.2 Train-only SVM scaling probe'
git -C 'D:\COMPACT-MI-RFE' push
```

Không thêm dataset, notebook output cũ, artifacts hoặc năm script CTU-13.
Nếu còn thay đổi user khác, chỉ commit các file Stage 7.2 vừa kiểm tra.

## Colab: chính bạn chạy, tuần tự, đúng một probe

1. Sau khi push, mở notebook mới trong GitHub ở Colab và kết nối Drive/runtime
   có `/content/datasets/CICIoMT24/WINDOWS_CPLUS` hợp lệ. Không xóa/retransfer
   dữ liệu chỉ vì mở notebook mới. Kiểm tra checkout
   `/content/RFforNetwork/huyen_code/ciciomt24_svm_scaling_probe.py` có mặt.
   Nếu checkout đã tồn tại trước push, **không** dùng `git pull` tự động trong
   notebook: tự kiểm tra thay đổi rồi tạo checkout mới hoặc đồng bộ file mã
   theo quy trình bạn phê duyệt, tránh ghi đè state đang chạy.
2. Chạy cell chuẩn bị/dataset có liên quan, xác nhận `DATASET_READY=True`;
   sau đó chạy cell **STAGE 7.2 CONFIG**. Mặc định scaling disabled.
3. Baseline: đặt `STAGE72_EXPERIMENT='baseline_svm'`, FW=256, FS=8,
   `STAGE72_RUN_NAME='baseline_svm_FW256_FS8_scaling01'`, sizes=(16000,32000,64000),
   `STAGE72_ENABLE=True`. Chạy **STAGE 7.2 RUN** một lần. Nó gọi đúng một
   subprocess với `--enable-scaling-probe`; xem progress từng worker.
4. Chỉ sau khi Baseline xong và kiểm tra JSON, đổi experiment thành
   `mi_rfe_svm`, tên mới `mi_rfe_svm_FW256_FS8_scaling01`, chạy lại CONFIG
   rồi RUN. Không chạy song song và không bật `MODE=full_train` ở Stage 6.
5. JSON tổng nằm tại `/content/stage72_runs/<run_name>.json`; JSON từng worker
   có hậu tố `.n<size>.json`. Chạy cell **STAGE 7.2 REVIEW** chỉ sau khi có cả
   hai JSON; nó đọc JSON, không fit hay mở CSV. Chạy **STAGE 7.2 BACKUP** với
   opt-in riêng để chép JSON/report nhỏ lên
   `/content/drive/MyDrive/COMPACT-MI-RFE/technical_probes/stage72/`; cell từ
   chối ghi đè file Drive đã tồn tại.

Nếu dùng CLI Colab thay vì cell, lệnh Baseline (chạy trong terminal Colab):

```bash
python /content/RFforNetwork/huyen_code/ciciomt24_svm_scaling_probe.py probe \
  --enable-scaling-probe --repo /content/RFforNetwork \
  --windows /content/datasets/CICIoMT24/WINDOWS_CPLUS \
  --manifest-root /content/RFforNetwork/configs/cplus_manifest \
  --config FW_256_FS_8 --experiment baseline_svm \
  --output /content/stage72_runs/baseline_svm_FW256_FS8_scaling01.json \
  --sizes 16000 32000 64000 --seed 42 --timeout-per-size 1200 --max-rss-gib 12
```

Sau khi có đủ hai JSON, báo cáo read-only:

```bash
python /content/RFforNetwork/huyen_code/ciciomt24_svm_scaling_probe.py review \
  --baseline /content/stage72_runs/baseline_svm_FW256_FS8_scaling01.json \
  --compact /content/stage72_runs/mi_rfe_svm_FW256_FS8_scaling01.json
```

Gửi ChatGPT: hai JSON tổng, các JSON worker nếu FAIL, text của cell REVIEW,
output unit tests và tài nguyên runtime thực tế. Không gửi CSV Train/Test.

## Tiêu chí trước khi xin xem xét Stage 7.3

Tests do bạn chạy thực sự PASS; cả hai JSON đúng experiment/config/seed và đủ
16k/32k/64k; sample actual=requested, Window_ID unique, Benign/Attack được
ghi; mọi SVC hội tụ, không timeout/RSS cap/low-RAM, đủ fit/predict/MI/RFE
measurement; Validation/Test CSV rows read đều 0; source/manifest hash đúng;
không ghi đè run cũ. Dù vậy vẫn **REVIEW_REQUIRED**: Stage 7.3/full SVM chỉ
có thể bắt đầu sau quyết định riêng của người dùng và đánh giá tài nguyên.
Không suy thời gian/RAM full từ ba điểm theo tuyến tính và không dùng probe
để chọn feature/best_k hay Test accuracy.
