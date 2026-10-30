# Checkpoint và resume B

Code đã viết, chưa thực thi tests/preflight/training. Policy gốc cố định coarse10,20,30,50,75,100 và radius5; RF-RFE step2 độc lập coarse grid.

## Đơn vị hoàn thành

- MI: một lần gọi trên Train380; checkpoint sau stage hoàn tất. Không có mid-MI fit resume.
- Scaler: Train Top100 scaler + binding, Validation transform only.
- RFE: mỗi vòng loại tối đa hai feature; final one-feature fit có final ranking artifact.
- SVM: một k gồm fit full Train, predict/metrics full Validation, local atomic record,
  rồi Drive checkpoint COMMITTED/readback trước khi bắt đầu k tiếp theo.
- Fine plan được COMMITTED sau đủ sáu coarse rows và trước fine candidate đầu tiên;
  bind digest coarse rows, coarse_best_k (hòa k nhỏ nhất), radius5 và danh sách fine
  còn thiếu. Fine winner mới không được làm thay đổi vùng quanh coarse_best_k.
- Session timing receipt và selection publication là các checkpoint riêng.

Không có checkpoint classifier giữa một SVC fit; interrupted fit chưa có candidate
record phải tính lại. Stage MI/RFE multi-file publication bị cắt dở mà không chứng
minh được consistency có thể BLOCKED để review; không xóa file để ép chạy.

## Mức local và durable

History là hash chain theo sequence/phase/k, không phải contiguous k=1..100.
Các fields science/metrics, feature prefix, dataset count/confusion matrix và timing
được kiểm tra trước khi một record được dùng lại.

Nếu upload lỗi sau local append, local record chưa là durable. Cùng local còn
nguyên thì resume kiểm tra identity/hash/ancestry và commit pending prefix trước
new fit. Mất runtime trước Drive commit thì record local không thể giả phục hồi;
candidate đó có thể cần fit lại. Sau commit mà process chết thì restore từ Drive
không fit lại candidate đã xác minh.

Checkpoint failed/pause chứa log/receipt, không bị tính là completed candidate.
Progress bar dùng durable units; local count và fit elapsed hiển thị riêng.

## Drive publication và non-overwrite

Delta checkpoint copy vào .incomplete_CP_*, kiểm tra bytes/SHA, ghi checkpoint.json
và COMMITTED.json, rename thành CP_*, rồi readback/parent-chain check trước ack.
Directory incomplete không được coi là completed, không tự xóa.
Drive FUSE/cloud replication durability chỉ bảo đảm ở mức API readback có thể kiểm
tra; không tuyên bố atomic cloud replication mạnh hơn giao diện cung cấp.

Run manifest, scientific files, checkpoint và completed publication bất biến.
Source/Git, step2, policy, dataset fingerprint, environment, config/run ID mismatch
trả BLOCKED; không auto migrate hoặc rewrite identity. RUN_A1_* bị API B từ chối.

## Lease/safe stop

Cell 11 REQUEST_STOP=True gửi STOP_REQUESTED và ngừng queue chưa bắt đầu; chờ
PAUSED + COMMITTED. STOP_AFTER_CONFIG=True cho config hiện tại complete/verify rồi
không bắt đầu config tiếp theo. Monitor interrupt không gửi stop.

Runtime mất có thể để lease chưa release; heartbeat cũ không chứng minh worker đã
dừng. Chỉ sau khi người dùng xác nhận runtime cũ thực sự dừng mới dùng phrase
PREVIOUS RUNTIME STOPPED <actual lease ID>. Không sửa file lease trực tiếp.

Resume giữ đúng source commit, policy, data/environment và Run ID B. Cell 12 chỉ
chuẩn bị acknowledge local stop; chạy lại Cell 7–8–9 để gate/xác nhận/run.
Cache arrays ở local có thể phải dựng lại, bằng đúng scaler đã lưu và bound data.
Không upload/copy các Parquet lớn hoặc cache arrays lên checkpoints.

## Thời gian và số fit

Statistics tách completed RF/SVM work khỏi recorded fit attempts. Một RF/SVC fit
bị ngắt có thể chưa để lại receipt, nên tổng attempts bị mất không được bịa.
Total selection seconds là tổng persisted session wall receipts tới work-complete
cutoff đầu tiên, gồm cache/checkpoint overhead trong các session đã lưu; loại
preflight, final publication/verify và thời gian hard interruption không lưu được.

Work-complete receipt khóa các prior receipt hashes. Resume publication không
được tạo timing khác để ghi đè báo cáo đã freeze; failure sau work complete được
ghi ở failure checkpoint/release thay vì sửa receipt completion.

Tất cả kết quả vẫn chỉ là Validation selection; không final refit hoặc Test.

## B gốc và version hiện tại

Algorithm MI_RFE_SVM_B_STEP2_COARSE6_RADIUS5_V2, FORMAT phuong_anB_step2.selection.v2.
Coarse6/radius5 khớp run_phuong_anB.py do người dùng cung cấp; thay đổi thuật toán chỉ
RF-RFE step1→2. Checkpoint/source bản nháp trước khi đối chiếu source gốc hoặc ranking
step1 không được nhập như tương thích. Giữ run/checkpoint cũ; mismatch trả BLOCKED,
không tự migration, sửa source SHA hoặc gán completed state.
