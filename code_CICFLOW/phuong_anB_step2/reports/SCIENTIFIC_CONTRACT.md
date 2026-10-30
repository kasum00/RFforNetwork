# Scientific contract — phương án B step2, policy gốc

## Nguồn đối chiếu

Đã đọc notebook E:/PHUONG_ANA_A1_COLAB_CLEAN_V2 (1).ipynb và source A1/A0/core.
Sau đó người dùng cung cấp toàn bộ run_phuong_anB.py trong cuộc trò chuyện, xác nhận
COARSE=(10,20,30,50,75,100), RADIUS=5 và chỉ đổi RF-RFE step1→step2.
Không có local file run_phuong_anB.py tại những vị trí source đã tìm; đối chiếu dựa
trên nội dung được dán, không giả khẳng định đọc một local file hoặc tính SHA toàn
bộ source gốc chưa lưu. evidence/original_policy_excerpt.py giữ nguyên constants và
hai pure helpers fine_grid/choose_best từ nội dung đó cho regression tests.

Notebook B gốc dùng CICIoMT2024 và cơ chế reuse original artifacts. Implementation
Colab này chỉ giữ policy/numerical contracts được yêu cầu, dùng adapter dataset
CICFLOW của A1. Không đưa CICIoMT2024 paths, counts, batch ID, Windows OS locks hoặc
step1 RFE checkpoints gốc sang CICFLOW.

## Dataset CICFLOW giữ nguyên

| Config | Train | Validation | Test theo manifest |
|---|---:|---:|---:|
| FW256_FS8 | 326157 | 104147 | 103494 |
| FW256_FS16 | 163203 | 52145 | 51824 |
| FW512_FS8 | 311423 | 96408 | 95865 |
| FW512_FS16 | 155814 | 48258 | 47991 |

Các số trên được đọc từ local manifest, không phải benchmark/actual Parquet read.
Runtime tiếp tục kiểm tra với metadata gốc; không dùng số khác hoặc sampling.
Dataset root duy nhất MyDrive/CICFLOWMETER_FEATURE_CSV. 380 feature + Window_Label,
row order, split/embargo, sidecars/CSV/proxy labels/preprocessing giữ nguyên.
28 sidecar nhỏ đã được đọc/hashes ghi vào evidence/dataset_reference.json. Codex
không mở hoặc copy Parquet. User preflight kiểm tra full Train/Validation SHA và
footer/schema; Test chỉ stat + declared hash metadata, không open/hash/footer.

Resolver FW512_FS8 xử lý input_paths tuyệt đối Windows bằng original frozen bytes
được kiểm tra SHA/size/mapping rồi dùng evidence_root hỗ trợ sẵn; không sửa sidecar,
không rewrite fingerprint để ép PASS. Frozen parent manifests bind label producer,
split membership/source ranges và counts như adapter A1.

## MI, scaler và model parameters

MI gọi core.reference_mi/saved_mi trên matrix Train380, một lần mutual_info_classif
sau StandardScaler, random_state=42, dense auto/n_neighbors=3 từ đúng profile.
Feature order, pandas sort tie semantics, Top100 giữ nguyên. Không per-feature seeds.

Top100 scaling/ranked arrays/svc_prefix dùng đúng helpers A1 được khóa SHA. Train
fit scaler, Validation transform; float32 layout/arithmetic trước C-float64 SVC.
RF RFE estimator100/depth15/balanced/seed42/n_jobs=-1, SVM RBF C10/max_iter20000 và
các default còn lại không đổi. Không áp RF classifier300/depth20 của nhánh khác.
Chỉ Train fit MI/RF/SVM; Validation chỉ chấm ứng viên.

## RF-RFE step2 là điều chỉnh duy nhất về thuật toán B gốc

Step1→2: square feature_importances, np.argsort mặc định tương ứng installed sklearn
RFE, loại min(2,remaining-1), increment native ranks, final one-feature estimator fit.
Lưu active/removed/retained indices, scores, support và ranks mỗi vòng. Top100 có
50 elimination và 1 fit cuối =51; step1 gốc100fits, A1step5=21fits.

Step2 tạo các nhóm rank thay vì unique permutation. Total order trong rank group
như A1: rank tăng, removal squared importance giảm, MI position tăng. Đây là cách
xác định prefix cho k lẻ khi step2 có tie group, được ghi trong scientific contract.
Không tái sử dụng ranking step1; không tuyên bố ranking/metrics tương đương tuyệt đối.

## Coarse/fine khớp source được dán

1. Coarse SVM đúng thứ tự 10,20,30,50,75,100, mỗi candidate một fit Train và predict
   full Validation; saved valid candidate cùng run thì load, không fit lại.
2. Coarse best Accuracy cao nhất, hòa nhỏ nhất k, như choose_best() sắp xếp k trước
   strict-greater của source gốc.
3. Fine list chính xác:
   range(max(1,coarse_best_k-5), min(100,coarse_best_k+5)+1), bỏ các k trong COARSE.
   Thứ tự tăng dần; không mở rộng vùng sau khi fine có winner mới.
4. Fine đã có checkpoint hợp lệ được khôi phục, không fit lại. Best cuối trên tất
   cả coarse+fine rows theo cùng Accuracy/max và k/min tie-break.

Ví dụ coarse_best=30 → fine25,26,27,28,29,31,32,33,34,35; coarse_best=100 → fine95..99.
Nếu sáu coarse bằng điểm, coarse_best=10; fine5..9,11..15; best cuối có thể là5 nếu
các điểm fine cũng cùng Accuracy. Không refine quanh tất cả coarse đồng hạng.

RF step2 độc lập coarse grid; không dùng 100,98,… hoặc grid cách2. Coarse6 và fine
5/10 cho tổng unique SVM11/16 của selection hoàn tất. Không exhaustive1..100; k1..4
không thuộc policy này khi coarse anchor nhỏ nhất10. Không đảm bảo optimum exhaustive,
không tự thêm k khác ngoài source gốc để cải thiện Accuracy.

## Identity và trạng thái

Algorithm MI_RFE_SVM_B_STEP2_COARSE6_RADIUS5_V2, format phuong_anB_step2.selection.v2.
Policy version B.step2.coarse6.radius5.v2 bind coarse/radius/selection rule cùng
RFE step2. Các proposed draft policies đã loại khỏi production; checkpoints draft
hoặc A1/step1/source mismatch đều BLOCKED, không rewrite artifact.

Không final refit Train+Validation hoặc Test; không automatic official reproduction.
EXPERIMENTAL_NON_OFFICIAL/OFFICIAL_BLOCKED, proxy-label/capture limitations giữ nguyên.
Regression tests chỉ được viết, chưa chạy. Không benchmark hoặc tuyên bố speedup.
