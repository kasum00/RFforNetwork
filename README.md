# COMPACT-MI-RFE trên CICIoMT2024

Repository này tái hiện pipeline **COMPACT-MI-RFE** từ mã CTU-13 trên bộ dữ liệu **CICIoMT2024**, với trọng tâm là chia dữ liệu theo thời gian, hạn chế data leakage do sliding window chồng lấn, lựa chọn đặc trưng bằng Mutual Information và RF-RFE, sau đó đánh giá bằng SVM hoặc Random Forest.

> **Cảnh báo khoa học:** nhãn sử dụng trong thực nghiệm là `SCENARIO_PROXY` suy ra từ tên kịch bản, không phải ground truth đã được xác minh cho từng flow. Pipeline đã kiểm soát source-row/window overlap nhưng chưa chứng minh các capture hoặc session hoàn toàn độc lập.

## Trạng thái hiện tại

| Hạng mục | Trạng thái |
|---|---|
| Audit 71 CSV / 19.891.102 flow | Hoàn tất |
| Chia C+ Train/Validation/Test và embargo | Hoàn tất, đã kiểm định full dataset |
| Sinh bốn cấu hình sliding window | Hoàn tất, 6.737.006 window |
| MI-RFE + RF, FW512/FS16 | Hoàn tất và kiểm định |
| MI-RFE + RF, FW256/FS16 | Hoàn tất và kiểm định |
| MI-RFE + RF, hai cấu hình FS8 | Chưa có đủ kết quả chính thức trong repository này |
| SVM bounded scaling probe | Hoàn tất đến 64.000 Train window |
| SVM full/resumable | Đang chạy theo checkpoint; chưa có kết quả Test chính thức |

Ảnh trạng thái gần nhất do người dùng cung cấp ngày 26/10/2026 cho thấy một lượt **MI-RFE + SVM, FW256/FS8** đang ở `k=24`, đã hoàn tất `k=1..23`, checkpoint gần nhất là `k_search/k_023`, chưa có lỗi. Đây là trạng thái vận hành tạm thời, **không phải kết quả thực nghiệm cuối cùng**.

## 1. Mục tiêu nghiên cứu

Ba thí nghiệm được tái hiện theo mã CTU-13:

1. **SVM Baseline** với toàn bộ 380 đặc trưng window.
2. **Mutual Information → RF-RFE → SVM**.
3. **Mutual Information → RF-RFE → Random Forest**.

Nguyên tắc chung:

- Không giảm mẫu hoặc undersampling trong run chính thức.
- Mutual Information và RFE chỉ học trên Train.
- Validation chỉ dùng để chọn số đặc trưng `k`.
- Test không dùng để chọn đặc trưng, tham số hoặc cấu hình.
- Sau khi khóa `k` và feature, scaler/classifier được fit lại trên Train+Validation.
- Test chỉ được đọc và đánh giá một lần sau final refit.

## 2. Pipeline tổng thể

```text
71 CSV CICIoMT2024 — 19.891.102 flow
                    │
                    ▼
       Audit schema, timestamp, duration,
       Flow ID, 5-tuple và provenance
                    │
                    ▼
       Gán nhãn SCENARIO_PROXY theo scenario
                    │
                    ▼
       Chia C+ theo block thời gian liên tục
       + timestamp bucket guard
       + embargo tối thiểu 511 flow
       + Flow Duration + sai số 1 giây
                    │
                    ▼
       Train / Validation / Test / Embargo
                    │
                    ▼
       Sliding window riêng trong từng split
       và từng source segment
                    │
                    ▼
       76 feature flow × 5 thống kê
       = 380 feature/window
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
      Baseline   MI-RFE     MI-RFE
        SVM       + SVM      + RF
          │         │         │
          └─────────┼─────────┘
                    ▼
       Chọn cấu hình bằng Validation
                    │
                    ▼
       Khóa feature/k → refit Train+Val
                    │
                    ▼
       Đánh giá Test đúng một lần
```

## 3. Các giai đoạn đã thực hiện

### Giai đoạn 1 — Audit bài báo, mã CTU-13 và dữ liệu

- Đối chiếu bài báo với năm script trong [`CODE_ctu13/`](CODE_ctu13/).
- Kiểm kê 71 CSV với tổng cộng **19.891.102 flow**.
- Kiểm tra schema, Timestamp, Flow Duration, Flow ID, 5-tuple và nhãn gốc.
- Tạo mapping scenario thành `Benign` hoặc `Attack`, đánh dấu `SCENARIO_PROXY`.
- Không ghi đè CSV hoặc cột Label gốc.

### Giai đoạn 2 — Thiết kế B2 và phát hiện nguy cơ leakage

Không chia ngẫu nhiên ở mức window. Với `FW=512`, `FS=8`, hai window liên tiếp dùng chung 504/512 flow, tương đương **98,4375%**. Chia ngẫu nhiên sau khi tạo window có thể đưa hai mẫu gần như giống nhau vào Train và Test.

Thiết kế B2:

- Sắp xếp ổn định theo Timestamp và giữ `Source_CSV`/`Original_Row_Index`.
- Giữ block thời gian liên tục, không shuffle từng dòng.
- Không cắt một timestamp bucket sang hai split.
- Giữ sáu nhóm nhỏ trong Train theo thiết kế đã duyệt.
- Dùng embargo tối thiểu 511 flow tại ranh giới phù hợp.

### Giai đoạn 3 — Flow Duration, full metadata audit và C+

- Đối chiếu CICFlowMeter và dữ liệu thực; `Flow Duration` được xử lý theo microsecond.
- Tính biên hoạt động bảo thủ:

```text
Flow_End_Time  = Timestamp + Flow_Duration_us / 1.000.000
Flow_End_Upper = Flow_End_Time + 1 giây
```

- Phân biệt time overlap, dùng lại 5-tuple và cùng session; ba điều này không tương đương.
- Audit metadata toàn bộ 19.891.102 flow theo cách tiết kiệm RAM.
- So sánh B2 gốc và các hướng mở rộng embargo.
- Tối ưu cutoff thành **C+**, hướng tới 60:20:20 trên phần dữ liệu sử dụng và không cho activity crossing theo guard bảo thủ.

### Giai đoạn 4 — Full split C+

| Phân vùng | Số flow | Tỷ lệ trong phần được sử dụng |
|---|---:|---:|
| Train | 10.798.603 | 60,00% |
| Validation | 3.599.535 | 20,00% |
| Test | 3.599.534 | 20,00% |
| Embargo | 1.893.430 | Không dùng để học/đánh giá |
| **Tổng nguồn** | **19.891.102** | — |

Kiểm định:

- Mỗi source row thuộc duy nhất Train, Validation, Test hoặc Embargo.
- Không dùng chung source row giữa các split.
- Không cắt timestamp bucket.
- Không có segment chồng lấn.
- Không có activity crossing theo guard đã duyệt.
- Có thể truy ngược output về CSV nguồn và chỉ số dòng gốc.
- CSV nguồn không bị sửa đổi.

Dữ liệu dẫn xuất nằm trong `CICIoMT24/CSV_SPLIT_CPLUS/` và không được commit lên Git.

### Giai đoạn 5 — Full sliding window

| FW/FS | Train | Validation | Test | Tổng |
|---|---:|---:|---:|---:|
| 256/8 | 1.348.317 | 449.343 | 449.251 | 2.246.911 |
| 256/16 | 674.169 | 224.677 | 224.631 | 1.123.477 |
| 512/8 | 1.347.101 | 448.735 | 448.563 | 2.244.399 |
| 512/16 | 673.559 | 224.373 | 224.287 | 1.122.219 |
| **Tổng** | — | — | — | **6.737.006** |

Với segment có `N` flow:

```text
windows = floor((N - FW) / FS) + 1, nếu N >= FW
windows = 0, nếu N < FW
```

Mỗi window sử dụng 76 đặc trưng hành vi và năm thống kê:

```text
mean, std, median, skew, kurtosis
```

Mỗi window có `76 × 5 = 380` feature. Nhãn window dùng existential rule: có ít nhất một flow Attack thì window là Attack; chỉ window gồm toàn bộ flow Benign mới là Benign.

### Giai đoạn 6 — Chuẩn bị feature matrix

`X` chỉ chứa đúng 380 đặc trưng thống kê. Label, Window ID/Index, Num Flows, Scenario, Split, Source CSV/segment và toàn bộ provenance bị loại khỏi mô hình.

Tiền xử lý:

1. Ép feature về numeric.
2. Chuyển `+Inf/-Inf` thành `NaN`.
3. `fillna(0)`.
4. Kiểm tra ma trận hữu hạn trước khi fit.

### Giai đoạn 7 — MI-RFE + Random Forest chính thức

```text
Train 380 feature
→ StandardScaler + Mutual Information trên Train
→ MI Top 100
→ scaler Top100 mới fit Train
→ RF-RFE step=1 tạo ranking cố định
→ thử k=1..100 bằng RF trên Validation
→ chọn Validation Accuracy cao nhất, hòa chọn k nhỏ hơn
→ khóa k và feature
→ fit scaler + RF mới trên Train+Validation
→ đọc và đánh giá Test một lần
```

RF-RFE:

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
)
```

RF classifier:

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=3,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
)
```

### Giai đoạn 8 — Resource audit cho FS8

- Preflight FW512/FS8 cần khoảng 23,99 GiB RAM available trong khi máy có khoảng 23,73 GiB usable.
- Memory lab thử các thay đổi kỹ thuật trên mẫu nhưng không đủ bằng chứng để hạ guard.
- Không giảm dataset, đổi MI hoặc đổi thuật toán để ép run.
- Benchmark kỹ thuật không được trình bày như kết quả chính thức.

### Giai đoạn 9 — Colab, SVM probe và SVM resumable

- Tạo notebook và adapter cho Google Colab/Drive.
- Audit riêng SVM Baseline và MI-RFE + SVM.
- Chạy bounded technical probe trên 16k/32k/64k Train window; không dùng Validation/Test chính thức và không chọn `best_k` từ probe.
- Triển khai authorization fail-closed, resource budget, supervisor, checkpoint và late-Test guard.
- Triển khai SVM nhiều phiên: tái dùng checkpoint đã hoàn tất và tiếp tục từ `k` đầu tiên chưa hoàn thành.
- Không thể resume bên trong một lần `SVC.fit`; fit dở phải chạy lại trong phiên được phê duyệt.

Các stage checkpoint:

```text
data_prepared
mi
rfe
k_search/k_001 ... k_search/k_100
selection
final
final_test
```

Checkpoint `.incomplete` không được coi là kết quả. Nếu Test đã được claim nhưng chưa có kết quả validated, pipeline dừng để manual review và không tự đánh giá Test lần hai.

## 4. Kết quả MI-RFE + Random Forest chính thức

| Chỉ số | FW512/FS16 | FW256/FS16 |
|---|---:|---:|
| Train window | 673.559 | 674.169 |
| Validation window | 224.373 | 224.677 |
| Test window | 224.287 | 224.631 |
| Best k | **8** | **18** |
| Validation Accuracy | 100% | 100% |
| Test Accuracy | 99,998662% | 99,976851% |
| Test Macro-F1 | 99,839580% | 97,263108% |
| Test Balanced Accuracy | 99,680851% | 94,820717% |
| Recall Benign | 99,361702% | 89,641434% |
| Recall Attack | 100% | 100% |
| Tổng thời gian | ~42 phút 9 giây | ~41 phút 17 giây |
| Peak process RSS | ~9,99 GiB | ~10,01 GiB |

Confusion matrix theo thứ tự `[Benign, Attack]`:

```text
FW512/FS16: [[467,  3],
             [  0, 223817]]

FW256/FS16: [[450, 52],
             [  0, 224129]]
```

Hai run dùng toàn bộ dữ liệu, thử đủ `k=1..100`, không dùng Test để chọn feature/k, final refit trên Train+Validation và chỉ đánh giá Test một lần.

Accuracy phải được đọc cùng Macro-F1, Balanced Accuracy và Recall Benign vì Test bị mất cân bằng nghiêm trọng.

## 5. Cấu hình SVM được bảo toàn

Tham số cốt lõi:

```python
C = 10.0
kernel = "rbf"
gamma = "scale"
max_iter = 20000
```

Baseline và MI-RFE + SVM giữ khác biệt có trong source CTU-13; không tự đồng nhất `probability` hoặc `class_weight`.

RF-RFE của nhánh MI-RFE + SVM:

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)
```

RFE dùng `step=1`, xếp hạng 100 feature MI và thử `k=1..100` bằng Validation Accuracy.

## 6. Kiểm soát data leakage

1. Chia flow trước khi tạo sliding window.
2. Không dùng chung source row giữa các split.
3. Không tạo window vượt split hoặc source segment.
4. Không cắt timestamp bucket.
5. Dùng embargo tối thiểu 511 flow và activity guard.
6. Không đưa provenance/metadata vào `X`.
7. Scaler, MI và RFE chỉ fit Train trong model selection.
8. Validation chỉ chọn `k`/cấu hình.
9. Final refit chỉ sau khi feature đã khóa.
10. Test được nạp muộn và đánh giá đúng một lần.

Các kiểm tra này **không chứng minh capture/session độc lập**. Metadata hiện có không đủ để xác nhận toàn bộ session; trùng 5-tuple không tự động là cùng session.

## 7. Cấu trúc repository

```text
COMPACT-MI-RFE/
├── CODE_ctu13/          # Năm script CTU-13 gốc
├── huyen_code/          # Split, window, training, tests và báo cáo
├── colab/               # Adapter, authorization, supervisor, resumable
├── notebooks/           # Notebook Colab
├── local/               # Bộ chạy Windows local/máy ảo
├── configs/             # Manifest/config nhỏ được version-control
├── CICIoMT24/           # Dữ liệu cục bộ; không đưa lên Git
└── README.md
```

Dataset, window CSV, cache, model, checkpoint, approval và kết quả runtime lớn không được đưa lên GitHub. Chúng được bao phủ bởi [`.gitignore`](.gitignore). Không commit secret, token Drive hoặc approval của một máy cụ thể.

## 7.1. Bản đồ repository và cleanup

Audit cấu trúc, dependency, Stage và phạm vi thay đổi được duy trì tại [`docs/repository_cleanup/`](docs/repository_cleanup/):

- [Repository audit](docs/repository_cleanup/REPOSITORY_AUDIT.md)
- [Dependency map](docs/repository_cleanup/DEPENDENCY_MAP.md)
- [Stage index](docs/repository_cleanup/STAGE_INDEX.md)
- [Cleanup plan](docs/repository_cleanup/CLEANUP_PLAN.md)
- [Cleanup report](docs/repository_cleanup/CLEANUP_REPORT.md)
- [Second-dataset readiness](docs/repository_cleanup/SECOND_DATASET_READINESS.md)

Các tài liệu này là bản đồ điều hướng và kế hoạch migration; chúng không thay thế manifest, checkpoint hoặc report khoa học gốc.

## 8. Báo cáo chính

- [Full split C+](huyen_code/STAGE34_REPORT.md)
- [Full sliding window](huyen_code/STAGE42_REPORT.md)
- [Audit trước huấn luyện](huyen_code/STAGE51_REPORT.md)
- [MI-RFE + RF FW512/FS16](huyen_code/STAGE52_REPORT.md)
- [MI-RFE + RF FW256/FS16](huyen_code/STAGE53_REPORT.md)
- [Resource preflight FW512/FS8](huyen_code/STAGE54_PREFLIGHT_REPORT.md)
- [Memory optimization lab](huyen_code/STAGE55_MEMORY_OPTIMIZATION_REPORT.md)
- [Triển khai Colab](STAGE6_REPORT.md)
- [SVM source audit](huyen_code/STAGE7_1_SVM_AUDIT.md)
- [SVM bounded scaling probe](huyen_code/STAGE7_2_REPORT.md)
- [SVM official preflight](huyen_code/STAGE7_3A_REPORT.md)
- [SVM controlled implementation](huyen_code/STAGE7_3B_REPORT.md)
- [SVM resumable guide](huyen_code/SVM_RESUMABLE_GUIDE.md)
- [SVM resumable report](huyen_code/SVM_RESUMABLE_REPORT.md)
- [Windows local guide](local/README_WINDOWS_LOCAL.md)

## 9. Trình tự tái hiện an toàn

1. Clone đúng commit được phê duyệt.
2. Tạo môi trường Python theo requirements tương ứng.
3. Đặt dataset/checkpoint đúng cấu trúc; không commit chúng.
4. Xác minh SHA256 source, manifest, validation và window summary.
5. Chạy unit test và lưu evidence gắn với đúng commit.
6. Chạy preflight; mọi mục bắt buộc phải PASS.
7. Kiểm tra RAM, disk, budget và worker cũ.
8. Tạo approval cho đúng experiment/config/run/session.
9. Chạy duy nhất một experiment/config mỗi lượt.
10. Theo dõi progress/checkpoint; không xóa run dở.
11. Chỉ publish khi final refit, một lần Test evaluation và validation đều PASS.

Không sửa cờ FAIL thành PASS, không dùng probe như kết quả chính thức và không dùng Test để quyết định cấu hình.

## 10. Giới hạn

- `SCENARIO_PROXY` có thể không phản ánh đúng từng flow.
- Benign rất hiếm; Accuracy không đủ để đánh giá chất lượng.
- Window trong cùng split vẫn chồng lấn mạnh và không phải quan sát IID độc lập.
- Chưa chứng minh capture/session independence.
- Snapshot size/mtime không tương đương hash toàn bộ dataset.
- Bounded SVM probe không chứng minh full RBF-SVM khả thi.
- Kết quả phụ thuộc phiên bản thư viện, hệ điều hành và tài nguyên máy.

## 11. Tóm tắt

Pipeline C+ và 6.737.006 window đã hoàn tất. Hai thí nghiệm MI-RFE + Random Forest với FS16 đã có kết quả chính thức theo đúng thứ tự Train → Validation → final refit → Test. Nhánh SVM đã có audit, probe, authorization và checkpoint resumable; full result phải chờ run hoàn tất và kiểm định artifact trước khi công bố.
