# Stage9 source/checkpoint review

Ngày soạn: 2026-11-02. Phương pháp: đọc trực tiếp source và notebook; không thực thi.

## Source đã đối chiếu

| Source | Bằng chứng dùng cho Stage9 |
|---|---|
| `colab/cicflow_reference_pipeline.py:191` | Contract preprocessing, MI, Top100 scaler, final role Train+Validation |
| `colab/cicflow_reference_pipeline.py:400` | `read_values`: cột yêu cầu theo thứ tự, cleaning, Test authorization |
| `colab/cicflow_reference_pipeline.py:743` | `final_refit`: concatenate Train rồi Val, C-float32, StandardScaler mới, SVC unchanged, save model/scaler |
| `phuong_anA/pipeline.py:195` | `verify`: ledger SHA/parent/identity, reconstruct small immutable state, validate exhaustive selection, compare publication |
| `phuong_anA/results.py` | MI/scaler/RFE binding, k history, frozen feature prefix, full Validation metrics |
| `phuong_anA/store.py` và `migration.py` | Immutable checkpoint chain, migrated k1 remains bound; completed FIX01 can verify without opening deleted origin |
| `phuong_anB_step2/pipeline.py:232` | B verifier additionally enforces individually committed sparse candidates and fine-plan ordering |
| `phuong_anB_step2/policy.py` | Coarse6, radius5, strict-greater Accuracy/min-k tie, no repeated valid candidate |
| A1 CLEAN_V2 và B STEP2 notebooks | Actual roots, profiles, selected source/commit, independent output layouts |

Không đồng nhất A1/B schema một cách ngầm định: Stage9 chọn module theo method; extraction đọc trường đã kiểm tra qua verifier của đúng phương án. Version khác/thiếu module/format thì BLOCKED.

## Giải quyết contract final scaler

Selection A1/B ghi `NOT_AUTHORIZED_SELECTION_ONLY` để chặn final trong notebook selection. Đây là giới hạn quyền của các notebook đó. Core reference có hợp đồng final rõ ràng và task Stage9 chủ động cấp giai đoạn final riêng: scaler mới fit full Train+Validation sau freeze. Stage9 lưu protocol này trong identity/checkpoint; không sửa contract hoặc artifact selection. Nếu archived core không có đúng final protocol này thì BLOCKED.

Stage9 helper hashes được khóa trong `reference_helpers.json`; original selection source phải có đúng saved commit, clean Git và toàn bộ source snapshot bytes khớp trước import. Numerical library versions và Python version phải khớp selection. OS/kernel lịch sử được giữ làm provenance, không giả rằng kernel của session mới bằng kernel cũ. Environment hiện tại được khóa trong Stage9 identity để resume.

Byte hashes của core preprocessing/final helper và common loader trong archived selection source còn phải trùng các helper Stage9 đang gọi. Chỉ cùng chuỗi contract hoặc cùng SVC parameters chưa đủ để coi implementation tương thích. Nếu helper khác, Stage9 BLOCKED để đối chiếu source; không tự dùng helper mới thay cho giao thức cũ.

## Checkpoint thực tế: chưa được đọc trên Drive trong nhiệm vụ này

| Method | Config | Run tham khảo để kiểm tra | Trạng thái hiện tại của audit này |
|---|---|---|---|
| A1 | FW256_FS8 | RUN_A1_01_FIX01 | Người dùng từng cung cấp verify selection PASS; chưa tái xác minh cho Stage9 |
| A1 | FW256_FS16 | RUN_A1_02 | Chờ discovery + archived verifier |
| A1 | FW512_FS8 | RUN_A1_01 | Chờ discovery + archived verifier |
| A1 | FW512_FS16 | RUN_A1_01 | Chờ discovery + archived verifier |
| B | FW256_FS8 | Đọc tên run thực tế; notebook đề xuất RUN_B2_01 | Chờ discovery + archived verifier |
| B | FW256_FS16 | Đọc tên run thực tế | Chờ discovery + archived verifier |
| B | FW512_FS8 | Đọc tên run thực tế | Chờ discovery + archived verifier |
| B | FW512_FS16 | Đọc tên run thực tế | Chờ discovery + archived verifier |

Các k A1 88/98/90/51 và B FW512_FS8=100 do người dùng quan sát không hardcode trong production; không xác nhận usable chỉ dựa vào những số này. Chỉ Cell7 verify PASS + Cell9 preflight READY mới cho phép chuyển sang xác nhận final fit.

## Blocker được giữ

- Selection chưa complete/không có full verified chain → BLOCKED_SELECTION_INCOMPLETE hoặc BLOCKED_SELECTION_NOT_VERIFIED.
- Sai dataset/split/schema/label producer/source/commit/library/Python/ranking/feature mapping/scaler → BLOCKED, không bypass.
- Không có exact original commit trên remote, không có metadata gốc hoặc biến thể B khác algorithm khóa → BLOCKED.
- Stage9 namespace đã có identity khác/partial corrupt artifact → BLOCKED, không đổi Run ID để đè.
- Claim Test không có completion hợp lệ → BLOCKED_AMBIGUOUS_TEST_CLAIM, không tự retry.
- Không có log unit-test PASS gắn source/environment hiện tại → chưa được fit/Test.
- Proxy labels và capture independence chưa thành ground truth; kết quả không trở thành official reproduction sau Test.

Không có kết luận reference-equivalence, tốc độ, Test Accuracy hay tám run hoàn thành từ lần đọc source này.
