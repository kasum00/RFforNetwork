# Stage 7.2 — Resource feasibility review (template)

**Trạng thái: REVIEW_REQUIRED.** Chưa chạy scaling probe trong Stage 7.2;
không có kết quả 16k/32k/64k. Không phải báo cáo Accuracy hay Test.

Sau khi người dùng tự chạy cả hai probe, đính kèm JSON tổng/worker và output
`review` read-only. Chỉ điền bảng dưới đây từ artifact đã xác minh; không
ngoại suy thành cam kết full SVM.

| Nhánh | Size yêu cầu/thực tế | Fit time | Predict time | Peak tree RSS | Support vectors / fit rows | Converged/warnings |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | 16k | Chưa chạy | — | — | — | — |
| Baseline | 32k | Chưa chạy | — | — | — | — |
| Baseline | 64k | Chưa chạy | — | — | — | — |
| MI-RFE-SVM | 16k | Chưa chạy | — | — | — | — |
| MI-RFE-SVM | 32k | Chưa chạy | — | — | — | — |
| MI-RFE-SVM | 64k | Chưa chạy | — | — | — | — |

Đối chiếu tăng trưởng 16k→32k→64k: **chưa có dữ liệu**.

- CPU/RAM/runtime thực tế: chưa có dữ liệu.
- Tỷ lệ Benign/Attack full Train so với sample/fit/holdout: chưa có dữ liệu.
- MI/RFE partial time/RSS, bốn RF fits và k=1/10/100: chưa có dữ liệu.
- Validation/Test CSV rows read, source/manifest hashes, timeout/RSS stop:
  chưa có dữ liệu Stage 7.2.
- Rủi ro còn lại: RBF/libsvm tăng phi tuyến, memory cache là cap chứ không phải
  allocation cố định; class-aware sample có thể lệch phân phối; polling RSS
  không phải OS hard limit; capture/session independence chưa chứng minh.
- Full Train FW256/FS8 ~1.348.317 window, Benign theo manifest ~3.182;
  không cam kết feasibility từ mẫu tối đa 64k.

**Quyết định:** Không đặt `svm_full_resource_gate_reviewed=True`. Chỉ xem xét
Stage 7.3 sau khi hai JSON đầy đủ, unit tests và resource review được người
dùng kiểm tra, rồi cấp phép riêng. Không dùng Test chọn feature, k hay policy.
