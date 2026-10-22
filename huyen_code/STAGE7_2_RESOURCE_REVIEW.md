# Stage 7.2 — Resource feasibility review, FW_256_FS_8

**Đủ hai nhánh technical probe 16k/32k/64k; full SVM vẫn REVIEW_REQUIRED.** Nguồn là các JSON người dùng cung cấp ở `E:\`, không phải huấn luyện full hay Test accuracy. Kiểm định chi tiết trong [STAGE7_2_REPORT.md](STAGE7_2_REPORT.md). File `mi_rfe_svm_FW256_FS8_scaling01.json` mang sai identity và **không được dùng**; nhánh MI-RFE đúng là `mi_rfe_svm_FW256_FS8_scaling02.json`.

| Nhánh | Sample / fit rows | Fit time SVC s | MI s | Partial RFE s | Peak tree RSS MiB | Support vectors |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| Baseline | 16k / 12.800 | 380 feature: 12,55 | — | — | 370,4 | 556 |
| Baseline | 32k / 25.600 | 380 feature: 46,41 | — | — | 530,1 | 856 |
| Baseline | 64k / 51.200 | 380 feature: 131,90 | — | — | 894,4 | 1.209 |
| MI-RFE-SVM | 16k / 12.800 | k1/10/100: 0,050 / 0,074 / 0,195 | 19,97 | 1,31 | 440,6 | 54 / 119 / 220 |
| MI-RFE-SVM | 32k / 25.600 | 1,650 / 0,154 / 0,617 | 42,30 | 1,59 | 647,1 | 2.322 / 75 / 331 |
| MI-RFE-SVM | 64k / 51.200 | 4,070 / 0,477 / 2,110 | 88,43 | 2,07 | 1.066,5 | 2.456 / 127 / 494 |

Tổng elapsed Baseline **515,28 s**, MI-RFE-SVM **485,05 s**, gồm ba lượt đọc Train. MI peak RSS 459,0/650,7/1.132,1 MiB. Runtime Colab: 4 CPU vật lý, 50,99 GiB RAM tổng và khoảng 48,8 GiB available trước run. Cap 12 GiB và timeout 1.200 s/worker không bị chạm; tất cả SVC fit hội tụ, không warning.

Baseline fit time tăng 3,70× rồi 2,84× khi gấp đôi mẫu; MI time tăng 2,12× rồi 2,09×, peak tree RSS MI-RFE tăng 1,47× rồi 1,65×. Support vectors k=1 nhảy 54→2.322→2.456; partial ranking được tính lại ở từng size, nên không coi là một đường scaling với feature cố định. Đây chỉ là **technical sampling**: 64k bằng 4,75% full Train 1.348.317 window, Benign rất hiếm, nhãn `SCENARIO_PROXY`.

Hai nhánh dùng cùng hash sample Window_ID và cùng source/data hashes trong Colab. Tuy nhiên hash `ciciomt24_training.py` Colab khác bản Windows hiện tại, chưa đối chiếu source/commit thực tế. RSS polling không phải hard limit; SVC RBF có thể tăng phi tuyến; full MI-RFE phải chạy ranking đầy đủ và k=1..100, không phải partial bốn RF fits. Capture/session independence chưa chứng minh. Không dùng Test hay probe để chọn best_k.

**Quyết định:** Stage 7.2 technical evidence **đầy đủ**; `resource_gate=REVIEW_REQUIRED`, `full_training_authorized=false`, `official_best_k=null`. **Không đặt `svm_full_resource_gate_reviewed=True` và không tự chạy Stage 7.3/full SVM.** Cần đối chiếu source và quyết định tài nguyên/thiết kế riêng của người dùng.
