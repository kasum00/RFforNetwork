# CICFLOW Colab V2 — cấu trúc ColabPro theo từng experiment

Notebook chính: `COMPACT_MI_RFE_CICFLOW_Colab_V2.ipynb`.
Đã sửa tại chính file này; không tạo notebook V3.

## Những gì được sửa

Bản cũ dừng ở MI/RFE và lưu theo Stage. Bản này chạy đủ **MI → RFE → k=1..100 → freeze selection → final refit Train+Validation → Test một lần**, theo luồng 12 cell của `COMPACT_MI_RFE_ColabPro.ipynb`. Source gốc CTU-13, notebook ColabPro, Stage 0–4, dataset và các completed run cũ được giữ nguyên.

## Cấu trúc kết quả như notebook mẫu

```text
MyDrive/
├── code_CICFLOW/
│   ├── colab/
│   │   ├── cicflow_reference_pipeline.py
│   │   └── test_cicflow_reference_pipeline.py
│   ├── stage4_mi/mi_selection.py       # source hiện có, chỉ dùng schema helpers
│   ├── stage5_rfe/rfe_selection.py     # source hiện có, dùng elimination loop
│   └── logs/
│       ├── ALL_EXECUTION_RESULTS.log  # append stdout + stderr, native exit/status
│       └── colab_reference/<run_name>/
├── CICFLOWMETER_FEATURE_CSV/           # chỉ đọc
│   ├── FW256_FS8/
│   ├── FW256_FS16/
│   ├── FW512_FS8/
│   └── FW512_FS16/
├── checkpoints/
│   └── mi_rfe_rf_FW256_FS8_run01.incomplete/
│       ├── steps/rfe_001.json ... rfe_099.json
│       ├── steps/k_001.json ... k_100.json
│       └── các artifact đã hoàn thành theo từng bước
└── experiment_results/
    └── mi_rfe_rf_FW256_FS8_run01/
        ├── artifact_manifest.json
        ├── events.jsonl
        ├── features.json
        ├── final_scaler.joblib
        ├── k_search_results.json
        ├── mi_feature_scores.csv
        ├── mi_top100.json
        ├── model.joblib
        ├── preflight_audit.json
        ├── progress.json
        ├── result.json
        ├── rfe_feature_ranking.csv
        ├── selection_frozen.json
        ├── hyperparameters.json       # metadata bổ sung để truy vết
        ├── run_identity.json
        ├── resource_usage.json
        ├── COMPLETE.json
        └── các state/binding nhỏ để verify
```

SVM có thư mục riêng `mi_rfe_svm_FW256_FS8_run01`. Bốn cấu hình độc lập; **một lần Cell 10 chỉ chạy một model/config**, như Cell 8 của mẫu. Không tự tạo batch 8 experiment. Thư mục hoàn tất chỉ xuất hiện sau khi đã kiểm tra toàn bộ checkpoint và copy/hash artifact. Lưu trực tiếp vào checkpoint trên Drive trong lúc tính, không đợi đến cell cuối.

## Ánh xạ 12 cell

| Cell | Mục đích như mẫu | Thích ứng CICFLOW |
|---|---|---|
| 1 | Runtime | Python, library, CPU/RAM/disk |
| 2 | Drive mount | /content/drive |
| 3 | Dataset paths | MyDrive/code_CICFLOW, CICFLOWMETER_FEATURE_CSV |
| 4 | Source setup | Import đúng file Drive, không clone/reset/legacy patch |
| 5 | Dependencies/tests | Imports + fixture regression trong worker có log |
| 6 | Dataset readiness | Parquet footer/schema/count, không đọc row values |
| 7 | Scientific contract | MI, scalers, branch RF-RFE và classifiers |
| 8 | RunSpec | Model, FW/FS, runNN, mode |
| 9 | Preflight | RAM/disk, source, tests, identity, confirmation |
| 10 | Fresh worker | Toàn experiment MI/RFE/k-search/final/Test |
| 11 | Verify | Hash/size, mapping, k winner, one-Test event order |
| 12 | Summary | result.json/artifact metadata, không refit |

## Thay đổi khoa học cần biết trước khi chạy lại

**MI mới không dùng kết quả Stage 4 cũ.** Theo `huyen_code/ciciomt24_training.py:select_mi`:

1. Giữ 380 feature theo schema Train, dữ liệu float32 sau khi thay NaN/Inf bằng 0.
2. `StandardScaler().fit_transform(Train)`.
3. **Một lần** `mutual_info_classif(scaled, y, random_state=42)` trên đủ 380 cột.
4. Dùng default source `discrete_features='auto', n_neighbors=3`; preflight chặn nếu defaults thư viện khác. Không dùng seed riêng theo cột, không chia 380 MI calls.
5. `pd.Series(scores, index=names).sort_values(ascending=False)`, giữ đúng tie semantics của pandas đang cài, được bind bằng library versions.
6. Fit scaler mới trên Train Top100; RFE nhận scaled Train, Validation chỉ transform.

Bản Stage 4 cũ dùng từng feature và seed42+index, không external scaler; score/ranking có thể khác. Thay đổi được thực hiện trong **helper mới**, không sửa `stage4_mi/mi_selection.py` hoặc manifests/checkpoints cũ.

RF-RFE: 100 cây, depth10, split2, sqrt, balanced, seed42, n_jobs=-1.
SVM-branch RF-RFE: 100 cây, depth15, balanced, seed42, n_jobs=-1, phần còn lại theo defaults source/library đã ghi. Manual elimination dùng source Stage 5 với square importances/np.argsort, có checkpoint từng bước; fit cuối trên một feature như sklearn.RFE. Có regression so sklearn RFE, cả tie và resume, **chưa được Codex thực thi**.

**Classifier RF: 300 cây, max_depth=20, min_samples_split=3**, sqrt/balanced/seed42/n_jobs=-1.
Classifier SVM: C10, RBF, gamma=scale, probability=False, cache10000MB, shrinking=True, max_iter20000, balanced, seed42 — đúng COMPACT_SVM_PARAMS trong source tham chiếu. Không lấy tham số classifier thay cho estimator RFE.

Validation Accuracy lớn nhất chọn k, hòa giữ k nhỏ nhất; grid luôn 1..100. Freeze trước final refit. Final scaler và classifier fit trên Train+Validation với feature đã chốt; Test chỉ đọc sau đó và chỉ predict/evaluate một lần. Test intent được lưu trước khi mở Test. Nếu bị ngắt sau intent nhưng chưa có test_result.json, resume BLOCKED để không đánh giá lại Test.

XGBoost: siêu tham số classifier 200/depth10/min_child_weight3 đã được ghi trong shared contract, nhưng ranking contract chưa chốt; nhánh này BLOCKED, không thay bằng ranking RF/SVM.

## Resume và bảo toàn

Identity gồm run/config/model, hash source+tests, git commit do người dùng nhập, library/Python, ordered schema/Top100 dependencies, tham số và fingerprint dataset (stat + footer + small JSON hashes). Fingerprint **không phải full Parquet content hash**.

- Full mode không ghi đè run/checkpoint tồn tại.
- Resume phải đúng run name, code, environment, parameters, dataset và GIT_COMMIT.
- MI được giữ khi mi_state và artifact hashes đầy đủ; MI đang dở chưa hoàn tất có thể phải tính lại từ đầu.
- RFE resume từ elimination đã persist; k-search resume từ k đã persist. Fit đang dở không thể resume bên trong cây/SVC.
- Run đã publish chỉ verify, không đọc Test lại.
- Checkpoint sealed COMPLETE có thể publish lại mà không fit/predict.
- Partial file hoặc final sealing dở không an toàn trả BLOCKED/UNSUPPORTED; không tự xóa để ép PASS.
- OS run lock chống hai writer. Atomicity của Google Drive FUSE không được chứng minh tuyệt đối.
- Các run trong COMPACT_MI_RFE_FRESH_RUNS và stage4_mi/runs cũ không bị di chuyển, ghi đè hay relabel.
- Helper V2 cũ được giữ, nhưng snapshot code cũ có thể khác sau khi thêm file; muốn resume experiment cũ phải dùng đúng source snapshot cũ, không tự bỏ identity check.

## Giới hạn và kiểm thử

Chỉ sửa code/docs và đọc source. **Codex chưa chạy unit tests, Python, notebook, training hoặc benchmark.** Cell 5 cung cấp bằng chứng khi người dùng chạy; fixture PASS không chứng minh toàn dataset hoặc khác version thư viện tương đương CTU-13. Scientific status luôn PROVISIONAL_RESEARCH, không official reproduction; labels/capture/session independence vẫn cần đối chiếu riêng.

Đọc `HUONG_DAN_CHAY_CICFLOW_COLAB_V2.md` để upload và chạy.
