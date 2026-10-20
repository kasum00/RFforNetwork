"""
scenario2_mi_rfe_svm.py (ĐÃ SỬA LỖI INDEX)
Pipeline MI-RFE với SVM (RBF kernel).
- RFE dùng RandomForest (nhanh) để xếp hạng 100 đặc trưng MI.
- Tìm k tối ưu bằng SVM trên tập validation.
- Huấn luyện lại trên train+val, đánh giá test.
"""

from pathlib import Path
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================
DATASET_ROOT = Path("/home/bkcs/Desktop/COMPACT-MI_RFE/window_CTU13")
OUTPUT_ROOT = Path("/home/bkcs/Desktop/COMPACT-MI_RFE/results/CT13/EX2/")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    # "FW_256_FS_8",
    # "FW_256_FS_16",
    "FW_512_FS_8",
    # "FW_512_FS_16",
]

LABEL_COL = "Label"
TOP_K_MI = 100
RANDOM_STATE = 42

# Các cột cần loại bỏ
DROP_COLUMNS = [
    "Window_ID", "Split", "Class", "Family", "Category", "Scenario",
    "Source_File", "SourceFile", "Window_Index", "Start_Flow", "End_Flow",
    "Num_Flows", "Label", "Window_Length",
    *[f"{p}_{s}" for p in ("Src Port", "Dst Port", "Protocol")
      for s in ("mean", "std", "median", "skew", "kurtosis")]
]

# Tham số SVM
SVM_PARAMS = {
    'C': 10.0,
    'gamma': 'scale',
    'kernel': 'rbf',
    'probability': False,   # Tắt để tăng tốc
    'cache_size': 10000,    # MB (có thể tăng nếu còn RAM)
    'shrinking': True,
    'max_iter': 20000,
    'class_weight': 'balanced',
}

# =========================================================
# LOAD DATA & PREPARE FEATURES (trả về mảng NumPy)
# =========================================================
def load_split(config_name, split_name):
    csv_path = DATASET_ROOT / config_name / f"{split_name}.csv"
    return pd.read_csv(csv_path)

def prepare_xy(df):
    y = df[LABEL_COL].values
    feature_cols = [c for c in df.columns if c not in DROP_COLUMNS]
    X = df[feature_cols].select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    # Trả về mảng NumPy và danh sách tên cột
    return X.values.astype(np.float32), y, feature_cols

# =========================================================
# MAIN
# =========================================================
def main():
    summary_rows = []

    for config_name in CONFIGS:
        print(f"\n{'='*80}")
        print(f"CONFIG: {config_name}")
        print(f"{'='*80}")

        config_out = OUTPUT_ROOT / config_name
        config_out.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------
        # 1. Load dữ liệu
        # -------------------------------------------------
        train_df = load_split(config_name, "train")
        val_df   = load_split(config_name, "val")
        test_df  = load_split(config_name, "test")

        X_train_full, y_train, feature_names = prepare_xy(train_df)
        X_val_full, y_val, _ = prepare_xy(val_df)
        X_test_full, y_test, _ = prepare_xy(test_df)

        print(f"\nTrain shape : {X_train_full.shape}")
        print(f"Val shape   : {X_val_full.shape}")
        print(f"Test shape  : {X_test_full.shape}")

        # -------------------------------------------------
        # 2. Scale toàn bộ đặc trưng (trên tập train, áp dụng val/test)
        # -------------------------------------------------
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_full)
        X_val_scaled   = scaler.transform(X_val_full)
        X_test_scaled  = scaler.transform(X_test_full)

        # -------------------------------------------------
        # 3. Mutual Information – chọn Top 100
        # -------------------------------------------------
        print("\nComputing Mutual Information ...")
        mi_scores = mutual_info_classif(X_train_scaled, y_train, random_state=RANDOM_STATE)
        mi_series = pd.Series(mi_scores, index=feature_names).sort_values(ascending=False)
        top100_features = mi_series.head(TOP_K_MI).index.tolist()
        mi_series.to_csv(config_out / "mi_feature_scores.csv")
        print(f"Top MI features: {len(top100_features)}")

        # Lấy chỉ số của các đặc trưng top100 trong mảng gốc
        top100_indices = [feature_names.index(f) for f in top100_features]

        # Tạo mảng con chỉ chứa top100
        X_train_top100 = X_train_full[:, top100_indices]
        X_val_top100   = X_val_full[:, top100_indices]
        X_test_top100  = X_test_full[:, top100_indices]

        # Scale lại trên top100
        scaler_top100 = StandardScaler()
        X_train_top100 = scaler_top100.fit_transform(X_train_top100)
        X_val_top100   = scaler_top100.transform(X_val_top100)
        X_test_top100  = scaler_top100.transform(X_test_top100)

        # -------------------------------------------------
        # 4. RFE Ranking (RandomForest)
        # -------------------------------------------------
        print("\nRunning RFE ranking ...")
        rfe_estimator = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
        rfe_selector = RFE(estimator=rfe_estimator, n_features_to_select=1, step=1)
        start_rfe = time.time()
        rfe_selector.fit(X_train_top100, y_train)
        rfe_time = time.time() - start_rfe
        print(f"RFE completed in {rfe_time:.2f} sec")

        rfe_ranks = pd.Series(rfe_selector.ranking_, index=top100_features).sort_values()
        rfe_ranks.to_csv(config_out / "rfe_feature_ranking.csv")
        ranked_features = rfe_ranks.index.tolist()
        print(f"Top 20 RFE features: {ranked_features[:20]}")

        # -------------------------------------------------
        # 5. Tìm k tối ưu trên validation (SVM)
        # -------------------------------------------------
        print("\nSearching best k using SVM on validation set ...")
        k_results = []
        best_k = None
        best_acc = -1

        detail_log_path = config_out / "val_metrics_details.txt"
        with open(detail_log_path, "w") as f_log:
            f_log.write("Detailed validation metrics for each k (SVM)\n")
            f_log.write("=" * 80 + "\n\n")

        for k in range(1, 101):
            print(f"  Evaluating Top-{k} features", end="")
            selected_features = ranked_features[:k]
            # Lấy chỉ số trong top100
            feature_indices = [top100_features.index(f) for f in selected_features]

            X_train_k = X_train_top100[:, feature_indices]
            X_val_k   = X_val_top100[:, feature_indices]

            svm_model = SVC(**SVM_PARAMS, random_state=RANDOM_STATE)
            svm_model.fit(X_train_k, y_train)
            y_val_pred = svm_model.predict(X_val_k)

            acc  = accuracy_score(y_val, y_val_pred)
            prec = precision_score(y_val, y_val_pred)
            rec  = recall_score(y_val, y_val_pred)
            f1   = f1_score(y_val, y_val_pred)

            k_results.append({"k": k, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1})

            with open(detail_log_path, "a") as f_log:
                f_log.write(f"k = {k}\n")
                f_log.write(f"Accuracy : {acc:.6f}\nPrecision: {prec:.6f}\nRecall   : {rec:.6f}\nF1-score : {f1:.6f}\n")
                f_log.write("Confusion Matrix:\n")
                f_log.write(str(confusion_matrix(y_val, y_val_pred)))
                f_log.write("\n\nClassification Report:\n")
                f_log.write(classification_report(y_val, y_val_pred, digits=6))
                f_log.write("\n" + "-" * 60 + "\n\n")

            print(f" | Acc={acc:.6f} F1={f1:.6f}")

            if acc > best_acc:
                best_acc = acc
                best_k = k

        k_df = pd.DataFrame(k_results)
        k_df.to_csv(config_out / "k_search_results.csv", index=False)
        print(f"\nBest k = {best_k} (val acc = {best_acc:.6f})")

        # -------------------------------------------------
        # 6. Huấn luyện cuối cùng (train+val) và test
        # -------------------------------------------------
        final_features = ranked_features[:best_k]
        pd.Series(final_features).to_csv(config_out / "best_features.csv", index=False)

        # Gộp train + val
        train_val_df = pd.concat([train_df, val_df], ignore_index=True)
        X_train_val_full, y_train_val, feature_names2 = prepare_xy(train_val_df)

        # Lấy vị trí của final_features trong full dataset
        final_indices = [feature_names2.index(f) for f in final_features]
        X_train_val_final = X_train_val_full[:, final_indices]
        X_test_final      = X_test_full[:, final_indices]

        # Scale
        final_scaler = StandardScaler()
        X_train_val_final = final_scaler.fit_transform(X_train_val_final)
        X_test_final      = final_scaler.transform(X_test_final)
        joblib.dump(final_scaler, config_out / "final_scaler.pkl")

        # Huấn luyện SVM cuối cùng
        print("\nTraining final SVM on train+val ...")
        final_svm = SVC(**SVM_PARAMS, random_state=RANDOM_STATE)
        start_train = time.time()
        final_svm.fit(X_train_val_final, y_train_val)
        train_time = time.time() - start_train

        y_test_pred = final_svm.predict(X_test_final)

        acc  = accuracy_score(y_test, y_test_pred)
        prec = precision_score(y_test, y_test_pred)
        rec  = recall_score(y_test, y_test_pred)
        f1   = f1_score(y_test, y_test_pred)

        with open(config_out / "final_test_metrics.txt", "w") as f:
            f.write(f"CONFIG: {config_name}\n")
            f.write(f"Model: SVM (RBF)\n")
            f.write(f"Best k : {best_k}\n")
            f.write(f"Train time : {train_time:.2f} sec\n")
            f.write(f"RFE time : {rfe_time:.2f} sec\n\n")
            f.write(f"Accuracy : {acc:.6f}\nPrecision: {prec:.6f}\nRecall   : {rec:.6f}\nF1-score : {f1:.6f}\n\n")
            f.write("Selected Features:\n")
            for feat in final_features:
                f.write(f"  {feat}\n")
            f.write("\nConfusion Matrix:\n")
            f.write(str(confusion_matrix(y_test, y_test_pred)))
            f.write("\n\nClassification Report:\n")
            f.write(classification_report(y_test, y_test_pred, digits=6))

        print(f"Test results: Acc={acc:.6f}, Prec={prec:.6f}, Rec={rec:.6f}, F1={f1:.6f}")

        # Tổng hợp
        summary_rows.append({
            "Config": config_name,
            "Best_k": best_k,
            "Test_Accuracy": acc,
            "Test_Precision": prec,
            "Test_Recall": rec,
            "Test_F1": f1,
            "Train_Time_sec": train_time,
            "RFE_Time_sec": rfe_time,
        })

    # Lưu bảng tổng hợp
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = OUTPUT_ROOT / "scenario2_svm_summary.csv"
    summary_df.to_csv(summary_csv, index=False, float_format="%.6f")
    print("\n" + "=" * 80)
    print("FINAL SUMMARY (SVM)")
    print("=" * 80)
    print(summary_df)
    print(f"\nKết quả đã lưu vào: {summary_csv}")

if __name__ == "__main__":
    main()