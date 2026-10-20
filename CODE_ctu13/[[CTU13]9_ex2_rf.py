"""
scenario2_mi_rfe_RF.py
==========================
Thực hiện đầy đủ pipeline MI + RFE + validation search + final test
cho mô hình Random Forest, chỉ chạy 1 seed (42).

Cấu trúc tương tự script XGBoost bạn cung cấp,
chỉ thay phần model và các tham số liên quan.
"""

from pathlib import Path
import time
import warnings
import joblib

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
DATASET_ROOT = Path("/home/bkcs/Desktop/COMPACT-MI_RFE/window_CTU13")
OUTPUT_ROOT = Path("/home/bkcs/Desktop/COMPACT-MI_RFE/results/CT13/EX2/RF")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    "FW_512_FS_8",
    # "FW_512_FS_16"
]

LABEL_COL = "Label"
TOP_K_MI = 100
RANDOM_STATE = 42

# ============================================================
# DROP COLUMNS (giống script XGB)
# ============================================================
# Các cột cần loại bỏ
DROP_COLUMNS = [
    "Window_ID", "Split", "Class", "Family", "Category", "Scenario",
    "Source_File", "SourceFile", "Window_Index", "Start_Flow", "End_Flow",
    "Num_Flows", "Label", "Window_Length",
    *[f"{p}_{s}" for p in ("Src Port", "Dst Port", "Protocol")
      for s in ("mean", "std", "median", "skew", "kurtosis")]
]

# ============================================================
# RANDOM FOREST PARAMETERS
# ============================================================
# Tham số chính cho model cuối cùng (và dùng trong validation search)
RF_PARAMS = dict(
    n_estimators=300,
    max_depth=20,
    min_samples_split=3,
    max_features='sqrt',
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# Tham số cho RFE estimator (nhẹ hơn để tăng tốc)
RFE_ESTIMATOR_PARAMS = dict(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# ============================================================
# LOAD SPLIT
# ============================================================
def load_split(config_name, split_name):
    csv_path = DATASET_ROOT / config_name / f"{split_name}.csv"
    return pd.read_csv(csv_path)

# ============================================================
# PREPARE XY
# ============================================================
def prepare_xy(df):
    y = df[LABEL_COL].values
    feature_cols = [c for c in df.columns if c not in DROP_COLUMNS]
    X = df[feature_cols].select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X.astype(np.float32), y, list(X.columns)

# ============================================================
# MAIN LOOP
# ============================================================
summary_rows = []

for config_name in CONFIGS:
    print("\n" + "=" * 80)
    print(f"CONFIG: {config_name}")
    print("=" * 80)

    config_out = OUTPUT_ROOT / config_name
    config_out.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------
    train_df = load_split(config_name, "train")
    val_df   = load_split(config_name, "val")
    test_df  = load_split(config_name, "test")

    X_train_df, y_train, feature_names = prepare_xy(train_df)
    X_val_df, y_val, _ = prepare_xy(val_df)
    X_test_df, y_test, _ = prepare_xy(test_df)

    print(f"\nTrain shape : {X_train_df.shape}")
    print(f"Val shape   : {X_val_df.shape}")
    print(f"Test shape  : {X_test_df.shape}")

    # --------------------------------------------------------
    # SCALE (fit trên train, transform val/test)
    # --------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_val_scaled   = scaler.transform(X_val_df)
    X_test_scaled  = scaler.transform(X_test_df)

    # --------------------------------------------------------
    # MI FEATURE SELECTION
    # --------------------------------------------------------
    print("\nComputing Mutual Information ...")
    mi_scores = mutual_info_classif(X_train_scaled, y_train,
                                    random_state=RANDOM_STATE)
    mi_series = pd.Series(mi_scores, index=feature_names).sort_values(ascending=False)
    top100_features = mi_series.head(TOP_K_MI).index.tolist()

    mi_series.to_csv(config_out / "mi_feature_scores.csv")
    print(f"Top MI features: {len(top100_features)}")

    # --------------------------------------------------------
    # BUILD TOP100 MATRICES (dữ liệu gốc)
    # --------------------------------------------------------
    X_train_top100_raw = X_train_df[top100_features]
    X_val_top100_raw   = X_val_df[top100_features]
    X_test_top100_raw  = X_test_df[top100_features]

    # Scale lại trên không gian 100 đặc trưng
    scaler_top100 = StandardScaler()
    X_train_top100 = scaler_top100.fit_transform(X_train_top100_raw)
    X_val_top100   = scaler_top100.transform(X_val_top100_raw)
    X_test_top100  = scaler_top100.transform(X_test_top100_raw)

    # --------------------------------------------------------
    # RFE RANKING
    # --------------------------------------------------------
    print("\nRunning RFE ranking ...")
    rfe_estimator = RandomForestClassifier(**RFE_ESTIMATOR_PARAMS)

    start_rfe = time.time()
    rfe_selector = RFE(estimator=rfe_estimator,
                       n_features_to_select=1,
                       step=1)
    rfe_selector.fit(X_train_top100, y_train)
    rfe_time = time.time() - start_rfe
    print(f"RFE completed in {rfe_time:.2f} sec")

    rfe_ranks = pd.Series(rfe_selector.ranking_, index=top100_features)
    rfe_ranks_sorted = rfe_ranks.sort_values(ascending=True)
    rfe_ranks_sorted.to_csv(config_out / "rfe_feature_ranking.csv")

    ranked_features = rfe_ranks_sorted.index.tolist()
    print("\nTop 20 RFE Features:")
    print(ranked_features[:20])

    # --------------------------------------------------------
    # VALIDATION SEARCH k = 1..100
    # --------------------------------------------------------
    print("\nSearching best k using validation set ...")
    k_results = []
    best_k = None
    best_acc = -1.0

    for k in range(1, 101):
        print(f"  Evaluating Top-{k} features")
        selected_features = ranked_features[:k]
        feature_indices = [top100_features.index(f) for f in selected_features]

        X_train_k = X_train_top100[:, feature_indices]
        X_val_k   = X_val_top100[:, feature_indices]

        # Model với tham số chuẩn, seed cố định
        model = RandomForestClassifier(**RF_PARAMS)
        model.fit(X_train_k, y_train)

        y_val_pred = model.predict(X_val_k)
        acc  = accuracy_score(y_val, y_val_pred)
        prec = precision_score(y_val, y_val_pred)
        rec  = recall_score(y_val, y_val_pred)
        f1   = f1_score(y_val, y_val_pred)

        k_results.append({
            "k": k,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1
        })

        print(f"k={k:3d} | Acc={acc:.6f} | Prec={prec:.6f} | Rec={rec:.6f} | F1={f1:.6f}")

        if acc > best_acc:
            best_acc = acc
            best_k = k

    k_results_df = pd.DataFrame(k_results)
    k_results_df.to_csv(config_out / "k_rfe_results.csv", index=False)

    print("\nBest k found:")
    print(f"  k = {best_k}")
    print(f"  val_acc = {best_acc:.6f}")

    final_features = ranked_features[:best_k]
    pd.Series(final_features).to_csv(config_out / "best_features.csv", index=False)

    # --------------------------------------------------------
    # RETRAIN trên TRAIN + VAL
    # --------------------------------------------------------
    print("\nRetraining using TRAIN + VAL ...")
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    X_train_val_df, y_train_val, _ = prepare_xy(train_val_df)
    X_train_val_df = X_train_val_df[final_features]
    X_test_final_df = X_test_df[final_features]

    final_scaler = StandardScaler()
    X_train_val_final = final_scaler.fit_transform(X_train_val_df)
    X_test_final = final_scaler.transform(X_test_final_df)

    joblib.dump(final_scaler, config_out / "final_scaler.pkl")

    final_model = RandomForestClassifier(**RF_PARAMS)
    start_train = time.time()
    final_model.fit(X_train_val_final, y_train_val)
    train_time = time.time() - start_train

    # --------------------------------------------------------
    # TEST EVALUATION
    # --------------------------------------------------------
    y_test_pred = final_model.predict(X_test_final)
    acc  = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred)
    rec  = recall_score(y_test, y_test_pred)
    f1   = f1_score(y_test, y_test_pred)
    cm   = confusion_matrix(y_test, y_test_pred)
    report = classification_report(y_test, y_test_pred, digits=6)

    # Lưu metrics
    metrics_file = config_out / "final_test_metrics.txt"
    with open(metrics_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"CONFIG: {config_name}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Best k : {best_k}\n")
        f.write(f"Train time : {train_time:.2f} sec\n\n")
        f.write(f"Accuracy : {acc:.6f}\n")
        f.write(f"Precision: {prec:.6f}\n")
        f.write(f"Recall   : {rec:.6f}\n")
        f.write(f"F1-score : {f1:.6f}\n\n")
        f.write("Selected Features\n")
        f.write("-" * 40 + "\n")
        for feat in final_features:
            f.write(f"{feat}\n")
        f.write("\nConfusion Matrix\n")
        f.write(str(cm))
        f.write("\n\nClassification Report\n")
        f.write(report)

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

# ============================================================
# FINAL SUMMARY
# ============================================================
summary_df = pd.DataFrame(summary_rows)
summary_csv = OUTPUT_ROOT / "scenario2_summary.csv"
summary_df.to_csv(summary_csv, index=False)

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(summary_df)
print("\nDone.")