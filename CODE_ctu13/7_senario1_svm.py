"""
evaluate_svm_single_run.py
==========================
Huấn luyện SVM (RBF kernel) trên FULL FEATURES – 1 lần cho mỗi config.
Lưu toàn bộ chỉ số: Accuracy, Precision, Recall, F1, Confusion Matrix,
Classification Report và THỜI GIAN HUẤN LUYỆN.
Tổng hợp CSV có cột "Model".
"""

from pathlib import Path
import pandas as pd
import numpy as np
import time
import warnings
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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
OUTPUT_ROOT = Path("/home/bkcs/Desktop/COMPACT-MI_RFE/results/CT13")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    "FW_256_FS_8",
    "FW_256_FS_16",
    "FW_512_FS_8",
    "FW_512_FS_16",
]

LABEL_COL = "Label"
MODEL_NAME = "SVM"

# Các cột cần loại bỏ
DROP_COLUMNS = [
    "Window_ID", "Split", "Class", "Family", "Category", "Scenario",
    "Source_File", "SourceFile", "Window_Index", "Start_Flow", "End_Flow",
    "Num_Flows", "Label", "Window_Length",
    *[f"{p}_{s}" for p in ("Src Port", "Dst Port", "Protocol")
      for s in ("mean", "std", "median", "skew", "kurtosis")]
]

# Tham số SVM (RBF kernel)
SVM_PARAMS = {
    'C': 10.0,
    'gamma': 'scale',
    'kernel': 'rbf',
    'probability': True,
    'cache_size': 10000,   # MB
    'shrinking': True,
    'max_iter': 20000,
    'random_state': 42    # cố định seed
}

# =========================================================
# LOAD DATA & PREPARE FEATURES
# =========================================================
def load_split(config_name, split_name):
    csv_path = DATASET_ROOT / config_name / f"{split_name}.csv"
    return pd.read_csv(csv_path)

def prepare_xy(df):
    y = df[LABEL_COL].values
    feature_cols = [c for c in df.columns if c not in DROP_COLUMNS]
    X = df[feature_cols].select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X.values.astype(np.float32), y, list(X.columns)

# =========================================================
# MAIN
# =========================================================
def main():
    summary_rows = []

    for config_name in CONFIGS:
        print(f"\n{'='*60}")
        print(f"Config: {config_name}")
        print(f"{'='*60}")

        # Load splits
        train_df = load_split(config_name, "train")
        val_df   = load_split(config_name, "val")
        test_df  = load_split(config_name, "test")
        train_val_df = pd.concat([train_df, val_df], ignore_index=True)

        X_train_val, y_train_val, _ = prepare_xy(train_val_df)
        X_test, y_test, _ = prepare_xy(test_df)

        # Scale
        scaler = StandardScaler()
        X_train_val_scaled = scaler.fit_transform(X_train_val)
        X_test_scaled = scaler.transform(X_test)

        # Lưu scaler
        config_out = OUTPUT_ROOT / config_name
        config_out.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, config_out / "scaler_svm.pkl")

        # Huấn luyện và đo thời gian
        svm = SVC(**SVM_PARAMS)
        print(f"Training SVM with {X_train_val_scaled.shape[1]} features...")
        start = time.time()
        svm.fit(X_train_val_scaled, y_train_val)
        elapsed = time.time() - start

        y_pred = svm.predict(X_test_scaled)

        # Tính toán các chỉ số
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Lưu file chi tiết
        metrics_file = config_out / "svm_metrics.txt"
        with open(metrics_file, "w") as f:
            f.write(f"Model: {MODEL_NAME} | Config: {config_name}\n")
            f.write(f"Training time: {elapsed:.2f} seconds\n\n")
            f.write(f"Accuracy : {acc:.6f}\n")
            f.write(f"Precision: {prec:.6f}\n")
            f.write(f"Recall   : {rec:.6f}\n")
            f.write(f"F1-score : {f1:.6f}\n\n")
            f.write("Confusion Matrix:\n")
            f.write(str(confusion_matrix(y_test, y_pred)))
            f.write("\n\nClassification Report:\n")
            f.write(classification_report(y_test, y_pred, digits=6))

        print(f"  Time={elapsed:.1f}s, Acc={acc:.6f}, Prec={prec:.6f}, "
              f"Rec={rec:.6f}, F1={f1:.6f}")

        # Thêm vào bảng tổng hợp
        summary_rows.append({
            "Model": MODEL_NAME,
            "Config": config_name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "Training_Time_sec": elapsed,
        })

    # Lưu CSV tổng hợp
    df_summary = pd.DataFrame(summary_rows)
    output_csv = OUTPUT_ROOT / "table_v_svm_fullmetrics.csv"
    df_summary.to_csv(output_csv, index=False, float_format="%.6f")
    print(f"\n✅ Đã lưu bảng tổng hợp vào: {output_csv}")
    print(df_summary)

if __name__ == "__main__":
    main()