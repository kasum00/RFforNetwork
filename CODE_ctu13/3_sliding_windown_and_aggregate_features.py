from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from tqdm import tqdm

# =========================================================
# CONFIG
# =========================================================

# Dataset sau khi split
DATASET_ROOT = Path(
    "/home/bkcs/Desktop/COMPACT-MI_RFE/CTU13_SPLIT"
)

# Output
OUTPUT_ROOT = Path(
    "/home/bkcs/Desktop/COMPACT-MI_RFE/window_CTU13"
)

# Sliding Window Config
WINDOW_SIZES = [256, 512]
STRIDES = [8, 16]

# Label column
LABEL_COL = "Label"

# =========================================================
# FEATURE AGGREGATION
# =========================================================

def compute_window_features(window_df, feature_cols):
    """
    Statistical aggregation:
    - mean
    - std
    - median
    - skewness
    - kurtosis
    """

    feature_vector = {}

    for col in feature_cols:

        values = window_df[col].values.astype(np.float64)

        # xử lý NaN / Inf
        values = np.nan_to_num(
            values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        feature_vector[f"{col}_mean"] = np.mean(values)
        feature_vector[f"{col}_std"] = np.std(values)
        feature_vector[f"{col}_median"] = np.median(values)
        feature_vector[f"{col}_skew"] = skew(values)
        feature_vector[f"{col}_kurtosis"] = kurtosis(values)

    return feature_vector


# =========================================================
# WINDOW LABELING
# =========================================================

def get_window_label(window_df):
    """
    Existential Rule

    Window = malicious nếu tồn tại
    ít nhất 1 malicious flow
    """

    labels = window_df[LABEL_COL].values

    return 1 if np.any(labels == 1) else 0


# =========================================================
# PROCESS SINGLE CSV
# =========================================================

def process_csv(csv_path, fw, fs, split_name):

    try:
        df = pd.read_csv(csv_path)

    except Exception as e:
        print(f"ERROR reading {csv_path}: {e}")
        return []

    # =====================================================
    # CHỌN NUMERIC FEATURES
    # =====================================================

    numeric_cols = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    # bỏ label
    feature_cols = [
        c for c in numeric_cols
        if c != LABEL_COL
    ]

    if len(feature_cols) == 0:
        return []

    windows = []

    # =====================================================
    # METADATA TỪ PATH
    # =====================================================

    relative_parts = csv_path.relative_to(
        DATASET_ROOT
    ).parts

    # Ví dụ:
    #
    # train/Adware/FakeApp/file.csv
    #
    # relative_parts[0] = train
    # relative_parts[1] = class
    # relative_parts[2] = family

    traffic_class = (
        relative_parts[1]
        if len(relative_parts) > 1
        else "Unknown"
    )

    family = (
        relative_parts[2]
        if len(relative_parts) > 2
        else "Unknown"
    )

    source_file = str(csv_path)

    # tên file không có .csv
    file_stem = csv_path.stem

    # =====================================================
    # SLIDING WINDOW
    # =====================================================

    window_index = 0

    for start in range(0, len(df) - fw + 1, fs):

        end = start + fw

        window_df = df.iloc[start:end]

        # -------------------------------------------------
        # statistical aggregation
        # -------------------------------------------------

        features = compute_window_features(
            window_df,
            feature_cols
        )

        # -------------------------------------------------
        # window label
        # -------------------------------------------------

        label = get_window_label(window_df)

        # -------------------------------------------------
        # Window ID
        # reset theo từng file
        # -------------------------------------------------

        window_id = (
            f"{file_stem}_WIN_{window_index:06d}"
        )

        # -------------------------------------------------
        # Metadata
        # -------------------------------------------------

        features["Window_ID"] = window_id

        features["Split"] = split_name

        features["Class"] = traffic_class

        features["Family"] = family

        features["Source_File"] = source_file

        features["Window_Index"] = window_index

        features["Start_Flow"] = start

        features["End_Flow"] = end - 1

        features["Num_Flows"] = fw

        features["Label"] = label

        windows.append(features)

        window_index += 1

    return windows


# =========================================================
# PROCESS DATASET
# =========================================================

for split in ["train", "val", "test"]:

    split_root = DATASET_ROOT / split

    csv_files = list(
        split_root.rglob("*.csv")
    )

    print(f"\n{'=' * 60}")
    print(f"{split.upper()} : {len(csv_files)} csv files")
    print(f"{'=' * 60}")

    for fw in WINDOW_SIZES:

        for fs in STRIDES:

            print(f"\nProcessing FW={fw}, FS={fs}")

            all_windows = []

            for csv_path in tqdm(csv_files):

                windows = process_csv(
                    csv_path=csv_path,
                    fw=fw,
                    fs=fs,
                    split_name=split
                )

                all_windows.extend(windows)

            # =================================================
            # SAVE
            # =================================================

            output_dir = (
                OUTPUT_ROOT /
                f"FW_{fw}_FS_{fs}"
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            output_csv = (
                output_dir /
                f"{split}.csv"
            )

            result_df = pd.DataFrame(all_windows)

            # -------------------------------------------------
            # đưa metadata lên đầu
            # -------------------------------------------------

            metadata_cols = [
                "Window_ID",
                "Split",
                "Class",
                "Family",
                "Source_File",
                "Window_Index",
                "Start_Flow",
                "End_Flow",
                "Num_Flows",
                "Label"
            ]

            feature_cols = [
                c for c in result_df.columns
                if c not in metadata_cols
            ]

            result_df = result_df[
                metadata_cols + feature_cols
            ]

            # -------------------------------------------------
            # save csv
            # -------------------------------------------------

            result_df.to_csv(
                output_csv,
                index=False
            )

            # =================================================
            # LOG
            # =================================================

            print(f"\nSaved: {output_csv}")

            print(f"Shape: {result_df.shape}")

            if "Label" in result_df.columns:

                print("\nLabel Distribution:")

                print(
                    result_df["Label"]
                    .value_counts()
                )

print("\nDONE!")