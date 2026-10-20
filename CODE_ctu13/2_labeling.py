from pathlib import Path
import pandas as pd
from tqdm import tqdm

# =========================================================
# CONFIG
# =========================================================

DATASET_ROOT = Path("/home/bkcs/Desktop/COMPACT-MI_RFE/CTU13_dataset")

# =========================================================
# LABELING RULE
# =========================================================
#
# Nếu file nằm trong folder:
#   Benign/
# => Label = 0
#
# Ngược lại:
# => Label = 1
#
# =========================================================

LABEL_COL = "Label"

# =========================================================
# GET ALL CSV FILES
# =========================================================

csv_files = list(DATASET_ROOT.rglob("*.csv"))

print(f"Total CSV files: {len(csv_files)}")

# =========================================================
# PROCESS
# =========================================================

for csv_path in tqdm(csv_files):

    try:
        df = pd.read_csv(csv_path)

    except Exception as e:
        print(f"\nERROR reading {csv_path}")
        print(e)
        continue

    # -----------------------------------------------------
    # Detect label from folder name
    # -----------------------------------------------------

    path_parts = [p.lower() for p in csv_path.parts]

    if "benign" in path_parts:
        label = 0
    else:
        label = 1

    # -----------------------------------------------------
    # Add Label column
    # -----------------------------------------------------

    df[LABEL_COL] = label

    # -----------------------------------------------------
    # Save back
    # -----------------------------------------------------

    df.to_csv(csv_path, index=False)

print("\nDone labeling dataset!")