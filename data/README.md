# Data

This project uses the **UNSW-NB15** network intrusion dataset. The raw CSVs are not tracked in this
repository (kept out via `.gitignore`) to keep clone sizes small and avoid redistributing a dataset
we don't own.

## Setup

1. Download the official train/test split of UNSW-NB15 (originally from UNSW Sydney's Cyber Range
   Lab; also commonly mirrored on Kaggle as "UNSW-NB15").
2. Place these two files directly in this `data/` folder:
   - `UNSW_NB15_training-set.csv` (175,341 flows)
   - `UNSW_NB15_testing-set.csv` (82,332 flows)

`src/train.py` reads these two files by default. Both are the standard pre-split files shipped with
UNSW-NB15 (49 raw features per flow, binary `label` column plus a multi-class `attack_cat` column).
