"""Step 1: clean the raw reviews and extract features.

Reads fake-reviews.csv, cleans the text (src/preprocess.py), adds the numeric and
part-of-speech features (src/feature_extraction.py), and writes processed-dataset.csv.

Usage:
    python build_dataset.py                 # writes processed-dataset.csv
    python build_dataset.py other-path.csv  # writes somewhere else
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from feature_extraction import extract_features  # noqa: E402
from preprocess import preprocess_text  # noqa: E402

output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "processed-dataset.csv"

df = pd.read_csv(ROOT / "fake-reviews.csv")
print(f"Loaded {len(df)} reviews")
df["cleaned_text"] = df["text_"].apply(preprocess_text)
df = extract_features(df, include_pos=True)  # part-of-speech tagging takes a few minutes
df.to_csv(output_path, index=False)
print(f"Processed dataset saved to {output_path}")
