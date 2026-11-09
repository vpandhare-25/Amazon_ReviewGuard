"""Unit tests for text cleaning and feature extraction. Run with: python -m pytest -q"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from feature_extraction import extract_features  # noqa: E402
from preprocess import preprocess_text  # noqa: E402


def test_lowercases_and_trims():
    assert preprocess_text("  I WANTED Different Flavors.  ") == "i wanted different flavors."


def test_removes_html_tags():
    raw = 'Great <a class="a-link-normal" href="/dp/B011J5MKPC">Pie Server</a> overall'
    assert preprocess_text(raw) == "great pie server overall"


def test_removes_urls():
    assert preprocess_text("See http://goo.gl/abc and www.example.com now") == "see  and  now"


def test_basic_features():
    df = pd.DataFrame({"cleaned_text": ["love it! great.", "ok"], "rating": [5.0, 3.0]})
    out = extract_features(df, include_pos=False)
    assert out["char_length"].tolist() == [15, 2]
    assert out["word_count"].tolist() == [3, 1]
    assert out["punctuation_ct"].tolist() == [2, 0]
    assert out["is_extreme_star"].tolist() == [True, False]


def test_cleaning_matches_processed_dataset():
    """The committed processed-dataset.csv must be exactly what preprocess_text produces."""
    root = Path(__file__).resolve().parent.parent
    df = pd.read_csv(root / "processed-dataset.csv", usecols=["text_", "cleaned_text"])
    mismatches = (df["text_"].apply(preprocess_text) != df["cleaned_text"]).sum()
    assert mismatches == 0
