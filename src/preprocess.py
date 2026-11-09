"""Text cleaning applied to every review before feature extraction.

Steps, in order:
  1. remove HTML tags (some reviews contain embedded Amazon product links and video blocks)
  2. remove URLs (anything starting with "http" or "www")
  3. trim surrounding whitespace and lowercase

These exact rules reproduce the `cleaned_text` column of processed-dataset.csv for all
36,389 reviews.
"""

import re

HTML_TAG = re.compile(r"<[^>]+>")
URL = re.compile(r"http\S+|www\S+")


def preprocess_text(text: str) -> str:
    text = HTML_TAG.sub("", text)
    text = URL.sub("", text)
    return text.strip().lower()
