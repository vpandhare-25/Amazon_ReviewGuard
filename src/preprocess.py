"""Text cleaning applied to every review before feature extraction.

Steps, in order:
  1. lowercase
  2. remove HTML tags (some reviews contain embedded Amazon product links and video blocks)
  3. remove links (anything starting with "http" or "www")
  4. trim surrounding whitespace

This is the original cleaning code from the first commit (it had been deleted by
mistake). It reproduces the `cleaned_text` column of processed-dataset.csv for all
36,389 reviews.
"""

import re


def preprocess_text(text: str) -> str:
    text = text.lower()                         # convert text to lowercase
    text = re.sub(r"<.*?>", "", text)           # remove HTML tags
    text = re.sub(r"http\S+|www\S+", "", text)  # remove links
    return text.strip()                         # strip remaining whitespace
