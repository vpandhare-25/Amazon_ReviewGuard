"""Turns cleaned review text into numeric features for the classifier.

Basic features (no NLP model needed):
  char_length      number of characters
  word_count       number of whitespace-separated words
  punctuation_ct   number of punctuation characters
  is_extreme_star  True for 1-star and 5-star ratings

Part-of-speech features (spaCy, optional): counts of VERB, NOUN and ADV tokens.
"""

import string
from collections import Counter

import pandas as pd

POS_WHITELIST = {"VERB", "NOUN", "ADV"}
_nlp = None  # spaCy pipeline, loaded on first use (slow to load, and not always needed)


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy

        # The parser and named-entity recognizer are not needed for part-of-speech tags.
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    return _nlp


def extract_features(df: pd.DataFrame, include_pos: bool = False) -> pd.DataFrame:
    df["char_length"] = df["cleaned_text"].apply(len)
    df["word_count"] = df["cleaned_text"].str.split().apply(len)
    df["punctuation_ct"] = df["cleaned_text"].apply(lambda text: sum(ch in string.punctuation for ch in text))
    df["is_extreme_star"] = df["rating"].isin([1.0, 5.0])
    if include_pos:
        df = add_pos_features(df)
    return df


def pos_counts(text: str) -> Counter:
    """Counts the whitelisted parts of speech in one review."""
    doc = _get_nlp()(text)
    return Counter(token.pos_ for token in doc if token.pos_ in POS_WHITELIST)


def add_pos_features(df: pd.DataFrame) -> pd.DataFrame:
    pos_df = pd.DataFrame(list(df["cleaned_text"].apply(pos_counts))).fillna(0)  # missing count = 0
    pos_df.index = df.index  # line rows up with the original dataframe
    return pd.concat([df, pos_df], axis=1)
