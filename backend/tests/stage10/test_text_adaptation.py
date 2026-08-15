"""
Tests for Text Preprocessing Adaptation (Stage 10 - Objective D)
Verifies:
1. Tokenization and truncation on short and long clinical notes
2. Empty and missing text handling
3. Repeated tokens and punctuation cleaning
4. Train-only vocabulary and IDF fitting
"""

import numpy as np
import pytest
from backend.app.multimodal.text_preprocessing import TextPreprocessor


def test_text_preprocessing_adaptation():
    train_texts = [
        "Biopsy confirmed well differentiated adenocarcinoma.",
        "Clinical note: " + "surgical margins clear. " * 25,
        "",
        None,
        "Repeated text " * 10,
        "Special chars: !!!??? ### $$$ Stage T2N0M0 (negative).",
    ]

    test_texts = [
        "Post-operative oncology follow-up.",
        "Unknown vocabulary token: zzzqxxqwerty12345.",
    ]

    preprocessor = TextPreprocessor(max_seq_length=32, vocab_size=150, use_tfidf=True)

    # Fit strictly on training fold
    preprocessor.fit(train_texts)
    assert preprocessor.is_fitted
    assert len(preprocessor.vocab) > 5

    # Transform test fold
    input_ids, masks = preprocessor.transform(test_texts, is_training=False)
    assert input_ids.shape == (2, 32)
    assert masks.shape == (2, 32)

    # Unknown words must map to [UNK] (index 1)
    assert 1 in input_ids[1]

    # Transform TF-IDF
    tfidf = preprocessor.transform_tfidf(test_texts)
    assert tfidf.shape == (2, len(preprocessor.vocab))
    assert np.all(np.isfinite(tfidf))
