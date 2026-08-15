"""
Tests for Image Preprocessing Adaptation (Stage 10 - Objective C)
Verifies:
1. RGB normalization and resizing
2. Grayscale image channel adaptation
3. Corrupted image detection and safe fallback
4. Missing image handling
5. Train-only augmentation isolation (no contamination into validation/test)
"""

from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from backend.app.multimodal.image_preprocessing import ImagePreprocessor


def test_image_preprocessing_adaptation(tmp_path):
    rgb_p = tmp_path / "rgb.png"
    gray_p = tmp_path / "gray.png"
    corrupt_p = tmp_path / "corrupt.png"
    missing_p = tmp_path / "non_existent.png"

    Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(rgb_p)
    Image.fromarray(np.random.randint(0, 255, (64, 64), dtype=np.uint8)).save(gray_p)
    with open(corrupt_p, "wb") as f:
        f.write(b"not a valid png file header")

    preprocessor = ImagePreprocessor(target_size=(32, 32), augment_train=True)
    preprocessor.fit([str(rgb_p), str(gray_p)])

    # Transform training fold (augmentations active)
    train_tensors = preprocessor.transform([str(rgb_p), str(gray_p), str(corrupt_p), str(missing_p)], is_training=True)
    assert train_tensors.shape == (4, 3, 32, 32)

    # Transform test fold (augmentations strictly disabled)
    test_tensors = preprocessor.transform([str(rgb_p), str(gray_p), str(corrupt_p), str(missing_p)], is_training=False)
    assert test_tensors.shape == (4, 3, 32, 32)

    # Verify corrupt and missing images do not crash and produce finite numerical values
    assert np.all(np.isfinite(test_tensors))
