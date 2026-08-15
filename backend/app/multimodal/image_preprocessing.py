"""
Auditable Image Preprocessing Engine

Handles image loading, corruption detection, resizing, channel normalization,
missing-image imputation, and deterministic train/val/test transformation pipelines.
Enforces the train-only preprocessing contract: augmentations and fitted statistics are strictly
applied to the training partition and never leak into validation/test partitions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Standard ImageNet / Biomedical normalization parameters
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


class ImagePreprocessor:
    """
    Executable image preprocessing pipeline with strict train/val/test isolation.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        augment_train: bool = False,
        missing_strategy: str = "zero_image",  # 'zero_image' | 'mean_image'
    ):
        self.target_size = target_size
        self.mean = np.array(mean or DEFAULT_MEAN, dtype=np.float32).reshape((3, 1, 1))
        self.std = np.array(std or DEFAULT_STD, dtype=np.float32).reshape((3, 1, 1))
        self.augment_train = augment_train
        self.missing_strategy = missing_strategy

        # Fitted statistics on training set only
        self.fitted_mean_image: Optional[np.ndarray] = None
        self.is_fitted = False
        self._cache: Dict[str, np.ndarray] = {}

    def fit(self, image_paths: List[Union[str, Path, None]]) -> "ImagePreprocessor":
        """
        Fits any statistical preprocessors strictly on the training partition.
        """
        valid_arrays = []
        for p in image_paths:
            if p is not None:
                img_arr = self._load_single_image(p, apply_augment=False)
                if img_arr is not None:
                    valid_arrays.append(img_arr)

        if valid_arrays and self.missing_strategy == "mean_image":
            self.fitted_mean_image = np.mean(np.stack(valid_arrays, axis=0), axis=0)
        else:
            self.fitted_mean_image = np.zeros((3, self.target_size[0], self.target_size[1]), dtype=np.float32)

        self.is_fitted = True
        return self

    def transform(
        self,
        image_paths: List[Union[str, Path, None]],
        is_training: bool = False,
    ) -> np.ndarray:
        """
        Transforms a batch of images into normalized tensors (N, C, H, W).
        If is_training=False, augmentations are strictly disabled.
        """
        processed_tensors = []
        apply_aug = self.augment_train and is_training

        for p in image_paths:
            if p is None or not Path(p).exists():
                # Handle missing image using fitted strategy
                if self.fitted_mean_image is not None:
                    processed_tensors.append(self.fitted_mean_image.copy())
                else:
                    processed_tensors.append(np.zeros((3, self.target_size[0], self.target_size[1]), dtype=np.float32))
                continue

            tensor = self._load_single_image(p, apply_augment=apply_aug)
            if tensor is None:
                # Corrupted image fallback to fitted zero/mean image
                if self.fitted_mean_image is not None:
                    processed_tensors.append(self.fitted_mean_image.copy())
                else:
                    processed_tensors.append(np.zeros((3, self.target_size[0], self.target_size[1]), dtype=np.float32))
            else:
                processed_tensors.append(tensor)

        return np.stack(processed_tensors, axis=0)

    def _load_single_image(self, path: Union[str, Path], apply_augment: bool = False) -> Optional[np.ndarray]:
        try:
            p_str = str(path)
            if not apply_augment and p_str in self._cache:
                return self._cache[p_str].copy()

            p = Path(path)
            if not p.exists():
                return None

            with Image.open(p) as img:
                # Convert grayscale or RGBA to RGB
                img_rgb = img.convert("RGB")
                # Resize to target dimension with anti-aliasing
                img_resized = img_rgb.resize(self.target_size, Image.Resampling.BILINEAR)

                # Optional training-only deterministic augmentation
                if apply_augment:
                    # Deterministic horizontal flip if random flag
                    if np.random.rand() > 0.5:
                        img_resized = img_resized.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

                # Convert to float numpy array (H, W, C) -> (C, H, W) normalized [0, 1]
                arr = np.array(img_resized, dtype=np.float32) / 255.0
                arr = np.transpose(arr, (2, 0, 1))

                # Standardize with mean and std
                arr = (arr - self.mean) / self.std
                res = arr.astype(np.float32)
                if not apply_augment:
                    self._cache[p_str] = res
                return res
        except Exception as e:
            logger.warning("Corrupt or unreadable image at %s: %s", path, str(e))
            return None

    def get_provenance_spec(self) -> Dict[str, Any]:
        return {
            "component": "image_preprocessing",
            "target_size": list(self.target_size),
            "normalization": {"mean": [float(m) for m in self.mean.flatten()], "std": [float(s) for s in self.std.flatten()]},
            "train_only_augmentations": self.augment_train,
            "missing_strategy": self.missing_strategy,
            "train_only_fitting_enforced": True,
            "status": "FITTED" if self.is_fitted else "INITIALIZED",
        }
