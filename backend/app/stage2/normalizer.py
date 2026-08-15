from backend.app.stage2.models import DatasetCharacteristics

class Normalizer:
    """
    Cleans dataset characteristics (modality counts, sample sizes) 
    leaving missing data as null (no hallucination).
    """
    def __init__(self):
        pass
        
    def normalize_dataset_characteristics(self, raw_data: dict) -> DatasetCharacteristics:
        """
        Creates DatasetCharacteristics ensuring unknown values remain None.
        """
        return DatasetCharacteristics(
            sample_count=self._safe_int(raw_data.get("sample_count")),
            feature_count=self._safe_int(raw_data.get("feature_count")),
            class_count=self._safe_int(raw_data.get("class_count")),
            class_imbalance=self._safe_bool(raw_data.get("class_imbalance")),
            missingness=self._safe_float(raw_data.get("missingness")),
            modality_count=self._safe_int(raw_data.get("modality_count")),
            modality_types=raw_data.get("modality_types", []),
            text_available=self._safe_bool(raw_data.get("text_available")),
            image_available=self._safe_bool(raw_data.get("image_available")),
            tabular_available=self._safe_bool(raw_data.get("tabular_available")),
            temporal_available=self._safe_bool(raw_data.get("temporal_available"))
        )
        
    def _safe_int(self, value) -> int | None:
        try:
            return int(value) if value is not None else None
        except ValueError:
            return None
            
    def _safe_float(self, value) -> float | None:
        try:
            return float(value) if value is not None else None
        except ValueError:
            return None
            
    def _safe_bool(self, value) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        return str(value).lower() in ("true", "1", "yes")
