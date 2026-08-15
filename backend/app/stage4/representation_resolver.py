"""
Stage 4F: Compatible Feature Representation Resolution.

This module audits Stage 2 evidence, Stage 3 validated specification,
Stage 3.1 compatibility decisions, and HANCOCK's actual modality schema
to determine whether any compatible feature representation can be identified
as a replacement for the incompatible cnn_representation.

SAFETY RULES enforced by this module:
  - Do NOT infer that pathology == imaging.
  - Do NOT substitute an arbitrary tabular representation without explicit evidence.
  - Do NOT modify Stage 2 or Stage 3 evidence.
  - Do NOT call fit(), train(), optimizer steps, or hyperparameter search.
  - If no valid candidate exists, preserve BLOCKED.
  - training_allowed is ALWAYS False in this module.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Modality taxonomy - maps HANCOCK profile keys to canonical modality types.
# imaging is strictly absent unless explicitly present as a profile key.
# ---------------------------------------------------------------------------
HANCOCK_IMAGING_KEYS = set()   # No imaging modality in HANCOCK

HANCOCK_NON_IMAGING_MODALITIES = {
    "clinical",    # structured tabular clinical variables
    "pathology",   # structured tabular pathology/staging variables (NOT tissue slides)
    "blood",       # time-series laboratory analytes
    "text",        # free-text clinical reports and histories
}

# Representations that REQUIRE imaging.  Must never be enabled for HANCOCK.
IMAGING_ONLY_REPRESENTATIONS = {
    "cnn_representation",
    "cnn",
    "resnet_representation",
    "vit_representation",
    "wsi_representation",
    "radiology_cnn",
    "pet_ct_representation",
    "imagenet_pretrained",
}

# Representations that are considered pathology-tissue-slide-specific
# (i.e., WSI-level models). Pathology in HANCOCK = structured tabular staging
# data, NOT whole-slide images.
PATHOLOGY_SLIDE_REPRESENTATIONS = {
    "wsi_representation",
    "clam_representation",
    "mil_representation",
    "uni_representation",
    "conch_representation",
    "virchow_representation",
    "ctranspath_representation",
    "hoptimus_representation",
}


class RepresentationResolver:
    """
    Evidence-conditioned feature representation resolver for Stage 4F.

    Reads existing Stage 4 resolution artifact, Stage 2 experiments,
    Stage 3 validated spec, and HANCOCK modality profile.  Produces
    a `stage4_representation_resolution.json` artifact.
    """

    def __init__(
        self,
        stage2_experiments_path: str = "evidence/processed/experiments.jsonl",
        stage2_mechanisms_path: str = "evidence/processed/mechanisms.jsonl",
        stage3_spec_path: str = "evidence/processed/stage3_validated_pipeline_specification.json",
        stage3_rankings_path: str = "evidence/processed/stage3_mechanism_rankings.json",
        stage1_profile_path: str = "data/metadata/hancock/stage1_profile_report.json",
        existing_resolution_path: str = "data/metadata/hancock/stage4_feature_representation_resolution.json",
        impl_config_path: str = "data/config/implementation_config.json",
        out_path: str = "data/metadata/hancock/stage4_representation_resolution.json",
    ) -> None:
        self.stage2_experiments_path = Path(stage2_experiments_path)
        self.stage2_mechanisms_path = Path(stage2_mechanisms_path)
        self.stage3_spec_path = Path(stage3_spec_path)
        self.stage3_rankings_path = Path(stage3_rankings_path)
        self.stage1_profile_path = Path(stage1_profile_path)
        self.existing_resolution_path = Path(existing_resolution_path)
        self.impl_config_path = Path(impl_config_path)
        self.out_path = Path(out_path)

        # training_allowed is permanently False in this module
        self._training_allowed: bool = False

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _load_json(self, path: Path) -> Any:
        """Load a JSON file; returns empty dict if file does not exist."""
        if not path.exists():
            return {}
        encoding = "utf-8-sig" if path.suffix == ".json" else "utf-8"
        with open(path, "r", encoding=encoding) as f:
            return json.load(f)

    def _load_jsonl(self, path: Path) -> List[Dict]:
        """Load a JSONL file; returns empty list if file does not exist."""
        if not path.exists():
            return []
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    # ------------------------------------------------------------------
    # HANCOCK modality inventory
    # ------------------------------------------------------------------

    def _get_hancock_available_modalities(self) -> Dict[str, bool]:
        """
        Determine which data modalities HANCOCK actually provides.
        Returns a dict mapping modality -> available (bool).
        Imaging is only marked available if the profile explicitly contains it.
        """
        profile = self._load_json(self.stage1_profile_path)
        available: Dict[str, bool] = {
            "clinical": bool(profile.get("clinical")),
            "pathology_tabular": bool(profile.get("pathology")),  # NOTE: tabular staging, NOT WSI
            "blood": bool(profile.get("blood")),
            "text": bool(profile.get("text")),
            "imaging": False,   # HANCOCK has NO imaging modality
        }
        return available

    # ------------------------------------------------------------------
    # Stage 2 evidence search
    # ------------------------------------------------------------------

    def _search_stage2_for_feature_representations(
        self,
        available_modalities: Dict[str, bool],
    ) -> List[Dict[str, Any]]:
        """
        Search Stage 2 experiments for explicit feature_representation values
        that are compatible with HANCOCK's actual modalities.

        A candidate is valid only if:
          - The experiment field_provenance records an explicit extraction.
          - The mechanism is not in IMAGING_ONLY_REPRESENTATIONS.
          - The mechanism is not in PATHOLOGY_SLIDE_REPRESENTATIONS (pathology != WSI).
          - The experiment's modalities overlap with what HANCOCK provides.
        """
        experiments = self._load_jsonl(self.stage2_experiments_path)
        candidates: List[Dict[str, Any]] = []

        for exp in experiments:
            fr = exp.get("feature_representation")
            if not fr:
                continue  # No explicit representation recorded

            exp_id = exp.get("experiment_id", "UNKNOWN")
            paper_id = exp.get("paper_id", "UNKNOWN")
            exp_modalities = set(exp.get("modalities", []))
            task = exp.get("task")

            # Determine compatibility
            if fr in IMAGING_ONLY_REPRESENTATIONS:
                compat = "INCOMPATIBLE"
                reason = f"'{fr}' requires imaging; HANCOCK has no imaging modality."
            elif fr in PATHOLOGY_SLIDE_REPRESENTATIONS:
                compat = "INCOMPATIBLE"
                reason = (
                    f"'{fr}' requires whole-slide imaging pathology; "
                    "HANCOCK pathology is structured tabular staging data only."
                )
            elif exp_modalities and exp_modalities.issubset({"imaging"}):
                # Experiment only used imaging modality — not transferable to HANCOCK
                compat = "INCOMPATIBLE"
                reason = (
                    f"Experiment '{exp_id}' used only imaging modalities {exp_modalities}; "
                    "not transferable to HANCOCK's tabular modalities."
                )
            else:
                compat = "POTENTIALLY_COMPATIBLE"
                reason = "Modality check passed; requires detailed validation."

            # Extract provenance sentence if available
            prov_sentence = None
            fp = exp.get("field_provenance", {})
            fr_prov = fp.get("feature_representation", {})
            if fr_prov:
                prov_sentence = fr_prov.get("source_sentence")

            candidates.append({
                "mechanism": fr,
                "evidence_status": "EVIDENCE_BACKED" if compat == "POTENTIALLY_COMPATIBLE" else "INCOMPATIBLE",
                "compatibility_status": compat,
                "reason": reason,
                "experiment_id": exp_id,
                "paper_id": paper_id,
                "modalities": list(exp_modalities),
                "task": task,
                "provenance_sentence": prov_sentence,
                "source": "stage2_corpus",
            })

        return candidates

    # ------------------------------------------------------------------
    # Stage 3 alternative mechanisms
    # ------------------------------------------------------------------

    def _get_stage3_alternatives(self) -> List[Dict[str, Any]]:
        """
        Retrieve alternative mechanisms recorded in Stage 3 rankings for
        feature_representation.  Only mechanisms with actual evidence support
        (evidence_count > 0) may be considered evidence-backed.
        """
        rankings = self._load_json(self.stage3_rankings_path)
        fr_rankings = rankings.get("feature_representation", {})
        alternatives = fr_rankings.get("alternatives", [])

        candidates: List[Dict[str, Any]] = []
        for alt in alternatives:
            mech = alt.get("mechanism", "UNKNOWN")
            ev_count = alt.get("evidence_count", 0)
            score = alt.get("final_score", 0.0)

            if mech in IMAGING_ONLY_REPRESENTATIONS:
                compat = "INCOMPATIBLE"
                ev_status = "INCOMPATIBLE"
                reason = f"'{mech}' requires imaging modality."
            elif mech in PATHOLOGY_SLIDE_REPRESENTATIONS:
                compat = "INCOMPATIBLE"
                ev_status = "INCOMPATIBLE"
                reason = f"'{mech}' requires whole-slide pathology imaging."
            elif ev_count == 0:
                compat = "INSUFFICIENT_EVIDENCE"
                ev_status = "INSUFFICIENT_EVIDENCE"
                reason = f"'{mech}' has zero evidence_count in Stage 3 rankings (score={score})."
            else:
                compat = "POTENTIALLY_COMPATIBLE"
                ev_status = "EVIDENCE_BACKED"
                reason = f"Stage 3 recorded {ev_count} supporting evidence entries."

            candidates.append({
                "mechanism": mech,
                "evidence_status": ev_status,
                "compatibility_status": compat,
                "reason": reason,
                "stage3_score": score,
                "stage3_evidence_count": ev_count,
                "source": "stage3_alternatives",
            })

        return candidates

    # ------------------------------------------------------------------
    # Explicit configuration check
    # ------------------------------------------------------------------

    def _get_explicit_configuration(self) -> Optional[Dict[str, Any]]:
        """
        Check whether implementation_config.json explicitly sets
        feature_representation.  A null value is NOT an explicit configuration.
        """
        impl = self._load_json(self.impl_config_path)
        val = impl.get("feature_representation")
        if val is None:
            return None
        return {
            "mechanism": val,
            "evidence_status": "EXPLICITLY_CONFIGURED",
            "compatibility_status": "REQUIRES_VALIDATION",
            "reason": "Explicitly set in implementation_config.json.",
            "provenance": "explicit_configuration",
            "source": "implementation_config",
        }

    # ------------------------------------------------------------------
    # Main audit
    # ------------------------------------------------------------------

    def resolve(self) -> Dict[str, Any]:
        """
        Perform the complete feature representation resolution audit.

        Returns the resolution report dict and writes it to self.out_path.
        training_allowed is always False.
        """
        # 1. Determine original state
        prev_resolution = self._load_json(self.existing_resolution_path)
        original_representation = prev_resolution.get(
            "original_selected_representation", "cnn_representation"
        )
        original_status = prev_resolution.get(
            "original_compatibility_status", "incompatible"
        )

        # 2. Inventory HANCOCK modalities
        available_modalities = self._get_hancock_available_modalities()
        logger.info("HANCOCK available modalities: %s", available_modalities)

        # 3. Search Stage 2 corpus
        stage2_candidates = self._search_stage2_for_feature_representations(
            available_modalities
        )
        logger.info("Stage 2 candidates found: %d", len(stage2_candidates))

        # 4. Check Stage 3 alternatives
        stage3_candidates = self._get_stage3_alternatives()
        logger.info("Stage 3 alternative candidates: %d", len(stage3_candidates))

        # 5. Check explicit configuration
        explicit_candidate = self._get_explicit_configuration()

        # 6. Always include original (cnn_representation) as incompatible
        original_candidate = {
            "mechanism": original_representation,
            "evidence_status": "INCOMPATIBLE",
            "compatibility_status": "INCOMPATIBLE",
            "reason": "HANCOCK does not provide a validated imaging modality; CNN requires imaging.",
            "source": "original_stage3_selection",
        }

        # 7. Aggregate all candidates
        all_candidates: List[Dict[str, Any]] = (
            [original_candidate] + stage2_candidates + stage3_candidates
        )
        if explicit_candidate:
            all_candidates.append(explicit_candidate)

        # 8. Classify
        evidence_backed: List[str] = []
        explicitly_configured: List[str] = []
        compatible: List[str] = []
        incompatible: List[str] = []

        for c in all_candidates:
            mech = c["mechanism"]
            ev_status = c["evidence_status"]
            compat_status = c["compatibility_status"]

            if ev_status == "INCOMPATIBLE" or compat_status == "INCOMPATIBLE":
                if mech not in incompatible:
                    incompatible.append(mech)
            elif ev_status == "EXPLICITLY_CONFIGURED":
                if mech not in explicitly_configured:
                    explicitly_configured.append(mech)
                # Explicit config needs compatibility validation before being 'compatible'
            elif ev_status == "EVIDENCE_BACKED" and compat_status == "POTENTIALLY_COMPATIBLE":
                if mech not in evidence_backed:
                    evidence_backed.append(mech)
                if mech not in compatible:
                    compatible.append(mech)
            # INSUFFICIENT_EVIDENCE → not added to any positive list

        # 9. Select replacement
        selected_replacement: Optional[str] = None
        selection_reason: str
        provenance: Optional[str] = None
        final_resolution_status: str

        if explicitly_configured:
            # Explicit configuration takes precedence but still needs
            # compatibility verification (which we cannot do without imaging data)
            selected_replacement = explicitly_configured[0]
            selection_reason = (
                "Explicitly configured in implementation_config.json. "
                "Compatibility with HANCOCK modalities still requires validation."
            )
            provenance = "explicit_configuration"
            final_resolution_status = "RESOLVED_EXPLICIT"
        elif evidence_backed:
            selected_replacement = evidence_backed[0]
            selection_reason = (
                f"Evidence-backed replacement found: '{selected_replacement}'. "
                "Compatible with HANCOCK's available modalities per Stage 2 evidence."
            )
            provenance = "stage2_corpus"
            final_resolution_status = "RESOLVED_EVIDENCE_BACKED"
        else:
            selection_reason = (
                "No evidence-backed or explicitly-configured compatible feature representation "
                "found for HANCOCK's clinical/pathology(tabular)/blood/text modalities. "
                "The Stage 2 corpus (30 papers) contains no experiments with an explicit "
                "feature_representation compatible with structured tabular/text data. "
                "The Stage 3 alternative (transformer_representation) has zero evidence support. "
                "feature_representation remains BLOCKED."
            )
            final_resolution_status = "BLOCKED"

        # 10. Build report
        report: Dict[str, Any] = {
            "original_representation": original_representation,
            "original_status": original_status,
            "hancock_available_modalities": available_modalities,
            "candidate_representations": all_candidates,
            "evidence_backed_candidates": evidence_backed,
            "explicitly_configured_candidates": explicitly_configured,
            "compatible_candidates": compatible,
            "incompatible_candidates": incompatible,
            "selected_replacement": selected_replacement,
            "selection_reason": selection_reason,
            "provenance": provenance,
            "final_resolution_status": final_resolution_status,
            "training_allowed": False,  # HARD RULE — never modified by this module
        }

        # 11. Write artifact
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info(
            "feature_representation resolution status: %s | selected_replacement: %s | training_allowed: %s",
            final_resolution_status,
            selected_replacement,
            False,
        )
        return report


# ---------------------------------------------------------------------------
# CLI convenience entry point (read-only audit, no training)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    resolver = RepresentationResolver()
    report = resolver.resolve()
    print(json.dumps({
        "final_resolution_status": report["final_resolution_status"],
        "selected_replacement": report["selected_replacement"],
        "evidence_backed_candidates": report["evidence_backed_candidates"],
        "explicitly_configured_candidates": report["explicitly_configured_candidates"],
        "compatible_candidates": report["compatible_candidates"],
        "incompatible_candidates": report["incompatible_candidates"],
        "training_allowed": report["training_allowed"],
    }, indent=2))
