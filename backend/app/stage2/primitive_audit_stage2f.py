"""
Stage 2F: Evidence Sufficiency Audit for Implementation Primitives

Audits the genuine Stage 2C evidence corpus for the five implementation primitives:
1. missing_value_handling
2. categorical_encoding
3. base_learner
4. loss_function
5. imbalance_handling

Generates:
- evidence/metadata/stage2f_primitive_evidence_inventory.json
- evidence/metadata/stage2f_taxonomy_gap_audit.json
- evidence/metadata/stage2f_resolution_report.json
- evidence/metadata/stage2f_final_summary.json
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PRIMITIVES = [
    "missing_value_handling",
    "categorical_encoding",
    "base_learner",
    "loss_function",
    "imbalance_handling",
]

PRIMITIVE_KEYWORDS = {
    "missing_value_handling": [
        r"\bimputation\b",
        r"\bmissing\s+values?\b",
        r"\bcomplete-case\b",
        r"\bmissingness\b",
        r"\bmean\s+imputation\b",
        r"\bknn\s+imputation\b",
        r"\bmice\b",
    ],
    "categorical_encoding": [
        r"\bone-hot(?:\s+encoding)?\b",
        r"\bone\s+hot(?:\s+encoding)?\b",
        r"\bordinal\s+encoding\b",
        r"\bcategorical\s+encoding\b",
        r"\bdummy\s+variables?\b",
        r"\btarget\s+encoding\b",
    ],
    "base_learner": [
        r"\blogistic\s+regression\b",
        r"\brandom\s+forest\b",
        r"\bgradient\s+boosting\b",
        r"\bxgboost\b",
        r"\blightgbm\b",
        r"\bsupport\s+vector\s+machine\b",
        r"\bsvm\b",
        r"\bdecision\s+tree\b",
        r"\bmultilayer\s+perceptron\b",
        r"\bann\b",
    ],
    "loss_function": [
        r"\bcross[- ]entropy\b",
        r"\bbinary\s+cross[- ]entropy\b",
        r"\bfocal\s+loss\b",
        r"\bbce\b",
        r"\blog\s+loss\b",
    ],
    "imbalance_handling": [
        r"\bsmote\b",
        r"\boversampling\b",
        r"\bundersampling\b",
        r"\bclass\s+weights?\b",
        r"\bclass-weighted\b",
        r"\bbalanced\s+learning\b",
    ],
}

# Explicit procedural / experimental patterns requiring actual model implementation
EXPLICIT_PRIMITIVE_PATTERNS = {
    "missing_value_handling": [
        r"missing\s+(?:data|values?)\s+(?:were|was)\s+(?:imputed|handled\s+using|replaced\s+by)\b",
        r"(?:we\s+)?applied\s+(?:mean|median|knn|mice|multiple)\s+imputation\b",
        r"imputation\s+(?:was|were)\s+performed\b",
    ],
    "categorical_encoding": [
        r"categorical\s+(?:variables?|features?)\s+(?:were|was)\s+(?:encoded|converted)\b",
        r"(?:we\s+)?used\s+one-hot\s+encoding\b",
        r"one-hot\s+encoding\s+(?:was|were)\s+applied\b",
    ],
    "base_learner": [
        r"(?:random\s+forest|xgboost|gradient\s+boosting|logistic\s+regression|svm)\s+(?:model|classifier)\s+(?:was|were)\s+trained\b",
        r"(?:we\s+)?trained\s+a\s+(?:random\s+forest|xgboost|gradient\s+boosting|logistic\s+regression|svm)\b",
        r"(?:evaluated|compared)\s+using\s+random\s+forest\b",
    ],
    "loss_function": [
        r"optimized\s+(?:using|with)\s+(?:binary\s+cross-entropy|focal\s+loss|cross-entropy\s+loss)\b",
        r"(?:we\s+)?used\s+(?:binary\s+cross-entropy|focal\s+loss|cross-entropy)\s+as\s+the\s+loss\b",
        r"(?:cross-entropy|focal\s+loss)\s+(?:was|were)\s+used\b",
    ],
    "imbalance_handling": [
        r"(?:smote|random\s+oversampling|class\s+weighting|class-weighted)\s+(?:was|were)\s+applied\b",
        r"to\s+handle\s+class\s+imbalance[^\.\n]*(?:smote|class\s+weights?|oversampling)\b",
    ],
}

TARGET_LEAKAGE_PATTERNS = [
    r"\b(?:features?|inputs?|variables?|predictors?)\s+(?:including|such\s+as|with|contain(?:ing)?)\s+[^\.\n]*\b(recurrence|survival_status|days_to_recurrence|days_to_last_information|days_to_progress|days_to_metastasis)\b",
    r"\b(recurrence|survival_status|survival_status_with_cause|days_to_recurrence|days_to_last_information|days_to_progress_1|days_to_progress_2|days_to_metastasis_1)\s+(?:as\s+(?:an?\s+)?(?:input|feature|predictor)|variable\s+used\s+for)",
]


class Stage2FPrimitiveAuditor:
    def __init__(
        self,
        metadata_dir: str = "evidence/metadata",
        processed_dir: str = "evidence/processed",
    ):
        self.metadata_dir = Path(metadata_dir)
        self.processed_dir = Path(processed_dir)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.papers_path = self.processed_dir / "papers.jsonl"
        self.experiments_path = self.processed_dir / "experiments.jsonl"
        self.claims_path = self.processed_dir / "evidence_claims.jsonl"
        self.mechanisms_path = self.processed_dir / "mechanisms.jsonl"

    def _load_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        data = []
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        return data

    def _save_json(self, path: Path, data: Any):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Audit Primitives in Corpus
    # ──────────────────────────────────────────────────────────────────────────
    def audit_primitives(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        papers = self._load_jsonl(self.papers_path)
        experiments = self._load_jsonl(self.experiments_path)
        mechanisms = self._load_jsonl(self.mechanisms_path)

        exps_by_paper = {}
        for e in experiments:
            pid = e.get("paper_id")
            if pid:
                exps_by_paper.setdefault(pid, []).append(e)

        inventory = []
        counts_by_primitive = {p: {"explicit": 0, "indirect": 0, "leakage": 0, "not_evidence": 0} for p in PRIMITIVES}

        for prim in PRIMITIVES:
            keywords = PRIMITIVE_KEYWORDS[prim]
            explicit_pats = EXPLICIT_PRIMITIVE_PATTERNS[prim]

            for p in papers:
                pid = p.get("paper_id") or p.get("id")
                title = p.get("title", "")
                abstract = p.get("abstract", "") or ""
                p_exps = exps_by_paper.get(pid, [])

                # Scan paper text & experiment provenances
                combined_texts = [(title + "\n" + abstract, "abstract", None)]
                for e in p_exps:
                    eid = e.get("experiment_id")
                    prov = e.get("field_provenance", {})
                    for f_name, p_data in prov.items():
                        if isinstance(p_data, dict) and p_data.get("source_sentence"):
                            combined_texts.append((p_data.get("source_sentence"), p_data.get("section", "experiment_provenance"), eid))

                for text_block, loc, exp_id in combined_texts:
                    if not text_block:
                        continue

                    # Check keyword match
                    matched_kw = None
                    for kw in keywords:
                        if re.search(kw, text_block, re.I):
                            matched_kw = kw
                            break

                    if not matched_kw:
                        continue

                    # Check for explicit sentence
                    sentences = re.split(r"(?<=[.!?])\s+", text_block)
                    explicit_sentence = None
                    for s in sentences:
                        for exp_pat in explicit_pats:
                            if re.search(exp_pat, s, re.I):
                                explicit_sentence = s.strip()
                                break
                        if explicit_sentence:
                            break

                    # Check target leakage
                    has_leakage = any(bool(re.search(l_pat, text_block, re.I)) for l_pat in TARGET_LEAKAGE_PATTERNS)

                    # Modality compatibility (reject if requires imaging or text only)
                    requires_imaging = bool(re.search(r"\b(?:ct|pet|mri|radiomic|wsi|histopatholog\w+|imaging)\s+(?:features?|images?|modality)\s+(?:was|were|is)\s+(?:required|indispensable|essential)\b", text_block, re.I))
                    is_compatible = not requires_imaging

                    # Check provenance
                    has_provenance = bool(exp_id or p.get("abstract_available") or p.get("full_text_available"))

                    if has_leakage:
                        classification = "LEAKAGE_RISK"
                        rationale = "Candidate text references target-derived outcome variables."
                        counts_by_primitive[prim]["leakage"] += 1
                    elif not has_provenance:
                        classification = "MISSING_PROVENANCE"
                        rationale = "Candidate lacks traceable paper/experiment provenance."
                    elif explicit_sentence and is_compatible:
                        classification = "EXPLICIT_SUPPORTED"
                        rationale = f"Explicit experimental procedural statement found for {prim}."
                        counts_by_primitive[prim]["explicit"] += 1
                    else:
                        classification = "INDIRECT_INSUFFICIENT"
                        rationale = f"Keyword mention ({matched_kw}) without explicit experimental methodology for {prim}."
                        counts_by_primitive[prim]["indirect"] += 1

                    inventory.append({
                        "primitive": prim,
                        "paper_id": pid,
                        "experiment_id": exp_id,
                        "term_matched": matched_kw,
                        "classification": classification,
                        "classification_rationale": rationale,
                        "source_sentence": explicit_sentence or text_block[:200],
                        "source_location": loc,
                        "provenance_complete": has_provenance,
                        "hancock_compatible": is_compatible,
                        "leakage_safe": not has_leakage,
                    })

        self._save_json(self.metadata_dir / "stage2f_primitive_evidence_inventory.json", inventory)
        return inventory, counts_by_primitive

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Audit Taxonomy for Implementation Primitives
    # ──────────────────────────────────────────────────────────────────────────
    def audit_taxonomy(self) -> Dict[str, Any]:
        mechanisms = self._load_jsonl(self.mechanisms_path)

        categories_present = set(str(m.get("category")).lower() for m in mechanisms if m.get("category"))
        roles_present = set(str(m.get("role")).lower() for m in mechanisms if m.get("role"))

        primitive_taxonomy_status = {}
        for prim in PRIMITIVES:
            has_mech = any(
                prim.lower() in str(m.get("canonical_name", "")).lower() or
                prim.lower() in str(m.get("role", "")).lower() or
                prim.lower() in str(m.get("category", "")).lower()
                for m in mechanisms
            )
            primitive_taxonomy_status[prim] = "TAXONOMY_PRESENT" if has_mech else "TAXONOMY_GAP"

        taxonomy_audit = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_mechanisms_in_corpus": len(mechanisms),
            "categories_present": list(categories_present),
            "primitive_taxonomy_status": primitive_taxonomy_status,
            "overall_taxonomy_status": "TAXONOMY_GAPS_IDENTIFIED" if any(v == "TAXONOMY_GAP" for v in primitive_taxonomy_status.values()) else "TAXONOMY_SUFFICIENT",
            "analysis": (
                "The Stage 2/3 mechanism taxonomy only maps mechanisms for Representation, Fusion, and Attention. "
                "The implementation primitives (missing_value_handling, categorical_encoding, base_learner, "
                "loss_function, imbalance_handling) are not defined as mapped canonical categories in the taxonomy."
            ),
        }
        self._save_json(self.metadata_dir / "stage2f_taxonomy_gap_audit.json", taxonomy_audit)
        return taxonomy_audit

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Resolution Report & Final Decision
    # ──────────────────────────────────────────────────────────────────────────
    def evaluate_resolutions(
        self,
        inventory: List[Dict[str, Any]],
        counts_by_primitive: Dict[str, Any],
        taxonomy_audit: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        resolutions = {}

        for prim in PRIMITIVES:
            prim_cands = [inv for inv in inventory if inv["primitive"] == prim]
            explicit_cands = [inv for inv in prim_cands if inv["classification"] == "EXPLICIT_SUPPORTED"]
            indirect_cands = [inv for inv in prim_cands if inv["classification"] == "INDIRECT_INSUFFICIENT"]
            leakage_cands = [inv for inv in prim_cands if inv["classification"] == "LEAKAGE_RISK"]
            tax_status = taxonomy_audit["primitive_taxonomy_status"].get(prim, "TAXONOMY_GAP")

            if explicit_cands and tax_status == "TAXONOMY_PRESENT":
                best = explicit_cands[0]
                res_status = "RESOLVED_BY_EXISTING_EVIDENCE"
                ev_status = "EXPLICIT_SUPPORTED"
                comp_status = "SUPPORTED" if best["hancock_compatible"] else "INCOMPATIBLE"
                leak_status = "SAFE" if best["leakage_safe"] else "LEAKAGE_RISK"
                pid = best["paper_id"]
                eid = best["experiment_id"]
                prov = "PROVENANCE_VERIFIED"
            elif explicit_cands and tax_status == "TAXONOMY_GAP":
                best = explicit_cands[0]
                res_status = "TAXONOMY_GAP"
                ev_status = "EXPLICIT_SUPPORTED"
                comp_status = "SUPPORTED" if best["hancock_compatible"] else "INCOMPATIBLE"
                leak_status = "SAFE" if best["leakage_safe"] else "LEAKAGE_RISK"
                pid = best["paper_id"]
                eid = best["experiment_id"]
                prov = "PROVENANCE_VERIFIED"
            elif leakage_cands and not indirect_cands and not explicit_cands:
                best = leakage_cands[0]
                res_status = "BLOCKED_LEAKAGE"
                ev_status = "LEAKAGE_RISK"
                comp_status = "INCOMPATIBLE"
                leak_status = "LEAKAGE_RISK"
                pid = best["paper_id"]
                eid = best["experiment_id"]
                prov = None
            else:
                best = indirect_cands[0] if indirect_cands else None
                res_status = "UNSUPPORTED"
                ev_status = "INDIRECT_INSUFFICIENT" if indirect_cands else "UNSUPPORTED"
                comp_status = "UNTESTED"
                leak_status = "SAFE"
                pid = best["paper_id"] if best else None
                eid = best["experiment_id"] if best else None
                prov = None

            resolutions[prim] = {
                "component": prim,
                "evidence_status": ev_status,
                "best_candidate": best["term_matched"] if best else None,
                "paper_id": pid,
                "experiment_id": eid,
                "provenance": prov,
                "compatibility": comp_status,
                "leakage_status": leak_status,
                "taxonomy_status": tax_status,
                "resolution_status": res_status,
            }

        resolution_report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "resolutions": resolutions,
        }
        self._save_json(self.metadata_dir / "stage2f_resolution_report.json", resolution_report)

        # Final decision determination:
        # ALL_PRIMITIVES_SUPPORTED, PARTIALLY_SUPPORTED, TAXONOMY_GAPS_FOUND, NO_PRIMITIVE_EVIDENCE
        resolved_count = sum(1 for r in resolutions.values() if r["resolution_status"] == "RESOLVED_BY_EXISTING_EVIDENCE")
        tax_gap_count = sum(1 for r in resolutions.values() if r["resolution_status"] == "TAXONOMY_GAP")
        
        if resolved_count == len(PRIMITIVES):
            final_decision = "ALL_PRIMITIVES_SUPPORTED"
        elif resolved_count > 0:
            final_decision = "PARTIALLY_SUPPORTED"
        elif tax_gap_count > 0:
            final_decision = "TAXONOMY_GAPS_FOUND"
        else:
            final_decision = "NO_PRIMITIVE_EVIDENCE"

        final_summary = {
            "execution_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_papers_audited": len(self._load_jsonl(self.papers_path)),
            "total_experiments_audited": len(self._load_jsonl(self.experiments_path)),
            "primitives_audited": len(PRIMITIVES),
            "primitive_resolutions": {k: v["resolution_status"] for k, v in resolutions.items()},
            "explicitly_supported_primitives": [k for k, v in resolutions.items() if v["resolution_status"] == "RESOLVED_BY_EXISTING_EVIDENCE"],
            "unsupported_primitives": [k for k, v in resolutions.items() if v["resolution_status"] == "UNSUPPORTED"],
            "taxonomy_gaps": [k for k, v in resolutions.items() if v["resolution_status"] == "TAXONOMY_GAP"],
            "final_decision": final_decision,
            "training_allowed": False,
        }
        self._save_json(self.metadata_dir / "stage2f_final_summary.json", final_summary)
        return resolution_report, final_summary

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Run Stage 2F Audit
    # ──────────────────────────────────────────────────────────────────────────
    def run(self) -> Dict[str, Any]:
        inventory, counts_by_prim = self.audit_primitives()
        taxonomy_audit = self.audit_taxonomy()
        res_report, final_summary = self.evaluate_resolutions(inventory, counts_by_prim, taxonomy_audit)
        return final_summary


if __name__ == "__main__":
    auditor = Stage2FPrimitiveAuditor()
    summary = auditor.run()
    print("Stage 2F Complete. Final Decision:", summary["final_decision"])
    print(json.dumps(summary, indent=2))
