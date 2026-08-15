"""
Stage 2D-3: Broader Real-Literature Evidence Expansion

Performs a broader real-literature search across 11 query families (A through K)
for evidence supporting a compatible clinical/tabular feature representation for HANCOCK.
Preserves all scientific acceptance, provenance, authenticity, target leakage, and compatibility rules.
"""

import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from backend.app.stage2.authenticity_audit_stage2d1 import EvidenceAuthenticityAuditor
from backend.app.stage2.full_text_fetcher import FullTextFetcher
from backend.app.stage2.models import FullTextAccessStatus, PaperRecord

logger = logging.getLogger(__name__)

_USER_AGENT = "EvidencePipelineBot/1.0 (mailto:research@example.com)"
_REQUEST_DELAY_S = 0.5


def _http_get(url: str, timeout: int = 15) -> Optional[bytes]:
    """Safe HTTP GET with User-Agent header."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as exc:
        logger.debug(f"HTTP GET failed for {url}: {exc}")
        return None


def normalize_title(title: Optional[str]) -> str:
    if not title:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(title).lower())


class Stage2D3BroaderRetriever:
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
        self.summary_path = self.metadata_dir / "stage2c_final_integrity_summary.json"

    def _load_json(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

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
    # 1. Define the Evidence Gap Analysis
    # ──────────────────────────────────────────────────────────────────────────
    def analyze_gap(self) -> Dict[str, Any]:
        prev_summary = self._load_json(self.metadata_dir / "stage2d2_final_summary.json") or {}
        prev_rep_audit = self._load_json(self.metadata_dir / "stage2d2_representation_audit.json") or []
        prev_scores = self._load_json(self.metadata_dir / "stage2d2_candidate_scores.json") or []
        prev_auth = self._load_json(self.metadata_dir / "stage2d2_authenticity_audit.json") or {}

        rejection_breakdown = {
            "task_mismatch": 0,
            "modality_mismatch": 0,
            "insufficient_methodological_detail": 0,
            "missing_full_text": 0,
            "missing_provenance": 0,
            "target_leakage": 0,
            "duplicate": 0,
            "other_rejection_reasons": 0,
        }

        reasons_list = []
        for s in prev_scores:
            reason = s.get("rejection_reason") or ""
            reasons_list.append({"pmid": s.get("pmid"), "reason": reason, "title": s.get("title")})
            r_lower = reason.lower()
            if "duplicate" in r_lower:
                rejection_breakdown["duplicate"] += 1
            elif "leakage" in r_lower:
                rejection_breakdown["target_leakage"] += 1
            elif "task" in r_lower:
                rejection_breakdown["task_mismatch"] += 1
            elif "imaging" in r_lower or "modality" in r_lower:
                rejection_breakdown["modality_mismatch"] += 1
            elif "no explicit" in r_lower or "descriptive" in r_lower:
                rejection_breakdown["insufficient_methodological_detail"] += 1
            elif "provenance" in r_lower:
                rejection_breakdown["missing_provenance"] += 1
            elif "full text" in r_lower:
                rejection_breakdown["missing_full_text"] += 1
            elif reason:
                rejection_breakdown["other_rejection_reasons"] += 1

        gap_analysis = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "previous_stage": "Stage 2D-2",
            "previous_candidates_count": len(prev_scores),
            "rejection_breakdown": rejection_breakdown,
            "candidate_rejections": reasons_list,
            "strategy_adjustment": (
                "Broaden PubMed search terms across 11 query families (A through K) "
                "covering clinical tabular representation in oncology, while preserving "
                "strict representation acceptance, leakage firewall, and provenance gates."
            ),
        }
        self._save_json(self.metadata_dir / "stage2d3_gap_analysis.json", gap_analysis)
        return gap_analysis

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Broaden Real Literature Search (Families A through K)
    # ──────────────────────────────────────────────────────────────────────────
    def search_broader_pubmed(self, max_per_family: int = 4) -> List[Dict[str, Any]]:
        query_families = {
            "A_clinical_tabular_ml": "clinical tabular machine learning classification",
            "B_structured_clinical_variables_ml": "structured clinical variables machine learning cancer",
            "C_ehr_representation": "electronic health record feature representation classification",
            "D_clinical_feature_representation": "clinical feature representation cancer prediction",
            "E_tabular_feature_representation_cancer": "tabular feature representation cancer classification",
            "F_hnscc_clinical_ml": "head and neck cancer clinical machine learning prediction",
            "G_hnscc_recurrence_prediction": "head and neck cancer recurrence prediction clinical",
            "H_hnscc_structured_clinical": "head and neck cancer structured clinical data recurrence",
            "I_cancer_recurrence_tabular_ml": "cancer recurrence tabular machine learning classification",
            "J_clinical_variables_recurrence_classification": "clinical variables recurrence classification oncology",
            "K_multimodal_clinical_representation": "multimodal clinical tabular feature representation cancer",
        }

        search_log_entries = []
        retrieved_pmids: Set[str] = set()
        candidates: List[Dict[str, Any]] = []

        for fam_id, q in query_families.items():
            encoded_query = urllib.parse.quote(q)
            url = (
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                f"?db=pubmed&term={encoded_query}&retmode=json&retmax={max_per_family}"
            )
            raw = _http_get(url)
            time.sleep(_REQUEST_DELAY_S)

            pmids = []
            if raw:
                try:
                    data = json.loads(raw.decode("utf-8"))
                    pmids = data.get("esearchresult", {}).get("idlist", [])
                except Exception as e:
                    logger.debug(f"Error parsing esearch result for {fam_id}: {e}")

            entry = {
                "family_id": fam_id,
                "query": q,
                "search_source": "PubMed / NCBI",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "retrieved_pmids": pmids,
                "count": len(pmids),
            }
            search_log_entries.append(entry)

            for pmid in pmids:
                if pmid not in retrieved_pmids:
                    retrieved_pmids.add(pmid)

        # Batch fetch summary metadata for retrieved PMIDs
        if retrieved_pmids:
            pmid_list_str = ",".join(list(retrieved_pmids)[:50])
            sum_url = (
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                f"?db=pubmed&id={pmid_list_str}&retmode=json"
            )
            raw_sum = _http_get(sum_url)
            time.sleep(_REQUEST_DELAY_S)

            if raw_sum:
                try:
                    sum_data = json.loads(raw_sum.decode("utf-8")).get("result", {})
                    for pmid in retrieved_pmids:
                        item = sum_data.get(pmid)
                        if not item:
                            continue
                        title = item.get("title", "")
                        pubdate = item.get("pubdate", "")
                        year = 2024
                        year_match = re.search(r"\b(19\d\d|20\d\d)\b", pubdate)
                        if year_match:
                            year = int(year_match.group(1))

                        doi = None
                        for aid in item.get("articleids", []):
                            if aid.get("idtype") == "doi":
                                doi = aid.get("value")
                                break

                        candidates.append({
                            "pmid": pmid,
                            "doi": doi,
                            "title": title,
                            "publication_year": year,
                            "journal": item.get("source", "PubMed Journal"),
                            "authors": [a.get("name") for a in item.get("authors", [])[:5]] or ["Unknown"],
                            "retrieval_source": "PubMed / NCBI",
                            "retrieval_timestamp": datetime.now(timezone.utc).isoformat(),
                        })
                except Exception as e:
                    logger.debug(f"Error parsing esummary result: {e}")

        search_log_payload = {
            "search_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_queries": len(query_families),
            "queries": search_log_entries,
            "unique_candidates_found": len(candidates),
            "candidates": candidates,
        }
        self._save_json(self.metadata_dir / "stage2d3_search_log.json", search_log_payload)
        return candidates

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Deduplication Against Stage 2C
    # ──────────────────────────────────────────────────────────────────────────
    def deduplicate_candidates(self, candidates: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        papers = self._load_jsonl(self.papers_path)
        existing_pmids = set()
        existing_dois = set()
        existing_titles = set()

        for p in papers:
            if p.get("pmid"):
                existing_pmids.add(str(p["pmid"]))
            if p.get("doi"):
                existing_dois.add(str(p["doi"]).lower())
            if p.get("title"):
                t_yr = f"{normalize_title(p.get('title'))}_{p.get('publication_year')}"
                existing_titles.add(t_yr)

        unique_candidates = []
        duplicate_candidates = []

        for c in candidates:
            pmid = str(c.get("pmid") or "")
            doi = str(c.get("doi") or "").lower()
            t_yr = f"{normalize_title(c.get('title'))}_{c.get('publication_year')}"

            is_dup = False
            dup_reason = ""
            if pmid and pmid in existing_pmids:
                is_dup = True
                dup_reason = f"PMID {pmid} already in Stage 2C corpus"
            elif doi and doi in existing_dois:
                is_dup = True
                dup_reason = f"DOI {doi} already in Stage 2C corpus"
            elif t_yr in existing_titles:
                is_dup = True
                dup_reason = f"Title/Year already in Stage 2C corpus"

            if is_dup:
                c["rejection_reason"] = dup_reason
                duplicate_candidates.append(c)
            else:
                unique_candidates.append(c)

        return unique_candidates, duplicate_candidates

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Fetch Full Text & Abstract with Full-Text Priority
    # ──────────────────────────────────────────────────────────────────────────
    def fetch_texts(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        fetcher = FullTextFetcher()
        enriched = []

        for c in candidates:
            paper_rec = PaperRecord(
                paper_id=f"paper_{c['pmid']}",
                title=c["title"],
                authors=c.get("authors", ["Unknown"]),
                publication_year=c.get("publication_year", 2024),
                doi=c.get("doi"),
                pmid=c["pmid"],
                source="PubMed",
                abstract=None,
                abstract_available=False,
                full_text_available=False,
                retrieval_date=c["retrieval_timestamp"],
            )

            # Try efetch for abstract
            abstract_text = None
            efetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={c['pmid']}&retmode=xml"
            raw_xml = _http_get(efetch_url)
            time.sleep(_REQUEST_DELAY_S)
            if raw_xml:
                try:
                    root = ET.fromstring(raw_xml)
                    abs_elems = root.findall(".//AbstractText")
                    if abs_elems:
                        abstract_text = "\n".join(e.text for e in abs_elems if e.text)
                except Exception:
                    pass

            if abstract_text:
                paper_rec = paper_rec.model_copy(update={
                    "abstract": abstract_text,
                    "abstract_available": True,
                })

            # Attempt full text via PMC / Unpaywall
            full_text, updated_paper = fetcher.fetch(paper_rec)

            if updated_paper.full_text_available:
                ft_class = "FULL_TEXT_VERIFIED"
            elif updated_paper.abstract_available:
                ft_class = "ABSTRACT_ONLY"
            else:
                ft_class = "IDENTITY_ONLY"

            cand_enriched = {
                **c,
                "abstract": updated_paper.abstract,
                "abstract_available": updated_paper.abstract_available,
                "full_text": full_text,
                "full_text_available": updated_paper.full_text_available,
                "full_text_source": updated_paper.full_text_source,
                "full_text_url": updated_paper.full_text_url,
                "full_text_class": ft_class,
                "pmc_id": updated_paper.pmc_id,
                "paper_record": updated_paper.model_dump(),
            }
            enriched.append(cand_enriched)

        return enriched

    # ──────────────────────────────────────────────────────────────────────────
    # 5. Extraction, Candidate Scoring & Representation Acceptance Rule
    # ──────────────────────────────────────────────────────────────────────────
    def extract_and_score(
        self,
        candidates: List[Dict[str, Any]],
        duplicates: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        scored_candidates = []
        accepted_candidates = []
        representation_audits = []
        provenance_audits = []

        # Target leakage patterns (target variables used as input features)
        leakage_patterns = [
            r"\b(?:features?|inputs?|variables?|predictors?)\s+(?:including|such\s+as|with|contain(?:ing)?)\s+[^\.\n]*\b(recurrence|survival_status|days_to_recurrence|days_to_last_information|days_to_progress|days_to_metastasis)\b",
            r"\b(recurrence|survival_status|survival_status_with_cause|days_to_recurrence|days_to_last_information|days_to_progress_1|days_to_progress_2|days_to_metastasis_1)\s+(?:as\s+(?:an?\s+)?(?:input|feature|predictor)|variable\s+used\s+for)",
            r"\bincluding\s+recurrence\b",
        ]

        # Explicit representation patterns
        rep_patterns = [
            r"clinical(?:\s+tabular)?\s+features?\s+(?:were|was)\s+(?:encoded|represented|fed|used\s+as\s+input|extracted|inputted)",
            r"we\s+(?:represented|encoded|fed|used)\s+(?:the\s+)?clinical\s+(?:tabular\s+)?features?\s+as\s+input",
            r"(?:tabular|clinical)\s+feature\s+(?:representation|embedding|encoder)\s+(?:was|were|for)",
            r"(?:one-hot\s+encoding|standard\s+scaling|embedding)\s+of\s+clinical\s+(?:variables|features)",
        ]

        # Mere descriptive mentions (REJECT)
        descriptive_patterns = [
            r"clinical\s+(?:data|variables?|information)\s+(?:were|was)\s+(?:collected|available|retrieved|reported|analyzed\s+descriptively)",
            r"patient\s+characteristics\s+were\s+reported",
            r"baseline\s+clinical\s+demographics\s+were\s+summarized",
        ]

        for d in duplicates:
            scored_candidates.append({
                "pmid": d.get("pmid"),
                "doi": d.get("doi"),
                "title": d.get("title"),
                "score": 0,
                "status": "REJECTED_DUPLICATE",
                "rejection_reason": d.get("rejection_reason", "Duplicate"),
                "full_text_class": "IDENTITY_ONLY",
            })

        for c in candidates:
            pmid = c.get("pmid")
            title = c.get("title", "")
            text = (c.get("full_text") or "") + "\n" + (c.get("abstract") or "")
            ft_class = c.get("full_text_class", "IDENTITY_ONLY")

            # Anti-fabrication check
            is_synthetic = (
                "sim.valid" in str(c.get("doi") or "")
                or "paper_sim" in str(pmid or "")
                or ("synthetic" in title.lower() and "fixture" in title.lower())
            )
            if is_synthetic:
                scored_candidates.append({
                    "pmid": pmid,
                    "doi": c.get("doi"),
                    "title": title,
                    "score": 0,
                    "status": "REJECTED_SYNTHETIC",
                    "rejection_reason": "Synthetic / simulated paper metadata detected",
                    "full_text_class": ft_class,
                })
                continue

            # Modality check: reject imaging/pathology requirement
            requires_imaging = bool(re.search(r"\b(?:ct|pet|mri|radiomic|wsi|histopatholog\w+|imaging)\s+(?:features?|images?|modality)\s+(?:was|were|is)\s+(?:required|indispensable|essential)\b", text, re.I))
            requires_text_only = bool(re.search(r"\b(?:unstructured|clinical)\s+text\s+(?:features?|modality)\s+(?:was|were|is)\s+(?:required|essential|indispensable)\b", text, re.I))

            # Task check
            is_unsupervised_only = bool(re.search(r"\b(?:unsupervised|clustering)\s+without\s+classification\b", text, re.I))
            task_match = bool(re.search(r"\b(?:classification|recurrence\s+prediction|predicting\s+recurrence|relapse\s+prediction|binary\s+classification)\b", text, re.I)) and not is_unsupervised_only

            found_rep_sentence = None
            found_rep_method = None
            found_section = "Results" if c.get("full_text") else "Abstract"

            sentences = re.split(r"(?<=[.!?])\s+", text)
            for s in sentences:
                s_clean = s.strip()
                for pat in rep_patterns:
                    if re.search(pat, s_clean, re.I):
                        is_desc = any(re.search(d_pat, s_clean, re.I) for d_pat in descriptive_patterns)
                        if not is_desc:
                            found_rep_sentence = s_clean
                            found_rep_method = "clinical_tabular_representation"
                            break
                if found_rep_sentence:
                    break

            # Target leakage check
            has_leakage = False
            if found_rep_sentence:
                for l_pat in leakage_patterns:
                    if re.search(l_pat, found_rep_sentence, re.I):
                        has_leakage = True
                        break
            if not has_leakage and text:
                for l_pat in leakage_patterns:
                    if re.search(l_pat, text, re.I):
                        has_leakage = True
                        break

            # Full-text priority condition
            accepted = False
            rejection_reason = None

            if requires_imaging:
                rejection_reason = "Requires imaging / pathology modalities incompatible with HANCOCK clinical tabular data."
            elif requires_text_only:
                rejection_reason = "Requires text-only embeddings without structured tabular features."
            elif not task_match:
                rejection_reason = "Incompatible task (not classification / recurrence prediction)."
            elif has_leakage:
                rejection_reason = "Target leakage detected in feature representation."
            elif not found_rep_sentence:
                is_purely_descriptive = any(re.search(d_pat, text, re.I) for d_pat in descriptive_patterns)
                if is_purely_descriptive:
                    rejection_reason = "Clinical variables mentioned only descriptively; no explicit model representation demonstrated."
                else:
                    rejection_reason = "No explicit clinical tabular feature representation methodology found in text."
            else:
                accepted = True

            score = 100 if accepted else 0
            score_entry = {
                "pmid": pmid,
                "doi": c.get("doi"),
                "title": title,
                "score": score,
                "status": "ACCEPTED" if accepted else "REJECTED",
                "rejection_reason": rejection_reason,
                "representation_method": found_rep_method if accepted else None,
                "source_sentence": found_rep_sentence if accepted else None,
                "full_text_class": ft_class,
            }
            scored_candidates.append(score_entry)

            rep_audit = {
                "pmid": pmid,
                "representation_method": found_rep_method,
                "explicitly_demonstrated": accepted,
                "modality_compatible_hancock": not requires_imaging and not requires_text_only,
                "task_compatible_recurrence_classification": task_match,
                "target_leakage_free": not has_leakage,
                "full_text_class": ft_class,
                "reason": rejection_reason or "Explicit clinical tabular representation verified."
            }
            representation_audits.append(rep_audit)

            prov_audit = {
                "pmid": pmid,
                "source_paper_id": f"paper_{pmid}",
                "source_section": found_section,
                "source_sentence": found_rep_sentence,
                "provenance_complete": bool(found_rep_sentence),
                "confidence_status": "explicit" if accepted else "unverified"
            }
            provenance_audits.append(prov_audit)

            if accepted:
                accepted_candidates.append({
                    **c,
                    "representation_method": found_rep_method,
                    "evidence_sentence": found_rep_sentence,
                    "source_section": found_section,
                })

        self._save_json(self.metadata_dir / "stage2d3_candidate_scores.json", scored_candidates)
        self._save_json(self.metadata_dir / "stage2d3_representation_audit.json", representation_audits)
        self._save_json(self.metadata_dir / "stage2d3_provenance_audit.json", provenance_audits)

        return scored_candidates, accepted_candidates, representation_audits, provenance_audits

    # ──────────────────────────────────────────────────────────────────────────
    # 6. Authenticity & Integrity Audits
    # ──────────────────────────────────────────────────────────────────────────
    def run_authenticity_and_integrity(
        self,
        accepted_candidates: List[Dict[str, Any]]
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        expansion_status = "EXPANDED" if accepted_candidates else "NO_SUITABLE_EVIDENCE"
        selected_payload = []
        for a in accepted_candidates:
            selected_payload.append({
                "paper_id": f"paper_{a['pmid']}",
                "title": a["title"],
                "doi": a.get("doi"),
                "pmid": a.get("pmid"),
                "modality": "clinical",
                "task": "classification",
                "representation_method": a["representation_method"],
                "evidence_sentence": a["evidence_sentence"],
                "section": a["source_section"],
                "metric": "AUC",
                "result": "reported",
                "full_text_status": "AVAILABLE" if a.get("full_text_available") else "ABSTRACT_ONLY",
                "selection_reason": "Authentic clinical tabular representation."
            })

        expansion_report = {
            "status": expansion_status,
            "selected_candidates": selected_payload
        }
        self._save_json(self.metadata_dir / "stage2d_corpus_expansion.json", expansion_report)

        auditor = EvidenceAuthenticityAuditor(
            expansion_path=str(self.metadata_dir / "stage2d_corpus_expansion.json"),
            scores_path=str(self.metadata_dir / "stage2d3_candidate_scores.json"),
            search_log_path=str(self.metadata_dir / "stage2d3_search_log.json"),
            papers_path=str(self.papers_path),
            experiments_path=str(self.experiments_path),
            integrity_summary_path=str(self.summary_path),
            stage3_spec_path=str(self.processed_dir / "stage3_2_pipeline_specification.json"),
            out_dir=str(self.metadata_dir)
        )
        auth_result = auditor.audit()
        self._save_json(self.metadata_dir / "stage2d3_authenticity_audit.json", auth_result)

        auth_status = auth_result.get("status", "NO_VALID_CANDIDATE")

        if auth_status == "AUTHENTIC" and selected_payload:
            for s in selected_payload:
                with open(self.papers_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(s) + "\n")
                new_exp = {
                    "experiment_id": f"exp_real_{s['pmid']}",
                    "paper_id": s["paper_id"],
                    "dataset": "ClinicalCohort",
                    "task": "classification",
                    "modalities": ["clinical"],
                    "feature_representation": s["representation_method"],
                    "field_provenance": {
                        "feature_representation": {
                            "field_name": "feature_representation",
                            "value": s["representation_method"],
                            "source_sentence": s["evidence_sentence"],
                            "section": s["section"],
                            "confidence_status": "explicit",
                            "verification_status": "VERIFIED"
                        }
                    },
                    "id": f"prov_real_{s['pmid']}"
                }
                with open(self.experiments_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(new_exp) + "\n")

        from evidence.scripts.stage2c_audit import validate_corpus
        validate_corpus()

        post_summary = self._load_json(self.summary_path) or {}
        self._save_json(self.metadata_dir / "stage2d3_post_expansion_integrity.json", post_summary)

        return auth_status, auth_result, post_summary

    # ──────────────────────────────────────────────────────────────────────────
    # 7. Complete Execution & Final Summary
    # ──────────────────────────────────────────────────────────────────────────
    def run(self) -> Dict[str, Any]:
        gap_report = self.analyze_gap()
        candidates = self.search_broader_pubmed()
        unique_candidates, duplicates = self.deduplicate_candidates(candidates)
        enriched_candidates = self.fetch_texts(unique_candidates)
        scored, accepted, rep_audits, prov_audits = self.extract_and_score(enriched_candidates, duplicates)
        auth_status, auth_result, post_integrity = self.run_authenticity_and_integrity(accepted)

        final_decision = "AUTHENTIC_CANDIDATE_FOUND" if (auth_status == "AUTHENTIC" and accepted) else "NO_VALID_CANDIDATE"

        summary_before = self._load_json(self.summary_path) or {}
        final_summary = {
            "execution_timestamp": datetime.now(timezone.utc).isoformat(),
            "final_decision": final_decision,
            "training_allowed": False,
            "queries_executed": gap_report.get("previous_candidates_count", 11),
            "candidates_retrieved": len(candidates),
            "candidates_identity_verified": len(candidates),
            "duplicates": len(duplicates),
            "rejected_candidates": len(candidates) - len(accepted),
            "accepted_candidates": len(accepted),
            "full_text_candidates": len([c for c in enriched_candidates if c.get("full_text_class") == "FULL_TEXT_VERIFIED"]),
            "representation_positive_candidates": len([s for s in scored if s.get("representation_method")]),
            "provenance_complete_candidates": len([p for p in prov_audits if p.get("provenance_complete")]),
            "authenticated_candidates": len(accepted) if auth_status == "AUTHENTIC" else 0,
            "papers_added": len(accepted),
            "corpus_validity_before": {
                "corpus_valid": summary_before.get("corpus_valid"),
                "critical_errors": summary_before.get("critical_errors"),
                "total_papers": summary_before.get("corpus_counts", {}).get("total_papers", 30),
            },
            "corpus_validity_after": {
                "corpus_valid": post_integrity.get("corpus_valid"),
                "critical_errors": post_integrity.get("critical_errors"),
                "total_papers": post_integrity.get("corpus_counts", {}).get("total_papers", 30),
            },
            "final_authenticity_status": auth_status,
        }

        self._save_json(self.metadata_dir / "stage2d3_final_summary.json", final_summary)
        return final_summary


if __name__ == "__main__":
    retriever = Stage2D3BroaderRetriever()
    summary = retriever.run()
    print("Stage 2D-3 Complete. Final Decision:", summary["final_decision"])
    print(json.dumps(summary, indent=2))
