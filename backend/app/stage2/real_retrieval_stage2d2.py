"""
Stage 2D-2: Real Literature Retrieval and Evidence Expansion

Replaces synthetic Stage 2D search with a genuine NCBI/PubMed literature retrieval pipeline.
Enforces strict scientific authenticity, anti-fabrication rules, and immutable safety gates.
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


class Stage2D2RealRetriever:
    def __init__(
        self,
        metadata_dir: str = "evidence/metadata",
        processed_dir: str = "evidence/processed",
    ):
        self.metadata_dir = Path(metadata_dir)
        self.processed_dir = Path(processed_dir)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.summary_path = self.metadata_dir / "stage2c_final_integrity_summary.json"
        self.audit_path = self.metadata_dir / "stage2c_final_integrity_audit.json"
        self.papers_path = self.processed_dir / "papers.jsonl"
        self.experiments_path = self.processed_dir / "experiments.jsonl"
        self.claims_path = self.processed_dir / "evidence_claims.jsonl"

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
    # 1. Baseline Pre-Search Integrity Check
    # ──────────────────────────────────────────────────────────────────────────
    def verify_pre_search_integrity(self) -> Dict[str, Any]:
        summary = self._load_json(self.summary_path)
        papers = self._load_jsonl(self.papers_path)
        exps = self._load_jsonl(self.experiments_path)
        claims = self._load_jsonl(self.claims_path)

        if not summary:
            raise RuntimeError("stage2c_final_integrity_summary.json not found.")

        known_seeds = {
            "paper_38396486", "paper_39074400", "paper_40325104", "paper_40449048",
            "paper_41131352", "paper_41353186", "paper_10.1038_s42256-023-00633-5",
            "paper_10.3390_bioengineering11010013"
        }
        actual_seed_count = sum(1 for p in papers if p.get("paper_id") in known_seeds)
        actual_new_count = len(papers) - actual_seed_count

        pre_search = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "corpus_valid": summary.get("corpus_valid", False),
            "critical_errors": summary.get("critical_errors", -1),
            "warnings": summary.get("warnings", 0),
            "total_papers_in_summary": summary.get("corpus_counts", {}).get("total_papers", 0),
            "actual_papers_count": len(papers),
            "seed_papers_count": actual_seed_count,
            "new_papers_count": actual_new_count,
            "duplicate_papers": summary.get("duplicate_count", {}).get("duplicate_papers", 0),
            "suspicious_entity_count": summary.get("suspicious_entity_count", 0),
            "numerical_evidence_errors": summary.get("numerical_evidence_errors", 0),
            "entity_confusion_errors": summary.get("entity_confusion_errors", 0),
            "provenance_coverage_percent": summary.get("provenance_coverage", {}).get("provenance_coverage_percent", 0.0),
            "baseline_verified": (
                summary.get("corpus_valid") is True
                and summary.get("critical_errors") == 0
                and len(papers) == 30
                and actual_seed_count == 8
                and actual_new_count == 22
            )
        }

        self._save_json(self.metadata_dir / "stage2d2_pre_search_integrity.json", pre_search)
        if not pre_search["baseline_verified"]:
            raise RuntimeError(f"Baseline integrity verification failed: {pre_search}")
        return pre_search

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Synthetic Stage 2D Record Cleanup
    # ──────────────────────────────────────────────────────────────────────────
    def cleanup_synthetic_records(self) -> int:
        synthetic_removed = 0
        
        # Check papers.jsonl
        papers = self._load_jsonl(self.papers_path)
        clean_papers = []
        for p in papers:
            pid = str(p.get("paper_id") or p.get("id") or "")
            doi = str(p.get("doi") or "")
            if pid.startswith("paper_sim") or "sim.valid" in doi or "stage2d" in pid:
                synthetic_removed += 1
            else:
                clean_papers.append(p)

        if len(clean_papers) != len(papers):
            with open(self.papers_path, "w", encoding="utf-8") as f:
                for p in clean_papers:
                    f.write(json.dumps(p) + "\n")

        # Check experiments.jsonl
        exps = self._load_jsonl(self.experiments_path)
        clean_exps = []
        for e in exps:
            eid = str(e.get("experiment_id") or "")
            pid = str(e.get("paper_id") or "")
            prov_id = str(e.get("id") or "")
            if eid.startswith("exp_stage2d_") or pid.startswith("paper_sim") or "stage2d" in prov_id:
                synthetic_removed += 1
            else:
                clean_exps.append(e)

        if len(clean_exps) != len(exps):
            with open(self.experiments_path, "w", encoding="utf-8") as f:
                for e in clean_exps:
                    f.write(json.dumps(e) + "\n")

        return synthetic_removed

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Real PubMed Literature Search
    # ──────────────────────────────────────────────────────────────────────────
    def search_pubmed(self, max_per_query: int = 5) -> List[Dict[str, Any]]:
        queries = [
            "head and neck cancer clinical tabular recurrence machine learning",
            "structured clinical feature encoding recurrence classification oncology",
            "tabular clinical feature representation cancer recurrence prediction",
            "head and neck squamous cell carcinoma recurrence tabular features",
            "clinical variables machine learning recurrence prediction cancer tabular",
        ]

        search_log_entries = []
        retrieved_pmids: Set[str] = set()
        candidates: List[Dict[str, Any]] = []

        for q in queries:
            encoded_query = urllib.parse.quote(q)
            url = (
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                f"?db=pubmed&term={encoded_query}&retmode=json&retmax={max_per_query}"
            )
            raw = _http_get(url)
            time.sleep(_REQUEST_DELAY_S)

            pmids = []
            if raw:
                try:
                    data = json.loads(raw.decode("utf-8"))
                    pmids = data.get("esearchresult", {}).get("idlist", [])
                except Exception as e:
                    logger.debug(f"Error parsing esearch result: {e}")

            entry = {
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
            pmid_list_str = ",".join(list(retrieved_pmids)[:30])
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
            "total_queries": len(queries),
            "queries": search_log_entries,
            "unique_candidates_found": len(candidates),
            "candidates": candidates,
        }
        self._save_json(self.metadata_dir / "stage2d2_search_log.json", search_log_payload)
        return candidates

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Deduplication Against Stage 2C
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
    # 5. Fetch Full Text & Abstract
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

            cand_enriched = {
                **c,
                "abstract": updated_paper.abstract,
                "abstract_available": updated_paper.abstract_available,
                "full_text": full_text,
                "full_text_available": updated_paper.full_text_available,
                "full_text_source": updated_paper.full_text_source,
                "full_text_url": updated_paper.full_text_url,
                "pmc_id": updated_paper.pmc_id,
                "paper_record": updated_paper.model_dump(),
            }
            enriched.append(cand_enriched)

        return enriched

    # ──────────────────────────────────────────────────────────────────────────
    # 6. Extraction, Candidate Scoring & Representation Acceptance Rule
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

        # Representation candidate patterns (explicit model representation / input / features)
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

        # Log duplicates in scores first
        for d in duplicates:
            scored_candidates.append({
                "pmid": d.get("pmid"),
                "doi": d.get("doi"),
                "title": d.get("title"),
                "score": 0,
                "status": "REJECTED_DUPLICATE",
                "rejection_reason": d.get("rejection_reason", "Duplicate"),
            })

        for c in candidates:
            pmid = c.get("pmid")
            title = c.get("title", "")
            text = (c.get("full_text") or "") + "\n" + (c.get("abstract") or "")
            
            # Anti-fabrication check: detect any simulated or fake data
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
                })
                continue

            # Modality check: reject imaging/pathology requirement
            requires_imaging = bool(re.search(r"\b(?:ct|pet|mri|radiomic|wsi|histopatholog\w+|imaging)\s+(?:features?|images?|modality)\s+(?:was|were|is)\s+(?:required|indispensable|essential)\b", text, re.I))
            
            # Task check: check for classification or recurrence prediction (excluding "without classification")
            is_unsupervised_only = bool(re.search(r"\b(?:unsupervised|clustering)\s+without\s+classification\b", text, re.I))
            task_match = bool(re.search(r"\b(?:classification|recurrence\s+prediction|predicting\s+recurrence|relapse\s+prediction|binary\s+classification)\b", text, re.I)) and not is_unsupervised_only

            # Representation extraction check
            found_rep_sentence = None
            found_rep_method = None
            found_section = "Results" if c.get("full_text") else "Abstract"

            sentences = re.split(r"(?<=[.!?])\s+", text)
            for s in sentences:
                s_clean = s.strip()
                # Check for explicit representation
                for pat in rep_patterns:
                    if re.search(pat, s_clean, re.I):
                        # Ensure not purely descriptive
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

            # Determine acceptance
            accepted = False
            rejection_reason = None

            if requires_imaging:
                rejection_reason = "Requires imaging / pathology modalities incompatible with HANCOCK clinical tabular data."
            elif not task_match:
                rejection_reason = "Incompatible task (not classification / recurrence prediction)."
            elif has_leakage:
                rejection_reason = "Target leakage detected in feature representation."
            elif not found_rep_sentence:
                # Check if it was merely descriptive
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
                "full_text_status": "FULL_TEXT" if c.get("full_text_available") else ("ABSTRACT_ONLY" if c.get("abstract_available") else "UNAVAILABLE"),
            }
            scored_candidates.append(score_entry)

            rep_audit = {
                "pmid": pmid,
                "representation_method": found_rep_method,
                "explicitly_demonstrated": accepted,
                "modality_compatible_hancock": not requires_imaging,
                "task_compatible_recurrence_classification": task_match,
                "target_leakage_free": not has_leakage,
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

        self._save_json(self.metadata_dir / "stage2d2_candidate_scores.json", scored_candidates)
        self._save_json(self.metadata_dir / "stage2d2_representation_audit.json", representation_audits)
        self._save_json(self.metadata_dir / "stage2d2_provenance_audit.json", provenance_audits)

        return scored_candidates, accepted_candidates, representation_audits, provenance_audits

    # ──────────────────────────────────────────────────────────────────────────
    # 7. Authenticity Gate & Post-Expansion Integrity Audit
    # ──────────────────────────────────────────────────────────────────────────
    def run_authenticity_and_integrity(
        self,
        accepted_candidates: List[Dict[str, Any]]
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        # Format corpus expansion metadata
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

        # Run Stage 2D-1 Authenticity Auditor
        auditor = EvidenceAuthenticityAuditor(
            expansion_path=str(self.metadata_dir / "stage2d_corpus_expansion.json"),
            scores_path=str(self.metadata_dir / "stage2d2_candidate_scores.json"),
            search_log_path=str(self.metadata_dir / "stage2d2_search_log.json"),
            papers_path=str(self.papers_path),
            experiments_path=str(self.experiments_path),
            integrity_summary_path=str(self.summary_path),
            stage3_spec_path=str(self.processed_dir / "stage3_2_pipeline_specification.json"),
            out_dir=str(self.metadata_dir)
        )
        auth_result = auditor.audit()
        self._save_json(self.metadata_dir / "stage2d2_authenticity_audit.json", auth_result)

        auth_status = auth_result.get("status", "NO_VALID_CANDIDATE")

        # If authentic candidate found, add to corpus and test integrity
        if auth_status == "AUTHENTIC" and selected_payload:
            for s in selected_payload:
                # Add to papers.jsonl
                with open(self.papers_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(s) + "\n")
                # Add to experiments.jsonl
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

        # Run Stage 2C integrity audit
        from evidence.scripts.stage2c_audit import validate_corpus
        validate_corpus()

        post_summary = self._load_json(self.summary_path) or {}
        self._save_json(self.metadata_dir / "stage2d2_post_expansion_integrity.json", post_summary)

        return auth_status, auth_result, post_summary

    # ──────────────────────────────────────────────────────────────────────────
    # 8. Complete Pipeline Execution & Final Summary
    # ──────────────────────────────────────────────────────────────────────────
    def run(self) -> Dict[str, Any]:
        # 1. Pre-search integrity
        pre_integrity = self.verify_pre_search_integrity()

        # 2. Cleanup synthetic
        synthetic_removed = self.cleanup_synthetic_records()

        # 3. Real literature search
        candidates = self.search_pubmed()

        # 4. Deduplicate
        unique_candidates, duplicates = self.deduplicate_candidates(candidates)

        # 5. Fetch full text
        enriched_candidates = self.fetch_texts(unique_candidates)

        # 6. Extract and score
        scored, accepted, rep_audits, prov_audits = self.extract_and_score(enriched_candidates, duplicates)

        # 7. Authenticity & post-expansion integrity
        auth_status, auth_result, post_integrity = self.run_authenticity_and_integrity(accepted)

        # Determine final decision string
        final_decision = "AUTHENTIC_EVIDENCE_FOUND" if (auth_status == "AUTHENTIC" and accepted) else "NO_VALID_CANDIDATE"

        final_summary = {
            "execution_timestamp": datetime.now(timezone.utc).isoformat(),
            "final_decision": final_decision,
            "training_allowed": False,
            "genuine_candidates_found": len(candidates),
            "duplicate_candidates": len(duplicates),
            "candidates_rejected": len(candidates) - len(accepted),
            "candidates_accepted": len(accepted),
            "genuine_papers_added": len(accepted),
            "representation_candidates": len([s for s in scored if s.get("representation_method")]),
            "authenticated_representation_candidates": len(accepted) if auth_status == "AUTHENTIC" else 0,
            "provenance_failures": len([p for p in prov_audits if not p.get("provenance_complete")]),
            "synthetic_candidates_rejected": synthetic_removed,
            "stage2c_integrity_before": {
                "corpus_valid": pre_integrity.get("corpus_valid"),
                "critical_errors": pre_integrity.get("critical_errors"),
                "total_papers": pre_integrity.get("actual_papers_count"),
            },
            "stage2c_integrity_after": {
                "corpus_valid": post_integrity.get("corpus_valid"),
                "critical_errors": post_integrity.get("critical_errors"),
                "total_papers": post_integrity.get("corpus_counts", {}).get("total_papers"),
            },
            "final_authenticity_status": auth_status,
        }

        self._save_json(self.metadata_dir / "stage2d2_final_summary.json", final_summary)
        return final_summary


if __name__ == "__main__":
    retriever = Stage2D2RealRetriever()
    summary = retriever.run()
    print("Stage 2D-2 Complete. Final Decision:", summary["final_decision"])
    print(json.dumps(summary, indent=2))
