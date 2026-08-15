"""
Stage 2A + 2B Orchestrator

Stage 2A flow (unchanged):
  EvidenceAcquisition → DocumentParser → EvidenceValidator → GraphBuilder → Outputs

Stage 2B additions:
  FullTextFetcher (per paper) → SectionParser → ExperimentExtractor →
  ContradictionDetector → Extended GraphBuilder → Extended Outputs

HANCOCK patient data is never sent to external APIs.
"""

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from backend.app.stage2.acquisition import EvidenceAcquisition
from backend.app.stage2.contradiction_detector import ContradictionDetector
from backend.app.stage2.document_parser import DocumentParser
from backend.app.stage2.evidence_validator import EvidenceValidator
from backend.app.stage2.experiment_extractor import ExperimentExtractor
from backend.app.stage2.full_text_fetcher import FullTextFetcher
from backend.app.stage2.graph_builder import GraphBuilder
from backend.app.stage2.models import (
    AblationRecord, ContradictionCandidate, EvidenceClaim,
    EvidenceStatus, ExperimentRecord, PaperRecord,
    SearchMetadata, SourceScope,
)
from backend.app.stage2.section_parser import SectionParser

logger = logging.getLogger(__name__)

_POLITE_DELAY_S = 1.5   # seconds between paper fetches


class Stage2Orchestrator:

    def __init__(self):
        self.acquisition = EvidenceAcquisition()
        self.parser = DocumentParser()
        self.validator = EvidenceValidator()
        self.graph_builder = GraphBuilder()
        # Stage 2B components
        self.fetcher = FullTextFetcher()
        self.section_parser = SectionParser()
        self.exp_extractor = ExperimentExtractor()
        self.contra_detector = ContradictionDetector()

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 2A (unchanged external contract)
    # ─────────────────────────────────────────────────────────────────────────

    def run_stage2a(self) -> dict:
        logger.info("Starting Stage 2A Orchestration")

        papers, total_seed_count, failed_retrievals = self.acquisition.fetch_seed_papers()
        unique_papers_count = len(papers)
        duplicate_count = max(0, total_seed_count - unique_papers_count - failed_retrievals)

        logger.info(
            f"Seeds: {total_seed_count}, resolved: {unique_papers_count}, "
            f"duplicates: {duplicate_count}, failed: {failed_retrievals}"
        )

        all_candidate_claims: List[EvidenceClaim] = []
        all_mechanisms: Dict = {}

        for paper in papers:
            candidates = self.parser.parse_paper(paper)
            for claim, mechanism in candidates:
                all_candidate_claims.append(claim)
                if mechanism.mechanism_id not in all_mechanisms:
                    all_mechanisms[mechanism.mechanism_id] = mechanism
                elif mechanism.mapping_status == "MAPPED":
                    all_mechanisms[mechanism.mechanism_id] = mechanism

        validated_claims = self.validator.validate_claims(all_candidate_claims)
        rejected_count = len(all_candidate_claims) - len(validated_claims)

        graph = self.graph_builder.build_graph(papers, validated_claims, list(all_mechanisms.values()))

        out_dir = Path("evidence/processed")
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "papers.jsonl", "w", encoding="utf-8") as f:
            for p in papers:
                f.write(p.model_dump_json() + "\n")
        with open(out_dir / "evidence_claims.jsonl", "w", encoding="utf-8") as f:
            for c in validated_claims:
                f.write(c.model_dump_json() + "\n")
        with open(out_dir / "mechanisms.jsonl", "w", encoding="utf-8") as f:
            for m in all_mechanisms.values():
                f.write(m.model_dump_json() + "\n")
        with open(out_dir / "evidence_graph.json", "w", encoding="utf-8") as f:
            f.write(graph.model_dump_json(indent=2))

        # Stats
        papers_with_abstract = sum(1 for p in papers if p.abstract_available)
        papers_without_abstract = unique_papers_count - papers_with_abstract
        claims_by_status: dict = defaultdict(int)
        claims_by_scope: dict = defaultdict(int)
        for c in validated_claims:
            claims_by_status[c.evidence_status.value] += 1
            claims_by_scope[c.source_scope.value] += 1

        coverage = self._build_coverage(papers, validated_claims)
        meta_dir = Path("evidence/metadata")
        meta_dir.mkdir(parents=True, exist_ok=True)
        with open(meta_dir / "paper_evidence_coverage.json", "w", encoding="utf-8") as f:
            json.dump(coverage, f, indent=2)

        search_metadata = SearchMetadata(
            search_query="seed_papers_manual_curation",
            source="PubMed / manual",
            retrieval_date=datetime.now().isoformat(),
            filters="multimodal_cancer",
            year_range="2023-2025",
            domain="multimodal_oncology",
            task="survival_prediction, classification",
            modality="clinical, imaging, text, omics",
            candidates_returned=total_seed_count,
            papers_selected=unique_papers_count,
        )

        negative_findings = sum(
            1 for c in validated_claims if c.result and c.result.direction == "degradation"
        )
        contradiction_candidates = sum(1 for c in validated_claims if c.contradiction_candidate)

        report = {
            "stage": "2A",
            "search_metadata": search_metadata.model_dump(),
            "number_of_papers_in_seed_corpus": total_seed_count,
            "number_of_papers_successfully_resolved": unique_papers_count,
            "papers_with_full_text": sum(1 for p in papers if p.full_text_available),
            "papers_with_abstract_only": sum(1 for p in papers if p.abstract_available and not p.full_text_available),
            "papers_with_no_abstract": papers_without_abstract,
            "duplicate_count": duplicate_count,
            "failed_paper_retrievals": failed_retrievals,
            "papers_without_evidence": sum(1 for p in papers if not any(c.paper_id == p.paper_id for c in validated_claims)),
            "candidate_claims_generated": len(all_candidate_claims),
            "claims_rejected_by_validator": rejected_count,
            "evidence_claims_extracted": len(validated_claims),
            "claims_by_evidence_status": dict(claims_by_status),
            "direct_empirical_claims": claims_by_status.get(EvidenceStatus.direct_empirical.value, 0),
            "qualitative_claims": claims_by_status.get(EvidenceStatus.qualitative.value, 0),
            "methodological_claims": claims_by_status.get(EvidenceStatus.methodological.value, 0),
            "claims_by_source_scope": dict(claims_by_scope),
            "negative_findings": negative_findings,
            "negative_evidence_coverage": "not_established" if negative_findings == 0 else "present",
            "contradiction_candidates": contradiction_candidates,
            "contradiction_coverage": "not_established" if contradiction_candidates == 0 else "flagged",
            "total_mechanisms_mapped": sum(1 for m in all_mechanisms.values() if m.mapping_status == "MAPPED"),
            "unmapped_mechanisms": sum(1 for m in all_mechanisms.values() if m.mapping_status == "UNMAPPED"),
        }

        with open(meta_dir / "evidence_ingestion_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Stage 2A complete. Papers: {unique_papers_count}, Claims: {len(validated_claims)}")
        return report

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 2B — Full-text acquisition + deep extraction
    # ─────────────────────────────────────────────────────────────────────────

    def run_stage2b(self) -> dict:
        """
        Attempt full-text retrieval for all papers, then run deep extraction.
        Updates all output files in evidence/processed/ and evidence/metadata/.
        """
        logger.info("Starting Stage 2B Orchestration")

        # ── 1. Load papers from Stage 2A output ──────────────────────────────
        papers_path = Path("evidence/processed/papers.jsonl")
        if not papers_path.exists():
            raise RuntimeError("Run Stage 2A before Stage 2B.")

        papers: List[PaperRecord] = []
        with open(papers_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    papers.append(PaperRecord.model_validate_json(line))

        # ── 2. Attempt full-text retrieval ────────────────────────────────────
        full_texts: Dict[str, Optional[str]] = {}
        enriched_papers: List[PaperRecord] = []

        for paper in papers:
            logger.info(f"Attempting full text for: {paper.paper_id}")
            try:
                ft_text, updated_paper = self.fetcher.fetch(paper)
                full_texts[paper.paper_id] = ft_text
                enriched_papers.append(updated_paper)
            except Exception as exc:
                logger.warning(f"Fetch error for {paper.paper_id}: {exc}")
                full_texts[paper.paper_id] = None
                enriched_papers.append(paper)
            time.sleep(_POLITE_DELAY_S)

        # ── 3. Load Stage 2A claims and mechanisms ────────────────────────────
        stage2a_claims: List[EvidenceClaim] = []
        with open(Path("evidence/processed/evidence_claims.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    stage2a_claims.append(EvidenceClaim.model_validate_json(line))

        all_mechanisms: Dict = {}
        with open(Path("evidence/processed/mechanisms.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    from backend.app.stage2.models import Mechanism
                    m = Mechanism.model_validate_json(line)
                    all_mechanisms[m.mechanism_id] = m

        # ── 4. Deep extraction from full text ─────────────────────────────────
        all_experiments: List[ExperimentRecord] = []
        all_ablations: List[AblationRecord] = []
        all_new_claims: List[EvidenceClaim] = []

        for paper in enriched_papers:
            ft = full_texts.get(paper.paper_id)

            # Always extract from abstract (Stage 2A claims cover this)
            # For full text: extract experiments and ablations
            if ft and paper.full_text_available:
                sections = self.section_parser.parse(ft)
                source_scope = SourceScope.full_text
            elif paper.abstract and paper.abstract.strip():
                # Fall back to abstract-level extraction
                sections = self.section_parser.parse(paper.abstract)
                source_scope = SourceScope.abstract
            else:
                continue

            exps, abls = self.exp_extractor.extract(
                paper_id=paper.paper_id,
                text=ft if ft else paper.abstract,
                source_scope=source_scope,
                sections=sections,
            )
            all_experiments.extend(exps)
            all_ablations.extend(abls)

        # ── 5. Upgrade Stage 2A claims where full text is now available ───────
        # If a paper now has full text, update source_scope of its claims
        ft_paper_ids = {p.paper_id for p in enriched_papers if p.full_text_available}
        upgraded_claims = []
        for c in stage2a_claims:
            if c.paper_id in ft_paper_ids and c.source_scope == SourceScope.abstract:
                # Upgrade: the paper now has full text but this claim was from abstract
                # Keep source_scope=abstract (accurate) — the claim text came from abstract
                pass
            upgraded_claims.append(c)

        # ── 6. Validate new claims (re-run) ───────────────────────────────────
        validated_claims = self.validator.validate_claims(upgraded_claims)

        # ── 7. Contradiction detection ────────────────────────────────────────
        contradictions = self.contra_detector.detect(validated_claims)

        # ── 8. Build extended graph ───────────────────────────────────────────
        graph = self.graph_builder.build_graph(
            enriched_papers,
            validated_claims,
            list(all_mechanisms.values()),
            experiments=all_experiments,
            ablations=all_ablations,
        )

        # ── 9. Persist all outputs ────────────────────────────────────────────
        out_dir = Path("evidence/processed")

        with open(out_dir / "papers.jsonl", "w", encoding="utf-8") as f:
            for p in enriched_papers:
                f.write(p.model_dump_json() + "\n")

        with open(out_dir / "evidence_claims.jsonl", "w", encoding="utf-8") as f:
            for c in validated_claims:
                f.write(c.model_dump_json() + "\n")

        with open(out_dir / "mechanisms.jsonl", "w", encoding="utf-8") as f:
            for m in all_mechanisms.values():
                f.write(m.model_dump_json() + "\n")

        with open(out_dir / "experiments.jsonl", "w", encoding="utf-8") as f:
            for e in all_experiments:
                f.write(e.model_dump_json() + "\n")

        with open(out_dir / "ablations.jsonl", "w", encoding="utf-8") as f:
            for a in all_ablations:
                f.write(a.model_dump_json() + "\n")

        with open(out_dir / "contradiction_candidates.jsonl", "w", encoding="utf-8") as f:
            for cc in contradictions:
                f.write(cc.model_dump_json() + "\n")

        with open(out_dir / "evidence_graph.json", "w", encoding="utf-8") as f:
            f.write(graph.model_dump_json(indent=2))

        # ── 10. Generate metadata outputs ─────────────────────────────────────
        meta_dir = Path("evidence/metadata")
        meta_dir.mkdir(parents=True, exist_ok=True)

        coverage = self._build_coverage_2b(enriched_papers, validated_claims, all_experiments, all_ablations)
        with open(meta_dir / "paper_evidence_coverage.json", "w", encoding="utf-8") as f:
            json.dump(coverage, f, indent=2)

        # Compute summary stats
        papers_with_ft = sum(1 for p in enriched_papers if p.full_text_available)
        papers_abstract_only = sum(1 for p in enriched_papers if p.abstract_available and not p.full_text_available)
        papers_no_evidence = sum(1 for p in enriched_papers if not any(c.paper_id == p.paper_id for c in validated_claims))
        direct_empirical = sum(1 for c in validated_claims if c.evidence_status == EvidenceStatus.direct_empirical)
        qualitative = sum(1 for c in validated_claims if c.evidence_status == EvidenceStatus.qualitative)
        negative_findings = sum(1 for c in validated_claims if c.result and c.result.direction == "degradation")
        # Also count from experiment results
        negative_findings += sum(
            1 for e in all_experiments
            for r in e.reported_results
            if r.direction == "degradation"
        )
        unmapped = sum(1 for m in all_mechanisms.values() if m.mapping_status == "UNMAPPED")

        report = {
            "stage": "2B",
            "retrieval_date": datetime.now().isoformat(),
            "papers_with_full_text": papers_with_ft,
            "papers_still_abstract_only": papers_abstract_only,
            "papers_with_no_evidence": papers_no_evidence,
            "experiments_extracted": len(all_experiments),
            "ablations_extracted": len(all_ablations),
            "evidence_claims_total": len(validated_claims),
            "direct_empirical_claims": direct_empirical,
            "qualitative_claims": qualitative,
            "negative_findings": negative_findings,
            "negative_evidence_coverage": "not_established" if negative_findings == 0 else "present",
            "contradiction_candidates": len(contradictions),
            "contradiction_coverage": "not_established" if not contradictions else "flagged",
            "unmapped_mechanisms": unmapped,
            "claims_rejected_as_unsupported": 0,  # validator keeps all well-formed claims
        }

        with open(meta_dir / "stage2b_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Stage 2B complete. Full text: {papers_with_ft}, Experiments: {len(all_experiments)}")
        return report

    # ─────────────────────────────────────────────────────────────────────────
    # Coverage helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_coverage(
        self, papers: List[PaperRecord], claims: List[EvidenceClaim]
    ) -> list:
        claims_by_paper: dict = defaultdict(list)
        for c in claims:
            claims_by_paper[c.paper_id].append(c)

        coverage = []
        for p in papers:
            paper_claims = claims_by_paper[p.paper_id]
            evidence_coverage = (
                "full" if p.full_text_available else
                "abstract_only" if p.abstract_available else
                "metadata_only" if p.title else
                "none"
            )
            scope = (
                paper_claims[0].source_scope.value if paper_claims else
                SourceScope.abstract.value if p.abstract_available else
                SourceScope.metadata_only.value if p.title else
                SourceScope.none.value
            )
            coverage.append({
                "paper_id": p.paper_id,
                "title": p.title,
                "doi": p.doi,
                "pmid": p.pmid,
                "abstract_available": p.abstract_available,
                "full_text_available": p.full_text_available,
                "claims_extracted": len(paper_claims),
                "claim_ids": [c.evidence_id for c in paper_claims],
                "source_scope": scope,
                "evidence_coverage": evidence_coverage,
                "evidence_statuses": list({c.evidence_status.value for c in paper_claims}),
            })
        return coverage

    def _build_coverage_2b(
        self,
        papers: List[PaperRecord],
        claims: List[EvidenceClaim],
        experiments: List[ExperimentRecord],
        ablations: List[AblationRecord],
    ) -> list:
        claims_by_paper: dict = defaultdict(list)
        exps_by_paper: dict = defaultdict(list)
        abls_by_paper: dict = defaultdict(list)
        for c in claims:
            claims_by_paper[c.paper_id].append(c)
        for e in experiments:
            exps_by_paper[e.paper_id].append(e)
        for a in ablations:
            abls_by_paper[a.paper_id].append(a)

        coverage = []
        for p in papers:
            paper_claims = claims_by_paper[p.paper_id]
            paper_exps = exps_by_paper[p.paper_id]
            paper_abls = abls_by_paper[p.paper_id]

            evidence_coverage = (
                "full" if p.full_text_available else
                "abstract_only" if p.abstract_available else
                "metadata_only" if p.title else
                "none"
            )
            source_scopes = list({c.source_scope.value for c in paper_claims})
            if not source_scopes:
                source_scopes = [SourceScope.abstract.value if p.abstract_available else SourceScope.none.value]

            coverage.append({
                "paper_id": p.paper_id,
                "title": p.title,
                "doi": p.doi,
                "pmid": p.pmid,
                "abstract_available": p.abstract_available,
                "full_text_available": p.full_text_available,
                "full_text_source": p.full_text_source,
                "full_text_access_status": p.full_text_access_status.value,
                "full_text_license": p.full_text_license,
                "claims_extracted": len(paper_claims),
                "experiments_extracted": len(paper_exps),
                "ablations_extracted": len(paper_abls),
                "claim_ids": [c.evidence_id for c in paper_claims],
                "source_scopes": source_scopes,
                "evidence_statuses": list({c.evidence_status.value for c in paper_claims}),
                "evidence_coverage": evidence_coverage,
            })
        return coverage


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    orchestrator = Stage2Orchestrator()

    stage = "2b" if len(sys.argv) < 2 else sys.argv[1].lower()
    if stage in ("2a", "stage2a"):
        report = orchestrator.run_stage2a()
    else:
        report = orchestrator.run_stage2b()

    print(json.dumps(report, indent=2))
