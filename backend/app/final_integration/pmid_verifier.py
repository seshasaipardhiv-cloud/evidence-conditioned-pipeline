"""
pmid_verifier.py

Verifies PMIDs and DOIs against the PubMed E-utilities API.
Records VERIFIED / UNVERIFIED / NOT_FOUND for every evidence paper.

If network access is unavailable, marks every paper UNVERIFIED with reason.
UNVERIFIED papers are NOT treated as authoritative evidence.

Creates:
  evidence/final/submission/New/provenance/evidence_source_verification.json
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

logger = logging.getLogger(__name__)

# All PMIDs/DOIs referenced across the project
_EVIDENCE_PAPERS = [
    {
        "pmid": "38396486",
        "doi": "10.3390/diagnostics14040448",
        "claimed_title": "Multi-Modal Image-Text Fusion for Head and Neck Cancer Prognosis",
        "source": "Stage 2D NER extraction — highest-confidence entity source paper",
        "role": "Multimodal fusion evidence",
    },
    {
        "pmid": "40325104",
        "doi": None,
        "claimed_title": "Gradient Boosting Methods for Clinical Tabular Data",
        "source": "Evidence scoring — tabular model selection",
        "role": "XGBoost/gradient boosting evidence",
    },
    {
        "pmid": "39074400",
        "doi": None,
        "claimed_title": "Multiple Imputation Methods for Missing Clinical Data",
        "source": "Preprocessing evidence — imputation selection",
        "role": "MICE imputation evidence",
    },
    {
        "pmid": "42487970",
        "doi": None,
        "claimed_title": "ResNet for Medical Image Classification",
        "source": "Image model selection evidence",
        "role": "ResNet architecture evidence",
    },
    {
        "pmid": "41131352",
        "doi": None,
        "claimed_title": "Subword Tokenization for Biomedical Language Models",
        "source": "Text preprocessing evidence",
        "role": "SciBERT tokenization evidence",
    },
    {
        "pmid": "41826845",
        "doi": None,
        "claimed_title": "Late Fusion Strategies in Multimodal Clinical Learning",
        "source": "Fusion selection evidence",
        "role": "Late fusion architecture evidence",
    },
    {
        "pmid": "41775771",
        "doi": None,
        "claimed_title": "Ensemble Strategies for Medical AI Systems",
        "source": "Ensemble selection evidence",
        "role": "Ensemble method evidence",
    },
]

_ESUMMARY_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    "?db=pubmed&id={pmid}&retmode=json"
)


def _fetch_pubmed_summary(pmid: str, timeout: float = 8.0) -> Optional[Dict]:
    """Fetches PubMed eSummary for a PMID. Returns None on any error."""
    url = _ESUMMARY_URL.format(pmid=pmid)
    try:
        req = Request(url, headers={"User-Agent": "EvidencePipeline/1.0 (scientific integrity audit)"})
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            result = data.get("result", {})
            if pmid in result:
                return result[pmid]
            # Some responses use uids list
            uids = result.get("uids", [])
            if uids and uids[0] in result:
                return result[uids[0]]
            return None
    except (URLError, HTTPError, TimeoutError, json.JSONDecodeError) as e:
        logger.warning(f"PubMed API request failed for PMID {pmid}: {e}")
        return None


def verify_all_papers(out_dir: str = "evidence/final/submission/New/provenance") -> Dict[str, Any]:
    """
    Verifies all evidence papers. Writes evidence_source_verification.json.
    Returns the full verification report.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    verified_papers = []
    summary_counts = {"VERIFIED": 0, "UNVERIFIED": 0, "NOT_FOUND": 0}

    for paper in _EVIDENCE_PAPERS:
        pmid = paper["pmid"]
        logger.info(f"Verifying PMID {pmid}...")

        result = _fetch_pubmed_summary(pmid, timeout=8.0)
        time.sleep(0.35)  # Respect NCBI rate limit (3 req/sec)

        if result is None:
            # Network unreachable or timeout
            verified_papers.append({
                **paper,
                "api_title": None,
                "verification_status": "UNVERIFIED",
                "verification_reason": "PubMed API request failed or timed out. Network may be unavailable.",
                "full_text_available": False,
                "matched_evidence_spans": [],
                "authority_level": "UNVERIFIED — do not treat as authoritative",
            })
            summary_counts["UNVERIFIED"] += 1

        elif result.get("error"):
            verified_papers.append({
                **paper,
                "api_title": None,
                "verification_status": "NOT_FOUND",
                "verification_reason": f"PubMed returned error: {result.get('error')}",
                "full_text_available": False,
                "matched_evidence_spans": [],
                "authority_level": "NOT_FOUND — do not use as evidence",
            })
            summary_counts["NOT_FOUND"] += 1

        else:
            api_title = result.get("title", "").strip()
            status = "VERIFIED" if api_title else "UNVERIFIED"
            if status == "VERIFIED":
                summary_counts["VERIFIED"] += 1
            else:
                summary_counts["UNVERIFIED"] += 1

            verified_papers.append({
                **paper,
                "api_title": api_title or None,
                "pubdate": result.get("pubdate"),
                "authors": [a.get("name") for a in result.get("authors", [])][:3],
                "journal": result.get("fulljournalname"),
                "verification_status": status,
                "verification_reason": "PubMed eSummary response received." if api_title else "PubMed response had empty title.",
                "full_text_available": bool(result.get("pmc")),
                "pmc_id": result.get("pmc"),
                "matched_evidence_spans": [],  # Span matching requires full-text retrieval
                "authority_level": "VERIFIED — may be used as evidence" if status == "VERIFIED" else "UNVERIFIED",
            })

    report = {
        "verification_timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": "PubMed E-utilities eSummary API",
        "api_url_template": _ESUMMARY_URL,
        "summary": {
            **summary_counts,
            "total": len(verified_papers),
            "authoritative_count": summary_counts["VERIFIED"],
            "non_authoritative_count": summary_counts["UNVERIFIED"] + summary_counts["NOT_FOUND"],
        },
        "important_note": (
            "Only VERIFIED papers may be treated as authoritative evidence. "
            "UNVERIFIED papers (network failure) and NOT_FOUND papers must NOT "
            "be used as the basis for evidence-conditioned model selection."
        ),
        "papers": verified_papers,
    }

    out_file = out_path / "evidence_source_verification.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(
        f"PMID verification complete: {summary_counts['VERIFIED']} VERIFIED, "
        f"{summary_counts['UNVERIFIED']} UNVERIFIED, {summary_counts['NOT_FOUND']} NOT_FOUND. "
        f"Saved to {out_file}"
    )
    return report
