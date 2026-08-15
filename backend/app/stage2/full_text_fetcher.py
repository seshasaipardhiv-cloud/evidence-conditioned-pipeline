"""
full_text_fetcher.py

Attempts to retrieve legally accessible full text for seed papers.

Priority:
  1. PubMed Central (PMC) via NCBI E-utilities — completely free and legal
  2. Unpaywall API (legal open-access resolver, no auth)
  3. Direct publisher URL for known open-access publishers (MDPI, BioMed Central)

Rules enforced:
  - No paywall bypass.
  - No pirated sources (Sci-Hub etc.).
  - If access fails: full_text_access_status = not_accessible or not_found.
  - No HANCOCK patient data is ever sent to any API.
  - All requests use a polite User-Agent with contact info.
"""

import hashlib
import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Optional, Tuple

from backend.app.stage2.models import FullTextAccessStatus, PaperRecord

logger = logging.getLogger(__name__)

_USER_AGENT = "EvidencePipelineBot/1.0 (mailto:research@example.com)"
_REQUEST_DELAY_S = 1.0   # politeness delay between HTTP requests


def _http_get(url: str, timeout: int = 20) -> Optional[bytes]:
    """Safe HTTP GET. Returns raw bytes or None on failure."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as exc:
        logger.debug(f"HTTP GET failed for {url}: {exc}")
        return None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# PMC lookup via E-utilities
# ──────────────────────────────────────────────────────────────────────────────

def _pmc_id_from_pmid(pmid: str) -> Optional[str]:
    """Convert PMID to PMC ID using NCBI ID converter."""
    url = (
        "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
        f"?ids={pmid}&format=json&email=research@example.com"
    )
    raw = _http_get(url)
    if not raw:
        return None
    try:
        data = json.loads(raw)
        records = data.get("records", [])
        if records:
            return records[0].get("pmcid")
    except Exception:
        pass
    return None


def _pmc_id_from_doi(doi: str) -> Optional[str]:
    """Search PMC by DOI using esearch."""
    encoded = urllib.parse.quote(doi)
    url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        f"?db=pmc&term={encoded}[doi]&retmode=json"
    )
    raw = _http_get(url)
    if not raw:
        return None
    try:
        data = json.loads(raw)
        ids = data.get("esearchresult", {}).get("idlist", [])
        if ids:
            return f"PMC{ids[0]}"
    except Exception:
        pass
    return None


def _fetch_pmc_fulltext(pmc_id: str) -> Optional[Tuple[str, str]]:
    """
    Fetch full text XML from PMC via efetch.
    Returns (text_content, license_text) or None.
    """
    clean_id = pmc_id.replace("PMC", "")
    url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=pmc&id={clean_id}&retmode=xml"
    )
    raw = _http_get(url)
    if not raw:
        return None

    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        logger.debug(f"XML parse error for PMC {pmc_id}: {exc}")
        return None

    # Extract all text content from body elements
    body_texts = []
    license_text = None

    for elem in root.iter():
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if tag in ("body", "p", "sec", "title"):
            if elem.text and elem.text.strip():
                body_texts.append(elem.text.strip())
        if tag == "license":
            license_text = " ".join(t for t in elem.itertext() if t.strip())

    if not body_texts:
        return None

    full_text = "\n".join(body_texts)
    return full_text, license_text or "Unknown"


# ──────────────────────────────────────────────────────────────────────────────
# Unpaywall API
# ──────────────────────────────────────────────────────────────────────────────

def _unpaywall_lookup(doi: str) -> Optional[dict]:
    """
    Query Unpaywall for open-access metadata.
    Returns dict with 'oa_url', 'license', 'is_oa' or None.
    """
    encoded = urllib.parse.quote(doi, safe="")
    url = f"https://api.unpaywall.org/v2/{encoded}?email=research@example.com"
    raw = _http_get(url)
    if not raw:
        return None
    try:
        data = json.loads(raw)
        if not data.get("is_oa"):
            return {"is_oa": False}
        best = data.get("best_oa_location") or {}
        return {
            "is_oa": True,
            "oa_url": best.get("url_for_pdf") or best.get("url"),
            "license": best.get("license"),
            "host_type": best.get("host_type"),
        }
    except Exception:
        return None


def _fetch_url_as_text(url: str) -> Optional[str]:
    """Fetch a URL and return decoded text (HTML/plain text)."""
    raw = _http_get(url)
    if not raw:
        return None
    # Try UTF-8 decode
    for enc in ("utf-8", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

class FullTextFetcher:
    """
    Attempts to fetch full text for a PaperRecord using legal open-access sources only.
    Mutates the paper in place with full_text_* fields.
    Returns (full_text_str, updated_paper) or (None, updated_paper).
    """

    def fetch(self, paper: PaperRecord) -> Tuple[Optional[str], PaperRecord]:
        retrieved_at = datetime.now(timezone.utc).isoformat()

        # ── Step 1: Try PMC via PMID or DOI ──────────────────────────────────
        pmc_id: Optional[str] = None

        if paper.pmid:
            time.sleep(_REQUEST_DELAY_S)
            pmc_id = _pmc_id_from_pmid(paper.pmid)

        if not pmc_id and paper.doi:
            time.sleep(_REQUEST_DELAY_S)
            pmc_id = _pmc_id_from_doi(paper.doi)

        if pmc_id:
            time.sleep(_REQUEST_DELAY_S)
            result = _fetch_pmc_fulltext(pmc_id)
            if result:
                full_text, license_text = result
                sha = _sha256(full_text.encode("utf-8"))
                url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/"
                paper = paper.model_copy(update={
                    "pmc_id": pmc_id,
                    "full_text_available": True,
                    "full_text_source": "PMC",
                    "full_text_url": url,
                    "full_text_retrieved_at": retrieved_at,
                    "full_text_sha256": sha,
                    "full_text_license": license_text,
                    "full_text_access_status": FullTextAccessStatus.accessible,
                })
                logger.info(f"PMC full text obtained for {paper.paper_id} ({pmc_id})")
                return full_text, paper

        # ── Step 2: Try Unpaywall ─────────────────────────────────────────────
        if paper.doi:
            time.sleep(_REQUEST_DELAY_S)
            oa_info = _unpaywall_lookup(paper.doi)

            if oa_info and oa_info.get("is_oa") and oa_info.get("oa_url"):
                oa_url = oa_info["oa_url"]
                time.sleep(_REQUEST_DELAY_S)
                text = _fetch_url_as_text(oa_url)
                if text and len(text) > 2000:
                    sha = _sha256(text.encode("utf-8"))
                    paper = paper.model_copy(update={
                        "full_text_available": True,
                        "full_text_source": oa_info.get("host_type", "open_access"),
                        "full_text_url": oa_url,
                        "full_text_retrieved_at": retrieved_at,
                        "full_text_sha256": sha,
                        "full_text_license": oa_info.get("license"),
                        "full_text_access_status": FullTextAccessStatus.accessible,
                    })
                    logger.info(f"Unpaywall full text for {paper.paper_id}: {oa_url}")
                    return text, paper

            elif oa_info and not oa_info.get("is_oa"):
                logger.info(f"Unpaywall confirms not open access: {paper.doi}")
                paper = paper.model_copy(update={
                    "full_text_access_status": FullTextAccessStatus.not_accessible,
                })
                return None, paper

        # ── Step 3: Not found ─────────────────────────────────────────────────
        if paper.full_text_access_status == FullTextAccessStatus.not_found:
            # If we have an abstract at least record abstract_only
            if paper.abstract_available:
                paper = paper.model_copy(update={
                    "full_text_access_status": FullTextAccessStatus.abstract_only,
                })
            else:
                paper = paper.model_copy(update={
                    "full_text_access_status": FullTextAccessStatus.not_found,
                })

        logger.info(
            f"Full text not obtained for {paper.paper_id} — "
            f"status: {paper.full_text_access_status.value}"
        )
        return None, paper
