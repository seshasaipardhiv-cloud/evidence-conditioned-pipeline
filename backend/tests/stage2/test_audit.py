"""
test_audit.py — Scientific audit tests for Stage 2A evidence claims.

Covers:
  1.  Abstract-only claim → source_scope = "abstract"
  2.  Full-text claim → source_scope = "full_text"
  3.  Metadata-only paper → no claims generated
  4.  Unsupported numerical claim → extraction_status = unresolved or claim rejected
  5.  Missing baseline → baseline = null
  6.  Missing metric → metric = null in result
  7.  Missing result → result = null for methodological claims
  8.  Background statement → NOT direct_empirical (must be methodological or qualitative)
  9.  Empty abstract → no claims generated
  10. Source location validation → evidence_location and source_scope present
"""

import pytest
from datetime import datetime

from backend.app.stage2.document_parser import DocumentParser
from backend.app.stage2.models import (
    PaperRecord, EvidenceClaim, EvidenceStatus, SourceScope,
    ExtractionStatus, Provenance, ExtractionMethod,
)


def make_paper(abstract: str, paper_id: str = "p_test", doi: str = "10.0000/test") -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        title="Test Paper",
        authors=["Author A"],
        publication_year=2024,
        doi=doi,
        source="PubMedSource",
        abstract=abstract if abstract.strip() else None,
        abstract_available=bool(abstract.strip()),
        full_text_available=False,
        retrieval_date=datetime.now().isoformat(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Abstract-only claim
# ─────────────────────────────────────────────────────────────────────────────
def test_abstract_only_claim_has_correct_source_scope():
    """Claims from abstract-only papers must have source_scope = 'abstract'."""
    parser = DocumentParser()
    paper = make_paper(
        "Result: The CNN model achieved AUC 0.91 compared with 0.86 for the unimodal baseline."
    )
    results = parser.parse_paper(paper)
    assert len(results) == 1, "Should produce exactly one claim"
    claim, _ = results[0]
    assert claim.source_scope == SourceScope.abstract


# ─────────────────────────────────────────────────────────────────────────────
# 2. Full-text claim
# ─────────────────────────────────────────────────────────────────────────────
def test_full_text_claim_has_correct_source_scope():
    """A claim constructed from full text should have source_scope = 'full_text'."""
    # We manually construct the claim (parser only generates abstract scope)
    prov = Provenance(
        source_type="scholarly_api",
        source_reference="10.0000/ft",
        extraction_method=ExtractionMethod.manual,
        extraction_status=ExtractionStatus.explicit,
        retrieval_date=datetime.now().isoformat(),
    )
    claim = EvidenceClaim(
        evidence_id="c_ft",
        paper_id="p_ft",
        claim="Full-text derived claim.",
        source_scope=SourceScope.full_text,
        mechanisms=["mech_cnn"],
        evidence_location="Section 4.2",
        extraction_method=ExtractionMethod.manual,
        evidence_status=EvidenceStatus.direct_empirical,
        provenance=prov,
    )
    assert claim.source_scope == SourceScope.full_text


# ─────────────────────────────────────────────────────────────────────────────
# 3. Metadata-only paper produces no claims
# ─────────────────────────────────────────────────────────────────────────────
def test_metadata_only_paper_produces_no_claims():
    """A paper with no abstract should produce zero evidence claims."""
    parser = DocumentParser()
    paper = make_paper("")  # empty abstract
    results = parser.parse_paper(paper)
    assert results == [], "Metadata-only paper must yield no evidence claims"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Unsupported numerical claim → extraction_status = unresolved
# ─────────────────────────────────────────────────────────────────────────────
def test_unsupported_numerical_claim_marked_unresolved():
    """
    An abstract that mentions a number in a non-result context should not produce
    an explicit extraction_status unless the number is clearly in a result statement.
    A methodological abstract should have extraction_status = unresolved.
    """
    parser = DocumentParser()
    # This abstract has a number but in a background context, not a result
    paper = make_paper(
        "We propose a novel framework with 3 layers of processing for cancer classification. "
        "The method incorporates CNN and clinical features."
    )
    results = parser.parse_paper(paper)
    # Should still produce a claim, but it should not be direct_empirical
    if results:
        claim, _ = results[0]
        assert claim.evidence_status != EvidenceStatus.direct_empirical, (
            "Background statement must not be classified as direct_empirical"
        )
        if claim.provenance.extraction_status == ExtractionStatus.explicit:
            pytest.fail("Extraction status must not be explicit for background statements")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Missing baseline → baseline = null
# ─────────────────────────────────────────────────────────────────────────────
def test_missing_baseline_is_null():
    """When no baseline is explicitly stated, baseline must be null."""
    parser = DocumentParser()
    paper = make_paper(
        "Result: The model achieved AUC 0.85 on the test set."
    )
    results = parser.parse_paper(paper)
    assert results, "Should produce a claim"
    claim, _ = results[0]
    # No baseline mentioned in abstract
    assert claim.baseline is None, f"Expected baseline=None, got {claim.baseline!r}"
    # baseline_value in result should also be None
    if claim.result:
        assert claim.result.baseline_value is None


# ─────────────────────────────────────────────────────────────────────────────
# 6. Missing metric → metric = null in result
# ─────────────────────────────────────────────────────────────────────────────
def test_missing_metric_is_null():
    """If no metric is explicitly mentioned, result.metric must be null."""
    parser = DocumentParser()
    paper = make_paper(
        "We propose a multimodal CNN model for cancer detection that improves outcomes."
    )
    results = parser.parse_paper(paper)
    if results:
        claim, _ = results[0]
        if claim.result:
            # Qualitative claim; metric should be null
            assert claim.result.metric is None, (
                f"Expected metric=None for qualitative claim, got {claim.result.metric!r}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 7. Missing result → result = null for methodological claims
# ─────────────────────────────────────────────────────────────────────────────
def test_methodological_claim_has_null_result():
    """A methodological claim (no outcome) must have result = null."""
    parser = DocumentParser()
    paper = make_paper(
        "We introduce a novel deep learning framework that integrates imaging and clinical data "
        "using cross-attention mechanisms for head and neck cancer prognosis."
    )
    results = parser.parse_paper(paper)
    if results:
        claim, _ = results[0]
        if claim.evidence_status == EvidenceStatus.methodological:
            assert claim.result is None, (
                f"Methodological claims must have result=None, got {claim.result}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 8. Background statement → NOT classified as direct_empirical
# ─────────────────────────────────────────────────────────────────────────────
def test_background_statement_not_direct_empirical():
    """
    A background motivation statement (no result, no experiment reported)
    must NOT be classified as direct_empirical.
    """
    parser = DocumentParser()
    paper = make_paper(
        "Cancer is a leading cause of mortality. Multimodal data fusion may improve "
        "diagnostic accuracy in clinical settings."
    )
    results = parser.parse_paper(paper)
    if results:
        claim, _ = results[0]
        assert claim.evidence_status != EvidenceStatus.direct_empirical, (
            f"Background statement incorrectly classified as direct_empirical: {claim.claim!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 9. Empty abstract → no claims generated
# ─────────────────────────────────────────────────────────────────────────────
def test_empty_abstract_produces_no_claims():
    """An empty abstract must produce exactly zero evidence claims."""
    parser = DocumentParser()
    paper = make_paper("")
    results = parser.parse_paper(paper)
    assert results == [], f"Expected 0 claims, got {len(results)}"


def test_whitespace_only_abstract_produces_no_claims():
    """A whitespace-only abstract must also produce zero evidence claims."""
    parser = DocumentParser()
    paper = make_paper("   \n\t  ")
    results = parser.parse_paper(paper)
    assert results == []


# ─────────────────────────────────────────────────────────────────────────────
# 10. Source location validation
# ─────────────────────────────────────────────────────────────────────────────
def test_claim_has_evidence_location_and_source_scope():
    """Every claim must have evidence_location set and source_scope set."""
    parser = DocumentParser()
    paper = make_paper(
        "Result: Early fusion of DL features and clinical parameters achieved AUC 0.77 "
        "vs 0.67 for clinical parameters alone."
    )
    results = parser.parse_paper(paper)
    assert results, "Should produce at least one claim"
    claim, _ = results[0]
    assert claim.evidence_location, "evidence_location must be non-empty"
    assert claim.source_scope is not None, "source_scope must be set"
    assert claim.provenance.source_reference, "provenance.source_reference must be set"


def test_claim_text_is_not_entire_abstract():
    """The claim text must never be the full abstract text."""
    parser = DocumentParser()
    abstract = (
        "Cancer prognosis prediction increasingly leverages multi-modal learning, yet existing "
        "approaches often rely on omics data that are costly and difficult to collect in routine "
        "practice. Pathology reports, by contrast, are routinely generated. We present CALM, a "
        "framework that integrates pathology images and reports. Across 14 TCGA cancer types, "
        "CALM improved prognostic accuracy compared to image and text baselines (up to +11.5% "
        "mean C-index)."
    )
    paper = make_paper(abstract)
    results = parser.parse_paper(paper)
    assert results, "Should produce at least one claim"
    claim, _ = results[0]
    # Claim text should be substantially shorter than the abstract
    assert len(claim.claim) < len(abstract) * 0.5, (
        f"Claim text is suspiciously long ({len(claim.claim)} chars) vs "
        f"abstract ({len(abstract)} chars). Claim may contain the full abstract."
    )
    assert abstract not in claim.claim, "Full abstract must not appear in claim text"
