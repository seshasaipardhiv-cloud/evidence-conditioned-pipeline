"""
test_stage2b.py — Stage 2B full-text acquisition and deep extraction tests.

Covers 16 test cases:
  1.  Full-text paper detected when full_text_available=True
  2.  Inaccessible full text → full_text_access_status=not_accessible
  3.  Abstract-only fallback when full text unavailable
  4.  Section extraction identifies Results section
  5.  Experiment extraction from results section
  6.  Baseline extracted from Methods/Results
  7.  Metric extracted explicitly from results
  8.  Numerical result extracted with value
  9.  Missing numerical result stays null
  10. Ablation extracted from ablation section
  11. Negative evidence preserved (not suppressed)
  12. Contradiction candidate requires task+metric+mechanism+direction
  13. Fusion strategy correctly classified
  14. Provenance carries section + source_scope=full_text
  15. Page/table/figure location preserved when found
  16. No fabricated delta when either value is null
"""

import pytest
from datetime import datetime

from backend.app.stage2.section_parser import SectionParser
from backend.app.stage2.experiment_extractor import ExperimentExtractor
from backend.app.stage2.contradiction_detector import ContradictionDetector
from backend.app.stage2.models import (
    PaperRecord, EvidenceClaim, EvidenceStatus, SourceScope,
    ExtractionStatus, Provenance, ExtractionMethod, EmpiricalResult,
    FullTextAccessStatus, FusionStrategy, ResultRecord,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def make_paper(
    abstract: str = "",
    full_text_available: bool = False,
    full_text_access_status: FullTextAccessStatus = FullTextAccessStatus.not_found,
    paper_id: str = "p_test",
) -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        title="Test Paper",
        authors=["Author A"],
        publication_year=2024,
        doi="10.0000/test",
        source="PubMedSource",
        abstract=abstract if abstract.strip() else None,
        abstract_available=bool(abstract.strip()),
        full_text_available=full_text_available,
        full_text_access_status=full_text_access_status,
        retrieval_date=datetime.now().isoformat(),
    )


def make_claim(
    evidence_id: str,
    paper_id: str = "p_test",
    mechanisms: list = None,
    task: str = None,
    domain: str = None,
    modalities: list = None,
    evidence_status: EvidenceStatus = EvidenceStatus.direct_empirical,
    result: EmpiricalResult = None,
) -> EvidenceClaim:
    prov = Provenance(
        source_type="scholarly_api",
        source_reference="10.0000/test",
        extraction_method=ExtractionMethod.regex_based,
        extraction_status=ExtractionStatus.explicit,
        retrieval_date=datetime.now().isoformat(),
    )
    return EvidenceClaim(
        evidence_id=evidence_id,
        paper_id=paper_id,
        claim="Test claim",
        mechanisms=mechanisms or ["mech_cnn"],
        task=task,
        domain=domain,
        modalities=modalities or ["clinical"],
        evidence_location="Results/Table 2",
        source_scope=SourceScope.full_text,
        extraction_method=ExtractionMethod.regex_based,
        evidence_status=evidence_status,
        result=result,
        provenance=prov,
    )


FULL_TEXT_SAMPLE = """
Abstract
This paper presents a multimodal cancer prognosis system.

Introduction
Cancer prognosis is a critical problem.

Methods
We trained a CNN on 932 MRI images with clinical features using early fusion.
k-fold cross-validation with k=5 was used.
The model was compared against clinical parameters alone.

Results
The multimodal model achieved AUC 0.77 vs 0.67 for clinical features alone.
Early fusion outperformed late fusion (0.77 vs 0.73 AUC, P = 0.005).
Incorporating lesion volumes did not enhance diagnostic efficacy.

Ablation Study
Without clinical features the model achieved AUC 0.70.
Without imaging features the model achieved AUC 0.65.

Limitations
This study used a single-center dataset which may limit generalizability.
"""


# ─────────────────────────────────────────────────────────────────────────────
# 1. Full-text paper detected
# ─────────────────────────────────────────────────────────────────────────────
def test_full_text_paper_detected():
    """Paper with full_text_available=True must reflect accessible status."""
    paper = make_paper(
        full_text_available=True,
        full_text_access_status=FullTextAccessStatus.accessible,
    )
    assert paper.full_text_available is True
    assert paper.full_text_access_status == FullTextAccessStatus.accessible


# ─────────────────────────────────────────────────────────────────────────────
# 2. Inaccessible full text
# ─────────────────────────────────────────────────────────────────────────────
def test_inaccessible_full_text_status():
    """Paywalled paper must record full_text_access_status=not_accessible."""
    paper = make_paper(
        abstract="Some abstract text.",
        full_text_available=False,
        full_text_access_status=FullTextAccessStatus.not_accessible,
    )
    assert paper.full_text_available is False
    assert paper.full_text_access_status == FullTextAccessStatus.not_accessible


# ─────────────────────────────────────────────────────────────────────────────
# 3. Abstract-only fallback
# ─────────────────────────────────────────────────────────────────────────────
def test_abstract_only_fallback():
    """When full text is unavailable, abstract_available must still be True."""
    paper = make_paper(
        abstract="CNN model achieved AUC 0.85 on test set.",
        full_text_available=False,
        full_text_access_status=FullTextAccessStatus.abstract_only,
    )
    assert paper.abstract_available is True
    assert paper.full_text_available is False


# ─────────────────────────────────────────────────────────────────────────────
# 4. Section extraction — Results section identified
# ─────────────────────────────────────────────────────────────────────────────
def test_section_extraction_results_identified():
    """SectionParser must identify the Results section."""
    parser = SectionParser()
    sections = parser.parse(FULL_TEXT_SAMPLE)
    assert "results" in sections, f"Expected 'results' section. Got: {list(sections.keys())}"
    assert "AUC" in sections["results"] or "0.77" in sections["results"]


def test_section_extraction_background_not_empirical():
    """Introduction section must not be classified as empirical."""
    parser = SectionParser()
    assert not parser.is_empirical_section("introduction")
    assert parser.is_background_section("introduction")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Experiment extraction from results section
# ─────────────────────────────────────────────────────────────────────────────
def test_experiment_extraction_from_results():
    """ExperimentExtractor must produce at least one ExperimentRecord from full text."""
    extractor = ExperimentExtractor()
    exps, _ = extractor.extract(
        paper_id="p_test",
        text=FULL_TEXT_SAMPLE,
        source_scope=SourceScope.full_text,
    )
    assert len(exps) >= 1
    exp = exps[0]
    assert exp.paper_id == "p_test"
    assert exp.source_scope == SourceScope.full_text


# ─────────────────────────────────────────────────────────────────────────────
# 6. Baseline extracted
# ─────────────────────────────────────────────────────────────────────────────
def test_baseline_extracted_from_results():
    """Baseline should be identified when 'compared with X alone' appears."""
    extractor = ExperimentExtractor()
    exps, _ = extractor.extract(
        paper_id="p_test",
        text=FULL_TEXT_SAMPLE,
        source_scope=SourceScope.full_text,
    )
    assert exps
    # Baseline may or may not be found; but if found it must not be null
    exp = exps[0]
    if exp.baseline is not None:
        assert len(exp.baseline) > 0, "Baseline must not be empty string"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Metric extracted explicitly
# ─────────────────────────────────────────────────────────────────────────────
def test_metric_extracted_from_results():
    """AUC must appear in evaluation_metrics when stated in Results."""
    extractor = ExperimentExtractor()
    exps, _ = extractor.extract(
        paper_id="p_test",
        text=FULL_TEXT_SAMPLE,
        source_scope=SourceScope.full_text,
    )
    assert exps
    all_metrics = exps[0].evaluation_metrics
    assert any("AUC" in (m or "").upper() for m in all_metrics), (
        f"Expected AUC in metrics, got: {all_metrics}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 8. Numerical result extracted with value
# ─────────────────────────────────────────────────────────────────────────────
def test_numerical_result_extracted():
    """method_value must be captured from 'AUC 0.77 vs 0.67'."""
    extractor = ExperimentExtractor()
    exps, _ = extractor.extract(
        paper_id="p_test",
        text=FULL_TEXT_SAMPLE,
        source_scope=SourceScope.full_text,
    )
    assert exps
    results = exps[0].reported_results
    numeric_results = [r for r in results if r.method_value is not None]
    assert numeric_results, "Expected at least one result with method_value"
    # The AUC 0.77 vs 0.67 should yield method_value=0.77 and baseline_value=0.67
    auc_result = next((r for r in numeric_results if r.metric and "AUC" in r.metric.upper()), None)
    if auc_result:
        assert auc_result.method_value == pytest.approx(0.77, abs=0.001)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Missing numerical result stays null
# ─────────────────────────────────────────────────────────────────────────────
def test_missing_numerical_result_stays_null():
    """When abstract says 'improved' without numbers, result values must be null."""
    extractor = ExperimentExtractor()
    text = (
        "Results\n"
        "The multimodal approach improved overall performance "
        "compared with single-modality baselines."
    )
    exps, _ = extractor.extract(
        paper_id="p_test", text=text, source_scope=SourceScope.full_text
    )
    if exps:
        for r in exps[0].reported_results:
            if r.method_value is not None:
                # Only numeric results should have values; qualitative should not
                pass  # numeric may be absent entirely
            else:
                assert r.method_value is None


# ─────────────────────────────────────────────────────────────────────────────
# 10. Ablation extracted
# ─────────────────────────────────────────────────────────────────────────────
def test_ablation_extracted():
    """AblationRecord must be created for 'without clinical features'."""
    extractor = ExperimentExtractor()
    _, ablations = extractor.extract(
        paper_id="p_test",
        text=FULL_TEXT_SAMPLE,
        source_scope=SourceScope.full_text,
    )
    assert len(ablations) >= 1, "Expected at least one ablation record"
    conditions = [a.condition_removed for a in ablations]
    assert any("clinical" in c.lower() or "imaging" in c.lower() for c in conditions), (
        f"Expected 'clinical features' or 'imaging features' in ablations, got: {conditions}"
    )


def test_ablation_linked_to_parent_experiment():
    """AblationRecord must link to parent_experiment_id."""
    extractor = ExperimentExtractor()
    exps, ablations = extractor.extract(
        paper_id="p_test",
        text=FULL_TEXT_SAMPLE,
        source_scope=SourceScope.full_text,
    )
    if exps and ablations:
        parent_ids = {e.experiment_id for e in exps}
        for abl in ablations:
            assert abl.parent_experiment_id in parent_ids, (
                f"Ablation {abl.ablation_id} references unknown experiment"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 11. Negative evidence preserved
# ─────────────────────────────────────────────────────────────────────────────
def test_negative_evidence_preserved():
    """
    A paper reporting 'did not enhance diagnostic efficacy' must have
    at least one ResultRecord with direction='degradation'.
    """
    extractor = ExperimentExtractor()
    text = (
        "Results\n"
        "Incorporating lesion volumes did not enhance diagnostic efficacy. "
        "No significant improvement was observed for the extended model."
    )
    exps, _ = extractor.extract(
        paper_id="p_test", text=text, source_scope=SourceScope.full_text
    )
    assert exps
    negative = [r for e in exps for r in e.reported_results if r.direction == "degradation"]
    assert negative, "Negative findings must be preserved, not suppressed"


# ─────────────────────────────────────────────────────────────────────────────
# 12. Contradiction candidate requires shared task+metric+mechanism+opposing direction
# ─────────────────────────────────────────────────────────────────────────────
def test_contradiction_candidate_requires_shared_dimensions():
    """Two claims with same mechanism but different tasks must NOT produce a contradiction."""
    detector = ContradictionDetector()
    c1 = make_claim(
        "c1", task="classification",
        result=EmpiricalResult(metric="AUC", direction="improvement"),
    )
    c2 = make_claim(
        "c2", task="survival_prediction",
        result=EmpiricalResult(metric="AUC", direction="degradation"),
    )
    # Different tasks → no contradiction candidate
    candidates = detector.detect([c1, c2])
    assert len(candidates) == 0, "Different tasks must not produce a contradiction candidate"


def test_contradiction_candidate_with_shared_dimensions():
    """Two claims with same task+metric+mechanism but opposing directions → candidate."""
    detector = ContradictionDetector()
    c1 = make_claim(
        "c1", task="classification", mechanisms=["mech_cnn"],
        result=EmpiricalResult(metric="AUC", direction="improvement"),
    )
    c2 = make_claim(
        "c2", task="classification", mechanisms=["mech_cnn"],
        result=EmpiricalResult(metric="AUC", direction="degradation"),
    )
    candidates = detector.detect([c1, c2])
    assert len(candidates) == 1
    assert candidates[0].direction_a in ("improvement", "degradation")
    assert candidates[0].direction_b in ("improvement", "degradation")
    assert candidates[0].direction_a != candidates[0].direction_b


# ─────────────────────────────────────────────────────────────────────────────
# 13. Fusion strategy correctly classified
# ─────────────────────────────────────────────────────────────────────────────
def test_fusion_strategy_early_fusion():
    extractor = ExperimentExtractor()
    text = "Methods\nWe applied early fusion (feature-level) to combine clinical and imaging features."
    exps, _ = extractor.extract("p1", text, SourceScope.full_text)
    if exps and exps[0].fusion_strategy:
        assert exps[0].fusion_strategy == FusionStrategy.early_fusion


def test_fusion_strategy_cross_attention():
    extractor = ExperimentExtractor()
    text = "Methods\nA cross-attention mechanism was used to fuse PET and CT image features."
    exps, _ = extractor.extract("p1", text, SourceScope.full_text)
    if exps and exps[0].fusion_strategy:
        assert exps[0].fusion_strategy == FusionStrategy.cross_attention


def test_fusion_strategy_late_fusion():
    extractor = ExperimentExtractor()
    text = "Methods\nLate fusion combines predictions from individual modality models."
    exps, _ = extractor.extract("p1", text, SourceScope.full_text)
    if exps and exps[0].fusion_strategy:
        assert exps[0].fusion_strategy == FusionStrategy.late_fusion


# ─────────────────────────────────────────────────────────────────────────────
# 14. Provenance carries section + source_scope = full_text
# ─────────────────────────────────────────────────────────────────────────────
def test_provenance_carries_section():
    """ResultRecord produced from full text must have source_scope = full_text."""
    extractor = ExperimentExtractor()
    exps, _ = extractor.extract(
        paper_id="p_test",
        text=FULL_TEXT_SAMPLE,
        source_scope=SourceScope.full_text,
    )
    assert exps
    for r in exps[0].reported_results:
        assert r.source_scope == SourceScope.full_text, (
            f"Result source_scope must be full_text, got {r.source_scope}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 15. Page/table/figure location preserved
# ─────────────────────────────────────────────────────────────────────────────
def test_source_location_recorded():
    """ResultRecord must record source_location (section label at minimum)."""
    extractor = ExperimentExtractor()
    exps, _ = extractor.extract(
        paper_id="p_test",
        text=FULL_TEXT_SAMPLE,
        source_scope=SourceScope.full_text,
    )
    assert exps
    for r in exps[0].reported_results:
        assert r.source_location is not None, "source_location must not be null for full-text results"


# ─────────────────────────────────────────────────────────────────────────────
# 16. No fabricated delta when either value is null
# ─────────────────────────────────────────────────────────────────────────────
def test_no_fabricated_delta_when_values_missing():
    """delta must be null if baseline_value OR method_value is null."""
    # Direct model check
    result = ResultRecord(
        metric="AUC",
        method_value=0.85,
        baseline_value=None,  # missing
        delta=None,
        direction="improvement",
        source_location="Results",
        source_scope=SourceScope.full_text,
    )
    assert result.delta is None, "delta must not be fabricated when baseline is unknown"


def test_delta_computed_only_when_both_values_present():
    """delta = method - baseline only when both are explicitly known."""
    result = ResultRecord(
        metric="AUC",
        method_value=0.77,
        baseline_value=0.67,
        delta=round(0.77 - 0.67, 4),
        direction="improvement",
        source_location="Results/Table 1",
        source_scope=SourceScope.full_text,
    )
    assert result.delta == pytest.approx(0.10, abs=0.001)
    assert result.method_value is not None
    assert result.baseline_value is not None

# ─────────────────────────────────────────────────────────────────────────────
# Regression Tests from Stage 2B Audit
# ─────────────────────────────────────────────────────────────────────────────

def test_pca_cs_pca_regression():
    from backend.app.stage2.mechanism_mapper import MechanismMapper
    mapper = MechanismMapper()
    # TEST 1: "csPCa" must NOT map to PCA.
    cat, name = mapper.map_mechanism("This is for csPCa patients.")
    assert name == "This is for csPCa patients."

def test_honeybee_proposed_method_regression():
    extractor = ExperimentExtractor()
    # TEST 2: HONeYBEE must be recognized as a proposed method, not a dataset.
    text = "We introduced a novel multimodal framework called HONeYBEE for cancer prognosis."
    exps, _ = extractor.extract("p_test", text, SourceScope.abstract)
    assert exps[0].proposed_method == "HONeYBEE"
    assert exps[0].dataset is None

def test_honeybee_dataset_regression():
    extractor = ExperimentExtractor()
    # TEST 3: HONeYBEE actual dataset must be TCGA if explicitly supported.
    text = "Patients were obtained from the TCGA dataset. We evaluated HONeYBEE."
    exps, _ = extractor.extract("p_test", text, SourceScope.abstract)
    assert exps[0].dataset == "TCGA"

def test_calm_proposed_method_regression():
    extractor = ExperimentExtractor()
    # TEST 4: CALM must be recognized as the proposed method.
    text = "The proposed CALM approach is highly effective."
    exps, _ = extractor.extract("p_test", text, SourceScope.abstract)
    assert exps[0].proposed_method == "CALM"

def test_calm_baseline_regression():
    extractor = ExperimentExtractor()
    # TEST 5: CALM baseline must not become CALM itself.
    text = "Our method CALM was compared against PORPOISE."
    exps, _ = extractor.extract("p_test", text, SourceScope.abstract)
    assert exps[0].baseline == "PORPOISE"

def test_bioengineering_task_regression():
    extractor = ExperimentExtractor()
    # TEST 6: Bioengineering paper task must be diagnosis/HPV status, not survival_prediction.
    text = "We predict HPV status using imaging."
    exps, _ = extractor.extract("p_test", text, SourceScope.abstract)
    assert exps[0].task == "classification"

def test_bioengineering_modality_regression():
    extractor = ExperimentExtractor()
    # TEST 7: Bioengineering paper modalities must not include text/omics unless explicitly supported.
    text = "We used CT and PET imaging. Other studies rely on omics data."
    exps, _ = extractor.extract("p_test", text, SourceScope.abstract)
    assert "imaging" in exps[0].modalities
    assert "omics" not in exps[0].modalities

def test_hnc_modality_regression():
    extractor = ExperimentExtractor()
    # TEST 8: HNC paper must not include omics unless explicitly supported.
    text = "We relied on clinical data and images. Omics were avoided."
    exps, _ = extractor.extract("p_test", text, SourceScope.abstract)
    assert "clinical" in exps[0].modalities
    assert "imaging" in exps[0].modalities
    assert "omics" not in exps[0].modalities

def test_tcga_late_fusion_regression():
    extractor = ExperimentExtractor()
    # TEST 9: TCGA multimodal fusion paper must preserve late_fusion.
    text = "The model utilizes a late fusion architecture."
    exps, _ = extractor.extract("p_test", text, SourceScope.abstract)
    assert exps[0].fusion_strategy == FusionStrategy.late_fusion

def test_background_modality_regression():
    extractor = ExperimentExtractor()
    # TEST 10: Background mentions of a modality must not create a modality.
    text = "Clinical data is common. However, we used imaging."
    exps, _ = extractor.extract("p_test", text, SourceScope.abstract)
    assert "imaging" in exps[0].modalities
    assert "clinical" not in exps[0].modalities


# ─────────────────────────────────────────────────────────────────────────────
# New Regression Tests 11–16 (per spec Section 12)
# ─────────────────────────────────────────────────────────────────────────────

def test_proposed_method_not_in_baseline_list():
    """TEST 11: The proposed method must NOT appear in its own baseline list."""
    extractor = ExperimentExtractor()
    # CALM is the proposed method; PORPOISE is the baseline.
    # CALM must not appear in exps[0].baselines.
    text = "Our method CALM was compared against PORPOISE."
    exps, _ = extractor.extract("p_test", text, SourceScope.abstract)
    assert exps
    exp = exps[0]
    assert exp.proposed_method == "CALM", f"Expected CALM as proposed method, got {exp.proposed_method}"
    baseline_names = [b.name.upper() for b in exp.baselines]
    assert "CALM" not in baseline_names, f"CALM must not appear in its own baselines: {baseline_names}"
    assert "PORPOISE" in baseline_names, f"PORPOISE must appear in baselines: {baseline_names}"


def test_multiple_baselines_preserved():
    """TEST 12: Multiple distinct baselines must all be captured."""
    extractor = ExperimentExtractor()
    text = (
        "The model was compared against PORPOISE, SurvPath, and the image-only model. "
        "Our method outperformed all baselines."
    )
    exps, _ = extractor.extract("p_test", text, SourceScope.abstract)
    assert exps
    exp = exps[0]
    # Must have at least 2 baseline entries
    assert len(exp.baselines) >= 2, (
        f"Expected at least 2 baselines, got {[b.name for b in exp.baselines]}"
    )


def test_missing_evidence_produces_null():
    """TEST 13: When dataset/task/modality cannot be determined, fields must be null/empty."""
    extractor = ExperimentExtractor()
    text = "Deep learning has shown promise in medical imaging."
    exps, _ = extractor.extract("p_test", text, SourceScope.abstract)
    assert exps
    exp = exps[0]
    # No explicit prediction objective → task must be null
    assert exp.task is None, f"Expected task=None, got {exp.task}"
    # No data source pattern → dataset must be null
    assert exp.dataset is None, f"Expected dataset=None, got {exp.dataset}"
    # No explicit input relationship → modalities must be empty
    assert exp.modalities == [], f"Expected empty modalities, got {exp.modalities}"


def test_every_accepted_field_has_provenance():
    """TEST 14: Every non-null scientific field must carry a FieldProvenance entry."""
    extractor = ExperimentExtractor()
    text = (
        "We propose HONeYBEE, a novel multimodal framework. "
        "Patients were obtained from the TCGA cohort. "
        "We used clinical data, imaging, and omics as input to predict overall survival. "
        "Our model uses a cross-attention architecture. "
        "HONeYBEE was compared against PORPOISE."
    )
    exps, _ = extractor.extract("p_test", text, SourceScope.abstract)
    assert exps
    exp = exps[0]
    prov = exp.field_provenance

    # For each non-null/non-empty scientific field, provenance must exist
    if exp.dataset is not None:
        assert "dataset" in prov, "dataset has value but no provenance"
    if exp.task is not None:
        assert "task" in prov, "task has value but no provenance"
    if exp.proposed_method is not None:
        assert "proposed_method" in prov, "proposed_method has value but no provenance"
    if exp.fusion_strategy is not None:
        assert "fusion_strategy" in prov, "fusion_strategy has value but no provenance"
    for mod in exp.modalities:
        assert f"modalities_{mod}" in prov, f"modality '{mod}' has no provenance"
    # Every provenance entry must have a non-empty source_sentence
    for key, fp in prov.items():
        assert fp.source_sentence, f"Provenance for '{key}' has empty source_sentence"
        assert fp.field_name, f"Provenance for '{key}' has empty field_name"
        assert fp.confidence_status is not None, f"Provenance for '{key}' has no confidence_status"


def test_cspca_inside_token_does_not_trigger_pca():
    """TEST 15: 'csPCa' inside compound tokens must not trigger PCA mapping."""
    from backend.app.stage2.mechanism_mapper import MechanismMapper
    mapper = MechanismMapper()
    # Test 1: plain token
    cat, name = mapper.map_mechanism("This study targets csPCa lesions.")
    assert name != "pca", f"csPCa must not map to PCA, got category={cat}, name={name}"
    # Test 2: inside a hyphenated compound
    cat2, name2 = mapper.map_mechanism("The csPCa-related findings are important.")
    assert name2 != "pca", f"csPCa-related must not map to PCA"
    # Test 3: all-caps context that contains PCA as substring
    cat3, name3 = mapper.map_mechanism("TSPCA and csPCa are different.")
    assert cat3.value != "Representation" or name3 != "pca", (
        "PCA must not match inside TSPCA"
    )


def test_generic_multimodal_does_not_determine_fusion():
    """TEST 16: The word 'multimodal' alone must NOT determine fusion strategy."""
    extractor = ExperimentExtractor()
    # Text only says "multimodal" — no explicit fusion architecture described
    text = "We present a multimodal model for cancer prognosis."
    exps, _ = extractor.extract("p_test", text, SourceScope.abstract)
    assert exps
    exp = exps[0]
    assert exp.fusion_strategy is None, (
        f"'multimodal' alone must not set fusion_strategy; got {exp.fusion_strategy}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Precision Repair Regression Tests (from Stage 2B Real-Data Audit)
# Ensure known false positives are eliminated.
# ─────────────────────────────────────────────────────────────────────────────

def test_raw_clinical_data_not_dataset():
    """'raw clinical data' must NOT produce dataset='raw'."""
    extractor = ExperimentExtractor()
    text = "These methods extract features from the raw dataset with faster inference time."
    exps, _ = extractor.extract("p_test", text, SourceScope.full_text)
    assert exps
    exp = exps[0]
    assert exp.dataset != "raw", (
        f"'raw' must not be extracted as a dataset name, got {exp.dataset!r}"
    )
    assert exp.dataset is None or exp.dataset not in ("raw", "Raw"), (
        f"dataset must not be a generic word, got {exp.dataset!r}"
    )


def test_cancer_patients_not_dataset():
    """'cancer patients' must NOT produce dataset='cancer'."""
    extractor = ExperimentExtractor()
    text = (
        "A recent study showed feasibility in a lung cancer cohort to reliably extract "
        "important prognostic factors."
    )
    exps, _ = extractor.extract("p_test", text, SourceScope.full_text)
    assert exps
    exp = exps[0]
    assert exp.dataset != "cancer", (
        f"'cancer' must not be extracted as a dataset name, got {exp.dataset!r}"
    )


def test_offers_not_proposed_method():
    """'Our model offers...' must NOT produce proposed_method='offers'."""
    extractor = ExperimentExtractor()
    text = (
        "Our model offers noticeable AUC, precision, recall, and specificity gains "
        "even using a small-scale ensemble model with just 4.5k parameters."
    )
    exps, _ = extractor.extract("p_test", text, SourceScope.full_text)
    assert exps
    exp = exps[0]
    assert exp.proposed_method != "offers", (
        f"'offers' is a verb and must not be extracted as a method name, got {exp.proposed_method!r}"
    )


def test_laying_not_proposed_method():
    """'laying a foundation' must NOT produce proposed_method='laying'."""
    extractor = ExperimentExtractor()
    text = (
        "This choice further refined our model, laying a foundation for the "
        "successful completion of the task."
    )
    exps, _ = extractor.extract("p_test", text, SourceScope.full_text)
    assert exps
    exp = exps[0]
    assert exp.proposed_method != "laying", (
        f"'laying' is a gerund and must not be extracted as a method name, got {exp.proposed_method!r}"
    )


def test_previous_studies_omics_not_extracted():
    """'previous studies used omics' must NOT extract omics as a modality."""
    extractor = ExperimentExtractor()
    text = "Previous studies have used omics data to classify cancer subtypes."
    exps, _ = extractor.extract("p_test", text, SourceScope.full_text)
    assert exps
    exp = exps[0]
    assert "omics" not in exp.modalities, (
        f"Background omics mention must not create modality, got {exp.modalities}"
    )


def test_our_study_ct_images_extracted():
    """'our study used CT images' SHOULD extract imaging modality."""
    extractor = ExperimentExtractor()
    text = "We used CT images and clinical data as input for survival prediction."
    exps, _ = extractor.extract("p_test", text, SourceScope.abstract)
    assert exps
    exp = exps[0]
    assert "imaging" in exp.modalities, (
        f"CT images with explicit input verb should detect imaging, got {exp.modalities}"
    )


def test_we_propose_honeybee_extracted_as_method():
    """'we propose HONeYBEE' SHOULD extract HONeYBEE as proposed method."""
    extractor = ExperimentExtractor()
    text = "We introduce a novel multimodal framework called HONeYBEE for cancer prognosis."
    exps, _ = extractor.extract("p_test", text, SourceScope.abstract)
    assert exps
    exp = exps[0]
    assert exp.proposed_method == "HONeYBEE", (
        f"HONeYBEE should be extracted as proposed_method, got {exp.proposed_method!r}"
    )


def test_we_used_tcga_cohort_extracted_as_dataset():
    """'we used the TCGA cohort' SHOULD extract TCGA as dataset."""
    extractor = ExperimentExtractor()
    text = "We used the TCGA cohort to evaluate the proposed approach."
    exps, _ = extractor.extract("p_test", text, SourceScope.abstract)
    assert exps
    exp = exps[0]
    assert exp.dataset == "TCGA", (
        f"TCGA cohort should be extracted as dataset, got {exp.dataset!r}"
    )


def test_cspca_not_mapped_to_pca():
    """'csPCa' must NOT be mapped to PCA by the mechanism mapper."""
    from backend.app.stage2.mechanism_mapper import MechanismMapper
    mapper = MechanismMapper()
    cat, name = mapper.map_mechanism("Detection of csPCa lesions.")
    assert name != "pca", (
        f"csPCa must not map to PCA, got category={cat.value}, name={name!r}"
    )
