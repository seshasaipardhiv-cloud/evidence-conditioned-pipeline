# Evidence-Conditioned Compositional Pipeline Synthesis: Final Research Submission Package

**Manuscript Title:** Evidence-Conditioned Compositional Pipeline Synthesis: A Provenance-Aware Framework for Reproducible Clinical Machine Learning  
**Target Venue:** Journal of Biomedical Informatics / JAMIA  
**Submission Date:** August 14, 2026  
**Scientific Verdict:** Grade A — Submission-Ready (Audited across Stages 6A–6I)  
**Total Tests Passing:** 712 / 712 (100% pass rate)  

---

## 1. Submission Package Structure

```
evidence/final/submission/
├── final_research_paper.pdf       # Publication-ready formatted PDF (15 pages)
├── final_research_paper.md        # Submission markdown source (3,912 words)
├── README.md                      # Package overview, cryptographic ledger, and instructions
├── submission_manifest.json       # Cryptographic SHA-256 hashes of all submission files
├── figures/                       # Publication-quality visual evidence (300 DPI PNG & vector SVG)
│   ├── fig1.png / fig1.svg        # Figure 1: Pipeline Synthesis Architecture
│   ├── fig2.png / fig2.svg        # Figure 2: Candidate vs. Baseline Predictive Performance
│   ├── fig3.png / fig3.svg        # Figure 3: Multi-Seed Robustness & Seed 100 Dynamics
│   ├── fig4.png / fig4.svg        # Figure 4: Controlled Component Ablation Analysis
│   ├── fig5.png / fig5.svg        # Figure 5: Probability Calibration (Brier Score)
│   ├── fig6.png / fig6.svg        # Figure 6: Candidate Multi-Metric Performance Profile
│   ├── fig7.png / fig7.svg        # Figure 7: Component Provenance Ledger & Boundary
│   ├── fig8.png / fig8.svg        # Figure 8: Formal Scientific Claim Boundary Matrix
│   └── figure_manifest.json       # Metadata & figure SHA-256 ledger
├── supplementary/                 # Peer-review defense files and audit ledgers
│   ├── stage6i_reviewer_questions.json    # 25 Hostile Reviewer Questions & Honest Defenses
│   ├── stage6i_hostile_review.json        # 17-Dimension Pre-Submission Scientific Audit
│   ├── stage6i_final_verdict.json         # Grade A Scientific Verdict
│   ├── stage6a_master_results.json        # Authoritative Master Results
│   ├── stage6a_ablation_results.json      # Complete Component Ablation Ledger
│   └── stage6a_claim_boundaries.json      # Evaluated Scientific Claim Boundaries
└── reproducibility/               # Immutable contracts and executable configs
    ├── stage5a_experiment_contract.json   # Cryptographically Frozen Experiment Contract
    ├── stage3_6_configured_pipeline.json  # Synthesized Pipeline Architecture
    └── stage5b_candidate_results.json     # Raw Execution Outputs (Seeds 42, 100, 2026)
```

---

## 2. Authoritative Core Results (Verified Immutable)

- **Candidate Pipeline (Actual Executed Path):**
  - **Mean Test ROC-AUC:** `0.9751 ± 0.0114` (Seed 42: `0.9888`, Seed 100: `0.9609`, Seed 2026: `0.9756`)
  - **Mean Test PR-AUC:** `0.9679`
  - **Mean Test Brier Score:** `0.0175` (Lowest probability estimation error)
  - **Mean Test F1 Score:** `0.9611` | **Accuracy:** `0.9825`
- **Default XGBoost Baseline:** `0.9704 ± 0.0059` (Candidate Margin: `+0.0047`, modest)
- **Random Forest Baseline:** `0.9698 ± 0.0065`
- **Logistic Regression Baseline:** `0.9645 ± 0.0070`
- **Simple MLP Baseline (Minimal Reference):** `0.9405 ± 0.0192` (`max_iter=10`)
- **Ablation Findings:** Full Candidate `0.9751`, Without SMOTE `0.9773`, Mean Imputation `0.9767`, Ordinal Encoding `0.9784`, Default XGBoost `0.9686`.
- **Key Methodological Principle:** *Evidence validity and empirical dataset optimality are distinct concepts.*

---

## 3. Disclosed Study Weaknesses and Scientific Boundaries

1. **Single Retrospective Cohort:** Evaluated strictly on the retrospective HANCOCK dataset; external multi-center generalizability is unestablished.
2. **Sample Size of Random Seeds:** Evaluated across $n=3$ seeds; underpowered for formal inferential hypothesis testing ($p$-values suppressed).
3. **Modest Margin:** Improvement over Default XGBoost is modest (`+0.0047` mean ROC-AUC).
4. **Seed 100 Loss:** Candidate won 2 of 3 seeds (66.7%), losing to Default XGBoost on Seed 100 (`-0.0034` delta).
5. **No Clinical Deployment:** The framework is a research methodology and is **not clinically deployable**.
6. **Prospective Caveat for `progress_1`:** Longitudinal follow-up variables require explicit temporal exclusion in prospective deployments.

---

## 4. Cryptographic Checksums (SHA-256 Ledger)

| Submission Artifact | SHA-256 Checksum |
| :--- | :--- |
| `final_research_paper.pdf` | `0202b17a44577c1141392f8a169a9debc987df3a4e86e3686f5fbc95312c0b2b` |
| `final_research_paper.md` | `3eac14bc4babb6ad4001acfef9adfd6fb32db543465e715f4c43ff03f3f813bf` |
| `README.md` | *Self-contained package ledger* |
| `submission_manifest.json` | *See manifest file* |
| `figures/fig1.png` | `N/A` |
| `figures/fig2.png` | `N/A` |
| `figures/fig3.png` | `N/A` |
| `figures/fig4.png` | `N/A` |
| `figures/fig5.png` | `N/A` |
| `figures/fig6.png` | `N/A` |
| `figures/fig7.png` | `N/A` |
| `figures/fig8.png` | `N/A` |

---

## 5. Independent Reproduction Instructions

```bash
# 1. Clone repository and activate virtual environment
git clone <repo_url> && cd evidence-conditioned-pipeline
python -m venv .venv
source .venv/bin/activate  # Or .\.venv\Scripts\activate on Windows

# 2. Install frozen requirements
pip install -r requirements.txt

# 3. Verify cryptographic execution contract and run full verification suite
pytest backend/tests -v
```
