# Scientific Revision Policy and Empirical Governance Charter

**Purpose:** This charter establishes binding scientific governance rules for all future manuscript revisions, peer-review responses, and supplementary analyses for the project:  
*“Evidence-Conditioned Compositional Pipeline Synthesis: A Provenance-Aware Framework for Reproducible Clinical Machine Learning”*.

---

## 1. Immutable Baseline Invariants (Strict Non-Negotiables)
Under no circumstances may a future revision or rebuttal:
1. **Alter Stage 5B Raw Results:** The empirical baseline results from Stage 5B (Candidate ROC-AUC: `0.9751 ± 0.0114`, Default XGBoost: `0.9704 ± 0.0059`, Delta: `+0.0047`, Brier: `0.0175`, Seeds: `[42, 100, 2026]`) are immutable historical records.
2. **Manufacture New Evidence or p-values:** No statistical significance claims ($p < 0.05$) may be retroactively computed or claimed from the $n=3$ seed benchmark.
3. **Hide Negative Findings:** The Seed 100 candidate loss (`0.9609` vs `0.9643`, `Δ = -0.0034`) and the inverted ablation results (omitting SMOTE: `0.9773`, ordinal encoding: `0.9784`) must remain prominently disclosed in all revised manuscripts.
4. **Claim General Deep Learning Superiority:** The Simple MLP baseline (`max_iter=10`) must remain characterized as a shallow, minimal reference comparator.
5. **Claim Clinical Deployment Readiness:** The framework must remain characterized as a research methodology evaluated on the single retrospective HANCOCK cohort, explicitly requiring prospective multi-center trials before clinical translation.
6. **Obscure Operational Imputation:** The distinction between the literature-derived taxonomy component family (MICE/MissForest) and the train-fitted univariate median/mode executor must remain explicit.
7. **Obscure Dormant Primitives:** `cross_attention` and `average_ensembling` must remain designated as dormant taxonomy capabilities during unimodal tabular evaluation.
8. **Obscure Temporal Prediction Epoch:** The prediction epoch must remain anchored to *Post-Adjuvant Recurrence Risk Prediction*, and the prospective exclusion requirement for `progress_1` must be preserved.

---

## 2. Protocol for Reviewer-Requested Experiments
If peer reviewers or journal editors request additional experimental evaluations (e.g., additional random seeds, new baseline algorithms, or external cohort testing):
1. **Separate Stage Versioning:** All reviewer-requested experiments must be executed under a separate, explicitly versioned revision directory (e.g., `evidence/processed/stage5r1_reviewer_experiments.json`).
2. **Zero Overwriting of Primary Package:** The original Stage 5B / 5C / 6A / 6H / 6I artifacts must never be overwritten, modified, or mutated.
3. **Side-by-Side Reporting:** Reviewer experiments must be presented in the revision as supplementary responses or clearly demarcated revision sections (e.g., "Section 5.5: Reviewer-Requested Robustness Resampling"), maintaining full provenance transparency.

---

## 3. Mandatory Reviewer Response Checklist
Every point-by-point reviewer response must satisfy:
- [ ] Direct citation of relevant manuscript sections and line numbers.
- [ ] Reference to underlying cryptographic audit artifacts in `evidence/final/reconciliation/` or `evidence/final/submission/`.
- [ ] Complete consistency with the 10 materialization safety gates and frozen execution contracts.
- [ ] Approval by all co-authors prior to portal resubmission.
