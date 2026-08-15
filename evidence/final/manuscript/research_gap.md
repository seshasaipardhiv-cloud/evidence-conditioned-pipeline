# Section 2: Methodological Research Gap

## 2.1 The Disconnect in Biomedical Machine Learning
Biomedical literature abounds with isolated studies reporting individual algorithms, preprocessing choices, and feature representations for clinical risk prediction. However, a major methodological gap exists between published literature and reproducible pipeline implementation:

1. **Arbitrary Default Proliferation**: When implementing published methods, practitioners routinely fill unmentioned or underspecified pipeline steps with arbitrary machine learning library defaults (e.g., default imputers, standard loss functions, default encoding) without documenting their provenance or scientific rationale.
2. **Fabricated Provenance Risk**: Underspecified choices are frequently retroactively justified or claimed as literature-backed without verifiable textual citations.
3. **Data and Target Leakage**: In biomedical risk prediction, preprocessing transformations (imputation, scaling, resampling) are frequently fitted across the entire cohort prior to cross-validation, or outcome-derived variables are inadvertently retained in the feature set, artificially inflating reported metrics.
4. **Silent Fallback and Unverified Substitution**: Failed components or incompatible representations often silently fall back to alternative algorithms without audit logging.

## 2.2 The Addressed Gap
Within the reviewed evidence and project corpus, existing research focuses primarily on developing individual ML algorithms or monolithic predictive models. This project addresses the overarching methodological question:

> *Can published biomedical literature evidence be systematically transformed into an end-to-end traceable, provenance-aware, executable, and reproducible machine learning pipeline while enforcing strict firewalls against arbitrary defaults, fabricated provenance, and target leakage?*

This work shifts the focus from ad-hoc manual model tuning to a principled, evidence-conditioned compositional pipeline synthesis methodology.
