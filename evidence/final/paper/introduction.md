# 1. Introduction

Biomedical machine learning pipelines require numerous interconnected architectural design choices, including feature representation, missing value imputation, categorical encoding, class imbalance handling, base learner selection, modality fusion, ensembling, and loss function configuration. While individual design decisions are frequently explored across published medical literature, synthesizing these heterogeneous findings into an end-to-end, reproducible, and executable predictive pipeline presents profound methodological and governance challenges.

In conventional machine learning workflows, practitioners routinely bridge gaps in literature descriptions by introducing arbitrary library defaults (e.g., default imputation algorithms or unverified hyperparameter settings) without establishing their empirical provenance or domain validity. Furthermore, ad-hoc composition frequently leads to subtle data leakage—such as fitting preprocessing scalers across entire patient cohorts prior to cross-validation—or inadvertent retention of outcome-derived clinical variables in the feature set.

To overcome these challenges, we introduce an **Evidence-Conditioned Compositional Pipeline Synthesis** framework. The framework transforms peer-reviewed biomedical literature findings into safe, executable, and provenance-tracked machine learning pipelines through a sequence of gated evolutionary audits. Crucially, the architecture establishes a formal boundary between *evidence-backed primitives* (grounded in literature citations) and *explicitly configured primitives* (human-controlled project inputs), strictly barring arbitrary defaults from entering the pipeline.

### Research Gap and Central Question
While existing biomedical ML research has developed sophisticated predictive algorithms, a critical methodological gap remains in how to systematically, safely, and reproducibly compose heterogeneous evidence into executable pipelines. We investigate the central research question:

> *Can heterogeneous evidence from biomedical machine learning literature be transformed into an executable clinical prediction pipeline while preserving provenance, preventing unsupported defaults, and enforcing reproducible experimental constraints?*

### Summary of Contributions
1. An end-to-end evidence-conditioned pipeline synthesis methodology bridging literature evidence to executable code.
2. A formal provenance and governance firewall distinguishing literature evidence from explicit configurations.
3. 10 independent materialization verification gates enforcing train-only preprocessing and target isolation.
4. Empirical demonstration on the retrospective HANCOCK clinical cohort with complete ablation analysis, operational transparency, and conservative claim boundaries.
