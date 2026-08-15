# Limitations and Non-Claims

## 1. Single Retrospective Cohort
The experimental evaluation was conducted exclusively on the retrospective HANCOCK clinical tabular dataset. Generalizability to external clinical environments, diverse healthcare institutions, or alternative cancer types has not been established.

## 2. Sample Size of Seeds
The empirical evaluation was conducted across $n=3$ random seeds (`42`, `100`, `2026`). While sufficient for descriptive robustness analysis, this sample size is underpowered for formal inferential hypothesis testing. We explicitly suppress claims of statistical significance.

## 3. Modest Improvement Margin
The primary predictive improvement over Default XGBoost is modest: `+0.0047` mean ROC-AUC (+0.48% relative improvement).

## 4. Inconsistent Seed-Level Dominance
The candidate pipeline won on 2 out of 3 seeds (Seed 42: `+0.0105`, Seed 2026: `+0.0071`), but exhibited a lower score than Default XGBoost on Seed 100 (`-0.0034`). Universal fold dominance is not established.

## 5. Ablation Divergence
Ablations omitting SMOTE (`0.9773`) or employing Ordinal Encoding (`0.9784`) achieved marginally higher ROC-AUC than the full candidate pipeline (`0.9751`). This demonstrates that literature-grounded mechanisms do not guarantee empirical performance optimality on a specific retrospective dataset.

## 6. Absence of Clinical Deployment Readiness
No multi-center prospective trial, decision-curve analysis, or clinical workflow integration has been performed. The synthesized pipeline is a research framework and is **not clinically deployable**.
