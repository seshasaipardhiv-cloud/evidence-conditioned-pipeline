# 8. Threats to Validity and Limitations

We explicitly document the limitations and non-claims of this study:

1. **Single Retrospective Cohort**: Evaluated solely on the single-center retrospective HANCOCK clinical tabular dataset. Generalizability to external clinical cohorts remains unestablished.
2. **Sample Size of Random Seeds**: Evaluated across $n=3$ seeds (`42`, `100`, `2026`). While providing descriptive robustness, this sample size is underpowered for formal inferential hypothesis testing or $p$-value estimation.
3. **Modest Performance Margin**: The predictive improvement over Default XGBoost is modest (`+0.0047` mean ROC-AUC, +0.48% relative).
4. **Lack of Universal Seed Dominance**: The candidate pipeline lost to Default XGBoost on Seed 100 (`-0.0034` delta), demonstrating that superiority is split-dependent.
5. **Ablation Divergence**: Configurations omitting SMOTE (`0.9773`) or utilizing Ordinal Encoding (`0.9784`) achieved marginally higher ROC-AUC, confirming that evidence backing does not equal empirical dataset optimality.
6. **No External or Prospective Validation**: External multi-center validation and prospective clinical trial validation have not been performed.
7. **No Clinical Deployment Readiness**: The framework is a research methodology and is **not clinically deployable**.
