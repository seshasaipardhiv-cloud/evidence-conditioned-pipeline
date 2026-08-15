# Section 6: Threats to Validity

We explicitly document the threats to validity across seven core scientific dimensions:

## 1. Internal Validity
- **Threat**: Potential data leakage across preprocessing transformations or target variable contamination.
- **Mitigation**: An 8-variable target isolation firewall was enforced, and all imputers, encoders, and resamplers were fitted strictly on the training fold.
- **Residual Threat**: The modest margin over Default XGBoost (`+0.0047`) and the Seed 100 loss indicate sensitivity to patient partition variance.

## 2. Dataset Validity
- **Threat**: Evaluation is limited to a single retrospective, single-center clinical cohort (HANCOCK structured tabular dataset).
- **Residual Threat**: Clinical characteristics, missingness rates, and class distributions may not represent broader clinical populations.

## 3. Statistical Validity
- **Threat**: Multi-seed evaluation was restricted to $n=3$ seeds (`42`, `100`, `2026`).
- **Residual Threat**: The sample size is underpowered for formal inferential hypothesis testing or $p$-value calculation. Claims are strictly descriptive.

## 4. External Validity & Generalizability
- **Threat**: High internal retrospective test performance (ROC-AUC > 0.97) may create an unwarranted assumption of clinical readiness.
- **Residual Threat**: External validation across independent multi-center hospital cohorts has not been performed.

## 5. Configuration Validity
- **Threat**: Two components (categorical encoding and loss function) were resolved via explicit project configuration rather than literature evidence.
- **Mitigation**: These components are explicitly labeled as project configurations and segregated from literature claims.

## 6. Evidence Corpus Limitations
- **Threat**: The literature retrieval corpus was focused on domain-specific cancer recurrence literature and may not capture all alternative ML techniques.

## 7. Model Comparison Limitations
- **Threat**: Baseline models were evaluated with standard configurations and may not represent the exhaustive upper bound of hyperparameter optimization.
