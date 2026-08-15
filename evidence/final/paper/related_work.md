# 2. Related Work and Research Gap

## 2.1 Tabular Clinical Machine Learning and Risk Prediction
Predictive modeling on structured clinical data has extensively explored regularized gradient tree boosting, random forests, and neural networks. Prior biomedical studies have demonstrated the efficacy of gradient boosted decision trees for tabular clinical risk prediction (e.g., PMID: 41775771), the necessity of principled missing-value handling for missing-at-random covariates (e.g., PMID: 41826845), and the utility of synthetic oversampling techniques like SMOTE for severe class imbalance in cancer recurrence cohorts (e.g., PMID: 41006422). Furthermore, multimodal architectures have investigated cross-attention mechanisms and ensembling strategies to combine clinical tabular features with imaging and molecular modalities (e.g., PMID: 42487970).

## 2.2 Automated Machine Learning (AutoML) vs. Evidence Synthesis
Automated Machine Learning (AutoML) frameworks—such as TPOT, Auto-sklearn, and Auto-PyTorch—automate pipeline construction through empirical search, genetic programming, or Bayesian optimization over unconstrained hyperparameter spaces. While effective for unconstrained performance maximization, conventional AutoML systems operate without domain provenance, frequently select opaque or clinically counter-intuitive preprocessing combinations, and lack formal governance against data leakage or arbitrary defaults. In contrast, evidence-conditioned synthesis constrains the search space strictly to domain mechanisms reported in peer-reviewed biomedical literature, ensuring that every architectural choice possesses an auditable provenance trail.

## 2.3 Reporting Guidelines, Data Leakage, and Reproducibility
The clinical machine learning community has increasingly emphasized reporting rigor and leakage prevention, as formalized in the TRIPOD+AI and PROBAST / PROBAST-AI guidelines. Data leakage—particularly the contamination of validation partitions during feature engineering, imputation, or scaling—remains a leading cause of reproducibility failures in medical AI. This work operationalizes these reporting standards by embedding cryptographic contracts and programmatic verification firewalls directly into the pipeline execution lifecycle.

## 2.4 The Compositional Synthesis Gap
Within the reviewed evidence corpus, existing research focuses almost exclusively on evaluating standalone algorithms or manual monolithic pipelines. This creates a major methodological void:
1. **Isolated Evidence**: Literature presents fragmented evidence for individual pipeline stages without a principled composition methodology.
2. **Arbitrary Default Proliferation**: Underspecified pipeline components are routinely filled with arbitrary library defaults without documented provenance.
3. **Target and Data Leakage**: Transformations are frequently fitted across validation boundaries, artificially inflating reported discrimination.
4. **Lack of Provenance Boundaries**: Unverified components are often retroactively claimed as literature-backed.

This work addresses this gap by developing a traceable, provenance-aware, and reproducible framework for evidence-conditioned compositional pipeline synthesis.
