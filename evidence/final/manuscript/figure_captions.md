# Publication Figure Captions

### Figure 1: Evidence-Conditioned Pipeline Synthesis Architecture
**Figure 1.** Schematic overview of the evidence-conditioned pipeline synthesis architecture. The workflow begins with biomedical literature retrieval from PubMed/PMC, extracts structured claims, audits provenance authenticity, and grounds mechanisms into a controlled taxonomy. Primitive slots lacking literature evidence (categorical encoding and loss function) are passed through an explicit configuration gate rather than populated by arbitrary library defaults. The finalized specification is materialized, subjected to 10 readiness verification gates, and executed under a frozen contract. Colors distinguish literature-backed components (blue) from explicitly configured components (amber).

### Figure 2: Candidate vs Baseline Predictive Performance
**Figure 2.** Comparison of mean test ROC-AUC across 3 random seeds (`[42, 100, 2026]`) on the retrospective HANCOCK clinical cohort. Error bars represent $\pm 1$ standard deviation. The evidence-conditioned candidate pipeline achieved a mean ROC-AUC of `0.9751 ± 0.0114`, compared to `0.9704 ± 0.0059` for Default XGBoost (mean $\Delta = +0.0047$, +0.48% relative), `0.9698 ± 0.0065` for Random Forest, `0.9645 ± 0.0070` for Logistic Regression, and `0.9405 ± 0.0192` for Simple MLP.

### Figure 3: Per-Seed Robustness and Margin Analysis
**Figure 3.** Seed-by-seed performance comparison between the Candidate Pipeline and Default XGBoost baseline. The candidate outperformed Default XGBoost on Seed 42 (`0.9888` vs `0.9783`, $\Delta = +0.0105$) and Seed 2026 (`0.9756` vs `0.9685`, $\Delta = +0.0071$), but achieved a lower score on Seed 100 (`0.9609` vs `0.9643`, $\Delta = -0.0034$), demonstrating a 66.7% win rate without universal fold dominance.

### Figure 4: Controlled Component Ablation Analysis
**Figure 4.** Controlled ablation analysis evaluating the empirical contribution of individual pipeline components across identical patient splits and seeds. The Full Candidate achieved `0.9751` ROC-AUC. Ablation without SMOTE achieved `0.9773`, simple mean imputation achieved `0.9767`, and ordinal encoding achieved `0.9784`. These findings illustrate that literature-backed validity does not guarantee empirical optimality on a single retrospective dataset.

### Figure 5: Probability Calibration Comparison
**Figure 5.** Test set Brier score comparison across candidate and baseline models. Lower values indicate superior probability calibration. The Candidate Pipeline achieved the lowest Brier score (`0.0175`), followed by Default XGBoost (`0.0180`), Logistic Regression (`0.0201`), Random Forest (`0.0207`), and Simple MLP (`0.0683`), confirming that high discrimination was achieved without probability distortion.

### Figure 6: Multi-Metric Candidate Pipeline Performance Profile
**Figure 6.** Holistic test-set performance profile of the synthesized candidate pipeline across primary and secondary metrics: ROC-AUC (`0.9751`), PR-AUC (`0.9679`), Accuracy (`0.9825`), Precision (`0.9801`), F1 Score (`0.9611`), and Recall (`0.9429`), with a calibration Brier score of `0.0175`.

### Figure 7: Synthesized Pipeline Component Provenance and Evidence Boundary
**Figure 7.** Provenance ledger mapping each of the 8 final pipeline primitives to its exact origin. Six primitives are cryptographically linked to peer-reviewed PubMed citations, while categorical encoding and loss function are explicitly demarcated as human-gated project configurations.

### Figure 8: Formal Scientific Claim Boundary Matrix
**Figure 8.** Evaluated claim boundary matrix establishing the definitive scientific boundaries of the project. Five methodological and descriptive claims are `SUPPORTED`, one baseline comparison claim is `PARTIALLY_SUPPORTED`, and four inferential and clinical deployment claims are strictly `NOT_SUPPORTED`.
