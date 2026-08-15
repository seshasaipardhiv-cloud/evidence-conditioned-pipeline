# Section 4: Experimental Results

## 4.1 Primary Discriminative Performance
The evidence-conditioned candidate pipeline was evaluated across three deterministic random seeds (`42`, `100`, `2026`) on the retrospective HANCOCK clinical tabular cohort. Across all test partitions, the candidate pipeline achieved a mean test ROC-AUC of `0.9751 ± 0.0114` (range: `[0.9609, 0.9888]`).

Comprehensive multi-metric evaluation yielded:
- **Mean Test ROC-AUC**: `0.9751 ± 0.0114`
- **Mean Test PR-AUC**: `0.9679`
- **Mean Test F1 Score**: `0.9611`
- **Mean Test Accuracy**: `0.9825`
- **Mean Test Precision**: `0.9801`
- **Mean Test Recall**: `0.9429`
- **Mean Test Brier Score**: `0.0175`

## 4.2 Baseline Model Comparison
We evaluated the candidate pipeline against four standardized baseline models trained on identical patient splits:
1. **Default XGBoost Baseline**: Mean ROC-AUC of `0.9704 ± 0.0059` (Candidate delta: `+0.0047`, +0.48% relative).
2. **Random Forest Baseline**: Mean ROC-AUC of `0.9698 ± 0.0065` (Candidate delta: `+0.0053`, +0.55% relative).
3. **Logistic Regression Baseline**: Mean ROC-AUC of `0.9645 ± 0.0070` (Candidate delta: `+0.0106`, +1.10% relative).
4. **Simple MLP Baseline**: Mean ROC-AUC of `0.9405 ± 0.0192` (Candidate delta: `+0.0346`, +3.68% relative).

The primary predictive improvement of the candidate pipeline over Default XGBoost (`+0.0047` ROC-AUC) is **modest**. Tree-based ensemble architectures consistently demonstrated high baseline discrimination on this structured clinical feature set.

## 4.3 Multi-Seed Robustness and Margin Dynamics
Evaluating the candidate against Default XGBoost across individual test folds reveals split-dependent performance variation:
- **Seed 42**: Candidate `0.9888` vs Default XGBoost `0.9783` (**Candidate Won**, $\Delta = +0.0105$).
- **Seed 100**: Candidate `0.9609` vs Default XGBoost `0.9643` (**Candidate Lost**, $\Delta = -0.0034$).
- **Seed 2026**: Candidate `0.9756` vs Default XGBoost `0.9685` (**Candidate Won**, $\Delta = +0.0071$).

The candidate pipeline won on 2 out of 3 seeds (66.7% win rate). While the candidate achieved a higher mean score, it did **not** achieve universal fold dominance across all test partitions. Because $n=3$ seeds is underpowered, we do not claim statistical significance for this margin.

## 4.4 Controlled Component Ablations
To isolate the contribution of individual pipeline mechanisms, ablations were executed on identical splits and seeds:
- **Full Candidate Pipeline** (MICE + OneHot + SMOTE + Tuned XGBoost): `0.9751` ROC-AUC.
- **Ablation B (Without SMOTE)**: `0.9773` ROC-AUC ($\Delta = +0.0022$).
- **Ablation C (Simple Mean Imputation)**: `0.9767` ROC-AUC ($\Delta = +0.0016$).
- **Ablation D (Ordinal Encoding)**: `0.9784` ROC-AUC ($\Delta = +0.0033$).
- **Ablation E (Default XGBoost)**: `0.9686` ROC-AUC ($\Delta = -0.0065$).

*Key Finding*: Ablations omitting SMOTE or employing Ordinal Encoding achieved marginally higher ROC-AUC on this specific dataset. This demonstrates that literature-grounded validity does not guarantee empirical performance optimality on a specific retrospective dataset.

## 4.5 Probability Calibration
Probability calibration was assessed via test-set Brier score:
- **Candidate Pipeline**: `0.0175` (Lowest probability error across all models)
- **Default XGBoost**: `0.0180`
- **Logistic Regression**: `0.0201`
- **Random Forest**: `0.0207`
- **Simple MLP**: `0.0683`

The candidate pipeline achieved the lowest Brier score, demonstrating that high discrimination was obtained alongside well-calibrated risk probabilities without calibration degradation.
