# 5. Experimental Results

## 5.1 Primary Predictive Performance
Table 1 reports the test-set performance metrics averaged across random seeds `[42, 100, 2026]`.

**Table 1: Test-Set Performance Across Candidate Pipeline and Baseline Models (Mean ± Std)**
| Pipeline / Model | Test ROC-AUC | $\Delta$ ROC-AUC vs Baseline | Test PR-AUC | Test F1 Score | Test Accuracy | Test Precision | Test Recall | Test Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Candidate Pipeline (Actual Executed Path)** | **0.9751 ± 0.0114** | — | **0.9679** | **0.9611** | **0.9825** | **0.9801** | **0.9429** | **0.0175** |
| **Default XGBoost Baseline** | 0.9704 ± 0.0059 | +0.0047 (+0.48%) | 0.9665 | 0.9611 | 0.9825 | 0.9801 | 0.9429 | 0.0180 |
| **Random Forest Baseline** | 0.9698 ± 0.0065 | +0.0053 (+0.55%) | 0.9494 | 0.9611 | 0.9825 | 0.9801 | 0.9429 | 0.0207 |
| **Logistic Regression Baseline** | 0.9645 ± 0.0070 | +0.0106 (+1.10%) | 0.9536 | 0.9558 | 0.9803 | 0.9798 | 0.9333 | 0.0201 |
| **Simple MLP Baseline (Minimal Reference)** | 0.9405 ± 0.0192 | +0.0346 (+3.68%) | 0.9060 | 0.9003 | 0.9561 | 0.9380 | 0.8667 | 0.0683 |

The candidate pipeline achieved a mean test ROC-AUC of `0.9751 ± 0.0114`. The primary performance margin over Default XGBoost (`+0.0047` ROC-AUC) is **modest**. Tree ensemble architectures demonstrated high baseline discrimination on this structured feature set. The candidate exceeded the minimal shallow MLP reference baseline under the frozen experimental contract; this comparison should not be interpreted as evidence of superiority over optimized neural architectures.

*(Refer to Figure 2 for the baseline comparison bar chart and Figure 6 for the multi-metric performance profile).*

## 5.2 Multi-Seed Robustness and Margin Dynamics
Table 2 breaks down performance across individual random seeds.

**Table 2: Per-Seed Performance Comparison (Candidate vs. Default XGBoost)**
| Random Seed | Candidate ROC-AUC | Default XGBoost ROC-AUC | Margin ($\Delta$) | Candidate Outcome |
| :---: | :---: | :---: | :---: | :---: |
| **Seed 42** | `0.9888` | `0.9783` | **+0.0105** | **Candidate Won** |
| **Seed 100** | `0.9609` | `0.9643` | **-0.0034** | Default XGBoost Won |
| **Seed 2026** | `0.9756` | `0.9685` | **+0.0071** | **Candidate Won** |

The candidate pipeline won on 2 out of 3 seeds (66.7% win rate). While achieving a higher mean score, the candidate did **not** universally dominate Default XGBoost across all folds. With $n=3$ seeds, this margin is not statistically significant.

*(Refer to Figure 3 for the per-seed robustness comparison).*

## 5.3 Controlled Component Ablation Analysis
Table 3 summarizes the ablation results across identical patient splits and seeds.

**Table 3: Controlled Component Ablation Results**
| Configuration | Changed Primitive | Mean ROC-AUC | Std ROC-AUC | Mean F1 | Mean Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Full Candidate Pipeline** | *None (Reference)* | **0.9751** | 0.0114 | 0.9611 | 0.0175 |
| **Ablation B (Without SMOTE)** | `imbalance_handling` | **0.9773** | 0.0095 | 0.9611 | 0.0177 |
| **Ablation C (Mean Imputation)** | `missing_value_handling` | **0.9767** | 0.0098 | 0.9611 | 0.0174 |
| **Ablation D (Ordinal Encoding)** | `categorical_encoding` | **0.9784** | 0.0111 | 0.9611 | 0.0177 |
| **Ablation E (Default XGBoost)** | `base_learner_config` | **0.9686** | 0.0063 | 0.9611 | 0.0181 |

*Crucial Finding*: Omitting SMOTE (`0.9773`) or employing Ordinal Encoding (`0.9784`) achieved marginally higher ROC-AUC on this specific dataset. The evidence-conditioned framework does not claim to identify the empirically optimal configuration for every dataset. Rather, it constrains pipeline composition to traceable evidence and explicit configuration, making the provenance and rationale of each choice auditable.

*(Refer to Figure 4 for the ablation comparison).*

## 5.4 Probability Calibration
The Candidate Pipeline achieved the lowest Brier score (`0.0175`), outperforming Default XGBoost (`0.0180`), Logistic Regression (`0.0201`), Random Forest (`0.0207`), and Simple MLP (`0.0683`). Lower Brier scores indicate lower probability estimation error, confirming that high discrimination was achieved alongside well-calibrated risk probabilities.

*(Refer to Figure 5 for the calibration comparison).*
