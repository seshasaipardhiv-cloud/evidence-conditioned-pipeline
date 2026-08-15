# Scientific Claim Boundary Matrix

## 1. Formal Scientific Claim Ledger

| Claim # | Statement | Status | Supporting / Refuting Evidence |
| :---: | :--- | :---: | :--- |
| **CLAIM 1** | Pipeline architecture is synthesized strictly from evidence-conditioned literature mechanisms and verified explicit configurations. | **`SUPPORTED`** | Verified provenance ledger and explicit configuration gate without silent defaults. |
| **CLAIM 2** | Pipeline components maintain traceable, cryptographically verified provenance from PubMed citations or explicit configuration. | **`SUPPORTED`** | End-to-end hash audit in `stage3_6_provenance_ledger.json` and `stage4_rematerialized_pipeline.json`. |
| **CLAIM 3** | Pipeline strictly avoids arbitrary ML library defaults and requires human-controlled explicit configuration when evidence is absent. | **`SUPPORTED`** | Stage 2F-4 and Stage 3.5 gates blocked unresolved components until explicit project configuration was provided. |
| **CLAIM 4** | Experimental execution protocol is deterministic and strictly reproducible under the tested protocol. | **`SUPPORTED`** | Multi-seed execution with zero patient overlap, locked contract hashes, and 100% test pass rate. |
| **CLAIM 5** | Candidate pipeline achieves high internal discriminative performance on the retrospective HANCOCK clinical cohort. | **`SUPPORTED`** | Mean test ROC-AUC of `0.9751 ± 0.0114` across seeds 42, 100, and 2026. |
| **CLAIM 6** | Candidate pipeline unconditionally outperforms all baseline models across all seeds. | **`PARTIALLY_SUPPORTED`** | Candidate achieved higher mean ROC-AUC (0.9751 vs 0.9704), but lost on Seed 100 (0.9609 vs 0.9643). |
| **CLAIM 7** | Candidate pipeline consistently dominates default XGBoost across every test fold. | **`NOT_SUPPORTED`** | Candidate lost to Default XGBoost on Seed 100 (-0.0034 delta). |
| **CLAIM 8** | Observed predictive performance improvement over default XGBoost is statistically significant. | **`NOT_SUPPORTED`** | Sample size $n=3$ seeds is underpowered for inferential claims; hypothesis testing was not performed; delta is modest (+0.0047). |
| **CLAIM 9** | Synthesized pipeline demonstrates generalizable clinical efficacy. | **`NOT_SUPPORTED`** | Evaluation is purely single-center retrospective internal testing. External validation has not been performed. |
| **CLAIM 10** | Pipeline is clinically deployable for recurrence risk assessment. | **`NOT_SUPPORTED`** | Clinical safety, prospective trials, multi-center calibration, and decision-curve analysis remain unestablished. |

## 2. Strongest Defensible Research Claim
> *"Evidence-conditioned pipeline synthesis provides a rigorous, traceable, and reproducible methodology for constructing valid machine learning pipelines from biomedical literature without unauthorized defaults or target leakage, yielding strong internal discrimination and calibration."*
