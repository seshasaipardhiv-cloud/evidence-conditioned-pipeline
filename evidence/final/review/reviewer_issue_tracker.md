# Reviewer Issue Tracker and Audit Ledger

**Manuscript Title:** Evidence-Conditioned Compositional Pipeline Synthesis: A Provenance-Aware Framework for Reproducible Clinical Machine Learning  
**Tracking Status:** `PEER_REVIEW_TRACKER_INITIALIZED`  
**Total Pre-Populated Defense Items:** 25  

---

## Structured Issue Tracking Table

| Reviewer | Comment ID | Exact Reviewer Comment | Manuscript Section | Requested Change | Scientific Impact | Evidence Required | Response Drafted | Manuscript Change Made | Verification Status |
| :--- | :---: | :--- | :---: | :--- | :---: | :--- | :---: | :---: | :---: |
| Hostile Reviewer | `Q01` | Why did the pipeline register MissForest/MICE when executor_stage5b.py executed univariate SimpleImputer(strategy='median')? | Section | Textual Clarification | Neutral / Clarification | `Q01` Evidence Ledger | Yes | Reconciled in Stage 6H | **RESOLVED** |
| Hostile Reviewer | `Q02` | Why are cross-attention and average ensembling in the pipeline specification if they were not executed in the HANCOCK benchmark? | Section | Textual Clarification | Neutral / Clarification | `Q02` Evidence Ledger | Yes | Reconciled in Stage 6H | **RESOLVED** |
| Hostile Reviewer | `Q03` | Is progress_1 (binary disease progression) an outcome proxy for recurrence? | Section | Textual Clarification | Neutral / Clarification | `Q03` Evidence Ledger | Yes | Reconciled in Stage 6H | **RESOLVED** |
| Hostile Reviewer | `Q04` | At what exact clinical timepoint does this model make predictions? | Section | Textual Clarification | Neutral / Clarification | `Q04` Evidence Ledger | Yes | Reconciled in Stage 6H | **RESOLVED** |
| Hostile Reviewer | `Q05` | The Simple MLP baseline used max_iter=10. Is this an intentionally weak strawman? | Section | Textual Clarification | Neutral / Clarification | `Q05` Evidence Ledger | Yes | Reconciled in Stage 6H | **RESOLVED** |
| Hostile Reviewer | `Q06` | Is a +0.0047 ROC-AUC gain over Default XGBoost practically meaningful? | Abstract, | Textual Clarification | Neutral / Clarification | `Q06` Evidence Ledger | Yes | Reconciled in Stage 6H | **RESOLVED** |
| Hostile Reviewer | `Q07` | The candidate pipeline lost to Default XGBoost on Seed 100 (0.9609 vs 0.9643). Does this undermine the claim of baseline superiority? | Table | Textual Clarification | Neutral / Clarification | `Q07` Evidence Ledger | Yes | Reconciled in Stage 6H | **RESOLVED** |
| Hostile Reviewer | `Q08` | Why did Ablation B (Without SMOTE: 0.9773) and Ablation D (Ordinal: 0.9784) outperform the Full Candidate (0.9751)? | Section | Textual Clarification | Neutral / Clarification | `Q08` Evidence Ledger | Yes | Reconciled in Stage 6H | **RESOLVED** |
| Hostile Reviewer | `Q09` | Is n=3 random seeds sufficient for statistical significance? | Section | Textual Clarification | Neutral / Clarification | `Q09` Evidence Ledger | Yes | Reconciled in Stage 6H | **RESOLVED** |
| Hostile Reviewer | `Q10` | Can these findings generalize to external medical centers? | Section | Textual Clarification | Neutral / Clarification | `Q10` Evidence Ledger | Yes | Reconciled in Stage 6H | **RESOLVED** |
| Hostile Reviewer | `Q11` | Is this pipeline ready for clinical deployment in oncology clinics? | Section | Textual Clarification | Neutral / Clarification | `Q11` Evidence Ledger | Yes | Reconciled in Stage 6H | **RESOLVED** |
| Hostile Reviewer | `Q12` | How does this differ from standard AutoML tools like TPOT or Auto-sklearn? | Section | Textual Clarification | Neutral / Clarification | `Q12` Evidence Ledger | Yes | Reconciled in Stage 6H | **RESOLVED** |
| Hostile Reviewer | `Q13` | Can a 4-paper literature corpus demonstrate generalizable literature synthesis? | Section | Textual Clarification | Neutral / Clarification | `Q13` Evidence Ledger | Yes | Reconciled in Stage 6H | **RESOLVED** |
| Hostile Reviewer | `Q14` | Why were one-hot encoding and binary logistic loss configured manually rather than extracted from literature? | Section | Textual Clarification | Neutral / Clarification | `Q14` Evidence Ledger | Yes | Reconciled in Stage 6H | **RESOLVED** |
| Hostile Reviewer | `Q15` | If omitting SMOTE improved ROC-AUC (0.9773), why was SMOTE selected in the candidate pipeline? | Section | Textual Clarification | Neutral / Clarification | `Q15` Evidence Ledger | Yes | Reconciled in Stage 6H | **RESOLVED** |
| Hostile Reviewer | `Q16` | Were imputers or encoders fitted on test data? | Section | Textual Clarification | Neutral / Clarification | `Q16` Evidence Ledger | Yes | Reconciled in Stage 6H | **RESOLVED** |
| Hostile Reviewer | `Q17` | Was there any patient overlap between training, validation, and test splits? | Section | Textual Clarification | Neutral / Clarification | `Q17` Evidence Ledger | Yes | Reconciled in Stage 6H | **RESOLVED** |
| Hostile Reviewer | `Q18` | Does a Brier score of 0.0175 prove clinical calibration? | Section | Textual Clarification | Neutral / Clarification | `Q18` Evidence Ledger | Yes | Reconciled in Stage 6H | **RESOLVED** |
| Hostile Reviewer | `Q19` | Can another researcher independently reproduce these exact results? | Section | Textual Clarification | Neutral / Clarification | `Q19` Evidence Ledger | Yes | Reconciled in Stage 6H | **RESOLVED** |
| Hostile Reviewer | `Q20` | Why did Random Forest (0.9698) and Default XGBoost (0.9704) perform so close to Candidate (0.9751)? | Section | Textual Clarification | Neutral / Clarification | `Q20` Evidence Ledger | Yes | Reconciled in Stage 6H | **RESOLVED** |
| Hostile Reviewer | `Q21` | Does the system assess the risk of bias in extracted literature papers? | Section | Textual Clarification | Neutral / Clarification | `Q21` Evidence Ledger | Yes | Reconciled in Stage 6H | **RESOLVED** |
| Hostile Reviewer | `Q22` | What missingness rate was present in the HANCOCK tabular data? | Section | Textual Clarification | Neutral / Clarification | `Q22` Evidence Ledger | Yes | Reconciled in Stage 6H | **RESOLVED** |
| Hostile Reviewer | `Q23` | What were the compute requirements for pipeline synthesis and execution? | Section | Textual Clarification | Neutral / Clarification | `Q23` Evidence Ledger | Yes | Reconciled in Stage 6H | **RESOLVED** |
| Hostile Reviewer | `Q24` | Does this paper follow established clinical ML reporting guidelines? | Section | Textual Clarification | Neutral / Clarification | `Q24` Evidence Ledger | Yes | Reconciled in Stage 6H | **RESOLVED** |
| Hostile Reviewer | `Q25` | What is the primary takeaway if the candidate model is just an XGBoost classifier? | Abstract, | Textual Clarification | Neutral / Clarification | `Q25` Evidence Ledger | Yes | Reconciled in Stage 6H | **RESOLVED** |

---

## Template for Incoming Post-Submission Peer Review Rounds

When official peer reviews are returned by JBI / Elsevier Editorial Manager, add new entries using this format:

| Reviewer | Comment ID | Exact Reviewer Comment | Manuscript Section | Requested Change | Scientific Impact | Evidence Required | Response Drafted | Manuscript Change Made | Verification Status |
| :--- | :---: | :--- | :---: | :--- | :---: | :--- | :---: | :---: | :---: |
| Reviewer 1 | `R1-C01` | *Verbatim comment* | Section 3.1 | Text revision | Minor | Source hash / experiment | Yes/No | Pending | `IN_PROGRESS` |
| Reviewer 2 | `R2-C01` | *Verbatim comment* | Section 5.1 | Additional analysis | Moderate | Ablation / statistical report | Yes/No | Pending | `IN_PROGRESS` |
