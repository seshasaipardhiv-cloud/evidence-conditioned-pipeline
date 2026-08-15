# Journal Submission Record and Lifecycle Ledger

**Manuscript Title:** Evidence-Conditioned Compositional Pipeline Synthesis: A Provenance-Aware Framework for Reproducible Clinical Machine Learning  
**Short Running Title:** Evidence-Conditioned Pipeline Synthesis  
**Target Journal:** *Journal of Biomedical Informatics* (Elsevier)  
**Submission Portal:** Elsevier Editorial Manager (https://www.editorialmanager.com/jbi/)  

---

## 1. Submission Identity and Cryptographic Hashes

- **Primary Target Journal:** *Journal of Biomedical Informatics* (JBI)
- **Manuscript Tracking ID:** `MANUSCRIPT_ID_PENDING_SUBMISSION` *(To be populated upon Editorial Manager confirmation)*
- **Submission Date:** `SUBMISSION_DATE_PENDING` *(To be populated upon clicking Submit)*
- **Submitted Manuscript File:** `final_research_paper.pdf` (15 pages, 3,987 words)
- **Submitted PDF SHA-256 Checksum:** `0202b17a44577c1141392f8a169a9debc987df3a4e86e3686f5fbc95312c0b2b`
- **Source Markdown File:** `final_research_paper.md`
- **Submission Package Root:** `evidence/final/submission/`
- **Submission Manifest:** `evidence/final/submission/submission_manifest.json`

---

## 2. Author and Institutional Registry

- **Submitting Author:** `SUBMITTING_AUTHOR_PLACEHOLDER`
- **Corresponding Author:** `CORRESPONDING_AUTHOR_PLACEHOLDER` (`CORRESPONDING_EMAIL_PLACEHOLDER@institution.edu`)
- **Co-Authors:**
  1. `AUTHOR_1_PLACEHOLDER` (Affiliation 1, ORCID: `0000-0000-0000-0000`)
  2. `AUTHOR_2_PLACEHOLDER` (Affiliation 1, ORCID: `0000-0000-0000-0000`)
  3. `AUTHOR_3_PLACEHOLDER` (Affiliation 2, ORCID: `0000-0000-0000-0000`)
- **Institutional Affiliations:**
  - `AFFILIATION_1_PLACEHOLDER`: Department of Biomedical Informatics, Institution Placeholder
  - `AFFILIATION_2_PLACEHOLDER`: Division of Clinical Oncology / Data Science, Institution Placeholder

---

## 3. Editorial Lifecycle and Review Milestones

| Stage / Milestone | Target Date | Actual Date | Status | Associated Artifact / Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Initial Portal Submission** | *Pending* | *Pending* | `READY_FOR_UPLOAD` | `final_research_paper.pdf` (`0202b17a44577c11...`) |
| **Editor-in-Chief Assignment** | *Pending* | *Pending* | `PENDING` | Editorial Manager |
| **Under Review (First Round)** | *Pending* | *Pending* | `PENDING` | JBI Reviewer Pool |
| **First Editorial Decision** | *Pending* | *Pending* | `PENDING` | Decision Letter (Accept / Minor / Major / Reject) |
| **Revision 1 Submission** | *Pending* | *Pending* | `PENDING` | `evidence/final/review/review_response_template.md` |
| **Final Editorial Acceptance** | *Pending* | *Pending* | `PENDING` | Production Proof |

---

## 4. Authoritative Empirical Invariants of Record

- **Candidate Pipeline (Actual Executed Path):**
  - **Mean Test ROC-AUC:** `0.9751 ± 0.0114` (Seed 42: `0.9888`, Seed 100: `0.9609`, Seed 2026: `0.9756`)
  - **Mean Test Brier Score:** `0.0175`
- **Default XGBoost Baseline:** `0.9704 ± 0.0059` (Candidate Margin: `+0.0047`, modest)
- **Random Forest:** `0.9698` | **Logistic Regression:** `0.9645` | **Simple MLP (Shallow Baseline):** `0.9405`
- **Ablations:** No SMOTE `0.9773`, Ordinal `0.9784`, Mean Imp `0.9767`, Default XGB `0.9686`
- **Configured Pipeline SHA-256:** `6b6bcb1b217793230dba3467ea09fd94339c7497ccc85b0f8ed86c59f16686da`
- **Frozen Experiment Contract SHA-256:** `6eb6b035c8f87bcf52d7d6107a5a4eafa6c6330ca9bf6c1ca837cdbd63910024`
