# Section 3: Multi-Level Novelty Analysis

The principal novelty of this work is the formal integration of literature evidence extraction, explicit configuration gating, and executable pipeline verification. We analyze this novelty across three distinct operational levels:

## Level 1 — Methodological Novelty: Evidence-Conditioned Compositional Synthesis
Unlike traditional automated machine learning (AutoML) frameworks that perform unconstrained empirical searches over arbitrary search spaces, our framework synthesizes pipelines strictly from biomedical literature evidence. Literature mechanisms are extracted, audited for provenance authenticity, grounded in a controlled domain taxonomy, and verified for compatibility before being composed into candidate pipelines.

## Level 2 — Governance & Safety Novelty: Strict Provenance Boundary Gating
The framework introduces a formal governance firewall that strictly distinguishes between:
- **`EVIDENCE_BACKED` Primitives**: Components possessing verifiable citations in peer-reviewed biomedical literature (e.g., MissForest/MICE, XGBoost, SMOTE).
- **`EXPLICITLY_CONFIGURED` Primitives**: Components where literature evidence is absent, requiring human-controlled explicit project configuration (e.g., one-hot encoding, binary logistic loss).

The architecture enforces hard safety gates that block execution if unresolved primitives attempt to silently use library defaults or falsely claim literature backing.

## Level 3 — Execution Novelty: End-to-End Constraint-Preserving Materialization
The framework carries literature provenance and safety constraints all the way into executable experimental code. It programmatically verifies:
- Strict 8-variable target isolation firewall
- Patient-level splitting with zero overlap across folds
- Train-only preprocessing fit enforcement
- Cryptographic contract and pipeline hash immutability
- Strict multi-seed reproducibility and compute budget limits
