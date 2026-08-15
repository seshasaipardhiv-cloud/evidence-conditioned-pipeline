# 7. Novelty and Research Contributions

## 7.1 Multi-Level Novelty Framework
The principal methodological contribution is the integration of literature evidence synthesis, explicit configuration gating, and constraint-preserving materialization. We analyze this novelty across three operational levels:

1. **Level 1 — Methodological Novelty (Evidence-Conditioned Synthesis)**: Shifting pipeline design from unconstrained AutoML empirical search to a structured synthesis workflow grounded in biomedical literature citations and domain taxonomies.
2. **Level 2 — Governance & Safety Novelty (Strict Provenance Gating)**: Establishing a formal firewall that strictly segregates `EVIDENCE_BACKED` primitives from `EXPLICITLY_CONFIGURED` inputs, barring silent ML library defaults from entering the pipeline.
3. **Level 3 — Execution Novelty (Constraint-Preserving Materialization)**: Carrying literature provenance and safety constraints directly into executable code through 10 readiness verification gates, a target isolation firewall, and immutable contract hashing.

*(Refer to Figure 8 for the formal scientific claim boundary matrix).*

## 7.2 Concrete Research Contributions
We formalize six concrete contributions, distinguishing what was architected from what was empirically demonstrated:

- **C1: Evidence-Conditioned Synthesis Framework** (*What was built*): An end-to-end framework translating biomedical literature citations into verified, executable machine learning pipelines.
- **C2: Formal Provenance and Evidence Boundary** (*What was built*): A cryptographic provenance ledger separating literature evidence from explicit project configurations.
- **C3: Controlled Mechanism and Primitive Resolution** (*What was built*): A human-gated resolution gate blocking execution when evidence is absent, barring silent defaults.
- **C4: Safety Gates Against Silent Defaults and Leakage** (*What was built*): 10 verification gates enforcing train-only preprocessing, zero patient overlap, and target isolation.
- **C5: Reproducible Executable Materialization** (*What was built*): An immutable experiment contract freezing seeds, splits, mappings, and compute constraints into verifiable hashes.
- **C6: Empirical Demonstration on HANCOCK Cohort** (*What was demonstrated*): Feasibility demonstration achieving high internal discrimination (`0.9751 ± 0.0114` ROC-AUC) and calibration (`0.0175` Brier score), alongside empirical ablation insights under operational tabular execution.
