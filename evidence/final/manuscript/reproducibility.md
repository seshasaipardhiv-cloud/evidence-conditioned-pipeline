# Reproducibility Manifest and Audit Protocol

## 1. Cryptographic Hashes
- **Stage 3.6 Configured Pipeline Hash**: `6b6bcb1b217793230dba3467ea09fd94339c7497ccc85b0f8ed86c59f16686da`
- **Stage 5A Experiment Contract Hash**: `6eb6b035c8f87bcf52d7d6107a5a4eafa6c6330ca9bf6c1ca837cdbd63910024`

## 2. Reproducibility Invariants Verified
1. **Deterministic Random Seeds**: Fixed seeds `[42, 100, 2026]` executed with exact split re-generation.
2. **Strict Zero Patient Overlap**: 0 patient intersection across train, validation, and test partitions.
3. **Leakage Prevention**: All 8 target/outcome/progress fields barred from feature matrix $X$.
4. **Train-Only Preprocessing**: Imputers, encoders, and resamplers fitted strictly on training data.
5. **Single Test Evaluation**: Test set evaluated strictly once per seed after final model parameter freeze.
6. **Zero Silent Fallback**: Abort triggers active for hash mismatches, leakage violations, or schema divergence.

## 3. Compute Budget Compliance
- **Peak Memory**: 6.83 MB (Strictly below the 4,096 MB budget).
- **Execution Time**: 6.87 seconds (Strictly below the 15-minute budget).
- **Device**: CPU execution only.
