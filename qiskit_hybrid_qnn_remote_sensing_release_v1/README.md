# Regime-Bounded Hybrid Quantum Heads Release v1

This package contains source code, clean tables, metadata, and figure assets needed to review and reproduce the reported DIOR low-shot hybrid QNN experiments.

The package intentionally excludes model checkpoints, raw DIOR image files, original DIOR annotations, IDE files, internal work logs, and raw copied experiment directories. DIOR data must be obtained from the original provider and preprocessed locally.

Key files:

- `REPRODUCIBILITY.md`: commands and expected metrics.
- `MANIFEST.md`: file inventory and release policy.
- `requirements.txt`: Python environment pins used by the project.
- `tables/main/`: main manuscript tables and clean summary CSVs.
- `tables/supplementary/`: supplementary data tables S1-S6.
- `figures/main/`: final Fig. 1-Fig. 3 in PNG and PDF.
- `metadata_clean/`: processed metadata and split summaries, without image data.
