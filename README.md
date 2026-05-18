# Regime-Bounded Hybrid Quantum Heads Release v2

This release is the GitHub-facing research package aligned with the v3 GRSL submission-ready manuscript:

`Regime-Bounded Metrics for Qiskit Hybrid Quantum Heads in Low-Shot Remote Sensing ROI Classification`

It contains the paper source/PDF, current cleaned project code, processed metadata, clean CSV evidence, figure-generation scripts, and reproducibility instructions for the DIOR low-shot hybrid QNN study.

## Version Relationship

- `GRSL_v3_submission_ready_20260518/` is the final manuscript delivery package.
- `release/release_v2/` is the corresponding GitHub/review package derived from that v3 content and the current single-mainline codebase.
- Earlier release material is superseded by this package for the submitted paper.

Recommended public upload target:

```text
https://github.com/jerry-neal-wang/qiskit_hybrid_qnn_remote_sensing_release
```

## Main Contents

- `manuscript/`: v3 main manuscript LaTeX source and compiled PDF.
- `supplement/`: v3 supplementary LaTeX source, compiled PDF, and Supplementary Table S1.
- `src/`: current source modules for datasets, models, QNN heads, metrics, and training helpers.
- `scripts/`: public preprocessing, training, analysis, figure, and environment scripts.
- `tests/`: lightweight package tests for current code contracts.
- `figures/`: final main figures and diagnostic/supplementary figure assets.
- `tables/`: paper-facing clean CSV tables.
- `experiment_evidence/`: per-experiment summary/detail CSVs only; raw run folders and checkpoints are excluded.
- `metadata_clean/`: processed metadata, split summaries, and label maps; image files are excluded.
- `configs_clean/`: sanitized experiment settings for the reported regimes.

## Data Policy

DIOR images and original annotations are not redistributed. Obtain DIOR from the original provider, place it under `dataset_DIOR/`, then run the preprocessing commands in `REPRODUCIBILITY.md`.

## Exclusions

This package intentionally excludes raw DIOR images, original DIOR annotations, model checkpoints, raw `shot_*` run directories, IDE files, chat/work logs, LaTeX build caches, and local absolute paths.
