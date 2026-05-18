# Release v2 Manifest

## Included

- `README.md`, `MANIFEST.md`, `REPRODUCIBILITY.md`, `ENVIRONMENT.md`, and `GITHUB_UPLOAD_CHECKLIST.md`.
- `requirements.txt`: unified Python dependency specification.
- `manuscript/`: final v3 manuscript source files, figures, and compiled `main.pdf`.
- `supplement/`: final v3 supplement source files, supplementary table source/CSV, and compiled `supplement.pdf`.
- `src/`: current project source for DIOR ROI data handling, CNN/hybrid models, QNN heads, metrics, seeding, and training helpers.
- `scripts/`: selected public scripts for preprocessing, experiment execution, table/figure generation, and environment bootstrap.
- `tests/`: lightweight tests for sample-cap behavior and QNN dependency reporting.
- `figures/main/`: final manuscript Fig. 1-Fig. 3 assets.
- `figures/diagnostic/`: supplementary and diagnostic figure assets used to support the evidence package.
- `tables/main/`: paper-facing main-result, pairwise-test, per-class, and boundary-summary CSV tables.
- `tables/supplementary/`: Supplementary Table S1 in CSV and LaTeX table form.
- `experiment_evidence/`: clean `lowshot_summary_rows.csv`, `lowshot_detail_rows.csv`, and paper-ready CSV tables for reported and boundary regimes.
- `metadata_clean/`: label maps, metadata CSVs, and split summaries for the processed DIOR-derived subsets.
- `configs_clean/experiment_settings_clean.json`: sanitized settings for the release.

## Excluded

- DIOR raw images and original annotations.
- Cropped ROI image files.
- Model checkpoints and weight files (`*.pt`, `*.pth`, `*.ckpt`).
- Raw `shot_*` run directories, training curves, and full per-seed artifact trees.
- LaTeX cache files (`*.aux`, `*.log`, `*.out`, `*.xdv`, etc.).
- IDE files, `.DS_Store`, `__pycache__`, and Python bytecode.
- Chat archives, internal work logs, and local-machine metadata.
- Obsolete BB84, secure-return, NEQR, route2, and legacy quantum-image code paths removed from the current mainline.

## Evidence Policy

The release evidence is the clean CSV payload under `tables/` and `experiment_evidence/`. Historical exploratory runs are retained only where they are explicitly used as boundary context and are never presented as universal quantum-advantage evidence.
