# Release Manifest

## Included

- `src/`: Python source modules required for datasets, models, QNN heads, metrics, and training helpers.
- `scripts/`: selected training, preprocessing, screening, and figure-generation scripts needed for the reported experiments.
- `requirements.txt` and `ENVIRONMENT.md`: environment specifications.
- `tables/main/`: main paper tables, architecture summary, setting summaries, paired deltas, subset context, and boundary summary.
- `tables/supplementary/`: supplementary data S1-S6 as clean CSV files.
- `figures/main/`: final Fig. 1-Fig. 3 as PNG and PDF.
- `figures/supplementary/`: selected supplementary per-class figure assets when available.
- `metadata_clean/`: processed metadata, split summaries, and label maps for the two paper subsets.
- `configs_clean/`: clean experiment settings without raw copied output directories.
- `manuscript_v5_1.md` and `supplement_v5_1.md`: review-ready text snapshots aligned with this release.

## Excluded

- raw DIOR images and original DIOR annotations;
- model checkpoints;
- copied raw experiment directories;
- IDE files;
- chat logs and internal work logs;
- older manuscript drafts;
- local absolute paths and local-machine metadata.

## Data Policy

DIOR data are not redistributed. Users should obtain DIOR from the original provider, then run the preprocessing command in `REPRODUCIBILITY.md`.

## Evidence Policy

The clean tables in `tables/main/` and `tables/supplementary/` are the release evidence tables. Historical exploratory screens are described only as context and are not treated as final confirmatory evidence.
