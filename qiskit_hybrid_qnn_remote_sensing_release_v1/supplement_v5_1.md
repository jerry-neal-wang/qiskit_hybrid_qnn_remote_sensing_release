# Supplementary Material for: Regime-Bounded Hybrid Quantum Heads for Low-Shot Remote-Sensing ROI Classification

This supplementary file reports seed-level deltas, subset-screening context, and reproducibility references for the main manuscript. Full CSV files are provided as named supplementary data files in the release package.

## Supplementary Table S1. Paired seed-level delta summary

The full per-seed data are stored in `main_pairwise_seed_deltas.csv`.

| Setting | Comparison | Paired seeds | Mean delta | Median delta | Better/Worse/Tie |
| --- | --- | --- | --- | --- | --- |
| transport_logistics4 resnet18 shot16 bblr0.1 | QNN No-Ent vs control | 7 | -0.0011 | -0.0029 | 2/5/0 |
| transport_logistics4 resnet18 shot16 bblr0.1 | QNN ZZ+RealAmp vs control | 7 | +0.0051 | -0.0000 | 3/4/0 |
| transport_logistics4 small_cnn shot16 bblr0.1 | QNN No-Ent vs control | 7 | +0.1976 | +0.1256 | 6/1/0 |
| transport_logistics4 small_cnn shot16 bblr0.1 | QNN ZZ+RealAmp vs control | 7 | +0.1491 | +0.1212 | 7/0/0 |
| transport_logistics4 small_cnn shot32 bblr0.1 | QNN No-Ent vs control | 7 | +0.0094 | -0.0031 | 3/4/0 |
| transport_logistics4 small_cnn shot32 bblr0.1 | QNN ZZ+RealAmp vs control | 7 | +0.0018 | +0.0223 | 4/3/0 |
| transport_logistics4 small_cnn shot64 bblr0.1 | QNN No-Ent vs control | 7 | -0.0021 | +0.0001 | 4/3/0 |
| transport_logistics4 small_cnn shot64 bblr0.1 | QNN ZZ+RealAmp vs control | 7 | +0.0001 | -0.0002 | 3/4/0 |
| urban_structural4 small_cnn shot16 bblr0.1 | QNN No-Ent vs control | 7 | -0.0063 | -0.0087 | 2/5/0 |
| urban_structural4 small_cnn shot16 bblr0.1 | QNN ZZ+RealAmp vs control | 7 | -0.0583 | -0.0604 | 2/5/0 |

## Supplementary Table S2. Subset-screening and boundary-selection chronology

`N/A` indicates that the non-entangled QNN was not run for the exploratory legacy screen. These legacy screens are not used as final confirmatory evidence.

| Subset candidate | Classes | Stage | Seeds | Shot | Control macro-F1 | No-Ent macro-F1 | ZZ macro-F1 | Decision | Role in paper |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| transport_logistics4_small_cnn_bblr01 | airplane;ship;vehicle;harbor | final 7-seed main confirmation | 42-48 | 16 | 0.3055 | 0.5031 | 0.4545 | keep as main positive low-shot window | main result |
| transport_logistics4_small_cnn_bblr01 | airplane;ship;vehicle;harbor | label-count boundary | 42-48 | 32 | 0.6034 | 0.6128 | 0.6053 | retain as negative/contracted boundary | larger-shot stress test |
| transport_logistics4_small_cnn_bblr01 | airplane;ship;vehicle;harbor | label-count boundary | 42-48 | 64 | 0.6705 | 0.6685 | 0.6707 | retain as null boundary | larger-shot stress test |
| transport_logistics4_resnet18_bblr01 | airplane;ship;vehicle;harbor | strong-backbone boundary | 42-48 | 16 | 0.8545 | 0.8534 | 0.8596 | retain as quantum-control margin compression boundary | ResNet18 transfer stress test |
| urban_structural4_small_cnn_bblr01 | vehicle;bridge;harbor;storagetank | cross-subset boundary | 42-48 | 16 | 0.4655 | 0.4593 | 0.4073 | retain as non-reproduced cross-subset boundary | negative subset-composition stress test |
| transport_logistics4_small_cnn_freezeonly | airplane;ship;vehicle;harbor | legacy 3-seed optimization screen | 42-44 | 16 | 0.2945 | N/A | 0.2207 | exclude from main public reproduction bundle unless explicitly documenting exploratory optimization failures | internal exploratory screen only |
| transport_logistics4_resnet18_legacy_screen | airplane;ship;vehicle;harbor | legacy 3-seed ResNet18 screen | 42-44 | 16 | 0.8412 | N/A | 0.8426 | superseded by 7-seed ResNet18 boundary | internal exploratory screen only |

## Supplementary Data S3-S6

- Supplementary Data S3: full per-seed detail table is provided as `S3_per_seed_detail_tables.csv`.
- Supplementary Data S4: full pairwise Wilcoxon table is provided as `S4_pairwise_wilcoxon_tables.csv`.
- Supplementary Data S5: full bootstrap confidence-interval table is provided as `S5_bootstrap_delta_ci.csv`.
- Supplementary Data S6: full per-class metrics table for the main `transport_logistics4` 16-shot setting is provided as `S6_per_class_metrics_transport_logistics4_shot16.csv`.

## Reproducibility-package note

Raw working artifacts should not be published directly because they may contain historical or local path traces. The accompanying sanitized release bundle contains clean summary CSVs, per-seed detail CSVs, source code, environment files, figure-generation scripts, and a reproducibility README.
