# Reproducibility Guide

## Environment

Recommended interpreter: Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the import check:

```bash
bash scripts/bootstrap_env.sh
```

## Data Preparation

DIOR images and original annotations are not redistributed in this package. Download DIOR from the original provider and place the raw dataset under `dataset_DIOR/` using the layout expected by `scripts/build_dior_multiclass_stratified.py`.

Build processed ROI subsets:

```bash
python scripts/build_dior_multiclass_stratified.py \
  --dataset-root dataset_DIOR \
  --output-root data/processed/subset_candidates \
  --classes airplane airport baseballfield bridge chimney dam Expressway-Service-area Expressway-toll-station golffield groundtrackfield harbor overpass ship storagetank tenniscourt trainstation vehicle windmill
```

If needed, screen candidate subsets:

```bash
python scripts/screen_dior_subset_candidates.py \
  --dataset-root dataset_DIOR \
  --processed-data-root data/processed/subset_candidates \
  --output-root artifacts/dior_subset_candidate_screening \
  --candidates transport_logistics4 urban_structural4 \
  --shots 16 \
  --seeds 42 43 44
```

## Main Small-CNN Experiments

Main 16-shot transport/logistics window:

```bash
python scripts/run_dior_multiclass_lowshot.py \
  --dataset-root data/processed/subset_candidates/transport_logistics4 \
  --output-dir artifacts/dior_subset_candidate_screening/transport_logistics4_shot16_bblr01_7seed \
  --shots 16 \
  --seeds 42 43 44 45 46 47 48 \
  --backbone-type small_cnn \
  --backbone-feature-dim 64 \
  --backbone-lr-scale 0.1 \
  --include-classical-control \
  --circuits no_entanglement zz_real_amplitudes \
  --full-eval-at-end
```

Label-count boundaries:

```bash
python scripts/run_dior_multiclass_lowshot.py \
  --dataset-root data/processed/subset_candidates/transport_logistics4 \
  --output-dir artifacts/dior_subset_candidate_screening/transport_logistics4_shot32_bblr01_7seed \
  --shots 32 \
  --seeds 42 43 44 45 46 47 48 \
  --backbone-type small_cnn \
  --backbone-feature-dim 64 \
  --backbone-lr-scale 0.1 \
  --include-classical-control \
  --circuits no_entanglement zz_real_amplitudes \
  --full-eval-at-end

python scripts/run_dior_multiclass_lowshot.py \
  --dataset-root data/processed/subset_candidates/transport_logistics4 \
  --output-dir artifacts/dior_subset_candidate_screening/transport_logistics4_shot64_bblr01_7seed \
  --shots 64 \
  --seeds 42 43 44 45 46 47 48 \
  --backbone-type small_cnn \
  --backbone-feature-dim 64 \
  --backbone-lr-scale 0.1 \
  --include-classical-control \
  --circuits no_entanglement zz_real_amplitudes \
  --full-eval-at-end
```

## Strong-Backbone Boundary

```bash
python scripts/run_dior_multiclass_lowshot.py \
  --dataset-root data/processed/subset_candidates/transport_logistics4 \
  --output-dir artifacts/transport_logistics4_resnet18_shot16_bblr01_7seed \
  --shots 16 \
  --seeds 42 43 44 45 46 47 48 \
  --backbone-type resnet18 \
  --backbone-feature-dim 128 \
  --pretrained-backbone \
  --backbone-lr-scale 0.1 \
  --include-classical-control \
  --circuits no_entanglement zz_real_amplitudes \
  --full-eval-at-end
```

## Cross-Subset Boundary

```bash
python scripts/run_dior_multiclass_lowshot.py \
  --dataset-root data/processed/subset_candidates/urban_structural4 \
  --output-dir artifacts/dior_subset_candidate_screening/urban_structural4_shot16_bblr01_7seed \
  --shots 16 \
  --seeds 42 43 44 45 46 47 48 \
  --backbone-type small_cnn \
  --backbone-feature-dim 64 \
  --backbone-lr-scale 0.1 \
  --include-classical-control \
  --circuits no_entanglement zz_real_amplitudes \
  --full-eval-at-end
```

## Figure and Table Generation

```bash
python scripts/generate_lowregime_per_class_analysis.py \
  --experiment-root artifacts/dior_subset_candidate_screening/transport_logistics4_shot16_bblr01_7seed \
  --shots 16

python scripts/generate_sci_composite_figures.py \
  --output-dir artifacts/paper_composite_figures
```

The release already includes the final clean CSVs and final Fig. 1-Fig. 3 under `tables/` and `figures/main/`.

## Expected Summary Metrics

Main transport/logistics, small CNN, 16 shots:

- Hybrid classical control: macro-F1 `0.3055 +/- 0.1729`
- QNN No-Ent: macro-F1 `0.5031 +/- 0.1421`
- QNN ZZ+RealAmp: macro-F1 `0.4545 +/- 0.1735`

Boundary checks:

- 32-shot No-Ent delta over control: `+0.0094`, Wilcoxon `p=0.8125`
- 64-shot No-Ent delta over control: `-0.0021`, Wilcoxon `p=0.8125`
- ResNet18 16-shot No-Ent delta over control: `-0.0011`, Wilcoxon `p=0.6875`
- Urban subset 16-shot No-Ent delta over control: `-0.0063`, Wilcoxon `p=0.6875`
