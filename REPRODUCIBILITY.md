# Reproducibility Guide

This guide documents the commands corresponding to the submitted-paper evidence. The release already includes clean CSV evidence and final figures; rerunning the full matrix requires DIOR data and substantial CPU/GPU time because Qiskit hybrid branches are simulated.

## 1. Environment

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Optional import check:

```bash
bash scripts/bootstrap_env.sh
```

## 2. Data Preparation

DIOR raw images and original annotations are not included. After obtaining DIOR, place it under `dataset_DIOR/` with the layout expected by `scripts/build_dior_multiclass_stratified.py`.

Build the main `transport_logistics4` subset:

```bash
.venv/bin/python scripts/build_dior_multiclass_stratified.py \
  --dataset-root dataset_DIOR \
  --output-dir data/processed/subset_candidates/transport_logistics4 \
  --classes airplane ship vehicle harbor
```

Build the cross-subset boundary dataset:

```bash
.venv/bin/python scripts/build_dior_multiclass_stratified.py \
  --dataset-root dataset_DIOR \
  --output-dir data/processed/subset_candidates/urban_structural4 \
  --classes vehicle bridge harbor storagetank
```

## 3. Main 16-Shot Experiment

```bash
.venv/bin/python scripts/run_dior_multiclass_lowshot.py \
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

## 4. Boundary Experiments

Shot-count boundaries:

```bash
.venv/bin/python scripts/run_dior_multiclass_lowshot.py \
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

.venv/bin/python scripts/run_dior_multiclass_lowshot.py \
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

Strong-backbone boundary:

```bash
.venv/bin/python scripts/run_dior_multiclass_lowshot.py \
  --dataset-root data/processed/subset_candidates/transport_logistics4 \
  --output-dir artifacts/transport_logistics4_resnet18_shot16_bblr01_7seed \
  --shots 16 \
  --seeds 42 43 44 45 46 47 48 \
  --baseline-epochs 8 \
  --hybrid-epochs 10 \
  --baseline-batch-size 16 \
  --hybrid-batch-size 16 \
  --image-size 96 \
  --backbone-type resnet18 \
  --backbone-feature-dim 128 \
  --pretrained-backbone \
  --input-normalization imagenet \
  --backbone-lr-scale 0.1 \
  --include-classical-control \
  --circuits no_entanglement zz_real_amplitudes \
  --final-val-max-samples-per-class 256 \
  --final-test-max-samples-per-class 256 \
  --full-eval-at-end
```

Cross-subset boundary:

```bash
.venv/bin/python scripts/run_dior_multiclass_lowshot.py \
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

## 5. Tables and Figures

Generate main per-experiment tables/figures:

```bash
.venv/bin/python scripts/generate_dior_lowregime_figures.py \
  --experiment-root artifacts/dior_subset_candidate_screening/transport_logistics4_shot16_bblr01_7seed
```

Generate per-class analysis:

```bash
.venv/bin/python scripts/generate_lowregime_per_class_analysis.py \
  --experiment-root artifacts/dior_subset_candidate_screening/transport_logistics4_shot16_bblr01_7seed \
  --shots 16
```

Generate the regime-boundary figure assets used by the v3 manuscript package:

```bash
.venv/bin/python scripts/generate_regime_boundary_evidence_map.py
```

Generate manuscript composite figures:

```bash
.venv/bin/python scripts/generate_sci_composite_figures.py \
  --output-dir artifacts/paper_composite_figures
```

## 6. Expected Main Metrics

For `transport_logistics4`, 16 shots/class, small CNN, seven paired seeds:

- Matched classical control macro-F1: `0.3055 +/- 0.1729`
- QNN no-entanglement macro-F1: `0.5031 +/- 0.1421`
- QNN ZZFeatureMap+RealAmplitudes macro-F1: `0.4545 +/- 0.1735`

Boundary checks:

- 32-shot no-entanglement delta over control: `+0.0094`
- 64-shot no-entanglement delta over control: `-0.0021`
- ResNet18 16-shot no-entanglement delta over control: `-0.0011`
- `urban_structural4` 16-shot no-entanglement delta over control: `-0.0063`
