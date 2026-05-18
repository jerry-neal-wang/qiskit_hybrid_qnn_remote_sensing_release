# Processed Metadata

This directory contains metadata only. It does not include DIOR raw images, original annotations, or cropped ROI image files.

Subdirectories:

- `transport_logistics4/`: main four-class subset used for the 16-shot evidence and shot/backbone boundary checks.
- `urban_structural4/`: four-class cross-subset boundary check.
- `dior_multiclass_stratified/`: current multiclass processed metadata snapshot retained for code-level reproducibility context.

Each subset contains:

- `label_map.json`
- `metadata.csv`
- `split_summary.csv`

The CSV rows reference generated ROI filenames, source image ids, labels, and bounding boxes, but no image payload is redistributed.
