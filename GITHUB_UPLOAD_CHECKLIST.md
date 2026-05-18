# GitHub Upload Checklist

Recommended repository:

```text
https://github.com/jerry-neal-wang/qiskit_hybrid_qnn_remote_sensing_release
```

Before uploading:

1. Confirm the public repository URL. If a different URL is used, update `manuscript/main.tex` and rebuild `manuscript/main.pdf` in Overleaf.
2. Upload either the whole `release_v2/` directory as repository contents or attach `release/release_v2.zip` to a GitHub release.
3. Do not add DIOR raw images, annotations, cropped ROI images, model checkpoints, IDE folders, or internal logs.
4. After upload, run a filename scan on the GitHub payload for `*.pt`, `*.pth`, `*.ckpt`, `JPEGImages`, and `Annotations`.

Suggested release tag:

```text
release_v2
```

Suggested release title:

```text
Regime-Bounded Hybrid Quantum Heads for Low-Shot DIOR ROI Classification - Release v2
```
