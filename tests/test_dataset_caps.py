from src.datasets.roi_dataset import resolve_split_sample_caps


def test_resolve_split_sample_caps_with_global_default() -> None:
    caps = resolve_split_sample_caps(max_samples_per_class=16)
    assert caps == {"train": 16, "val": 16, "test": 16}


def test_resolve_split_sample_caps_with_split_override() -> None:
    caps = resolve_split_sample_caps(
        max_samples_per_class=16,
        split_max_samples_per_class={"val": 256, "test": None},
    )
    assert caps == {"train": 16, "val": 256, "test": None}
