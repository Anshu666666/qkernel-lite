"""
Stage 1 Tests: Data Loading & Stratified Subset
=================================================
Pass when:
  - Prints split sizes and class counts
  - Same (seed, N) returns identical indices twice
  - Every subset is 50/50
  - Sample grid saves and can be inspected
"""

import pytest
import numpy as np
from data import (
    load_pneumoniamnist,
    stratified_subset_indices,
    get_subset_data,
    get_class_counts,
    save_sample_grid,
    print_split_info,
)
import config


@pytest.fixture(scope="module")
def datasets():
    return load_pneumoniamnist()


def test_split_sizes(datasets):
    """All splits have data."""
    train_ds, val_ds, test_ds, info = datasets
    assert len(train_ds.labels) > 0
    assert len(val_ds.labels) > 0
    assert len(test_ds.labels) > 0
    print_split_info(train_ds, val_ds, test_ds)


def test_class_counts(datasets):
    """Each split has both classes."""
    train_ds, val_ds, test_ds, _ = datasets
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        counts = get_class_counts(ds)
        assert 0 in counts and 1 in counts, f"{name} missing a class: {counts}"


def test_deterministic_sampling(datasets):
    """Same (seed, N) → identical indices."""
    train_ds, _, _, _ = datasets
    for seed in [0, 1, 42]:
        for n in [10, 20, 50]:
            idx1 = stratified_subset_indices(train_ds, n=n, seed=seed)
            idx2 = stratified_subset_indices(train_ds, n=n, seed=seed)
            assert np.array_equal(idx1, idx2), \
                f"Non-deterministic at seed={seed}, N={n}"


def test_different_seeds_different_subsets(datasets):
    """Different seeds produce different subsets."""
    train_ds, _, _, _ = datasets
    idx0 = stratified_subset_indices(train_ds, n=20, seed=0)
    idx1 = stratified_subset_indices(train_ds, n=20, seed=1)
    assert not np.array_equal(idx0, idx1), "Seeds 0 and 1 gave same indices!"


def test_50_50_balance(datasets):
    """Every subset is exactly 50/50."""
    train_ds, _, _, _ = datasets
    for n in config.N_VALUES:
        for seed in range(5):
            indices = stratified_subset_indices(train_ds, n=n, seed=seed)
            _, labels = get_subset_data(train_ds, indices)
            n_0 = np.sum(labels == 0)
            n_1 = np.sum(labels == 1)
            assert n_0 == n // 2, f"N={n}, seed={seed}: class 0 has {n_0}, expected {n//2}"
            assert n_1 == n // 2, f"N={n}, seed={seed}: class 1 has {n_1}, expected {n//2}"


def test_sample_grid_saves(datasets):
    """Sample grid image saves to disk."""
    train_ds, _, _, _ = datasets
    indices = stratified_subset_indices(train_ds, n=20, seed=0)
    path = save_sample_grid(train_ds, indices, title="Test Grid")
    assert path.exists(), f"Grid image not saved to {path}"
    assert path.stat().st_size > 0, "Grid image is empty"
    print(f"  ✓ Grid saved: {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
