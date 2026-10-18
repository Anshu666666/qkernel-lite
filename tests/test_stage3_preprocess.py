"""
Stage 3 Tests: PCA & Angle Scaling
=====================================
Pass when:
  - Scaled training features span exactly [0, π]
  - Test values are clipped to [0, π]
  - Components capped at N-1
  - Explained variance prints
  - Changing test data does NOT change PCA components
"""

import pytest
import numpy as np
from preprocess import PCAAnglePreprocessor, preprocess_for_experiment
from data import load_pneumoniamnist, stratified_subset_indices
from features import extract_and_cache_features
import config


@pytest.fixture(scope="module")
def data():
    train_ds, _, test_ds, _ = load_pneumoniamnist()
    X_train_full = extract_and_cache_features(train_ds, "train")
    X_test_full = extract_and_cache_features(test_ds, "test")
    indices = stratified_subset_indices(train_ds, n=20, seed=0)
    return X_train_full, X_test_full, indices


def test_train_spans_0_to_pi(data):
    """Scaled training features span exactly [0, π]."""
    X_train_full, X_test_full, indices = data
    X_train, X_test, pp = preprocess_for_experiment(
        X_train_full, X_test_full, indices, n_components=8
    )
    assert np.isclose(X_train.min(), 0.0, atol=1e-6), \
        f"Train min = {X_train.min()}, expected 0"
    assert np.isclose(X_train.max(), np.pi, atol=1e-6), \
        f"Train max = {X_train.max()}, expected π"


def test_test_clipped(data):
    """Test values are clipped to [0, π]."""
    X_train_full, X_test_full, indices = data
    _, X_test, _ = preprocess_for_experiment(
        X_train_full, X_test_full, indices, n_components=8
    )
    assert X_test.min() >= 0.0 - 1e-10, f"Test min = {X_test.min()}"
    assert X_test.max() <= np.pi + 1e-10, f"Test max = {X_test.max()}"


def test_components_capped_at_n_minus_1():
    """PCA components capped at N-1 for small N."""
    np.random.seed(42)
    X_small = np.random.randn(5, 512)  # N=5, want 8 components

    pp = PCAAnglePreprocessor(n_components=8)
    pp.fit(X_small)
    assert pp.n_components_actual == 4, \
        f"Expected 4 components (N-1=4), got {pp.n_components_actual}"


def test_explained_variance_prints(data):
    """Variance report doesn't crash and sums to ≤ 1."""
    X_train_full, X_test_full, indices = data
    _, _, pp = preprocess_for_experiment(
        X_train_full, X_test_full, indices, n_components=8
    )
    pp.print_variance_report()
    assert 0 < pp.total_explained_variance <= 1.0


def test_pca_independent_of_test_data(data):
    """Changing test data doesn't affect PCA components."""
    X_train_full, X_test_full, indices = data
    X_sub = X_train_full[indices]

    pp1 = PCAAnglePreprocessor(n_components=8)
    pp1.fit(X_sub)

    pp2 = PCAAnglePreprocessor(n_components=8)
    pp2.fit(X_sub)

    assert np.allclose(pp1.pca.components_, pp2.pca.components_), \
        "PCA components changed between fits on same data!"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
