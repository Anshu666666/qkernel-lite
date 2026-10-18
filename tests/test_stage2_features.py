"""
Stage 2 Tests: Feature Extraction
===================================
Pass when:
  - Cache has shape (n_images, 512) with no NaNs
  - Second run loads from cache
  - Logistic regression on 200 train images scores > chance on test
"""

import pytest
import numpy as np
from data import load_pneumoniamnist
from features import extract_and_cache_features, get_cache_path
import config


@pytest.fixture(scope="module")
def datasets():
    return load_pneumoniamnist()


@pytest.fixture(scope="module")
def features(datasets):
    train_ds, val_ds, test_ds, _ = datasets
    X_train = extract_and_cache_features(train_ds, "train")
    X_test = extract_and_cache_features(test_ds, "test")
    return X_train, X_test, train_ds, test_ds


def test_feature_shape(features):
    """Features have shape (n, 512)."""
    X_train, X_test, train_ds, test_ds = features
    assert X_train.shape == (len(train_ds.labels), 512), \
        f"Train features shape: {X_train.shape}"
    assert X_test.shape == (len(test_ds.labels), 512), \
        f"Test features shape: {X_test.shape}"


def test_no_nans(features):
    """No NaN values in features."""
    X_train, X_test, _, _ = features
    assert not np.any(np.isnan(X_train)), "NaN in train features!"
    assert not np.any(np.isnan(X_test)), "NaN in test features!"


def test_cache_exists():
    """Cache files exist on disk."""
    for split in ["train", "test"]:
        path = get_cache_path(split)
        assert path.exists(), f"Cache not found: {path}"


def test_cache_reload(datasets):
    """Second call loads from cache (fast)."""
    train_ds, _, _, _ = datasets
    import time
    t0 = time.time()
    X = extract_and_cache_features(train_ds, "train")
    elapsed = time.time() - t0
    assert elapsed < 5.0, f"Cache reload too slow: {elapsed:.1f}s"
    print(f"  ✓ Cache reload: {elapsed:.2f}s")


def test_logistic_regression_above_chance(features):
    """LogReg on ResNet features scores clearly above chance."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    X_train, X_test, train_ds, test_ds = features

    n = min(200, len(X_train))
    lr = LogisticRegression(max_iter=1000, random_state=0)
    lr.fit(X_train[:n], train_ds.labels.flatten()[:n])

    probs = lr.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(test_ds.labels.flatten(), probs)

    print(f"  LogReg AUC (N={n}): {auc:.4f}")
    assert auc > 0.6, f"AUC too low ({auc:.4f}) — features may be broken"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
