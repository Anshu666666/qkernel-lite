"""
Stage 4 Tests: Classical Models
=================================
Pass when:
  - Manual RBF kernel matches sklearn's built-in
  - Predictions match between manual and sklearn RBF SVC
  - Tuning function cannot receive test arrays
  - M1 training loss decreases
  - M1 is reproducible for fixed seed
"""

import pytest
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel as sklearn_rbf
from classical import (
    compute_rbf_kernel,
    SmallCNN,
    train_m1,
    predict_m1,
    tune_svm,
)


def test_rbf_kernel_matches_sklearn():
    """Manual RBF kernel matches sklearn's implementation."""
    np.random.seed(42)
    X = np.random.randn(30, 8)
    for gamma in [0.01, 0.1, 0.5, 1.0, 5.0]:
        K_manual = compute_rbf_kernel(X, X, gamma)
        K_sklearn = sklearn_rbf(X, X, gamma=gamma)
        assert np.allclose(K_manual, K_sklearn, atol=1e-10), \
            f"Mismatch at gamma={gamma}"
    print("  ✓ RBF kernel matches sklearn at 5 gamma values")


def test_rbf_predictions_match_sklearn():
    """Precomputed RBF SVM gives same predictions as built-in."""
    np.random.seed(42)
    X = np.random.randn(30, 8)
    y = np.array([0]*15 + [1]*15)
    gamma = 0.5

    K = compute_rbf_kernel(X, X, gamma)
    svm_pre = SVC(kernel="precomputed", C=1.0)
    svm_pre.fit(K, y)
    pred_pre = svm_pre.predict(K)

    svm_rbf = SVC(kernel="rbf", gamma=gamma, C=1.0)
    svm_rbf.fit(X, y)
    pred_rbf = svm_rbf.predict(X)

    assert np.array_equal(pred_pre, pred_rbf), \
        f"Predictions differ: {np.sum(pred_pre != pred_rbf)} mismatches"


def test_tuning_does_not_accept_test():
    """tune_svm signature doesn't have a test_X parameter."""
    import inspect
    sig = inspect.signature(tune_svm)
    param_names = list(sig.parameters.keys())
    for bad_name in ["X_test", "test_X", "test_data", "X_test_scaled"]:
        assert bad_name not in param_names, \
            f"tune_svm has a '{bad_name}' parameter — possible test leakage!"


def test_m1_loss_decreases():
    """M1 CNN training loss decreases over epochs."""
    np.random.seed(0)
    images = np.random.randint(0, 255, size=(20, 28, 28)).astype(np.uint8)
    labels = np.array([0]*10 + [1]*10)

    model, losses = train_m1(images, labels, seed=0, epochs=30)
    assert losses[-1] < losses[0], \
        f"Loss didn't decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
    print(f"  ✓ Loss: {losses[0]:.4f} → {losses[-1]:.4f}")


def test_m1_reproducible():
    """M1 with same seed gives same loss trajectory."""
    np.random.seed(0)
    images = np.random.randint(0, 255, size=(20, 28, 28)).astype(np.uint8)
    labels = np.array([0]*10 + [1]*10)

    _, losses1 = train_m1(images, labels, seed=42, epochs=10)
    _, losses2 = train_m1(images, labels, seed=42, epochs=10)

    assert np.allclose(losses1, losses2, atol=1e-5), \
        "M1 not reproducible with same seed!"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
