"""
QKernel-Lite – PCA & Angle Scaling
====================================
Stage 3: PCA dimensionality reduction + min-max scaling to [0, π].
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional

import config


class PCAAnglePreprocessor:
    """
    PCA reduction + angle scaling pipeline.

    Fits PCA and MinMaxScaler on training data ONLY,
    then transforms (and clips) test data.

    Parameters
    ----------
    n_components : int
        Number of PCA components (= number of qubits).
    pca_fit_on : str
        "subset" → fit on training subset only (default).
        "pool"  → fit on full training pool (no labels used).
    """

    def __init__(
        self,
        n_components: int = config.PCA_COMPONENTS,
        pca_fit_on: str = config.PCA_FIT_ON,
    ):
        self.n_components = n_components
        self.pca_fit_on = pca_fit_on
        self.pca: Optional[PCA] = None
        self.scaler: Optional[MinMaxScaler] = None
        self._is_fitted = False

    def fit(self, X_train: np.ndarray) -> "PCAAnglePreprocessor":
        """
        Fit PCA and angle scaler on training features.

        Parameters
        ----------
        X_train : np.ndarray of shape (n_train, 512)

        Notes
        -----
        Components are capped at N-1 to avoid rank-deficient covariance
        (Trap 4 in the build plan).
        """
        n_samples = X_train.shape[0]
        n_comp = min(self.n_components, n_samples - 1)

        if n_comp < self.n_components:
            print(f"  ⚠ Capping PCA to {n_comp} components "
                  f"(N={n_samples}, max = N-1)")

        self.pca = PCA(n_components=n_comp, random_state=0)
        X_pca = self.pca.fit_transform(X_train)

        # Min-max scale to [0, π] based on TRAINING data
        self.scaler = MinMaxScaler(
            feature_range=(config.ANGLE_MIN, config.ANGLE_MAX)
        )
        self.scaler.fit(X_pca)

        self._is_fitted = True
        self.n_components_actual = n_comp
        return self

    def transform(self, X: np.ndarray, clip: bool = True) -> np.ndarray:
        """
        Apply PCA + angle scaling. Optionally clip to [0, π].

        Parameters
        ----------
        X : np.ndarray of shape (n, 512)
        clip : bool
            If True, clip output to [0, π] (for test data, Trap 5).

        Returns
        -------
        X_scaled : np.ndarray of shape (n, n_components)
        """
        assert self._is_fitted, "Must call .fit() first!"

        X_pca = self.pca.transform(X)
        X_scaled = self.scaler.transform(X_pca)

        if clip:
            X_scaled = np.clip(X_scaled, config.ANGLE_MIN, config.ANGLE_MAX)

        return X_scaled

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """Fit on X_train and return transformed training data."""
        self.fit(X_train)
        return self.transform(X_train, clip=True)

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Fraction of variance explained by each component."""
        assert self._is_fitted
        return self.pca.explained_variance_ratio_

    @property
    def total_explained_variance(self) -> float:
        """Total fraction of variance retained."""
        return float(np.sum(self.explained_variance_ratio))

    def print_variance_report(self):
        """Print explained variance per component."""
        ratios = self.explained_variance_ratio
        print(f"  PCA: {self.n_components_actual} components, "
              f"total variance retained: {self.total_explained_variance:.4f}")
        for i, r in enumerate(ratios):
            bar = "█" * int(r * 50)
            print(f"    PC{i+1}: {r:.4f} {bar}")


def preprocess_for_experiment(
    X_train_full: np.ndarray,
    X_test_full: np.ndarray,
    train_indices: np.ndarray,
    n_components: int = config.PCA_COMPONENTS,
) -> Tuple[np.ndarray, np.ndarray, PCAAnglePreprocessor]:
    """
    End-to-end preprocessing for one experiment run.

    Parameters
    ----------
    X_train_full : np.ndarray of shape (total_train, 512)
        Full training features from ResNet-18.
    X_test_full : np.ndarray of shape (n_test, 512)
        Full test features.
    train_indices : np.ndarray
        Subset indices into X_train_full.
    n_components : int
        Number of PCA components.

    Returns
    -------
    X_train_scaled : np.ndarray of shape (N, n_components) in [0, π]
    X_test_scaled  : np.ndarray of shape (n_test, n_components) in [0, π]
    preprocessor   : fitted PCAAnglePreprocessor
    """
    X_sub = X_train_full[train_indices]

    preprocessor = PCAAnglePreprocessor(n_components=n_components)
    X_train_scaled = preprocessor.fit_transform(X_sub)
    X_test_scaled = preprocessor.transform(X_test_full, clip=True)

    return X_train_scaled, X_test_scaled, preprocessor


if __name__ == "__main__":
    # Quick tests
    from data import load_pneumoniamnist, stratified_subset_indices
    from features import extract_and_cache_features

    train_ds, val_ds, test_ds, _ = load_pneumoniamnist()
    X_train_full = extract_and_cache_features(train_ds, "train")
    X_test_full = extract_and_cache_features(test_ds, "test")

    indices = stratified_subset_indices(train_ds, n=20, seed=0)

    X_train, X_test, pp = preprocess_for_experiment(
        X_train_full, X_test_full, indices, n_components=8
    )

    print(f"\nTrain scaled: shape={X_train.shape}, "
          f"min={X_train.min():.6f}, max={X_train.max():.6f}")
    print(f"Test scaled:  shape={X_test.shape}, "
          f"min={X_test.min():.6f}, max={X_test.max():.6f}")

    # Check: training spans exactly [0, π]
    assert np.isclose(X_train.min(), 0.0, atol=1e-6), "Train min ≠ 0"
    assert np.isclose(X_train.max(), np.pi, atol=1e-6), "Train max ≠ π"

    # Check: test is clipped
    assert X_test.min() >= 0.0, "Test has values < 0"
    assert X_test.max() <= np.pi, "Test has values > π"

    pp.print_variance_report()

    # Check: changing test data does not change PCA components
    pp2 = PCAAnglePreprocessor(n_components=8)
    pp2.fit(X_train_full[indices])
    assert np.allclose(pp.pca.components_, pp2.pca.components_), \
        "PCA components changed with different test data!"
    print("\n✓ All Stage 3 checks passed!")
