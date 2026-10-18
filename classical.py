"""
QKernel-Lite – Classical Models (M1 & M2)
==========================================
Stage 4:
  M1: Small CNN trained from scratch
  M2: RBF kernel SVM with precomputed kernel
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from typing import Tuple, Optional, Dict
import time

import config


# ═══════════════════════════════════════════════════════════════════════
#  M1: Small CNN
# ═══════════════════════════════════════════════════════════════════════

class SmallCNN(nn.Module):
    """
    Tiny CNN for the from-scratch baseline (M1).
    Architecture from the build plan:
      conv 1→16 (3×3), ReLU, MaxPool
      conv 16→32 (3×3), ReLU, MaxPool
      flatten → linear →64, ReLU → linear →1
    """

    def __init__(self, input_size: int = 28):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Calculate flattened size
        dummy = torch.zeros(1, 1, input_size, input_size)
        with torch.no_grad():
            flat_size = self.conv_block(dummy).view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train_m1(
    images: np.ndarray,
    labels: np.ndarray,
    seed: int = 0,
    epochs: int = config.M1_EPOCHS,
    lr: float = config.M1_LR,
    weight_decay: float = config.M1_WEIGHT_DECAY,
    input_size: int = 28,
    device: str = "cpu",
) -> Tuple[SmallCNN, list]:
    """
    Train M1 CNN from scratch.

    Parameters
    ----------
    images : np.ndarray of shape (N, H, W) – raw grayscale
    labels : np.ndarray of shape (N,) – 0 or 1
    seed : int – for reproducibility

    Returns
    -------
    model : trained SmallCNN
    losses : list of per-epoch training losses
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Prepare tensors: (N, 1, H, W) float32
    if images.ndim == 4 and images.shape[-1] == 1:
        images = images[:, :, :, 0]
    X = torch.FloatTensor(images).unsqueeze(1) / 255.0  # normalize to [0,1]

    # Resize to input_size if needed
    if X.shape[-1] != input_size:
        X = torch.nn.functional.interpolate(
            X, size=(input_size, input_size), mode='bilinear', align_corners=False
        )

    y = torch.FloatTensor(labels)
    dataset = TensorDataset(X, y)
    batch_size = min(config.M1_BATCH_SIZE_CAP, len(labels))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SmallCNN(input_size=input_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    losses = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_X).squeeze(-1)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_X)
        epoch_loss /= len(labels)
        losses.append(epoch_loss)

    return model, losses


@torch.no_grad()
def predict_m1(
    model: SmallCNN,
    images: np.ndarray,
    input_size: int = 28,
    device: str = "cpu",
) -> np.ndarray:
    """Return probability of class 1 for each image."""
    if images.ndim == 4 and images.shape[-1] == 1:
        images = images[:, :, :, 0]
    X = torch.FloatTensor(images).unsqueeze(1) / 255.0

    if X.shape[-1] != input_size:
        X = torch.nn.functional.interpolate(
            X, size=(input_size, input_size), mode='bilinear', align_corners=False
        )

    model.eval()
    logits = model(X.to(device)).squeeze(-1).cpu()
    probs = torch.sigmoid(logits).numpy()
    return probs


# ═══════════════════════════════════════════════════════════════════════
#  M2: RBF Kernel SVM
# ═══════════════════════════════════════════════════════════════════════

def compute_rbf_kernel(
    X1: np.ndarray,
    X2: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    Compute RBF (Gaussian) kernel matrix manually.

    k(x, x') = exp(-gamma * ||x - x'||^2)

    Parameters
    ----------
    X1 : np.ndarray of shape (n1, d)
    X2 : np.ndarray of shape (n2, d)
    gamma : float

    Returns
    -------
    K : np.ndarray of shape (n1, n2)
    """
    # ||x - x'||^2 = ||x||^2 + ||x'||^2 - 2 x·x'
    sq1 = np.sum(X1 ** 2, axis=1, keepdims=True)  # (n1, 1)
    sq2 = np.sum(X2 ** 2, axis=1, keepdims=True)  # (n2, 1)
    dist_sq = sq1 + sq2.T - 2 * (X1 @ X2.T)      # (n1, n2)
    dist_sq = np.maximum(dist_sq, 0.0)             # numerical safety
    return np.exp(-gamma * dist_sq)


def compute_base_gamma(X: np.ndarray) -> float:
    """
    Base gamma via median heuristic:
    1 / median(||x_i - x_j||^2) for i < j.
    """
    n = len(X)
    if n < 2:
        return 1.0
    sq = np.sum(X ** 2, axis=1, keepdims=True)
    dist_sq = sq + sq.T - 2 * (X @ X.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    i_tri = np.triu_indices(n, k=1)
    dists = dist_sq[i_tri]
    med = float(np.median(dists)) if len(dists) > 0 else 0.0
    if med < 1e-12:
        return 1.0
    return 1.0 / med


def compute_feature_diagnostics(
    features: np.ndarray,
    pca_reference: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute diagnostics on feature representations:
      - feat_variance: total variance of the feature array
      - dist_std_over_median: std(pairwise distances) / median(pairwise distances)
      - dist_spearman_rank_corr: Spearman rank correlation between pairwise
        distances in PCA space and in expectation space.
    """
    n = len(features)
    var = float(np.var(features))

    sq = np.sum(features ** 2, axis=1, keepdims=True)
    dist_sq = np.maximum(sq + sq.T - 2 * (features @ features.T), 0.0)
    i_tri = np.triu_indices(n, k=1)
    dists = np.sqrt(dist_sq[i_tri])

    med_dist = float(np.median(dists)) if len(dists) > 0 else 0.0
    if med_dist > 1e-12:
        dist_std_over_med = float(np.std(dists) / med_dist)
    else:
        dist_std_over_med = 0.0

    if pca_reference is not None and len(pca_reference) == n:
        sq_ref = np.sum(pca_reference ** 2, axis=1, keepdims=True)
        dist_sq_ref = np.maximum(sq_ref + sq_ref.T - 2 * (pca_reference @ pca_reference.T), 0.0)
        dists_ref = np.sqrt(dist_sq_ref[i_tri])
        if np.std(dists) > 1e-12 and np.std(dists_ref) > 1e-12:
            corr, _ = spearmanr(dists_ref, dists)
            rank_corr = float(corr)
        else:
            rank_corr = 0.0
    else:
        rank_corr = 1.0

    return {
        "feat_variance": var,
        "dist_std_over_median": dist_std_over_med,
        "dist_spearman_rank_corr": rank_corr,
    }


def tune_svm(
    K_train: np.ndarray,
    y_train: np.ndarray,
    X_train_for_kernel: np.ndarray,
    gamma_multipliers: list = config.GAMMA_MULTIPLIERS,
    C_values: list = config.C_VALUES,
    base_gamma: Optional[float] = None,
    n_folds: Optional[int] = None,
    kernel_fn=None,
    seed: int = 0,
) -> Dict:
    """
    Tune SVM hyperparameters (gamma, C) via stratified k-fold CV.

    IMPORTANT: This function never sees test data (Trap 3).

    Parameters
    ----------
    K_train : np.ndarray of shape (N, N)
        Precomputed kernel on training data (used only if kernel_fn is None).
    y_train : np.ndarray of shape (N,)
    X_train_for_kernel : np.ndarray of shape (N, d)
        Raw features (used to recompute kernel on folds).
    gamma_multipliers : list of float
    C_values : list of float
    base_gamma : float, optional
        If None, computed from X_train_for_kernel via median heuristic.
    n_folds : int, optional
        If None, determined by config.get_cv_folds.
    kernel_fn : callable, optional
        Function (X1, X2, gamma) → kernel matrix.
        Defaults to compute_rbf_kernel.
    seed : int

    Returns
    -------
    dict with keys: best_gamma, best_C, best_multiplier, gamma_edge, C_edge, best_auc, cv_results
    """
    if base_gamma is None:
        base_gamma = compute_base_gamma(X_train_for_kernel)
    if n_folds is None:
        n_folds = config.get_cv_folds(len(y_train))
    if kernel_fn is None:
        kernel_fn = compute_rbf_kernel

    best_auc = -1.0
    best_gamma = None
    best_C = None
    best_multiplier = None
    best_tie_score = float("inf")
    cv_results = []

    for gm in gamma_multipliers:
        gamma = base_gamma * gm

        for C in C_values:
            fold_aucs = []
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

            for train_idx, val_idx in skf.split(X_train_for_kernel, y_train):
                X_fold_train = X_train_for_kernel[train_idx]
                X_fold_val = X_train_for_kernel[val_idx]
                y_fold_train = y_train[train_idx]
                y_fold_val = y_train[val_idx]

                K_fold_train = kernel_fn(X_fold_train, X_fold_train, gamma)
                K_fold_val = kernel_fn(X_fold_val, X_fold_train, gamma)

                svm = SVC(kernel="precomputed", C=C, probability=False)
                svm.fit(K_fold_train, y_fold_train)

                scores = svm.decision_function(K_fold_val)
                try:
                    auc = roc_auc_score(y_fold_val, scores)
                except ValueError:
                    auc = 0.5  # single class in fold

                fold_aucs.append(auc)

            mean_auc = float(np.mean(fold_aucs))
            curr_tie_score = abs(np.log(gm)) + abs(np.log(C))
            cv_results.append({
                "gamma": gamma, "multiplier": gm, "C": C, "mean_auc": mean_auc,
                "fold_aucs": fold_aucs,
            })

            is_better = False
            if mean_auc > best_auc + 1e-9:
                is_better = True
            elif abs(mean_auc - best_auc) <= 1e-9:
                # Deterministic tie-break: prefer multiplier closest to 1.0 and C closest to 1.0
                if curr_tie_score < best_tie_score:
                    is_better = True

            if is_better:
                best_auc = mean_auc
                best_gamma = gamma
                best_C = C
                best_multiplier = gm
                best_tie_score = curr_tie_score

    min_gm = min(gamma_multipliers)
    max_gm = max(gamma_multipliers)
    min_C = min(C_values)
    max_C = max(C_values)
    gamma_edge = bool(best_multiplier == min_gm or best_multiplier == max_gm)
    C_edge = bool(best_C == min_C or best_C == max_C)

    return {
        "best_gamma": best_gamma,
        "best_C": best_C,
        "best_multiplier": best_multiplier,
        "gamma_edge": gamma_edge,
        "C_edge": C_edge,
        "best_auc": best_auc,
        "cv_results": cv_results,
    }


def train_and_predict_m2(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    seed: int = 0,
) -> Tuple[np.ndarray, Dict]:
    """
    Full M2 pipeline: tune gamma/C → train SVM → predict.

    Parameters
    ----------
    X_train : np.ndarray of shape (N, d) – scaled PCA features
    y_train : np.ndarray of shape (N,)
    X_test  : np.ndarray of shape (n_test, d)
    seed    : int

    Returns
    -------
    decision_scores : np.ndarray of shape (n_test,)
    info : dict with tuning results and kernel diagnostics
    """
    t0 = time.time()

    # Compute base gamma & Tune
    base_gamma = compute_base_gamma(X_train)
    tune_result = tune_svm(
        K_train=None, y_train=y_train,
        X_train_for_kernel=X_train,
        kernel_fn=compute_rbf_kernel,
        base_gamma=base_gamma,
        seed=seed,
    )

    gamma = tune_result["best_gamma"]
    C = tune_result["best_C"]

    # Train with best params
    K_train = compute_rbf_kernel(X_train, X_train, gamma)
    svm = SVC(kernel="precomputed", C=C, probability=False)
    svm.fit(K_train, y_train)

    # Predict
    K_test = compute_rbf_kernel(X_test, X_train, gamma)
    decision_scores = svm.decision_function(K_test)

    # Untuned baseline (multiplier = 1.0, C = 1.0)
    K_train_untuned = compute_rbf_kernel(X_train, X_train, base_gamma)
    svm_untuned = SVC(kernel="precomputed", C=1.0, probability=False)
    svm_untuned.fit(K_train_untuned, y_train)
    K_test_untuned = compute_rbf_kernel(X_test, X_train, base_gamma)
    scores_notuned = svm_untuned.decision_function(K_test_untuned)

    elapsed = time.time() - t0

    # Kernel & feature diagnostics
    diag = kernel_diagnostics(K_train)
    feat_diag = compute_feature_diagnostics(X_train, pca_reference=None)

    return decision_scores, {
        "best_gamma": gamma,
        "best_C": C,
        "gamma_edge": tune_result["gamma_edge"],
        "C_edge": tune_result["C_edge"],
        "scores_notuned": scores_notuned,
        "best_cv_auc": tune_result["best_auc"],
        "runtime_seconds": elapsed,
        **feat_diag,
        **diag,
    }


def kernel_diagnostics(K: np.ndarray) -> Dict:
    """
    Compute kernel matrix health diagnostics.

    Returns dict with:
      - offdiag_mean, offdiag_std
      - min_eigenvalue
      - is_symmetric
      - diag_is_one
    """
    n = K.shape[0]
    mask = ~np.eye(n, dtype=bool)
    offdiag = K[mask]

    eigenvalues = np.linalg.eigvalsh(K)

    return {
        "kernel_offdiag_mean": float(np.mean(offdiag)),
        "kernel_offdiag_std": float(np.std(offdiag)),
        "kernel_min_eigenvalue": float(eigenvalues[0]),
        "kernel_is_symmetric": bool(np.allclose(K, K.T)),
        "kernel_diag_is_one": bool(np.allclose(np.diag(K), 1.0)),
    }


if __name__ == "__main__":
    # Quick test: verify manual RBF matches sklearn's
    from sklearn.metrics.pairwise import rbf_kernel as sklearn_rbf

    np.random.seed(42)
    X = np.random.randn(20, 8)
    gamma = 0.5

    K_manual = compute_rbf_kernel(X, X, gamma)
    K_sklearn = sklearn_rbf(X, X, gamma=gamma)

    assert np.allclose(K_manual, K_sklearn, atol=1e-10), \
        "Manual RBF kernel doesn't match sklearn!"
    print("✓ Manual RBF kernel matches sklearn")

    # Verify equivalence of predictions
    y = np.array([0]*10 + [1]*10)

    svm_manual = SVC(kernel="precomputed", C=1.0)
    svm_manual.fit(K_manual, y)
    pred_manual = svm_manual.predict(K_manual)

    svm_sklearn = SVC(kernel="rbf", gamma=gamma, C=1.0)
    svm_sklearn.fit(X, y)
    pred_sklearn = svm_sklearn.predict(X)

    assert np.array_equal(pred_manual, pred_sklearn), \
        "Predictions differ between manual and sklearn RBF!"
    print("✓ Predictions match between manual and sklearn RBF SVC")
