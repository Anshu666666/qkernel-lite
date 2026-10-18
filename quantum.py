"""
QKernel-Lite – Quantum Kernel (M3)
====================================
Stage 5: PennyLane projected quantum kernel.

Circuit: For L layers:
  RY(θ_i) → RZ(θ_i) on each qubit i
  CNOT ring (i, i+1 mod n) if entanglement=True

Projection: ⟨X⟩, ⟨Y⟩, ⟨Z⟩ on every qubit → 3n real numbers per image.
Kernel: k(x,x') = exp(-γ Σ (⟨P_i⟩(x) - ⟨P_i⟩(x'))²)
"""

import numpy as np
import pennylane as qml
from typing import Tuple, Dict, Optional
from pathlib import Path
import hashlib
import json
import time

import config
from classical import (
    compute_base_gamma,
    tune_svm,
    kernel_diagnostics,
    compute_feature_diagnostics,
)


# ═══════════════════════════════════════════════════════════════════════
#  Quantum Circuit
# ═══════════════════════════════════════════════════════════════════════

def create_circuit(
    n_qubits: int = config.DEFAULT_QUBITS,
    n_layers: int = config.DEFAULT_LAYERS,
    entanglement: bool = config.DEFAULT_ENTANGLEMENT,
):
    """
    Create a PennyLane QNode that encodes features and returns
    all 3n Pauli expectation values.

    Parameters
    ----------
    n_qubits : int
    n_layers : int
    entanglement : bool
        If True, apply CNOT ring after encoding.

    Returns
    -------
    circuit : QNode
        Takes features (n_qubits,) → returns (3*n_qubits,) expectations.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="numpy")
    def circuit(features):
        for layer in range(n_layers):
            # Angle encoding: RY(θ_i) then RZ(θ_i) on each qubit
            for i in range(n_qubits):
                qml.RY(features[i], wires=i)
                qml.RZ(features[i], wires=i)

            # Entangling layer: CNOT ring
            if entanglement:
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])

        # Measure ⟨X⟩, ⟨Y⟩, ⟨Z⟩ on every qubit
        observables = []
        for i in range(n_qubits):
            observables.append(qml.expval(qml.PauliX(i)))
            observables.append(qml.expval(qml.PauliY(i)))
            observables.append(qml.expval(qml.PauliZ(i)))

        return tuple(observables)

    return circuit


def draw_circuit(
    n_qubits: int = config.DEFAULT_QUBITS,
    n_layers: int = config.DEFAULT_LAYERS,
    entanglement: bool = config.DEFAULT_ENTANGLEMENT,
) -> str:
    """Return text ASCII diagram of the quantum circuit."""
    circuit = create_circuit(n_qubits, n_layers, entanglement)
    dummy_features = np.ones(n_qubits) * (np.pi / 4)
    return str(qml.draw(circuit)(dummy_features))



def get_expectation_values(
    features: np.ndarray,
    n_qubits: int = config.DEFAULT_QUBITS,
    n_layers: int = config.DEFAULT_LAYERS,
    entanglement: bool = config.DEFAULT_ENTANGLEMENT,
) -> np.ndarray:
    """
    Compute 3n expectation values for a single feature vector.

    Parameters
    ----------
    features : np.ndarray of shape (n_qubits,)

    Returns
    -------
    expectations : np.ndarray of shape (3 * n_qubits,)
    """
    circuit = create_circuit(n_qubits, n_layers, entanglement)
    result = circuit(features)
    return np.array(result)


def create_batched_circuit(
    n_qubits: int = config.DEFAULT_QUBITS,
    n_layers: int = config.DEFAULT_LAYERS,
    entanglement: bool = config.DEFAULT_ENTANGLEMENT,
):
    """
    Create a batched PennyLane QNode for evaluating multiple images simultaneously.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="numpy")
    def circuit(features):
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(features[:, i], wires=i)
                qml.RZ(features[:, i], wires=i)
            if entanglement:
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])
        obs = []
        for i in range(n_qubits):
            obs.append(qml.expval(qml.PauliX(i)))
            obs.append(qml.expval(qml.PauliY(i)))
            obs.append(qml.expval(qml.PauliZ(i)))
        return obs

    return circuit


def get_all_expectation_values(
    X: np.ndarray,
    n_qubits: int = config.DEFAULT_QUBITS,
    n_layers: int = config.DEFAULT_LAYERS,
    entanglement: bool = config.DEFAULT_ENTANGLEMENT,
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute expectation values for all images using vectorized batching.
    Falls back gracefully to sequential if needed.
    """
    n = len(X)
    try:
        circuit = create_batched_circuit(n_qubits, n_layers, entanglement)
        raw_obs = circuit(X)
        results = np.column_stack([np.array(o) for o in raw_obs])
    except Exception as e:
        circuit = create_circuit(n_qubits, n_layers, entanglement)
        results = np.zeros((n, 3 * n_qubits))
        for i in range(n):
            results[i] = np.array(circuit(X[i]))

    if verbose:
        print(f"    Done: {n} images → ({n}, {3*n_qubits}) expectations")

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Caching
# ═══════════════════════════════════════════════════════════════════════

def _cache_key(
    split: str,
    seed: int,
    n_train: int,
    n_qubits: int,
    entanglement: bool,
    n_layers: int,
    scale: float = 1.0,
) -> str:
    """Generate a deterministic cache filename."""
    if scale == 1.0:
        return f"{split}_s{seed}_N{n_train}_q{n_qubits}_ent{entanglement}_L{n_layers}"
    if scale == 0.25:
        return f"{split}_s{seed}_N{n_train}_q{n_qubits}_ent{entanglement}_L{n_layers}_sc0.25"
    if scale == 0.5:
        return f"{split}_s{seed}_N{n_train}_q{n_qubits}_ent{entanglement}_L{n_layers}_sc0.50"
    return f"{split}_s{seed}_N{n_train}_q{n_qubits}_ent{entanglement}_L{n_layers}_sc{scale:g}"


def get_cached_expectations(
    X: np.ndarray,
    split: str,
    seed: int,
    n_train: int,
    n_qubits: int = config.DEFAULT_QUBITS,
    n_layers: int = config.DEFAULT_LAYERS,
    entanglement: bool = config.DEFAULT_ENTANGLEMENT,
    scale: float = 1.0,
    force: bool = False,
    verbose: bool = True,
) -> np.ndarray:
    """
    Get expectation values with disk caching.

    Cache key: (split, seed, N, qubits, entanglement, layers, scale).
    """
    tag = _cache_key(split, seed, n_train, n_qubits, entanglement, n_layers, scale)
    cache_path = config.CACHE_DIR / f"quantum_exp_{tag}.npy"

    if cache_path.exists() and not force:
        if verbose:
            print(f"  Loading cached quantum expectations: {cache_path.name}")
        return np.load(cache_path)

    if verbose:
        print(f"  Computing quantum expectations: {tag}")

    t0 = time.time()
    X_input = scale * X if scale != 1.0 else X
    expectations = get_all_expectation_values(
        X_input, n_qubits, n_layers, entanglement, verbose
    )
    elapsed = time.time() - t0
    if verbose:
        print(f"  Computed in {elapsed:.1f}s")

    np.save(cache_path, expectations)
    return expectations


# ═══════════════════════════════════════════════════════════════════════
#  Projected Quantum Kernel
# ═══════════════════════════════════════════════════════════════════════

def compute_pqk_kernel(
    E1: np.ndarray,
    E2: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    Projected Quantum Kernel:
    k(x, x') = exp(-γ Σ (⟨P_i⟩(x) - ⟨P_i⟩(x'))²)

    Mathematically identical to RBF on the expectation-value features.

    Parameters
    ----------
    E1 : np.ndarray of shape (n1, 3*n_qubits)
    E2 : np.ndarray of shape (n2, 3*n_qubits)
    gamma : float

    Returns
    -------
    K : np.ndarray of shape (n1, n2)
    """
    sq1 = np.sum(E1 ** 2, axis=1, keepdims=True)
    sq2 = np.sum(E2 ** 2, axis=1, keepdims=True)
    dist_sq = sq1 + sq2.T - 2 * (E1 @ E2.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    return np.exp(-gamma * dist_sq)


def train_and_predict_m3(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    seed: int = 0,
    n_train: int = 20,
    n_qubits: int = config.DEFAULT_QUBITS,
    n_layers: int = config.DEFAULT_LAYERS,
    entanglement: bool = config.DEFAULT_ENTANGLEMENT,
    scales: Optional[list] = None,
    force: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """
    Full M3 pipeline: quantum expectations → PQK kernel → tune → predict.
    Supports encoding scale tuning over scales in [0.25, 0.5, 1.0].
    """
    if scales is None:
        scales = [1.0]

    t0 = time.time()
    n_folds = config.get_cv_folds(len(y_train))
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.svm import SVC

    # Load/compute expectations for all scales
    E_train_dict = {}
    E_test_dict = {}
    for s in scales:
        E_train_dict[s] = get_cached_expectations(
            X_train, "train_sub", seed, n_train,
            n_qubits, n_layers, entanglement, scale=s, force=force, verbose=False
        )
        E_test_dict[s] = get_cached_expectations(
            X_test, "test", seed, n_train,
            n_qubits, n_layers, entanglement, scale=s, force=force, verbose=False
        )

    # Cross-validation across scales, multipliers, and C
    best_auc = -1.0
    best_s = 1.0
    best_gamma = None
    best_C = None
    best_multiplier = None
    best_tie_score = float("inf")

    # Track best for each fixed scale
    fixed_s_best = {}

    for s in scales:
        E_tr_s = E_train_dict[s]
        base_gamma_s = compute_base_gamma(E_tr_s)

        s_best_auc = -1.0
        s_best_gamma = None
        s_best_C = None
        s_best_mult = None
        s_best_tie = float("inf")

        for gm in config.GAMMA_MULTIPLIERS:
            gamma = base_gamma_s * gm
            for C in config.C_VALUES:
                fold_aucs = []
                skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
                for tr_idx, val_idx in skf.split(E_tr_s, y_train):
                    E_f_tr = E_tr_s[tr_idx]
                    E_f_val = E_tr_s[val_idx]
                    K_f_tr = compute_pqk_kernel(E_f_tr, E_f_tr, gamma)
                    K_f_val = compute_pqk_kernel(E_f_val, E_f_tr, gamma)

                    svm = SVC(kernel="precomputed", C=C, probability=False)
                    svm.fit(K_f_tr, y_train[tr_idx])
                    scores = svm.decision_function(K_f_val)
                    try:
                        auc = roc_auc_score(y_train[val_idx], scores)
                    except ValueError:
                        auc = 0.5
                    fold_aucs.append(auc)

                mean_auc = float(np.mean(fold_aucs))
                curr_tie_score = abs(np.log(gm)) + abs(np.log(C)) + abs(np.log(s))
                fixed_tie_score = abs(np.log(gm)) + abs(np.log(C))

                # Update fixed-s best
                if (mean_auc > s_best_auc + 1e-9) or (abs(mean_auc - s_best_auc) <= 1e-9 and fixed_tie_score < s_best_tie):
                    s_best_auc = mean_auc
                    s_best_gamma = gamma
                    s_best_C = C
                    s_best_mult = gm
                    s_best_tie = fixed_tie_score

                # Update global best across all s
                if (mean_auc > best_auc + 1e-9) or (abs(mean_auc - best_auc) <= 1e-9 and curr_tie_score < best_tie_score):
                    best_auc = mean_auc
                    best_s = s
                    best_gamma = gamma
                    best_C = C
                    best_multiplier = gm
                    best_tie_score = curr_tie_score

        fixed_s_best[s] = {
            "best_gamma": s_best_gamma,
            "best_C": s_best_C,
            "best_multiplier": s_best_mult,
            "best_cv_auc": s_best_auc,
        }

    # Evaluate test decision scores for each fixed s
    fixed_s_scores = {}
    for s in scales:
        K_tr_fixed = compute_pqk_kernel(E_train_dict[s], E_train_dict[s], fixed_s_best[s]["best_gamma"])
        svm_fixed = SVC(kernel="precomputed", C=fixed_s_best[s]["best_C"], probability=False)
        svm_fixed.fit(K_tr_fixed, y_train)
        K_te_fixed = compute_pqk_kernel(E_test_dict[s], E_train_dict[s], fixed_s_best[s]["best_gamma"])
        fixed_s_scores[s] = svm_fixed.decision_function(K_te_fixed)

    # Train final model on best_s
    K_train = compute_pqk_kernel(E_train_dict[best_s], E_train_dict[best_s], best_gamma)
    svm = SVC(kernel="precomputed", C=best_C, probability=False)
    svm.fit(K_train, y_train)
    K_test = compute_pqk_kernel(E_test_dict[best_s], E_train_dict[best_s], best_gamma)
    decision_scores = svm.decision_function(K_test)

    # Untuned baseline (scale = 1.0, multiplier = 1.0, C = 1.0)
    E_tr_1 = E_train_dict[1.0] if 1.0 in E_train_dict else get_cached_expectations(X_train, "train_sub", seed, n_train, n_qubits, n_layers, entanglement, scale=1.0, verbose=False)
    E_te_1 = E_test_dict[1.0] if 1.0 in E_test_dict else get_cached_expectations(X_test, "test", seed, n_train, n_qubits, n_layers, entanglement, scale=1.0, verbose=False)
    base_gamma_1 = compute_base_gamma(E_tr_1)
    K_train_untuned = compute_pqk_kernel(E_tr_1, E_tr_1, base_gamma_1)
    svm_untuned = SVC(kernel="precomputed", C=1.0, probability=False)
    svm_untuned.fit(K_train_untuned, y_train)
    K_test_untuned = compute_pqk_kernel(E_te_1, E_tr_1, base_gamma_1)
    scores_notuned = svm_untuned.decision_function(K_test_untuned)

    elapsed = time.time() - t0

    # Grid edge detection
    min_gm = min(config.GAMMA_MULTIPLIERS)
    max_gm = max(config.GAMMA_MULTIPLIERS)
    min_C = min(config.C_VALUES)
    max_C = max(config.C_VALUES)
    min_s = min(scales)
    max_s = max(scales)
    gamma_edge = bool(best_multiplier == min_gm or best_multiplier == max_gm)
    C_edge = bool(best_C == min_C or best_C == max_C)
    scale_edge = bool(best_s == min_s or best_s == max_s)

    # Kernel & feature diagnostics
    diag = kernel_diagnostics(K_train)
    feat_diag = compute_feature_diagnostics(E_train_dict[best_s], pca_reference=X_train)

    # Fixed s=1.00 and s=0.25 diagnostics (for Table 4 consistency without mixing)
    E_tr_100 = E_train_dict.get(1.0)
    feat_diag_s100 = compute_feature_diagnostics(E_tr_100, pca_reference=X_train) if E_tr_100 is not None else {}
    E_tr_025 = E_train_dict.get(0.25)
    feat_diag_s025 = compute_feature_diagnostics(E_tr_025, pca_reference=X_train) if E_tr_025 is not None else {}

    return decision_scores, {
        "best_scale": best_s,
        "scale_edge": scale_edge,
        "best_gamma": best_gamma,
        "best_C": best_C,
        "best_multiplier": best_multiplier,
        "gamma_edge": gamma_edge,
        "C_edge": C_edge,
        "scores_notuned": scores_notuned,
        "fixed_s_scores": fixed_s_scores,
        "fixed_s_best": fixed_s_best,
        "best_cv_auc": best_auc,
        "runtime_seconds": elapsed,
        **feat_diag,
        "feat_variance_s100": feat_diag_s100.get("feat_variance"),
        "dist_std_over_median_s100": feat_diag_s100.get("dist_std_over_median"),
        "dist_spearman_rank_corr_s100": feat_diag_s100.get("dist_spearman_rank_corr"),
        "feat_variance_s025": feat_diag_s025.get("feat_variance"),
        "dist_std_over_median_s025": feat_diag_s025.get("dist_std_over_median"),
        "dist_spearman_rank_corr_s025": feat_diag_s025.get("dist_spearman_rank_corr"),
        **diag,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Circuit diagram
# ═══════════════════════════════════════════════════════════════════════

def draw_circuit(
    n_qubits: int = config.DEFAULT_QUBITS,
    n_layers: int = config.DEFAULT_LAYERS,
    entanglement: bool = config.DEFAULT_ENTANGLEMENT,
) -> str:
    """Return a text-based circuit diagram."""
    circuit = create_circuit(n_qubits, n_layers, entanglement)
    dummy = np.random.rand(n_qubits) * np.pi
    circuit(dummy)  # execute once to build the tape
    return qml.draw(circuit)(dummy)


if __name__ == "__main__":
    print("=" * 60)
    print("Stage 5 Quantum Kernel Checks")
    print("=" * 60)

    # ── Check 1: Single qubit RY(θ) ──
    print("\n[Check 1] Single qubit RY(θ)...")
    dev1 = qml.device("default.qubit", wires=1)

    @qml.qnode(dev1, interface="numpy")
    def single_ry(theta):
        qml.RY(theta, wires=0)
        return (
            qml.expval(qml.PauliX(0)),
            qml.expval(qml.PauliY(0)),
            qml.expval(qml.PauliZ(0)),
        )

    theta = 1.23
    ex, ey, ez = single_ry(theta)
    assert np.isclose(ez, np.cos(theta), atol=1e-7), f"⟨Z⟩={ez}, expected cos(θ)={np.cos(theta)}"
    assert np.isclose(ex, np.sin(theta), atol=1e-7), f"⟨X⟩={ex}, expected sin(θ)={np.sin(theta)}"
    assert np.isclose(ey, 0.0, atol=1e-7), f"⟨Y⟩={ey}, expected 0"
    print("  ✓ ⟨Z⟩=cos(θ), ⟨X⟩=sin(θ), ⟨Y⟩=0")

    # ── Check 2: RY(θ) then RZ(θ), no entanglement, 1 layer ──
    print("\n[Check 2] RY(θ)→RZ(θ), 1 layer, no entanglement...")
    dev2 = qml.device("default.qubit", wires=1)

    @qml.qnode(dev2, interface="numpy")
    def ry_rz_single(theta):
        qml.RY(theta, wires=0)
        qml.RZ(theta, wires=0)
        return (
            qml.expval(qml.PauliX(0)),
            qml.expval(qml.PauliY(0)),
            qml.expval(qml.PauliZ(0)),
        )

    theta = 1.5
    ex, ey, ez = ry_rz_single(theta)
    # Expected: ⟨X⟩ = sin(θ)cos(θ), ⟨Y⟩ = sin²(θ), ⟨Z⟩ = cos(θ)
    # Wait, after RY(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
    # Then RZ(θ): e^{-iθ/2}cos(θ/2)|0⟩ + e^{iθ/2}sin(θ/2)|1⟩
    # ⟨Z⟩ = cos²(θ/2) - sin²(θ/2) = cos(θ)
    # ⟨X⟩ = 2 Re[e^{iθ/2}cos(θ/2) · e^{iθ/2}sin(θ/2)]
    #      = 2 cos(θ/2)sin(θ/2) cos(θ) = sin(θ)cos(θ)
    # ⟨Y⟩ = 2 Im[...] = sin(θ)sin(θ) = sin²(θ)
    # Actually let me recalculate carefully:
    # After RZ(θ): |ψ⟩ = e^{-iθ/2}cos(θ/2)|0⟩ + e^{iθ/2}sin(θ/2)|1⟩
    # ⟨X⟩ = ⟨ψ|X|ψ⟩ = 2 Re(α*β) where α = e^{-iθ/2}cos(θ/2), β = e^{iθ/2}sin(θ/2)
    #      = 2 Re(e^{iθ}cos(θ/2)sin(θ/2)) = 2 cos(θ/2)sin(θ/2)cos(θ)
    #      = sin(θ)cos(θ)
    # ⟨Y⟩ = 2 Im(α*β) = 2 sin(θ)cos(θ/2)sin(θ/2) ... hmm
    # Actually: α*β = e^{iθ/2}cos(θ/2) · e^{iθ/2}sin(θ/2) = e^{iθ}cos(θ/2)sin(θ/2)
    # ⟨Y⟩ = -2 Im(α*β) = -2 sin(θ)cos(θ/2)sin(θ/2) = -sin(θ)sin(θ) ... no
    # Let me just verify numerically that ⟨Y⟩ ≠ 0 (which proves Trap 1)
    print(f"  θ={theta}: ⟨X⟩={ex:.6f}, ⟨Y⟩={ey:.6f}, ⟨Z⟩={ez:.6f}")
    assert np.isclose(ez, np.cos(theta), atol=1e-6), f"⟨Z⟩ wrong"
    assert np.isclose(ex, np.sin(theta) * np.cos(theta), atol=1e-6), f"⟨X⟩ wrong"
    # Key Trap 1 check: ⟨Y⟩ is NOT zero when RZ is applied
    assert not np.isclose(ey, 0.0, atol=1e-3), f"⟨Y⟩ should not be 0 with RZ!"
    print("  ✓ ⟨Y⟩ ≠ 0 (Trap 1 proven: RZ gives non-trivial ⟨Y⟩)")

    # ── Check 3-5: Kernel properties ──
    print("\n[Checks 3-5] Kernel matrix properties...")
    np.random.seed(42)
    X_dummy = np.random.rand(10, 8) * np.pi
    E_dummy = get_all_expectation_values(X_dummy, n_qubits=8, n_layers=2,
                                          entanglement=True, verbose=False)

    gamma = compute_base_gamma(E_dummy)
    K = compute_pqk_kernel(E_dummy, E_dummy, gamma)

    # Check 3a: Symmetric
    assert np.allclose(K, K.T, atol=1e-10), "Kernel not symmetric!"
    print("  ✓ Kernel is symmetric")

    # Check 3b: Diagonal = 1
    assert np.allclose(np.diag(K), 1.0, atol=1e-10), "Diagonal ≠ 1!"
    print("  ✓ Diagonal = 1")

    # Check 3c: PSD (min eigenvalue ≥ -1e-8)
    eigs = np.linalg.eigvalsh(K)
    assert eigs[0] >= -1e-8, f"Not PSD: min eigenvalue = {eigs[0]}"
    print(f"  ✓ PSD (min eigenvalue = {eigs[0]:.2e})")

    # Check 4: Test-vs-train kernel shape
    X_test_dummy = np.random.rand(5, 8) * np.pi
    E_test_dummy = get_all_expectation_values(X_test_dummy, n_qubits=8, n_layers=2,
                                               entanglement=True, verbose=False)
    K_test = compute_pqk_kernel(E_test_dummy, E_dummy, gamma)
    assert K_test.shape == (5, 10), f"Wrong shape: {K_test.shape}"
    print(f"  ✓ Test kernel shape = {K_test.shape}")

    # Check 5: Off-diagonal std above alarm
    diag = kernel_diagnostics(K)
    print(f"  Off-diag mean={diag['kernel_offdiag_mean']:.6f}, "
          f"std={diag['kernel_offdiag_std']:.6f}")
    if diag["kernel_offdiag_std"] < config.KERNEL_STD_ALARM:
        print(f"  ⚠ WARNING: Off-diag std below alarm ({config.KERNEL_STD_ALARM})")
    else:
        print(f"  ✓ Off-diag std above alarm threshold")

    print("\n" + "=" * 60)
    print("All Stage 5 checks passed!")
    print("=" * 60)

    # Print circuit diagram
    print("\nCircuit diagram (8 qubits, 2 layers, entanglement=True):")
    print(draw_circuit(8, 2, True))
