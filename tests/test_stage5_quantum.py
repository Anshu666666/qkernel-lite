"""
Stage 5 Tests: Quantum Kernel
================================
All 6 checks from the build plan:
  1. Single qubit RY(θ): ⟨Z⟩=cos(θ), ⟨X⟩=sin(θ), ⟨Y⟩=0
  2. RY(θ)→RZ(θ), no ent, 1 layer: ⟨X⟩=sin(θ)cos(θ), ⟨Y⟩≠0, ⟨Z⟩=cos(θ)
  3. Kernel is symmetric, diagonal=1, PSD (min eigenvalue ≥ -1e-8)
  4. Test-vs-train kernel has correct shape
  5. Off-diagonal std above alarm level at n=8
  6. Performance is reasonable (not hours)
"""

import pytest
import numpy as np
import pennylane as qml
import time
from quantum import (
    create_circuit,
    get_all_expectation_values,
    compute_pqk_kernel,
)
from classical import compute_base_gamma, kernel_diagnostics
import config


def test_check1_single_ry():
    """Check 1: Single qubit RY(θ) expectations."""
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev, interface="numpy")
    def circuit(theta):
        qml.RY(theta, wires=0)
        return (
            qml.expval(qml.PauliX(0)),
            qml.expval(qml.PauliY(0)),
            qml.expval(qml.PauliZ(0)),
        )

    for theta in [0.5, 1.0, 1.5, 2.0, np.pi/4]:
        ex, ey, ez = circuit(theta)
        assert np.isclose(ez, np.cos(theta), atol=1e-7), \
            f"⟨Z⟩={ez}, expected cos({theta})={np.cos(theta)}"
        assert np.isclose(ex, np.sin(theta), atol=1e-7), \
            f"⟨X⟩={ex}, expected sin({theta})={np.sin(theta)}"
        assert np.isclose(ey, 0.0, atol=1e-7), \
            f"⟨Y⟩={ey}, expected 0"


def test_check2_ry_rz_trap1():
    """Check 2: RY→RZ proves Trap 1 (⟨Y⟩ ≠ 0)."""
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev, interface="numpy")
    def circuit(theta):
        qml.RY(theta, wires=0)
        qml.RZ(theta, wires=0)
        return (
            qml.expval(qml.PauliX(0)),
            qml.expval(qml.PauliY(0)),
            qml.expval(qml.PauliZ(0)),
        )

    theta = 1.5
    ex, ey, ez = circuit(theta)

    assert np.isclose(ez, np.cos(theta), atol=1e-6), \
        f"⟨Z⟩ wrong: {ez}"
    assert np.isclose(ex, np.sin(theta) * np.cos(theta), atol=1e-6), \
        f"⟨X⟩ wrong: {ex}"
    assert not np.isclose(ey, 0.0, atol=1e-3), \
        "⟨Y⟩ = 0! Trap 1 not fixed: RZ should make ⟨Y⟩ non-trivial"


def test_check3_kernel_properties():
    """Check 3: Kernel is symmetric, diagonal=1, PSD."""
    np.random.seed(42)
    X = np.random.rand(15, 8) * np.pi
    E = get_all_expectation_values(X, n_qubits=8, n_layers=2,
                                    entanglement=True, verbose=False)

    gamma = compute_base_gamma(E)
    K = compute_pqk_kernel(E, E, gamma)

    # Symmetric
    assert np.allclose(K, K.T, atol=1e-10), "Not symmetric"

    # Diagonal = 1
    assert np.allclose(np.diag(K), 1.0, atol=1e-10), "Diagonal ≠ 1"

    # PSD
    eigs = np.linalg.eigvalsh(K)
    assert eigs[0] >= -1e-8, f"Not PSD: min eigenvalue = {eigs[0]}"


def test_check4_test_kernel_shape():
    """Check 4: Test-vs-train kernel shape is correct."""
    np.random.seed(42)
    X_train = np.random.rand(10, 8) * np.pi
    X_test = np.random.rand(5, 8) * np.pi

    E_train = get_all_expectation_values(X_train, n_qubits=8, n_layers=2,
                                          entanglement=True, verbose=False)
    E_test = get_all_expectation_values(X_test, n_qubits=8, n_layers=2,
                                         entanglement=True, verbose=False)

    gamma = compute_base_gamma(E_train)
    K = compute_pqk_kernel(E_test, E_train, gamma)
    assert K.shape == (5, 10), f"Shape {K.shape}, expected (5, 10)"


def test_check5_offdiag_std():
    """Check 5: Off-diagonal std above alarm at n=8."""
    np.random.seed(42)
    X = np.random.rand(15, 8) * np.pi
    E = get_all_expectation_values(X, n_qubits=8, n_layers=2,
                                    entanglement=True, verbose=False)

    gamma = compute_base_gamma(E)
    K = compute_pqk_kernel(E, E, gamma)
    diag = kernel_diagnostics(K)

    print(f"  Off-diag std = {diag['kernel_offdiag_std']:.6f}")
    assert diag["kernel_offdiag_std"] > config.KERNEL_STD_ALARM, \
        f"Off-diag std {diag['kernel_offdiag_std']} < alarm {config.KERNEL_STD_ALARM}"


def test_check6_performance():
    """Check 6: Computing expectations doesn't take hours."""
    np.random.seed(42)
    X = np.random.rand(20, 8) * np.pi

    t0 = time.time()
    E = get_all_expectation_values(X, n_qubits=8, n_layers=2,
                                    entanglement=True, verbose=False)
    elapsed = time.time() - t0

    assert E.shape == (20, 24), f"Shape {E.shape}"
    print(f"  20 images × 8 qubits: {elapsed:.2f}s")
    assert elapsed < 120, f"Too slow: {elapsed:.0f}s for 20 images"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
