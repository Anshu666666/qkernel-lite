"""
Stage 0 Smoke Test
===================
Verifies all imports work, PennyLane runs a 2-qubit circuit,
PneumoniaMNIST downloads, and ResNet18 weights load.
"""

import pytest
import numpy as np


def test_imports():
    """All key packages import without error."""
    import torch
    import torchvision
    import medmnist
    import pennylane
    import sklearn
    import pandas
    import matplotlib
    import streamlit
    print(f"  torch={torch.__version__}")
    print(f"  pennylane={pennylane.__version__}")
    print(f"  medmnist={medmnist.__version__}")
    print(f"  sklearn={sklearn.__version__}")


def test_pennylane_2qubit_circuit():
    """A 2-qubit PennyLane circuit runs and returns expected values."""
    import pennylane as qml

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

    z0, z1 = circuit()
    # Bell state: ⟨Z⟩ = 0 for both qubits
    assert abs(z0) < 1e-6, f"⟨Z₀⟩ = {z0}, expected 0"
    assert abs(z1) < 1e-6, f"⟨Z₁⟩ = {z1}, expected 0"
    print("  ✓ 2-qubit Bell state: ⟨Z₀⟩≈0, ⟨Z₁⟩≈0")


def test_resnet18_loads():
    """ResNet18 ImageNet weights load successfully."""
    import torchvision.models as models
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    # Count params
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 11_000_000, f"Too few params: {n_params}"
    print(f"  ✓ ResNet18 loaded: {n_params:,} parameters")


def test_pneumoniamnist_loads():
    """PneumoniaMNIST downloads and has expected structure."""
    from data import load_pneumoniamnist, get_class_counts

    train_ds, val_ds, test_ds, info = load_pneumoniamnist()

    assert len(train_ds.labels) > 0, "Empty training set"
    assert len(test_ds.labels) > 0, "Empty test set"

    # Check binary classification
    unique_labels = np.unique(train_ds.labels)
    assert set(unique_labels.tolist()) == {0, 1}, \
        f"Expected binary labels, got {unique_labels}"

    counts = get_class_counts(train_ds)
    print(f"  ✓ PneumoniaMNIST loaded: {len(train_ds.labels)} train, "
          f"{len(test_ds.labels)} test | counts={counts}")


def test_config_module():
    """Config values are sensible."""
    import config

    assert config.PCA_COMPONENTS == 8
    assert config.ANGLE_MIN == 0.0
    assert np.isclose(config.ANGLE_MAX, np.pi)
    assert config.M1_EPOCHS == 60
    assert len(config.N_VALUES) == 4
    assert config.get_cv_folds(10) == 3
    assert config.get_cv_folds(50) == 5
    print("  ✓ Config values valid")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
