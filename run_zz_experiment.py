"""
Run ZZ Correlator Experiment
Tests whether adding two-qubit ZZ correlators ⟨Z_i Z_{i+1}⟩ to the 
single-qubit ⟨X_i⟩, ⟨Y_i⟩, ⟨Z_i⟩ expectation values restores performance 
for the 8-qubit 2-layer entangled circuit.
"""
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import pennylane as qml
import time

import config
from data import load_pneumoniamnist, stratified_subset_indices, get_subset_data
from features import extract_and_cache_features
from preprocess import preprocess_for_experiment
from classical import compute_base_gamma

def create_circuit_with_zz(n_qubits=8, n_layers=2):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="numpy")
    def circuit(features):
        for _ in range(n_layers):
            for i in range(n_qubits):
                qml.RY(features[:, i], wires=i)
                qml.RZ(features[:, i], wires=i)
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
        
        obs = []
        # Single-qubit observables
        for i in range(n_qubits):
            obs.append(qml.expval(qml.PauliX(i)))
            obs.append(qml.expval(qml.PauliY(i)))
            obs.append(qml.expval(qml.PauliZ(i)))
        # Two-qubit ZZ correlators along CNOT ring
        for i in range(n_qubits):
            obs.append(qml.expval(qml.PauliZ(i) @ qml.PauliZ((i + 1) % n_qubits)))
        return obs

    return circuit

def run_zz_comparison(seeds=range(5), n_values=[10, 20, 50, 100]):
    print("Loading data...")
    train_ds, val_ds, test_ds, info = load_pneumoniamnist()
    X_train_full = extract_and_cache_features(train_ds, "train", "cpu")
    X_test_full = extract_and_cache_features(test_ds, "test", "cpu")
    y_test = test_ds.labels.flatten()

    circ = create_circuit_with_zz(8, 2)

    results = []

    for n in n_values:
        for seed in seeds:
            train_idx = stratified_subset_indices(train_ds, n=n, seed=seed)
            y_train = train_ds.labels.flatten()[train_idx]

            X_tr_sc, X_te_sc, _ = preprocess_for_experiment(
                X_train_full, X_test_full, train_idx, n_components=8
            )

            # Evaluate at fixed s = 0.25 (bandwidth parity) and s = 1.00 (default)
            for scale in [0.25, 1.00]:
                X_tr_in = scale * X_tr_sc
                X_te_in = scale * X_te_sc

                raw_tr = circ(X_tr_in)
                E_tr_full = np.column_stack([np.array(o) for o in raw_tr])

                raw_te = circ(X_te_in)
                E_te_full = np.column_stack([np.array(o) for o in raw_te])

                # 1. Single-qubit features only (first 24 features: 3*8)
                E_tr_single = E_tr_full[:, :24]
                E_te_single = E_te_full[:, :24]

                # 2. Single-qubit + ZZ correlators (all 32 features: 24 + 8)
                E_tr_zz = E_tr_full
                E_te_zz = E_te_full

                for feat_type, E_tr, E_te in [("XYZ_only", E_tr_single, E_te_single), ("XYZ_plus_ZZ", E_tr_zz, E_te_zz)]:
                    gamma = compute_base_gamma(E_tr)
                    diff_tr = E_tr[:, np.newaxis, :] - E_tr[np.newaxis, :, :]
                    K_tr = np.exp(-gamma * np.sum(diff_tr ** 2, axis=-1))

                    diff_te = E_te[:, np.newaxis, :] - E_tr[np.newaxis, :, :]
                    K_te = np.exp(-gamma * np.sum(diff_te ** 2, axis=-1))

                    svm = SVC(kernel="precomputed", C=1.0)
                    svm.fit(K_tr, y_train)
                    scores = svm.decision_function(K_te)
                    auc = float(roc_auc_score(y_test, scores))

                    results.append({
                        "N": n, "seed": seed, "scale": scale,
                        "feature_type": feat_type, "auc": auc
                    })

    df_res = pd.DataFrame(results)
    df_res.to_csv("results/zz_experiment_results.csv", index=False)
    print("\nSaved ZZ results to results/zz_experiment_results.csv")

    print("\nSummary of ZZ Experiment (Mean AUC ± Std):")
    summary = df_res.groupby(["scale", "feature_type", "N"])["auc"].agg(["mean", "std"]).reset_index()
    print(summary.to_string(index=False))

if __name__ == "__main__":
    run_zz_comparison(seeds=range(5), n_values=[10, 20, 50, 100])
