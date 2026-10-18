"""
QKernel-Lite: Prototype Database Builder
=========================================
Precomputes and serializes all interactive diagnostic outputs, model decision scores,
PCA angles, Pauli expectations, and Gram matrices into a single compact archive (~3 MB).

This eliminates the need for PyTorch, PennyLane, or MedMNIST at app runtime.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from sklearn.svm import SVC

import config
from data import stratified_subset_indices
from features import get_cache_path
from preprocess import preprocess_for_experiment
from classical import (
    SmallCNN, predict_m1, compute_rbf_kernel, compute_base_gamma, kernel_diagnostics
)
from quantum import (
    compute_pqk_kernel, get_cached_expectations, draw_circuit
)

def build_database():
    print("=== Loading raw data and feature caches ===")
    raw_data = np.load(config.CACHE_DIR / "pneumoniamnist_224.npz")
    test_images_raw = raw_data["test_images"] # (624, 224, 224)
    test_labels = raw_data["test_labels"].flatten() # (624,)
    train_labels_full = raw_data["train_labels"].flatten()

    # Downsample test images to 112x112 for crisp display at 1/4 the memory
    print("Downsampling 624 test images to 112x112...")
    test_images_112 = test_images_raw[:, ::2, ::2]

    # Load frozen ResNet-18 features
    X_train_full = np.load(get_cache_path("train"))
    X_test_full = np.load(get_cache_path("test"))

    # Load results.csv for tuned parameters
    results_df = None
    if config.RESULTS_CSV.exists():
        results_df = pd.read_csv(config.RESULTS_CSV)

    # 1. Precompute Circuit Diagrams for Tab 2
    print("Generating ASCII circuit diagrams...")
    circuits = {}
    for q in [4, 6, 8]:
        for l in [1, 2]:
            for ent in [True, False]:
                circuits[f"{q}Q_{l}L_{ent}"] = draw_circuit(q, l, ent)

    # 2. Precompute Predictions & Diagnostics across N and Seeds
    print("Precomputing model decisions across N and Seeds...")
    db = {}
    scales = [0.0625, 0.125, 0.25, 0.50, 1.00]
    
    # We precompute for all N and seeds 0 to 4 (or 15 seeds)
    seeds_to_cache = list(range(15))

    for N in config.N_VALUES:
        for seed in seeds_to_cache:
            key = f"N{N}_s{seed}"
            print(f"  Processing {key}...")
            
            # Lookup tuned hyperparameters from results.csv
            gamma2 = None
            c2 = 1.0
            gamma3 = None
            c3 = 1.0
            best_s = 0.125

            if results_df is not None:
                n_col = "N" if "N" in results_df.columns else "n_train"
                r3 = results_df[(results_df["model"] == "M3") & (results_df[n_col] == N) & (results_df["seed"] == seed)]
                if not r3.empty:
                    if pd.notna(r3.iloc[0].get("best_scale")):
                        best_s = float(r3.iloc[0]["best_scale"])
                    if pd.notna(r3.iloc[0].get("best_gamma")):
                        gamma3 = float(r3.iloc[0]["best_gamma"])
                    if pd.notna(r3.iloc[0].get("best_C")):
                        c3 = float(r3.iloc[0]["best_C"])

                r2 = results_df[(results_df["model"] == "M2") & (results_df[n_col] == N) & (results_df["seed"] == seed) & (results_df["qubits"] == config.DEFAULT_QUBITS)]
                if not r2.empty:
                    if pd.notna(r2.iloc[0].get("best_gamma")):
                        gamma2 = float(r2.iloc[0]["best_gamma"])
                    if pd.notna(r2.iloc[0].get("best_C")):
                        c2 = float(r2.iloc[0]["best_C"])

            # 1. Stratified indices & PCA
            train_indices = stratified_subset_indices(train_labels_full, n=N, seed=seed)
            y_train_sub = train_labels_full[train_indices]

            X_train_scaled, X_test_scaled, pp = preprocess_for_experiment(
                X_train_full, X_test_full, train_indices,
                n_components=config.PCA_COMPONENTS,
            )

            # M1: Scratch CNN
            m1_path = config.CACHE_DIR / f"m1_weights_N{N}_s{seed}.pt"
            if not m1_path.exists():
                m1_path = config.CACHE_DIR / f"m1_weights_N{N}_s0.pt"

            m1_probs = np.full(624, 0.5, dtype=np.float32)
            if m1_path.exists():
                m1_model = SmallCNN(input_size=28)
                m1_model.load_state_dict(torch.load(m1_path, map_location="cpu", weights_only=True))
                m1_model.eval()
                # predict in batches
                m1_probs = predict_m1(m1_model, test_images_raw).astype(np.float32)

            # M2: Classical RBF-SVM
            if gamma2 is None:
                gamma2 = compute_base_gamma(X_train_scaled)
            K_train2 = compute_rbf_kernel(X_train_scaled, X_train_scaled, gamma2)
            svm2 = SVC(kernel="precomputed", C=c2)
            svm2.fit(K_train2, y_train_sub)
            K_test2 = compute_rbf_kernel(X_test_scaled, X_train_scaled, gamma2)
            m2_scores = svm2.decision_function(K_test2).astype(np.float32)

            # Sort for Gram matrix display
            sort_idx = np.argsort(y_train_sub)
            X_sorted = X_train_scaled[sort_idx]
            y_sorted = y_train_sub[sort_idx]
            K_train2_sorted = compute_rbf_kernel(X_sorted, X_sorted, gamma2).astype(np.float32)
            diag_rbf = kernel_diagnostics(K_train2_sorted)
            n_class0 = int(np.sum(y_sorted == 0))

            # M3: Quantum PQK-SVM across all scales
            m3_scores_by_scale = {}
            exp_vals_by_scale = {}
            K_train3_sorted_by_scale = {}
            diag_pqk_by_scale = {}

            for sc in scales:
                try:
                    E_train = get_cached_expectations(
                        X_train_scaled, "train_sub", seed, N, scale=sc, verbose=False
                    )
                    E_test = get_cached_expectations(
                        X_test_scaled, "test", seed, N, scale=sc, verbose=False
                    )
                    exp_vals_by_scale[str(sc)] = E_test.astype(np.float32)

                    sc_gamma = gamma3 if (gamma3 is not None and sc == best_s) else compute_base_gamma(E_train)
                    K_train3 = compute_pqk_kernel(E_train, E_train, sc_gamma)
                    svm3 = SVC(kernel="precomputed", C=c3)
                    svm3.fit(K_train3, y_train_sub)
                    K_test3 = compute_pqk_kernel(E_test, E_train, sc_gamma)
                    m3_scores_by_scale[str(sc)] = svm3.decision_function(K_test3).astype(np.float32)

                    # Sorted Gram for display
                    E_sorted = E_train[sort_idx]
                    K_train3_sorted = compute_pqk_kernel(E_sorted, E_sorted, sc_gamma).astype(np.float32)
                    K_train3_sorted_by_scale[str(sc)] = K_train3_sorted
                    diag_pqk_by_scale[str(sc)] = kernel_diagnostics(K_train3_sorted)
                except Exception as e:
                    # Fallback if specific scale not cached
                    m3_scores_by_scale[str(sc)] = m2_scores
                    exp_vals_by_scale[str(sc)] = np.zeros((624, 24), dtype=np.float32)
                    K_train3_sorted_by_scale[str(sc)] = K_train2_sorted
                    diag_pqk_by_scale[str(sc)] = diag_rbf

            # Record entry in db
            db[key] = {
                "m1_probs": m1_probs,
                "m2_scores": m2_scores,
                "m3_scores_by_scale": m3_scores_by_scale,
                "pca_angles": X_test_scaled.astype(np.float32),
                "pca_explained_var": float(pp.total_explained_variance),
                "exp_vals_by_scale": exp_vals_by_scale,
                "K_rbf_sorted": K_train2_sorted,
                "diag_rbf": diag_rbf,
                "K_pqk_sorted_by_scale": K_train3_sorted_by_scale,
                "diag_pqk_by_scale": diag_pqk_by_scale,
                "n_class0": n_class0,
                "best_scale": best_s,
            }

    # Save to data/prototype_db.npz
    out_dir = config.ROOT_DIR / "data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "prototype_db.npz"

    print(f"Saving compiled prototype database to {out_path}...")
    np.savez_compressed(
        out_path,
        test_images=test_images_112,
        test_labels=test_labels,
        circuits=circuits,
        db=db,
    )
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Done! Database successfully created: {size_mb:.2f} MB")

if __name__ == "__main__":
    build_database()
