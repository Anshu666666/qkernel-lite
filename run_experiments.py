"""
QKernel-Lite – Experiment Runner
==================================
Stage 6: Loop over seeds, N, models.
Append one row per run to results.csv.
Skip rows already done (crash-safe).
"""

import numpy as np
import pandas as pd
import torch
import time
from pathlib import Path
from typing import Optional

import config
from data import (
    load_pneumoniamnist,
    stratified_subset_indices,
    get_subset_data,
    print_split_info,
)
from features import extract_and_cache_features
from preprocess import preprocess_for_experiment
from classical import (
    train_m1, predict_m1,
    train_and_predict_m2,
)
from quantum import train_and_predict_m3
from evaluate import (
    evaluate_decision_scores,
    evaluate_probabilities,
    print_metrics,
)


from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def load_or_create_results() -> pd.DataFrame:
    """Load existing results.csv or create empty DataFrame."""
    if config.RESULTS_CSV.exists():
        return pd.read_csv(config.RESULTS_CSV)
    return pd.DataFrame(columns=config.RESULTS_COLUMNS)


def is_done(df: pd.DataFrame, model: str, n: int, seed: int,
            qubits: int = 0, entanglement: bool = False, layers: int = 0) -> bool:
    """Check if a specific run is already in results."""
    mask = (
        (df["model"] == model) &
        (df["N"] == n) &
        (df["seed"] == seed)
    )
    if model in ("M2", "M3", "M4"):
        mask = mask & (df["qubits"] == qubits)
    if model in ("M3", "M4"):
        mask = mask & (
            (df["entanglement"] == entanglement) &
            (df["layers"] == layers)
        )
    return mask.any()


def append_result(df: pd.DataFrame, row: dict) -> pd.DataFrame:
    """Append a row and save to CSV."""
    new_row = pd.DataFrame([row])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(config.RESULTS_CSV, index=False)
    return df


def run_m1_experiment(
    train_ds, test_ds,
    train_indices: np.ndarray,
    y_test: np.ndarray,
    seed: int, n: int,
) -> dict:
    """Run M1 (small CNN) experiment."""
    print(f"\n  ── M1 (CNN from scratch) | N={n}, seed={seed} ──")
    t0 = time.time()

    images_train, y_train = get_subset_data(train_ds, train_indices)
    images_test = test_ds.imgs

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, losses = train_m1(images_train, y_train, seed=seed, device=device)
    probs = predict_m1(model, images_test, device=device)
    metrics = evaluate_probabilities(y_test, probs)
    elapsed = time.time() - t0

    print_metrics(metrics, "M1")
    print(f"    Loss: start={losses[0]:.4f} → end={losses[-1]:.4f} | {elapsed:.1f}s")

    # Save model weights for seed=0 (for Streamlit)
    if seed == 0:
        model_path = config.CACHE_DIR / f"m1_weights_N{n}_s{seed}.pt"
        torch.save(model.state_dict(), model_path)

    return {
        "model": "M1", "N": n, "seed": seed,
        "qubits": 0, "entanglement": False, "layers": 0,
        **metrics,
        "auc_notuned": None,
        "best_scale": None, "scale_edge": False,
        "auc_s00625": None, "auc_s0125": None, "auc_s025": None, "auc_s050": None, "auc_s100": None,
        "best_gamma": None, "best_C": None,
        "gamma_edge": False, "C_edge": False,
        "feat_variance": None, "dist_std_over_median": None, "dist_spearman_rank_corr": None,
        "feat_variance_s100": None, "dist_std_over_median_s100": None, "dist_spearman_rank_corr_s100": None,
        "feat_variance_s025": None, "dist_std_over_median_s025": None, "dist_spearman_rank_corr_s025": None,
        "kernel_offdiag_mean": None, "kernel_offdiag_std": None,
        "kernel_min_eigenvalue": None,
        "runtime_seconds": elapsed,
    }


def run_m2_experiment(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
    seed: int, n: int,
    qubits: int = config.PCA_COMPONENTS,
) -> dict:
    """Run M2 (classical RBF-SVM) experiment."""
    q_str = f", q={qubits}" if qubits != config.PCA_COMPONENTS else ""
    print(f"\n  ── M2 (RBF-SVM){q_str} | N={n}, seed={seed} ──")
    t0 = time.time()

    X_tr = X_train_scaled[:, :qubits]
    X_te = X_test_scaled[:, :qubits]

    scores, info = train_and_predict_m2(X_tr, y_train, X_te, seed)
    metrics = evaluate_decision_scores(y_test, scores)
    auc_notuned = float(roc_auc_score(y_test, info["scores_notuned"])) if "scores_notuned" in info else None
    elapsed = time.time() - t0

    print_metrics(metrics, "M2")
    edge_str = " [EDGE HIT]" if info.get("gamma_edge") or info.get("C_edge") else ""
    print(f"    γ={info['best_gamma']:.6f}, C={info['best_C']:.1f}{edge_str} | Untuned AUC={auc_notuned:.4f} | {elapsed:.1f}s")

    fv = info.get("feat_variance")
    sp = info.get("dist_std_over_median")
    rho = info.get("dist_spearman_rank_corr")

    return {
        "model": "M2", "N": n, "seed": seed,
        "qubits": qubits, "entanglement": False, "layers": 0,
        **metrics,
        "auc_notuned": auc_notuned,
        "best_scale": 1.0, "scale_edge": False,
        "auc_s00625": None, "auc_s0125": None, "auc_s025": None, "auc_s050": None, "auc_s100": None,
        "best_gamma": info["best_gamma"], "best_C": info["best_C"],
        "gamma_edge": info.get("gamma_edge", False),
        "C_edge": info.get("C_edge", False),
        "feat_variance": fv,
        "dist_std_over_median": sp,
        "dist_spearman_rank_corr": rho,
        "feat_variance_s100": fv,
        "dist_std_over_median_s100": sp,
        "dist_spearman_rank_corr_s100": rho,
        "feat_variance_s025": fv,
        "dist_std_over_median_s025": sp,
        "dist_spearman_rank_corr_s025": rho,
        "kernel_offdiag_mean": info["kernel_offdiag_mean"],
        "kernel_offdiag_std": info["kernel_offdiag_std"],
        "kernel_min_eigenvalue": info["kernel_min_eigenvalue"],
        "runtime_seconds": elapsed,
    }


def run_m3_experiment(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
    seed: int, n: int,
    qubits: int = config.DEFAULT_QUBITS,
    layers: int = config.DEFAULT_LAYERS,
    entanglement: bool = config.DEFAULT_ENTANGLEMENT,
    scales: Optional[list] = None,
    model_name: str = "M3",
) -> dict:
    """Run M3/M4 (quantum kernel SVM) experiment."""
    ent_str = "ent" if entanglement else "no-ent"
    print(f"\n  ── {model_name} (PQK-SVM) | N={n}, seed={seed}, "
          f"q={qubits}, L={layers}, {ent_str} ──")

    # Only use as many features as qubits
    X_tr = X_train_scaled[:, :qubits]
    X_te = X_test_scaled[:, :qubits]

    t0 = time.time()
    scores, info = train_and_predict_m3(
        X_tr, y_train, X_te,
        seed=seed, n_train=n,
        n_qubits=qubits, n_layers=layers,
        entanglement=entanglement,
        scales=scales,
    )
    metrics = evaluate_decision_scores(y_test, scores)
    auc_notuned = float(roc_auc_score(y_test, info["scores_notuned"])) if "scores_notuned" in info else None

    fixed_s = info.get("fixed_s_scores", {})
    auc_s00625 = float(roc_auc_score(y_test, fixed_s[0.0625])) if 0.0625 in fixed_s else None
    auc_s0125 = float(roc_auc_score(y_test, fixed_s[0.125])) if 0.125 in fixed_s else None
    auc_s025 = float(roc_auc_score(y_test, fixed_s[0.25])) if 0.25 in fixed_s else None
    auc_s050 = float(roc_auc_score(y_test, fixed_s[0.5])) if 0.5 in fixed_s else None
    auc_s100 = float(roc_auc_score(y_test, fixed_s[1.0])) if 1.0 in fixed_s else None

    elapsed = time.time() - t0

    print_metrics(metrics, model_name)
    edge_str = " [EDGE HIT]" if info.get("gamma_edge") or info.get("C_edge") or info.get("scale_edge") else ""
    best_sc = info.get("best_scale", 1.0)
    print(f"    Best scale={best_sc}, γ={info['best_gamma']:.6f}, C={info['best_C']:.1f}{edge_str} | Untuned AUC={auc_notuned:.4f} | {elapsed:.1f}s")
    if auc_s025 is not None:
        print(f"    Fixed scale AUCs: s=0.0625: {auc_s00625}, s=0.125: {auc_s0125}, s=0.25: {auc_s025:.4f}, s=0.50: {auc_s050:.4f}, s=1.00: {auc_s100:.4f}")
    print(f"    Kernel off-diag std={info['kernel_offdiag_std']:.6f}")

    return {
        "model": model_name, "N": n, "seed": seed,
        "qubits": qubits, "entanglement": entanglement, "layers": layers,
        **metrics,
        "auc_notuned": auc_notuned,
        "best_scale": best_sc,
        "scale_edge": info.get("scale_edge", False),
        "auc_s00625": auc_s00625,
        "auc_s0125": auc_s0125,
        "auc_s025": auc_s025,
        "auc_s050": auc_s050,
        "auc_s100": auc_s100,
        "best_gamma": info["best_gamma"], "best_C": info["best_C"],
        "gamma_edge": info.get("gamma_edge", False),
        "C_edge": info.get("C_edge", False),
        "feat_variance": info.get("feat_variance"),
        "dist_std_over_median": info.get("dist_std_over_median"),
        "dist_spearman_rank_corr": info.get("dist_spearman_rank_corr"),
        "feat_variance_s100": info.get("feat_variance_s100"),
        "dist_std_over_median_s100": info.get("dist_std_over_median_s100"),
        "dist_spearman_rank_corr_s100": info.get("dist_spearman_rank_corr_s100"),
        "feat_variance_s025": info.get("feat_variance_s025"),
        "dist_std_over_median_s025": info.get("dist_std_over_median_s025"),
        "dist_spearman_rank_corr_s025": info.get("dist_spearman_rank_corr_s025"),
        "kernel_offdiag_mean": info["kernel_offdiag_mean"],
        "kernel_offdiag_std": info["kernel_offdiag_std"],
        "kernel_min_eigenvalue": info["kernel_min_eigenvalue"],
        "runtime_seconds": elapsed,
    }


def run_all_experiments(
    n_values: list = config.N_VALUES,
    seeds: list = config.SEEDS,
    run_m1: bool = True,
    run_m2: bool = True,
    run_m3: bool = True,
    run_ablations: bool = True,
):
    """
    Main experiment loop. Crash-safe: skips completed runs.
    """
    print("=" * 70)
    print("QKernel-Lite Experiment Runner")
    print("=" * 70)

    # Load data
    train_ds, val_ds, test_ds, info = load_pneumoniamnist()
    print_split_info(train_ds, val_ds, test_ds)

    # Extract and cache features
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_train_full = extract_and_cache_features(train_ds, "train", device)
    X_test_full = extract_and_cache_features(test_ds, "test", device)
    y_test = test_ds.labels.flatten()

    print(f"\nTest set: {len(y_test)} images, "
          f"Normal={np.sum(y_test==0)}, Pneumonia={np.sum(y_test==1)}")

    results_df = load_or_create_results()

    for n in n_values:
        for seed in seeds:
            print(f"\n{'='*70}")
            print(f"N={n}, seed={seed}")
            print(f"{'='*70}")

            # Get subset
            train_indices = stratified_subset_indices(train_ds, n=n, seed=seed)
            _, y_train = get_subset_data(train_ds, train_indices)

            # PCA + scaling
            X_train_scaled, X_test_scaled, pp = preprocess_for_experiment(
                X_train_full, X_test_full, train_indices,
                n_components=config.PCA_COMPONENTS,
            )
            pp.print_variance_report()

            # ── M1 ──
            if run_m1 and not is_done(results_df, "M1", n, seed):
                row = run_m1_experiment(
                    train_ds, test_ds, train_indices, y_test, seed, n
                )
                results_df = append_result(results_df, row)

            # ── M2 (8 components) ──
            if run_m2 and not is_done(results_df, "M2", n, seed, qubits=config.PCA_COMPONENTS):
                row = run_m2_experiment(
                    X_train_scaled, y_train, X_test_scaled, y_test, seed, n,
                    qubits=config.PCA_COMPONENTS,
                )
                results_df = append_result(results_df, row)

            # ── M3 (main: 8 qubits, 2 layers, entanglement) with scale tuning ──
            if run_m3 and not is_done(results_df, "M3", n, seed,
                                      qubits=config.DEFAULT_QUBITS,
                                      entanglement=config.DEFAULT_ENTANGLEMENT,
                                      layers=config.DEFAULT_LAYERS):
                row = run_m3_experiment(
                    X_train_scaled, y_train, X_test_scaled, y_test,
                    seed, n,
                    qubits=config.DEFAULT_QUBITS,
                    layers=config.DEFAULT_LAYERS,
                    entanglement=config.DEFAULT_ENTANGLEMENT,
                    scales=config.ENCODING_SCALES,
                    model_name="M3",
                )
                results_df = append_result(results_df, row)

            # ── M4 ablations & matching M2 baselines (Full 2x2: L in {1, 2} x ent in {False, True}) ──
            if run_ablations and seed in config.ABLATION_SEEDS:
                for q in config.QUBIT_ABLATION:
                    # Matching M2 baseline at component count q
                    if q != config.PCA_COMPONENTS:
                        if not is_done(results_df, "M2", n, seed, qubits=q):
                            row = run_m2_experiment(
                                X_train_scaled, y_train, X_test_scaled, y_test, seed, n,
                                qubits=q,
                            )
                            results_df = append_result(results_df, row)

                    for L in config.LAYER_ABLATION:
                        for ent in config.ENTANGLEMENT_ABLATION:
                            # Skip the main M3 config (already run with scale tuning)
                            if (q == config.DEFAULT_QUBITS and
                                L == config.DEFAULT_LAYERS and
                                ent == config.DEFAULT_ENTANGLEMENT):
                                continue

                            if not is_done(results_df, "M4", n, seed,
                                          qubits=q, entanglement=ent, layers=L):
                                row = run_m3_experiment(
                                    X_train_scaled, y_train,
                                    X_test_scaled, y_test,
                                    seed, n,
                                    qubits=q,
                                    layers=L,
                                    entanglement=ent,
                                    scales=config.ENCODING_SCALES,
                                    model_name="M4",
                                )
                                results_df = append_result(results_df, row)

    print(f"\n{'='*70}")
    print(f"All experiments complete! Results saved to: {config.RESULTS_CSV}")
    print(f"Total rows: {len(results_df)}")
    print(f"{'='*70}")

    return results_df


def run_mini_grid():
    """Quick mini-grid for Stage 6 pass/fail check."""
    return run_all_experiments(
        n_values=[20],
        seeds=[0, 1],
        run_m1=True,
        run_m2=True,
        run_m3=True,
        run_ablations=False,
    )


if __name__ == "__main__":
    import sys
    if "--mini" in sys.argv:
        run_mini_grid()
    else:
        run_all_experiments()
