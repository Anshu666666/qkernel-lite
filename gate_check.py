"""
Gate Verification Script
========================
Compares Unentangled 1-layer M3 versus M2 (8 components) across all seeds and N.
Uses median-heuristic base gamma and wide grid for both.
Computes t-based 95% confidence interval on paired difference (M3 - M2).
"""

import numpy as np
import pandas as pd
from scipy import stats
import time
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

import config
from data import load_pneumoniamnist, stratified_subset_indices, get_subset_data
from features import extract_and_cache_features
from preprocess import preprocess_for_experiment
from classical import compute_rbf_kernel, tune_svm, compute_base_gamma
from quantum import create_circuit, compute_pqk_kernel, get_cached_expectations

def run_gate():
    print("=" * 70)
    print("GATE CHECK: Unentangled 1-Layer M3 vs M2 (8 components)")
    print("Evaluating across N in {10, 20, 50, 100} and seeds 0..4 (20 runs)")
    print("=" * 70)

    train_ds, val_ds, test_ds, _ = load_pneumoniamnist()
    device = "cpu"
    X_train_full = extract_and_cache_features(train_ds, "train", device)
    X_test_full = extract_and_cache_features(test_ds, "test", device)
    y_test = test_ds.labels.flatten()

    records = []

    for n in config.N_VALUES:
        for seed in config.SEEDS:
            t0 = time.time()
            train_indices = stratified_subset_indices(train_ds, n=n, seed=seed)
            _, y_train = get_subset_data(train_ds, train_indices)

            X_tr, X_te, _ = preprocess_for_experiment(
                X_train_full, X_test_full, train_indices, n_components=8
            )

            # ── M2 (8 components) ──
            base_gamma_m2 = compute_base_gamma(X_tr)
            tune_m2 = tune_svm(
                K_train=None, y_train=y_train, X_train_for_kernel=X_tr,
                gamma_multipliers=config.GAMMA_MULTIPLIERS,
                C_values=config.C_VALUES,
                base_gamma=base_gamma_m2,
                seed=seed,
            )
            K_tr_m2 = compute_rbf_kernel(X_tr, X_tr, tune_m2["best_gamma"])
            svm_m2 = SVC(kernel="precomputed", C=tune_m2["best_C"])
            svm_m2.fit(K_tr_m2, y_train)
            K_te_m2 = compute_rbf_kernel(X_te, X_tr, tune_m2["best_gamma"])
            auc_m2 = roc_auc_score(y_test, svm_m2.decision_function(K_te_m2))

            # ── Unentangled 1-layer M3 (8 qubits, L=1, ent=False) ──
            E_tr = get_cached_expectations(
                X_tr, "train_sub", seed, n, n_qubits=8, n_layers=1, entanglement=False, verbose=False
            )
            E_te = get_cached_expectations(
                X_te, "test", seed, n, n_qubits=8, n_layers=1, entanglement=False, verbose=False
            )

            base_gamma_m3 = compute_base_gamma(E_tr)
            tune_m3 = tune_svm(
                K_train=None, y_train=y_train, X_train_for_kernel=E_tr,
                kernel_fn=compute_pqk_kernel,
                gamma_multipliers=config.GAMMA_MULTIPLIERS,
                C_values=config.C_VALUES,
                base_gamma=base_gamma_m3,
                seed=seed,
            )
            K_tr_m3 = compute_pqk_kernel(E_tr, E_tr, tune_m3["best_gamma"])
            svm_m3 = SVC(kernel="precomputed", C=tune_m3["best_C"])
            svm_m3.fit(K_tr_m3, y_train)
            K_te_m3 = compute_pqk_kernel(E_te, E_tr, tune_m3["best_gamma"])
            auc_m3 = roc_auc_score(y_test, svm_m3.decision_function(K_te_m3))

            diff = auc_m3 - auc_m2
            elapsed = time.time() - t0

            records.append({
                "N": n, "seed": seed,
                "auc_m2": auc_m2, "auc_m3_unent1L": auc_m3,
                "diff": diff, "elapsed": elapsed,
            })
            print(f"  N={n:3d}, seed={seed} | M2={auc_m2:.4f}, M3_unent={auc_m3:.4f} | Diff={diff:+.4f} ({elapsed:.1f}s)")

    df = pd.DataFrame(records)

    # Statistical summary
    overall_diff = df["diff"].values
    mean_diff = float(np.mean(overall_diff))
    std_diff = float(np.std(overall_diff, ddof=1))
    n_total = len(overall_diff)
    t_crit = float(stats.t.ppf(0.975, df=n_total - 1))
    ci_half = t_crit * (std_diff / np.sqrt(n_total))
    ci_lower = mean_diff - ci_half
    ci_upper = mean_diff + ci_half

    print("\n" + "=" * 70)
    print(f"OVERALL GATE RESULTS (n={n_total}):")
    print(f"  Mean AUC M2:           {df['auc_m2'].mean():.4f}")
    print(f"  Mean AUC M3 (unent 1L): {df['auc_m3_unent1L'].mean():.4f}")
    print(f"  Mean Difference (M3-M2): {mean_diff:+.4f}")
    print(f"  Std Deviation:           {std_diff:.4f}")
    print(f"  t-critical (df={n_total-1}):     {t_crit:.4f}")
    print(f"  95% CI (t-based):       [{ci_lower:+.4f}, {ci_upper:+.4f}]")
    print(f"  Half-width:             ±{ci_half:.4f}")
    print("=" * 70)

    # Per-N breakdown
    print("\nPer-N Breakdown (t-based 95% CI with df=4, t=2.776):")
    t_crit_4 = float(stats.t.ppf(0.975, df=4))
    for n_val, group in df.groupby("N"):
        m_m2 = group["auc_m2"].mean()
        m_m3 = group["auc_m3_unent1L"].mean()
        diffs = group["diff"].values
        m_diff = np.mean(diffs)
        s_diff = np.std(diffs, ddof=1)
        ci_h = t_crit_4 * (s_diff / np.sqrt(len(diffs)))
        print(f"  N={n_val:3d}: M2={m_m2:.4f}, M3={m_m3:.4f} | Diff={m_diff:+.4f} ± {ci_h:.4f} [{m_diff-ci_h:+.4f}, {m_diff+ci_h:+.4f}]")

    print("\n" + "=" * 70)
    if abs(mean_diff) <= 0.03:
        print(f"✓ GATE PASSED: Absolute mean gap |{mean_diff:+.4f}| <= 0.03")
    else:
        print(f"✗ GATE FAILED: Absolute mean gap |{mean_diff:+.4f}| > 0.03")
    print("=" * 70)

    df.to_csv("results/gate_unentangled_1L.csv", index=False)
    return df

if __name__ == "__main__":
    run_gate()
