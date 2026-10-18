import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

RESULTS_CSV = Path("results/results.csv")

def t_ci(data, confidence=0.95):
    """Compute mean, half-width, lower, and upper bounds using exact Student's t."""
    a = np.array(data, dtype=float)
    a = a[~np.isnan(a)]
    n = len(a)
    if n < 2:
        return np.mean(a), 0.0, np.mean(a), np.mean(a)
    mean = np.mean(a)
    se = stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return mean, h, mean - h, mean + h

def paired_t_test(a, b, confidence=0.95):
    """Compute paired difference (a - b) with exact Student's t CI and two-sided p-value."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    diff = a - b
    n = len(diff)
    mean = np.mean(diff)
    se = stats.sem(diff)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    res = stats.ttest_1samp(diff, 0.0)
    p_val = float(res.pvalue)
    return mean, h, mean - h, mean + h, p_val

def run_analysis():
    if not RESULTS_CSV.exists():
        print(f"File {RESULTS_CSV} does not exist.")
        return

    df = pd.read_csv(RESULTS_CSV)
    print(f"Total rows in {RESULTS_CSV}: {len(df)}")

    n_values = [10, 20, 50, 100]

    # ---------------------------------------------------------
    # TABLE 1: Main Models (15 seeds)
    # ---------------------------------------------------------
    print("\n" + "="*100)
    print("TABLE 1: MAIN MODEL PERFORMANCE (ROC-AUC) ACROSS N (15 seeds, Student's t 95% CI, df=14)")
    print("Tuning Budget Disparity: M2 searches 36/54 configs; M3-tuned searches 270 configs.")
    print("="*100)
    print(f"{'N':<4} | {'Model':<14} | {'ROC-AUC (Mean ± 95% CI)':<34} | {'Untuned / Baseline':<20} | {'Edge Hits (γ, C, s-low, s-high)'}")
    print("-"*100)

    for n in n_values:
        # M1
        df_m1 = df[(df["model"] == "M1") & (df["N"] == n)].sort_values("seed")
        m1_mean, m1_h, m1_l, m1_u = t_ci(df_m1["auc"])
        print(f"{n:<4} | {'M1 (Scratch)':<14} | {m1_mean:.4f} ± {m1_h:.4f} [{m1_l:.4f}, {m1_u:.4f}]   | {'N/A':<20} | N/A")

        # M2 (8 components)
        df_m2 = df[(df["model"] == "M2") & (df["N"] == n) & (df["qubits"] == 8)].sort_values("seed")
        m2_mean, m2_h, m2_l, m2_u = t_ci(df_m2["auc"])
        m2_notuned = df_m2["auc_notuned"].mean() if "auc_notuned" in df_m2 else np.nan
        g_edges_m2 = int(df_m2["gamma_edge"].sum()) if "gamma_edge" in df_m2 else 0
        c_edges_m2 = int(df_m2["C_edge"].sum()) if "C_edge" in df_m2 else 0
        print(f"{n:<4} | {'M2 (RBF 8-c)':<14} | {m2_mean:.4f} ± {m2_h:.4f} [{m2_l:.4f}, {m2_u:.4f}]   | {m2_notuned:.4f} (γ=base,C=1)    | γ:{g_edges_m2}/15, C:{c_edges_m2}/15")

        # M3-default (s=1.00 fixed)
        df_m3 = df[(df["model"] == "M3") & (df["N"] == n) & (df["qubits"] == 8) & (df["layers"] == 2) & (df["entanglement"] == True)].sort_values("seed")
        s100_col = "auc_s100" if "auc_s100" in df_m3 and df_m3["auc_s100"].notna().all() else "auc"
        m3_def_mean, m3_def_h, m3_def_l, m3_def_u = t_ci(df_m3[s100_col])
        m3_untuned = df_m3["auc_notuned"].mean() if "auc_notuned" in df_m3 else np.nan
        print(f"{n:<4} | {'M3-default':<14} | {m3_def_mean:.4f} ± {m3_def_h:.4f} [{m3_def_l:.4f}, {m3_def_u:.4f}]   | {m3_untuned:.4f} (untuned)       | s=1.00 fixed")

        # M3-tuned (post-hoc)
        m3_t_mean, m3_t_h, m3_t_l, m3_t_u = t_ci(df_m3["auc"])
        g_edges_m3 = int(df_m3["gamma_edge"].sum()) if "gamma_edge" in df_m3 else 0
        c_edges_m3 = int(df_m3["C_edge"].sum()) if "C_edge" in df_m3 else 0
        s_low = int((df_m3["best_scale"] == 0.0625).sum()) if "best_scale" in df_m3 else 0
        s_high = int((df_m3["best_scale"] == 1.0).sum()) if "best_scale" in df_m3 else 0
        print(f"{n:<4} | {'M3-tuned':<14} | {m3_t_mean:.4f} ± {m3_t_h:.4f} [{m3_t_l:.4f}, {m3_t_u:.4f}]   | Post-hoc CV tuned    | γ:{g_edges_m3}/15, C:{c_edges_m3}/15, s_low:{s_low}/15, s_high:{s_high}/15")
        print("-" * 100)

    # ---------------------------------------------------------
    # TABLE 1b: Scale Trend
    # ---------------------------------------------------------
    print("\n" + "="*100)
    print("TABLE 1b: EXTENDED ENCODING SCALE TREND FOR M3 (8Q 2L Ent) ACROSS N")
    print("="*100)
    print(f"{'N':<4} | {'s=0.0625':<10} | {'s=0.125':<10} | {'s=0.25':<10} | {'s=0.50':<10} | {'s=1.00':<10} | {'CV-Tuned':<10} | {'Scale Choices'}")
    print("-"*100)
    for n in n_values:
        df_m3 = df[(df["model"] == "M3") & (df["N"] == n) & (df["qubits"] == 8) & (df["layers"] == 2) & (df["entanglement"] == True)].sort_values("seed")
        s00625 = df_m3["auc_s00625"].mean() if "auc_s00625" in df_m3 else np.nan
        s0125 = df_m3["auc_s0125"].mean() if "auc_s0125" in df_m3 else np.nan
        s025 = df_m3["auc_s025"].mean() if "auc_s025" in df_m3 else np.nan
        s050 = df_m3["auc_s050"].mean() if "auc_s050" in df_m3 else np.nan
        s100 = df_m3["auc_s100"].mean() if "auc_s100" in df_m3 else np.nan
        tuned = df_m3["auc"].mean()
        scale_counts = df_m3["best_scale"].value_counts().to_dict() if "best_scale" in df_m3 else {}
        scale_str = ", ".join([f"s={k}:{v}" for k, v in sorted(scale_counts.items())])
        print(f"{n:<4} | {s00625:<10.4f} | {s0125:<10.4f} | {s025:<10.4f} | {s050:<10.4f} | {s100:<10.4f} | {tuned:<10.4f} | {scale_str}")

    # ---------------------------------------------------------
    # TABLE 2: Paired Differences with 12 Tests Bonferroni Threshold (α = 0.05 / 12 = 0.004167)
    # ---------------------------------------------------------
    bonf_alpha = 0.05 / 12.0
    print("\n" + "="*115)
    print("TABLE 2a: PAIRED DIFFERENCES (M3 − M2) PER N (15 seeds, Exact Student's t 95% CI, df=14)")
    print(f"Bonferroni-corrected significance threshold for 12 comparisons: α = 0.05 / 12 ≈ {bonf_alpha:.5f}")
    print("="*115)
    print(f"{'N':<4} | {'Comparison':<22} | {'M2 Mean':<8} | {'M3 Mean':<8} | {'Paired Diff (Mean ± 95% CI)':<36} | {'p-value':<12} | {'Bonferroni (α=0.00417)'}")
    print("-"*115)

    for n in n_values:
        df_m2 = df[(df["model"] == "M2") & (df["N"] == n) & (df["qubits"] == 8)].sort_values("seed")
        df_m3 = df[(df["model"] == "M3") & (df["N"] == n) & (df["qubits"] == 8) & (df["layers"] == 2) & (df["entanglement"] == True)].sort_values("seed")
        
        m2_vals = df_m2["auc"].values
        m2_un_vals = df_m2["auc_notuned"].values
        m3_def_vals = df_m3["auc_s100"].values if "auc_s100" in df_m3 and df_m3["auc_s100"].notna().all() else df_m3["auc"].values
        m3_t_vals = df_m3["auc"].values

        # M3-default - M2
        d_mean, d_h, d_l, d_u, p_val = paired_t_test(m3_def_vals, m2_vals)
        sig = "SURVIVES" if p_val < bonf_alpha else ("Suggestive" if p_val < 0.05 else "Not sig")
        print(f"{n:<4} | {'M3-default − M2':<22} | {np.mean(m2_vals):<8.4f} | {np.mean(m3_def_vals):<8.4f} | {d_mean:+.4f} ± {d_h:.4f} [{d_l:+.4f}, {d_u:+.4f}]      | {p_val:<12.4e} | {sig}")

        # M3-tuned - M2
        d_mean, d_h, d_l, d_u, p_val = paired_t_test(m3_t_vals, m2_vals)
        sig = "SURVIVES" if p_val < bonf_alpha else ("Suggestive" if p_val < 0.05 else "Not sig")
        print(f"{n:<4} | {'M3-tuned − M2':<22} | {np.mean(m2_vals):<8.4f} | {np.mean(m3_t_vals):<8.4f} | {d_mean:+.4f} ± {d_h:.4f} [{d_l:+.4f}, {d_u:+.4f}]      | {p_val:<12.4e} | {sig}")

        # M3-tuned - M2-untuned (additional comparison against best M2 variant)
        d_mean, d_h, d_l, d_u, p_val = paired_t_test(m3_t_vals, m2_un_vals)
        sig_un = "SURVIVES" if p_val < bonf_alpha else ("Suggestive" if p_val < 0.05 else "Not sig")
        print(f"{n:<4} | {'M3-tuned − M2-untuned':<22} | {np.mean(m2_un_vals):<8.4f} | {np.mean(m3_t_vals):<8.4f} | {d_mean:+.4f} ± {d_h:.4f} [{d_l:+.4f}, {d_u:+.4f}]      | {p_val:<12.4e} | {sig_un} (vs untuned)")
        print("-" * 115)

    print("\n" + "="*115)
    print("TABLE 2b: PAIRED DIFFERENCES (M2 − M1) PER N (15 seeds, Exact Student's t 95% CI, df=14)")
    print(f"Bonferroni-corrected significance threshold for 12 comparisons: α = 0.05 / 12 ≈ {bonf_alpha:.5f}")
    print("="*115)
    print(f"{'N':<4} | {'Comparison':<22} | {'M1 Mean':<8} | {'M2 Mean':<8} | {'Paired Diff (Mean ± 95% CI)':<36} | {'p-value':<12} | {'Bonferroni (α=0.00417)'}")
    print("-"*115)

    for n in n_values:
        df_m1 = df[(df["model"] == "M1") & (df["N"] == n)].sort_values("seed")
        df_m2 = df[(df["model"] == "M2") & (df["N"] == n) & (df["qubits"] == 8)].sort_values("seed")
        m1_vals = df_m1["auc"].values
        m2_vals = df_m2["auc"].values
        d_mean, d_h, d_l, d_u, p_val = paired_t_test(m2_vals, m1_vals)
        sig = "SURVIVES" if p_val < bonf_alpha else ("Suggestive" if p_val < 0.05 else "Not sig")
        print(f"{n:<4} | {'M2 − M1':<22} | {np.mean(m1_vals):<8.4f} | {np.mean(m2_vals):<8.4f} | {d_mean:+.4f} ± {d_h:.4f} [{d_l:+.4f}, {d_u:+.4f}]      | {p_val:<12.4e} | {sig}")

    # ---------------------------------------------------------
    # TABLE 3: Full Ablation Grid on ALL 15 SEEDS
    # ---------------------------------------------------------
    print("\n" + "="*120)
    print("TABLE 3: FULL 2x2 ABLATION GRID ON ALL 15 SEEDS (seeds 0 to 14) WITH MATCHING M2 BASELINES")
    print("="*120)

    for q in [4, 6, 8]:
        print(f"\n--- Qubits = {q} ---")
        print(f"{'Configuration':<32} | {'Scale Mode':<15} | {'N=10':<18} | {'N=20':<18} | {'N=50':<18} | {'N=100':<18}")
        print("-"*125)

        # M2 baseline
        m2_str = []
        for n in n_values:
            df_m2_q = df[(df["model"] == "M2") & (df["N"] == n) & (df["qubits"] == q)].sort_values("seed")
            mean, h, l, u = t_ci(df_m2_q["auc"])
            m2_str.append(f"{mean:.4f} ± {h:.4f} (n={len(df_m2_q)})")
        print(f"{'Classical M2':<32} | {'Standard':<15} | {m2_str[0]:<18} | {m2_str[1]:<18} | {m2_str[2]:<18} | {m2_str[3]:<18}")

        cells = [
            (1, False, "1L Unentangled"),
            (1, True,  "1L Entangled"),
            (2, False, "2L Unentangled"),
            (2, True,  "2L Entangled"),
        ]

        for L, ent, name in cells:
            for sc_name, sc_col in [("Cell-Tuned s", "auc"), ("Fixed s = 0.25", "auc_s025"), ("Fixed s = 1.00", "auc_s100")]:
                vals_str = []
                for n in n_values:
                    if q == 8 and L == 2 and ent == True:
                        df_cell = df[(df["model"] == "M3") & (df["N"] == n) & (df["qubits"] == 8) & (df["layers"] == 2) & (df["entanglement"] == True)].sort_values("seed")
                    else:
                        df_cell = df[(df["model"] == "M4") & (df["N"] == n) & (df["qubits"] == q) & (df["layers"] == L) & (df["entanglement"] == ent)].sort_values("seed")
                    
                    if len(df_cell) == 0:
                        vals_str.append("N/A")
                    else:
                        col_to_use = sc_col if sc_col in df_cell and df_cell[sc_col].notna().all() else "auc"
                        mean, h, l, u = t_ci(df_cell[col_to_use])
                        vals_str.append(f"{mean:.4f} ± {h:.4f} (n={len(df_cell)})")
                print(f"{name:<32} | {sc_name:<15} | {vals_str[0]:<18} | {vals_str[1]:<18} | {vals_str[2]:<18} | {vals_str[3]:<18}")

    # ---------------------------------------------------------
    # EQUIVALENCE TEST: 1-Layer Unentangled (CV-tuned s) vs M2
    # ---------------------------------------------------------
    print("\n" + "="*115)
    print("EQUIVALENCE CHECK: 1-LAYER UNENTANGLED (CV-TUNED s) vs CLASSICAL M2 (8 Qubits, 15 Seeds)")
    print("Equivalence Bound Margin: ±0.02 AUC ([-0.0200, +0.0200])")
    print("="*115)
    print(f"{'N':<4} | {'1L Unent Mean':<14} | {'M2 Mean':<10} | {'Paired Diff (Mean ± 95% CI)':<36} | {'Inside [-0.02, +0.02]?':<24}")
    print("-"*115)
    for n in n_values:
        df_m2 = df[(df["model"] == "M2") & (df["N"] == n) & (df["qubits"] == 8)].sort_values("seed")
        df_1l = df[(df["model"] == "M4") & (df["N"] == n) & (df["qubits"] == 8) & (df["layers"] == 1) & (df["entanglement"] == False)].sort_values("seed")
        if len(df_1l) > 0 and len(df_m2) > 0 and len(df_1l) == len(df_m2):
            d_mean, d_h, d_l, d_u, p_val = paired_t_test(df_1l["auc"].values, df_m2["auc"].values)
            inside = "YES (EQUIVALENT)" if (d_l >= -0.02 and d_u <= 0.02) else f"NO (CI: [{d_l:+.4f}, {d_u:+.4f}])"
            print(f"{n:<4} | {np.mean(df_1l['auc']):<14.4f} | {np.mean(df_m2['auc']):<10.4f} | {d_mean:+.4f} ± {d_h:.4f} [{d_l:+.4f}, {d_u:+.4f}]      | {inside}")
        else:
            print(f"{n:<4} | Data count mismatch (1L: {len(df_1l)}, M2: {len(df_m2)})")

    # ---------------------------------------------------------
    # PAIRED DIFFERENCES FOR ALL ABLATION CELLS MINUS MATCHING M2 (15 SEEDS)
    # ---------------------------------------------------------
    print("\n" + "="*125)
    print("PAIRED DIFFERENCES (ABLATION CELL − MATCHING M2) PER N (15 seeds, Exact Student's t 95% CI, df=14)")
    print("="*125)
    for q in [4, 6, 8]:
        print(f"\n--- Qubits = {q} (Paired against Classical M2 with {q} components) ---")
        print(f"{'Cell (CV-Tuned s)':<20} | {'N':<4} | {'Cell Mean':<10} | {'M2 Mean':<10} | {'Paired Diff (Mean ± 95% CI)':<36} | {'p-value':<12}")
        print("-"*105)
        for L, ent, name in cells:
            for n in n_values:
                df_m2_q = df[(df["model"] == "M2") & (df["N"] == n) & (df["qubits"] == q)].sort_values("seed")
                if q == 8 and L == 2 and ent == True:
                    df_c = df[(df["model"] == "M3") & (df["N"] == n) & (df["qubits"] == 8) & (df["layers"] == 2) & (df["entanglement"] == True)].sort_values("seed")
                else:
                    df_c = df[(df["model"] == "M4") & (df["N"] == n) & (df["qubits"] == q) & (df["layers"] == L) & (df["entanglement"] == ent)].sort_values("seed")
                
                if len(df_c) == 15 and len(df_m2_q) == 15:
                    d_mean, d_h, d_l, d_u, p_val = paired_t_test(df_c["auc"].values, df_m2_q["auc"].values)
                    print(f"{name:<20} | {n:<4} | {np.mean(df_c['auc']):<10.4f} | {np.mean(df_m2_q['auc']):<10.4f} | {d_mean:+.4f} ± {d_h:.4f} [{d_l:+.4f}, {d_u:+.4f}]      | {p_val:<12.4e}")


if __name__ == "__main__":
    run_analysis()
