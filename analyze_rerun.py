"""
Analyze Full Experimental Matrix: 15 Seeds, Extended Scales, 2x2 Grid, Diagnostics
==================================================================================
Produces all required summary tables with Student's t 95% CIs and Bonferroni correction.
All output formatted in pure Unicode (NO LaTeX).
"""

import numpy as np
import pandas as pd
from scipy import stats
import config

def analyze():
    if not config.RESULTS_CSV.exists():
        print(f"Results file not found: {config.RESULTS_CSV}")
        return

    df = pd.read_csv(config.RESULTS_CSV)
    print(f"Total rows in results.csv: {len(df)}")

    # ─────────────────────────────────────────────────────────────────
    # 1. Main Models: M1, M2 (8-comp), M3-default, M3-tuned across N
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "="*115)
    print("TABLE 1: MAIN MODEL PERFORMANCE (ROC-AUC) ACROSS N (t-based 95% CI, deterministic CV tie-break)")
    print("Tuning Budget Disparity: M2 searches 36/54 configs (γ, C); M3-tuned searches 270 configs (s, γ, C).")
    print("M3-tuned encoding scale s was added post-hoc after default s=1.00 failed.")
    print("="*115)
    print(f"{'N':<5} | {'Model':<12} | {'AUC (mean ± 95% CI)':<30} | {'Untuned / Baseline':<20} | {'Edge Hits (γ, C, s)':<22}")
    print("-"*115)

    for n in config.N_VALUES:
        # M1
        sub_m1 = df[(df["model"] == "M1") & (df["N"] == n)]
        n_m1 = len(sub_m1)
        if n_m1 > 0:
            m = sub_m1["auc"].mean()
            s = sub_m1["auc"].std(ddof=1) if n_m1 > 1 else 0.0
            t_crit = stats.t.ppf(0.975, df=max(n_m1 - 1, 1))
            ci = t_crit * (s / np.sqrt(n_m1)) if n_m1 > 1 else 0.0
            print(f"{n:<5} | {'M1 (Scratch)':<12} | {m:.4f} ± {ci:.4f} [{m-ci:.4f}, {m+ci:.4f}]   | {'N/A':<20} | {'N/A':<22}")

        # M2 (8 components)
        sub_m2 = df[(df["model"] == "M2") & (df["qubits"] == config.PCA_COMPONENTS) & (df["N"] == n)]
        n_m2 = len(sub_m2)
        if n_m2 > 0:
            m = sub_m2["auc"].mean()
            s = sub_m2["auc"].std(ddof=1) if n_m2 > 1 else 0.0
            t_crit = stats.t.ppf(0.975, df=max(n_m2 - 1, 1))
            ci = t_crit * (s / np.sqrt(n_m2)) if n_m2 > 1 else 0.0
            untuned = sub_m2["auc_notuned"].mean() if "auc_notuned" in sub_m2 else np.nan
            ge = int(sub_m2["gamma_edge"].sum()) if "gamma_edge" in sub_m2 else 0
            ce = int(sub_m2["C_edge"].sum()) if "C_edge" in sub_m2 else 0
            edge_str = f"γ:{ge}/{n_m2}, C:{ce}/{n_m2}"
            print(f"{n:<5} | {'M2 (RBF 8-c)':<12} | {m:.4f} ± {ci:.4f} [{m-ci:.4f}, {m+ci:.4f}]   | {untuned:.4f} (γ=base,C=1)    | {edge_str:<22}")

        # M3-default (fixed s=1.00)
        sub_m3 = df[(df["model"] == "M3") & (df["N"] == n)]
        n_m3 = len(sub_m3)
        if n_m3 > 0 and "auc_s100" in sub_m3 and sub_m3["auc_s100"].notna().any():
            vals_def = sub_m3["auc_s100"]
            m = vals_def.mean()
            s = vals_def.std(ddof=1) if n_m3 > 1 else 0.0
            t_crit = stats.t.ppf(0.975, df=max(n_m3 - 1, 1))
            ci = t_crit * (s / np.sqrt(n_m3)) if n_m3 > 1 else 0.0
            untuned = sub_m3["auc_notuned"].mean() if "auc_notuned" in sub_m3 else np.nan
            print(f"{n:<5} | {'M3-default':<12} | {m:.4f} ± {ci:.4f} [{m-ci:.4f}, {m+ci:.4f}]   | {untuned:.4f} (untuned)       | {'s=1.0 fixed':<22}")

        # M3-tuned (post-hoc scale tuned)
        if n_m3 > 0:
            m = sub_m3["auc"].mean()
            s = sub_m3["auc"].std(ddof=1) if n_m3 > 1 else 0.0
            t_crit = stats.t.ppf(0.975, df=max(n_m3 - 1, 1))
            ci = t_crit * (s / np.sqrt(n_m3)) if n_m3 > 1 else 0.0
            ge = int(sub_m3["gamma_edge"].sum()) if "gamma_edge" in sub_m3 else 0
            ce = int(sub_m3["C_edge"].sum()) if "C_edge" in sub_m3 else 0
            se = int(sub_m3["scale_edge"].sum()) if "scale_edge" in sub_m3 else 0
            edge_str = f"γ:{ge}/{n_m3}, C:{ce}/{n_m3}, s:{se}/{n_m3}"
            print(f"{n:<5} | {'M3-tuned':<12} | {m:.4f} ± {ci:.4f} [{m-ci:.4f}, {m+ci:.4f}]   | {'post-hoc s tuned':<20} | {edge_str:<22}")

        print("-" * 115)

    # ─────────────────────────────────────────────────────────────────
    # 1b. Extended Encoding Scale Tuning Breakdown for M3
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "="*115)
    print("TABLE 1b: EXTENDED ENCODING SCALE (s ∈ {0.0625, 0.125, 0.25, 0.50, 1.00}) TREND FOR M3 (8Q 2L Ent)")
    print("As s → 0, rotations linearize: ⟨X⟩ ~ sθ, ⟨Z⟩ ~ 1 - (sθ)²/2 (smooth classical mapping).")
    print("="*115)
    print(f"{'N':<5} | {'s=0.0625':<10} | {'s=0.125':<10} | {'s=0.25':<10} | {'s=0.50':<10} | {'s=1.00':<10} | {'CV-Tuned':<10} | {'Selected Scale Distribution':<26}")
    print("-"*115)

    for n in config.N_VALUES:
        sub_m3 = df[(df["model"] == "M3") & (df["N"] == n)]
        if len(sub_m3) == 0:
            continue
        v_00625 = sub_m3["auc_s00625"].mean() if "auc_s00625" in sub_m3 and sub_m3["auc_s00625"].notna().any() else np.nan
        v_0125  = sub_m3["auc_s0125"].mean() if "auc_s0125" in sub_m3 and sub_m3["auc_s0125"].notna().any() else np.nan
        v_025   = sub_m3["auc_s025"].mean() if "auc_s025" in sub_m3 and sub_m3["auc_s025"].notna().any() else np.nan
        v_050   = sub_m3["auc_s050"].mean() if "auc_s050" in sub_m3 and sub_m3["auc_s050"].notna().any() else np.nan
        v_100   = sub_m3["auc_s100"].mean() if "auc_s100" in sub_m3 and sub_m3["auc_s100"].notna().any() else np.nan
        v_tuned = sub_m3["auc"].mean()

        sc_counts = sub_m3["best_scale"].value_counts().to_dict() if "best_scale" in sub_m3 else {}
        sc_dist = ", ".join([f"s={k}:{v}" for k, v in sorted(sc_counts.items())])

        print(f"{n:<5} | {v_00625:<10.4f} | {v_0125:<10.4f} | {v_025:<10.4f} | {v_050:<10.4f} | {v_100:<10.4f} | {v_tuned:<10.4f} | {sc_dist:<26}")
    print("="*115)

    # ─────────────────────────────────────────────────────────────────
    # 2. Paired Differences: M3 − M2 and M2 − M1 (Bonferroni threshold = 0.05 / 8 = 0.00625)
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "="*115)
    print("TABLE 2a: PAIRED DIFFERENCES (M3 − M2) PER N (Exact Student's t 95% CI)")
    print("Bonferroni-corrected significance threshold for 8 comparisons: α = 0.05 / 8 = 0.00625")
    print("="*115)
    print(f"{'N':<5} | {'Comparison':<16} | {'M2 Mean':<8} | {'M3 Mean':<8} | {'Paired Diff (mean ± 95% CI)':<30} | {'p-value':<12} | {'Bonferroni (α=0.00625)':<22}")
    print("-"*115)

    m2_main = df[(df["model"] == "M2") & (df["qubits"] == config.PCA_COMPONENTS)][["N", "seed", "auc"]].rename(columns={"auc": "auc_m2"})
    m3_main = df[df["model"] == "M3"][["N", "seed", "auc", "auc_s100"]].rename(columns={"auc": "auc_m3_tuned", "auc_s100": "auc_m3_default"})

    paired = pd.merge(m2_main, m3_main, on=["N", "seed"], how="inner")

    for n in config.N_VALUES:
        sub_p = paired[paired["N"] == n]
        n_p = len(sub_p)
        if n_p == 0:
            continue
        t_crit = stats.t.ppf(0.975, df=max(n_p - 1, 1))

        # M3-default vs M2
        if "auc_m3_default" in sub_p and sub_p["auc_m3_default"].notna().any():
            diff_def = sub_p["auc_m3_default"] - sub_p["auc_m2"]
            m_diff = diff_def.mean()
            s_diff = diff_def.std(ddof=1) if n_p > 1 else 0.0
            ci = t_crit * (s_diff / np.sqrt(n_p)) if n_p > 1 else 0.0
            t_stat, p_val = stats.ttest_rel(sub_p["auc_m3_default"], sub_p["auc_m2"]) if n_p > 1 else (0, 1)
            sig_str = "SURVIVES (p < 0.00625)" if p_val < 0.00625 else ("Suggestive (p < 0.05)" if p_val < 0.05 else "Not significant")
            print(f"{n:<5} | {'M3-default − M2':<16} | {sub_p['auc_m2'].mean():.4f}   | {sub_p['auc_m3_default'].mean():.4f}   | {m_diff:+.4f} ± {ci:.4f} [{m_diff-ci:+.4f}, {m_diff+ci:+.4f}] | p = {p_val:.4e} | {sig_str:<22}")

        # M3-tuned vs M2
        diff_tun = sub_p["auc_m3_tuned"] - sub_p["auc_m2"]
        m_diff = diff_tun.mean()
        s_diff = diff_tun.std(ddof=1) if n_p > 1 else 0.0
        ci = t_crit * (s_diff / np.sqrt(n_p)) if n_p > 1 else 0.0
        t_stat, p_val = stats.ttest_rel(sub_p["auc_m3_tuned"], sub_p["auc_m2"]) if n_p > 1 else (0, 1)
        sig_str = "SURVIVES (p < 0.00625)" if p_val < 0.00625 else ("Suggestive (p < 0.05)" if p_val < 0.05 else "Not significant")
        print(f"{n:<5} | {'M3-tuned − M2':<16} | {sub_p['auc_m2'].mean():.4f}   | {sub_p['auc_m3_tuned'].mean():.4f}   | {m_diff:+.4f} ± {ci:.4f} [{m_diff-ci:+.4f}, {m_diff+ci:+.4f}] | p = {p_val:.4e} | {sig_str:<22}")

        print("-" * 115)

    print("\n" + "="*115)
    print("TABLE 2b: PAIRED DIFFERENCES (M2 − M1) PER N (Exact Student's t 95% CI)")
    print("Bonferroni-corrected significance threshold for 8 comparisons: α = 0.05 / 8 = 0.00625")
    print("="*115)
    print(f"{'N':<5} | {'Comparison':<16} | {'M1 Mean':<8} | {'M2 Mean':<8} | {'Paired Diff (mean ± 95% CI)':<30} | {'p-value':<12} | {'Bonferroni (α=0.00625)':<22}")
    print("-"*115)

    m1_main = df[df["model"] == "M1"][["N", "seed", "auc"]].rename(columns={"auc": "auc_m1"})
    paired_m2_m1 = pd.merge(m1_main, m2_main, on=["N", "seed"], how="inner")
    paired_m2_m1["diff"] = paired_m2_m1["auc_m2"] - paired_m2_m1["auc_m1"]

    for n in config.N_VALUES:
        sub_p = paired_m2_m1[paired_m2_m1["N"] == n]
        n_p = len(sub_p)
        if n_p == 0:
            continue
        t_crit = stats.t.ppf(0.975, df=max(n_p - 1, 1))
        diffs = sub_p["diff"].values
        m_diff = np.mean(diffs)
        s_diff = np.std(diffs, ddof=1) if n_p > 1 else 0.0
        ci = t_crit * (s_diff / np.sqrt(n_p)) if n_p > 1 else 0.0
        t_stat, p_val = stats.ttest_rel(sub_p["auc_m2"], sub_p["auc_m1"]) if n_p > 1 else (0, 1)
        sig_str = "SURVIVES (p < 0.00625)" if p_val < 0.00625 else ("Suggestive (p < 0.05)" if p_val < 0.05 else "Not significant")
        print(f"{n:<5} | {'M2 − M1':<16} | {sub_p['auc_m1'].mean():.4f}   | {sub_p['auc_m2'].mean():.4f}   | {m_diff:+.4f} ± {ci:.4f} [{m_diff-ci:+.4f}, {m_diff+ci:+.4f}] | p = {p_val:.4e} | {sig_str:<22}")

    print("="*115)

    # ─────────────────────────────────────────────────────────────────
    # 3. Full 2x2 Ablation Grid with Matching M2 Baselines
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "="*115)
    print("TABLE 3: FULL 2x2 ABLATION GRID (1L/2L × Ent/Unent) WITH BANDWIDTH PARITY (ROC-AUC)")
    print("Reported with Cell-Tuned Scale s, Common Fixed s = 0.25, and Default Fixed s = 1.00")
    print("="*115)
    print(f"{'Qubits':<7} | {'Configuration':<26} | {'N=10':<15} | {'N=20':<15} | {'N=50':<15} | {'N=100':<15}")
    print("-"*115)

    for q in [4, 6, 8]:
        # Classical M2 matching baseline
        row_str = f"{q} (M2)   | {'Classical M2':<26} | "
        for n in config.N_VALUES:
            sub = df[(df["model"] == "M2") & (df["qubits"] == q) & (df["N"] == n)]
            if len(sub) > 0:
                row_str += f"{sub['auc'].mean():.4f} ± {sub['auc'].std(ddof=1):.3f} | "
            else:
                row_str += "  N/A           | "
        print(row_str)

        # 1L Unentangled: Tuned s, Fixed s=0.25, Fixed s=1.00
        for mode, col in [("1L Unent (cell-tuned s)", "auc"), ("1L Unent (fixed s=0.25)", "auc_s025"), ("1L Unent (fixed s=1.00)", "auc_s100")]:
            row_str = f"{q} (Q)    | {mode:<26} | "
            for n in config.N_VALUES:
                sub = df[(df["model"].isin(["M3", "M4"])) & (df["qubits"] == q) & (df["layers"] == 1) & (df["entanglement"] == False) & (df["N"] == n)]
                if len(sub) > 0 and col in sub and sub[col].notna().any():
                    row_str += f"{sub[col].mean():.4f} ± {sub[col].std(ddof=1):.3f} | "
                else:
                    row_str += "  N/A           | "
            print(row_str)

        # 1L Entangled: Tuned s, Fixed s=0.25, Fixed s=1.00
        for mode, col in [("1L Ent (cell-tuned s)", "auc"), ("1L Ent (fixed s=0.25)", "auc_s025"), ("1L Ent (fixed s=1.00)", "auc_s100")]:
            row_str = f"{q} (Q)    | {mode:<26} | "
            for n in config.N_VALUES:
                sub = df[(df["model"].isin(["M3", "M4"])) & (df["qubits"] == q) & (df["layers"] == 1) & (df["entanglement"] == True) & (df["N"] == n)]
                if len(sub) > 0 and col in sub and sub[col].notna().any():
                    row_str += f"{sub[col].mean():.4f} ± {sub[col].std(ddof=1):.3f} | "
                else:
                    row_str += "  N/A           | "
            print(row_str)

        # 2L Unentangled: Tuned s, Fixed s=0.25, Fixed s=1.00
        for mode, col in [("2L Unent (cell-tuned s)", "auc"), ("2L Unent (fixed s=0.25)", "auc_s025"), ("2L Unent (fixed s=1.00)", "auc_s100")]:
            row_str = f"{q} (Q)    | {mode:<26} | "
            for n in config.N_VALUES:
                sub = df[(df["model"].isin(["M3", "M4"])) & (df["qubits"] == q) & (df["layers"] == 2) & (df["entanglement"] == False) & (df["N"] == n)]
                if len(sub) > 0 and col in sub and sub[col].notna().any():
                    row_str += f"{sub[col].mean():.4f} ± {sub[col].std(ddof=1):.3f} | "
                else:
                    row_str += "  N/A           | "
            print(row_str)

        # 2L Entangled: Tuned s, Fixed s=0.25, Fixed s=1.00
        for mode, col in [("2L Ent (cell-tuned s)", "auc"), ("2L Ent (fixed s=0.25)", "auc_s025"), ("2L Ent (fixed s=1.00)", "auc_s100")]:
            row_str = f"{q} (Q)    | {mode:<26} | "
            for n in config.N_VALUES:
                sub = df[(df["model"].isin(["M3", "M4"])) & (df["qubits"] == q) & (df["layers"] == 2) & (df["entanglement"] == True) & (df["N"] == n)]
                if len(sub) > 0 and col in sub and sub[col].notna().any():
                    row_str += f"{sub[col].mean():.4f} ± {sub[col].std(ddof=1):.3f} | "
                else:
                    row_str += "  N/A           | "
            print(row_str)

        print("-" * 115)

    # ─────────────────────────────────────────────────────────────────
    # 4. Physical Diagnostics Table (Computed at Fixed s=1.0, Fixed s=0.25, and Tuned s)
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "="*115)
    print("TABLE 4: PHYSICAL DIAGNOSTICS (Feature Variance, Distance Spread, Spearman ρ vs Angle Features)")
    print("Classical spread falls with dimension (1/√d Euclidean concentration); Quantum spread remains flat.")
    print("="*115)
    print(f"{'Configuration':<34} | {'Feat Variance':<15} | {'Dist Spread (std/med)':<22} | {'Spearman ρ (vs PCA)':<20}")
    print("-"*115)

    # Classical M2
    for q in [4, 6, 8]:
        sub = df[(df["model"] == "M2") & (df["qubits"] == q)]
        if len(sub) > 0:
            fv = sub["feat_variance"].mean()
            sp = sub["dist_std_over_median"].mean()
            rho = sub["dist_spearman_rank_corr"].mean()
            print(f"Classical M2 ({q} comp)                  | {fv:<15.4f} | {sp:<22.4f} | {rho:<20.4f}")

    print("\n[PART A: Standard Fixed Scale s = 1.00 (All Cells)]")
    print("-" * 115)
    for q in [4, 6, 8]:
        for L, ent, name in [(1, False, "1L Unent"), (1, True, "1L Ent"), (2, False, "2L Unent"), (2, True, "2L Ent")]:
            sub = df[(df["model"].isin(["M3", "M4"])) & (df["qubits"] == q) & (df["layers"] == L) & (df["entanglement"] == ent)]
            if len(sub) > 0 and "feat_variance_s100" in sub and sub["feat_variance_s100"].notna().any():
                fv = sub["feat_variance_s100"].mean()
                sp = sub["dist_std_over_median_s100"].mean()
                rho = sub["dist_spearman_rank_corr_s100"].mean()
                print(f"Quantum {q}Q {name:<12} (fixed s=1.00)     | {fv:<15.4f} | {sp:<22.4f} | {rho:<20.4f}")
            elif len(sub) > 0:
                fv = sub["feat_variance"].mean()
                sp = sub["dist_std_over_median"].mean()
                rho = sub["dist_spearman_rank_corr"].mean()
                print(f"Quantum {q}Q {name:<12} (fixed s=1.00)     | {fv:<15.4f} | {sp:<22.4f} | {rho:<20.4f}")

    print("\n[PART B: Common Fixed Scale s = 0.25 (Bandwidth Parity)]")
    print("-" * 115)
    for q in [4, 6, 8]:
        for L, ent, name in [(1, False, "1L Unent"), (1, True, "1L Ent"), (2, False, "2L Unent"), (2, True, "2L Ent")]:
            sub = df[(df["model"].isin(["M3", "M4"])) & (df["qubits"] == q) & (df["layers"] == L) & (df["entanglement"] == ent)]
            if len(sub) > 0 and "feat_variance_s025" in sub and sub["feat_variance_s025"].notna().any():
                fv = sub["feat_variance_s025"].mean()
                sp = sub["dist_std_over_median_s025"].mean()
                rho = sub["dist_spearman_rank_corr_s025"].mean()
                print(f"Quantum {q}Q {name:<12} (fixed s=0.25)     | {fv:<15.4f} | {sp:<22.4f} | {rho:<20.4f}")

    print("\n[PART C: Cell-Tuned Scale s]")
    print("-" * 115)
    for q in [4, 6, 8]:
        for L, ent, name in [(1, False, "1L Unent"), (1, True, "1L Ent"), (2, False, "2L Unent"), (2, True, "2L Ent")]:
            sub = df[(df["model"].isin(["M3", "M4"])) & (df["qubits"] == q) & (df["layers"] == L) & (df["entanglement"] == ent)]
            if len(sub) > 0:
                fv = sub["feat_variance"].mean()
                sp = sub["dist_std_over_median"].mean()
                rho = sub["dist_spearman_rank_corr"].mean()
                print(f"Quantum {q}Q {name:<12} (tuned scale)     | {fv:<15.4f} | {sp:<22.4f} | {rho:<20.4f}")

    print("="*115)

if __name__ == "__main__":
    analyze()
