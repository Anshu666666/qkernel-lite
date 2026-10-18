"""
QKernel-Lite – Figure Generation
==================================
Stage 8: Publication-quality plots from results.csv.
Uses exact Student's t multiplier for 95% confidence intervals.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path

import config


def set_style():
    """Set publication-quality matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 150,
    })


COLORS = {
    "M1": "#E24A33",
    "M2": "#348ABD",
    "M3": "#988ED5",
    "M4": "#8EBA42",
}
MARKERS = {
    "M1": "s",
    "M2": "o",
    "M3": "D",
    "M4": "^",
}


def plot_auc_vs_n(df: pd.DataFrame, save_path: Path = None):
    """
    Figure 1: AUC versus N for M1, M2 (8 components), M3 with exact t-based 95% CI.
    Includes M3-default (s = 1.00) and untuned M2 as references.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    # M1
    df_m1 = df[df["model"] == "M1"]
    if len(df_m1) > 0:
        g1 = df_m1.groupby("N")["auc"].agg(["mean", "std", "count"]).reset_index()
        t1 = np.array([stats.t.ppf(0.975, df=max(c - 1, 1)) if c > 1 else 1.96 for c in g1["count"]])
        ci1 = t1 * (g1["std"] / np.sqrt(g1["count"]))
        ax.errorbar(g1["N"], g1["mean"], yerr=ci1, label="M1: Scratch CNN",
                     color=COLORS["M1"], marker=MARKERS["M1"], capsize=5, linewidth=2.2, markersize=8)

    # M2 Tuned
    df_m2 = df[(df["model"] == "M2") & (df["qubits"] == config.PCA_COMPONENTS)]
    if len(df_m2) > 0:
        g2 = df_m2.groupby("N")["auc"].agg(["mean", "std", "count"]).reset_index()
        t2 = np.array([stats.t.ppf(0.975, df=max(c - 1, 1)) if c > 1 else 1.96 for c in g2["count"]])
        ci2 = t2 * (g2["std"] / np.sqrt(g2["count"]))
        ax.errorbar(g2["N"], g2["mean"], yerr=ci2, label="M2: Classical RBF-SVM (8-comp, tuned)",
                     color=COLORS["M2"], marker=MARKERS["M2"], capsize=5, linewidth=2.5, markersize=8)

        # M2 Untuned reference
        if "auc_notuned" in df_m2 and df_m2["auc_notuned"].notna().any():
            g2_un = df_m2.groupby("N")["auc_notuned"].mean().reset_index()
            ax.plot(g2_un["N"], g2_un["auc_notuned"], label="M2: Untuned baseline (γ=base, C=1)",
                    color=COLORS["M2"], linestyle=":", marker="x", linewidth=1.8, alpha=0.85)

    # M3-tuned
    df_m3 = df[df["model"] == "M3"]
    if len(df_m3) > 0:
        g3 = df_m3.groupby("N")["auc"].agg(["mean", "std", "count"]).reset_index()
        t3 = np.array([stats.t.ppf(0.975, df=max(c - 1, 1)) if c > 1 else 1.96 for c in g3["count"]])
        ci3 = t3 * (g3["std"] / np.sqrt(g3["count"]))
        ax.errorbar(g3["N"], g3["mean"], yerr=ci3, label="M3-tuned: CV-tuned encoding scale (post-hoc)",
                     color=COLORS["M3"], marker=MARKERS["M3"], capsize=5, linewidth=2.5, markersize=8)

        # M3-default reference (s = 1.00)
        s100_col = "auc_s100" if "auc_s100" in df_m3 and df_m3["auc_s100"].notna().all() else None
        if s100_col:
            g3_def = df_m3.groupby("N")[s100_col].agg(["mean", "std", "count"]).reset_index()
            t3_def = np.array([stats.t.ppf(0.975, df=max(c - 1, 1)) if c > 1 else 1.96 for c in g3_def["count"]])
            ci3_def = t3_def * (g3_def["std"] / np.sqrt(g3_def["count"]))
            ax.errorbar(g3_def["N"], g3_def["mean"], yerr=ci3_def, label="M3-default: fixed s = 1.00 (unscaled)",
                        color="#7f7f7f", linestyle="--", marker="v", capsize=4, linewidth=1.8, alpha=0.85)

    ax.set_xlabel("Training Set Size N")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("QKernel-Lite: Model Comparison Across Training Sizes (Exact 95% t-CI, n=15)")
    ax.legend(loc="lower right", fontsize=9.5)
    ax.set_ylim(0.48, 0.98)
    ax.set_xticks(config.N_VALUES)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    return fig


def plot_paired_difference(df: pd.DataFrame, save_path: Path = None):
    """
    Figure 2: Paired difference M3-tuned − M2 (8 components) versus N with exact t-based 95% CI.
    Also plots paired difference against untuned M2.
    """
    set_style()

    m2 = df[(df["model"] == "M2") & (df["qubits"] == config.PCA_COMPONENTS)][["N", "seed", "auc", "auc_notuned"]].rename(columns={"auc": "auc_m2", "auc_notuned": "auc_m2_untuned"})
    m3 = df[df["model"] == "M3"][["N", "seed", "auc"]].rename(columns={"auc": "auc_m3"})
    paired = pd.merge(m2, m3, on=["N", "seed"], how="inner")
    paired["diff_tuned"] = paired["auc_m3"] - paired["auc_m2"]
    paired["diff_untuned"] = paired["auc_m3"] - paired["auc_m2_untuned"]

    sum_tun = paired.groupby("N")["diff_tuned"].agg(["mean", "std", "count"]).reset_index()
    t_mult = np.array([stats.t.ppf(0.975, df=max(c - 1, 1)) if c > 1 else 1.96 for c in sum_tun["count"]])
    sum_tun["ci95"] = t_mult * (sum_tun["std"] / np.sqrt(sum_tun["count"]))

    sum_un = paired.groupby("N")["diff_untuned"].agg(["mean", "std", "count"]).reset_index()
    sum_un["ci95"] = t_mult * (sum_un["std"] / np.sqrt(sum_un["count"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        sum_tun["N"], sum_tun["mean"], yerr=sum_tun["ci95"],
        color=COLORS["M3"], marker="D", capsize=5,
        linewidth=2.5, markersize=8, label="M3-tuned − M2 (tuned RBF-SVM)",
    )
    ax.fill_between(
        sum_tun["N"],
        sum_tun["mean"] - sum_tun["ci95"],
        sum_tun["mean"] + sum_tun["ci95"],
        color=COLORS["M3"], alpha=0.15,
    )

    ax.errorbar(
        sum_un["N"], sum_un["mean"], yerr=sum_un["ci95"],
        color="#2b5c8f", marker="o", capsize=4, linestyle="--",
        linewidth=1.8, markersize=7, label="M3-tuned − M2-untuned (best classical baseline)",
    )

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Training Set Size N")
    ax.set_ylabel("Paired AUC Difference")
    ax.set_title("M3-tuned − M2 (Paired Difference, Exact 95% t-CI, n=15)")
    ax.set_xticks(sorted(sum_tun["N"].unique()))
    ax.legend(loc="lower right", fontsize=9.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    return fig


def plot_ablation(df: pd.DataFrame, save_path: Path = None):
    """
    Figure 3: Ablation – Full 2×2 (1L/2L × Entangled/Unentangled) and qubit scaling vs M2 baselines.
    Error bars indicate exact Student's t 95% confidence intervals (df = 14, n = 15).
    """
    set_style()

    m3 = df[df["model"] == "M3"]
    m4 = df[df["model"] == "M4"]
    ablation = pd.concat([m3, m4])
    m2 = df[df["model"] == "M2"]

    if len(ablation) == 0:
        print("  No ablation data found.")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Full 2x2 grid at 8 qubits vs M2 (8 components)
    ax = axes[0]
    configs = [
        (1, False, "1L Unentangled", "#2ca02c", "o", "-"),
        (1, True,  "1L Entangled",   "#1f77b4", "^", "-"),
        (2, False, "2L Unentangled", "#ff7f0e", "s", "--"),
        (2, True,  "2L Entangled (M3)", "#9467bd", "D", "-"),
    ]
    for layers, ent, label, color, marker, ls in configs:
        sub = ablation[(ablation["qubits"] == 8) & (ablation["layers"] == layers) & (ablation["entanglement"] == ent)]
        if len(sub) > 0:
            grouped = sub.groupby("N")["auc"].agg(["mean", "std", "count"]).reset_index()
            t_mult = np.array([stats.t.ppf(0.975, df=max(c - 1, 1)) if c > 1 else 1.96 for c in grouped["count"]])
            ci95 = t_mult * (grouped["std"] / np.sqrt(grouped["count"]))
            ax.errorbar(
                grouped["N"], grouped["mean"], yerr=ci95,
                label=label, color=color, marker=marker, linestyle=ls,
                capsize=4, linewidth=2, markersize=7,
            )
    # Matching M2 (8 components)
    m2_8 = m2[m2["qubits"] == 8]
    if len(m2_8) > 0:
        grouped_m2_8 = m2_8.groupby("N")["auc"].agg(["mean", "std", "count"]).reset_index()
        t_mult = np.array([stats.t.ppf(0.975, df=max(c - 1, 1)) if c > 1 else 1.96 for c in grouped_m2_8["count"]])
        ci95_m2 = t_mult * (grouped_m2_8["std"] / np.sqrt(grouped_m2_8["count"]))
        ax.errorbar(
            grouped_m2_8["N"], grouped_m2_8["mean"], yerr=ci95_m2,
            label="M2 Classical (8-comp)", color="#348ABD", linestyle=":",
            marker="x", linewidth=2.5, markersize=8, capsize=4,
        )
    ax.set_xlabel("Training Set Size N")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("(a) 8Q: Depth × Entanglement (Exact 95% t-CI)")
    ax.legend(fontsize=8.5, loc="lower right")

    # (b) Qubit scaling for 1-Layer vs matching M2
    ax = axes[1]
    color_map = {4: "#1f77b4", 6: "#2ca02c", 8: "#9467bd"}
    for q in config.QUBIT_ABLATION:
        sub = ablation[(ablation["qubits"] == q) & (ablation["layers"] == 1) & (ablation["entanglement"] == True)]
        if len(sub) > 0:
            grouped = sub.groupby("N")["auc"].agg(["mean", "std", "count"]).reset_index()
            t_mult = np.array([stats.t.ppf(0.975, df=max(c - 1, 1)) if c > 1 else 1.96 for c in grouped["count"]])
            ci95 = t_mult * (grouped["std"] / np.sqrt(grouped["count"]))
            ax.errorbar(
                grouped["N"], grouped["mean"], yerr=ci95,
                label=f"1L Ent {q}Q", marker="^", color=color_map[q],
                capsize=4, linewidth=1.8,
            )
        # Matching M2
        sub_m2 = m2[m2["qubits"] == q]
        if len(sub_m2) > 0:
            grouped_m2 = sub_m2.groupby("N")["auc"].agg(["mean", "std"]).reset_index()
            ax.plot(
                grouped_m2["N"], grouped_m2["mean"],
                label=f"M2 {q}-comp", linestyle="--", color=color_map[q],
                alpha=0.7, marker="x",
            )
    ax.set_xlabel("Training Set Size N")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("(b) 1-Layer Qubit Scaling vs Matching M2")
    ax.legend(ncol=2, fontsize=8, loc="lower right")

    # (c) Qubit scaling for 2-Layer vs matching M2
    ax = axes[2]
    for q in config.QUBIT_ABLATION:
        sub = ablation[(ablation["qubits"] == q) & (ablation["layers"] == 2) & (ablation["entanglement"] == True)]
        if len(sub) > 0:
            grouped = sub.groupby("N")["auc"].agg(["mean", "std", "count"]).reset_index()
            t_mult = np.array([stats.t.ppf(0.975, df=max(c - 1, 1)) if c > 1 else 1.96 for c in grouped["count"]])
            ci95 = t_mult * (grouped["std"] / np.sqrt(grouped["count"]))
            ax.errorbar(
                grouped["N"], grouped["mean"], yerr=ci95,
                label=f"2L Ent {q}Q (M3)", marker="D", color=color_map[q],
                capsize=4, linewidth=1.8,
            )
        # Matching M2
        sub_m2 = m2[m2["qubits"] == q]
        if len(sub_m2) > 0:
            grouped_m2 = sub_m2.groupby("N")["auc"].agg(["mean", "std"]).reset_index()
            ax.plot(
                grouped_m2["N"], grouped_m2["mean"],
                label=f"M2 {q}-comp", linestyle="--", color=color_map[q],
                alpha=0.7, marker="x",
            )
    ax.set_xlabel("Training Set Size N")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("(c) 2-Layer Qubit Scaling vs Matching M2")
    ax.legend(ncol=2, fontsize=8, loc="lower right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    return fig


def plot_diagnostics(df: pd.DataFrame, save_path: Path = None):
    """
    Figure 4: Physical diagnostics across qubit counts:
      (a) Total feature variance vs qubit count (log scale)
      (b) Distance spread (std / median) vs qubit count (dimensional concentration)
      (c) Spearman rank correlation between PCA distance and Quantum distance
    """
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    quantum_df = df[df["model"].isin(["M3", "M4"])].copy()
    m2_df = df[df["model"] == "M2"].copy()

    # Panel (a): Feature Variance
    ax = axes[0]
    m2_var = m2_df.groupby("qubits")["feat_variance"].mean().reset_index()
    if len(m2_var) > 0:
        ax.plot(m2_var["qubits"], m2_var["feat_variance"], label="Classical PCA", color="#348ABD", marker="s", linewidth=2)
    q_unent = quantum_df[(quantum_df["entanglement"] == False) & (quantum_df["layers"] == 1)].groupby("qubits")["feat_variance"].mean().reset_index()
    if len(q_unent) > 0:
        ax.plot(q_unent["qubits"], q_unent["feat_variance"], label="Quantum 1L Unent", color="#2ca02c", marker="o", linewidth=2)
    q_ent = quantum_df[(quantum_df["entanglement"] == True) & (quantum_df["layers"] == 2)].groupby("qubits")["feat_variance"].mean().reset_index()
    if len(q_ent) > 0:
        ax.plot(q_ent["qubits"], q_ent["feat_variance"], label="Quantum 2L Ent (M3)", color="#988ED5", marker="D", linewidth=2)

    ax.set_xlabel("Number of Qubits / Components")
    ax.set_ylabel("Feature Variance")
    ax.set_title("(a) Feature Variance (Concentration)")
    ax.set_xticks(config.QUBIT_ABLATION)
    ax.set_yscale("log")
    ax.legend(fontsize=9, loc="center right")

    # Panel (b): Distance Spread (std / median)
    ax = axes[1]
    m2_spread = m2_df.groupby("qubits")["dist_std_over_median"].mean().reset_index()
    if len(m2_spread) > 0:
        ax.plot(m2_spread["qubits"], m2_spread["dist_std_over_median"], label="Classical PCA (roughly 1/√d concentration)", color="#348ABD", marker="s", linewidth=2)
    q_unent_sp = quantum_df[(quantum_df["entanglement"] == False) & (quantum_df["layers"] == 1)].groupby("qubits")["dist_std_over_median"].mean().reset_index()
    if len(q_unent_sp) > 0:
        ax.plot(q_unent_sp["qubits"], q_unent_sp["dist_std_over_median"], label="Quantum 1L Unent", color="#2ca02c", marker="o", linewidth=2)
    q_ent_sp = quantum_df[(quantum_df["entanglement"] == True) & (quantum_df["layers"] == 2)].groupby("qubits")["dist_std_over_median"].mean().reset_index()
    if len(q_ent_sp) > 0:
        ax.plot(q_ent_sp["qubits"], q_ent_sp["dist_std_over_median"], label="Quantum 2L Ent (M3)", color="#988ED5", marker="D", linewidth=2)

    ax.set_xlabel("Number of Qubits / Components")
    ax.set_ylabel("Pairwise Distance Spread (std / median)")
    ax.set_title("(b) Relative Distance Spread\n(Classical drop is dimensional: roughly 1/√d)")
    ax.set_xticks(config.QUBIT_ABLATION)
    ax.legend(fontsize=8.5, loc="upper right")

    # Panel (c): Pairwise Distance Rank Correlation (Spearman rho vs PCA)
    ax = axes[2]
    corr_configs = [
        (1, False, "1L Unentangled", "#2ca02c", "o"),
        (1, True,  "1L Entangled",   "#1f77b4", "^"),
        (2, False, "2L Unentangled", "#ff7f0e", "s"),
        (2, True,  "2L Entangled (M3)", "#988ED5", "D"),
    ]
    for layers, ent, label, color, marker in corr_configs:
        sub_rho = quantum_df[(quantum_df["layers"] == layers) & (quantum_df["entanglement"] == ent)].groupby("qubits")["dist_spearman_rank_corr"].mean().reset_index()
        if len(sub_rho) > 0:
            ax.plot(sub_rho["qubits"], sub_rho["dist_spearman_rank_corr"], label=label, color=color, marker=marker, linewidth=2)

    ax.set_xlabel("Number of Qubits")
    ax.set_ylabel("Spearman Rank Correlation (ρ)")
    ax.set_title("(c) Metric Preservation vs PCA Space (Spearman ρ)")
    ax.set_xticks(config.QUBIT_ABLATION)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=8.5, loc="center left")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    return fig


def plot_single_qubit_phase_wrapping(save_path: Path = None):
    """
    Figure 5: Physical mechanism of phase wrapping and bandwidth scaling.
    Plots single-qubit ⟨Z⟩ and ⟨X⟩ vs input feature θ ∈ [0, π]
    for 1-layer and 2-layer circuits at s = 1.0 and s = 0.25.
    """
    import pennylane as qml
    set_style()

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev, interface="numpy")
    def single_qubit_circuit(theta, n_layers, scale):
        for _ in range(n_layers):
            qml.RY(scale * theta, wires=0)
            qml.RZ(scale * theta, wires=0)
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(0))

    thetas = np.linspace(0, np.pi, 200)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    configs = [
        (1, 1.0,  "1L, s = 1.0 (monotonic, metric preserved)", "#2ca02c", "-", 2.5),
        (2, 1.0,  "2L, s = 1.0 (phase wrapping / frequency folding)", "#d62728", "--", 2.5),
        (1, 0.25, "1L, s = 0.25 (linearized regime)", "#1f77b4", ":", 2.0),
        (2, 0.25, "2L, s = 0.25 (wrapping suppressed, monotonic)", "#9467bd", "-", 2.5),
    ]

    for L, s, label, color, ls, lw in configs:
        z_vals = []
        x_vals = []
        for th in thetas:
            z, x = single_qubit_circuit(th, L, s)
            z_vals.append(float(z))
            x_vals.append(float(x))
        axes[0].plot(thetas, z_vals, label=label, color=color, linestyle=ls, linewidth=lw)
        axes[1].plot(thetas, x_vals, label=label, color=color, linestyle=ls, linewidth=lw)

    # Key endpoint and turnaround annotations - carefully positioned to avoid line overlap
    axes[0].plot([np.pi/2], [0.0], 'ro', markersize=6)
    axes[0].annotate("2L (s=1.0) crosses ⟨Z⟩=0 at θ=π/2",
                     xy=(np.pi/2, 0.0), xytext=(np.pi/2 - 1.1, -0.65),
                     arrowprops=dict(arrowstyle="->", color="red", lw=1.2), fontsize=8, color="red")
    axes[0].plot([np.pi], [1.0], 'ro', markersize=6)
    axes[0].annotate("2L (s=1.0) wraps back to +1.0 at θ=π\n(U(π)²|0⟩ = |0⟩ up to global phase)",
                     xy=(np.pi, 1.0), xytext=(np.pi - 1.45, 0.62),
                     arrowprops=dict(arrowstyle="->", color="red", lw=1.2), fontsize=8, color="red")

    axes[0].plot([np.pi], [0.1464], 'mo', markersize=6)
    axes[0].annotate("2L (s=0.25) ends at +0.1464 (Monotonic)",
                     xy=(np.pi, 0.1464), xytext=(np.pi - 1.35, -0.2),
                     arrowprops=dict(arrowstyle="->", color="#9467bd", lw=1.2), fontsize=8, color="#9467bd")

    axes[0].set_xlabel("Input Feature Angle θ (radians)")
    axes[0].set_ylabel("Expectation Value ⟨Z⟩")
    axes[0].set_title("(a) Single-Qubit ⟨Z⟩ vs. Feature: Phase Wrapping")
    axes[0].set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    axes[0].set_xticklabels(["0", "π/4", "π/2", "3π/4", "π"])
    axes[0].set_ylim(-1.2, 1.2)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=8, loc="lower left")

    # Annotations for axes[1]
    axes[1].plot([np.pi/2], [-1.0], 'ro', markersize=6)
    axes[1].annotate("2L (s=1.0) reaches ⟨X⟩ = -1.0 at θ=π/2",
                     xy=(np.pi/2, -1.0), xytext=(np.pi/2 - 0.7, -0.6),
                     arrowprops=dict(arrowstyle="->", color="red", lw=1.2), fontsize=8, color="red")
    axes[1].set_xlabel("Input Feature Angle θ (radians)")
    axes[1].set_ylabel("Expectation Value ⟨X⟩")
    axes[1].set_title("(b) Single-Qubit ⟨X⟩ vs. Feature: Harmonic Generation")
    axes[1].set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    axes[1].set_xticklabels(["0", "π/4", "π/2", "3π/4", "π"])
    axes[1].set_ylim(-1.2, 1.2)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=8, loc="upper right")


def plot_kernel_heatmaps(
    K_m2: np.ndarray, K_m3: np.ndarray,
    y_sorted: np.ndarray,
    save_path: Path = None,
):
    """
    Figure 5: Side-by-side kernel heatmaps (M2 vs M3) at N=50.
    """
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    n_class0 = np.sum(y_sorted == 0)

    for ax, K, title in [(axes[0], K_m2, "M2: RBF Kernel"),
                          (axes[1], K_m3, "M3: PQK Kernel")]:
        im = ax.imshow(K, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axhline(y=n_class0-0.5, color="red", linewidth=1)
        ax.axvline(x=n_class0-0.5, color="red", linewidth=1)
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    return fig


def plot_single_qubit_phase_wrapping(save_path: Path = None):
    """
    Figure 5: Physical mechanism of phase wrapping and bandwidth scaling.
    Plots single-qubit ⟨Z⟩ and ⟨X⟩ vs input feature θ ∈ [0, π]
    for 1-layer and 2-layer circuits at s = 1.0 and s = 0.25.
    """
    import pennylane as qml
    set_style()

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev, interface="numpy")
    def single_qubit_circuit(theta, n_layers, scale):
        for _ in range(n_layers):
            qml.RY(scale * theta, wires=0)
            qml.RZ(scale * theta, wires=0)
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(0))

    thetas = np.linspace(0, np.pi, 200)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    configs = [
        (1, 1.0,  "1L, s = 1.0 (monotonic, metric preserved)", "#2ca02c", "-", 2.5),
        (2, 1.0,  "2L, s = 1.0 (phase wrapping / frequency folding)", "#d62728", "--", 2.5),
        (1, 0.25, "1L, s = 0.25 (linearized regime)", "#1f77b4", ":", 2.0),
        (2, 0.25, "2L, s = 0.25 (wrapping suppressed, monotonic)", "#9467bd", "-", 2.5),
    ]

    for L, s, label, color, ls, lw in configs:
        z_vals = []
        x_vals = []
        for th in thetas:
            z, x = single_qubit_circuit(th, L, s)
            z_vals.append(float(z))
            x_vals.append(float(x))
        axes[0].plot(thetas, z_vals, label=label, color=color, linestyle=ls, linewidth=lw)
        axes[1].plot(thetas, x_vals, label=label, color=color, linestyle=ls, linewidth=lw)

    # Key endpoint and turnaround annotations
    # 2L, s = 1.0 turnaround and wrap
    axes[0].plot([np.pi/2], [0.0], 'ro', markersize=6)
    axes[0].annotate("2L (s=1.0) crosses ⟨Z⟩=0 at θ=π/2",
                     xy=(np.pi/2, 0.0), xytext=(np.pi/2 - 0.5, -0.4),
                     arrowprops=dict(arrowstyle="->", color="red", lw=1.2), fontsize=8, color="red")
    axes[0].plot([np.pi], [1.0], 'ro', markersize=6)
    axes[0].annotate("2L (s=1.0) wraps back to +1.0 at θ=π\n(Exact identity: U(π)²|0⟩ = |0⟩)",
                     xy=(np.pi, 1.0), xytext=(np.pi - 1.2, 0.75),
                     arrowprops=dict(arrowstyle="->", color="red", lw=1.2), fontsize=8, color="red")
    # 2L, s = 0.25 monotonic endpoint
    axes[0].plot([np.pi], [0.1464], 'mo', markersize=6)
    axes[0].annotate("2L (s=0.25) ends at +0.1464 (Monotonic)",
                     xy=(np.pi, 0.1464), xytext=(np.pi - 1.1, 0.35),
                     arrowprops=dict(arrowstyle="->", color="#9467bd", lw=1.2), fontsize=8, color="#9467bd")

    axes[0].set_xlabel("Input Feature Angle θ (radians)")
    axes[0].set_ylabel("Expectation Value ⟨Z⟩")
    axes[0].set_title("(a) Single-Qubit ⟨Z⟩ vs. Feature: Phase Wrapping")
    axes[0].set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    axes[0].set_xticklabels(["0", "π/4", "π/2", "3π/4", "π"])
    axes[0].set_ylim(-1.15, 1.15)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=8, loc="lower left")

    # Annotations for axes[1]
    axes[1].plot([np.pi/2], [-1.0], 'ro', markersize=6)
    axes[1].annotate("2L (s=1.0) reaches ⟨X⟩ = -1.0 at θ=π/2",
                     xy=(np.pi/2, -1.0), xytext=(np.pi/2 - 0.4, -0.65),
                     arrowprops=dict(arrowstyle="->", color="red", lw=1.2), fontsize=8, color="red")
    axes[1].set_xlabel("Input Feature Angle θ (radians)")
    axes[1].set_ylabel("Expectation Value ⟨X⟩")
    axes[1].set_title("(b) Single-Qubit ⟨X⟩ vs. Feature: Harmonic Generation")
    axes[1].set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    axes[1].set_xticklabels(["0", "π/4", "π/2", "3π/4", "π"])
    axes[1].set_ylim(-1.15, 1.15)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    return fig


def generate_all_figures(results_path: Path = config.RESULTS_CSV):
    """Generate all figures from results.csv."""
    print("Generating figures...")

    df = pd.read_csv(results_path)
    out = config.RESULTS_DIR

    plot_auc_vs_n(df, out / "fig1_auc_vs_n.png")
    plot_paired_difference(df, out / "fig2_paired_difference.png")
    plot_ablation(df, out / "fig3_ablation.png")
    plot_diagnostics(df, out / "fig4_kernel_concentration.png")
    plot_single_qubit_phase_wrapping(out / "fig5_phase_wrapping.png")

    print("Done!")


if __name__ == "__main__":
    generate_all_figures()
