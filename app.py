"""
QKernel-Lite – Clinical Diagnostic Prototype & Quantum Benchmark Platform
========================================================================
Design System: Cobalt Grid Editorial Theme (Warm Ivory #F0EBDE & Cobalt Blue #1F2BE0)
Fonts: Newsreader (Editorial Serif), Hanken Grotesk (Sans), DM Mono (Monospace)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from data import load_pneumoniamnist, stratified_subset_indices, get_subset_data
from features import extract_and_cache_features, get_cache_path
from preprocess import preprocess_for_experiment
from classical import (
    compute_rbf_kernel, compute_base_gamma,
    SmallCNN, predict_m1, kernel_diagnostics,
)
from quantum import (
    compute_pqk_kernel, get_cached_expectations,
    draw_circuit, create_circuit,
)
import torch
from sklearn.svm import SVC

# ═══════════════════════════════════════════════════════════════════════
#  Page Config & Editorial Light Styling (Cobalt Grid Theme)
# ═══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="QKernel-Lite | Clinical Quantum Benchmark",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400;1,6..72,500&family=Hanken+Grotesk:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Hanken Grotesk', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #1F2BE0;
}

/* Canvas & Graph Paper Background */
.stApp {
    background-color: #F0EBDE;
    background-image: 
        linear-gradient(to right, rgba(31, 43, 224, 0.08) 1px, transparent 1px),
        linear-gradient(to bottom, rgba(31, 43, 224, 0.08) 1px, transparent 1px);
    background-size: 32px 32px;
    color: #1F2BE0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #E6E0CE;
    border-right: 1.5px solid #1F2BE0;
}

section[data-testid="stSidebar"] * {
    color: #1F2BE0 !important;
}

/* Editorial Hero Frame */
.hero-container {
    border-top: 2px solid #1F2BE0;
    border-bottom: 2px solid #1F2BE0;
    background: #E6E0CE;
    padding: 22px 28px;
    margin-bottom: 22px;
}

.hero-kicker {
    font-family: 'Hanken Grotesk', sans-serif;
    font-weight: 700;
    font-size: 0.78rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #1F2BE0;
    margin-bottom: 6px;
}

.hero-title {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 2.5rem;
    font-weight: 400;
    line-height: 1.0;
    color: #1F2BE0;
    margin: 0 0 8px 0;
}

.hero-subtitle {
    font-family: 'Hanken Grotesk', sans-serif;
    font-size: 0.95rem;
    color: #1F2BE0;
    line-height: 1.45;
}

/* Editorial Cards */
.editorial-card {
    background: #EAE4D4;
    border: 1.2px solid #1F2BE0;
    border-radius: 2px;
    padding: 16px 20px;
    margin-bottom: 16px;
}

.card-title {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1.25rem;
    font-weight: 500;
    color: #1F2BE0;
    margin-bottom: 10px;
    border-bottom: 1px solid rgba(31, 43, 224, 0.25);
    padding-bottom: 4px;
}

/* Metric Display */
div[data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace;
    font-weight: 500;
    color: #1F2BE0 !important;
    font-size: 1.8rem !important;
}

div[data-testid="stMetricLabel"] {
    font-family: 'Hanken Grotesk', sans-serif;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.74rem !important;
    color: #1F2BE0 !important;
}

/* Editorial Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background-color: #E6E0CE;
    border: 1.5px solid #1F2BE0;
    padding: 4px;
    border-radius: 2px;
    margin-bottom: 20px;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Hanken Grotesk', sans-serif;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.82rem;
    color: #1F2BE0;
    padding: 8px 18px;
    border-radius: 0;
}

.stTabs [aria-selected="true"] {
    background-color: #1F2BE0 !important;
    color: #F0EBDE !important;
}

/* Callouts */
.consensus-box {
    border-left: 3px solid #1F2BE0;
    background: #E6E0CE;
    padding: 10px 14px;
    font-size: 0.88rem;
    line-height: 1.45;
    margin-top: 12px;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
#  Cached Data & Benchmark Results Loading
# ═══════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_data():
    train_ds, val_ds, test_ds, info = load_pneumoniamnist()
    X_train_full = np.load(get_cache_path("train"))
    X_test_full = np.load(get_cache_path("test"))
    return train_ds, test_ds, X_train_full, X_test_full


@st.cache_data
def load_results():
    if config.RESULTS_CSV.exists():
        return pd.read_csv(config.RESULTS_CSV)
    return None

if not get_cache_path("train").exists():
    st.error("Feature cache not found. Please verify cache directory.")
    st.stop()

train_ds, test_ds, X_train_full, X_test_full = load_data()
y_test = test_ds.labels.flatten()
results_df = load_results()

# ═══════════════════════════════════════════════════════════════════════
#  Top Hero Frame
# ═══════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero-container">
    <div class="hero-kicker">IEG313: Quantum Computing for Engineers · Capstone Prototype</div>
    <div class="hero-title">QKernel-Lite: Projected Quantum Kernels on Medical Scarcity</div>
    <div class="hero-subtitle">
        Empirical evaluation of Projected Quantum Kernel Support Vector Machines (PQK-SVM) against classical deep transfer learning baselines across 960 runs on PneumoniaMNIST (df = 14, 15 seeds).
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
#  Sidebar Controls
# ═══════════════════════════════════════════════════════════════════════

st.sidebar.markdown("### Benchmark Parameters")
N = st.sidebar.selectbox("Training Subset Size N", config.N_VALUES, index=1)
seed = st.sidebar.number_input("Random Seed (0 to 14)", min_value=0, max_value=14, value=0)

# Retrieve tuned hyperparameters from results.csv
tuned_scale_default = 0.25
tuned_gamma_m3 = None
tuned_c_m3 = 1.0
tuned_gamma_m2 = None
tuned_c_m2 = 1.0

if results_df is not None:
    m3_row = results_df[(results_df["model"] == "M3") & (results_df["N"] == N) & (results_df["seed"] == seed)]
    if len(m3_row) > 0:
        if "best_scale" in m3_row and pd.notna(m3_row["best_scale"].values[0]):
            tuned_scale_default = float(m3_row["best_scale"].values[0])
        if "best_gamma" in m3_row and pd.notna(m3_row["best_gamma"].values[0]):
            tuned_gamma_m3 = float(m3_row["best_gamma"].values[0])
        if "best_C" in m3_row and pd.notna(m3_row["best_C"].values[0]):
            tuned_c_m3 = float(m3_row["best_C"].values[0])

    m2_row = results_df[(results_df["model"] == "M2") & (results_df["qubits"] == config.PCA_COMPONENTS) & (results_df["N"] == N) & (results_df["seed"] == seed)]
    if len(m2_row) > 0:
        if "best_gamma" in m2_row and pd.notna(m2_row["best_gamma"].values[0]):
            tuned_gamma_m2 = float(m2_row["best_gamma"].values[0])
        if "best_C" in m2_row and pd.notna(m2_row["best_C"].values[0]):
            tuned_c_m2 = float(m2_row["best_C"].values[0])

st.sidebar.markdown("---")
st.sidebar.markdown("### Quantum Encoding Scale")
scale_override = st.sidebar.select_slider(
    "Angle Compression Factor s",
    options=[0.0625, 0.125, 0.25, 0.50, 1.00],
    value=tuned_scale_default,
    help="θ → s·θ. Suppresses phase wrapping turnaround at θ = π/2."
)

if scale_override == tuned_scale_default:
    st.sidebar.caption(f"✓ Using CV-Tuned Scale (s = {scale_override})")
else:
    st.sidebar.caption(f"⚡ Manual Override: s = {scale_override} (Tuned: {tuned_scale_default})")

st.sidebar.markdown("---")
st.sidebar.caption("Dirac's Army · Group 5 · Anshuman Biswas (2024BCD0008)")

# ═══════════════════════════════════════════════════════════════════════
#  Tabs Navigation
# ═══════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "01 / Live Diagnostic Predict",
    "02 / Quantum Circuit & Mechanics",
    "03 / Metric Space & Kernel Explorer",
    "04 / Empirical Benchmark Results",
])

# ─────────────────────────────────────────────────────────────────────
# TAB 1: Live Diagnostic Predict
# ─────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="card-title">Patient Chest X-Ray Diagnostic Evaluation</div>', unsafe_allow_html=True)

    col_img, col_pred = st.columns([1, 2])

    with col_img:
        st.markdown('<div class="editorial-card">', unsafe_allow_html=True)
        test_idx = st.number_input("Test Scan Index (0 to 623)", min_value=0, max_value=len(y_test)-1, value=42)
        if st.button("Shuffle Random Patient Scan"):
            test_idx = int(np.random.randint(0, len(y_test)))

        img = test_ds.imgs[test_idx]
        img_show = img[:, :, 0] if (img.ndim == 3 and img.shape[-1] == 1) else img

        fig, ax = plt.subplots(figsize=(3.4, 3.4), facecolor="#F0EBDE")
        ax.imshow(img_show, cmap="bone")
        true_label = "PNEUMONIA" if y_test[test_idx] == 1 else "NORMAL"
        ax.set_title(f"Clinical Ground Truth: {true_label}", fontsize=10, fontweight="bold", color="#1F2BE0", pad=8)
        ax.axis("off")
        st.pyplot(fig)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_pred:
        train_indices = stratified_subset_indices(train_ds, n=N, seed=seed)
        _, y_train = get_subset_data(train_ds, train_indices)

        X_train_scaled, X_test_scaled, pp = preprocess_for_experiment(
            X_train_full, X_test_full, train_indices,
            n_components=config.PCA_COMPONENTS,
        )

        st.markdown('<div class="editorial-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Tri-Model Decision Comparison</div>', unsafe_allow_html=True)

        pred_cols = st.columns(3)

        # M1: Scratch CNN
        with pred_cols[0]:
            st.markdown("**M1: Scratch CNN**")
            m1_path = config.CACHE_DIR / f"m1_weights_N{N}_s{seed}.pt"
            prob1 = 0.5
            pred1 = "NORMAL"
            if m1_path.exists():
                model = SmallCNN(input_size=28)
                model.load_state_dict(torch.load(m1_path, map_location="cpu", weights_only=True))
                model.eval()
                test_img = test_ds.imgs[test_idx:test_idx+1]
                prob1 = float(predict_m1(model, test_img)[0])
                pred1 = "PNEUMONIA" if prob1 > 0.5 else "NORMAL"
                st.metric("Posterior", f"{prob1:.3f}")
                st.markdown(f"**Pred:** `{pred1}`")
            else:
                st.info("Weights not cached.")

        # M2: Classical RBF-SVM
        with pred_cols[1]:
            st.markdown("**M2: Classical RBF-SVM**")
            gamma2 = tuned_gamma_m2 if tuned_gamma_m2 is not None else compute_base_gamma(X_train_scaled)
            c2 = tuned_c_m2 if tuned_c_m2 is not None else 1.0
            K_train2 = compute_rbf_kernel(X_train_scaled, X_train_scaled, gamma2)
            svm2 = SVC(kernel="precomputed", C=c2)
            svm2.fit(K_train2, y_train)
            K_test2 = compute_rbf_kernel(X_test_scaled[test_idx:test_idx+1], X_train_scaled, gamma2)
            score2 = float(svm2.decision_function(K_test2)[0])
            pred2 = "PNEUMONIA" if score2 > 0 else "NORMAL"
            st.metric("Margin Score", f"{score2:+.3f}")
            st.markdown(f"**Pred:** `{pred2}`")

        # M3: Quantum PQK-SVM
        with pred_cols[2]:
            st.markdown(f"**M3: PQK-SVM (s={scale_override})**")
            score3 = 0.0
            pred3 = "NORMAL"
            try:
                E_train = get_cached_expectations(
                    X_train_scaled, "train_sub", seed, N, scale=scale_override, verbose=False
                )
                E_test = get_cached_expectations(
                    X_test_scaled, "test", seed, N, scale=scale_override, verbose=False
                )
                gamma3 = tuned_gamma_m3 if tuned_gamma_m3 is not None else compute_base_gamma(E_train)
                c3 = tuned_c_m3 if tuned_c_m3 is not None else 1.0
                K_train3 = compute_pqk_kernel(E_train, E_train, gamma3)
                svm3 = SVC(kernel="precomputed", C=c3)
                svm3.fit(K_train3, y_train)
                K_test3 = compute_pqk_kernel(E_test[test_idx:test_idx+1], E_train, gamma3)
                score3 = float(svm3.decision_function(K_test3)[0])
                pred3 = "PNEUMONIA" if score3 > 0 else "NORMAL"
                st.metric("Quantum Margin", f"{score3:+.3f}")
                st.markdown(f"**Pred:** `{pred3}`")
            except Exception as e:
                st.warning(f"Expectations: {e}")

        # Consensus verdict
        agreement = (pred1 == pred2 == pred3 == true_label)
        if agreement:
            st.markdown(f"""
            <div class="consensus-box">
                <strong>Full Clinical Consensus:</strong> All three architectures agree with ground truth ({true_label}).
            </div>
            """, unsafe_allow_html=True)
        elif pred2 == pred3 == true_label:
            st.markdown(f"""
            <div class="consensus-box">
                <strong>Kernel Baseline Agreement:</strong> Classical RBF-SVM and Quantum PQK correctly identified {true_label}, while scratch CNN failed due to small-sample memorization.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="consensus-box">
                <strong>Diagnostic Boundary Conflict:</strong> Models diverge on this scan (Ground Truth: {true_label}). Highlights margin sensitivity under N = {N}.
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# TAB 2: Quantum Circuit & Mechanics
# ─────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="card-title">PennyLane Hardware Circuit & Pauli Projections</div>', unsafe_allow_html=True)

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    with col_ctrl1:
        circ_qubits = st.selectbox("Number of Qubits", [4, 6, 8], index=2)
    with col_ctrl2:
        circ_layers = st.selectbox("Circuit Layers", [1, 2], index=1)
    with col_ctrl3:
        circ_ent = st.selectbox("Entanglement Mode", [True, False], format_func=lambda x: "Circular CNOT Ring" if x else "None (Product State)", index=0)

    col_diag, col_angles = st.columns([1, 1])

    with col_diag:
        st.markdown('<div class="editorial-card">', unsafe_allow_html=True)
        st.markdown(f"**QNode Hardware Diagram ({circ_qubits}Q, {circ_layers}L)**")
        diag = draw_circuit(circ_qubits, circ_layers, circ_ent)
        st.code(diag, language="text")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_angles:
        st.markdown('<div class="editorial-card">', unsafe_allow_html=True)
        st.markdown("**ResNet-18 PCA Rotation Angles θ_i ∈ [0, π]**")
        pca_vals = X_test_scaled[test_idx]
        fig, ax = plt.subplots(figsize=(5.5, 3.2), facecolor="#F0EBDE")
        ax.set_facecolor("#F0EBDE")
        ax.bar(range(len(pca_vals)), pca_vals, color="#1F2BE0", edgecolor="#1F2BE0")
        ax.set_xlabel("Principal Component i", color="#1F2BE0")
        ax.set_ylabel("Rotation Angle (rad)", color="#1F2BE0")
        ax.tick_params(colors="#1F2BE0")
        ax.grid(True, linestyle="--", alpha=0.3, color="#1F2BE0")
        for spine in ax.spines.values():
            spine.set_color("#1F2BE0")
        st.pyplot(fig)
        plt.close()
        st.caption(f"Explained Variance (8 PCs): {pp.total_explained_variance:.2%}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="editorial-card">', unsafe_allow_html=True)
    st.markdown(f"**Local Pauli Expectations (s = {scale_override}): ⟨X_i⟩, ⟨Y_i⟩, ⟨Z_i⟩**")
    try:
        E_test = get_cached_expectations(X_test_scaled, "test", seed, N, scale=scale_override, verbose=False)
        exp_vals = E_test[test_idx]
        fig, ax = plt.subplots(figsize=(11, 2.8), facecolor="#F0EBDE")
        ax.set_facecolor("#F0EBDE")
        colors = ["#1F2BE0", "#5560E5", "#0f172a"] * config.DEFAULT_QUBITS
        ax.bar(range(len(exp_vals)), exp_vals, color=colors[:len(exp_vals)], edgecolor="#1F2BE0")
        labels = []
        for q in range(config.DEFAULT_QUBITS):
            labels.extend([f"Q{q}X", f"Q{q}Y", f"Q{q}Z"])
        ax.set_xticks(range(len(exp_vals)))
        ax.set_xticklabels(labels, rotation=90, fontsize=8, color="#1F2BE0")
        ax.set_ylabel("Expectation", color="#1F2BE0")
        ax.set_ylim(-1.15, 1.15)
        ax.tick_params(colors="#1F2BE0")
        ax.grid(True, linestyle="--", alpha=0.3, color="#1F2BE0")
        for spine in ax.spines.values():
            spine.set_color("#1F2BE0")
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.info("Loading expectation values...")
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# TAB 3: Metric Space & Kernel Explorer
# ─────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="card-title">Metric Space & Gram Matrix Condition</div>', unsafe_allow_html=True)

    col_k1, col_k2 = st.columns(2)
    sort_idx = np.argsort(y_train)
    X_sorted = X_train_scaled[sort_idx]
    y_sorted = y_train[sort_idx]
    n_class0 = int(np.sum(y_sorted == 0))

    with col_k1:
        st.markdown('<div class="editorial-card">', unsafe_allow_html=True)
        st.markdown("**Classical M2: RBF Gram Matrix**")
        gamma_rbf = tuned_gamma_m2 if tuned_gamma_m2 is not None else compute_base_gamma(X_sorted)
        K_rbf = compute_rbf_kernel(X_sorted, X_sorted, gamma_rbf)
        diag_rbf = kernel_diagnostics(K_rbf)

        fig, ax = plt.subplots(figsize=(4.2, 4.2), facecolor="#F0EBDE")
        im = ax.imshow(K_rbf, cmap="Blues", vmin=0, vmax=1)
        ax.axhline(y=n_class0 - 0.5, color="#1F2BE0", linewidth=1.2, linestyle="--")
        ax.axvline(x=n_class0 - 0.5, color="#1F2BE0", linewidth=1.2, linestyle="--")
        ax.axis("off")
        plt.colorbar(im, ax=ax, shrink=0.75)
        st.pyplot(fig)
        plt.close()
        st.markdown(f"• Off-diagonal Spread (std): `{diag_rbf['kernel_offdiag_std']:.4f}`")
        st.markdown(f"• Minimum Eigenvalue: `{diag_rbf['kernel_min_eigenvalue']:.4e}`")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_k2:
        st.markdown('<div class="editorial-card">', unsafe_allow_html=True)
        st.markdown(f"**Quantum M3: PQK Gram Matrix (s={scale_override})**")
        try:
            E_train = get_cached_expectations(X_train_scaled, "train_sub", seed, N, scale=scale_override, verbose=False)
            E_sorted = E_train[sort_idx]
            gamma_pqk = tuned_gamma_m3 if tuned_gamma_m3 is not None else compute_base_gamma(E_sorted)
            K_pqk = compute_pqk_kernel(E_sorted, E_sorted, gamma_pqk)
            diag_pqk = kernel_diagnostics(K_pqk)

            fig, ax = plt.subplots(figsize=(4.2, 4.2), facecolor="#F0EBDE")
            im = ax.imshow(K_pqk, cmap="Blues", vmin=0, vmax=1)
            ax.axhline(y=n_class0 - 0.5, color="#1F2BE0", linewidth=1.2, linestyle="--")
            ax.axvline(x=n_class0 - 0.5, color="#1F2BE0", linewidth=1.2, linestyle="--")
            ax.axis("off")
            plt.colorbar(im, ax=ax, shrink=0.75)
            st.pyplot(fig)
            plt.close()
            st.markdown(f"• Off-diagonal Spread (std): `{diag_pqk['kernel_offdiag_std']:.4f}`")
            st.markdown(f"• Minimum Eigenvalue: `{diag_pqk['kernel_min_eigenvalue']:.4e}`")
            if diag_pqk['kernel_offdiag_std'] < 0.02:
                st.warning("⚠️ High concentration: kernel entries collapsing toward uniform similarity.")
        except Exception as e:
            st.info(f"Computing quantum kernel: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# TAB 4: Empirical Benchmark Results
# ─────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="card-title">Empirical Benchmark Findings (15 Seeds, 960 Completed Runs)</div>', unsafe_allow_html=True)

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.metric("Total Executions", "960 Runs", "df = 14")
    with sc2:
        st.metric("M3 Deficit (N=100)", "−0.0288 AUC", "p = 0.00015")
    with sc3:
        st.metric("1L Unentangled vs M2", "±0.02 Bound", "Equivalent (N≥20)")
    with sc4:
        st.metric("Bonferroni Gate", "α = 0.00417", "12 Hypotheses")

    st.markdown("---")
    st.markdown("**Synchronized Publication Figures**")

    fig_tabs = st.tabs([
        "Fig 1: AUC vs N",
        "Fig 2: Paired Difference",
        "Fig 3: 2×2 Ablation",
        "Fig 4: Diagnostics",
        "Fig 5: Phase Wrapping",
    ])

    with fig_tabs[0]:
        st.image("results/fig1_auc_vs_n.png", caption="Figure 1: Mean Test ROC-AUC vs N across 15 seeds (M3 labeled as post-hoc CV-tuned).", width="stretch")
    with fig_tabs[1]:
        st.image("results/fig2_paired_difference.png", caption="Figure 2: Paired difference (M3-tuned − M2) across training set sizes with exact 95% t-CI.", width="stretch")
    with fig_tabs[2]:
        st.image("results/fig3_ablation.png", caption="Figure 3: Systematic 2×2 ablation grid (1L/2L × Entangled/Unentangled) with exact 95% t-CIs.", width="stretch")
    with fig_tabs[3]:
        st.image("results/fig4_kernel_concentration.png", caption="Figure 4: Physical diagnostics: feature variance collapse and Spearman rank correlation.", width="stretch")
    with fig_tabs[4]:
        st.image("results/fig5_phase_wrapping.png", caption="Figure 5: Physical mechanism of phase wrapping and bandwidth scaling.", width="stretch")

    st.markdown("---")
    st.markdown("**Interactive Dataset Query**")
    if results_df is not None:
        sel_models = st.multiselect("Filter Models", options=sorted(results_df["model"].unique()), default=["M1", "M2", "M3"])
        filtered_df = results_df[results_df["model"].isin(sel_models)]
        st.dataframe(filtered_df, width="stretch")
        st.download_button(
            "Download Filtered Results CSV",
            data=filtered_df.to_csv(index=False).encode('utf-8'),
            file_name="qkernel_lite_results.csv",
            mime="text/csv"
        )
