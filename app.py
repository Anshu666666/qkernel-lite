"""
QKernel-Lite – Research Prototype & Empirical Benchmark Platform
================================================================
Design System: Cobalt Grid Editorial Theme (Warm Ivory #F0EBDE & Cobalt Blue #1F2BE0)
Fonts: Newsreader (Editorial Serif), Hanken Grotesk (Sans), DM Mono (Monospace)

Architecture: Self-contained, lightweight runtime driven by precomputed prototype database
(data/prototype_db.npz). Zero heavy dependencies (no PyTorch, no PennyLane, no MedMNIST at runtime).
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent
DB_PATH = ROOT_DIR / "data" / "prototype_db.npz"
RESULTS_CSV = ROOT_DIR / "results" / "results.csv"

# ═══════════════════════════════════════════════════════════════════════
#  Page Config & Editorial Light Styling (Cobalt Grid Theme)
# ═══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="QKernel-Lite | Quantum vs Classical Benchmark",
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

/* Top Hero Frame */
.hero-container {
    border-top: 2px solid #1F2BE0;
    border-bottom: 2px solid #1F2BE0;
    background: #E6E0CE;
    padding: 20px 26px;
    margin-bottom: 18px;
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
    font-size: 2.3rem;
    font-weight: 400;
    line-height: 1.05;
    color: #1F2BE0;
    letter-spacing: -0.01em;
    margin-bottom: 6px;
}

.hero-subtitle {
    font-family: 'Hanken Grotesk', sans-serif;
    font-size: 0.95rem;
    line-height: 1.45;
    color: #1F2BE0;
    opacity: 0.9;
}

/* Streamlit Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: #E6E0CE;
    border-bottom: 2px solid #1F2BE0;
    gap: 4px;
    padding: 4px 8px 0 8px;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    color: #1F2BE0 !important;
    background-color: transparent !important;
    border: 1px solid transparent !important;
    border-bottom: none !important;
    padding: 8px 18px !important;
    border-top-left-radius: 4px !important;
    border-top-right-radius: 4px !important;
}

.stTabs [aria-selected="true"] {
    background-color: #F0EBDE !important;
    color: #1F2BE0 !important;
    border: 1.5px solid #1F2BE0 !important;
    border-bottom: 2px solid #F0EBDE !important;
    margin-bottom: -2px !important;
}

/* Metric Boxes */
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1.2px solid #1F2BE0;
    padding: 10px 14px;
    border-radius: 4px;
}
div[data-testid="stMetricValue"] {
    font-family: 'Newsreader', Georgia, serif !important;
    color: #1F2BE0 !important;
    font-size: 1.9rem !important;
}
div[data-testid="stMetricLabel"] {
    font-family: 'Hanken Grotesk', sans-serif !important;
    color: #1F2BE0 !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-size: 0.75rem !important;
}

/* Native Container Borders */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1.2px solid #1F2BE0 !important;
    border-radius: 4px !important;
}

/* Buttons */
.stButton > button {
    background-color: #1F2BE0 !important;
    color: #F0EBDE !important;
    font-family: 'Hanken Grotesk', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    border: 1.5px solid #1F2BE0 !important;
    border-radius: 3px !important;
    padding: 6px 16px !important;
}
.stButton > button:hover {
    background-color: #5560E5 !important;
    color: #ffffff !important;
    border-color: #5560E5 !important;
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
#  Cached Prototype Database Loading
# ═══════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_prototype_data():
    if not DB_PATH.exists():
        return None, None, None, None, None
    raw = np.load(DB_PATH, allow_pickle=True)
    test_images = raw["test_images"]
    test_labels = raw["test_labels"]
    circuits = raw["circuits"].item()
    db = raw["db"].item()
    
    results_df = None
    if RESULTS_CSV.exists():
        results_df = pd.read_csv(RESULTS_CSV)
    return test_images, test_labels, circuits, db, results_df

test_images, test_labels, circuits, db, results_df = load_prototype_data()

if db is None:
    st.error("Prototype database (data/prototype_db.npz) not found. Please run scripts/build_prototype_db.py.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════
#  Top Hero Frame & Mandatory Research Disclaimer
# ═══════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero-container">
    <div class="hero-kicker">IEG313: Quantum Computing for Engineers · Capstone Research Platform</div>
    <div class="hero-title">QKernel-Lite: Projected Quantum Kernels on Medical Scarcity</div>
    <div class="hero-subtitle">
        Empirical evaluation of Projected Quantum Kernel Support Vector Machines (PQK-SVM) against classical deep transfer learning baselines across 960 runs on PneumoniaMNIST (df = 14, 15 seeds).
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background:#fef2f2; border:1.5px solid #dc2626; color:#991b1b; padding:10px 16px; border-radius:4px; margin-bottom:18px; font-size:0.86rem; line-height:1.45;">
    <strong>RESEARCH EVALUATION PROTOTYPE — NOT FOR CLINICAL DIAGNOSTIC USE:</strong>
    This software is an engineering capstone benchmark exploring quantum kernel methods in medical imaging. It is NOT certified as a medical device or diagnostic decision-support system and must never be used to guide patient treatment or clinical triage.
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
#  Sidebar Controls
# ═══════════════════════════════════════════════════════════════════════

st.sidebar.markdown("### Benchmark Parameters")
N = st.sidebar.selectbox("Training Subset Size N", [10, 20, 50, 100], index=1)
seed = st.sidebar.number_input("Random Seed (0 to 14)", min_value=0, max_value=14, value=0)

key = f"N{N}_s{seed}"
if key not in db:
    key = "N20_s0"
entry = db[key]

default_scale = entry.get("best_scale", 0.125)
st.sidebar.markdown("---")
st.sidebar.markdown("### Quantum Encoding Scale")
scale_override = st.sidebar.select_slider(
    "Angle Compression Factor s",
    options=[0.0625, 0.125, 0.25, 0.50, 1.00],
    value=default_scale,
    help="Scales PCA features before angle encoding. s <= 0.25 suppresses phase wrapping and linearizes quantum features."
)

if scale_override == default_scale:
    st.sidebar.caption(f"✓ Using CV-Tuned Scale (s = {default_scale})")
else:
    st.sidebar.caption(f"⚡ Interactive Override: s = {scale_override}")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Dirac's Army · Group 5**
* Course IEG313 · IIIT Kottayam
* Authors: Daksh Singh, Anshuman Biswas, Himmat Sirwee, Sawan Kumar, Aryan Gupta
""")

# ═══════════════════════════════════════════════════════════════════════
#  Main Application Tabs
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

    st.markdown("""
    <div style="background:#EAE4D4; border:1.2px solid #1F2BE0; padding:12px 16px; border-radius:4px; margin-bottom:16px; font-size:0.86rem; line-height:1.45;">
        <strong>How to Interpret the Decision Scores:</strong><br>
        • <strong>M1: Scratch CNN</strong> — Posterior probability P(Pneumonia) ∈ [0, 1]. Values > 0.5 predict Pneumonia; values ≤ 0.5 predict Normal.<br>
        • <strong>M2: Classical RBF-SVM</strong> — Signed distance f(x) = w·Φ(x) + b from the separating hyperplane. Positive values indicate Pneumonia; negative values indicate Normal.<br>
        • <strong>M3: Quantum PQK-SVM</strong> — Signed distance f(x) in Hilbert-projected metric space after 8-qubit evolution and 24 Pauli projections.<br>
        • <strong>Empirical Takeaway:</strong> Classical transfer learning (M2) accounts for 100% of diagnostic capability (+0.36 AUC over scratch CNN at N=10). Quantum encoding adds no statistical lift.
    </div>
    """, unsafe_allow_html=True)

    col_img, col_pred = st.columns([1, 2])

    with col_img:
        with st.container(border=True):
            test_idx = st.number_input("Test Scan Index (0 to 623)", min_value=0, max_value=len(test_labels)-1, value=42)
            if st.button("Shuffle Random Patient Scan"):
                test_idx = int(np.random.randint(0, len(test_labels)))

            img = test_images[test_idx]
            fig, ax = plt.subplots(figsize=(3.4, 3.4), facecolor="#F0EBDE")
            ax.imshow(img, cmap="bone")
            true_label = "PNEUMONIA" if test_labels[test_idx] == 1 else "NORMAL"
            ax.set_title(f"Ground Truth Label: {true_label}", fontsize=10, fontweight="bold", color="#1F2BE0", pad=8)
            ax.axis("off")
            st.pyplot(fig)
            plt.close()

    with col_pred:
        with st.container(border=True):
            st.markdown('<div class="card-title">Tri-Model Decision Outputs</div>', unsafe_allow_html=True)

            pred_cols = st.columns(3)

            # M1: Scratch CNN
            prob1 = float(entry["m1_probs"][test_idx])
            pred1 = "PNEUMONIA" if prob1 > 0.5 else "NORMAL"
            with pred_cols[0]:
                st.markdown("**M1: Scratch CNN**")
                st.metric("Posterior Prob", f"{prob1:.3f}")
                st.markdown(f"**Pred:** `{pred1}`")

            # M2: Classical RBF-SVM
            score2 = float(entry["m2_scores"][test_idx])
            pred2 = "PNEUMONIA" if score2 > 0 else "NORMAL"
            with pred_cols[1]:
                st.markdown("**M2: Classical RBF-SVM**")
                st.metric("Margin Score", f"{score2:+.3f}")
                st.markdown(f"**Pred:** `{pred2}`")

            # M3: Quantum PQK-SVM
            m3_scores = entry["m3_scores_by_scale"].get(str(scale_override), entry["m2_scores"])
            score3 = float(m3_scores[test_idx])
            pred3 = "PNEUMONIA" if score3 > 0 else "NORMAL"
            with pred_cols[2]:
                st.markdown(f"**M3: PQK-SVM (s={scale_override})**")
                st.metric("Quantum Margin", f"{score3:+.3f}")
                st.markdown(f"**Pred:** `{pred3}`")

            # Agreement & Consensus Assessment
            st.markdown("---")
            if pred1 == pred2 == pred3 == true_label:
                st.markdown(f"""
                <div class="consensus-box">
                    <strong>Complete Architectural Agreement:</strong> All three models concordantly classified this patient as <strong>{true_label}</strong>.
                </div>
                """, unsafe_allow_html=True)
            elif pred2 == pred3 == true_label:
                st.markdown(f"""
                <div class="consensus-box">
                    <strong>Kernel Baseline Agreement:</strong> Classical RBF-SVM and Quantum PQK correctly identified <strong>{true_label}</strong>, while scratch CNN diverged.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="consensus-box">
                    <strong>Decision Boundary Conflict:</strong> Models diverge on this patient scan (Ground Truth: <strong>{true_label}</strong>). Illustrates classification sensitivity under scarce training size N = {N}.
                </div>
                """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# TAB 2: Quantum Circuit & Mechanics
# ─────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="card-title">PennyLane Hardware Circuit & Pauli Projections</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#EAE4D4; border:1.2px solid #1F2BE0; padding:12px 16px; border-radius:4px; margin-bottom:16px; font-size:0.86rem; line-height:1.45;">
        <strong>Quantum Circuit Physics:</strong><br>
        • <strong>Angle Encoding:</strong> Each orthogonal PCA component θ_i is converted to qubit rotations R_Y(sθ_i) and R_Z(sθ_i).<br>
        • <strong>Phase Wrapping:</strong> At default s = 1.0, rotations double in 2 layers. Since U(π)²|0⟩ = |0⟩, ⟨Z⟩ returns to +1 at θ = π, folding distant features together.<br>
        • <strong>Entanglement Collapse:</strong> Circular CNOT gates correlate qubits, shrinking reduced 1-qubit Bloch vectors toward the center of the sphere and crushing expectation variance from 0.29 down to 0.0085.
    </div>
    """, unsafe_allow_html=True)

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    with col_ctrl1:
        circ_qubits = st.selectbox("Number of Qubits", [4, 6, 8], index=2)
    with col_ctrl2:
        circ_layers = st.selectbox("Circuit Layers", [1, 2], index=1)
    with col_ctrl3:
        circ_ent = st.selectbox("Entanglement Mode", [True, False], format_func=lambda x: "Circular CNOT Ring" if x else "None (Product State)", index=0)

    col_diag, col_angles = st.columns([1, 1])

    with col_diag:
        with st.container(border=True):
            st.markdown(f"**QNode Hardware Diagram ({circ_qubits}Q, {circ_layers}L)**")
            circ_key = f"{circ_qubits}Q_{circ_layers}L_{circ_ent}"
            diag_text = circuits.get(circ_key, "Circuit diagram precomputed.")
            st.code(diag_text, language="text")

    with col_angles:
        with st.container(border=True):
            st.markdown("**ResNet-18 PCA Rotation Angles θ_i ∈ [0, π]**")
            pca_vals = entry["pca_angles"][test_idx][:circ_qubits]
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
            st.caption(f"Total Explained Variance (8 PCs): {entry['pca_explained_var']:.2%}")

    with st.container(border=True):
        st.markdown(f"**Local Pauli Expectations (s = {scale_override}): ⟨X_i⟩, ⟨Y_i⟩, ⟨Z_i⟩**")
        exp_all = entry["exp_vals_by_scale"].get(str(scale_override), np.zeros((624, 24)))
        exp_vals = exp_all[test_idx]
        fig, ax = plt.subplots(figsize=(11, 2.8), facecolor="#F0EBDE")
        ax.set_facecolor("#F0EBDE")
        colors = ["#1F2BE0", "#5560E5", "#0f172a"] * 8
        ax.bar(range(len(exp_vals)), exp_vals, color=colors[:len(exp_vals)], edgecolor="#1F2BE0")
        labels = []
        for q in range(8):
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

# ─────────────────────────────────────────────────────────────────────
# TAB 3: Metric Space & Kernel Explorer
# ─────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="card-title">Metric Space & Gram Matrix Condition</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#EAE4D4; border:1.2px solid #1F2BE0; padding:12px 16px; border-radius:4px; margin-bottom:16px; font-size:0.86rem; line-height:1.45;">
        <strong>Kernel Diagnostics & Spread Metrics:</strong><br>
        • <strong>Off-Diagonal Spread (std):</strong> Measures discriminative contrast. When spread is high, the kernel easily separates distinct patients. If std < 0.02, the kernel is collapsing to uniform similarity.<br>
        • <strong>Minimum Eigenvalue:</strong> Confirms positive semi-definiteness (Mercer's condition) ensuring dual SVM optimization remains strictly convex.
    </div>
    """, unsafe_allow_html=True)

    col_k1, col_k2 = st.columns(2)
    n_class0 = entry["n_class0"]

    with col_k1:
        with st.container(border=True):
            st.markdown("**Classical M2: RBF Gram Matrix**")
            K_rbf = entry["K_rbf_sorted"]
            diag_rbf = entry["diag_rbf"]

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

    with col_k2:
        with st.container(border=True):
            st.markdown(f"**Quantum M3: PQK Gram Matrix (s={scale_override})**")
            K_pqk = entry["K_pqk_sorted_by_scale"].get(str(scale_override), K_rbf)
            diag_pqk = entry["diag_pqk_by_scale"].get(str(scale_override), diag_rbf)

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

# ─────────────────────────────────────────────────────────────────────
# TAB 4: Empirical Benchmark Results
# ─────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="card-title">Empirical Benchmark Findings (15 Seeds, 960 Completed Runs)</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#EAE4D4; border:1.2px solid #1F2BE0; padding:12px 16px; border-radius:4px; margin-bottom:16px; font-size:0.86rem; line-height:1.45;">
        <strong>Statistical Takeaways Across 15 Independent Seeds:</strong><br>
        • <strong>M3 Deficit Survives Bonferroni:</strong> M3-tuned is detectably inferior to classical RBF-SVM (M2) at all N: deficits of −0.124 at N=10, −0.085 at N=20, −0.030 at N=50, and −0.029 at N=100 (all p < 0.001).<br>
        • <strong>Classical Equivalence:</strong> The 1-layer unentangled product state is the only quantum model indistinguishable from M2 within ±0.02 AUC (N ≥ 20). It computes per-feature trig functions and is classically simulable.<br>
        • <strong>Tuning Budget Asymmetry:</strong> M3 explored 270 hyperparameter configurations vs. 36–54 for M2. Despite this 5× search advantage, quantum never surpassed the classical control.
    </div>
    """, unsafe_allow_html=True)

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.metric("Total Executions", "960 Runs", "df = 14")
    with sc2:
        st.metric("M3 Deficit (N=100)", "−0.0288 AUC", "p = 0.00015")
    with sc3:
        st.metric("1L Product vs M2", "±0.02 Bound", "Equivalent (N≥20)")
    with sc4:
        st.metric("Entangled Variance", "0.0085", "34× Collapse")

    st.markdown("---")
    st.markdown("### Published Capstone Benchmark Figures")
    fig_cols1 = st.columns(2)
    with fig_cols1[0]:
        f1_path = ROOT_DIR / "results" / "fig1_auc_vs_n.png"
        if f1_path.exists():
            st.image(str(f1_path), caption="Figure 1: Mean ROC-AUC vs Training Set Size N (15 seeds, 95% CI)")
    with fig_cols1[1]:
        f2_path = ROOT_DIR / "results" / "fig2_paired_difference.png"
        if f2_path.exists():
            st.image(str(f2_path), caption="Figure 2: Paired Differences (M3 - M2) with Bonferroni Threshold")

    fig_cols2 = st.columns(3)
    with fig_cols2[0]:
        f3_path = ROOT_DIR / "results" / "fig3_ablation.png"
        if f3_path.exists():
            st.image(str(f3_path), caption="Figure 3: 2x2 Entanglement vs Layers Ablation Grid")
    with fig_cols2[1]:
        f4_path = ROOT_DIR / "results" / "fig4_kernel_concentration.png"
        if f4_path.exists():
            st.image(str(f4_path), caption="Figure 4: Kernel Distance Spread Diagnostics")
    with fig_cols2[2]:
        f5_path = ROOT_DIR / "results" / "fig5_phase_wrapping.png"
        if f5_path.exists():
            st.image(str(f5_path), caption="Figure 5: Phase Wrapping Turnaround at s = 1.00")

    if results_df is not None:
        st.markdown("---")
        st.markdown("### Complete Benchmark Data Ledger (960 Rows)")
        n_col = "N" if "N" in results_df.columns else "n_train"
        sel_models = st.multiselect("Filter Models", options=sorted(results_df["model"].unique()), default=["M1", "M2", "M3"])
        filtered_df = results_df[results_df["model"].isin(sel_models)]
        st.dataframe(filtered_df, use_container_width=True, height=280)
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Results CSV",
            data=csv_data,
            file_name="qkernel_lite_benchmark_filtered.csv",
            mime="text/csv",
        )
