# QKernel-Lite: Empirical Study of Projected Quantum Kernels on Medical Scarcity

> **Dirac's Army** · Academic Capstone & Empirical Benchmark  
> Comparing Scratch CNN (M1), Classical RBF-SVM (M2), and Projected Quantum Kernel SVM (M3/M4) under extreme few-shot regimes (N ∈ {10, 20, 50, 100}) on Chest Radiographs (`PneumoniaMNIST`).

### 📦 Key Project Deliverables
- 📄 **10-Page Formal Academic Report**: [`report/index.html`](report/index.html) (Strict 1-to-1 physical A4 PDF printing)
- 📐 **Archify Interactive Architecture Diagram**: [`report/architecture.html`](report/architecture.html) · [Vector SVG](report/images/architecture.svg)
- 📊 **Cobalt Grid Presentation Slide Deck**: [`slides/index.html`](slides/index.html) (Zero-dependency, keyboard & touch enabled)
- 🌐 **Interactive Streamlit Prototype Platform**: [`app.py`](app.py) (Cobalt Grid editorial light aesthetic)
- ☁️ **Oracle Cloud Always-Free VM Deployment Guide**: [`deploy/DEPLOYMENT_GUIDE.md`](deploy/DEPLOYMENT_GUIDE.md)

---

## 1. Executive Summary & Core Findings

**QKernel-Lite** provides a mathematically rigorous, fully reproducible empirical evaluation of Projected Quantum Kernels (PQK; Huang et al., 2021; Alagiyawanna et al., arXiv:2601.03375) applied to pediatric chest X-ray classification under simulated clinical data scarcity ($N \in \{10, 20, 50, 100\}$). 

Across **960 complete experimental runs** with **100% seed parity** ($n = 15$ seeds, df = 14) and rigorous Bonferroni family-wise error control across 12 statistical hypotheses ($\alpha_{\text{bonf}} = 0.05 / 12 \approx 0.00417$), our empirical findings establish:

1. **Default Encoding Fails:** At default angle encoding ($s = 1.00$), the 2-layer entangled quantum kernel (M3-default) collapses to near-chance performance ($0.5557 \pm 0.0543$ at $N = 10$, $0.5739 \pm 0.0582$ at $N = 20$). This collapse is caused by single-qubit phase wrapping turnaround ($U(\pi)^2|0\rangle = |0\rangle$ up to global phase) and frequency folding.
2. **Tuned Scale Mitigates, But M3 Stays Detectably Below M2:** With a post-hoc CV-tuned encoding scale $s \in \{0.0625, 0.125, 0.25, 0.50, 1.00\}$, M3-tuned recovers substantial accuracy ($0.7717$ to $0.8962$ AUC), but **remains statistically detectably below classical RBF-SVM (M2) across all sample sizes**:
   - $N = 10$: $\Delta_{\text{AUC}} = -0.1241 \pm 0.0543$ ($p = 2.35 \times 10^{-4}$, **Survives Bonferroni**)
   - $N = 20$: $\Delta_{\text{AUC}} = -0.0846 \pm 0.0391$ ($p = 3.84 \times 10^{-4}$, **Survives Bonferroni**)
   - $N = 50$: $\Delta_{\text{AUC}} = -0.0296 \pm 0.0149$ ($p = 7.91 \times 10^{-4}$, **Survives Bonferroni**)
   - $N = 100$: $\Delta_{\text{AUC}} = -0.0288 \pm 0.0120$ ($p = 1.46 \times 10^{-4}$, **Survives Bonferroni**)
3. **Only Classical Product States Achieve Statistical Equivalence:** The only quantum circuit family indistinguishable from classical M2 within a conservative $\pm 0.02$ AUC margin (for $N \ge 20$) is the **1-layer unentangled product state**. This circuit performs independent single-qubit trigonometric mappings and is **trivially classically simulable**.
4. **Entanglement Systematically Degrades Performance:** At matched circuit depth, adding circular CNOT entangling gates causes a statistically significant performance drop across all sample sizes (e.g. at 8Q 2L: drop of $-0.0843$ at $N = 10$, $-0.0859$ at $N = 20$, $-0.0291$ at $N = 50$, and $-0.0206$ at $N = 100$; all $p < 0.01$).
5. **ZZ Correlator Hypothesis Tested & Refuted:** Expanding Pauli projections from 24 single-qubit expectations to 32 observables by adding nearest-neighbor $\langle Z_i Z_{i+1} \rangle$ correlators did not reverse the degradation (AUC dropped by $0.01$ to $0.034$ at $s = 0.25$). Discarding correlations is not the root cause.
6. **M2 Modestly Beats M1 at Intermediate Scarcity:** Classical RBF-SVM significantly outperforms scratch CNN at $N = 20$ ($\Delta = +0.0445$, 95% CI $[+0.0213, +0.0677]$, $p = 0.0010$) and $N = 50$ ($\Delta = +0.0212$, 95% CI $[+0.0084, +0.0339]$, $p = 0.0032$).
7. **Benchmark Saturation:** The ResNet-18 + PCA representation saturates near $0.925$–$0.930$ ROC-AUC, leaving virtually no margin for quantum advantage.

---

## 2. Experimental Architecture

```mermaid
flowchart LR
    A[Raw Chest X-Ray\nPneumoniaMNIST] --> B[Frozen ResNet-18\nImageNet Backbone]
    B --> C[Global Average Pool\n512-d Vector]
    C --> D[PCA Reduction\n8 Principal Components]
    D --> E[Min-Max Scaling\nAngles θ ∈ [0, π]]
    E --> F[Quantum Feature Map\ns·θ Encoding + CNOT Ring]
    F --> G[Pauli Projections\n⟨X⟩, ⟨Y⟩, ⟨Z⟩ per qubit\n24-d Vector]
    G --> H[Projected Kernel\nk(x,x') = exp(-γ||E-E'||²)]
    H --> I[Dual C-SVM Classifier\nROC-AUC, BalAcc, F1]
```

### Models Evaluated

| Model | Architecture | Dimension $d$ | Hyperparameter Optimization | Role & Baseline Definition |
| :--- | :--- | :--- | :--- | :--- |
| **M1: Scratch CNN** | 2-stage Conv-ReLU-Pool $\to$ Linear (105k weights) | Raw $28 \times 28$ | Adam ($\text{lr}=10^{-3}$, 60 epochs) | **Negative Control:** Demonstrates overfitting when $P \gg N$. |
| **M2: Classical RBF-SVM** | Frozen ResNet-18 $\to$ PCA $\to$ $[0, \pi]$ | $d = 8$ (also $4, 6$) | Dual convex QP, CV grid on $(\gamma, C)$ | **Primary Scientific Control:** Holds representation constant. |
| **M3-default: PQK-SVM** | ResNet-18 $\to$ PCA $\to$ 2L Ent Circuit $\to$ 24 Paulis | $d = 24$ | Fixed unscaled encoding ($s = 1.00$), CV on $(\gamma, C)$ | **Literature Default:** Direct application of standard angle encoding. |
| **M3-tuned: PQK-SVM** | ResNet-18 $\to$ PCA $\to$ 2L Ent Circuit $\to$ 24 Paulis | $d = 24$ | Post-hoc CV grid on $(s, \gamma, C)$ (270 configs) | **Proposed Quantum Model:** Angle-compressed quantum kernel. |
| **M4: Ablation Grid** | Qubits $n \in \{4, 6, 8\}$, Layers $L \in \{1, 2\}$, Ent On/Off | $3n$ Paulis | CV grid on $(s, \gamma, C)$ | **11 Ablation Cells:** Isolates depth, width, and entanglement. |

---

## 3. Full Benchmark Results (15 Seeds, Exact Student's t 95% CIs)

### Table 1: Model Comparison Across Sample Sizes ($n = 15$, df = 14)

| Metric | N = 10 | N = 20 | N = 50 | N = 100 |
| :--- | :---: | :---: | :---: | :---: |
| **M1: Scratch CNN** | $0.8469 \pm 0.0423$ | $0.8653 \pm 0.0252$ | $0.9011 \pm 0.0086$ | $0.9110 \pm 0.0103$ |
| **M2: Classical RBF (8Q, tuned)** | $\mathbf{0.8957 \pm 0.0257}$ | $\mathbf{0.9098 \pm 0.0268}$ | $\mathbf{0.9222 \pm 0.0073}$ | $\mathbf{0.9250 \pm 0.0074}$ |
| **M2: Classical RBF (8Q, untuned)** | $0.9066 \pm 0.0208$ | $0.9196 \pm 0.0182$ | $0.9247 \pm 0.0069$ | $0.9271 \pm 0.0064$ |
| **M2: Classical RBF (4Q sensitivity)** | $0.9074 \pm 0.0177$ | $0.9142 \pm 0.0151$ | $0.9142 \pm 0.0088$ | $0.9127 \pm 0.0081$ |
| **M3-default (s = 1.00 unscaled)** | $0.5557 \pm 0.0543$ | $0.5739 \pm 0.0582$ | $0.7479 \pm 0.0277$ | $0.8228 \pm 0.0242$ |
| **M3-tuned (CV-tuned s, post-hoc)** | $0.7717 \pm 0.0610$ | $0.8252 \pm 0.0425$ | $0.8926 \pm 0.0156$ | $0.8962 \pm 0.0126$ |

### Table 2: Paired Differences vs. Classical Baselines & Significance

| Comparison | N = 10 | N = 20 | N = 50 | N = 100 | Bonferroni ($\alpha=0.00417$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **M3-tuned − M2 (8Q tuned)** | $-0.1241 \pm 0.0543$ | $-0.0846 \pm 0.0391$ | $-0.0296 \pm 0.0149$ | $-0.0288 \pm 0.0120$ | **Survives at all N** ($p \le 7.9 \times 10^{-4}$) |
| **M3-tuned − M2 (8Q untuned)** | $-0.1350 \pm 0.0582$ | $-0.0943 \pm 0.0387$ | $-0.0320 \pm 0.0162$ | $-0.0309 \pm 0.0119$ | **Survives at all N** ($p \le 1.8 \times 10^{-4}$) |
| **M3-tuned − M2 (4Q best baseline)** | $-0.1358 \pm 0.0579$ | $-0.0890 \pm 0.0411$ | $-0.0215 \pm 0.0136$ | $-0.0165 \pm 0.0093$ | **Survives at all N** ($p \le 4.3 \times 10^{-3}$) |
| **M2 − M1 (Classical vs CNN)** | $+0.0489 \pm 0.0479$ | $+0.0445 \pm 0.0232$ | $+0.0212 \pm 0.0128$ | $+0.0141 \pm 0.0099$ | **Survives at N=20, 50** ($p \le 0.0032$) |

> **CI Sign Clarification:** For M2 − M1, both lower and upper bounds are strictly positive at $N = 20$ ($[+0.0213, +0.0677]$) and $N = 50$ ($[+0.0084, +0.0339]$), confirming that M2's advantage over M1 is statistically significant.

### Table 3: Equivalence Test: 1-Layer Unentangled (CV-Tuned s) vs. Classical M2 (Bound: $\pm 0.02$ AUC)

| Training Size N | 1L Unentangled Mean | Matched M2 Mean | Paired Difference | Exact 95% t-CI | Result within $\pm 0.02$ Bound |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **N = 10** | $0.8842$ | $0.8957$ | $-0.0115 \pm 0.0127$ | $[-0.0242, +0.0012]$ | **Inconclusive / Fails** (variance extends beyond $-0.02$) |
| **N = 20** | $0.9115$ | $0.9098$ | $+0.0017 \pm 0.0046$ | $[-0.0029, +0.0063]$ | **Statistically Equivalent** |
| **N = 50** | $0.9215$ | $0.9222$ | $-0.0007 \pm 0.0064$ | $[-0.0072, +0.0057]$ | **Statistically Equivalent** |
| **N = 100** | $0.9211$ | $0.9250$ | $-0.0039 \pm 0.0072$ | $[-0.0111, +0.0033]$ | **Statistically Equivalent** |

*Note: This is an exact 95% CI equivalence check (equivalent to two one-sided tests at $\alpha = 0.025$), which is strictly conservative compared to standard 90% CI equivalence testing.*

### Table 4: Entanglement Degradation at Matched Depth (8 Qubits, 2 Layers, 15 Seeds)

| Sample Size N | Entangled (M3) | Unentangled (2L) | Paired Difference (Ent − Unent) | 95% CI | p-value | Scale Edge Hits (Ent vs Unent) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **N = 10** | $0.7717$ | $0.8560$ | $-0.0843 \pm 0.0592$ | $[-0.1436, -0.0251]$ | $p = 0.0086$ | Ent: 4 low, 3 high \| Unent: 0 low, 2 high |
| **N = 20** | $0.8252$ | $0.9111$ | $-0.0859 \pm 0.0416$ | $[-0.1274, -0.0443]$ | $p = 5.69 \times 10^{-4}$ | Ent: 0 low, 0 high \| Unent: 0 low, 0 high |
| **N = 50** | $0.8926$ | $0.9217$ | $-0.0291 \pm 0.0156$ | $[-0.0447, -0.0134]$ | $p = 1.35 \times 10^{-3}$ | Ent: 3 low, 0 high \| Unent: 2 low, 0 high |
| **N = 100** | $0.8962$ | $0.9168$ | $-0.0206 \pm 0.0103$ | $[-0.0310, -0.0103]$ | $p = 7.77 \times 10^{-4}$ | Ent: 4 low, 0 high \| Unent: 3 low, 1 high |

---

## 4. Methodological Disclosures & Scientific Honesty

In accordance with rigorous empirical standards, the following disclosures must accompany all presentations of this work:

1. **Post-Hoc Scale Optimization:** The encoding scale search grid $s \in \{0.0625, 0.125, 0.25, 0.50, 1.00\}$ was introduced post-hoc after observing that default angle encoding ($s = 1.00$) failed completely. While this successfully restored competitive performance for unentangled states, it was guided by knowledge of the default failure mode.
2. **Asymmetric Hyperparameter Tuning Budgets:** Model M3-tuned explored 270 hyperparameter configurations per CV fold ($5 \text{ scales} \times 6\ \gamma \times 9\ C$), compared to 36 to 54 configurations for classical Model M2 ($6\ \gamma \times 6\text{ to }9\ C$). Despite a $5\times$ to $7.5\times$ larger optimization budget, M3-tuned did not surpass M2.
3. **Framework Translation:** Reference literature (Alagiyawanna et al., 2024/2026) originally developed prototypes using TensorFlow and TensorFlow Quantum. We ported the entire methodology to PyTorch and PennyLane (`lightning.qubit`) for medical imaging workflows.
4. **Simulated Data Scarcity:** PneumoniaMNIST comprises 5,856 total images. We artificially enforced small-sample regimes ($N \le 100$) using stratified subsampling to evaluate few-shot behavior while retaining a fixed 624-image test set for uncorrupted statistical power.

---

## 5. Claims You Can and Cannot Make

### Permissible Empirical Claims
* At default encoding ($s = 1.00$), M3 is far below M2 at every sample size $N$.
* With a CV-tuned encoding scale ($s \le 0.25$), M3 improves substantially but stays detectably below M2: deficits of approx $-0.12, -0.08, -0.03, -0.03$ AUC for $N = 10, 20, 50, 100$.
* The only circuit family statistically equivalent to M2 within $\pm 0.02$ AUC ($N \ge 20$) is the 1-layer product state, which is a separable classical trigonometric mapping and trivially classically simulable.
* Entangled variants scored systematically lower in this circuit family and measurement set.
* Adding nearest-neighbor ZZ correlators did not reverse the degradation (tested negative result).
* Classical M2 beats scratch CNN (M1) modestly at $N = 20$ and $N = 50$.
* The ResNet-18 + PCA representation saturates near $0.93$ AUC, leaving little room for any advantage.

### Impermissible Over-Claims
* **DO NOT** claim general quantum kernel inferiority: our findings apply strictly to this circuit family (fixed CNOT ring, angle encoding, 1-body Pauli measurements, no trainable gates).
* **DO NOT** claim that entanglement "scrambles medical pathology": entanglement flattens metric distance spread and reduces single-qubit expectation variance across coordinate space.
* **DO NOT** claim that M3 "ties" or "matches" M2: M3-tuned exhibits a statistically significant deficit surviving Bonferroni correction across all $N$.

---

## 6. Teammate Walkthroughs & Viva Defense Guide

### 1. Daksh Singh (2024BCS0042): Data, Problem Framing & Evaluation
* **Module Ownership:** `data.py`, `evaluate.py`, stratified subsampling, evaluation metrics.
* **Core Defense:** Explain why ROC-AUC is threshold-free and immune to clinical class imbalance (unlike raw accuracy); explain why 15 seeds are required to obtain tight Student's t confidence intervals under small $N$; defend why data scarcity is simulated on PneumoniaMNIST to maintain a large 624-image test set.
* **Key Viva Question:** *"Why is accuracy misleading on clinical data?"*  
  *Answer: PneumoniaMNIST has a 62/38 class imbalance. A naive classifier predicting Pneumonia for every patient achieves 62% accuracy with zero diagnostic sensitivity.*

### 2. Anshuman Biswas (2024BCD0008): Classical Models & Baselines
* **Module Ownership:** `features.py`, `preprocess.py`, `classical.py`, ResNet-18 backbone, PCA, Model M1, Model M2.
* **Core Defense:** Explain why M1 with 105k parameters overfits $N=20$; why ImageNet optical primitives transfer to chest radiographs; why PCA is fitted exclusively on training data; why M2 is the primary scientific control holding data representation and RBF kernel structure identical to M3.
* **Key Viva Question:** *"Why is M2 more important than M1?"*  
  *Answer: M1 is a strawman negative control proving that deep networks cannot be trained from scratch on 20 samples. M2 is the true scientific control: it takes the exact same 8 PCA components into the exact same RBF mathematical formula, isolating the quantum feature map as the sole independent variable.*

### 3. Himmat Sirwee (2024BCS0040): Quantum Circuit & Encoding
* **Module Ownership:** `quantum.py`, PennyLane QNodes, angle encoding, circular CNOT rings, Pauli expectation values.
* **Core Defense:** Explain the mapping of PCA coordinates to rotation angles $\theta \in [0, \pi]$; explain the Bloch sphere representation and why single-qubit states are captured by $\langle X \rangle, \langle Y \rangle, \langle Z \rangle$; explain why simulating 8 qubits ($2^8 = 256$ state amplitudes) is classically trivial on `lightning.qubit`.
* **Key Viva Question:** *"What does the CNOT layer change in the quantum state?"*  
  *Answer: CNOT entangles adjacent qubits, creating non-separable bipartite states. However, because our measurement operators are strictly local 1-body Paulis, tracing out entangled subsystems shrinks the local Bloch vectors toward the origin, collapsing expectation variance.*

### 4. Sawan Kumar (2024BCS0164): Quantum Kernel Theory & SVM
* **Module Ownership:** Projected Quantum Kernel formula, Gram matrix construction, dual convex QP, 2×2 ablations.
* **Core Defense:** Explain fidelity kernels vs. projected quantum kernels (Huang et al., 2021); explain how fidelity kernels suffer exponential concentration in Hilbert space while PQK projects states back to reduced 1-particle density matrices; explain why 1-layer unentangled states reduce to classical trigonometric RBF kernels.
* **Key Viva Question:** *"Does PQK provide an exponential speedup?"*  
  *Answer: No. Kernel evaluation on classical data requires simulating or running the quantum circuit for all pairs ($O(N^2)$). Quantum kernels seek an inductive bias advantage in data representation geometry, not a computational speedup.*

### 5. Aryan Gupta (2024BCS0020): Integration, Prototype & Conclusions
* **Module Ownership:** `app.py`, `run_experiments.py`, `plots.py`, dashboard architecture, conclusion synthesis.
* **Core Defense:** Defend the complete pipeline end-to-end; explain the 5-tab Streamlit dashboard; articulate the bounded conclusions; explain why the project is a successful, rigorous empirical investigation regardless of whether quantum beat classical.
* **Key Viva Question:** *"If the quantum model didn't win, was this project a failure?"*  
  *Answer: No. A null or negative result with 100% seed parity, 960 completed runs, and rigorous Bonferroni validation is a high-value scientific contribution. It identifies phase wrapping and entanglement variance collapse as precise failure modes of angle-encoded PQKs on optical data.*

---

## 7. Reproduction Instructions

### Environment Setup
```powershell
# Clone and enter workspace
cd qkernel-lite

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install pinned dependencies
pip install -r requirements.txt
```

### Run Test Verification Suite (32 Unit Tests)
```powershell
.\venv\Scripts\pytest tests/ -v
```

### Reproduce Full 960-Run Benchmark
```powershell
# Run the complete experiment matrix across seeds 0 to 14
python run_experiments.py

# Run the ZZ correlator ablation experiment
python run_zz_experiment.py

# Compute statistical tables and Bonferroni checks
python analyze_ablation_15seeds.py
```

### Generate Publication Figures (Stage 8)
```powershell
python plots.py
```

### Launch Streamlit Prototype Platform
```powershell
streamlit run app.py
```

---

## 8. Core References

1. **A. M. A. S. D. Alagiyawanna, Asoka Karunananda, A. Mahasinghe, Thushari Silva**, *"Enhancing Small Dataset Classification Using Projected Quantum Kernels with Convolutional Neural Networks,"* 8th SLAAI International Conference on Artificial Intelligence (SLAAI-ICAI 2024); arXiv:2601.03375 [cs.LG] (2024/2026).
2. **H.-Y. Huang, M. Broughton, M. Mohseni, et al.**, *"Power of data in quantum machine learning,"* Nature Communications 12, 2631 (2021).
3. **J. Yang et al.**, *"MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification,"* Scientific Data 10, 41 (2023).
