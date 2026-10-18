# Module 5: The Scientific Control, Quantum Comparisons, & Viva Defense Masterclass

**Author:** Dirac’s Army (Lead: Anshuman Biswas)  
**Target Mastery:** Classical Models & Baselines (Primary Viva Defense & Project Integration)  
**Prerequisites Assumed:** Modules 1 through 4 (CNNs, ResNet-18, PCA, RBF-SVM).

---

## 1. The Big Picture: How Your Module Powers the Entire Project

As the lead builder and owner of the classical pipeline, your components form the **bedrock of the entire QKernel-Lite system**:

```
[ PneumoniaMNIST Image (224x224) ]
               │
               ▼
┌─────────────────────────────────────────────────────────┐
│  ANSHUMAN: Module 2 (Frozen ResNet-18 Feature Extractor) │
│  Transforms 50,176 pixels ──► 512 high-level features   │
└─────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────┐
│  ANSHUMAN: Module 3 (Principal Component Analysis - PCA) │
│  Compresses 512 features ──► n = 4, 6, 8 components     │
└─────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────┐
│  ANSHUMAN: Angle Rescaling                              │
│  Scales PCA numbers to rotation angles θ ∈ [0, π]       │
└─────────────────────────────────────────────────────────┘
               │
       ┌───────┴────────────────────────────────┐
       ▼                                        ▼
┌───────────────────────────────┐  ┌────────────────────────────────────┐
│  ANSHUMAN: Model M2           │  │  HIMMAT & SAWAN: Model M3          │
│  (Classical Primary Control)  │  │  (Projected Quantum Kernel - PQK)   │
│                               │  │                                    │
│  Classical RBF Formula:       │  │  PennyLane Quantum Circuit:        │
│  k(x, x') = exp(-γ ||x-x'||²) │  │  Encode θ into n qubits via RY/RX  │
│                               │  │  Measure ⟨X⟩, ⟨Y⟩, ⟨Z⟩ on qubits   │
│                               │  │  PQK Formula:                      │
│                               │  │  k(x, x') = exp(-γ Σ (⟨P⟩-⟨P'⟩)²)  │
└──────────────┬────────────────┘  └─────────────────┬──────────────────┘
               │                                     │
               └──────────────────┬──────────────────┘
                                  ▼
               ┌─────────────────────────────────────┐
               │  Shared Scikit-Learn SVM Optimizer  │
               │  SVC(kernel="precomputed")          │
               │  Evaluates ROC-AUC, Balanced Acc, F1 │
               └─────────────────────────────────────┘
```

---

## 2. The Striking Mathematical Symmetry Between M2 and M3

Notice the extraordinary symmetry between your classical model (M2) and the quantum model (M3):

### Classical RBF Kernel (Model M2):
$$k_{\text{RBF}}(\mathbf{x}, \mathbf{x}') = \exp\left( -\gamma \sum_{k=1}^n (x_k - x'_k)^2 \right)$$
Where $x_k$ and $x'_k$ are the classical PCA coordinates of the two images.

### Projected Quantum Kernel (Model M3 - Huang et al., 2021):
$$k_{\text{PQK}}(\mathbf{x}, \mathbf{x}') = \exp\left( -\gamma \sum_{i=1}^n \sum_{P \in \{X, Y, Z\}} \Big( \langle P_i \rangle(\mathbf{x}) - \langle P_i \rangle(\mathbf{x}') \Big)^2 \right)$$
Where $\langle P_i \rangle(\mathbf{x})$ is the quantum mechanical expectation value of Pauli operator $P$ measured on qubit $i$ after encoding image $\mathbf{x}$!

### The Profound Realization:
**A Projected Quantum Kernel is mathematically an RBF Gaussian kernel evaluated on the physical expectation values of the quantum circuit!**

- In M2, the feature space is $\mathbb{R}^n$ (the $n$ classical PCA numbers).
- In M3, the feature space is $\mathbb{R}^{3n}$ (the $3n$ quantum measurement numbers $\langle X \rangle, \langle Y \rangle, \langle Z \rangle$ across $n$ qubits).

This proves that **Model M2 is the single fairest, most rigorous baseline possible**. It holds the data representation, dimensionality, kernel family (Gaussian/exponential), and downstream SVM optimizer completely constant, isolating the quantum circuit as the **sole independent variable**!

---

## 3. The 3 Master Viva Questions: Scripted Answers

These three questions are guaranteed to be asked during your project review. Here is your structured, unshakeable defense:

---

### Viva Question 1: "Why not train the CNN with the quantum circuit end-to-end using backpropagation?"

#### Common Misconception:
*Students often say: "Because PennyLane doesn't support PyTorch" (Wrong! PennyLane integrates seamlessly with PyTorch).*

#### The True Scientific & Mathematical Defense:
1. **Severe Overfitting on Extreme Scarcity ($N \le 100$):**
   An end-to-end hybrid quantum-classical neural network requires backpropagating gradients from the quantum circuit back into all 11.7 million weights of ResNet-18. When you have only 20 training images, attempting to optimize 11 million weights causes instantaneous memorization and catastrophic overfitting.
2. **Barren Plateaus in Hybrid Gradients:**
   Computing gradients across hybrid quantum-classical boundaries introduces barren plateaus—an exponential flattening of the quantum gradient landscape where $\text{Var}[\partial \mathcal{L} / \partial \theta] \in \mathcal{O}(2^{-n})$.
3. **Training Time and Simulation Overhead:**
   In end-to-end training, the quantum simulator must be executed on every mini-batch, for every epoch, and for every parameter update using the Parameter-Shift Rule ($2 \times \text{number of gates}$ circuit runs per gradient step). With 100 epochs, this takes hours.
   In our kernel approach, **the quantum circuit is executed exactly once per image ($O(N)$)**, cached, and the SVM convex quadratic program solves globally in milliseconds!

---

### Viva Question 2: "What information is lost by PCA when you reduce 512 dimensions to 8?"

#### The Mathematical Defense:
1. **Trace Decomposition:**
   The total sample variance across all 512 deep ResNet features is $\text{Tr}(\mathbf{\Sigma}) = \sum_{k=1}^{512} \lambda_k$.
   Our 8 principal components retain an **Explained Variance Ratio (EVR)**:
   $$\text{EVR}_8 = \frac{\sum_{k=1}^8 \lambda_k}{\sum_{k=1}^{512} \lambda_k}$$
2. **The Discarded Nullspace:**
   The exact discarded information is the orthogonal residual projection residing in the 504-dimensional subspace:
   $$\mathbf{x}_{\text{lost}} = (\mathbf{I} - \mathbf{V}_8 \mathbf{V}_8^T) \tilde{\mathbf{x}}$$
   The amount of variance lost is precisely $\sum_{k=9}^{512} \lambda_k$.

#### The Clinical / Radiological Defense:
- **What is Preserved:** The top 8 eigenvectors align with the dominant, macroscopic geometric structures across the patient cohort: overall thoracic cavity volume, bilateral lung field expansion, and large-scale hazy consolidations (pneumonia fluid density).
- **What is Discarded:** The remaining 504 dimensions capture fine interstitial micro-textures, localized vascular markings, and scanner exposure calibration noise.
- **The Trade-Off Justification:** While discarding the trailing dimensions acts as a beneficial noise filter, it also discards subtle, localized pathology. We explicitly accept this trade-off because simulating more than 8 qubits on classical hardware causes exponential slowdown ($2^n$ statevector scaling).

---

### Viva Question 3: "Why is Model M2 more important than Model M1?"

#### The Strategic Scientific Defense:
1. **M1 is an Asymmetric Strawman (Negative Control):**
   Model M1 (a CNN trained from scratch) is tested in a regime where it is mathematically guaranteed to fail ($10^5$ parameters vs. 20 images). Beating M1 proves nothing about quantum computing; it only proves that deep networks need data. Any high school model with transfer learning beats M1.
2. **M2 is the True Scientific Control:**
   Model M2 (Frozen ResNet-18 + PCA + Classical RBF-SVM) uses the **exact same pretrained visual knowledge**, the **exact same PCA subspace**, and the **exact same SVM margin optimizer** as our quantum model M3.
3. **Isolating the Quantum Advantage:**
   If M3 outperforms M1, the credit belongs to Transfer Learning, not Quantum Computing.
   **Only if M3 outperforms M2 can we legitimately claim that the quantum Hilbert space embedding provided real diagnostic value!**
   Therefore, M2 is the only benchmark that possesses scientific validity.

---

## 4. The "Win, Tie, or Loss" Philosophy: Why Dirac's Army Cannot Lose

In college project presentations, students often panic if their proposed quantum model doesn't beat the classical baseline by 10%.

As lead builder, here is your guiding philosophy (rooted in **Huang et al., Nature Communications 2021**):

| Outcome | Physical Interpretation | How to Present It in Viva |
| :--- | :--- | :--- |
| **M3 > M2 (Quantum Wins)** | The quantum feature map (superposition + entanglement) successfully unrolls non-linear correlations in the PCA features that the classical Gaussian kernel could not capture. | A triumphant proof-of-concept for quantum kernel methods in data-scarce medical imaging. |
| **M3 $\approx$ M2 (Quantum Ties)** | Expected by Huang et al.: classical Gaussian RBF kernels are universal approximators on natural image features; the quantum circuit matches classical performance on low qubit counts ($n \le 8$). | A defensible confirmation of theoretical predictions: quantum kernels are viable and robust in medical domains without suffering performance degradation. |
| **M3 < M2 (Quantum Loses)** | Projecting continuous radiological features through discrete quantum gates with entanglement introduces artificial Hilbert space distortions or phase sensitivity that classical smooth distances avoid. | A rigorous, honest empirical finding highlighting the limitations of NISQ angle embeddings on classical image manifolds. |

> **Key Takeaway:** In real science, a carefully controlled, honest comparison with a clear explanation of *why* the result occurred is worth an A+; an exaggerated or uncontrolled claim that falls apart under faculty questioning results in failure.

---

## 5. Master Summary Table of Anshuman's Ownership

| Concept | Your Core Ownership Responsibility | Primary Defense Argument |
| :--- | :--- | :--- |
| **M1 (Scratch CNN)** | Negative control baseline showing scratch deep learning. | Parameter under-determination ($P \gg N$). While initial theory predicted collapse to AUC ~ 0.5, our empirical runs showed M1 reaches ~0.86 AUC at N=10 and N=20 by latching onto coarse macroscopic lung opacities. |
| **ResNet-18** | Universal inductive representation extractor. | Optical primitives (edges, textures, gradients) transfer across domains; freezing eliminates $11.7\text{M}$ parameter estimation risk. |
| **PCA** | Dimensionality bridge from classical deep features to qubits. | Diagonalizes covariance matrix ($\mathbf{\Sigma} \mathbf{v} = \lambda \mathbf{v}$); retains maximum variance; fitted on training data only to prevent data leakage. |
| **RBF-SVM (M2)** | Primary scientific control baseline. | Convex QP optimization; margin maximization; Gaussian similarity $\exp(-\gamma d^2)$; acts as identical counterpart to PQK. |

---

## 6. The Rerun Resolution: Depth vs. Entanglement Disentangled, Bandwidth Parity, & Empirical Diagnostics

### 6.1 The γ Heuristic vs. Circuit Frequency Deficit
In our initial experimental diagnostic, recalibrating the γ heuristic via the median distance resolved numerical identity collapse on extreme outliers (e.g., N=20 seed 0). However, the full multi-seed rerun revealed that γ scaling alone accounted for only modest shifts (~0.02 to 0.04 AUC) across the broader distribution. The remaining deficit of Model M3 at small N is a genuine physical property of the circuit architecture:
1. **Circuit Depth & Data Re-uploading:** Repeating layers doubles the angular frequency components $e^{i n \theta}$, causing high-frequency oscillations that distort smooth decision boundaries.
2. **Entangling Scrambling:** CNOT rings scramble single-qubit expectation values, reducing feature variance and degrading distance ranks.

### 6.2 Disentangling Depth and Entanglement: The Full 2×2 Grid
To systematically separate the effects of circuit depth from quantum entanglement, we executed a complete 2×2 ablation grid (1 Layer vs. 2 Layers × Unentangled vs. Entangled) across 4, 6, and 8 qubits with matching classical M2 baselines across all N ∈ {10, 20, 50, 100} and 5 seeds:

| Qubits | Configuration | N=10 | N=20 | N=50 | N=100 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **8 (M2)** | **Classical M2 (8-comp)** | **0.9152 ± 0.027** | **0.9277 ± 0.012** | **0.9310 ± 0.005** | **0.9294 ± 0.012** |
| **8 (Q)** | **1L Unentangled** | **0.8744 ± 0.021** | **0.9243 ± 0.008** | **0.9308 ± 0.006** | **0.9271 ± 0.020** |
| 8 (Q) | 1L Entangled | 0.6163 ± 0.109 | 0.6565 ± 0.093 | 0.7058 ± 0.113 | 0.7177 ± 0.034 |
| 8 (Q) | 2L Unentangled | 0.6662 ± 0.124 | 0.7721 ± 0.146 | 0.8784 ± 0.020 | 0.8892 ± 0.022 |
| 8 (Q) | 2L Entangled (fixed s=1.0) | 0.5320 ± 0.077 | 0.5610 ± 0.119 | 0.7274 ± 0.037 | 0.8320 ± 0.036 |
| 8 (Q) | 2L Entangled (CV-tuned s) | 0.7668 ± 0.109 | 0.8188 ± 0.098 | 0.9249 ± 0.017 | 0.8958 ± 0.024 |

#### Key Empirical Insights:
1. **The 1-Layer Unentangled Gate:** At 8 qubits, the 1-layer unentangled circuit shows **no detectable difference** from Classical M2 at N ≥ 20 (0.9243 vs. 0.9277 at N=20; 0.9308 vs. 0.9310 at N=50). At 1 layer with no CNOTs, the circuit applies a smooth, monotonic trigonometric warp $x_i \mapsto (\sin\theta_i\cos\theta_i, \sin^2\theta_i, \cos\theta_i)$ that preserves pairwise metric ranks ($\rho = 0.8175$).
2. **The Cost of Depth:** Moving from 1 layer to 2 layers in the unentangled circuit drops AUC by ~0.21 at N=10 and ~0.15 at N=20. Data re-uploading doubles the frequency harmonics, wrapping angles modulo $2\pi$ and collapsing Spearman rank correlation to $\rho \approx 0.2594$.
3. **The Cost of Entanglement:** Adding CNOT entangling gates to the 1-layer circuit drops AUC from 0.8744 to 0.6163 at N=10 and from 0.9243 to 0.6565 at N=20. Entanglement scrambles single-qubit expectation values, reducing feature variance and collapsing rank correlation to $\rho \approx 0.2556$.

### 6.3 Bandwidth Parity: Encoding Scale Tuning ($s \in \{0.25, 0.50, 1.00\}$)
While classical M2 had its RBF bandwidth γ tuned across a wide multiplier grid, standard M3 used fixed angular encoding $\theta_i = \pi \cdot \tilde{x}_i \in [0, \pi]$. In quantum kernel literature, input encoding bandwidth directly controls frequency spectrum width.

By introducing an encoding scale factor $s \in \{0.25, 0.50, 1.00\}$ tuned via 3-fold CV alongside γ and C:
- Compressing angles by $s=0.25$ constrains phase rotations to $[0, \pi/4]$, suppressing spurious high-frequency harmonics caused by 2-layer re-uploading.
- **M3 Performance Rescued:** At N=50, M3 AUC surges from **0.7274 (fixed s=1.0)** to **0.9249 (tuned s)**, closely matching M2's 0.9310 (paired difference $-0.0061 \pm 0.0263$, spanning zero).
- At N=100, M3 AUC rises from **0.8320 (fixed s=1.0)** to **0.8958 (tuned s)**.

### 6.4 The 3 Physical Diagnostics
1. **Total Feature Variance (Hilbert Space Concentration):**
   - Classical PCA: $\text{Var} \approx 0.69 - 0.73$
   - Quantum Unentangled (1L): $\text{Var} \approx 0.29$ (independent of qubit count)
   - Quantum Entangled (2L): drops to $\approx 0.015 - 0.025$ — a ~30× to 50× reduction in variance. In exact statevector simulation, γ rescaling compensates for this shrinkage; however, on physical NISQ hardware with finite measurement shots ($M \le 10^4$), shot noise $\mathcal{O}(1/\sqrt{M})$ would dominate these tiny expectation signals.
2. **Relative Distance Spread ($\text{std} / \text{median}$):**
   - Classical PCA spread falls monotonically from $0.3185$ (4D) $\to 0.2418$ (6D) $\to 0.1890$ (8D). This decline is purely a **classical Euclidean dimensional concentration effect** ($1/\sqrt{d}$), not a quantum property.
   - Quantum Entangled spread remains stable around $0.24 - 0.41$, verifying that median-heuristic γ scaling successfully prevents kernel matrix collapse.
3. **Metric Preservation (Spearman Rank Correlation $\rho$ vs. Angle Features):**
   - Classical PCA: $\rho = 1.0000$ (reference)
   - **1L Unentangled:** $\rho = 0.9067$ (4Q), $0.8650$ (6Q), $0.8175$ (8Q) — strongly metric-preserving, confirming the gate behavior!
   - **1L Entangled:** $\rho = 0.3801$ (4Q), $0.2945$ (6Q), $0.2556$ (8Q) — rank preservation collapses immediately upon applying CNOTs.
   - **2L Unentangled:** $\rho = 0.2986$ (4Q), $0.2757$ (6Q), $0.2594$ (8Q) — data re-uploading wraps angles and scrambles rank preservation even without entanglement.
   - **2L Entangled:** $\rho = 0.1629$ (4Q), $0.0885$ (6Q), $0.3726$ (8Q).

### 6.5 Statistical Rigor: Per-N Hypothesis Testing & Edge Hits
1. **No Pooled All-N Rows:** All statistical tests are conducted strictly per-N (n=5 independent seed pairs per N, Student's $t$ with $df=4$, $t=2.776$). Pooled tests violate independence because runs share random seeds across N.
2. **Paired M2 − M1 Tests:**
   - N=10: $+0.0498 \pm 0.0439$, $p = 0.0345$ (M2 statistically significantly outperforms M1).
   - N=20: $+0.0638 \pm 0.0576$, $p = 0.0371$ (M2 statistically significantly outperforms M1).
   - N=50: $+0.0313 \pm 0.0239$, $p = 0.0218$ (M2 statistically significantly outperforms M1).
   - N=100: $+0.0089 \pm 0.0162$, $p = 0.2017$ (No statistically significant difference; CIs overlap zero as M1 converges on 100 images).
3. **Deterministic CV Tie-Breaking:**
   - To eliminate false edge hits at small N caused by discrete fold ties, the CV grid search employs a deterministic tie-breaker preferring hyperparameters closest to 1.0: $\text{score} = |\ln(\gamma_{\text{mult}})| + |\ln(C)| + |\ln(s)|$.
   - Result: Edge hits at N=10 dropped to 0/5 for both M2 and M3.

