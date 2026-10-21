# 🏛️ QKernel-Lite: Overall Project Viva Defense Q&A
**Course:** IEG313: Quantum Computing for Engineers · IIIT Kottayam  
**Team:** Dirac's Army (Group 5)  
**Evaluators:** Head of Department (Computer Science & Engineering) & Head of Department (Mathematics & Computing)  
**Focus:** Comprehensive viva defense guide spanning general introductory questions, literature replication, architectural methodology, and deep mathematical defense.

---

## 📌 Part 1: General & Foundational Project Questions

---

### Q1: "In simple terms, what is the core novelty and original contribution of your project?"
**The Team's Collective Defense:**  
*"Our project provides the first rigorous, controlled scientific refutation of ungrounded quantum advantage claims in medical image classification.

While prior papers claimed dramatic 'quantum advantage' by comparing quantum models against artificially weak classical baselines, our novelty lies in three core contributions:

1. **A Matched Scientific Control (Model M2):** We built a 3-model benchmark that isolates the quantum circuit as the single independent variable. Model M2 and Quantum Model M3 receive identical ImageNet ResNet-18 features, identical PCA reduction, and solve the identical dual SVM quadratic program.
2. **Analytical Derivation of Quantum Failure Modes:** We discovered and mathematically derived two physical failure mechanisms:
   - The **Phase Wrapping Turnaround Bug**, where default scaling ($s = 1.00$) causes healthy and severely diseased lungs to map to the exact same point in Hilbert space ($U(\pi)^2|0\rangle = |0\rangle$).
   - The **Small-$s$ Taylor Duality**, proving that when quantum models perform well at small scale factors ($s \le 0.125$), they mathematically reduce to classical polynomial feature approximations.
3. **Reproducible Engineering Platform:** We published a transparent 960-run experimental ledger across 15 seeds with Bonferroni-corrected hypothesis tests, backed by an interactive sub-5ms research prototype deployed live on Oracle Cloud."*

---

### Q2: "Which research paper did you refer to as your primary reference, and were you trying to replicate their findings?"
**The Team's Collective Defense:**  
*"Our investigation began as a replication and extension of a specific recent study:

- **Primary Reference Paper:**  
  **Alagiyawanna et al. (SLAAI-ICAI 2024 / arXiv:2601.03375)**: *'Enhancing Small Dataset Classification Using Projected Quantum Kernels with Convolutional Neural Networks.'*
- **Foundational Theoretical Reference:**  
  **Huang et al. (Nature Communications 2021)**: *'Power of data in quantum machine learning'* (introducing Projected Quantum Kernels to resolve fidelity barren plateaus).
- **Clinical Dataset Reference:**  
  **Kermany et al. (Cell 2018)** / **Yang et al. (Nature Scientific Data 2023, MedMNIST v2)** for `PneumoniaMNIST`.

**Were we trying to replicate their findings?**  
Yes, initially we sought to verify their reported finding that Projected Quantum Kernels dramatically boost accuracy on small datasets ($N \le 100$). However, when examining their methodology, we identified a critical flaw known as the **strawman baseline fallacy**:
- Alagiyawanna et al. compared their quantum kernel against an unregularized, 100,000-parameter CNN trained from scratch on tiny datasets, which collapsed to near-random accuracy (~12% on multi-class benchmarks).
- They omitted a matched classical transfer learning control.

When we replicated their angle-encoded circuit in PyTorch and PennyLane (`lightning.qubit`) and introduced a fair classical control (Model M2 with identical pretrained ResNet-18 features and classical RBF kernel), **their claimed quantum advantage completely vanished**. Quantum M3 lagged behind classical M2 across all sample sizes ($p < 0.001$).

Thus, our project transitioned from a simple replication into a **definitive confirmatory benchmark and root-cause diagnostic study**."*

---

### Q3: "Can you walk us through the high-level workflow of your project from an X-ray image to the final diagnostic prediction?"
**The Team's Collective Defense:**  
*"Our pipeline processes a patient radiograph through seven deterministic stages:

1. **Input Radiograph:** A standardized $28 \times 28$ grayscale chest X-ray from the `PneumoniaMNIST` benchmark.
2. **Deep Feature Extraction:** We pass the image through a frozen pretrained ResNet-18 backbone (ImageNet weights, locked in evaluation mode). This extracts 512 high-level visual features.
3. **Dimensionality Reduction (PCA):** Because quantum statevector simulation scales exponentially ($2^n$), we apply Principal Component Analysis (PCA) fitted strictly on the $N$ training samples, compressing 512 dimensions down to 8 orthogonal principal components ($77\%$ explained variance).
4. **Coordinate Min-Max Normalization:** The 8 continuous PCA components are linearly mapped into bounded angular coordinates: $\theta_i \in [0, \pi]$.
5. **Quantum Angle Encoding:** An 8-qubit quantum register initialized in $|0\rangle^{\otimes 8}$ undergoes single-qubit rotations $R_Y(s\theta_i)$ followed by $R_Z(s\theta_i)$, where $s$ is an encoding scale parameter, interleaved with optional circular CNOT entangling gates.
6. **Local Pauli Expectation Measurement:** We measure the expectation values of local Pauli operators $\langle X_i \rangle, \langle Y_i \rangle, \langle Z_i \rangle$ on each of the 8 qubits, producing a 24-dimensional classical vector $\mathbf{E}(\mathbf{x}) \in \mathbb{R}^{24}$.
7. **Dual SVM Classification:** A classical Support Vector Machine evaluates a Gaussian RBF kernel on these 24 expectation values to output the final diagnostic score: positive for Pneumonia, negative for Normal."*

---

### Q4: "Why did you choose pediatric pneumonia classification and the PneumoniaMNIST dataset specifically?"
**The Team's Collective Defense:**  
*"We selected pediatric pneumonia on PneumoniaMNIST for two compelling reasons:

1. **High-Stakes Clinical Reality:** Pediatric pneumonia remains one of the leading infectious causes of childhood mortality worldwide. In rural emergency clinics and developing nations, expert pediatric radiologists are scarce, and clinical datasets for emerging epidemic strains are inherently tiny ($N = 10, 20$). If quantum computing could genuinely improve few-shot diagnostic accuracy, this is where it would have life-saving impact.
2. **Benchmark Standardization & Scientific Rigor:** Many quantum machine learning papers evaluate on toy synthetic datasets (e.g. concentric circles or artificial moons) where quantum circuits can easily be engineered to win. PneumoniaMNIST (Kermany et al., Cell 2018; Yang et al., Nature Scientific Data 2023) is a recognized, peer-reviewed clinical benchmark with an untouched 624-patient test set. It provides an authentic real-world test for continuous optical radiographic data."*

---

### Q5: "If your quantum model achieved lower accuracy than classical models, why is this still a valid and high-value engineering project?"
**The Team's Collective Defense:**  
*"In experimental science and computer engineering, **a reproducible null result that refutes an unverified claim is a successful investigation, not a failed project**.

Consider three realities:
1. **Preventing Wasted Compute & Capital:** In the quantum software industry, millions of dollars are currently spent attempting to deploy quantum kernels for computer vision. By proving mathematically that angle encoding on continuous optical images suffers from phase wrapping and entanglement variance collapse, we identify exactly where the boundaries of quantum utility lie.
2. **The Scientific Method:** If every student project had to confirm a hyped hypothesis, science would suffer from severe publication bias. The highest mark of engineering maturity is reporting exactly what was measured under rigorous statistical control ($p < 0.001$, Bonferroni-corrected across 15 seeds).
3. **Rigorous Diagnostic Explanations:** We did not simply stop at 'quantum lost.' We derived the exact closed form of the circuit, explained the 34-fold variance collapse under entanglement, proved the Small-$s$ Taylor Duality, and refuted the ZZ-correlator hypothesis."*

---

### Q6: "What are the major limitations of your study, and what would you do next if you had 6 more months or access to physical QPUs?"
**The Team's Collective Defense:**  
*"We maintain full intellectual transparency regarding our study's limitations:

1. **Current Limitations:**
   - **Classical Simulation Scale:** We simulated an 8-qubit register using PennyLane's `lightning.qubit` statevector engine. Simulating beyond 16 qubits on standard workstations incurs exponential RAM overhead.
   - **Downsampled Resolution:** We utilized $28 \times 28$ images. While sufficient for global consolidations, subtle focal micro-infiltrates in full $1024 \times 1024$ hospital DICOM radiographs require localized patch processing.
   - **Circuit Family Scope:** We investigated angle-encoded product and circular CNOT circuits as specified by Alagiyawanna et al. We did not evaluate Hamiltonian evolution or IQP kernels.

2. **Next Steps (6-Month Roadmap):**
   - **Natively Quantum Data:** Pivot the kernel machine from classical optical radiographs to naturally quantum data structures—such as molecular electronic Hamiltonians or quantum sensor magnetometry—where Hilbert space distance is the native physical metric.
   - **Physical NISQ Deployment:** Run the 1-layer unentangled product state on physical IBM Quantum superconducting hardware to quantify the impact of physical thermal relaxation ($T_1$) and dephasing ($T_2$) on kernel stability.
   - **Patch-Based Quantum Ensembles:** Extract localized high-resolution visual tokens via Vision Transformers (ViT) and feed local patch tokens to distinct quantum registers."*

---

## 🏛️ Part 2: High-Level Architectural & Engineering Questions

---

### Q7 (CSE HOD): "Does evaluating a Projected Quantum Kernel offer an asymptotic computational speedup over classical kernels? If not, what was the theoretical motivation for using quantum kernels on classical data?"
**The Team's Collective Defense:**  
*"No, quantum kernels offer **zero computational complexity speedup** when processing classical data.

To construct a Gram matrix for an SVM on $N$ classical data points, one must evaluate the pairwise kernel distance for all pairs:

$$
\text{Computational Cost} = \mathcal{O}(N^2 \cdot T_{\text{kernel}})
$$

Regardless of whether $T_{\text{kernel}}$ is computed via classical dot products or quantum circuit evaluations, the quadratic $\mathcal{O}(N^2)$ complexity of training a kernel machine remains unchanged.

**The Theoretical Motivation was Inductive Bias, not Speed:**  
In learning theory, when data is extremely scarce ($N \le 20$), a model cannot rely on empirical data alone; its success depends entirely on its **inductive bias**—the geometric prior it imposes on the hypothesis space. The hypothesis in literature (Huang et al., 2021; Kübler et al., 2021) was that mapping classical coordinates into a non-linear $2^n$-dimensional quantum Hilbert space might construct a geometric representation where complex pathology becomes linearly separable with a wider margin than any classical kernel could achieve. Our study tested and disproved this inductive bias hypothesis for angle-encoded medical radiographs."*

---

### Q8 (CSE HOD): "A critic could object: 'Your classical baseline (M2) benefited from ResNet-18, which trained on 14 million ImageNet images. Your quantum circuit was untrained and randomly structured. Isn't that an unfair comparison?' How do you answer this?"
**The Team's Collective Defense:**  
*"That objection actually highlights why our scientific control is so airtight:

1. **Both Models Received the Exact Same Features:** We did NOT feed raw pixels to M2 and ResNet features to M3. The classical baseline and the quantum pipeline were given the exact same 8 PCA components extracted from frozen ResNet-18.
2. **Clinical Reality:** In practical healthcare, an emergency clinic facing an epidemic does not operate in a vacuum—it has access to pretrained classical backbones. Any proposed quantum technology must prove that it adds value *on top of* existing classical tools.
3. **$5\times$ Larger Tuning Budget for Quantum:** Quantum Model M3 was given a **$5\times$ larger hyperparameter tuning search budget** (270 grid combinations per fold vs. 36–54 for M2), specifically tuning the angle scale $s$ to rescue the quantum model. Even with ResNet features provided to both, and with the search budget heavily biased in quantum’s favor, quantum coordinate encoding actively degraded the classical representation."*

---

### Q9 (CSE HOD): "How does your Streamlit prototype handle model disagreement, and how did you achieve sub-5ms inference latency?"
**The Team's Collective Defense:**  
*"We addressed both latency and clinical decision support through software architecture:

1. **Sub-5ms Inference via Pre-Compiled Cache:**  
   Running live PyTorch forward passes and PennyLane quantum statevector simulations inside an Oracle Cloud ARM VM produces 2 to 3 seconds of latency per user interaction. To eliminate this, we pre-compiled an offline database (`prototype_db.npz`, 28.6 MB) containing all 624 patient test tensors, ResNet embeddings, pre-computed quantum expectation vectors, and trained dual SVM coefficients. In live inference, the app computes a vectorized NumPy matrix-vector dot product:

$$
f(\mathbf{x}) = \sum_{i \in \text{SVs}} \alpha_i y_i \exp\left(-\gamma \|\mathbf{u}_i - \mathbf{u}_{\text{test}}\|^2\right) + b
$$

   This executes in **3.8 milliseconds** and consumes only 24.5 MB of RAM.

2. **Tri-Model Diagnostic Consensus Engine:**  
   For every scan, the interface evaluates all three models simultaneously:
   - **Complete Consensus (3/3):** M1, M2, and M3 agree on the diagnosis.
   - **Kernel Consensus vs. CNN Divergence (2/3):** Classical RBF and Quantum PQK agree, while Scratch CNN diverges—demonstrating that transfer learning overrides scratch CNN noise.
   - **Model Divergence:** The interface triggers an alert banner warning clinicians that the radiograph falls within an uncertain margin boundary."*

---

### Q10: "In your live demo and report, you enforce a strict Clinical Research Disclaimer. From a bioethics and medical device regulatory standpoint (e.g. FDA / CDSCO), why is this mandatory?"
**The Team's Collective Defense:**  
*"Pediatric pulmonology involves critical, life-or-death diagnoses: untreated bacterial pneumonia can cause respiratory failure within hours, while misdiagnosing a normal lung can lead to unnecessary antibiotic exposure or invasive interventions.

Under FDA Software as a Medical Device (SaMD) and CDSCO digital health guidelines, any diagnostic system deployed in a clinical setting requires rigorous prospective clinical trials, multisite validation across different X-ray machine manufacturers, and certified ISO 13485 quality management.

Because our benchmark uses simulated few-shot regimes ($N=10$) on downsampled $28 \times 28$ images to explore quantum computing theory, deploying or presenting this tool as a clinical diagnostic instrument would be unethical and legally impermissible. Our platform is strictly an engineering benchmark to evaluate machine learning representations."*

---

## 🔬 Part 3: Deep Theoretical & Mathematical Defense

---

### Q11 (Math & Computing HOD): "Why did you choose angle encoding with circular CNOT gates? Could an alternative quantum feature map—such as IQP (Instantaneous Quantum Polynomial) or Hamiltonian evolution—have succeeded?"
**The Team's Collective Defense:**  
*"We benchmarked the exact circuit family proposed in the foundational study by Alagiyawanna et al. (SLAAI-ICAI 2024 / arXiv:2601.03375), which utilized $R_Y-R_Z$ angle encoding with entangling chains.

However, from a mathematical perspective, an IQP or Hamiltonian feature map would face the same fundamental metric breakdown:

1. **Continuous Optical Data vs. Periodic Phases:** Clinical opacity in an X-ray is a continuous, monotonic physical property (higher tissue density = more optical attenuation = brighter pixels). Quantum gates rely on periodic trigonometric rotations ($e^{i \theta H}$). Any periodic mapping inherently risks folding widely separated clinical extremes together whenever features scale beyond the first monotonic quadrant.
2. **Bipartite Entanglement Scrambling:** In both CNOT circuits and Hamiltonian evolution ($e^{-i t \sum Z_i Z_j}$), generating multi-qubit entanglement spreads local information into high-order tensor correlations. Unless the downstream measurement operators measure the global $n$-body observable (which triggers exponential concentration), projecting onto local 1-qubit or 2-qubit density matrices causes the local Bloch vectors to contract toward zero, destroying margin variance."*

---

### Q12 (CSE HOD): "Why did you use a fixed Quantum Kernel with a classical SVM instead of training a Parameterized Variational Quantum Circuit (VQC / QNN) directly with gradient descent?"
**The Team's Collective Defense:**  
*"Three decisive theoretical and practical reasons:

1. **Convex Optimization vs. Non-Convex Landscapes:** Training a VQC using gradient descent via the parameter-shift rule optimizes a non-convex loss surface plagued by local minima, saddle points, and barren plateaus ($\text{Var}[\partial_\theta \mathcal{L}] \sim \mathcal{O}(2^{-n})$). In contrast, an SVM with a positive semi-definite Projected Quantum Kernel solves a strictly convex dual quadratic program ($\min_{\boldsymbol{\alpha}} \frac{1}{2}\boldsymbol{\alpha}^T \mathbf{Q} \boldsymbol{\alpha} - \mathbf{1}^T \boldsymbol{\alpha}$). Convexity guarantees a mathematically unique, global optimum.
2. **Sample Efficiency:** On tiny datasets ($N = 10, 20$), optimizing dozens of variational circuit angles $\boldsymbol{\theta}$ using gradient steps leads to severe empirical overfitting, as the circuit parameters quickly memorize individual training samples.
3. **Representer Theorem:** By the classical Representer Theorem, the optimal decision boundary for any kernel machine lies directly in the span of the training data: $f(\mathbf{x}) = \sum_{i=1}^N \alpha_i y_i k(\mathbf{x}_i, \mathbf{x}) + b$. This makes kernel methods fundamentally superior to VQCs for scarce sample regimes."*

---

### Q13 (Math & Computing HOD): "Entanglement is the primary quantum resource that enables Bell inequality violations and quantum computational supremacy. Why did entanglement actively destroy performance in your benchmark?"
**The Team's Collective Defense:**  
*"In quantum computing, entanglement is a powerful computational resource for algorithms that process global quantum states (such as Shor's factoring algorithm or Grover search). However, in **Projected Quantum Machine Learning**, entanglement acts as a destructive entropy source.

Mathematically, let $|\psi(\mathbf{x})\rangle \in \mathcal{H}_1 \otimes \dots \otimes \mathcal{H}_n$ be the global entangled state. To evaluate local Pauli observables, we compute:

$$
\langle \sigma_z^{(i)} \rangle = \text{Tr}\left(\sigma_z^{(i)} \rho_i(\mathbf{x})\right), \quad \text{where } \rho_i(\mathbf{x}) = \text{Tr}_{\bar{i}}\left(|\psi(\mathbf{x})\rangle\langle\psi(\mathbf{x})|\right)
$$

By the subadditivity of von Neumann entropy, tracing out an entangled subsystem increases the entropy of the reduced state: $S(\rho_i) > 0$. The reduced density matrix $\rho_i$ transitions from a pure state on the Bloch sphere surface to a mixed state inside the sphere:

$$
\|\mathbf{r}_i\|^2 = 2\text{Tr}(\rho_i^2) - 1 < 1
$$

As entanglement increases, the local Bloch vector contracts toward the origin $(0, 0, 0)$.  
In our experiments, this caused a **34-fold collapse in expectation variance** ($0.290 \to 0.0085$). In clinical terms, diverse lung pathologies are projected to the same origin, erasing margin boundaries."*

---

### Q14 (Math & Computing HOD): "You showed that the 1-layer unentangled state is the only quantum model matching classical performance within $\pm 0.02$ AUC. Can you prove that this model is trivially classically simulable in polynomial time?"
**The Team's Collective Defense:**  
*"Yes, we proved this analytically in Section 7.2 of our academic report.

For an $n$-qubit 1-layer unentangled circuit, because there are zero entangling gates, the global state is a separable tensor product of single-qubit states:

$$
|\psi(\mathbf{x})\rangle = \bigotimes_{i=1}^n |\psi_i(x_i)\rangle
$$

The expectation value of an observable acting on qubit $i$ is completely independent of all other qubits:

$$
\langle \sigma_z^{(i)} \rangle = \langle \psi_i(x_i) | \sigma_z | \psi_i(x_i) \rangle = \cos(s \theta_i)
$$

$$
\langle X^{(i)} \rangle = \sin(s \theta_i)\cos(s \theta_i), \quad \langle Y^{(i)} \rangle = \sin^2(s \theta_i)
$$

Evaluating these $3n$ expectation values requires **$\mathcal{O}(n)$ basic scalar trigonometric operations on a standard classical CPU**. It requires zero Hilbert space statevectors, zero complex matrix exponentiation, and zero quantum hardware.

Therefore, any performance achieved by the 1-layer unentangled state lies entirely within classical polynomial time complexity."*

---

### Q15 (Math & Computing HOD): "Why did you use Bonferroni correction instead of the Benjamini-Hochberg False Discovery Rate (FDR) procedure?"
**The Team's Collective Defense:**  
*"The Benjamini-Hochberg procedure controls the False Discovery Rate (FDR)—the expected proportion of false positives among all declared discoveries ($\mathbb{E}[V/R]$). FDR is appropriate for exploratory, high-throughput discovery screening (such as scanning 20,000 genes in genomics) where a small fraction of false positives is acceptable.

In contrast, our benchmark was an **evaluative confirmatory hypothesis test** with a discrete set of 12 primary comparisons. In clinical and confirmatory scientific benchmarks, the standard is to control the **Family-Wise Error Rate (FWER)**—the probability of making even a single false positive across the entire study:

$$
\text{FWER} = P(V \ge 1) \le \alpha
$$

The Bonferroni correction is the gold standard for FWER control. It is strictly conservative: because our quantum deficits survived the rigorous Bonferroni threshold ($\alpha = 0.00417$), they are guaranteed to survive any less conservative correction, including Holm-Bonferroni or Benjamini-Hochberg."*

---

### Q16 (CSE & Math HOD): "What would happen if you scaled this pipeline to 20 or 50 qubits? Could Projected Quantum Kernels unlock advantage at scale?"
**The Team's Collective Defense:**  
*"Our empirical and theoretical findings indicate that scaling to 50 qubits under this architecture would actually **accelerate performance degradation**, not reverse it:

1. **Information Concentration in PCA:** In pediatric chest radiographs, optical variance drops off rapidly across eigenvectors. The first 8 principal components explain 77% of cumulative variance. Components 20 through 50 capture high-frequency scanner noise and patient clothing fibers, which degrades generalization.
2. **Entanglement Entropy Scaling:** As qubit count $n$ increases from 8 to 50, the entropy of the reduced density matrix $\rho_i = \text{Tr}_{\bar{i}}(|\psi\rangle\langle\psi|)$ under circular or all-to-all entanglement scales up, driving the mixed state even closer to the maximally mixed state $\frac{1}{2}\mathbf{I}$. Expectation values will concentrate even closer to zero, compounding the 34-fold variance collapse we already documented.

Scaling qubit count without fundamentally changing the feature encoding and measurement strategy will simply amplify the metric collapse."*

---

### Q17: "What is the single most important lesson that a computer science and quantum engineer must take away from project QKernel-Lite?"
**The Team's Collective Defense:**  
*"The central lesson is that **quantum features must match the natural mathematical geometry of the problem domain**.

Simply taking classical continuous optical data, chopping it into PCA angles, and passing it through entangling gates does not create a magic inductive advantage. Entanglement is not a universal performance booster—when paired with local observable projections, it acts as an entropy source that collapses variance and distorts metric distance.

The true future of Quantum Machine Learning lies not in force-fitting quantum circuits onto classical images where classical transfer learning already reigns supreme, but in applying quantum kernels to **natively quantum data structures**—such as molecular electronic Hamiltonians, quantum chemistry ground states, and quantum sensor networks—where quantum state geometry is the authentic physical language of the data."*

---
*(End of Overall Project Viva Defense Guide)*
