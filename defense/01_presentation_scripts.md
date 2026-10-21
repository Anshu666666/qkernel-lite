# 🎙️ QKernel-Lite: Complete Team Presentation Scripts
**Course:** IEG313: Quantum Computing for Engineers · IIIT Kottayam  
**Team:** Dirac's Army (Group 5)  
**Total Target Duration:** 9 minutes 30 seconds (Budget: 9–10 minutes)  
**Evaluators:** Head of Department (Computer Science & Engineering) & Head of Department (Mathematics & Computing)

---

## ⏱️ Master Timing & Stage Coordination Table

| Presenter | Assigned Modules | Slides | Target Time | Cumulative Time |
| :--- | :--- | :--- | :---: | :---: |
| **Daksh Singh** | Title, Thesis, Data Scarcity, Paired $t$-Test, Bonferroni | 01, 02, 03, 04, 05 | **2 min 00 s** | 0:00 – 2:00 |
| **Anshuman Biswas** | Scratch CNN (M1), Frozen ResNet-18, PCA, Classical Control (M2) | 06, 07 | **1 min 50 s** | 2:00 – 3:50 |
| **Himmat Sirwee** | 8Q Angle Encoding, Statevector Sim, 1-Layer Closed Form, Small-$s$ Duality | 08, 09 | **1 min 50 s** | 3:50 – 5:40 |
| **Sawan Kumar** | Phase Wrapping Bug, PQK vs. Fidelity Plateaus, ZZ Correlators | 10, 11, 12 | **2 min 00 s** | 5:40 – 7:40 |
| **Aryan Gupta** | 960-Run Ledger, Budget Asymmetry, Prototype Platform, Bounded Conclusions | 13, 14, 15, 16 | **2 min 05 s** | 7:40 – 9:45 |

---

# 1. Daksh Singh (2024BCS0042)
**Role:** Opening, Problem Definition, Dataset Protocol, Statistical Framework  
**Assigned Slides:** 01, 02, 03, 04, 05  
**Allotted Time:** ~2 minutes 00 seconds (~330 words)

---

### slide-1 | Slide 01: Title & Cover
*(Target Time: ~15 seconds)*

> **[Spoken Delivery]**  
> *"Respected Heads of Departments, Faculty, and colleagues, good morning.  
>
> We are **Dirac’s Army**, and today we present **QKernel-Lite**: an empirical, paired benchmark evaluating Projected Quantum Kernel Support Vector Machines against classical transfer learning baselines under extreme medical data scarcity."*

---

### slide-2 | Slide 02: The Project in Six Incontrovertible Scientific Facts
*(Target Time: ~35 seconds)*

> **[Spoken Delivery]**  
> *(Point to Left Card — Methodological Framework)*  
> *"To orient our defense, our entire study reduces to six rigorous findings.  
>
> Simply put: recent literature hypothesizes that mapping classical data into quantum Hilbert spaces grants an inductive advantage when training labels are scarce. To test this honestly, we took pediatric chest X-rays with sample sizes as small as 10 to 100 scans, extracted 8 clinical features via frozen ResNet-18, and compared classical Gaussian RBF kernels against projected quantum kernels using the exact same convex dual quadratic solver.  
>
> *(Point to Right Card — Measured Scientific Reality)*  
> *The measured reality was unambiguous: **the classical model won decisively across every sample size**. In fact, quantum entanglement did not create advantage—it collapsed feature variance by 34-fold and systematically degraded diagnostic performance. The only quantum circuit matching classical accuracy was a 1-layer unentangled product state, which is trivially classically simulable."*

---

### slide-3 | Slide 03: Module Ownership & Presentation Sequence
*(Target Time: ~15 seconds)*

> **[Spoken Delivery]**  
> *(Glance at Architecture Diagram)*  
> *"To ensure complete transparency, every member of our team independently designed, implemented, and defends their assigned module—from statistical discipline and deep feature extraction to quantum circuit mathematics and platform engineering."*

---

### slide-4 | Slide 04: Pediatric Radiography & Extreme Label Scarcity
*(Target Time: ~25 seconds)*

> **[Spoken Delivery]**  
> *(Point to Left Box — Experimental Protocol)*  
> *"Our clinical testing ground is **PneumoniaMNIST**, consisting of pediatric chest radiographs. In rare pediatric illnesses or novel outbreaks, clinicians often have fewer than 20 verified scans.  
>
> We simulated four extreme scarcity regimes: $N = 10, 20, 50,$ and $100$ balanced training scans. But crucially, look at the right card: we maintained an uncompromised, fixed evaluation set of **624 real patient scans** ($234$ normal, $390$ pneumonia).  
>
> Why ROC-AUC instead of raw accuracy? Because the test set is 62.5% pneumonia. A naive model predicting pneumonia every single time scores 62% accuracy with zero diagnostic sensitivity. ROC-AUC evaluates true ranking invariant to class imbalance."*

---

### slide-5 | Slide 05: Paired Student's t-Test & Bonferroni Family-Wise Control
*(Target Time: ~30 seconds)*

> **[Spoken Delivery]**  
> *(Point to Left Equation — Paired Student's t)*  
> *"To ensure statistical discipline, we ran a fully paired experimental design across **15 independent random seeds** ($df = 14$). By evaluating all models on the identical patient subsets per seed, between-sample variance cancels out, isolating true model deltas.  
>
> With $n=15$, population variance is unknown, requiring a Student’s $t$-distribution ($t_{0.025, 14} = 2.145$). Furthermore, because we conducted 12 primary pairwise comparisons, standard $\alpha = 0.05$ guarantees false positives. We enforced a strict **Bonferroni correction**:  
> $$\alpha_{\text{bonf}} = \frac{0.05}{12} = 0.00417$$  
>
> *(Point to Right Card — Equivalence Gate)*  
> *Every single quantum deficit survived this threshold ($p < 0.001$). Furthermore, under a strict clinical equivalence interval of $\pm 0.02$ AUC, entangled circuits failed catastrophically.  
>
> Now, I hand over to Anshuman to walk through our deep classical baselines."*

---

# 2. Anshuman Biswas (2024BCD0008)
**Role:** M1 (Scratch CNN), ResNet-18 Transfer Feature Pipeline, PCA Low-Rank Reduction, Model M2 (Classical Control)  
**Assigned Slides:** 06, 07  
**Allotted Time:** ~1 minute 50 seconds (~320 words)

---

### slide-6 | Slide 06: M1 Scratch CNN & Frozen ResNet-18 Transfer Learning
*(Target Time: ~55 seconds)*

> **[Spoken Delivery]**  
> *"Thank you, Daksh. Good morning Professors.  
>
> To benchmark quantum kernels fairly, we first need to know what classical deep learning actually does under extreme scarcity.  
>
> *(Point to Left Card — M1 Scratch CNN)*  
> *Our first baseline, **Model M1**, is a standard convolutional neural network trained from scratch. In plain terms: when an emergency clinic only has 10 X-rays, training 105,000 weights is like asking a student to memorize an entire textbook after seeing only two practice questions. Mathematically, our parameter-to-data ratio exceeds $10,000\text{ to }1$ ($P/N > 10^4$). Under such severe parameter under-determination, gradient descent easily overfits scanner artifacts and bedframe shadows.  
>
> Yet, notice an honest empirical finding: M1 did not collapse to random guessing—it achieved $0.847$ AUC at $N=10$. Pneumonia opacities produce large, distinct optical consolidations that simple spatial convolutions can partially capture.  
>
> *(Point to Right Card — Frozen ResNet-18)*  
> *To solve overfitting properly, we used **Transfer Learning**. Instead of training on 10 scans, we freeze an 11.7-million parameter ResNet-18 pretrained on ImageNet. Why freeze it completely? Because fine-tuning 11 million weights on 10 scans guarantees catastrophic representation collapse. By locking the weights and fixing batch-normalization in `eval()` mode, its convolutional primitives—edges, lung textures, and tissue boundaries—transfer directly to chest radiographs. Passing each scan through Global Average Pooling extracts a rich, noise-free **512-dimensional latent vector**."*

---

### slide-7 | Slide 07: PCA Dimensionality Reduction & Classical Control (M2)
*(Target Time: ~55 seconds)*

> **[Spoken Delivery]**  
> *(Point to Left Card — PCA Compression)*  
> *"Now comes the mathematical bridge between our classical features and quantum simulation.  
>
> Simulating an $n$-qubit register scales as $2^n$. Simulating 512 qubits is physically impossible on classical hardware, so we must compress 512 dimensions to **8 principal components**.  
>
> Here, we enforced strict linear algebra discipline: PCA is fitted **exclusively on the training fold**. At $N=10$, the sample covariance matrix has a rank of at most $N-1 = 9$. We extract the top 8 eigenvectors—retaining **77% of cumulative variance**—and min-max scale the coordinates to $[0, \pi]$ so they can drive physical quantum rotation angles $\theta_i$. Crucially, test scans are projected using training fold eigenvectors and min-max boundaries, guaranteeing **zero data leakage**.  
>
> *(Point to Right Card — Model M2 RBF-SVM)*  
> *This establishes **Model M2**, the scientific linchpin of our entire research capstone.  
>
> In published literature, authors frequently compare quantum models against unregularized scratch CNNs and claim 'quantum advantage'. That is an artificially weak strawman. Model M2 consumes the **exact same 8 PCA components** and maps them through a classical Gaussian RBF kernel into an identical dual convex quadratic programming SVM solver.  
>
> The result? Classical transfer learning provides a massive diagnostic lift: **$+0.0488$ AUC at $N=10$**, and a statistically verified **$+0.0445$ lift at $N=20$** ($p = 0.0010$, surviving Bonferroni). This is the true benchmark quantum must beat.  
>
> Now, Himmat will explain how these 8 clinical angles are encoded into quantum circuits."*

---

# 3. Himmat Sirwee (2024BCS0040)
**Role:** Quantum Register Topology, Statevector Simulation, 1-Layer Closed Form, Small-$s$ Taylor Duality  
**Assigned Slides:** 08, 09  
**Allotted Time:** ~1 minute 50 seconds (~310 words)

---

### slide-8 | Slide 08: 8-Qubit Angle Encoding & Circuit Topology
*(Target Time: ~55 seconds)*

> **[Spoken Delivery]**  
> *"Thank you, Anshuman. Good morning Professors.  
>
> *(Point to Left Card — Circuit Topology)*  
> *To build our quantum pipeline, we initialize an 8-qubit register in the ground state $|0^{\otimes 8}\rangle = |00000000\rangle$, where each qubit corresponds to one compressed clinical feature coordinate $\theta_i \in [0, \pi]$.  
>
> On each qubit $i$, we apply a rotation block $R_Y(s\theta_i)$ followed by $R_Z(s\theta_i)$, where $s$ is an encoding scale parameter. In entangled architectures, we interleave these rotations with a circular CNOT chain—connecting qubit 0 to 1, 1 to 2, and wrapping 7 back to 0. In 2-layer circuits, this entire block is repeated twice.  
>
> *(Point to Right Card — Simulation vs. Physical Hardware)*  
> *Now, look at the simulation methodology on the right. We execute exact statevector simulation using PennyLane’s `lightning.qubit` engine. An 8-qubit statevector requires $2^8 = 256$ complex amplitudes, which is classically simulated without Monte Carlo shot noise.  
>
> This is a crucial scientific distinction: physical NISQ QPUs suffer from thermal relaxation ($T_1$), dephasing ($T_2$), and finite measurement sampling error. Because our statevector simulation has **zero physical hardware noise**, any deficit we observe in quantum performance is purely algorithmic and mathematical, not an artifact of noisy hardware."*

---

### slide-9 | Slide 09: Mathematical Derivation: 1-Layer Closed Form & Small-s Duality
*(Target Time: ~55 seconds)*

> **[Spoken Delivery]**  
> *(Point to Left Box — Exact Closed Form)*  
> *"To understand what the quantum circuit is doing geometrically, we derived the analytical closed form for the 1-layer unentangled product state.  
>
> Applying $R_Y(s\theta)$ to $|0\rangle$ rotates the state in the $X$-$Z$ plane:  
> $$|\psi_1\rangle = \cos(s\theta/2)|0\rangle + \sin(s\theta/2)|1\rangle$$  
> Applying $R_Z(s\theta)$ injects a relative phase shift $e^{is\theta}$ into the $|1\rangle$ component:  
> $$|\psi\rangle = \cos(s\theta/2)|0\rangle + e^{is\theta}\sin(s\theta/2)|1\rangle$$  
>
> By calculating the expectation values of the Pauli operators $\langle P \rangle = \langle \psi | P | \psi \rangle$, we obtain the exact coordinates of the Bloch vector:  
> $$\langle X \rangle = \sin(s\theta)\cos(s\theta), \quad \langle Y \rangle = \sin^2(s\theta), \quad \langle Z \rangle = \cos(s\theta)$$  
>
> *(Point to Right Box — Taylor Duality)*  
> *Now examine the right box: what happens when the scale factor $s$ is tuned small ($s \le 0.125$)?  
> Taking the first-order Taylor series around zero reveals a profound duality:  
> $$\langle X \rangle \approx s\theta, \quad \langle Y \rangle \approx (s\theta)^2, \quad \langle Z \rangle \approx 1 - \frac{(s\theta)^2}{2}$$  
>
> Notice that $\langle X \rangle$ is strictly linear in the input feature, while $\langle Y \rangle$ and $\langle Z \rangle$ are quadratic. When these 24 expectation values feed an RBF kernel, **the quantum model mathematically reduces to a classical polynomial approximation**!  
>
> This proves why the quantum model performs well at small $s$: it recovers classical accuracy precisely because its quantum characteristics have been suppressed to mimic a classical kernel.  
>
> Now, Sawan will demonstrate what happens when default quantum parameters break down."*

---

# 4. Sawan Kumar (2024BCS0164)
**Role:** Phase Wrapping Bug, PQK vs. Fidelity Plateaus, Metric Diagnostics, Refuted ZZ Correlator Hypothesis  
**Assigned Slides:** 10, 11, 12  
**Allotted Time:** ~2 minutes 00 seconds (~330 words)

---

### slide-10 | Slide 10: Physical Failure Modes: Phase Wrapping & Entanglement Collapse
*(Target Time: ~40 seconds)*

> **[Spoken Delivery]**  
> *"Thank you, Himmat. Good morning Professors.  
>
> *(Point to Figure 5 Plot on Right)*  
> *When researchers run projected quantum kernels using default literature settings ($s = 1.00$), the model collapses catastrophically to **55.6% AUC at $N=10$**—barely better than a coin flip! Why?  
>
> We identified the exact physical failure mode: the **Phase Wrapping Turnaround Bug**.  
> In a 2-layer circuit, rotation angles accumulate across layers ($U(\pi)^2|0\rangle = |0\rangle$). As shown in Figure 5, the $Z$-axis expectation drops monotonically to $-1$ at $\theta = \pi/2$, but then turns around and climbs back to $+1$ at $\theta = \pi$:  
> $$\langle Z(0)\rangle = +1 \quad \text{and} \quad \langle Z(\pi)\rangle = +1$$  
>
> In clinical terms, this is a disaster: a healthy lung ($\theta \approx 0$) and severe consolidative pneumonia ($\theta \approx \pi$) get mapped to the exact same point in Hilbert space! Compressing the angles ($s \le 0.25$) confines rotations to the monotonic quadrant $[0, \pi/4]$, which eliminates turnaround and recovers stability to $77.2\%$ AUC."*

---

### slide-11 | Slide 11: Projected Quantum Kernels vs. Fidelity Barren Plateaus
*(Target Time: ~40 seconds)*

> **[Spoken Delivery]**  
> *(Point to Left Box — Fidelity Kernel)*  
> *"Next, we address why we used Projected Quantum Kernels instead of standard fidelity kernels.  
>
> A conventional fidelity kernel evaluates global state overlap: $k_{\text{fid}}(\mathbf{x}, \mathbf{x}') = |\langle \phi(\mathbf{x}) | \phi(\mathbf{x}') \rangle|^2$. As proved by Huang et al. (Nature Communications 2021), in an $n$-qubit Hilbert space of dimension $2^n$, random states become orthogonal with high probability. The kernel variance concentrates exponentially toward zero:  
> $$\text{Var}[k_{\text{fid}}] \sim \mathcal{O}(2^{-n})$$  
> For large $n$, the Gram matrix collapses into the identity matrix $I$, destroying margin separability.  
>
> *(Point to Right Box — Projected Quantum Kernel)*  
> *Projected Quantum Kernels bypass this barren plateau by projecting the state back into classical metric space using 1-particle reduced density matrices, measuring 24 local Pauli expectations:  
> $$E(\mathbf{x}) = [\langle X_1 \rangle, \langle Y_1 \rangle, \dots, \langle Z_8 \rangle]^T \in \mathbb{R}^{24}$$  
> A classical Gaussian RBF kernel is then evaluated on these expectations. Because the outer function is a standard Gaussian on $\mathbb{R}^{24}$, **Mercer’s condition is strictly preserved**, guaranteeing a positive semi-definite Gram matrix and a convex dual optimization problem."*

---

### slide-12 | Slide 12: Empirical Diagnostics & The Tested ZZ Correlator Hypothesis
*(Target Time: ~40 seconds)*

> **[Spoken Delivery]**  
> *(Point to Figure 4 on Left)*  
> *"However, look at our geometric diagnostics in Figure 4. When we measured rank correlation between input Euclidean distance and kernel distance, classical M2 preserved geometry almost perfectly ($r = 0.998$). But under quantum entanglement, the Spearman rank collapsed to **$r = 0.412$**, scrambling clinical distances! Furthermore, CNOT entangling gates caused single-qubit expectation variance to collapse **34-fold** (from $0.290$ down to $0.0085$).  
>
> *(Point to Right Box — ZZ Correlators)*  
> *At this stage, we asked a fundamental question:  
> *'Did entanglement fail simply because we measured single qubits individually and threw away two-qubit quantum correlations?'*  
>
> To test this, we added 8 nearest-neighbor correlators $\langle Z_i Z_{i+1} \rangle$, expanding to 32 total expectations.  
> The result refuted the hypothesis: **test AUC dropped further across all sample sizes** (falling by up to $-0.034$ AUC at $N=20$). Discarding correlations is not the flaw—multi-qubit entanglement actively corrupts the clinical metric space!  
>
> Now, Aryan will present our complete experimental ledger and platform implementation."*

---

# 5. Aryan Gupta (2024BCS0020)
**Role:** 960-Run Ledger, Tuning Budget Accounting, Interactive Streamlit Platform, Bounded Conclusions & Viva Defense  
**Assigned Slides:** 13, 14, 15, 16  
**Allotted Time:** ~2 minutes 05 seconds (~340 words)

---

### slide-13 | Slide 13: Exact Experimental Accounting: 960 Controlled Runs
*(Target Time: ~35 seconds)*

> **[Spoken Delivery]**  
> *"Thank you, Sawan. Good morning Professors.  
>
> *(Point to Left Table — Row Accounting)*  
> *To guarantee scientific reproducibility, we executed an exact ledger of **960 controlled runs** across 15 seeds:  
> - 60 runs for Model M1,  
> - 180 runs for classical Model M2 across feature counts,  
> - 60 runs for default quantum M3, and  
> - 660 runs across 11 systematic quantum ablation cells.  
> Every single run was evaluated on the identical 624-image test set. There was zero cherry-picking and zero missing data.  
>
> *(Point to Right Box — Budget Asymmetry)*  
> *Now, look at the right card: **Tuning Budget Asymmetry**.  
> For classical M2, we searched over 36 to 54 hyperparameter configurations per fold. But for quantum M3, we searched over $C$, $\gamma$, AND scale $s$, evaluating **270 configurations per fold**—a **$5\times$ larger tuning budget**.  
> Even with this massive search advantage skewed in quantum's favor, quantum failed to outperform the classical control."*

---

### slide-14 | Slide 14: Empirical Benchmark Ledger & Bonferroni Significance
*(Target Time: ~35 seconds)*

> **[Spoken Delivery]**  
> *(Point to Table — Benchmark Results)*  
> *"Here is the definitive empirical ledger across 15 seeds ($df = 14$):  
> At $N = 10$, classical M2 reaches **0.8957 AUC**, while tuned quantum M3 reaches **0.7717 AUC**—a significant deficit of **$-0.1241$ ($p = 2.4 \times 10^{-4}$)**.  
> At $N = 20$, the deficit is **$-0.0846$ ($p = 3.8 \times 10^{-4}$)**.  
> At $N = 50$, the deficit is **$-0.0296$ ($p = 7.9 \times 10^{-4}$)**.  
> At $N = 100$, the deficit is **$-0.0288$ ($p = 1.5 \times 10^{-4}$)**.  
>
> Every single paired difference survives the Bonferroni threshold ($\alpha = 0.00417$). The hypothesis of quantum advantage on scarce radiographs is conclusively rejected."*

---

### slide-15 | Slide 15: Streamlit Research Prototype & Disclaimer
*(Target Time: ~30 seconds)*

> **[Spoken Delivery]**  
> *(Point to Prototype Architecture on Left)*  
> *"To make our benchmark interactive, we engineered a production platform in Streamlit. Instead of running expensive quantum simulations during inference, we pre-compiled a lightweight 28.6 MB internal database (`prototype_db.npz`). This enables instant switching across all 624 patients, seeds, and circuits in under 5 milliseconds.  
>
> It is deployed live on Oracle Cloud under Caddy with verified Let’s Encrypt SSL at `qkernel-lite.duckdns.org`, passing all 32 unit tests in 28 seconds.  
>
> *(Point to Disclaimer on Right)*  
> *Crucially, we enforce a mandatory clinical disclaimer: this is strictly an academic benchmark and not a certified medical device."*

---

### slide-16 | Slide 16: Bounded Conclusions & Viva Defense Integrity
*(Target Time: ~25 seconds)*

> **[Spoken Delivery]**  
> *(Point to Permissible Claims on Left)*  
> *"In conclusion, we stand firmly behind four bounded claims:  
> 1. Pretrained transfer learning accounts for 100% of diagnostic capability.  
> 2. Projected quantum kernels remain statistically inferior to matched classical RBF-SVM across all sample sizes.  
> 3. The only quantum circuit matching classical performance is a 1-layer product state, which is classically simulable.  
> 4. Multi-qubit entanglement actively degrades diagnostic metric geometry.  
>
> *(Point to Bottom Card)*  
> *We do not claim quantum kernels can never work on images—we report what we measured with complete statistical rigor. In engineering science, **a reproducible null result is vastly superior to an unreproducible claim of advantage**.  
>
> Thank you, Professors. We now welcome your questions."*

---
*(All five presenters stand together, ready for viva questions).*
