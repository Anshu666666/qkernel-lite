# 🏛️ QKernel-Lite: Overall Project Viva Defense Q&A
**Course:** IEG313: Quantum Computing for Engineers · IIIT Kottayam  
**Team:** Dirac's Army (Group 5)  
**Evaluators:** Head of Department (Computer Science & Engineering) & Head of Department (Mathematics & Computing)  
**Focus:** Cross-cutting architectural, theoretical, algorithmic, statistical, and clinical questions directed to the team as a whole.

---

### Q1: "Quantum computing currently has massive industry hype. Your project demonstrates that classical transfer learning beats quantum kernels decisively across all sample sizes. Why did you not pivot to a synthetic dataset or cherry-pick parameters to claim quantum advantage?"
> **The Team's Collective Defense:**  
> *"Because genuine engineering science requires rigorous intellectual honesty, not hype.  
> Much of the published literature in Quantum Machine Learning suffers from what is known as the 'strawman fallacy': authors evaluate quantum algorithms against unregularized, poorly tuned classical models on synthetic or toy datasets, claiming 'quantum advantage' that vanishes the moment a competent classical baseline is introduced.  
> Our mandate in IEG313 was to investigate whether Projected Quantum Kernels solve a genuine real-world problem: **pediatric radiograph scarcity in emergency clinics**. By testing on authentic clinical data (`PneumoniaMNIST`), maintaining an uncompromised 624-image test set, and establishing Model M2 as a strict scientific control holding features and optimization identical, our negative result is far more scientifically valuable than a contrived positive result. It identifies the exact mathematical boundaries—specifically phase wrapping and entanglement partial-trace collapse—where angle-encoded quantum kernels fail on optical continuous data."*

---

### Q2 (CSE HOD): "Does evaluating a Projected Quantum Kernel offer an asymptotic computational speedup over classical kernels? If not, what was the theoretical motivation for using quantum kernels on classical data?"
> **The Team's Collective Defense:**  
> *"No, quantum kernels offer **zero computational complexity speedup** when processing classical data.  
> To construct a Gram matrix for an SVM on $N$ classical data points, one must evaluate the pairwise kernel distance for all pairs:  
> $$\text{Computational Cost} = \mathcal{O}(N^2 \cdot T_{\text{kernel}})$$  
> Regardless of whether $T_{\text{kernel}}$ is computed via classical dot products or quantum circuit evaluations, the quadratic $\mathcal{O}(N^2)$ complexity of training a kernel machine remains unchanged.  
> **The Theoretical Motivation was Inductive Bias, not Speed:**  
> In learning theory, when data is extremely scarce ($N \le 20$), a model cannot rely on empirical data alone; its success depends entirely on its **inductive bias**—the geometric prior it imposes on the hypothesis space. The hypothesis in literature (Huang et al., 2021; Kübler et al., 2021) was that mapping classical coordinates into a non-linear $2^n$-dimensional quantum Hilbert space might construct a geometric representation where complex pathology becomes linearly separable with a wider margin than any classical kernel could achieve. Our study tested and disproved this inductive bias hypothesis for angle-encoded medical radiographs."*

---

### Q3 (CSE HOD): "A critic could object: 'Your classical baseline (M2) benefited from ResNet-18, which trained on 14 million ImageNet images. Your quantum circuit was untrained and randomly structured. Isn't that an unfair comparison?' How do you answer this?"
> **The Team's Collective Defense:**  
> *"That objection actually highlights why our scientific control is so airtight.  
> First, **both Model M2 and Model M3 consumed the exact same ResNet-18 features**. We did NOT feed raw pixels to M2 and ResNet features to M3. The classical baseline and the quantum pipeline were given the exact same 8 PCA components extracted from frozen ResNet-18.  
> Second, in clinical reality, an emergency hospital facing an epidemic does not operate in a vacuum—it has access to pretrained classical backbones. A proposed quantum technology must prove that it adds value over existing classical tools.  
> Third, quantum Model M3 was given a **$5\times$ larger hyperparameter tuning search budget** (270 grid combinations per fold vs. 36–54 for M2), specifically tuning the angle scale $s$ to rescue the quantum model. Even with ResNet features provided to both, and with the search budget heavily biased in quantum’s favor, quantum coordinate encoding actively degraded the classical representation."*

---

### Q4 (Math & Computing HOD): "Why did you choose angle encoding with circular CNOT gates? Could an alternative quantum feature map—such as IQP (Instantaneous Quantum Polynomial) or Hamiltonian evolution—have succeeded?"
> **The Team's Collective Defense:**  
> *"We benchmarked the exact circuit family proposed in the foundational study by Alagiyawanna et al. (SLAAI-ICAI 2024 / arXiv:2601.03375), which utilized $R_Y-R_Z$ angle encoding with entangling chains.  
> However, from a mathematical perspective, an IQP or Hamiltonian feature map would face the same fundamental metric breakdown:  
> 1. **Continuous Optical Data vs. Periodic Phases:** Clinical opacity in an X-ray is a continuous, monotonic physical property (higher tissue density = more optical attenuation = brighter pixels). Quantum gates rely on periodic trigonometric rotations ($e^{i \theta H}$). Any periodic mapping inherently risks folding widely separated clinical extremes together whenever features scale beyond the first monotonic quadrant.  
> 2. **Bipartite Entanglement Scrambling:** In both CNOT circuits and Hamiltonian evolution ($e^{-i t \sum Z_i Z_j}$), generating multi-qubit entanglement spreads local information into high-order tensor correlations. Unless the downstream measurement operators measure the global $n$-body observable (which triggers exponential concentration), projecting onto local 1-qubit or 2-qubit density matrices causes the local Bloch vectors to contract toward zero, destroying margin variance."*

---

### Q5 (CSE HOD): "Why did you use a fixed Quantum Kernel with a classical SVM instead of training a Parameterized Variational Quantum Circuit (VQC / QNN) directly with gradient descent?"
> **The Team's Collective Defense:**  
> *"Three decisive theoretical and practical reasons:  
> 1. **Convex Optimization vs. Non-Convex Landscapes:** Training a VQC using gradient descent via the parameter-shift rule optimizes a non-convex loss surface plagued by local minima, saddle points, and barren plateaus ($\text{Var}[\partial_\theta \mathcal{L}] \sim \mathcal{O}(2^{-n})$). In contrast, an SVM with a positive semi-definite Projected Quantum Kernel solves a strictly convex dual quadratic program ($\min_{\boldsymbol{\alpha}} \frac{1}{2}\boldsymbol{\alpha}^T \mathbf{Q} \boldsymbol{\alpha} - \mathbf{1}^T \boldsymbol{\alpha}$). Convexity guarantees a mathematically unique, global optimum.  
> 2. **Sample Efficiency:** On tiny datasets ($N = 10, 20$), optimizing dozens of variational circuit angles $\boldsymbol{\theta}$ using gradient steps leads to severe empirical overfitting, as the circuit parameters quickly memorize individual training samples.  
> 3. **Representer Theorem:** By the classical Representer Theorem, the optimal decision boundary for any kernel machine lies directly in the span of the training data: $f(\mathbf{x}) = \sum_{i=1}^N \alpha_i y_i k(\mathbf{x}_i, \mathbf{x}) + b$. This makes kernel methods fundamentally superior to VQCs for scarce sample regimes."*

---

### Q6 (Math & Computing HOD): "Entanglement is the primary quantum resource that enables Bell inequality violations and quantum computational supremacy. Why did entanglement actively destroy performance in your benchmark?"
> **The Team's Collective Defense:**  
> *"In quantum computing, entanglement is a powerful computational resource for algorithms that process global quantum states (such as Shor's factoring algorithm or Grover search). However, in **Projected Quantum Machine Learning**, entanglement acts as a destructive entropy source.  
> Mathematically, let $|\psi(\mathbf{x})\rangle \in \mathcal{H}_1 \otimes \dots \otimes \mathcal{H}_n$ be the global entangled state. To evaluate local Pauli observables, we compute:  
> $$\langle \sigma_z^{(i)} \rangle = \text{Tr}\left(\sigma_z^{(i)} \rho_i(\mathbf{x})\right), \quad \text{where } \rho_i(\mathbf{x}) = \text{Tr}_{\bar{i}}\left(|\psi(\mathbf{x})\rangle\langle\psi(\mathbf{x})|\right)$$  
> By the subadditivity of von Neumann entropy, tracing out an entangled subsystem increases the entropy of the reduced state: $S(\rho_i) > 0$. The reduced density matrix $\rho_i$ transitions from a pure state on the Bloch sphere surface to a mixed state inside the sphere:  
> $$\|\mathbf{r}_i\|^2 = 2\text{Tr}(\rho_i^2) - 1 < 1$$  
> As entanglement increases, the local Bloch vector contracts toward the origin $(0, 0, 0)$.  
> In our experiments, this caused a **34-fold collapse in expectation variance** ($0.290 \to 0.0085$). In clinical terms, diverse lung pathologies are projected to the same origin, erasing margin boundaries."*

---

### Q7 (Math & Computing HOD): "You showed that the 1-layer unentangled state is the only quantum model matching classical performance within $\pm 0.02$ AUC. Can you prove that this model is trivially classically simulable in polynomial time?"
> **The Team's Collective Defense:**  
> *"Yes, we proved this analytically in Section 7.2 of our academic report.  
> For an $n$-qubit 1-layer unentangled circuit, because there are zero entangling gates, the global state is a separable tensor product of single-qubit states:  
> $$|\psi(\mathbf{x})\rangle = \bigotimes_{i=1}^n |\psi_i(x_i)\rangle$$  
> The expectation value of an observable acting on qubit $i$ is completely independent of all other qubits:  
> $$\langle \sigma_z^{(i)} \rangle = \langle \psi_i(x_i) | \sigma_z | \psi_i(x_i) \rangle = \cos(s \theta_i)$$  
> $$\langle X^{(i)} \rangle = \sin(s \theta_i)\cos(s \theta_i), \quad \langle Y^{(i)} \rangle = \sin^2(s \theta_i)$$  
> Evaluating these $3n$ expectation values requires **$\mathcal{O}(n)$ basic scalar trigonometric operations on a standard classical CPU**. It requires zero Hilbert space statevectors, zero complex matrix exponentiation, and zero quantum hardware.  
> Therefore, any performance achieved by the 1-layer unentangled state lies entirely within classical polynomial time complexity."*

---

### Q8 (Math & Computing HOD): "Why did you use Bonferroni correction instead of the Benjamini-Hochberg False Discovery Rate (FDR) procedure?"
> **The Team's Collective Defense:**  
> *"The Benjamini-Hochberg procedure controls the False Discovery Rate (FDR)—the expected proportion of false positives among all declared discoveries ($\mathbb{E}[V/R]$). FDR is appropriate for exploratory, high-throughput discovery screening (such as scanning 20,000 genes in genomics) where a small fraction of false positives is acceptable.  
> In contrast, our benchmark was an **evaluative confirmatory hypothesis test** with a discrete set of 12 primary comparisons. In clinical and confirmatory scientific benchmarks, the standard is to control the **Family-Wise Error Rate (FWER)**—the probability of making even a single false positive across the entire study:  
> $$\text{FWER} = P(V \ge 1) \le \alpha$$  
> The Bonferroni correction is the gold standard for FWER control. It is strictly conservative: because our quantum deficits survived the rigorous Bonferroni threshold ($\alpha = 0.00417$), they are guaranteed to survive any less conservative correction, including Holm-Bonferroni or Benjamini-Hochberg."*

---

### Q9 (CSE HOD): "How does your Streamlit prototype handle model disagreement? If M1 predicts normal while M2 and M3 predict pneumonia, what does the interface show?"
> **The Team's Collective Defense:**  
> *"Our prototype interface implements a dedicated **Tri-Model Diagnostic Consensus Engine** directly in `app.py`.  
> For every patient scan, the platform simultaneously evaluates:  
> 1. M1: Scratch CNN posterior probability $P(Y=1|\mathbf{x}) \in [0, 1]$.  
> 2. M2: Classical RBF-SVM signed distance margin $f_{M2}(\mathbf{x}) = \mathbf{w} \cdot \phi(\mathbf{x}) + b$.  
> 3. M3: Quantum PQK-SVM signed distance margin $f_{M3}(\mathbf{x}) = \mathbf{w} \cdot \mathbf{E}(\mathbf{x}) + b$.  
> It categorizes the output into three explicit clinical states:  
> - **Complete Consensus (3/3 Agree):** All models concordantly classify the scan.  
> - **Kernel Consensus vs. CNN Divergence (2/3 Agree):** Classical RBF and Quantum PQK agree, while Scratch CNN diverges—visually demonstrating that transfer learning representations override scratch CNN noise memorization.  
> - **Model Divergence:** The UI renders an alert banner flagging that the radiograph lies within a decision margin boundary, explicitly warning the user that model confidence is split."*

---

### Q10 (CSE & Math HOD): "What would happen if you scaled this pipeline to 20 or 50 qubits? Could Projected Quantum Kernels unlock advantage at scale?"
> **The Team's Collective Defense:**  
> *"Our empirical and theoretical findings indicate that scaling to 50 qubits under this architecture would actually **accelerate performance degradation**, not reverse it.  
> 1. **Information Concentration in PCA:** In pediatric chest radiographs, optical variance drops off rapidly across eigenvectors. The first 8 principal components explain 77% of cumulative variance. Components 20 through 50 capture high-frequency scanner noise and patient clothing fibers, which degrades generalization.  
> 2. **Entanglement Entropy Scaling:** As qubit count $n$ increases from 8 to 50, the entropy of the reduced density matrix $\rho_i = \text{Tr}_{\bar{i}}(|\psi\rangle\langle\psi|)$ under circular or all-to-all entanglement scales up, driving the mixed state even closer to the maximally mixed state $\frac{1}{2}\mathbf{I}$. Expectation values will concentrate even closer to zero, compounding the 34-fold variance collapse we already documented.  
> Scaling qubit count without fundamentally changing the feature encoding and measurement strategy will simply amplify the metric collapse."*

---

### Q11: "In your live demo and report, you enforce a strict Clinical Research Disclaimer. From a bioethics and medical device regulatory standpoint (e.g. FDA / CDSCO), why is this mandatory?"
> **The Team's Collective Defense:**  
> *"Pediatric pulmonology involves critical, life-or-death diagnoses: untreated bacterial pneumonia can cause respiratory failure within hours, while misdiagnosing a normal lung can lead to unnecessary antibiotic exposure or invasive interventions.  
> Under FDA Software as a Medical Device (SaMD) and CDSCO digital health guidelines, any diagnostic system deployed in a clinical setting requires rigorous prospective clinical trials, multisite validation across different X-ray machine manufacturers, and certified ISO 13485 quality management.  
> Because our benchmark uses simulated few-shot regimes ($N=10$) on downsampled $28 \times 28$ images to explore quantum computing theory, deploying or presenting this tool as a clinical diagnostic instrument would be unethical and legally impermissible. Our platform is strictly an engineering benchmark to evaluate machine learning representations."*

---

### Q12: "What is the single most important lesson that a computer science and quantum engineer must take away from project QKernel-Lite?"
> **The Team's Collective Defense:**  
> *"The central lesson is that **quantum features must match the natural mathematical geometry of the problem domain**.  
> Simply taking classical continuous optical data, chopping it into PCA angles, and passing it through entangling gates does not create a magic inductive advantage. Entanglement is not a universal performance booster—when paired with local observable projections, it acts as an entropy source that collapses variance and distorts metric distance.  
> The true future of Quantum Machine Learning lies not in force-fitting quantum circuits onto classical images where classical transfer learning already reigns supreme, but in applying quantum kernels to **natively quantum data structures**—such as molecular electronic Hamiltonians, quantum chemistry ground states, and quantum sensor networks—where quantum state geometry is the authentic physical language of the data."*

---
