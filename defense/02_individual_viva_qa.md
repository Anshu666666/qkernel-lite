# 🎯 QKernel-Lite: Individual Viva Defense Q&A
**Course:** IEG313: Quantum Computing for Engineers · IIIT Kottayam  
**Team:** Dirac's Army (Group 5)  
**Evaluators:** Head of Department (Computer Science & Engineering) & Head of Department (Mathematics & Computing)  
**Purpose:** Highly technical, rigorous questions targeting each student's specific assigned modules, with bulletproof academic answers addressing both CSE and Mathematical angles.

---

# 1. Daksh Singh (2024BCS0042)
**Assigned Modules:** Dataset Protocol (PneumoniaMNIST), Scarcity Regimes ($N \in \{10, 20, 50, 100\}$), Evaluation Metrics (ROC-AUC), Paired Student's $t$-Test, Bonferroni Family-Wise Error Control, Clinical Equivalence Testing.

---

### Q1.1 (CSE HOD): Why did you use PneumoniaMNIST ($28 \times 28$ grayscale) instead of full-resolution clinical chest X-rays ($1024 \times 1024$ DICOMs from CheXpert or NIH)?
**Bulletproof Answer:**  
*"Two reasons: computational tractability of quantum simulation, and experimental control.  
First, exact classical statevector simulation of quantum circuits scales exponentially ($2^n$). Compressing a $1024 \times 1024$ image directly into 8 qubits would require an extreme compression ratio of $131,072 : 1$, which would conflate image downsampling loss with quantum kernel efficacy. PneumoniaMNIST (standardized from Kermany et al., Cell 2018) is already an accepted benchmark in biomedical machine learning (Yang et al., Nature Scientific Data 2023).  
Second, PneumoniaMNIST provides a pristine, standardized $28 \times 28$ format that isolates biological density variations while allowing us to execute an exact 960-run ledger across 15 seeds. Furthermore, we maintain a completely untouched 624-image test set to ensure rigorous generalizability."*

---

### Q1.2 (Math & Computing HOD): Why is raw classification accuracy a misleading metric on clinical radiograph datasets, and how is ROC-AUC mathematically formulated?
**Bulletproof Answer:**  
*"PneumoniaMNIST's 624-image test set exhibits severe class imbalance: 390 pneumonia cases (62.5%) versus 234 healthy cases (37.5%). If a naive or degenerate model simply outputs 'Pneumonia' for every patient without looking at the image, it trivially achieves **62.5% accuracy** despite having zero clinical diagnostic utility and 0% specificity.  
Mathematically, ROC-AUC measures the probability that a classifier ranks a randomly chosen positive instance higher than a randomly chosen negative instance. It is equivalent to the normalized Wilcoxon-Mann-Whitney rank statistic:  

$$\text{AUC} = \frac{1}{N_+ N_-} \sum_{i \in \mathcal{P}} \sum_{j \in \mathcal{N}} \mathbb{I}\left(f(\mathbf{x}_i) > f(\mathbf{x}_j)\right)$$

Where $N_+, N_-$ are the counts of pneumonia and normal scans, and $\mathbb{I}$ is the indicator function. ROC-AUC is completely threshold-free and scale-invariant, making it the mathematically honest standard for imbalanced clinical diagnostics."*

---

### Q1.3 (Math & Computing HOD): Why did you use a paired Student's $t$-test with $df = 14$ instead of an independent two-sample $t$-test or a standard $z$-test?
**Bulletproof Answer:**  
*"An independent two-sample $t$-test assumes two uncorrelated groups and absorbs between-sample variability into the denominator, inflating the pooled standard error and drastically reducing statistical power.  
In our experimental design, Model M1, M2, and M3 were evaluated on the **exact same random patient subsets** for each seed $k \in \{0, \dots, 14\}$. Therefore, we compute the paired difference:  

$$\Delta_k = \text{AUC}_{M3, k} - \text{AUC}_{M2, k}$$

This directly subtracts out between-patient subset variation, isolating the model architecture as the sole source of variance.  
Furthermore, because our sample size is $n = 15$, the central limit theorem does not allow assuming an asymptotic Gaussian $z$-distribution since the true population variance $\sigma^2$ is unknown. We must use the exact Student’s $t$-distribution with $n - 1 = 14$ degrees of freedom ($t_{0.025, 14} = 2.145$)."*

---

### Q1.4 (Math & Computing HOD): What is Family-Wise Error Rate (FWER), and how does your Bonferroni correction threshold ($\alpha = 0.00417$) protect against $p$-hacking?
**Bulletproof Answer:**  
*"When conducting multiple statistical hypothesis tests simultaneously, the probability of obtaining at least one false positive (Type I error) by pure chance compounds. If you perform $m$ independent tests at standard $\alpha = 0.05$, the Family-Wise Error Rate is:  

$$\text{FWER} = 1 - (1 - \alpha)^m \approx 1 - (0.95)^{12} \approx 46.0\%$$

That means nearly a 50% chance of claiming a false breakthrough!  
To guarantee that the overall probability of even a single false positive across the entire benchmark remains bounded by 0.05, we apply the Bonferroni inequality across our 12 primary pairwise tests (3 model pairs $\times$ 4 sample sizes $N$):  

$$\alpha_{\text{bonf}} = \frac{\alpha_{\text{global}}}{m} = \frac{0.05}{12} = 0.004167$$

Every paired deficit we reported ($p = 2.4 \times 10^{-4}$ at $N=10$, $3.8 \times 10^{-4}$ at $N=20$, $7.9 \times 10^{-4}$ at $N=50$, and $1.5 \times 10^{-4}$ at $N=100$) easily survives this stringent threshold."*

---

### Q1.5 (CSE & Math HOD): Explain the Clinical Equivalence Gate ($\pm 0.02$ AUC). Why is 'failing to reject the null hypothesis' ($p > 0.05$) NOT proof that two models are equivalent?
**Bulletproof Answer:**  
*"In statistical inference, **absence of evidence is not evidence of absence**. A high $p$-value ($p > 0.05$) often simply reflects low statistical power, high sample noise, or an insufficient sample size—it never proves that two models are identical.  
To scientifically assert that a quantum model matches a classical model, we formalized a Two One-Sided Tests (TOST) equivalence interval:  

$$\text{Equivalence Gate: } \text{CI}_{95\%}(\text{AUC}_{M3} - \text{AUC}_{M2}) \subseteq [-0.02, +0.02]$$

In medical radiology, a delta within $\pm 0.02$ AUC represents clinical indifference.  
Under this criterion:  
1. The **1-layer unentangled product state** at $s \le 0.25$ passes the gate for $N \ge 20$ (its 95% confidence interval is entirely contained within $[-0.02, +0.02]$).  
2. The **2-layer entangled circuit** fails catastrophically, with confidence intervals dropping down to $-0.18$.  
This proves mathematically that entanglement fails to achieve clinical equivalence."*

---

# 2. Anshuman Biswas (2024BCD0008)
**Assigned Modules:** Model M1 (Scratch CNN Architecture & Overfitting), Pretrained ResNet-18 Feature Extractor, PCA Dimensionality Reduction, Model M2 (Classical RBF-SVM Scientific Control).

---

### Q2.1 (CSE HOD): Your Model M1 Scratch CNN achieved ~0.847 AUC at $N=10$ despite having 105,826 parameters. Why did it not collapse to 0.50 (random guessing) given $P/N > 10,000$?
**Bulletproof Answer:**  
*"In published quantum literature, authors frequently assert that unregularized deep networks completely fail or collapse to 0.50 AUC under scarce data. Our empirical measurement proved that claim is exaggerated.  
Model M1 achieved 0.8469 AUC at $N=10$ because pediatric pneumonia on PneumoniaMNIST is characterized by massive, distinct biological consolidations: large areas of fluid-filled alveolar sacs that create severe global optical density shifts. Even with $P/N > 10,000$, the first $3 \times 3$ convolutional layer (16 filters) quickly converges on basic threshold and spatial pooling filters that detect lung-field brightness.  
However, M1 exhibits severe variance and memorizes scanner boundaries. While it doesn't collapse to 0.50, it remains substantially inferior to transfer learning (0.8957 AUC), proving that parameter under-determination impairs generalizability even if it doesn't drop to zero."*

---

### Q2.2 (CSE HOD): Why did you freeze the entire 11.7M weights of ResNet-18 rather than fine-tuning the last residual block or classification head?
**Bulletproof Answer:**  
*"Fine-tuning requires backpropagating loss gradients through deep layers. When $N = 10$ or $N = 20$, computing gradients on 11 million weights with an unregularized cross-entropy loss causes catastrophic representation drift: the carefully calibrated features in the upper layers rapidly distort to fit the 10 training scans, destroying their generalization.  
Furthermore, by locking the network in `eval()` mode (`model.eval()`), we freeze the running mean and variance of all Batch Normalization layers to the pre-computed ImageNet population statistics. If BatchNorm were left in training mode with a batch size of 10, batch variance estimation would be severely biased and degenerate. Freezing converts ResNet-18 into a strictly deterministic, non-linear mathematical operator."*

---

### Q2.3 (CSE HOD): What specific visual primitives learned from ImageNet (trained on animals, vehicles, objects) actually transfer to pediatric chest radiographs?
**Bulletproof Answer:**  
*"Deep convolutional neural networks exhibit a well-known hierarchical feature taxonomy (Zeiler & Fergus, ECCV 2014):  
- **Layers 1 and 2** learn Gabor-like directional spatial frequency filters: edge detectors, corner boundaries, and continuous intensity gradients.  
- **Layers 3 and 4** compose these edges into geometric textures, contour curvatures, and contrast transitions.  
In a chest radiograph, the diagnostic signal consists precisely of these low- and mid-level optical primitives: the sharp contrast between black radiolucent air in healthy parenchyma versus white radiopaque consolidation in inflamed tissue, as well as the sharp delineation of rib cages and diaphragm borders. These low-level primitives transfer 100% invariant of the high-level semantic object class."*

---

### Q2.4 (Math & Computing HOD): In PCA, when $N = 10$, the sample covariance matrix has rank at most 9. Explain the algebraic consequence of this rank deficiency, and prove why there is zero data leakage.
**Bulletproof Answer:**  
*"Given an $N \times D$ centered data matrix $\mathbf{X}_c$ where $N = 10$ and $D = 512$, the sample covariance matrix is:  

$$\mathbf{\Sigma} = \frac{1}{N-1} \mathbf{X}_c^T \mathbf{X}_c \in \mathbb{R}^{512 \times 512}$$

By the rank inequality of matrix products, $\text{rank}(\mathbf{\Sigma}) \le \min(N-1, D) = 9$.  
This means that only 9 eigenvalues are strictly non-zero; the remaining 503 eigenvalues are identically zero! Thus, taking the top 8 eigenvectors ($k = 8$) extracts almost the entirety of the spanning subspace available in the training fold ($\approx 77\%$ of cumulative variance).  
**Zero Data Leakage:** A common fatal flaw in ML papers is fitting PCA on the entire dataset before splitting. We fit PCA **strictly on the $N$ training samples of each fold**. The test samples are then projected into this pre-calculated basis:  

$$\mathbf{z}_{\text{test}} = (\mathbf{x}_{\text{test}} - \boldsymbol{\mu}_{\text{train}}) \mathbf{V}_{8, \text{train}}$$

And scaled using the training fold's minimum and maximum coordinates. Test data never influences the eigenvectors or scaling bounds."*

---

### Q2.5 (CSE & Math HOD): Why is Model M2 (Classical RBF-SVM) the critical scientific control of this project rather than Model M1 (Scratch CNN)?
**Bulletproof Answer:**  
*"In experimental science, an honest benchmark requires holding all variables constant except the single independent variable under test.  
Model M1 differs from the quantum model in three separate confounding dimensions: it uses raw pixels instead of ResNet features, backpropagation instead of convex optimization, and deep convolutional capacity instead of kernel margins. Comparing quantum against M1 proves nothing about quantum mechanics.  
In contrast, **Model M2 holds every single variable identical to quantum Model M3**:  
1. It consumes the exact same 8 PCA components extracted from frozen ResNet-18.  
2. It uses the exact same Gaussian RBF kernel formulation: $k(\mathbf{u}, \mathbf{v}) = \exp(-\gamma \|\mathbf{u} - \mathbf{v}\|^2)$.  
3. It solves the exact same convex dual quadratic program:  

$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i,j=1}^N \alpha_i \alpha_j y_i y_j k(\mathbf{u}_i, \mathbf{u}_j) \quad \text{s.t. } 0 \le \alpha_i \le C, \, \sum \alpha_i y_i = 0$$

The **sole independent variable** is whether the coordinates $\mathbf{u}$ are the classical PCA features $\mathbf{x} \in \mathbb{R}^8$ (in M2) or the quantum expectation values $\mathbf{E}(\mathbf{x}) \in \mathbb{R}^{24}$ (in M3). Any delta in performance isolates the quantum feature map completely."*

---

# 3. Himmat Sirwee (2024BCS0040)
**Assigned Modules:** Quantum Circuit Architecture, 8-Qubit Register, Angle Encoding, PennyLane `lightning.qubit` Statevector Engine, 1-Layer Analytical Closed Form, Small-$s$ Taylor Duality.

---

### Q3.1 (CSE HOD): Why did you execute statevector simulation instead of running your benchmark on physical NISQ hardware (like IBM Quantum)?
**Bulletproof Answer:**  
*"Running on physical NISQ processors would introduce three confounding hardware error channels: thermal relaxation ($T_1$), phase dephasing ($T_2$), and finite measurement shot noise ($1/\sqrt{N_{\text{shots}}}$). If a physical quantum computer had achieved lower accuracy, a reviewer could argue: *'The algorithm was superior, but NISQ hardware noise destroyed it.'*  
By using PennyLane's `lightning.qubit` C++ statevector simulator, we compute the state vector $|\psi\rangle$ exactly with infinite precision ($2^8 = 256$ complex amplitudes) and zero shot noise.  
This provides an airtight scientific proof: **because our quantum execution was mathematically noiseless, the observed quantum deficit is an inherent property of the feature mapping and entanglement geometry, not hardware error.**"*

---

### Q3.2 (Math & Computing HOD): Step through the mathematical derivation of the 1-layer unentangled state. How do you obtain the exact expectation values $\langle X \rangle, \langle Y \rangle, \langle Z \rangle$?
**Bulletproof Answer:**  
*"Starting from ground state $|0\rangle = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ on a single qubit:  
Applying single-qubit rotation $R_Y(s\theta) = \exp(-i \frac{s\theta}{2} Y) = \begin{bmatrix} \cos(s\theta/2) & -\sin(s\theta/2) \\ \sin(s\theta/2) & \cos(s\theta/2) \end{bmatrix}$:  

$$|\psi_1\rangle = R_Y(s\theta)|0\rangle = \cos(s\theta/2)|0\rangle + \sin(s\theta/2)|1\rangle$$

Next, applying $R_Z(s\theta) = \exp(-i \frac{s\theta}{2} Z) = \begin{bmatrix} e^{-i s\theta/2} & 0 \\ 0 & e^{i s\theta/2} \end{bmatrix}$, and dropping an unobservable global phase of $e^{-i s\theta/2}$:  

$$|\psi\rangle = \cos(s\theta/2)|0\rangle + e^{is\theta}\sin(s\theta/2)|1\rangle$$

We now evaluate the expectation values of Pauli matrices $\langle P \rangle = \langle \psi | P | \psi \rangle$:  
1. For $Z = |0\rangle\langle 0| - |1\rangle\langle 1|$:  

$$\langle Z \rangle = \cos^2(s\theta/2) - \sin^2(s\theta/2) = \cos(s\theta)$$

2. For $X = |0\rangle\langle 1| + |1\rangle\langle 0|$:  

$$\langle X \rangle = 2\text{Re}\left(\cos(s\theta/2) \cdot e^{is\theta}\sin(s\theta/2)\right) = 2\cos(s\theta/2)\sin(s\theta/2)\cos(s\theta) = \sin(s\theta)\cos(s\theta)$$

3. For $Y = -i|0\rangle\langle 1| + i|1\rangle\langle 0|$:  

$$\langle Y \rangle = 2\text{Im}\left(\cos(s\theta/2) \cdot e^{is\theta}\sin(s\theta/2)\right) = \sin(s\theta)\sin(s\theta) = \sin^2(s\theta)$$

This yields the complete analytical Bloch vector $\mathbf{r} = (\langle X \rangle, \langle Y \rangle, \langle Z \rangle) \in \mathbb{R}^3$."*

---

### Q3.3 (Math & Computing HOD): Prove the Small-$s$ Taylor Duality. What is its consequence for claims of quantum advantage?
**Bulletproof Answer:**  
*"When the scaling parameter $s$ is tuned small ($s \to 0$), we take the Maclaurin series expansion of the closed-form Pauli expectations:  
1. $\sin(s\theta) = s\theta - \frac{(s\theta)^3}{6} + \mathcal{O}((s\theta)^5)$  
2. $\cos(s\theta) = 1 - \frac{(s\theta)^2}{2} + \mathcal{O}((s\theta)^4)$  
Substituting these into our derived expectation formulas:  

$$\langle X \rangle = \sin(s\theta)\cos(s\theta) = (s\theta)(1 - \frac{(s\theta)^2}{2}) \approx s\theta$$

$$\langle Y \rangle = \sin^2(s\theta) \approx (s\theta)^2$$

$$\langle Z \rangle = \cos(s\theta) \approx 1 - \frac{(s\theta)^2}{2}$$

**Consequence:** Notice that $\langle X \rangle$ is strictly a linear coordinate transform, while $\langle Y \rangle$ and $\langle Z \rangle$ are quadratic polynomial transforms.  
When these values are fed into an outer Gaussian RBF kernel, the overall mapping mathematically behaves as a classical quadratic-polynomial kernel!  
This proves an essential theoretical truth: **the quantum model recovers classical accuracy at small $s$ precisely because its quantum non-linearities have been linearized away to emulate a classical kernel.**"*

---

### Q3.4 (CSE HOD): What is the physical topology of your circular CNOT entangling chain, and what is its gate depth?
**Bulletproof Answer:**  
*"Our 8-qubit register employs a closed 1D nearest-neighbor circular ring topology:  

$$\text{CNOT pairs: } (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0)$$

In terms of circuit scheduling, this can be executed in two alternating hardware layers (even-odd CNOTs followed by odd-even CNOTs plus the wrap-around gate), achieving a minimal entangling depth of 3.  
Circular entanglement was specifically selected because it couples all 8 feature dimensions into a single connected graph without open edge boundary effects."*

---

### Q3.5 (CSE & Math HOD): How does the statevector dimension scale with qubit count ($2^n$), and why was 8 qubits chosen as the exact boundary for this benchmark?
**Bulletproof Answer:**  
*"The Hilbert space dimension of an $n$-qubit register is $2^n$ complex amplitudes ($2 \times 8 \times 2^n$ bytes).  
- For 4 qubits: $2^4 = 16$ amplitudes ($0.25$ KB).  
- For 8 qubits: $2^8 = 256$ amplitudes ($4$ KB).  
- For 16 qubits: $2^{16} = 65,536$ amplitudes ($1$ MB).  
- For 30 qubits: $2^{30} \approx 10^9$ amplitudes ($16$ GB RAM).  
We chose $n = 8$ qubits for two decisive reasons:  
1. **PCA Information Threshold:** 8 principal components explain $\approx 77\%$ of cumulative clinical variance from ResNet-18. Dropping to 4 qubits loses significant variance ($< 58\%$).  
2. **Benchmark Feasibility:** Our rigorous protocol required 960 controlled training runs across 15 seeds. Evaluating an 8-qubit circuit takes $\approx 1.2$ milliseconds per evaluation, enabling the execution of tens of thousands of kernel matrix inner products deterministically within feasible compute limits."*

---

# 4. Sawan Kumar (2024BCS0164)
**Assigned Modules:** Phase Wrapping Turnaround Bug, Global Fidelity Barren Plateaus, Projected Quantum Kernel Formulation, Metric Diagnostics, Refuted ZZ Correlator Hypothesis.

---

### Q4.1 (CSE & Math HOD): Explain the Phase Wrapping Turnaround Bug physically. Why does $U(\pi)^2|0\rangle = |0\rangle$, and why is this clinically disastrous?
**Bulletproof Answer:**  
*"In a 2-layer circuit with default scaling ($s = 1.00$), data coordinates $\theta \in [0, \pi]$ drive single-qubit rotations twice across the layers. The effective unitary operator is $U(s\theta) = (R_Z(s\theta) R_Y(s\theta))^2$.  
When $\theta = \pi$:  
$R_Y(\pi)$ represents a $180^\circ$ rotation around the $Y$-axis, mapping $|0\rangle \to |1\rangle$. In the second layer, another $180^\circ$ rotation maps $|1\rangle \to -|0\rangle$. Up to an unobservable global phase, the state has performed a full $360^\circ$ revolution around the Bloch sphere and returned to the ground state:  

$$U(\pi)^2|0\rangle = |0\rangle$$

As a result, the expectation value $\langle Z \rangle$ drops from $+1$ at $\theta = 0$, reaches $-1$ at $\theta = \pi/2$, and then turns around, climbing back to $+1$ at $\theta = \pi$.  
**Clinical Disaster:** A completely normal infant lung ($\theta \approx 0$) and a critically consolidated pneumonia lung ($\theta \approx \pi$) produce the **exact same quantum expectation value $\langle Z \rangle = +1$**! The metric space folds in half, blinding the SVM classifier to severe disease. Compressing scale ($s \le 0.25$) confines the rotation to a strictly monotonic quarter-arc ($[0, \pi/4]$), curing the turnaround."*

---

### Q4.2 (Math & Computing HOD): What is the mathematical difference between a global fidelity kernel and a projected quantum kernel (PQK)? Why do fidelity kernels suffer exponential concentration?
**Bulletproof Answer:**  
*"A global fidelity kernel measures the squared Hilbert space inner product between two $n$-qubit quantum states:  

$$k_{\text{fid}}(\mathbf{x}, \mathbf{x}') = |\langle \psi(\mathbf{x}) | \psi(\mathbf{x}') \rangle|^2 = \text{Tr}\left(\rho(\mathbf{x}) \rho(\mathbf{x}')\right)$$

As proven by Huang et al. (Nature Communications 2021) and McClean et al., as qubit count $n$ grows, the geometry of an $n$-qubit Hilbert space undergoes geometric concentration of measure (barren plateaus): two independently sampled states become orthogonal with probability approaching 1:  

$$\mathbb{E}[k_{\text{fid}}] = \frac{1}{2^n}, \quad \text{Var}[k_{\text{fid}}] \sim \mathcal{O}(2^{-n})$$

For $n \ge 8$, the off-diagonal entries of the Gram matrix concentrate exponentially toward 0, causing $K \approx \mathbf{I}$ (the identity matrix). The SVM margin shrinks to zero, destroying generalization.  
**The PQK Solution:** PQK does not take the global inner product. Instead, it traces out all other qubits to obtain 1-particle reduced density matrices $\rho_i = \text{Tr}_{\bar{i}}(\rho)$, measures 24 local Pauli expectations, and evaluates a classical RBF kernel in $\mathbb{R}^{24}$. Local observables bypass global concentration entirely."*

---

### Q4.3 (Math & Computing HOD): How does Projected Quantum Kernel guarantee that the resulting Gram matrix satisfies Mercer's condition for convex SVM dual optimization?
**Bulletproof Answer:**  
*"By Mercer’s Theorem (1909), a continuous symmetric kernel $k(\mathbf{u}, \mathbf{v})$ guarantees a convex quadratic programming dual with a unique global optimum if and only if the kernel is positive semi-definite (PSD):  

$$\iint f(\mathbf{u}) k(\mathbf{u}, \mathbf{v}) f(\mathbf{v}) \, d\mathbf{u} \, d\mathbf{v} \ge 0 \quad \forall f \in L^2$$

In Projected Quantum Kernels, the quantum circuit is treated strictly as a non-linear feature extractor:  

$$\phi_Q: \mathbf{x} \in \mathbb{R}^8 \mapsto \mathbf{E}(\mathbf{x}) \in \mathbb{R}^{24}$$

The kernel function evaluated between two scans is:  

$$k_{\text{PQK}}(\mathbf{x}, \mathbf{x}') = \exp\left(-\gamma \|\mathbf{E}(\mathbf{x}) - \mathbf{E}(\mathbf{x}')\|^2\right)$$

Because the outer kernel is a standard Gaussian Radial Basis Function on Euclidean space $\mathbb{R}^{24}$, and the composition of a PSD Gaussian kernel with any arbitrary deterministic coordinate mapping $\phi_Q$ is guaranteed to remain positive semi-definite (Schoenberg’s Theorem), Mercer’s condition is strictly satisfied, ensuring zero non-convex local minima."*

---

### Q4.4 (Math & Computing HOD): You observed a 34-fold collapse in expectation variance under entanglement ($0.290 \to 0.0085$). What is the exact quantum-mechanical mechanism behind this?
**Bulletproof Answer:**  
*"This collapse is an inevitable mathematical consequence of **quantum entanglement combined with partial trace projection**.  
When qubits are unentangled, the global state is a pure product state $|\psi\rangle = \bigotimes_{i=1}^n |\psi_i\rangle$. Each individual qubit's reduced density matrix is pure: $\text{Tr}(\rho_i^2) = 1$, and its Bloch vector lies on the surface of the Bloch sphere ($|\mathbf{r}| = 1$), producing a wide expectation spread across diverse patient scans ($\text{Var} = 0.290$).  
When circular CNOT entangling gates are activated, the global state becomes highly entangled. When we compute single-qubit expectation values $\langle \sigma_z^{(i)} \rangle$, we mathematically trace out the other 7 qubits: $\rho_i = \text{Tr}_{\bar{i}}(|\psi\rangle\langle\psi|)$.  
Tracing out an entangled subsystem converts pure states into **mixed states** ($\text{Tr}(\rho_i^2) < 1$). The local Bloch vector contracts deeply toward the origin ($|\mathbf{r}| \to 0$). All patient X-rays, regardless of whether they have pneumonia or clear lungs, project to nearly identical, concentrated expectation values near $(0, 0, 0)$. Measurement variance collapses 34-fold to $0.0085$, erasing clinical discriminability."*

---

### Q4.5 (CSE & Math HOD): You hypothesized that entanglement failed because 1-qubit Paulis discarded 2-qubit correlations, so you tested $\langle Z_i Z_{i+1} \rangle$. Why did adding these correlators make accuracy worse?
**Bulletproof Answer:**  
*"We rigorously tested this exact hypothesis in our ablation study (`run_zz_experiment.py`). We added 8 nearest-neighbor correlators $\langle Z_i Z_{i+1} \rangle$, expanding the projection space from 24 to 32 dimensions.  
If discarded quantum correlations were the missing key, test ROC-AUC should have rebounded. Instead, **test AUC dropped across all sample sizes**: falling by $-0.013$ at $N=10$, $-0.034$ at $N=20$, $-0.010$ at $N=50$, and $-0.014$ at $N=100$.  
Why? Because nearest-neighbor ZZ correlators on chest radiographs do not correlate with biological pathology. Instead, they introduce noisy, high-frequency oscillatory terms that increase the condition number of the Gram matrix and scatter the SVM margin hyperplane. This proved conclusively that discarding correlations is not the flaw—multi-qubit entanglement actively corrupts the metric space."*

---

# 5. Aryan Gupta (2024BCS0020)
**Assigned Modules:** 960-Run Ledger Accounting, Hyperparameter Tuning Budget Asymmetry, Interactive Streamlit Prototype Architecture, Live Test Suite Verification, Bounded Conclusions & Viva Defense Integrity.

---

### Q5.1 (CSE HOD): Walk through the exact 960-run experimental ledger. Why were 15 independent seeds necessary rather than the standard 5 seeds?
**Bulletproof Answer:**  
*"Our 960-run accounting is structured as follows:  
- **Model M1 (Scratch CNN):** 4 sample sizes ($N=10, 20, 50, 100$) $\times$ 15 seeds = **60 runs**.  
- **Model M2 (Classical RBF-SVM):** 4 $N$ values $\times$ 3 PCA dimensions (4, 6, 8) $\times$ 15 seeds = **180 runs**.  
- **Model M3 (Default PQK, $s=1$):** 4 $N$ values $\times$ 15 seeds = **60 runs**.  
- **Model M4 (11 Quantum Ablation Cells):** 4 $N$ values $\times$ 11 circuit combinations (1L/2L $\times$ Ent/Unent $\times$ 4Q/6Q/8Q) $\times$ 15 seeds = **660 runs**.  
Total: $60 + 180 + 60 + 660 = \mathbf{960\text{ completed runs}}$.  
**Why 15 seeds?** In extreme few-shot learning ($N=10$), individual seed variance is high. With 5 seeds ($df = 4$), the critical Student’s $t$-value is $t_{0.025, 4} = 2.776$, creating wide confidence intervals that obscure small model differences. With 15 seeds ($df = 14$), the critical value drops to $t_{0.025, 14} = 2.145$, tightening our confidence intervals by over 40% and providing the statistical power needed to detect Bonferroni-corrected significance."*

---

### Q5.2 (CSE & Math HOD): Explain the Tuning Budget Asymmetry. Why did quantum M3 receive a $5\times$ larger search budget, and what does this prove?
**Bulletproof Answer:**  
*"In classical Model M2, inner 5-fold cross-validation searched over 6 values of $C$ and 6 to 9 values of $\gamma$, resulting in **36 to 54 configurations evaluated per fold**.  
In quantum Model M3, because default scaling ($s = 1.00$) collapsed, we introduced an extensive post-hoc scaling grid: $s \in \{0.0625, 0.125, 0.25, 0.50, 1.00\}$. Cross-validation searched over 6 values of $C$, 9 values of $\gamma$, and 5 values of $s$, totaling **270 configurations evaluated per fold**—a **$5\times$ larger hyperparameter tuning budget**!  
**What this proves:** In machine learning, giving a model 5 times more search attempts introduces an optimistic selection bias. Yet, even with this massive search advantage heavily skewed in quantum's favor, **quantum still remained statistically detectably below classical M2 across all sample sizes**. This eliminates any objection that quantum lost due to insufficient hyperparameter tuning."*

---

### Q5.3 (CSE HOD): How did you engineer the Streamlit prototype to achieve sub-5ms inference latency on a low-resource Oracle Cloud VM without running live quantum simulations?
**Bulletproof Answer:**  
*"Running live PennyLane statevector simulations and PyTorch forward passes on an Oracle Cloud Always-Free ARM VM (limited RAM and CPU) would produce 2 to 3 seconds of latency per user click, degrading user experience.  
We engineered a decoupled, pre-compiled internal database (`prototype_db.npz`, 28.6 MB). During our offline benchmark pipeline, we computed and cached:  
1. All 624 patient test image tensors and labels.  
2. Frozen ResNet-18 latent embeddings and training fold PCA transformation matrices.  
3. Pre-calculated 24-observable Pauli expectation vectors for all circuit variants.  
4. Dual SVM support vectors, dual coefficients $\alpha_i y_i$, and intercept $b$.  
In the live Streamlit app (`app.py`), inference computes a purely vectorized matrix-vector dot product:  

$$f(\mathbf{x}) = \sum_{i \in \text{SVs}} \alpha_i y_i \exp\left(-\gamma \|\mathbf{u}_i - \mathbf{u}_{\text{test}}\|^2\right) + b$$

This executes in NumPy in **3.8 milliseconds**, consuming only 24.5 MB of RAM, and is served securely over HTTPS via Caddy."*

---

### Q5.4 (CSE & Math HOD): What are the permissible vs. impermissible scientific claims of this study?
**Bulletproof Answer:**  
*"As rigorous engineers, we enforce strict scientific boundaries:  
**What We Stand Behind (Permissible Claims):**  
1. On `PneumoniaMNIST` at $N \le 100$, tuned projected quantum kernels remain statistically detectably inferior to matched classical RBF-SVM by $-0.030$ to $-0.124$ AUC ($p < 0.001$, surviving Bonferroni).  
2. Pretrained transfer learning accounts for 100% of diagnostic performance (+0.36 AUC lift over scratch CNN).  
3. The only quantum circuit matching classical performance within $\pm 0.02$ AUC is a 1-layer unentangled product state, which is classically simulable.  
4. Circular CNOT entanglement collapses expectation variance 34-fold and degrades diagnostic metric geometry.  

**What We Explicitly Do Not Claim (Impermissible Claims):**  
1. We do *not* claim a general mathematical proof that quantum kernels can never classify images in any setting.  
2. We do *not* claim entanglement scrambles information in non-optical data structures (e.g. quantum chemistry or molecular graphs).  
3. We do *not* claim quantum M3 ties or beats classical transfer learning."*

---

### Q5.5 (CSE & Math HOD): If an evaluator says: 'Your quantum model achieved lower accuracy than classical, so your project is a failure,' how do you defend the academic merit of this work?
**Bulletproof Answer:**  
*"In science and engineering, **a hypothesis test that conclusively refutes an unverified claim is a successful investigation, not a failed project**.  
The published literature (e.g., Alagiyawanna et al., 2024/2026) claimed that projected quantum kernels provide dramatic advantages on small datasets. However, their paper compared quantum against an unregularized, artificially weak scratch CNN and did not include a matched classical transfer learning control.  
By introducing Model M2, enforcing 100% seed parity across 960 runs, proving the Phase Wrapping Turnaround Bug, demonstrating Small-$s$ Taylor Duality, and disproving the ZZ correlator hypothesis, **QKernel-Lite provides the definitive empirical refutation of ungrounded quantum advantage claims in medical image classification.**  
In engineering, discovering where a technology fails and deriving the mathematical mechanism of that failure prevents wasted compute and guides the scientific community toward genuine opportunities."*

---
