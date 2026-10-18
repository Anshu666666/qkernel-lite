# QKernel-Lite: Projected Quantum Kernels for Data-Scarce Medical Image Classification

**Team:** Dirac’s Army | **Course:** IEG313 Quantum Computing for Engineers | **Lead builder:** Anshuman Biswas (2024BCD0008)

## 1. Project in one paragraph

QKernel-Lite asks a focused question: when only a handful of labeled medical images are available, does a projected quantum kernel (PQK) classifier do better than classical approaches? Images from the MedMNIST chest X-ray dataset (PneumoniaMNIST) are converted into compact feature vectors by a frozen pretrained CNN, reduced with PCA to a few numbers, encoded into a small simulated quantum circuit, and read out through single-qubit measurements. The resulting quantum kernel is fed to an SVM. We compare it against a small CNN trained from scratch and a classical RBF-kernel SVM on identical tiny training sets, and report the result honestly, whether the quantum model wins, ties, or loses. A Streamlit prototype lets anyone interact with the trained models live.

## 2. Scope changes from the submitted plan

The title and abstract we submitted mention TensorFlow, ChestX-ray14 and “rare disease”. The build is narrower and more realistic. We should tell the faculty this in one line during the first review so nothing looks like a mismatch.

| Submitted | Actual build | Reason |
| --- | --- | --- |
| TensorFlow + PennyLane | PennyLane + PyTorch + scikit-learn | TensorFlow Quantum is poorly maintained and hard to install |
| ChestX-ray14, rare disease | PneumoniaMNIST, with data scarcity simulated by limiting training to N = 10 to 100 images | Auto-downloads, tiny, removes days of data work |
| “Will outperform baselines” | “Evaluates whether it helps” | Quantum kernels often only match classical ones; an honest comparison is more defensible |
| Reference \[1\] arXiv:2601.03375 | Verify it on arxiv.org first. If it does not exist, cite Huang et al., “Power of data in quantum machine learning,” Nature Communications 12, 2631 (2021) | Unverified source |

## 3. Pipeline (what happens technically)

1. **Data:** PneumoniaMNIST images (28×28 grayscale, upscaled to 224×224 or 64×64 for the CNN). Fixed test set; training subsets of size N ∈ {10, 20, 50, 100}, sampled with 5 random seeds.
2. **Features:** frozen ResNet18 turns each image into a 512-dimensional vector.
3. **PCA:** reduce 512 numbers to n (6 to 8), so each number maps to one qubit. Fit on the training subset only.
4. **Scaling:** rescale each PCA output to a rotation-angle range such as \[0, π\].
5. **Quantum encoding:** one qubit per feature, RY or RX rotation by that feature (angle encoding), optionally followed by CNOT entangling layers.
6. **Projection:** measure ⟨X⟩, ⟨Y⟩, ⟨Z⟩ on every qubit, giving 3n real numbers per image. The circuit runs once per image, never on pairs.
7. **Kernel:** k(x, x′) = exp(−γ Σ over qubits and Paulis of (⟨P⟩(x) − ⟨P⟩(x′))²).
8. **Classifier:** `SVC(kernel="precomputed")` on the kernel matrix.

## 4. Models compared

| ID | Model | Purpose |
| --- | --- | --- |
| M1 | Small CNN trained from scratch | The “plain CNN” baseline from our abstract |
| M2 | Frozen ResNet18 + PCA + classical RBF-SVM | Key control: isolates the effect of the quantum kernel |
| M3 | Frozen ResNet18 + PCA + PQK-SVM | Our method |
| M4 (ablation) | M3 with different qubit counts (4, 6, 8) and with or without entanglement | One cheap ablation to discuss |

**Metrics:** ROC-AUC, balanced accuracy, F1. Report mean ± std across seeds, plotted against N. Plain accuracy is not used as a headline metric because the classes are imbalanced.

## 5. The prototype (instead of just a notebook)

A **Streamlit web app** (`app.py`) that runs locally with `streamlit run app.py`. The heavy experiments run once offline in the notebook or scripts; the app loads saved results and small saved models so it is instant during a demo.

**Tabs:**

1. **Live Predict:** pick a test X-ray (or random), choose training size N, and see predictions and confidence from M1, M2 and M3 side by side, with the true label.
2. **Under the Hood:** shows the PCA features for that image, the quantum circuit diagram, and the 3n measured expectation values as a bar chart.
3. **Kernel Explorer:** heatmap of the kernel matrix for a small sample, sorted by class, with a toggle between the classical RBF and the quantum projected kernel, plus a slider for γ. Block structure in the heatmap shows when the kernel is separating classes.
4. **Results:** the main plot of AUC versus N for all models, the ablation plot, and a short written conclusion.

This tab layout also gives each team member a natural section to present (see Section 7).

**Fallback:** if Streamlit gives trouble, Gradio is an equally quick substitute. Keep the notebook as a backup so the team always has something runnable.

## 6. Four-day schedule (Anshuman)

| Day | Goal | Output |
| --- | --- | --- |
| 1 | Learn concepts hands-on: PCA, SVM, kernels in scikit-learn; qubits, rotations, expectation values in PennyLane | Small scratch notebooks proving each concept (e.g., kernel matrix by hand for 4 points) |
| 2 | Data loader, subsampling, feature extraction, M1 and M2, evaluation function | Working baselines with metrics |
| 3 | PennyLane circuit, projected features, PQK kernel, M3, sanity checks | M3 produces sensible kernel matrix and scores |
| 4 | Run full experiments, ablation, plots, build Streamlit app, write report | Results plot, running prototype, report draft |

Sanity checks on Day 3: kernel matrix is symmetric, diagonal is exactly 1, values are not all nearly identical (that would signal concentration), and M3 beats chance on N = 100.

## 7. Virtual work division

Anshuman builds everything. Every member “owns” one module and must be able to explain and defend it afterwards. Anshuman runs a 45 to 60 minute walkthrough with each owner after the build (on Day 4 or the day after) and uses the checklists below. The owner should be able to run their module’s code, change a parameter, and explain what changed.

### Daksh Singh (2024BCS0042): Data, problem framing and evaluation

**Owns:** dataset, subsampling, splits, metrics, the “data-scarce” framing.

**Must understand and defend:**

- What PneumoniaMNIST is, its class balance, and why we use it instead of ChestX-ray14
- Why a model memorizes 20 images (overfitting), and how the train/test split is kept fair across all models
- Why we use multiple seeds and a fixed test set
- Confusion matrix, precision, recall, F1, balanced accuracy and ROC-AUC, and why plain accuracy is misleading on imbalanced data

**Likely viva questions:** Why is this “data-scarce” if you simulated it? Why AUC and not accuracy? What happens to your results if you change the seed?

### Anshuman Biswas (2024BCD0008): Classical models and baselines

**Owns:** M1 (small CNN), frozen ResNet18 feature extractor, PCA, M2 (classical RBF-SVM).

**Must understand and defend:**

- What a convolutional layer does, and why a small CNN fails on 20 images
- Transfer learning: why a frozen pretrained network gives usable features with so little data
- PCA in plain terms: directions of maximum variance, why we need it (512 features to n qubits), and why it is fitted on training data only
- The RBF kernel and the role of γ
- Why M2 is the fairest baseline for judging the quantum kernel

**Likely viva questions:** Why not train the CNN with the quantum part end to end? What information is lost by PCA to 8 dimensions? Why is M2 more important than M1?

### Himmat Sirwee (2024BCS0040): Quantum circuit and encoding

**Owns:** the PennyLane circuit, angle encoding, entangling layers, measurement of expectation values.

**Must understand and defend:**

- Qubit state vector, Bloch sphere, and why a single qubit’s state is captured by ⟨X⟩, ⟨Y⟩, ⟨Z⟩
- RX, RY, RZ, Hadamard and CNOT; how a classical number becomes a rotation angle
- Multi-qubit states, tensor products (n qubits means a 2ⁿ-dimensional state), and entanglement
- Expectation values ⟨ψ|P|ψ⟩, and what the simulator actually computes
- Reading the circuit diagram and how it ties to the IEG313 syllabus topics (gates, circuits, superposition, entanglement)

**Likely viva questions:** Why angle encoding? What does the CNOT layer change in the output? Why does simulating 8 qubits cost almost nothing but 50 would be impossible?

### Sawan Kumar (2024BCS0164): Quantum kernel theory and SVM

**Owns:** the projected kernel formula, kernel matrix construction, SVM with precomputed kernel, ablation analysis.

**Must understand and defend:**

- What a kernel is (a similarity score), what the kernel trick buys, and what a kernel matrix looks like
- How an SVM uses a precomputed kernel (max-margin idea, support vectors)
- Fidelity kernel versus projected kernel, and why the projected one is used: exponential concentration of fidelity kernels at higher qubit counts
- “Projected” means looking at reduced single-qubit states (partial trace, intuitively)
- Meaning of the ablation: what changing qubit count or entanglement does to the kernel and the scores

**Likely viva questions:** Why is the projected kernel cheaper to compute? What would the kernel matrix look like if everything concentrated? Does the quantum kernel give you something the RBF kernel cannot?

### Aryan Gupta (2024BCS0020): Integration, experiments, prototype and conclusions

**Owns:** overall architecture, experiment runner, Streamlit prototype, results analysis, report.

**Must understand and defend:** the whole pipeline end to end, how the modules connect, the experimental design, the honest interpretation of the results, and the answer to “when can a quantum kernel help?” (Huang et al.: only when data has structure that is hard to simulate classically; real medical images likely do not, so a null result is plausible). Also the limitations: simulation only, no noise, small test sets, high variance.

**Likely viva questions:** Did the quantum model win? If not, was the project a failure? What would change on real hardware?

### Shared slide and demo split

| Presenter | Segment |
| --- | --- |
| Daksh | Problem, dataset, evaluation setup |
| Anshuman | Classical baselines and feature pipeline |
| Himmat | Quantum circuit and encoding (Under the Hood tab) |
| Sawan | Kernel theory and ablation (Kernel Explorer tab) |
| Aryan | Architecture, live demo, results and conclusions |

Every member should also know the one-sentence summary of the project and the three-sentence answer to “why quantum here?”.

## 8. Deliverables

- GitHub repository or zip with a clear structure: `data.py`, `features.py`, `classical.py`, `quantum.py`, `experiments.py`, `app.py`, `results/`, `README.md`
- Runnable Streamlit prototype
- Backup notebook reproducing all results
- Results plot (AUC versus N for all models) and ablation plot
- Short report (3 to 5 pages): problem, method, results, discussion, limitations, references
- Slide deck with the presenter split above

## 9. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| PQK does not beat the classical RBF-SVM | Expected possibility. Present it as a finding and discuss why, using the concentration and data-structure arguments |
| Kernel values all near 1 or near 0 | Tune γ, reduce qubits, rescale angles, check the diagonal and symmetry |
| Results noisy at small N | More seeds (5 to 10), report mean ± std |
| Simulation slow | Stay at 8 qubits or fewer; use `lightning.qubit`; cache measured features to disk |
| Time overrun | Cut order: ablation first, then M1 fine-tuning, then extra Streamlit tabs. Never cut M2 |
| Reference \[1\] cannot be verified | Replace with Huang et al. (2021) |

## 10. Core references

&#91;1\] H.-Y. Huang et al., “Power of data in quantum machine learning,” Nature Communications 12, 2631 (2021). Origin of projected quantum kernels.

&#91;2\] PennyLane documentation and demos on quantum kernels, pennylane.ai.

&#91;3\] J. Yang et al., “MedMNIST v2: a large-scale lightweight benchmark for 2D and 3D biomedical image classification,” Scientific Data, 2023.

&#91;4\] scikit-learn documentation: PCA, SVC with precomputed kernels.

&#91;5\] Optionally, arXiv:2601.03375 as submitted, only after confirming it exists and the title matches.
