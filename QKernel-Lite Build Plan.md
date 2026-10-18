# QKernel-Lite: Build Plan

Oct 18, 2026 · @Anshuman Biswas

This is the build spec for the plan you submitted: same pipeline, same four models, with ten traps fixed in advance and a pass/fail check after every stage. Build one stage at a time and do not start the next until the check passes.

## 1. Traps to know before writing any code

| # | Trap | Why it bites | What this plan does |
| --- | --- | --- | --- |
| 1 | Circuits without entanglement make the quantum kernel classical | If each qubit is encoded from its own feature only, its ⟨X⟩, ⟨Y⟩, ⟨Z⟩ depend only on that feature (for RY alone: ⟨X⟩ = sin θ, ⟨Y⟩ = 0, ⟨Z⟩ = cos θ). The projected kernel is then an RBF kernel on sin/cos of your features, so a tie with M2 is guaranteed, and ⟨Y⟩ is a dead feature | Main M3 uses RY then RZ encoding (all three Paulis non-trivial) plus a CNOT ring over 2 layers. The no-entanglement version is the ablation, and a unit test proves the equivalence |
| 2 | M2 and M3 see different inputs | If M2 gets raw PCA values and M3 gets angle-scaled ones, you are comparing preprocessing, not kernels | Both use the same PCA outputs scaled to \[0, π\] |
| 3 | Tuning leakage or unequal tuning | Picking γ or C on the test set inflates scores; tuning only one model biases the comparison | Same grid and same cross-validation for M2 and M3, run only inside the training subset. This is the one addition to the submitted plan, and it needs no new theory |
| 4 | PCA from a tiny sample | With N = 10, centered data has rank at most 9, so the 8th component is mostly noise | Config flag `pca_fit_on`: subset (default, as submitted) or pool (all unlabeled training images, no labels used, so no leakage). Cap components at N − 1. Run pool as a check if time allows |
| 5 | Test values outside \[0, π\] | A min-max scaler fit on 10 images will not cover all test values | Fit the scaler on the training subset only, then clip test values to \[0, π\] |
| 6 | Dead kernel | Too many layers or qubits, or a bad γ, make all off-diagonal entries nearly equal and the SVM predicts one class | Log off-diagonal mean and std plus the smallest eigenvalue on every run. Treat std below about 0.01 as an alarm (a rule of thumb, not a theorem) |
| 7 | PneumoniaMNIST is easy and noisy | Classical features may already score high at small N, so M2 versus M3 differences can be small next to seed variance. I believe the official test split also differs somewhat from train, so test AUC can sit below validation AUC | Every model uses the same subsets per seed (paired design). Report the per-seed difference M3 − M2 with a 95% confidence interval. Start with 5 seeds, scale to 10 to 20 once everything works |
| 8 | Class imbalance | Train is roughly three quarters pneumonia and test roughly 62% pneumonia (verify with code, do not trust my numbers) | Training subsets are stratified 50/50. Report AUC, balanced accuracy and F1, and print test class counts |
| 9 | M1 may not collapse to AUC 0.5 | The study notes promise this. A tiny CNN can pick up easy cues such as overall brightness and score above chance | Report whatever M1 gets and call it the weak baseline, not a guaranteed failure |
| 10 | Overclaims in the study notes | Faculty who know quantum computing will probe them | See section 7 |

## 2. Frozen experiment spec

Fix these before coding so the AI cannot drift.

| Item | Value |
| --- | --- |
| Dataset | PneumoniaMNIST via `medmnist`, official train/val/test splits. Use size 224 if your installed version supports the `size` argument, otherwise 64 upsampled. Check the download size first |
| Training pool | Official train split. Subsets are drawn from it, stratified 50/50, N in {10, 20, 50, 100} |
| Test set | Full official test split, fixed, never used for tuning |
| Seeds | 0 to 4 at first (a seed fixes the subset and all randomness), extend later |
| Features | ResNet18 with ImageNet weights, eval mode, no gradients, grayscale repeated to 3 channels, ImageNet mean/std normalization, 512 numbers from global average pooling, cached to disk |
| Reduction | PCA to n components, n = 8 for the main run, fit on the training subset |
| Scaling | Min-max to \[0, π\], fit on the training subset, test clipped |
| M1 | Small CNN trained from scratch on raw 28×28 (or 64×64) images: conv 1→16 (3×3), ReLU, max-pool; conv 16→32 (3×3), ReLU, max-pool; flatten; linear →64, ReLU; linear →1. Adam, learning rate 1e-3, weight decay 1e-4, fixed 60 epochs, batch size min(16, N), no early stopping on test |
| M2 | RBF kernel on the scaled PCA features, SVC with a precomputed kernel |
| M3 | Projected quantum kernel on the same features, same SVC |
| Circuit | For each of L = 2 layers: RY(θᵢ) then RZ(θᵢ) on every qubit i, then CNOT(i, i+1 mod n) around a ring. Measure ⟨X⟩, ⟨Y⟩, ⟨Z⟩ on every qubit, giving 3n numbers. A reasonable design, not the only one |
| Kernel | k(x, x′) = exp(−γ · sum of squared differences of the 3n expectation values) |
| M4 ablations | Qubits 4, 6, 8. Entanglement on or off. Layers 1 or 2 if time allows |
| Tuning | γ in base × {0.1, 0.3, 1, 3, 10}, C in {0.1, 1, 10, 100}. Stratified k-fold inside the training subset (k = 3 for N up to 20, otherwise 5). Base γ = 1 / (d × variance of the features), with d = n for M2 and d = 3n for M3 |
| Metrics | ROC-AUC from `decision_function`, balanced accuracy, F1. Plus kernel diagnostics |

## 3. Stages with pass/fail checks

Give each stage to your AI coding agent separately. Ask it to write the checks as tests first, run them yourself, and commit after each pass.

### Stage 0: Setup (about 1 hour)

- Create a Python 3.11 virtual environment (avoid brand-new Python versions until the packages confirm support). Install torch, torchvision, medmnist, pennylane, scikit-learn, pandas, matplotlib, streamlit, pytest.
- Folder layout: `config.py`, `data.py`, `features.py`, `preprocess.py`, `classical.py`, `quantum.py`, `evaluate.py`, `run_experiments.py`, `app.py`, `tests/`, `cache/`, `results/`, `notes/`.
- Run `git init`, and save versions in `requirements.txt` once it works.
- **Pass when:** one smoke script imports everything, runs a 2-qubit PennyLane circuit, downloads PneumoniaMNIST and loads the ResNet18 weights. These downloads need internet. If versions conflict, pin whatever combination works.

### Stage 1: Data

- Build the loader and a deterministic, stratified subset sampler.
- **Pass when:** it prints split sizes and class counts, the same (seed, N) returns identical indices twice, every subset is 50/50, and it saves an image grid of 8 samples that you inspect by eye.

### Stage 2: ResNet18 features

- Extract 512-number features for train, val and test once, and cache them.
- **Pass when:** the cache has shape (images, 512) with no NaNs, a second run loads from cache, and a logistic regression trained on 200 training images scores clearly above chance on the test set. If it scores near 0.5, the preprocessing is wrong (usual causes: missing normalization, or the model left in training mode).

### Stage 3: PCA and angle scaling

- Implement PCA plus min-max scaling plus clipping, with the `pca_fit_on` flag.
- **Pass when:** the scaled training features span exactly \[0, π\], test values are clipped, components are capped at N − 1, explained variance prints, and changing the test data does not change the PCA components.

### Stage 4: Classical models (M1 and M2)

- M2: compute the RBF kernel matrix yourself and fit SVC with a precomputed kernel. Add the tuning function from section 2.
- M1: the CNN from section 2.
- **Pass when:** your manual RBF kernel gives the same predictions as scikit-learn’s built-in RBF SVC at the same γ and C, the tuning function cannot receive test arrays at all, M1 training loss decreases, and M1 results are reproducible for a fixed seed.

### Stage 5: Quantum kernel (M3)

- Build the circuit once as a PennyLane QNode that returns all 3n expectation values. Run it once per image, cache by (seed, N, qubits, entanglement, layers), then build the kernel from the cached numbers.
- **Pass when all of these hold:**
  1. A single qubit with RY(θ) gives ⟨Z⟩ = cos θ, ⟨X⟩ = sin θ, ⟨Y⟩ = 0.
  2. With entanglement off and 1 layer, RY(θ) then RZ(θ) gives ⟨X⟩ = sin θ cos θ, ⟨Y⟩ = sin² θ, ⟨Z⟩ = cos θ. This proves trap 1 in code.
  3. The kernel matrix is symmetric, has diagonal exactly 1, and its smallest eigenvalue is at least −1e-8.
  4. The test-versus-train kernel has shape (test size, train size).
  5. Off-diagonal std is logged and above the alarm level at n = 8.
  6. Computing expectation values for 100 train plus 624 test images at 8 qubits takes minutes, not hours. If slower, use parameter broadcasting or `lightning.qubit`, or drop to 6 qubits.

### Stage 6: Experiment runner

- Loop over seeds, N and models, and append one row per run to `results.csv`: model, N, seed, qubits, entanglement, layers, AUC, balanced accuracy, F1, chosen γ and C, kernel diagnostics, runtime. Skip rows already done so a crash costs nothing.
- **Pass when:** a mini-grid (N = 20, seeds 0 and 1, all models) finishes end to end, a rerun skips finished rows, and plots render. Measure the mini-grid runtime and extrapolate before launching the full grid.

### Stage 7: Streamlit prototype

- Tabs as in the plan doc: Live Predict, Under the Hood, Kernel Explorer, Results.
- It loads cached features, cached expectation values and `results.csv`. It fits only the small SVMs live. Save M1 weights once per N for seed 0 instead of training in the app.
- **Pass when:** every interaction answers in a few seconds from a cold start with no internet, and a screenshot set exists as a backup for the demo.

### Stage 8: Figures and report

- AUC versus N for M1, M2, M3 with confidence intervals.
- Paired difference M3 − M2 versus N.
- Ablations: qubits, entanglement on versus off.
- Kernel off-diagonal std versus qubit count (already logged, so almost free, and it directly supports the concentration discussion).
- Kernel heatmaps, M2 versus M3, at N = 50.

## 4. Schedule (two working days plus buffer)

| When | Work | Gate |
| --- | --- | --- |
| Tonight, 1 to 2 h | Stages 0 and 1 | Smoke script and data checks pass |
| Day 1 morning, 3 to 4 h | Stages 2 to 4 | Classical equivalence test passes |
| Day 1 afternoon, 3 h | Stage 5 | All six quantum checks pass |
| Day 1 night | Stage 6 mini-grid, then launch full grid overnight | Mini-grid passes first |
| Day 2 morning | Inspect results, fix problems, run ablations, Stage 7 | App runs offline |
| Day 2 afternoon | Stage 8, report, slides | Figures final |
| Buffer, half a day | Teammate walkthroughs and rehearsal | Everyone passes their sign-off |

## 5. What teammates do in parallel

| Member | Learns | Checks in the finished code | Sign-off |
| --- | --- | --- | --- |
| Daksh | Dataset, overfitting, metrics | Stage 1 class counts, metric code in Stage 6 | Explains class counts, why AUC, what a seed changes |
| Anshuman | Classical pipeline (your notes) | Stages 2 to 4 | Explains ResNet features, PCA, RBF, why M2 is the control |
| Himmat | Qubits, gates, expectation values | Stage 5 checks 1 and 2 | Changes qubits or entanglement and predicts the effect |
| Sawan | Kernels, SVM, concentration | Stage 4 equivalence test, Stage 5 checks 3 and 5 | Explains diagonal, PSD, γ and the diagnostics |
| Aryan | Pipeline, experiments, app | Stages 6 and 7 | Reruns the mini-grid and explains each column of `results.csv` |

## 6. Vibecoding without losing control

- Paste this doc and the relevant study note into the agent. For each stage say: implement exactly this spec, write the pass/fail checks as tests first, list any deviation.
- Run the tests yourself. Do not accept “it works” from the agent.
- One stage per commit, so you can roll back.
- Search the code for the test arrays inside any tuning function.
- Ask the agent to explain any line you cannot read. That is your viva preparation.

## 7. Claims that are safe in the viva

Several statements in the generated study notes are stronger than the evidence supports.

| Avoid | Say instead |
| --- | --- |
| “M1 will collapse to AUC 0.5” | “M1 is the weak baseline. With 10 to 100 images it is unstable, and we report what we measured” |
| “Chest X-rays may contain entanglement-like structure” or “classical kernels need thousands of patients where quantum needs 20” | “Huang et al. show an advantage is possible only for data with structure that is hard to learn classically. We test empirically whether any appears. A tie is the expected outcome” |
| “PQK avoids exponential concentration” | “Projected kernels are designed to mitigate the concentration of fidelity kernels. We log kernel spread versus qubit count to check. To my knowledge later work shows projected kernels can still concentrate for expressive circuits” |
| Quantum MRI, quantum sensors, microsecond 100-qubit runs | Leave out, or call it a possible extension outside our scope. This project is a noiseless simulation |
| “A tie proves the quantum model works” | “A tie shows the pipeline is sound and the quantum kernel is not worse here. We claim no advantage” |
| “Almost all QML papers cheat” | Do not say it |

## 8. If things go wrong

- **Circuits too slow:** fewer qubits (6), `lightning.qubit`, parameter broadcasting, cache everything.
- **Kernel nearly constant:** reduce layers, reduce γ, check angle scaling.
- **M3 ties M2 everywhere:** that is a valid result. The entanglement ablation plus the diagnostics explain it.
- **Out of time, cut in this order:** extra seeds, qubit ablation, extra app tabs, M1 tuning. Never cut M2, the paired comparison, or the stage checks.
- **Reference \[1\] (arXiv:2601.03375)** is still unverified. Check it on arxiv.org, and otherwise cite Huang et al., Nature Communications 12, 2631 (2021).
