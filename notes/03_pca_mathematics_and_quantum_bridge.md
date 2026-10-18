# Module 3: Principal Component Analysis (PCA) & The Quantum Feature Bridge

**Author:** Dirac’s Army (Lead: Anshuman Biswas)  
**Target Mastery:** Classical Models & Baselines (PCA Ownership & Quantum Interface)  
**Prerequisites Assumed:** Modules 1 & 2, Linear Algebra (Eigenvalues, Eigenvectors, Covariance Matrices), Optimization with Lagrange Multipliers.

---

## 1. The Quantum Motivation: The Qubit Dimension Bottleneck

In Module 2, our frozen ResNet-18 transformed each raw chest X-ray into a 512-dimensional feature vector:
$$\mathbf{z} \in \mathbb{R}^{512}$$

Why can we not feed all 512 numbers directly into a quantum circuit?

### 1.1 The Hilbert Space Dimension Explosion
In quantum computing, an $n$-qubit register lives in a $2^n$-dimensional complex Hilbert space:
$$\mathcal{H} = (\mathbb{C}^2)^{\otimes n} \cong \mathbb{C}^{2^n}$$

If we use **Angle Encoding** (where each feature is mapped to the rotation angle of one qubit):
- Feeding 512 features requires **$n = 512$ qubits**.
- To simulate a 512-qubit statevector on a classical computer, we must store $2^{512}$ complex amplitudes:
  $$2^{512} \approx 1.34 \times 10^{154} \text{ complex numbers}$$
- For perspective, the observable universe contains approximately $10^{80}$ subatomic particles. Simulating 512 qubits is physically impossible.

### 1.2 The NISQ Simulation Regime
On modern classical simulators (such as PennyLane's `default.qubit` or `lightning.qubit`):
- $n = 4$ qubits $\implies 2^4 = 16$ amplitudes (instantaneous, microseconds).
- $n = 6$ qubits $\implies 2^6 = 64$ amplitudes (milliseconds).
- $n = 8$ qubits $\implies 2^8 = 256$ amplitudes (smooth, scalable simulation).

Therefore, we face an engineering imperative: **we must compress 512 deep ResNet features down to $n \in \{4, 6, 8\}$ numbers** while preserving as much diagnostic clinical information as mathematically possible.

The optimal linear technique for this dimensionality reduction is **Principal Component Analysis (PCA)**.

---

## 2. Mathematical Derivation of PCA from First Principles

Let our training dataset of ResNet features be represented as a design matrix:
$$\mathbf{X} \in \mathbb{R}^{N \times D}$$
Where:
- $N$ is the number of training samples ($N \in \{10, 20, 50, 100\}$).
- $D = 512$ is the original feature dimension.
- Each row $\mathbf{x}_i^T \in \mathbb{R}^D$ is the feature vector of image $i$.

---

### 2.1 Step 1: Mean Centering
Before analyzing variance, we compute the empirical sample mean vector $\boldsymbol{\mu} \in \mathbb{R}^D$:
$$\boldsymbol{\mu} = \frac{1}{N} \sum_{i=1}^N \mathbf{x}_i$$

We construct the centered data matrix $\tilde{\mathbf{X}} \in \mathbb{R}^{N \times D}$:
$$\tilde{\mathbf{X}} = \mathbf{X} - \mathbf{1}_N \boldsymbol{\mu}^T$$
*(Every feature column now has an empirical mean of exactly zero).*

---

### 2.2 Step 2: The Sample Covariance Matrix
The sample covariance matrix $\mathbf{\Sigma} \in \mathbb{R}^{D \times D}$ measures how the 512 features vary together:

$$\mathbf{\Sigma} = \frac{1}{N - 1} \tilde{\mathbf{X}}^T \tilde{\mathbf{X}}$$

#### Fundamental Mathematical Properties of $\mathbf{\Sigma}$:
1. **Symmetric:** $\mathbf{\Sigma}^T = \left(\frac{1}{N-1} \tilde{\mathbf{X}}^T \tilde{\mathbf{X}}\right)^T = \frac{1}{N-1} \tilde{\mathbf{X}}^T \tilde{\mathbf{X}} = \mathbf{\Sigma}$.
2. **Positive Semi-Definite (PSD):** For any non-zero vector $\mathbf{v} \in \mathbb{R}^D$:
   $$\mathbf{v}^T \mathbf{\Sigma} \mathbf{v} = \mathbf{v}^T \left(\frac{1}{N-1} \tilde{\mathbf{X}}^T \tilde{\mathbf{X}}\right) \mathbf{v} = \frac{1}{N-1} \|\tilde{\mathbf{X}} \mathbf{v}\|^2 \ge 0$$
   This guarantees that all eigenvalues of $\mathbf{\Sigma}$ are real and non-negative ($\lambda_k \ge 0$).

---

### 2.3 Step 3: Maximizing Projected Variance
We wish to find a unit direction vector $\mathbf{u}_1 \in \mathbb{R}^D$ ($\|\mathbf{u}_1\|^2 = \mathbf{u}_1^T \mathbf{u}_1 = 1$) such that when all centered data points $\tilde{\mathbf{x}}_i$ are projected onto $\mathbf{u}_1$, the variance of the projected points is maximized.

The projection of point $\tilde{\mathbf{x}}_i$ onto $\mathbf{u}_1$ is the scalar:
$$z_{i, 1} = \mathbf{u}_1^T \tilde{\mathbf{x}}_i$$

Since the data is zero-mean, the sample variance along direction $\mathbf{u}_1$ is:
$$\sigma_{\mathbf{u}_1}^2 = \frac{1}{N-1} \sum_{i=1}^N (z_{i, 1})^2 = \frac{1}{N-1} \sum_{i=1}^N (\mathbf{u}_1^T \tilde{\mathbf{x}}_i)(\tilde{\mathbf{x}}_i^T \mathbf{u}_1) = \mathbf{u}_1^T \left( \frac{1}{N-1} \tilde{\mathbf{X}}^T \tilde{\mathbf{X}} \right) \mathbf{u}_1 = \mathbf{u}_1^T \mathbf{\Sigma} \mathbf{u}_1$$

---

### 2.4 Step 4: Constrained Optimization via Lagrange Multipliers
We formulate the optimization problem:
$$\max_{\mathbf{u}_1} \mathbf{u}_1^T \mathbf{\Sigma} \mathbf{u}_1 \quad \text{subject to } \mathbf{u}_1^T \mathbf{u}_1 = 1$$

We construct the Lagrangian function $\mathcal{L}(\mathbf{u}_1, \lambda_1)$:
$$\mathcal{L}(\mathbf{u}_1, \lambda_1) = \mathbf{u}_1^T \mathbf{\Sigma} \mathbf{u}_1 - \lambda_1 (\mathbf{u}_1^T \mathbf{u}_1 - 1)$$

Taking the gradient with respect to $\mathbf{u}_1$ and setting it to zero:
$$\nabla_{\mathbf{u}_1} \mathcal{L} = 2 \mathbf{\Sigma} \mathbf{u}_1 - 2 \lambda_1 \mathbf{u}_1 = \mathbf{0}$$

$$\mathbf{\Sigma} \mathbf{u}_1 = \lambda_1 \mathbf{u}_1$$

### The Core Revelation:
1. The direction of maximum variance $\mathbf{u}_1$ is an **eigenvector** of the covariance matrix $\mathbf{\Sigma}$!
2. Premultiplying both sides by $\mathbf{u}_1^T$:
   $$\mathbf{u}_1^T \mathbf{\Sigma} \mathbf{u}_1 = \lambda_1 \mathbf{u}_1^T \mathbf{u}_1 = \lambda_1 (1) = \lambda_1$$
3. The variance along direction $\mathbf{u}_1$ is **precisely the eigenvalue $\lambda_1$**!

To maximize variance, we choose the eigenvector corresponding to the **largest eigenvalue** $\lambda_1$.
The second principal direction $\mathbf{u}_2$ is the eigenvector corresponding to the second largest eigenvalue $\lambda_2$ (orthogonal to $\mathbf{u}_1$), and so forth.

---

### 2.5 The Spectral Decomposition and Dimension Reduction
By the **Spectral Theorem**, any real symmetric matrix $\mathbf{\Sigma} \in \mathbb{R}^{D \times D}$ can be factored as:
$$\mathbf{\Sigma} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^T$$
Where:
- $\mathbf{V} = [\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_D] \in \mathbb{R}^{D \times D}$ is an orthonormal matrix of eigenvectors ($\mathbf{V}^T \mathbf{V} = \mathbf{I}$).
- $\mathbf{\Lambda} = \text{diag}(\lambda_1, \lambda_2, \dots, \lambda_D)$ is a diagonal matrix of eigenvalues sorted in descending order:
  $$\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_D \ge 0$$

To reduce dimensions from $D = 512$ down to $n \in \{4, 6, 8\}$, we truncate $\mathbf{V}$ to retain only the top $n$ columns:
$$\mathbf{V}_n = [\mathbf{v}_1, \dots, \mathbf{v}_n] \in \mathbb{R}^{512 \times n}$$

The reduced, $n$-dimensional representation of any centered sample $\tilde{\mathbf{x}} \in \mathbb{R}^{512}$ is computed by linear projection:
$$\mathbf{z} = \mathbf{V}_n^T \tilde{\mathbf{x}} \in \mathbb{R}^n$$

---

## 3. Equivalence to Reconstruction Error Minimization

There is a second, equally important geometric interpretation of PCA:
> **Maximizing projected variance is mathematically equivalent to minimizing the mean squared reconstruction error.**

Suppose we project $\tilde{\mathbf{x}}$ onto the $n$-dimensional subspace spanned by $\mathbf{V}_n$ and then attempt to reconstruct the original 512-dimensional vector:
$$\hat{\mathbf{x}} = \mathbf{V}_n \mathbf{z} = \mathbf{V}_n \mathbf{V}_n^T \tilde{\mathbf{x}}$$

The reconstruction error vector is the orthogonal residual:
$$\mathbf{e} = \tilde{\mathbf{x}} - \hat{\mathbf{x}} = (\mathbf{I} - \mathbf{V}_n \mathbf{V}_n^T) \tilde{\mathbf{x}}$$

By the Pythagorean theorem:
$$\|\tilde{\mathbf{x}}\|^2 = \|\hat{\mathbf{x}}\|^2 + \|\mathbf{e}\|^2$$
$$\sum_{i=1}^N \|\tilde{\mathbf{x}}_i\|^2 = \sum_{i=1}^N \|\mathbf{V}_n^T \tilde{\mathbf{x}}_i\|^2 + \sum_{i=1}^N \|\tilde{\mathbf{x}}_i - \hat{\mathbf{x}}_i\|^2$$

$$\text{Total Original Variance} = \text{Projected Variance} + \text{Reconstruction Error}$$

Because the total original variance is a fixed constant, **maximizing the variance retained automatically minimizes the reconstruction error!**

---

## 4. Why PCA Must be Fitted on Training Data ONLY (Data Leakage)

In machine learning and clinical trials, the cardinal rule of statistical validity is:
> **The test set must simulate the arrival of future, unseen patients in a hospital.**

### 4.1 The Data Leakage Catastrophe
What happens if an engineer writes:
```python
# CATASTROPHIC ERROR: Data Leakage!
pca.fit(X_all) # Fitted on combined Train + Test
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
```

1. **Mean Leakage:** The centering mean $\boldsymbol{\mu}$ now includes pixel values and features from the test patients.
2. **Subspace Leakage:** The covariance matrix $\mathbf{\Sigma}$ and its eigenvectors $\mathbf{V}_n$ are rotated so that they align with the variance of the test patients.
3. **Consequence:** The training data now implicitly contains geometric information about the test distribution. The model achieves an artificially inflated test score during cross-validation, but completely fails when deployed in a real-world clinical setting.

### 4.2 The Legitimate Engineering Protocol
The transformation parameters $\{\boldsymbol{\mu}_{\text{train}}, \mathbf{V}_{n, \text{train}}\}$ must be computed **strictly on the training split**:

```python
# CORRECT SCIENTIFIC PROTOCOL
pca = PCA(n_components=n)

# 1. Learn mu and V_n strictly on training data
pca.fit(X_train)

# 2. Transform the training data
X_train_pca = pca.transform(X_train)

# 3. Project test samples using the FROZEN training parameters:
# z_test = V_train^T * (x_test - mu_train)
X_test_pca = pca.transform(X_test)
```

---

## 5. What Information is Lost by Reducing 512 Dimensions to 8?

In your viva examination, the evaluators will almost certainly ask:
> *"512 features down to 8 is a 98.4% reduction in coordinates. What clinical information did you just throw away?"*

Here is your rigorous mathematical and physical defense:

### 5.1 The Mathematical Formulation of Lost Information
The total empirical variance across all 512 dimensions is the trace of the covariance matrix:
$$\text{Total Variance} = \text{Tr}(\mathbf{\Sigma}) = \sum_{k=1}^{512} \lambda_k$$

The **Explained Variance Ratio** of our $n = 8$ components is:
$$\text{EVR}_8 = \frac{\sum_{k=1}^8 \lambda_k}{\sum_{k=1}^{512} \lambda_k}$$

The exact mathematical information discarded is the residual projection residing in the $(512 - n)$-dimensional nullspace:
$$\mathbf{x}_{\text{lost}} = (\mathbf{I} - \mathbf{V}_8 \mathbf{V}_8^T) \tilde{\mathbf{x}}$$
The total variance lost is precisely the sum of the remaining 504 eigenvalues:
$$\text{Variance Lost} = \sum_{k=9}^{512} \lambda_k$$

### 5.2 The Radiological Meaning of the Lost Variance
1. **Dominant Signals Retained (Top 8 Components):**
   - The top 8 principal components capture the largest macro-scale geometric variations across the patient population:
     - Overall lung aeration and field expansion.
     - Large-scale diffuse radiopacity (bilateral pneumonia consolidations).
     - Broad bone density and thoracic cavity dimensions.
2. **Signals Discarded (Components 9 to 512):**
   - High-frequency micro-textures and subtle interstitial markings.
   - Minor contrast variations caused by X-ray machine exposure calibration.
   - Subtle localized opacities that span only a tiny fraction of total image variance.
3. **The Engineering Trade-Off:**
   Discarding the trailing 504 dimensions acts as a **denoising filter**, stripping away scanner-specific artifacts. However, if a rare pathological feature has small global variance across the cohort, PCA may discard it. This is the fundamental trade-off of bridging classical deep representations to near-term quantum processors.

---

## 6. Angle Rescaling: Mapping PCA Features to Quantum Rotation Angles

Once PCA produces an 8-dimensional vector $\mathbf{z} \in \mathbb{R}^8$, there is one final critical mathematical step before quantum encoding.

### 6.1 The Problem with Unbounded Linear Values
PCA features are unbounded real numbers: $z_k \in (-\infty, \infty)$ (typically distributed with mean 0 and standard deviation $\sqrt{\lambda_k}$).

However, quantum rotation gates like $R_Y(\theta)$ are periodic:
$$R_Y(\theta) = \exp\left(-i \frac{\theta}{2} Y\right) = \begin{bmatrix} \cos(\theta/2) & -\sin(\theta/2) \\ \sin(\theta/2) & \cos(\theta/2) \end{bmatrix}$$

Notice:
- $R_Y(\theta + 2\pi) = -R_Y(\theta) \implies$ density matrix is $2\pi$-periodic.
- If raw PCA values like $-14.2$ or $+38.7$ are plugged into $R_Y$, the angle wraps around the circle multiple times ($38.7 \pmod{2\pi}$).
- Two patients who had very different PCA values could accidentally map to the exact same quantum state! Monotonic distance relationships are destroyed.

### 6.2 The Scaling Transformation
To guarantee that Euclidean distances in PCA space map smoothly to state separations on the Bloch sphere without wrap-around aliasing, we rescale each PCA coordinate into a bounded interval such as $[0, \pi]$:

$$\theta_k = \pi \cdot \frac{z_k - z_{\min, k}}{z_{\max, k} - z_{\min, k}}$$

Where $z_{\min, k}$ and $z_{\max, k}$ are the minimum and maximum values of the $k$-th principal component **computed strictly on the training set**.

Now, $\theta_k \in [0, \pi]$ maps monotonically to unique rotations from the north pole $|0\rangle$ toward the south pole $|1\rangle$ on the Bloch sphere!

---

## 7. Viva Defense Checklist: What You Need to Defend

| Viva Question | Crystal-Clear Technical Defense |
| :--- | :--- |
| **Why not feed all 512 ResNet features into the quantum circuit?** | 512 features require 512 qubits. Simulating 512 qubits requires storing $2^{512} \approx 1.34 \times 10^{154}$ complex numbers, which exceeds the atoms in the universe. PCA compresses the features to $n \in \{4, 6, 8\}$ qubits to fit the simulation regime. |
| **What does PCA do in simple terms?** | It finds an orthonormal coordinate system where the axes point along the directions of maximum sample variance, diagonalizing the covariance matrix $\mathbf{\Sigma} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^T$. |
| **Why must PCA be fitted only on the training set?** | Fitting on the full dataset causes data leakage: the mean and principal axes become contaminated with test set distribution properties, producing optimistically biased, invalid results. |
| **What information is lost when dropping from 512 to 8 dimensions?** | The orthogonal projection residing in the nullspace of the top 8 eigenvectors, with lost variance equal to $\sum_{k=9}^{512} \lambda_k$. This discards high-frequency textures and scanner noise, but may also discard subtle, small-scale clinical findings. |
| **Why rescale PCA outputs to $[0, \pi]$ before the quantum gates?** | Quantum rotation gates $R_Y(\theta)$ are periodic. Unbounded PCA values cause modulo wrap-around, destroying metric distances. Rescaling to $[0, \pi]$ ensures monotonic, bijective mapping onto the Bloch sphere. |
