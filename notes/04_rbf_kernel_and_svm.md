# Module 4: Support Vector Machines (SVM) & The RBF Kernel from Absolute Scratch

**Author:** Dirac’s Army (Lead: Anshuman Biswas)  
**Target Mastery:** Classical Models & Baselines (M2 Ownership: Classical RBF-SVM)  
**Prerequisites Assumed:** None! Starting from absolute visual intuition.

---

## 0. Intuitive First Principles: What is an SVM in Plain English?

Before looking at any equations, let us build the physical and visual intuition.

### 0.1 What is "Linear Separability"?
Suppose you plot patient data on a 2D sheet of paper:
- **Red circles** = Pneumonia patients
- **Blue crosses** = Healthy patients

Can you lay down a straight wooden ruler between the two groups such that **all red circles are on one side of the ruler, and all blue crosses are on the other side?**
- If **YES**: The data is called **Linearly Separable** (separable by a straight line in 2D, a flat plane in 3D, or a "hyperplane" in higher dimensions).
- If **NO**: The data is **Non-Linearly Separable**. For example, imagine all the Red circles are clustered in a ball in the center, and the Blue crosses surround them like a donut. No matter how you orient a straight ruler, you cannot separate them!

#### Why do computers care about straight lines?
Because mathematically, a straight line is the simplest, most stable decision rule possible:
$$\text{Score} = w_1 x_1 + w_2 x_2 + b$$
- If $\text{Score} > 0 \implies \text{Pneumonia}$
- If $\text{Score} < 0 \implies \text{Normal}$

It requires very little computation and avoids erratic, crazy predictions. But real-world medical data is almost never neatly separable by a simple straight line.

---

### 0.2 Why Map Points to a Higher Dimension? (The "Magic" Trick)

Suppose you have points on a simple 1D number line:
- Blue points at $-2$ and $+2$
- Red points at $-1, 0, +1$

```
Blue          Red     Red    Red          Blue
 ✕ ──────────── ○ ───── ○ ──── ○ ────────── ✕
-2            -1       0      +1           +2
```

Can you separate the Red points from the Blue points with a single cut?
**Impossible!** The red points are trapped in the middle between the blue points.

Now, let's do a simple trick: **add a second dimension $y = x^2$!**
- The point $0$ stays at $(0, 0^2) = (0, 0)$
- The points $-1$ and $+1$ move up to $(-1, 1)$ and $(1, 1)$
- The points $-2$ and $+2$ move way up to $(-2, 4)$ and $(2, 4)$

```
y-axis (height = x^2)
 4 ┬      ✕ (-2, 4)                               ✕ (2, 4)
 3 │
 2.5 ┈┈┈┈┈┈┈┈┈┈┈┈┈ [ STRAIGHT LINE CUT at y = 2.5 ] ┈┈┈┈┈┈┈┈┈┈┈┈
 2 │
 1 │                 ○ (-1, 1)        ○ (1, 1)
 0 ┴───────────────────────── ○ (0, 0) ─────────────────────── x-axis
```

Look at that graph!
In 1D, separating them was impossible. But in 2D, you can draw a **straight horizontal ruler line at $y = 2.5$**!
- Everything below the line ($y < 2.5$) is Red (Pneumonia).
- Everything above the line ($y > 2.5$) is Blue (Healthy).

> **The Core Intuition:**
> When data is tangled up in low dimensions, lifting it into a higher dimension untangles the points so a flat, straight cut can separate them!

---

### 0.3 What is a Support Vector Machine (The "Widest Road" Analogy)?

Suppose your data *is* separable by a straight line. There are **infinite** possible straight lines you could draw between the red and blue dots:
- One line might skim right next to the red dots.
- Another line might hug the blue dots.
- Which line is the best?

**The Highway Analogy:**
Imagine building a two-lane highway between Red Town and Blue Town. You want to pave the **widest possible road** between them before the road's curbs touch the nearest houses on either side!

```
     Red Town (Pneumonia)
        ○      ○
           ○  [Curb 1] ─── Touching House A (Support Vector!)
  ═══════════════════════════════════════════════════════════════
  ┈┈┈┈┈┈┈┈┈┈┈┈┈┈  CENTER LINE: Decision Boundary  ┈┈┈┈┈┈┈┈┈┈┈┈┈┈
  ═══════════════════════════════════════════════════════════════
              [Curb 2] ─── Touching House B (Support Vector!)
           ✕
        ✕      ✕
     Blue Town (Normal)
```

1. **The Center Line:** This is the classification boundary ($\mathbf{w}^T \mathbf{x} + b = 0$).
2. **The Street Width (Margin):** The distance between Curb 1 and Curb 2. An SVM searches for the line that makes this road as wide as possible.
3. **The Support Vectors:** Notice that 95% of the houses deep inside Red Town or Blue Town don't matter at all. The **only** points that determine the road are the critical borderline patients touching the curbs! Those critical boundary points are called the **Support Vectors**.

---

### 0.4 How the Kernel Trick Fits In
What if your data is non-linear?
1. An **RBF Kernel** mathematically curls and lifts your data into a high-dimensional space.
2. In that high-dimensional space, the SVM paves a **wide, straight highway**.
3. When you look at that straight highway back down in your original 2D or 8D space, it looks like a **smooth, flexible boundary** curving naturally around the pneumonia patients!


---

## 2. Mathematical Formulation of the Support Vector Machine

Let our training dataset of PCA-reduced features be:
$$\mathcal{S} = \{(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \dots, (\mathbf{x}_N, y_N)\}, \quad \mathbf{x}_i \in \mathbb{R}^n, \quad y_i \in \{-1, +1\}$$
*(In SVM literature, class labels are conventionally defined as $\pm 1$ rather than $0/1$ to simplify margin geometry).*

---

### 2.1 The Primal Optimization Problem (Hard Margin)
A linear hyperplane is defined by a normal weight vector $\mathbf{w} \in \mathbb{R}^n$ and bias $b \in \mathbb{R}$:
$$f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b = 0$$

For a linearly separable dataset, we require every training point to be correctly classified with a functional margin of at least 1:
$$y_i (\mathbf{w}^T \mathbf{x}_i + b) \ge 1, \quad \forall i \in \{1, \dots, N\}$$

The geometric perpendicular distance from any hyperplane to a support vector is $\frac{1}{\|\mathbf{w}\|}$.
The total separation margin between the positive and negative classes is:
$$\text{Margin} = \frac{2}{\|\mathbf{w}\|}$$

To **maximize the margin**, we minimize $\|\mathbf{w}\|$, or equivalently minimize $\frac{1}{2} \|\mathbf{w}\|^2$:

$$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 \quad \text{subject to } y_i (\mathbf{w}^T \mathbf{x}_i + b) \ge 1, \quad \forall i = 1, \dots, N$$

---

### 2.2 The Soft Margin Formulation (Handling Noise & Overlap)
In real-world medical data, classes may overlap slightly. We introduce non-negative slack variables $\xi_i \ge 0$ and a regularization penalty $C > 0$:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^N \xi_i$$
$$\text{subject to } y_i (\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$

- $C$ controls the trade-off: large $C$ strictly penalizes misclassifications (narrow margin, risk of overfitting); small $C$ tolerates errors to achieve a wider, more robust margin.

---

### 2.3 The Lagrangian Dual Formulation
To solve this constrained optimization problem, we construct the Lagrangian:
$$\mathcal{L}(\mathbf{w}, b, \boldsymbol{\xi}, \boldsymbol{\alpha}, \boldsymbol{\beta}) = \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^N \xi_i - \sum_{i=1}^N \alpha_i \Big[ y_i (\mathbf{w}^T \mathbf{x}_i + b) - 1 + \xi_i \Big] - \sum_{i=1}^N \beta_i \xi_i$$
Where $\alpha_i \ge 0$ and $\beta_i \ge 0$ are Lagrange multipliers.

Setting the partial derivatives with respect to the primal variables to zero:
1. $\nabla_{\mathbf{w}} \mathcal{L} = \mathbf{w} - \sum_{i=1}^N \alpha_i y_i \mathbf{x}_i = \mathbf{0} \implies \mathbf{w} = \sum_{i=1}^N \alpha_i y_i \mathbf{x}_i$
2. $\frac{\partial \mathcal{L}}{\partial b} = -\sum_{i=1}^N \alpha_i y_i = 0 \implies \sum_{i=1}^N \alpha_i y_i = 0$
3. $\frac{\partial \mathcal{L}}{\partial \xi_i} = C - \alpha_i - \beta_i = 0 \implies 0 \le \alpha_i \le C$

Substituting these relations back into the Lagrangian yields the **Wolfe Dual Problem**:

$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j (\mathbf{x}_i^T \mathbf{x}_j)$$
$$\text{subject to } 0 \le \alpha_i \le C \quad \text{and} \quad \sum_{i=1}^N \alpha_i y_i = 0$$

---

### 2.4 The Critical Observation
Look carefully at the objective function of the Dual Problem:
$$\sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j \mathbf{(x_i^T x_j)}$$

Notice:
**The feature vectors $\mathbf{x}_i$ and $\mathbf{x}_j$ do not appear anywhere in isolation!**
They enter the optimization problem **strictly as an inner product (dot product)**:
$$\langle \mathbf{x}_i, \mathbf{x}_j \rangle = \mathbf{x}_i^T \mathbf{x}_j$$

And once the dual problem is solved, the classification decision for a new patient $\mathbf{x}$ is:
$$f(\mathbf{x}) = \text{sign}\left( \sum_{i \in \text{Support Vectors}} \alpha_i y_i (\mathbf{x}_i^T \mathbf{x}) + b \right)$$

This mathematical structure makes the **Kernel Trick** possible.

---

## 3. The Kernel Trick & Mercer's Theorem

Suppose the patient data is not linearly separable in the original $n$-dimensional space $\mathbb{R}^n$.
We wish to map each point into a higher-dimensional feature space $\mathcal{H}$ via a non-linear mapping:
$$\mathbf{\Phi}: \mathbb{R}^n \longrightarrow \mathcal{H}$$

In this new space, the dot product between two transformed points is:
$$\langle \mathbf{\Phi}(\mathbf{x}_i), \mathbf{\Phi}(\mathbf{x}_j) \rangle_{\mathcal{H}}$$

### The Kernel Trick
What if we never explicitly compute $\mathbf{\Phi}(\mathbf{x})$?
What if there exists a scalar function $K(\mathbf{x}_i, \mathbf{x}_j)$ that directly computes this inner product in the low-dimensional space?

$$K(\mathbf{x}_i, \mathbf{x}_j) = \langle \mathbf{\Phi}(\mathbf{x}_i), \mathbf{\Phi}(\mathbf{x}_j) \rangle$$

### Mercer’s Theorem
A symmetric function $K: \mathbb{R}^n \times \mathbb{R}^n \to \mathbb{R}$ is a valid Mercer kernel if and only if for any set of points $\{\mathbf{x}_1, \dots, \mathbf{x}_N\}$, the resulting $N \times N$ Gram matrix $\mathbf{K}$ (where $K_{ij} = K(\mathbf{x}_i, \mathbf{x}_j)$) is **symmetric and positive semi-definite (PSD)**:
$$\mathbf{c}^T \mathbf{K} \mathbf{c} \ge 0, \quad \forall \mathbf{c} \in \mathbb{R}^N$$

When Mercer's condition holds, $K(\mathbf{x}_i, \mathbf{x}_j)$ is mathematically guaranteed to compute an inner product in some **Reproducing Kernel Hilbert Space (RKHS)** $\mathcal{H}$, even if $\mathcal{H}$ has **infinite dimensions**!

---

## 4. The Radial Basis Function (RBF / Gaussian) Kernel

The most widely used and powerful classical kernel is the **Radial Basis Function (RBF)** kernel:

$$K_{\text{RBF}}(\mathbf{x}, \mathbf{x}') = \exp\left( -\gamma \|\mathbf{x} - \mathbf{x}'\|^2 \right)$$

Where:
- $\|\mathbf{x} - \mathbf{x}'\|^2 = \sum_{k=1}^n (x_k - x'_k)^2$ is the squared Euclidean distance between two patient feature vectors.
- $\gamma > 0$ is a user-defined scaling hyperparameter.

---

### 4.1 Proof: Why the RBF Kernel Lives in an Infinite-Dimensional Space
Why is the RBF kernel so expressive? Let us expand it for 1-dimensional inputs ($x, x' \in \mathbb{R}$), with $\gamma = 1$:

$$K(x, x') = \exp\big(-(x - x')^2\big) = \exp(-x^2) \cdot \exp(-{x'}^2) \cdot \exp(2 x x')$$

Using the Taylor series expansion of $\exp(z) = \sum_{k=0}^{\infty} \frac{z^k}{k!}$:

$$\exp(2 x x') = \sum_{k=0}^{\infty} \frac{2^k}{k!} (x x')^k = \sum_{k=0}^{\infty} \left( \sqrt{\frac{2^k}{k!}} x^k \right) \left( \sqrt{\frac{2^k}{k!}} {x'}^k \right)$$

Combining this with the prefactors:
$$K(x, x') = \sum_{k=0}^{\infty} \underbrace{\left( \exp(-x^2) \sqrt{\frac{2^k}{k!}} x^k \right)}_{\phi_k(x)} \underbrace{\left( \exp(-{x'}^2) \sqrt{\frac{2^k}{k!}} {x'}^k \right)}_{\phi_k(x')}$$

$$K(x, x') = \langle \mathbf{\Phi}(x), \mathbf{\Phi}(x') \rangle$$
Where:
$$\mathbf{\Phi}(x) = \left[ \phi_0(x), \phi_1(x), \phi_2(x), \dots, \phi_{\infty}(x) \right]^T \in \mathbb{R}^{\infty}$$

### The Revelation:
The RBF kernel computes an exact dot product in an **infinite-dimensional polynomial feature space**, yet computing it takes only a single, instantaneous exponential evaluation $\exp(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2)$!

---

## 5. The Exact Role and Geometric Meaning of Hyperparameter $\gamma$

In your viva, the examiners will probe your understanding of $\gamma$:
> *"What does $\gamma$ physically do to the decision boundary, and what happens when $\gamma \to 0$ vs $\gamma \to \infty$?"*

In Gaussian probability, the standard normal density has variance $\sigma^2$:
$$\exp\left( -\frac{\|\mathbf{x} - \mathbf{x}'\|^2}{2\sigma^2} \right) \iff \gamma = \frac{1}{2\sigma^2} \iff \sigma = \frac{1}{\sqrt{2\gamma}}$$

$\sigma$ is the **length-scale (bandwidth)**: it represents the "radius of influence" of each support vector.

---

### 5.1 Case A: $\gamma \longrightarrow 0$ (Infinite Bandwidth $\sigma \to \infty$)
- When $\gamma$ is extremely small:
  $$\lim_{\gamma \to 0} \exp(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2) = \exp(0) = 1.0$$
- Every pair of patients, no matter how far apart, has a kernel similarity of $1.0$.
- The kernel matrix becomes a flat block of all 1s:
  $$\mathbf{K} \approx \begin{bmatrix} 1 & 1 & \dots & 1 \\ 1 & 1 & \dots & 1 \\ \vdots & \vdots & \ddots & \vdots \\ 1 & 1 & \dots & 1 \end{bmatrix}$$
- **Consequence:** The model cannot differentiate between distinct points. The decision boundary becomes a flat, rigid hyperplane. This causes **severe underfitting (high bias)**.

---

### 5.2 Case B: $\gamma \longrightarrow \infty$ (Zero Bandwidth $\sigma \to 0$)
- When $\gamma$ is extremely large:
  - If $\mathbf{x} = \mathbf{x}' \implies \|\mathbf{x} - \mathbf{x}'\|^2 = 0 \implies K(\mathbf{x}, \mathbf{x}) = \exp(0) = 1.0$.
  - If $\mathbf{x} \neq \mathbf{x}' \implies \|\mathbf{x} - \mathbf{x}'\|^2 > 0 \implies K(\mathbf{x}, \mathbf{x}') = \exp(-\infty) = 0.0$.
- The kernel matrix collapses into the **Identity Matrix**:
  $$\mathbf{K} \approx \mathbf{I} = \begin{bmatrix} 1 & 0 & \dots & 0 \\ 0 & 1 & \dots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \dots & 1 \end{bmatrix}$$
- **Consequence:** Every training point is an isolated Dirac delta spike. The SVM simply memorizes each training point individually by putting a tiny bubble around it. Training accuracy is 100%, but for any new test point, the kernel outputs 0. This causes **catastrophic overfitting (high variance)**.

---

### 5.3 Optimal Tuning: The "Sweet Spot"
The ideal $\gamma$ creates smooth Gaussian contours that group patients of the same class together while leaving a wide separation valley between classes:
In `scikit-learn`, the default heuristic is `gamma='scale'`:
$$\gamma_{\text{scale}} = \frac{1}{n \cdot \text{Var}(\mathbf{X})}$$
This automatically scales $\gamma$ inversely with the number of features $n$ and the empirical variance of the dataset.

---

## 6. Precomputed Kernels in Scikit-Learn: The Bridge to M3

Notice how our project plan structures both M2 and M3:
In `classical.py` and `quantum.py`, we do NOT use `SVC(kernel="rbf")` directly.
Instead, we compute the $N \times N$ matrix $\mathbf{K}$ explicitly, and pass it to:
```python
from sklearn.svm import SVC

# 1. Compute Gram matrix explicitly
K_train = compute_kernel_matrix(X_train, X_train) # Shape: (N, N)

# 2. Train SVM on precomputed matrix
clf = SVC(kernel="precomputed")
clf.fit(K_train, y_train)

# 3. Predict on test set using cross-kernel matrix
K_test = compute_kernel_matrix(X_test, X_train)   # Shape: (N_test, N)
y_pred = clf.predict(K_test)
```

### Why This Architecture is Essential:
By using `kernel="precomputed"`, **the exact same SVM optimizer, loss function, and evaluation code is used for BOTH Model M2 and Model M3!**
- For **M2**: $\mathbf{K}$ is generated by the classical RBF formula $\exp(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2)$.
- For **M3**: $\mathbf{K}$ is generated by the Projected Quantum Kernel circuit.

This ensures an **unbiased, scientifically controlled comparison**: any difference in performance between M2 and M3 is 100% attributable to the **kernel geometry itself**, not differences in optimization or classification algorithms!

---

## 7. Viva Defense Checklist: What You Need to Defend

| Viva Question | Crystal-Clear Technical Defense |
| :--- | :--- |
| **Why does an SVM work so well on 20 images when a CNN failed?** | SVM optimizes a convex quadratic objective with a unique global minimum (no vanishing gradients or local minima) and enforces margin maximization, which bounds test error independently of dimensionality. |
| **What is the Kernel Trick?** | An algebraic identity that computes the inner product $\langle \mathbf{\Phi}(\mathbf{x}), \mathbf{\Phi}(\mathbf{x}') \rangle$ in a high-dimensional Hilbert space directly using a scalar function $K(\mathbf{x}, \mathbf{x}')$ in the input space, without ever calculating coordinate representations in that space. |
| **Why is the RBF kernel called "infinite-dimensional"?** | By Taylor series expansion of $\exp(2\mathbf{x}^T \mathbf{x}')$, the Gaussian kernel corresponds to an inner product over an infinite sequence of polynomial basis functions $\phi_0, \phi_1, \dots, \phi_{\infty}$. |
| **What does $\gamma$ do, and what happens at the extremes?** | $\gamma = 1 / (2\sigma^2)$ is the inverse bandwidth of the Gaussian bell. As $\gamma \to 0$, similarity becomes 1 everywhere (underfitting). As $\gamma \to \infty$, the kernel collapses to the identity matrix $\mathbf{I}$, isolating each point in a tiny bubble (overfitting). |
| **Why compute the kernel matrix manually (`kernel="precomputed"`)?** | It standardizes the downstream classifier. Both classical RBF (M2) and quantum kernel (M3) feed into the exact same scikit-learn optimizer, isolating the kernel calculation as the only independent variable. |
