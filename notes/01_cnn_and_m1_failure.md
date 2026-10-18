# Module 1: Convolutional Neural Networks (CNNs) & Why M1 Fails on $N=20$ Images

**Author:** Dirac’s Army (Lead: Anshuman Biswas)  
**Target Mastery:** Classical Models & Baselines (M1 Ownership)  
**Prerequisites Assumed:** Multi-Layer Perceptrons (MLPs/ANNs), Backpropagation, Gradient Descent, Basic Linear Algebra.

---

## 1. Motivation: Why Fully Connected ANNs Break Down on Images

In standard Artificial Neural Networks (ANNs / MLPs), every input neuron is connected to every hidden neuron. Let us examine what happens when an image is fed to an MLP.

### 1.1 The Curse of Dimensionality & Parameter Explosion
Suppose we take an image $X$. Even for a modest medical scan of dimensions $224 \times 224$ pixels with 1 grayscale channel:
$$\text{Input dimension } d = 224 \times 224 \times 1 = 50,176 \text{ numbers.}$$

If the first hidden layer has a modest size of $h_1 = 512$ neurons, the weight matrix $W^{[1]}$ has dimensions $512 \times 50,176$:
$$\text{Parameters in Layer 1} = 512 \times 50,176 \approx 25.69 \text{ million weights!}$$

- Storing and computing gradients for 25.7 million weights in the *very first layer* is computationally wasteful.
- More critically from a statistical perspective: fitting 25.7 million parameters on a few training samples is mathematically ill-posed (infinite overfitting).

### 1.2 Destruction of Spatial Topology (2D Geometry)
An image is not an arbitrary bag of independent numbers. Its semantic content is governed by **spatial locality**:
- Pixels at $(i, j)$ and $(i+1, j)$ are strongly correlated.
- Pixels at opposite corners $(1, 1)$ and $(224, 224)$ have little direct local covariance.

When we flatten an image matrix $X \in \mathbb{R}^{H \times W}$ into a 1D vector $x \in \mathbb{R}^{H \cdot W}$, we completely discard spatial proximity. If we randomly permuted all pixels with a fixed permutation $\pi$, an MLP would achieve the exact same mathematical loss, completely oblivious to physical geometry.

### 1.3 Lack of Translation Invariance / Equivariance
Consider a patch representing a focal pneumonia opacity.
- In an MLP, if the patch appears in the top-left corner, it activates weights $W_{[\text{top-left}]}$.
- If the exact same opacity appears in the bottom-right corner, it activates a disjoint set of weights $W_{[\text{bottom-right}]}$.
- An MLP cannot share what it learned about top-left patterns with bottom-right patterns. It must re-learn the exact same visual feature at every single pixel coordinate!

This motivates the architecture of the **Convolutional Neural Network (CNN)**.

---

## 2. Mathematical Formulation of the Convolutional Layer

A CNN enforces two inductive biases derived from physics and optics:
1. **Local Connectivity (Receptive Field):** Neurons only process small local neighborhoods.
2. **Weight Sharing (Stationarity):** The same feature detector (filter) is applied across the entire visual field.

---

### 2.1 The 2D Discrete Cross-Correlation Operation
In signal processing, mathematicians distinguish between *cross-correlation* and *convolution* (which flips the kernel). Deep learning libraries (PyTorch, TensorFlow) call this operation "convolution", but mathematically implement **discrete cross-correlation**.

Given:
- Input 2D matrix (single channel) $I \in \mathbb{R}^{H \times W}$
- Kernel (filter) $K \in \mathbb{R}^{k_h \times k_w}$ (typically $3 \times 3$ or $5 \times 5$)
- Bias scalar $b \in \mathbb{R}$

The output feature map $S$ at coordinate $(i, j)$ is computed as:

$$(I * K)(i, j) + b = \sum_{m=0}^{k_h - 1} \sum_{n=0}^{k_w - 1} I(i + m, j + n) K(m, n) + b$$

#### Concrete Numerical Example ($3 \times 3$ Kernel)
Let $I$ be a patch of an X-ray and $K$ be a vertical edge detector (Sobel filter):

$$I_{\text{patch}} = \begin{bmatrix} 10 & 10 & 80 \\ 12 & 10 & 82 \\ 10 & 11 & 79 \end{bmatrix}, \quad K = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix}$$

Computing the elementwise dot product and sum:
$$\begin{aligned}
(I * K)(0, 0) &= (-1 \cdot 10) + (0 \cdot 10) + (1 \cdot 80) \\
&\quad + (-2 \cdot 12) + (0 \cdot 10) + (2 \cdot 82) \\
&\quad + (-1 \cdot 10) + (0 \cdot 11) + (1 \cdot 79) \\
&= (-10 + 0 + 80) + (-24 + 0 + 164) + (-10 + 0 + 79) \\
&= 70 + 140 + 69 = 279
\end{aligned}$$

Because there is a sharp gradient from dark ($~10$) to bright ($~80$), the filter produces a strong response ($279$). If the patch had been uniform (e.g., all pixels equal to $50$), the sum would evaluate to $0$.

---

### 2.2 Multi-Channel Formulation (Tensor Mathematics)
In real architectures:
- The input is not 1 channel; it has $C_{\text{in}}$ channels (e.g., $C_{\text{in}} = 1$ for grayscale X-rays, $C_{\text{in}} = 3$ for RGB, or $C_{\text{in}} = 64$ for intermediate hidden layers).
- We wish to detect $C_{\text{out}}$ distinct features simultaneously (e.g., horizontal edges, textures, circular shapes). Therefore, we need $C_{\text{out}}$ distinct 3D filter banks.

#### Formal Dimensions
* **Input Tensor:** $\mathcal{X} \in \mathbb{R}^{C_{\text{in}} \times H_{\text{in}} \times W_{\text{in}}}$
* **Weight Tensor (Kernels):** $\mathcal{W} \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times k_h \times k_w}$
* **Bias Vector:** $\mathbf{b} \in \mathbb{R}^{C_{\text{out}}}$
* **Output Tensor:** $\mathcal{Y} \in \mathbb{R}^{C_{\text{out}} \times H_{\text{out}} \times W_{\text{out}}}$

For the $c$-th output channel at spatial coordinate $(i, j)$:

$$\mathcal{Y}(c, i, j) = \mathbf{b}(c) + \sum_{k=0}^{C_{\text{in}} - 1} \sum_{m=0}^{k_h - 1} \sum_{n=0}^{k_w - 1} \mathcal{X}(k, i + m, j + n) \cdot \mathcal{W}(c, k, m, n)$$

Following this linear transformation, an element-wise non-linear activation function is applied:
$$\mathcal{A}(c, i, j) = \sigma\Big(\mathcal{Y}(c, i, j)\Big) \quad \text{where } \sigma(z) = \max(0, z) \text{ (ReLU)}$$

---

### 2.3 Stride, Padding, and Spatial Dimension Calculation

When applying a filter across an image, two spatial hyperparameters govern the geometry:
1. **Padding ($p$):** Adding zeros around the boundary of the input image.
   - *Why?* Prevents the spatial dimensions from shrinking too rapidly and preserves border information.
2. **Stride ($s$):** The step size with which the kernel window slides across the input grid.
   - A stride $s=1$ moves 1 pixel at a time.
   - A stride $s=2$ downsamples the image by skipping alternate pixels.

#### Spatial Dimension Formula
Given input height $H_{\text{in}}$, kernel height $k_h$, padding $p$, and stride $s$:

$$H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} - k_h + 2p}{s} \right\rfloor + 1$$

*(An identical formula holds for width $W_{\text{out}}$).*

> **Example:** Input $28 \times 28$, $k=3$, $p=1$, $s=1$:
> $$H_{\text{out}} = \left\lfloor \frac{28 - 3 + 2(1)}{1} \right\rfloor + 1 = 28$$
> Padding $p=1$ with $k=3, s=1$ is called **"same padding"** because it preserves spatial size.

---

### 2.4 Pooling Layers: Translation Invariance & Downsampling
Following convolutional layers, we frequently insert **Max Pooling** layers:
$$\mathcal{P}(c, i, j) = \max_{0 \le m < k_p, 0 \le n < k_p} \mathcal{A}(c, i \cdot s + m, j \cdot s + n)$$

- A $2 \times 2$ max pool with stride $2$ cuts both height and width in half ($H \to H/2, W \to W/2$).
- **No learnable parameters!** It purely extracts the maximum activation in each $2 \times 2$ window.
- **Invariance:** If a pneumonia opacity shifts by 1 pixel, the maximum value in that local neighborhood remains unchanged.

---

### 2.5 Feature Hierarchy: What CNNs Actually Learn
By stacking convolutional and pooling layers, the **effective receptive field** of deeper neurons grows:

```
[Layer 1]: Sees 3x3 pixels   --> Learns: Edges, high-frequency lines, dark/light contrasts
    │
[Layer 2]: Sees 7x7 pixels   --> Learns: Corners, junctions, textures, curved ribs
    │
[Layer 3]: Sees 15x15 pixels --> Learns: Anatomical contours, lung field boundaries
    │
[Layer 4]: Sees entire image --> Learns: Global patterns (hazy consolidation, bilateral opacity)
```

---

## 3. Parameter Count Analysis: CNN vs. MLP

To appreciate why CNNs are efficient, let us compute the exact number of trainable parameters for a convolutional layer:

$$\text{Trainable Parameters} = \Big(C_{\text{in}} \times k_h \times k_w \times C_{\text{out}}\Big) + C_{\text{out}} \text{ (biases)}$$

Notice what is missing from this equation: **$H_{\text{in}}$ and $W_{\text{in}}$ do not appear anywhere!**
Whether the image is $28 \times 28$ or $1024 \times 1024$, a $3 \times 3$ filter uses the exact same number of weights.

#### Comparison:
- **MLP Layer (50,176 inputs to 64 outputs):**
  $$50,176 \times 64 + 64 = 3,211,328 \text{ parameters}$$
- **Conv Layer ($1$ input channel, $64$ filters of size $3 \times 3$):**
  $$(1 \times 3 \times 3 \times 64) + 64 = 576 + 64 = 640 \text{ parameters}$$

The convolutional layer achieves a **$5,000\times$ reduction** in parameters while preserving spatial geometry!

---

## 4. Why Model M1 (Small CNN Trained From Scratch) Fails on $N=20$ Images

In our project plan, **Model M1** is defined as:
> *"A small CNN trained from scratch on the restricted training set ($N \in \{10, 20, 50, 100\}$)."*

Why is M1 destined to perform poorly (often achieving ROC-AUC $\approx 0.50$, no better than a coin flip) when $N = 20$?

---

### 4.1 Parameter-to-Data Ratio (Severe Under-Determination)
Even a very "small" CNN (e.g., 2 conv layers + 1 fully connected classification head):
1. **Conv1:** $1 \to 16$ channels, $3 \times 3 \implies (1 \times 9 \times 16) + 16 = 160$ parameters.
2. **Conv2:** $16 \to 32$ channels, $3 \times 3 \implies (16 \times 9 \times 32) + 32 = 4,640$ parameters.
3. **FC Head:** If the image was $28 \times 28$, after two $2 \times 2$ max pools, spatial size is $7 \times 7$. Flattened vector dimension = $32 \times 7 \times 7 = 1,568$.
   Connecting 1,568 to 64 hidden units:
   $$1,568 \times 64 + 64 = 100,416 \text{ parameters}.$$
4. **Output Layer:** $64 \times 2 + 2 = 130$ parameters.

$$\text{Total Parameters } P \approx 105,346.$$

Now examine the training data:
$$N = 20 \text{ images (e.g., 10 normal, 10 pneumonia).}$$

We have **105,346 tunable degrees of freedom** to fit **20 scalar targets ($y_i \in \{0, 1\}$)**!
Mathematically, this is an under-determined system of equations with an infinite number of zero-loss solutions.

---

### 4.2 Statistical Learning Theory: Empirical Risk vs. True Risk
Let $\mathcal{D}$ be the true underlying data distribution of chest X-rays.
- **True Expected Risk (Generalization Error):**
  $$\mathcal{R}(f) = \mathbb{E}_{(x, y) \sim \mathcal{D}} \big[\mathcal{L}(f(x), y)\big]$$
- **Empirical Risk (Training Error on our sample $S_N$):**
  $$\hat{\mathcal{R}}_N(f) = \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f(x_i), y_i)$$

Gradient descent optimizes $\hat{\mathcal{R}}_N(f)$ directly. By uniform convergence bounds (Vapnik-Chervonenkis / Rademacher Complexity theory):

$$\mathcal{R}(f) \le \hat{\mathcal{R}}_N(f) + \mathcal{O}\left(\sqrt{\frac{\text{VCdim}(\mathcal{F}) + \ln(1/\delta)}{N}}\right)$$

Where:
- $\text{VCdim}(\mathcal{F})$ is proportional to the parameter count $P \approx 10^5$.
- $N = 20$.
- The generalization gap term $\sqrt{\frac{P}{N}} = \sqrt{\frac{105,346}{20}} \approx \sqrt{5267} \approx 72.5$ is astronomically large!

The bound tells us that an empirical training loss of $\hat{\mathcal{R}}_N(f) = 0$ provides **zero theoretical guarantee** that the true test risk $\mathcal{R}(f)$ will be small.

---

### 4.3 Spurious Correlations and High-Frequency Memorization
In natural chest X-rays, each image contains millions of irrelevant high-frequency signals:
- Slight contrast variations in the scanner calibration.
- Small text labels (e.g., "L" or "R" markers stamped by the radiologist).
- Differences in patient rib-cage size or body habitus.

Because $N=20$, by pure random chance, all 10 pneumonia images might have slightly brighter background borders or a minor artifact in row 3.
Because the CNN has 105,000 weights, gradient descent will rapidly latch onto these **spurious correlations** rather than learning the subtle, low-contrast alveolar infiltrates in the lung parenchyma.

The model achieves **100% training accuracy in 15 epochs**, but when evaluated on an unseen test set:
- The spurious artifacts are no longer correlated with the disease.
- The model's predictions degrade to random guesses.

---

### 4.4 The Role of M1 in Our Project
Why do we include M1 if we know it fails?
**In scientific research, M1 serves as the negative control.**
It proves to our evaluators and professors that simply applying deep learning "from scratch" to a data-scarce medical domain fails catastrophically. It justifies the entire rationale for:
1. Using **Transfer Learning** (frozen pretrained features).
2. Projecting to low dimensions via **PCA**.
3. Exploring **Kernel Methods** (Classical RBF vs. Projected Quantum Kernels).

---

## 5. Summary Checklist: What You Need to Defend in Viva

| Question | Short, Defensible Technical Answer |
| :--- | :--- |
| **What does a convolutional layer do?** | It computes a parameterized 2D discrete cross-correlation between sliding kernel banks and input feature maps, enforcing local connectivity and weight sharing to capture spatial patterns translationally. |
| **Why not use a standard MLP?** | MLPs discard spatial topology by flattening, suffer from parameter explosion ($O(H \cdot W \cdot h_1)$), and lack translation equivariance. |
| **Why does a small CNN fail on 20 images?** | Overparameterization (~10⁵ parameters vs 20 targets). In this under-determined regime, empirical risk minimization risks memorizing spurious artifacts. However, our empirical runs showed that M1 does NOT collapse to 0.5 chance: it reaches ~0.86 ROC-AUC at N=10 and N=20 because early filters capture coarse global lung contrast and macroscopic opacities. |
| **What is the purpose of M1 in the benchmark?** | It is our baseline control verifying the performance of end-to-end training from scratch in extreme data scarcity, serving as the weak baseline to compare against transfer learning and kernel machines. |

---

## 6. Deep Dive FAQ: Clarifying the Building Blocks

### 6.1 What is a Kernel (Filter)?
A kernel is simply a tiny matrix of weights (typically 3×3 or 5×5). Think of it as a **pattern template** or **stencil**.
- When placed over a patch of pixels, you compute the element-wise product and sum them up.
- If the pixel patch matches the pattern of the filter (e.g., dark on the left, bright on the right), the sum produces a large positive value (the filter "detects" the pattern).
- If the patch does not match (e.g., flat, uniform gray), the positive and negative weights cancel out, yielding near zero.
- In classical computer vision, humans manually designed these numbers (e.g., Sobel filters for edge detection). In a CNN, these weights start as random numbers and are **learned automatically** via gradient descent and backpropagation.

### 6.2 What are Channels, $C_{\text{in}}$, and $C_{\text{out}}$?
- **Channels** represent the depth dimension of an image or feature map:
  - A grayscale X-ray has **1 channel** (only intensity/brightness at each pixel). Thus, $C_{\text{in}} = 1$.
  - A color photo has **3 channels** (Red, Green, Blue stacked). Thus, $C_{\text{in}} = 3$.
- **$C_{\text{in}}$ (Input Channels):** The number of 2D slices entering the layer.
- **$C_{\text{out}}$ (Output Channels):** The number of different features we want to detect simultaneously in that layer.
  - If we want the layer to detect 16 different patterns (e.g., vertical edges, horizontal edges, dark spots, corners), we create 16 distinct 3D filters.
  - Each filter processes the input and produces its own 2D output map.
  - Stacking these 16 maps together yields an output tensor with depth $C_{\text{out}} = 16$.
  - For the next convolutional layer, this $C_{\text{out}} = 16$ becomes its input depth $C_{\text{in}} = 16$.

### 6.3 Why Does the Kernel Window Slide?
1. **Weight Sharing:** Instead of having a different set of weights for every pixel coordinate, one tiny 3×3 filter (9 weights) scans across all 50,000 pixels.
2. **Translation Equivariance:** A lung opacity or fracture has the exact same visual properties whether it appears in the upper lobe or lower lobe. Sliding the filter ensures that if the pattern shifts, the detection response shifts by the exact same amount.

### 6.4 What is a "Connected Classification Head"?
A complete CNN is split into two functional halves:
1. **The Feature Extractor (Backbone):** The sequence of convolutional and pooling layers. Their sole purpose is spatial vision—processing raw pixels and progressively distilling them into a compact vector of high-level semantic descriptors (e.g., 512 numbers).
2. **The Classification Head:** Once the spatial feature maps are flattened into a 1D vector, we attach standard Dense (Fully Connected) layers ending in a final output neuron (with Sigmoid/Softmax).
   - This dense section at the end is called the **"Classification Head"**.
   - It takes the 512 extracted features and learns the final decision boundary: $\hat{y} = \sigma(W_{\text{head}} f + b_{\text{head}})$.

### 6.5 What is a Pooling Layer?
Pooling is a downsampling operation that reduces spatial dimensions while retaining the most salient features.
- In **Max Pooling** with a 2×2 window and stride 2:
  - The layer looks at non-overlapping 2×2 squares of pixels.
  - It picks the **maximum value** in that square and discards the other three.
  - It cuts both height and width in half (reducing total pixel count by 75%).
- **Crucial properties:**
  - Has **zero learnable parameters** (it is purely a hardcoded $\max$ function).
  - Provides **translation invariance**: if a feature moves slightly (by 1 pixel) within that 2×2 square, the maximum value remains unchanged.
  - Dramatically lowers the memory and compute required for subsequent layers.

### 6.6 What is ROC-AUC and Why is it Essential?
- In medical imaging, data is often heavily imbalanced (e.g., 95 healthy scans, 5 disease scans).
- A trivial model predicting "healthy" for all cases achieves **95% accuracy** while missing every sick patient.
- Neural networks output a continuous probability $p \in [0, 1]$. To classify, we compare $p$ against a threshold $T$:
  - If $p \ge T \implies \text{Pneumonia}$.
  - If $p < T \implies \text{Normal}$.
- Changing $T$ creates a trade-off between **True Positive Rate (Sensitivity)** and **False Positive Rate (1 - Specificity)**.
- The **ROC Curve** plots True Positive Rate vs. False Positive Rate as $T$ varies from $1.0$ down to $0.0$.
- **AUC (Area Under the Curve):**
  - Measures the entire two-dimensional area under the ROC curve.
  - An AUC of $0.50$ represents a random coin flip.
  - An AUC of $1.00$ represents perfect separation.
  - **Statistical Meaning:** AUC equals the exact probability that the model ranks a randomly chosen positive case higher than a randomly chosen negative case. It is completely independent of the classification threshold and invariant to class imbalance.

