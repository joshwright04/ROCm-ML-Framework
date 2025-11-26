# GPU Deep Learning Framework Roadmap (HIP/C++)

This roadmap outlines the progression of small HIP/C++ programs you should write to build toward a minimal Deep Learning framework **from scratch**.  
Each bullet represents a standalone `.hip` program you should implement, test, and benchmark.

---

## **Stage 0 — Infrastructure**

### **0.1 Error Checking & Timing**
- Implement a `HIP_CHECK(expr)` macro that prints `hipGetErrorString` on failure.
- Add kernel timing using:
  - `hipEvent_t start, stop`
  - `hipEventRecord()`, `hipEventElapsedTime()`

### **0.2 Simple Build Setup**
- Use one `.hip` or `.cpp` file compiled via `hipcc`.
- Optional: Write a batch script or Makefile/CMake for easier builds.

---

## **Stage 1 — Vector Operations (1D) **

### **Vector Add)**
- `c[i] = a[i] + b[i]`
- Compare GPU vs CPU results.

### **AXPY (BLAS Level 1)**
- Operation:  
  ```
  y = α * x + y
  ```
- Kernel: `y[i] = alpha * x[i] + y[i]`.

### **Dot Product with Parallel Reduction**
1. Version A: **atomicAdd per thread** (simple).
2. Version B: **shared-memory reduction** (fast & ML-relevant).
3. Compare GPU vs CPU and benchmark timing.

### **Elementwise Unary Ops (Activations)**
Implement GPU kernels for:
- ReLU: `y[i] = max(0, x[i])`
- Sigmoid
- Tanh
- (Later) GELU, Softplus, etc.

These become the activation functions in your neural networks.

---

## **Stage 2 — Matrix Operations (2D)**

Now extend the indexing concepts to 2D tensors.

### **Matrix–Vector Multiply (GEMV)**
Assume row-major storage:

```
y[row] = Σ (A[row * n + col] * x[col])
```

Implement two kernel versions:

- **Option A:** One thread per output row.
- **Option B:** One block per row + shared-memory reduction.

### **Naive Matrix–Matrix Multiply (GEMM)**
For matrices A (M×K), B (K×N), compute C (M×N):

```
C[i, j] = Σ_k (A[i, k] * B[k, j])
```

Kernel:
- One thread per output element `(i, j)`

This is your first **linear layer**.

### **Tiled Matrix Multiply with Shared Memory**
- Use 2D thread blocks (e.g., 16×16).
- Load tiles of A and B into `__shared__` memory.
- Perform block-level partial sums.
- This teaches memory coalescing, tiling, and GPU locality.

---

## **Stage 3 — Essential ML Operations**

### **Matrix-Wide Activations**
Reuse your vector activations, treating the matrix as flat memory.

### **Row-wise Softmax**
For each row:
1. Subtract max (numerical stability)
2. Exponentiate
3. Sum the exponentials (reduction)
4. Divide

Requires:
- row-wise max reduction  
- row-wise sum reduction

### **Cross-Entropy Loss**
For predicted probabilities and target labels:
- Compute per-sample loss on GPU.
- Reduce across the batch to get the mean loss.

---

## **Stage 4 — Forward Pass for a Tiny Neural Network**

### **Dense (Fully-Connected) Layer Forward**
Implement:

```
y = W x + b
```

Components:
- GEMV or GEMM (for batch)
- Add bias vector
- Activation (optional)

### **Batch Support**
For inputs X (batch × in_features):

```
Y = X · Wᵀ + b
```

Requires:
- Matrix–matrix multiply
- Broadcasted bias add

### **Multi-Layer MLP Forward**
Combine layers:
```
h1 = relu(W1 x + b1)
y  = W2 h1 + b2
```

Test correctness by comparing GPU output vs CPU baseline.

---

## **Stage 5 — Backpropagation & Training**

### **Dense Layer Backward**
Given upstream gradient `dY`:

```
dW = dYᵀ * X     (matmul)
db = sum_rows(dY) (reduction)
dX = dY * W       (matmul)
```

Reuses all earlier matrix ops.

### **Activation Backward**
- ReLU backward:
  ```
  dX[i] = (X[i] > 0 ? dY[i] : 0)
  ```
- Sigmoid/tanh backward formulas.

### **SGD Weight Update**
On CPU or GPU:

```
W[i] -= lr * dW[i];
b[i] -= lr * db[i];
```

### **Train a Tiny Neural Network**
Use a simple dataset:
- XOR  
- Two Gaussian blobs  
- Spiral dataset  

Training loop:
1. Forward (GPU)  
2. Loss (GPU)  
3. Backward (GPU)  
4. Update (CPU or GPU)  
5. Repeat for N epochs  
6. Print loss curve  

Verify:
- Loss decreases  
- Network learns correct decision boundary  

---

## 🎉 End Result

After completing this roadmap you will have:

- A minimal deep learning engine from scratch  
- Implemented entirely in **HIP/C++**  
- With GPU acceleration for:
  - activations  
  - matrix ops  
  - softmax  
  - cross-entropy  
  - dense layers  
  - backpropagation  
- Fully capable of training a small neural network end-to-end

And you'll deeply understand what frameworks like PyTorch/TensorFlow do under the hood.
