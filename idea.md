# Thesis Project Plan: Minimal Rust GPU ML Runtime with `wgpu`

## Feasibility verdict

This is reasonable and doable as a thesis project if the scope stays focused.

The realistic version is **not** a full ML framework. It is a small inference runtime that supports a limited set of tensor operations, runs those operations on the GPU through `wgpu`, compares against a CPU backend, and evaluates where GPU execution helps or hurts.

The strongest thesis contribution is:

```text
A minimal Rust tensor runtime with a wgpu backend, CPU backend, MLP inference,
benchmarks, and one or two ML-specific kernel fusion optimizations.
```

This is technically meaningful because it touches systems programming, GPU programming, tensor execution, benchmarking, and runtime optimization without requiring autograd, training, CUDA-specific code, or a full compiler.

## Goal

Build a minimal Rust-based tensor runtime that can execute selected ML inference operations on the GPU using `wgpu`, then benchmark it against a Rust CPU backend and, optionally, PyTorch.

Target user-facing API:

```rust
let logits = x
    .matmul(&w1)
    .add(&b1)
    .relu()
    .matmul(&w2)
    .add(&b2);
```

Internally, the runtime should execute the supported operations on either:

```text
CPU backend
wgpu backend
```

The final evaluation should explain when GPU execution is faster, when it is slower, and how much kernel fusion improves performance.

## Non-goals

To keep the project finishable, avoid these unless everything else is complete:

```text
training
autograd
dynamic computation graphs
large model support
convolutions
mixed precision
advanced memory planning
CUDA-specific optimization
full NumPy/PyTorch-style broadcasting
```

The first version should be inference-only.

---

## Phase 1: Learn the GPU execution model

Before thinking about ML, implement one simple GPU computation.

Concepts to understand:

```text
CPU host code prepares data
GPU buffers store data
WGSL shaders define kernels
dispatch launches GPU threads
readback copies results back to CPU
```

Target example:

```rust
let a = vec![1.0, 2.0, 3.0];
let b = vec![4.0, 5.0, 6.0];

let c = gpu_add(a, b);

assert_eq!(c, vec![5.0, 7.0, 9.0]);
```

Tools:

```text
Rust
wgpu
pollster
bytemuck
WGSL
```

Success criteria:

```text
create a wgpu device
upload two buffers
run an add shader
read the result back
verify correctness
```

---

## Phase 2: Build a minimal GPU operation runner

Create a small abstraction over `wgpu` instead of writing raw setup code for every operation.

Suggested project structure:

```text
rust_gpu/
  crates/
    tensor-core/
      src/
        lib.rs
        tensor.rs
        shape.rs
        backend.rs
    tensor-cpu/
      src/
        lib.rs
        ops.rs
    tensor-wgpu/
      src/
        lib.rs
        device.rs
        buffer.rs
        kernels.rs
        ops/
          add.rs
          relu.rs
          matmul.rs
      shaders/
        add.wgsl
        relu.wgsl
        matmul.wgsl
    examples/
      mlp.rs
    benchmarks/
      benches/
        ops.rs
        mlp.rs
```

Core types:

```rust
pub struct GpuDevice {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

pub struct GpuTensor {
    buffer: wgpu::Buffer,
    shape: Vec<usize>,
    len: usize,
}
```

Target API:

```rust
let a = GpuTensor::from_vec(&device, vec![1.0, 2.0, 3.0], &[3]);
let b = GpuTensor::from_vec(&device, vec![4.0, 5.0, 6.0], &[3]);

let c = a.add(&b);
let result = c.to_vec();
```

Success criteria:

```text
basic tensor allocation
upload from Vec<f32>
download to Vec<f32>
shape validation
one reusable operation dispatch path
```

---

## Phase 3: Implement the minimum operation set

Start with a small set of operations and only expand if needed.

Required for the MLP:

```text
add
relu
matmul
softmax, optional for final probabilities
```

Useful for testing and benchmarks:

```text
sub
mul
div
sum
mean
max
```

Recommended order:

```text
1. add
2. relu
3. matmul
4. softmax
5. optional extra elementwise ops
6. optional reductions
```

For the thesis, correctness matters more than having many operations. A small reliable op set is better than a broad incomplete one.

---

## Phase 4: Implement a CPU backend

Add a CPU backend early so every GPU operation has a simple correctness reference.

Example trait:

```rust
pub trait Backend {
    type Tensor;

    fn add(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;
    fn matmul(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;
    fn relu(x: &Self::Tensor) -> Self::Tensor;
}
```

Backends:

```text
CpuBackend
WgpuBackend
```

This enables:

```text
same model
same input
same weights
CPU result vs GPU result
CPU timing vs GPU timing
```

Success criteria:

```text
CPU and GPU results match within floating-point tolerance
tests cover shape errors
tests cover small known matmul examples
```

---

## Phase 5: Build the first model: MLP inference

Use a simple multilayer perceptron.

Architecture:

```text
input: 784
hidden: 128
output: 10
```

Forward pass:

```text
x -> matmul(w1) -> add(b1) -> relu -> matmul(w2) -> add(b2)
```

Optional:

```text
softmax for probabilities
```

Dataset:

```text
MNIST
```

The project only needs inference. Training can be done elsewhere, or weights can be loaded from a small exported file.

Success criteria:

```text
load or generate model weights
run one batch through CPU backend
run the same batch through wgpu backend
compare outputs
benchmark latency across batch sizes
```

---

## Phase 6: Benchmark individual operations

Benchmark operations before benchmarking the full model.

Elementwise sizes:

```text
1M elements
10M elements
100M elements, if memory allows
```

Matmul sizes:

```text
128x128
512x512
1024x1024
```

Softmax batch sizes:

```text
1
32
128
```

Measure:

```text
CPU time
GPU time including upload and download
GPU time excluding upload and download
kernel launch count
approximate memory traffic
```

This distinction is important because GPU kernels may be fast internally while end-to-end execution is slow due to data transfer or dispatch overhead.

---

## Phase 7: Benchmark full model inference

Benchmark the MLP forward pass.

Compare:

```text
Rust CPU backend
Rust wgpu backend
PyTorch CPU, optional
PyTorch CUDA, optional and only if available
```

Batch sizes:

```text
1
16
64
256
1024
```

Expected findings:

```text
GPU is often inefficient for batch size 1
GPU becomes more useful for larger batches
upload and download overhead can dominate
keeping tensors resident on the GPU matters
matmul dominates total inference time
```

---

## Phase 8: Add kernel fusion

Kernel fusion should be the main optimization section of the thesis.

Naive execution:

```text
matmul -> write output
add bias -> write output
relu -> write output
```

Fused execution:

```text
matmul + bias + relu -> one kernel
```

Target pattern:

```rust
x.matmul(&w).add(&b).relu()
```

The runtime can detect this pattern and dispatch a fused kernel instead of launching three separate operations.

Start with:

```text
matmul + add + relu
```

Optional extra fusion patterns:

```text
matmul + add
add + relu
```

Success criteria:

```text
same numerical result as naive execution
fewer kernel launches
less intermediate memory traffic
measurable speedup for relevant batch sizes
```

---

## Phase 9: Thesis evaluation

The evaluation should answer:

```text
Can Rust and wgpu support a minimal ML inference pipeline?
What overhead does wgpu introduce?
When is GPU execution faster than CPU execution?
When is it slower?
How much does kernel fusion improve inference performance?
What limitations remain compared with PyTorch/CUDA?
```

Useful graphs:

```text
CPU vs GPU matmul time
naive GPU vs fused GPU
batch size vs inference latency
upload/download overhead
kernel launch count before and after fusion
```

---

## Recommended MVP

If time gets tight, finish this version first:

```text
1. CPU backend
2. wgpu backend
3. add, relu, matmul
4. MLP inference with fixed weights
5. operation benchmarks
6. model benchmarks
7. one fused matmul + bias + relu kernel
8. thesis analysis
```

Everything else is optional.

## Main risks

```text
wgpu setup and synchronization can take longer than expected
GPU timing is easy to measure incorrectly
matmul performance depends heavily on tiling and memory access patterns
readback can dominate small benchmarks
full softmax and cross entropy are unnecessary if the thesis focuses on inference latency
too many operations will dilute the core contribution
```

## Final deliverables

```text
1. Rust library
2. CPU backend
3. wgpu backend
4. WGSL kernels
5. MLP inference example
6. Benchmark suite
7. Kernel fusion experiment
8. Performance analysis
9. Thesis writeup
```

## Final scope statement

Build a minimal Rust tensor runtime for GPU inference using `wgpu`, evaluate it on an MLP model, and show how kernel fusion affects performance.

That scope is realistic, Rust-centered, and strong enough for a thesis.
