# Lite Thesis Sketch

## Working Title

Building and Evaluating a Minimal Rust/WGPU Tensor Runtime for GPU-Resident Neural Network Inference

## One-Sentence Thesis

This project studies how a small Rust tensor runtime can execute neural network
operations on the GPU through `wgpu` and WGSL, and measures when GPU execution
becomes useful once tensors stay resident on the GPU instead of crossing the
CPU/GPU boundary after every operation.

## Research Question

When building a minimal tensor runtime in Rust using portable GPU compute, which
parts of the execution path dominate performance: data transfer, command
submission, kernel implementation, or tensor residency?

Secondary questions:

- How much does keeping tensors on the GPU improve multi-operation inference?
- How much does a tiled matmul kernel improve over a naive one-thread-per-output
  kernel?
- At what tensor or MLP sizes does GPU execution become competitive with a simple
  CPU reference?
- What architectural pieces are needed before higher-level optimizations such as
  fusion or reusable buffers become meaningful?

## Why This Matters

Modern machine learning workloads rely heavily on matrix multiplication and
elementwise tensor operations. Production frameworks hide most of the runtime
machinery behind high-level APIs. That is useful for users, but it makes it hard
to learn where performance actually comes from.

This project is important because it makes the full stack visible:

- Rust owns tensor metadata, validation, memory allocation, and benchmark logic.
- `wgpu` owns portable GPU setup, buffers, queues, bind groups, and pipelines.
- WGSL owns the actual GPU kernel code.
- Benchmarks separate CPU reference execution, GPU end-to-end execution, and
  GPU-resident execution.

The result is not just another tensor library. It is an experimental system that
can explain the cost model of GPU inference in small steps.

## What Exists Already

There are already strong Rust and WebGPU machine learning systems.

Burn is a full Rust deep learning framework. Its documentation describes goals
such as performance, portability, automatic kernel fusion, asynchronous
execution, memory management, automatic kernel selection, and multiple backends
including WGPU, CUDA, ROCm, LibTorch, Candle, and NdArray:

<https://burn.dev/docs/burn/index.html>

Candle is a minimalist Rust ML framework from Hugging Face. It focuses on
performance, ease of use, GPU support, PyTorch-like syntax, CUDA, optimized CPU
backends, WASM support, and real model examples:

<https://github.com/huggingface/candle>

dfdx is a Rust tensor and neural network library with CUDA support and strong
compile-time shape checking:

<https://docs.rs/dfdx>

ONNX Runtime Web has a WebGPU execution provider. Its documentation explicitly
discusses keeping tensors on the GPU through I/O binding, pre-allocated GPU
tensors, graph capture, and avoiding CPU/GPU copies:

<https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html>

`wgpu` itself is a safe, portable Rust graphics and compute API based on WebGPU.
It can target Vulkan, Metal, DirectX 12, OpenGL ES, browsers through WebGPU, and
WebGL2:

<https://wgpu.rs/>

Chrome's WebGPU compute documentation shows the same basic pipeline this project
uses: create buffers, describe shader input/output, compile shader code, create
a compute pipeline, submit commands, and read back results:

<https://developer.chrome.com/docs/capabilities/web-apis/gpu-compute>

WebGPU Fundamentals explains the workgroup model used by WGSL compute shaders:
workgroups contain parallel invocations, and `global_invocation_id` identifies
which logical element a shader invocation should process:

<https://webgpufundamentals.org/webgpu/lessons/webgpu-compute-shaders.html>

## What Is Original Here

This project is not original because it is the first tensor library in Rust. It
is not. Burn, Candle, dfdx, Tensor Frame, xnn, and ONNX Runtime Web already cover
nearby territory.

The originality is in the research shape:

1. The runtime is intentionally minimal and inspectable.

   Instead of starting from a large framework, the project builds the pieces in
   order: shape validation, CPU references, GPU tensors, add, ReLU, matmul, MLP,
   pipeline caching, resident benchmarks, and tiled kernels.

2. Every GPU operation has a CPU reference.

   This gives a correctness-first path. Performance measurements are only useful
   after the GPU result is known to match a simple CPU implementation.

3. The benchmark design separates transfer cost from resident execution.

   `gpu_e2e/*` measures upload, dispatch, wait, and readback. `gpu_resident/*`
   measures a more realistic inference path where tensors stay on the GPU.
   This directly tests the common claim that GPU acceleration is only useful
   when enough work stays on the device.

4. The project compares naive and tiled matmul inside the same runtime.

   The naive kernel is easy to understand. The tiled kernel introduces
   workgroup memory and data reuse. Keeping both makes the optimization visible
   rather than magical.

5. The implementation is portable by construction.

   The runtime uses `wgpu` and WGSL instead of CUDA-only code. On the current
   Windows machine it runs through Vulkan on an NVIDIA GeForce RTX 5070, but the
   architecture is meant to map to other `wgpu` backends later.

## Current Evidence

The runtime already supports:

- CPU add, ReLU, matmul, and fixed MLP forward.
- GPU add, ReLU, naive matmul, tiled matmul, and MLP forward.
- GPU-resident tensor-native APIs.
- Cached WGPU compute pipelines.
- Criterion benchmarks with selectable benchmark filters.

Current large GPU-resident matmul results on NVIDIA GeForce RTX 5070:

| Size | Naive Matmul | Tiled Matmul | Approximate Speedup |
| --- | ---: | ---: | ---: |
| `256x256` | `98.9 us` | `90.2 us` | `1.10x` |
| `512x512` | `328.7 us` | `223.6 us` | `1.47x` |
| `1024x1024` | `2.00 ms` | `1.28 ms` | `1.56x` |

This is an early but useful result. It shows that the tiled implementation
matters more as matrix size grows, which matches the expected memory-reuse
argument.

## Proposed Thesis Structure

### Chapter 1: Introduction

Motivate GPU acceleration for tensor operations and neural network inference.
Explain why Rust and `wgpu` are interesting: memory safety, explicit systems
programming, and portable GPU compute.

State the main research question: what does it take for a minimal Rust/WGPU
runtime to make GPU-resident inference useful?

### Chapter 2: Background

Explain:

- tensors and row-major layout
- elementwise operations
- matrix multiplication
- MLP forward pass
- CPU vs GPU execution
- WebGPU, WGPU, WGSL, buffers, bind groups, compute pipelines, workgroups
- why CPU/GPU transfer costs matter

### Chapter 3: Related Work

Compare this project with:

- Burn: full Rust deep learning framework with multiple backends and fusion.
- Candle: Rust ML framework focused on performance and practical model support.
- dfdx: Rust tensor library with compile-time shape checking.
- ONNX Runtime Web: production WebGPU execution provider with graph capture and
  GPU tensor I/O binding.
- WebGPU educational examples: useful for learning compute shaders, but not a
  Rust tensor runtime with CPU/GPU benchmark comparison.

The gap: there is room for a small, reproducible, thesis-sized runtime that
shows the internal steps and measures them directly.

### Chapter 4: Design

Describe the workspace:

- `tensor-core`: shape validation
- `tensor-cpu`: CPU reference operations
- `tensor-wgpu`: GPU context, tensors, operations, kernels
- root demo binary
- Criterion benchmark harness

Explain why the runtime avoids broadcasting, dynamic model loading, autodiff,
and training for now. The point is to isolate inference execution mechanics.

### Chapter 5: Implementation

Describe:

- `GpuContext`
- `GpuTensor`
- upload and readback
- WGSL inclusion through `include_str!`
- pipeline caching
- add, ReLU, matmul, tiled matmul
- MLP as `matmul -> add -> relu -> matmul -> add`
- benchmark selection and GPU gating

### Chapter 6: Evaluation

Measure:

- CPU reference operations
- GPU end-to-end operations
- GPU-resident operations
- naive vs tiled matmul
- fixed and generated MLP cases

The key comparisons should be:

- CPU vs `gpu_e2e`
- `gpu_e2e` vs `gpu_resident`
- naive matmul vs tiled matmul
- small MLP vs larger MLP

### Chapter 7: Discussion

Explain what the results mean:

- GPU is not automatically faster.
- Small operations are dominated by overhead.
- Transfers can hide useful GPU speed.
- Resident tensors are necessary for meaningful GPU inference.
- Kernel quality matters, especially for matmul.
- A minimal runtime makes the performance story easier to understand than a
  full framework.

### Chapter 8: Future Work

Possible next steps:

- Expanded GPU timestamp-query coverage to separate true GPU kernel time from
  host-side overhead.
- Reusable output buffers to reduce allocation overhead.
- Fused kernels such as `matmul + bias + relu`.
- Batched matmul.
- FP16 support.
- Softmax and cross-entropy.
- Loading a tiny real model.
- Comparing against Burn, Candle, or ONNX Runtime Web for selected workloads.

## Strong Thesis Claim

A good claim for the final thesis could be:

> A minimal Rust/WGPU tensor runtime can demonstrate the essential performance
> tradeoffs of GPU inference: GPU acceleration becomes compelling only when data
> remains resident on the device and when compute-heavy kernels, especially
> matrix multiplication, are implemented with memory reuse in mind.

This claim is defensible because the project can show it experimentally:

- end-to-end GPU can be slower when transfer overhead dominates
- GPU-resident benchmarks reduce transfer cost
- tiled matmul improves over naive matmul as matrix size grows
- MLP performance depends strongly on matmul and operation chaining

## Potentially Novel Contributions

For a master's thesis, the contribution does not need to be a world-first
framework. It needs to be a clear, original investigation with implementation
and evidence.

Possible contributions:

1. A minimal Rust/WGPU tensor runtime with CPU reference validation.
2. A benchmark methodology that separates end-to-end GPU cost from GPU-resident
   execution.
3. A documented comparison of naive and tiled WGSL matmul kernels.
4. A small MLP inference path that keeps tensors resident across multiple GPU
   operations.
5. A clear explanation of where overhead appears in the Rust/WGPU/WGSL stack.

## Risks

The main risk is that the project may look too small if it stops at add, ReLU,
matmul, and toy MLPs.

To avoid that, the evaluation must be strong. The thesis should not say, "I
built a tiny framework." It should say, "I built a controlled runtime to measure
specific GPU inference tradeoffs."

Another risk is comparing against mature frameworks too directly. Burn, Candle,
and ONNX Runtime are much larger systems. The thesis should use them as related
work and motivation, not as enemies to beat.

## Best Next Technical Step

Expand the GPU timestamp-query evaluation.

The runtime now has timestamp-query matmul helpers and targeted
`gpu_timestamp/*` benchmark groups. The next step is to run them on Windows and
compare those numbers against `gpu_resident/*`. That separates shader execution
time from Rust function calls, WGPU command encoding, command submission,
synchronization, allocation, and query readback.

After the first timestamp results, the next implementation step should be one
of:

- reusable output buffers
- fused `matmul + bias + relu`
- larger MLP benchmark cases

The best thesis path is probably:

```text
timestamp-query results -> reusable buffers -> fused MLP layer -> final benchmark report
```

That sequence turns the project from "it runs on the GPU" into "we can explain
and improve the cost model."
