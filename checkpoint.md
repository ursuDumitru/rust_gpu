# Checkpoint: Minimal wgpu Tensor Runtime

Last updated: 2026-05-07

## Current State

The project is now a small Rust `wgpu` runtime that can run several tensor operations on a real GPU:

```text
add:  [1.0, 2.0, 3.0] + [4.0, 5.0, 6.0] = [5.0, 7.0, 9.0]
relu: [-2.0, 0.0, 3.5] -> [0.0, 0.0, 3.5]
matmul: [2, 3] @ [3, 2] -> [2, 2]
mlp: matmul -> add -> relu -> matmul -> add
```

The latest Windows-native validation succeeded on:

```text
NVIDIA GeForce RTX 5070 (Vulkan, DiscreteGpu)
```

This matters because the project is no longer only compiling or running CPU reference code. It is creating GPU buffers, reusing cached compute pipelines, dispatching WGSL compute shaders, reading results back, and comparing them against CPU results.

## Workspace File Map

```text
src/main.rs                         root smoke-test demo binary

crates/tensor-core/src/shape.rs      pure shape and compatibility validation
crates/tensor-cpu/src/ops.rs         CPU reference add, relu, matmul
crates/tensor-cpu/src/mlp.rs         CPU MLP forward pass
crates/tensor-wgpu/src/context.rs    reusable wgpu instance/adapter/device/queue setup
crates/tensor-wgpu/src/kernels.rs    cached bind group layouts and compute pipelines
crates/tensor-wgpu/src/tensor.rs     GpuTensor buffer, shape, upload, output allocation, readback
crates/tensor-wgpu/src/mlp.rs        GPU-resident MLP forward pass
crates/tensor-wgpu/src/ops/          split GPU operation dispatch
crates/tensor-wgpu/shaders/          WGSL compute shaders
```

The root package remains named `rust_gpu`, so `cargo run` still runs the demo. Reusable library code now lives in workspace crates:

```text
tensor-core
tensor-cpu
tensor-wgpu
```

## How the Rust Code Uses WGSL

WGSL is the shader language that describes the code executed by the GPU. Rust does not execute the WGSL directly. Instead, Rust gives the shader source to `wgpu`, builds a compute pipeline from it, binds GPU buffers to the shader's bindings, then dispatches workgroups.

For example, `KernelCache::new` in `crates/tensor-wgpu/src/kernels.rs` loads the add shader like this:

```rust
let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
    label: Some("add shader"),
    source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/add.wgsl").into()),
});
```

The important part is:

```rust
include_str!("../shaders/add.wgsl")
```

This embeds the WGSL file as a string at compile time. When Rust compiles, the shader text becomes part of the binary. At runtime, `wgpu` receives that string and compiles/validates it for the selected backend, currently Vulkan on Windows.

The ReLU and matmul paths do the same thing with:

```rust
include_str!("../shaders/relu.wgsl")
include_str!("../shaders/matmul.wgsl")
```

So the flow is:

```text
WGSL file -> include_str! -> ShaderModule -> ComputePipeline cached in GpuContext
GpuTensor inputs -> BindGroup -> ComputePass -> dispatch_workgroups
```

The important optimization is that shader modules, bind group layouts, pipeline layouts, and compute pipelines are now created once when `GpuContext` is created. Individual operations still create per-call bind groups because the actual input and output buffers change every time.

## GPU Setup

`GpuContext::new()` in `crates/tensor-wgpu/src/context.rs` owns the reusable GPU setup.

It creates a `wgpu::Instance` with:

```rust
wgpu::InstanceDescriptor::new_without_display_handle_from_env()
```

This allows environment variables such as `WGPU_BACKEND` to influence backend selection. Then it requests a high-performance adapter:

```rust
power_preference: wgpu::PowerPreference::HighPerformance
```

The code rejects CPU/software adapters:

```rust
if adapter_info.device_type == wgpu::DeviceType::Cpu {
    bail!(...)
}
```

That guard is intentional. It prevents the project from silently running on software Vulkan such as `llvmpipe` and pretending GPU execution works.

After a real adapter is selected, the context creates:

```text
wgpu::Device - creates buffers, shaders, pipelines, bind groups, encoders
wgpu::Queue  - submits finished command buffers to the GPU
```

After the device and queue exist, `GpuContext::new()` also builds a `KernelCache`. The cache stores reusable bind group layouts and compute pipelines for add, ReLU, and matmul. This avoids rebuilding pipeline state inside every tensor operation and makes the GPU-resident benchmarks measure more of the actual dispatch path.

## Tensor Representation

`GpuTensor` in `crates/tensor-wgpu/src/tensor.rs` is the reusable GPU tensor abstraction.

It stores:

```rust
pub struct GpuTensor {
    pub(crate) buffer: wgpu::Buffer,
    shape: Vec<usize>,
    len: usize,
}
```

The `buffer` is the actual GPU memory. The `shape` and `len` are CPU-side metadata that let Rust validate operation compatibility and know how much memory to read back.

`GpuTensor::from_vec` uploads CPU data into a GPU buffer:

```text
&[f32] on CPU -> bytemuck::cast_slice -> wgpu storage buffer
```

`bytemuck::cast_slice` converts `&[f32]` into `&[u8]` without changing the data. GPU buffers are byte buffers, so this is needed when uploading typed Rust data.

`GpuTensor::empty_output` allocates a GPU buffer with the right size for an operation result.

`GpuTensor::to_vec` downloads a GPU tensor back to CPU memory. It works by:

```text
1. Creating a staging buffer with MAP_READ usage.
2. Encoding a copy from the tensor buffer into the staging buffer.
3. Submitting the copy command to the queue.
4. Waiting for the GPU with device.poll(...).
5. Mapping the staging buffer for reading.
6. Casting bytes back to Vec<f32>.
```

This staging-buffer pattern is required because GPU storage buffers are not directly readable by normal Rust code.

## Shape Validation

`tensor-core` validates that:

```text
shape product == element count
elementwise binary ops use identical shapes
empty tensors are valid
```

For now, there is no broadcasting. That means add requires exactly matching shapes. This is deliberate: it keeps the runtime simple until the core GPU execution path is stable.

An empty shape `[]` is treated as a scalar shape and expects one element. A shape containing a zero dimension, such as `[0]` or `[2, 0, 4]`, represents an empty tensor.

## CPU Reference Code

`crates/tensor-cpu` contains simple CPU implementations:

```rust
cpu_add(a, b) -> Result<Vec<f32>>
cpu_relu(input) -> Vec<f32>
cpu_matmul(a, a_shape, b, b_shape) -> Result<Vec<f32>>
cpu_mlp_forward(...) -> Result<Vec<f32>>
```

The CPU code is not just a fallback. It is the correctness reference for GPU results. The demo computes the expected result on CPU, runs the same operation on GPU, and asserts equality.

That pattern is important for the thesis because every GPU operation should have a simple CPU reference before performance is measured.

## GPU Add

The public convenience API is:

```rust
gpu_add(&GpuContext, &[f32], &[f32]) -> Result<Vec<f32>>
```

It:

```text
1. Validates equal input lengths.
2. Uploads both slices into GpuTensor buffers.
3. Calls gpu_add_tensor.
4. Reads the output tensor back into Vec<f32>.
```

The tensor-native API is:

```rust
gpu_add_tensor(&GpuContext, &GpuTensor, &GpuTensor) -> Result<GpuTensor>
```

It keeps the result resident on the GPU. This is the more important API for future MLP inference because later operations can consume GPU tensors without downloading after every kernel.

The reusable add pipeline state is created once in `KernelCache`:

```text
ShaderModule       from add.wgsl
BindGroupLayout    bindings 0, 1, 2
PipelineLayout     uses the bind group layout
ComputePipeline    uses the WGSL main function
```

Each `gpu_add_tensor` call still creates:

```text
BindGroup          connects actual buffers to shader bindings
CommandEncoder     records commands
ComputePass        sets pipeline, bind group, and dispatches work
```

The add shader declares three buffers:

```wgsl
@group(0) @binding(0) var<storage, read> a: Data;
@group(0) @binding(1) var<storage, read> b: Data;
@group(0) @binding(2) var<storage, read_write> c: Data;
```

Those bindings match the Rust bind group entries:

```text
binding 0 -> a.buffer
binding 1 -> b.buffer
binding 2 -> output.buffer
```

The shader runs one invocation per element:

```wgsl
let index = id.x;
c.values[index] = a.values[index] + b.values[index];
```

The guard:

```wgsl
if (index >= arrayLength(&c.values)) {
    return;
}
```

exists because Rust dispatches work in groups of 64 threads. If the tensor length is not exactly divisible by 64, extra GPU invocations are launched. Those extra invocations must return without touching memory.

## GPU ReLU

The ReLU path mirrors add, but with one input buffer and one output buffer.

The public convenience API is:

```rust
gpu_relu(&GpuContext, &[f32]) -> Result<Vec<f32>>
```

The tensor-native API is:

```rust
gpu_relu_tensor(&GpuContext, &GpuTensor) -> Result<GpuTensor>
```

The ReLU shader declares:

```wgsl
@group(0) @binding(0) var<storage, read> input: Data;
@group(0) @binding(1) var<storage, read_write> output: Data;
```

Rust binds:

```text
binding 0 -> input.buffer
binding 1 -> output.buffer
```

The shader operation is:

```wgsl
output.values[index] = max(input.values[index], 0.0);
```

This proves the tensor abstraction works for both:

```text
binary elementwise op: add(a, b)
unary elementwise op: relu(x)
```

## Workgroups

Both shaders use:

```wgsl
@compute @workgroup_size(64)
```

Rust computes the number of workgroups with:

```rust
let workgroups = (input.len() as u32).div_ceil(WORKGROUP_SIZE);
```

If there are 3 elements and a workgroup size of 64, Rust dispatches 1 workgroup. That creates 64 possible invocations, but only indices 0, 1, and 2 do useful work. The shader bounds check protects the rest.

This is normal GPU programming. You usually dispatch enough threads to cover the data, then guard the tail.

## Demo Flow

`src/main.rs` is a smoke test that runs the current runtime end to end:

```text
1. Create GpuContext.
2. Print the selected adapter.
3. Run CPU add and GPU add.
4. Print both results.
5. Assert equality.
6. Run CPU ReLU and GPU ReLU.
7. Print both results.
8. Assert equality.
9. Run CPU matmul and GPU matmul.
10. Run CPU MLP forward and GPU MLP forward.
```

The latest successful output confirms:

```text
wgpu selected the RTX 5070
add shader produced the expected result
relu shader produced the expected result
matmul shader produced the expected result
MLP forward produced the expected result
GPU readback worked
CPU/GPU correctness checks passed
```

## Tests

The normal test suite currently covers:

```text
CPU add
CPU ReLU
CPU matmul
CPU MLP
shape validation
elementwise shape mismatch
empty tensor shapes
```

GPU tests are opt-in with:

```bash
RUST_GPU_RUN_GPU_TESTS=1 cargo test
```

They are opt-in because not every development environment has a working hardware GPU backend. Normal `cargo test` should stay usable in WSL or CI.

## Why the Design Looks Like This

The project is intentionally not a full tensor framework yet. The current design is a small foundation:

```text
GpuContext owns GPU access and cached kernel pipeline state.
GpuTensor owns GPU memory plus shape metadata.
tensor-cpu gives correctness references.
tensor-wgpu ops dispatch concrete GPU kernels.
WGSL files contain the code that actually runs on the GPU.
main.rs proves the whole path works on real hardware.
```

This keeps the next steps manageable. The runtime can now add more operations without rewriting setup, upload, readback, or shape handling every time.

## Next Natural Step

The next optimization should be chosen from benchmark output after pipeline caching:

```text
reusable output buffers
GPU-resident MLP benchmark
matmul kernel tiling
GPU timestamp-query timing
```

Pipeline caching removes one large fixed cost. The remaining benchmark gaps should show whether allocation, transfer, synchronization, or naive matmul math is the next biggest issue.
