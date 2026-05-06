# rust_gpu Progress and Context

Last updated: 2026-05-06

## Project goal

`rust_gpu` is a thesis-oriented Rust project for building a minimal GPU tensor/ML inference runtime with `wgpu`.

The refined target scope is:

```text
A minimal Rust tensor runtime with a CPU backend, wgpu backend, MLP inference,
benchmarks, and one or two kernel fusion optimizations.
```

The project is intentionally inference-only for now. Training, autograd, convolutions, large model support, CUDA-specific implementation, and a full PyTorch-like API are out of scope.

## Current implementation

The project is currently a single Rust crate named `rust_gpu`.

Implemented so far:

```text
src/lib.rs       library entrypoint
src/context.rs   reusable GPU context setup
src/cpu.rs       CPU reference operation
src/ops.rs       reusable GPU vector add operation
src/main.rs      small binary demo
src/add.wgsl     WGSL compute shader for vector add
```

The demo does:

```text
1. Create a wgpu instance.
2. Request a high-performance adapter.
3. Reject CPU/software adapters.
4. Create a device and queue.
5. Upload two Vec<f32> inputs.
6. Dispatch a WGSL compute shader.
7. Read results back through a staging buffer.
8. Compare GPU output with CPU output.
```

Expected demo computation:

```text
[1.0, 2.0, 3.0] + [4.0, 5.0, 6.0] = [5.0, 7.0, 9.0]
```

`GpuContext` now uses:

```rust
wgpu::InstanceDescriptor::new_without_display_handle_from_env()
```

This means `wgpu` environment variables such as `WGPU_BACKEND` can influence backend selection.

## Validation status

These checks passed after the latest refactor:

```bash
cargo check
cargo test
```

Unit tests currently cover CPU/reference behavior:

```text
equal-length add succeeds
empty vector add succeeds
mismatched lengths fail
shared length validation accepts/rejects correctly
```

`cargo run` has not succeeded on a real GPU inside WSL yet.

## WSL GPU investigation

Environment:

```text
WSL2
Ubuntu 26.04
NVIDIA GeForce RTX 5070
Windows NVIDIA driver visible from WSL
```

The user ran `nvidia-smi` successfully inside WSL:

```text
NVIDIA-SMI 590.57
Driver Version: 591.86
CUDA Version: 13.1
GPU: NVIDIA GeForce RTX 5070
```

This proves the Windows NVIDIA driver and WSL CUDA/NVML bridge are working.

However, `wgpu` does not use CUDA for this project. It needs a graphics/compute backend such as Vulkan, DX12, Metal, or GL.

Inside WSL, `wgpu` currently sees only software adapters:

```text
llvmpipe (LLVM 21.1.8, 256 bits) (Vulkan, Cpu)
llvmpipe (LLVM 21.1.8, 256 bits) (Gl, Cpu)
```

The project intentionally rejects these CPU/software adapters so we do not mistake software execution for GPU execution.

The WSL Vulkan checks showed:

```text
/usr/lib/x86_64-linux-gnu/libvulkan_dzn.so: missing
/usr/share/vulkan/icd.d/*dzn*: missing
```

Installed Vulkan ICDs include software/virtual Mesa drivers such as:

```text
lvp_icd.json
gfxstream_vk_icd.json
virtio_icd.json
```

The system has WSL D3D12 bridge libraries:

```text
/usr/lib/wsl/lib/libd3d12.so
/usr/lib/wsl/lib/libdxcore.so
```

The system also has Mesa D3D12 DRI files:

```text
/usr/lib/x86_64-linux-gnu/dri/d3d12_dri.so
/usr/lib/x86_64-linux-gnu/dri/d3d12_drv_video.so
```

But attempts to force GL through D3D12 still selected software:

```bash
WGPU_BACKEND=gl MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA cargo run
```

Result:

```text
llvmpipe (Gl, Cpu)
```

A further command suggested for user-side testing:

```bash
WGPU_BACKEND=gl \
MESA_LOADER_DRIVER_OVERRIDE=d3d12 \
MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA \
cargo run
```

If this still reports `llvmpipe`, then WSL is not currently a good hardware path for this `wgpu` project on this distro.

## Windows-native path

The most reliable route for real GPU execution is to run the project natively on Windows so `wgpu` can use DX12 directly.

Building from this path failed:

```text
\\wsl$\Ubuntu-26.04\home\ursu\projects\rust_gpu
```

Error:

```text
incremental compilation: could not create session directory lock file
Incorrect function. (os error -2147024895)
```

Cause:

```text
Cargo/rustc incremental compilation does not work reliably from the WSL UNC path.
```

Temporary workaround:

```powershell
cd "\\wsl$\Ubuntu-26.04\home\ursu\projects\rust_gpu"

$env:CARGO_INCREMENTAL="0"
$env:CARGO_TARGET_DIR="$env:TEMP\rust_gpu_target"

cargo run
```

Preferred Windows-native workflow:

```powershell
mkdir "$env:USERPROFILE\projects"

robocopy "\\wsl$\Ubuntu-26.04\home\ursu\projects\rust_gpu" "$env:USERPROFILE\projects\rust_gpu" /E /XD target

cd "$env:USERPROFILE\projects\rust_gpu"
cargo run
```

Best long-term path:

```text
Keep a Windows-side checkout at C:\Users\dumit\projects\rust_gpu
and run cargo from that native Windows path.
```

## Important commands

Non-GPU checks safe for assistant to run:

```bash
cargo check
cargo test
```

GPU-related commands should be run by the user or require explicit approval:

```bash
cargo run
vulkaninfo --summary
nvidia-smi
WGPU_BACKEND=gl cargo run
WGPU_BACKEND=vulkan cargo run
```

## Next practical steps

Recommended next step:

```text
Get the vector-add demo running on a real GPU through Windows-native DX12.
```

Then continue implementation milestones:

```text
1. Add a reusable GpuBuffer/GpuTensor shape.
2. Implement add as the first tensor operation.
3. Add relu.
4. Add simple benchmarks for CPU add vs GPU add.
5. Implement matmul after the buffer/tensor shape is stable.
```

Avoid spending too much thesis time fighting WSL Vulkan unless WSL is mandatory.
