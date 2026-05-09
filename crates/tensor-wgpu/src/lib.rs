//! WGPU backend for GPU-resident tensor operations.
//!
//! This crate owns GPU context creation, GPU tensor buffers, WGSL compute
//! dispatch, and convenience APIs that upload inputs and read outputs back.

/// GPU adapter, device, and queue setup.
pub mod context;
mod kernels;
/// GPU implementation of the fixed MLP forward pass.
pub mod mlp;
/// GPU implementations of individual tensor operations.
pub mod ops;
/// GPU-resident tensor buffer type.
pub mod tensor;
/// GPU timestamp-query result types.
pub mod timing;

pub use context::GpuContext;
pub use mlp::{
    gpu_mlp_forward, gpu_mlp_forward_tensor, gpu_mlp_forward_tiled, gpu_mlp_forward_tiled_tensor,
};
pub use ops::{
    gpu_add, gpu_add_tensor, gpu_matmul, gpu_matmul_tensor, gpu_matmul_tensor_timed,
    gpu_matmul_tiled, gpu_matmul_tiled_tensor, gpu_matmul_tiled_tensor_timed, gpu_relu,
    gpu_relu_tensor,
};
pub use tensor::GpuTensor;
pub use timing::GpuKernelTiming;
