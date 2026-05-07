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

pub use context::GpuContext;
pub use mlp::gpu_mlp_forward;
pub use ops::{gpu_add, gpu_add_tensor, gpu_matmul, gpu_matmul_tensor, gpu_relu, gpu_relu_tensor};
pub use tensor::GpuTensor;
