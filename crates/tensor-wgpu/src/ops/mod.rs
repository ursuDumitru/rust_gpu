//! GPU tensor operations implemented with WGSL compute shaders.

mod add;
pub(crate) mod common;
mod matmul;
mod relu;

pub use add::{gpu_add, gpu_add_tensor};
pub use matmul::{gpu_matmul, gpu_matmul_tensor};
pub use relu::{gpu_relu, gpu_relu_tensor};
