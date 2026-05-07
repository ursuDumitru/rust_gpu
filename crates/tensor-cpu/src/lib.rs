//! CPU reference backend for tensor operations.
//!
//! These functions are simple and correctness-focused. They are used by tests,
//! demos, and benchmarks as the reference behavior for the GPU backend.

/// CPU implementation of the fixed MLP forward pass.
pub mod mlp;
/// CPU implementations of individual tensor operations.
pub mod ops;

pub use mlp::cpu_mlp_forward;
pub use ops::{cpu_add, cpu_matmul, cpu_relu};
