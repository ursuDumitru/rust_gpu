//! GPU implementation of the small fixed MLP used by the smoke demo.
//!
//! Intermediate tensors stay resident on the GPU between operations.

use anyhow::Result;
use tensor_core::shape::validate_mlp_shapes;

use crate::{
    GpuContext, GpuTensor,
    ops::{gpu_add_tensor, gpu_matmul_tensor, gpu_matmul_tiled_tensor, gpu_relu_tensor},
};

/// Runs the fixed two-layer MLP forward pass on the GPU.
///
/// Inputs are uploaded once, intermediate tensors stay GPU-resident, and only
/// the final output is downloaded to the CPU.
pub async fn gpu_mlp_forward(
    context: &GpuContext,
    x: &[f32],
    x_shape: &[usize],
    w1: &[f32],
    w1_shape: &[usize],
    b1: &[f32],
    b1_shape: &[usize],
    w2: &[f32],
    w2_shape: &[usize],
    b2: &[f32],
    b2_shape: &[usize],
) -> Result<Vec<f32>> {
    let x = GpuTensor::from_vec(context, x, x_shape)?;
    let w1 = GpuTensor::from_vec(context, w1, w1_shape)?;
    let b1 = GpuTensor::from_vec(context, b1, b1_shape)?;
    let w2 = GpuTensor::from_vec(context, w2, w2_shape)?;
    let b2 = GpuTensor::from_vec(context, b2, b2_shape)?;

    let output = gpu_mlp_forward_tensor(context, &x, &w1, &b1, &w2, &b2).await?;

    output.to_vec(context).await
}

/// Runs the fixed two-layer MLP forward pass with tiled matmul and downloads it.
///
/// Inputs are uploaded once, intermediate tensors stay GPU-resident, and only
/// the final output is downloaded to the CPU.
pub async fn gpu_mlp_forward_tiled(
    context: &GpuContext,
    x: &[f32],
    x_shape: &[usize],
    w1: &[f32],
    w1_shape: &[usize],
    b1: &[f32],
    b1_shape: &[usize],
    w2: &[f32],
    w2_shape: &[usize],
    b2: &[f32],
    b2_shape: &[usize],
) -> Result<Vec<f32>> {
    let x = GpuTensor::from_vec(context, x, x_shape)?;
    let w1 = GpuTensor::from_vec(context, w1, w1_shape)?;
    let b1 = GpuTensor::from_vec(context, b1, b1_shape)?;
    let w2 = GpuTensor::from_vec(context, w2, w2_shape)?;
    let b2 = GpuTensor::from_vec(context, b2, b2_shape)?;

    let output = gpu_mlp_forward_tiled_tensor(context, &x, &w1, &b1, &w2, &b2).await?;

    output.to_vec(context).await
}

/// Runs the fixed two-layer MLP forward pass on GPU-resident tensors.
///
/// The returned tensor also stays on the GPU. Call `GpuTensor::to_vec` only at
/// the boundary where CPU code needs to inspect the final result.
pub async fn gpu_mlp_forward_tensor(
    context: &GpuContext,
    x: &GpuTensor,
    w1: &GpuTensor,
    b1: &GpuTensor,
    w2: &GpuTensor,
    b2: &GpuTensor,
) -> Result<GpuTensor> {
    validate_mlp_shapes(x.shape(), w1.shape(), b1.shape(), w2.shape(), b2.shape())?;

    let h = gpu_matmul_tensor(context, &x, &w1).await?;
    let h = gpu_add_tensor(context, &h, &b1).await?;
    let h = gpu_relu_tensor(context, &h).await?;
    let output = gpu_matmul_tensor(context, &h, &w2).await?;

    gpu_add_tensor(context, &output, &b2).await
}

/// Runs the fixed two-layer MLP forward pass with tiled matmul kernels.
///
/// This is useful for comparing the current naive matmul path against the tiled
/// matmul path while keeping add and ReLU behavior unchanged.
pub async fn gpu_mlp_forward_tiled_tensor(
    context: &GpuContext,
    x: &GpuTensor,
    w1: &GpuTensor,
    b1: &GpuTensor,
    w2: &GpuTensor,
    b2: &GpuTensor,
) -> Result<GpuTensor> {
    validate_mlp_shapes(x.shape(), w1.shape(), b1.shape(), w2.shape(), b2.shape())?;

    let h = gpu_matmul_tiled_tensor(context, x, w1).await?;
    let h = gpu_add_tensor(context, &h, b1).await?;
    let h = gpu_relu_tensor(context, &h).await?;
    let output = gpu_matmul_tiled_tensor(context, &h, w2).await?;

    gpu_add_tensor(context, &output, b2).await
}

#[cfg(test)]
mod tests {
    use super::*;

    const X: &[f32] = &[1.0, -2.0, 3.0];
    const W1: &[f32] = &[
        0.5, -1.0, 2.0, 0.0, //
        1.0, 0.5, -0.5, 2.0, //
        -1.5, 1.0, 0.0, 0.5,
    ];
    const B1: &[f32] = &[0.5, 1.0, -1.0, 0.0];
    const W2: &[f32] = &[
        1.0, -1.0, //
        0.5, 0.25, //
        -2.0, 1.5, //
        1.0, 0.0,
    ];
    const B2: &[f32] = &[0.25, -0.75];

    #[test]
    fn gpu_mlp_forward_test_is_opt_in_for_gpu_validation() {
        let run_gpu_test = std::env::var("RUST_GPU_RUN_GPU_TESTS").ok().as_deref() == Some("1");
        if !run_gpu_test {
            return;
        }

        pollster::block_on(async {
            let context = GpuContext::new().await.unwrap();
            let result = gpu_mlp_forward(
                &context,
                X,
                &[1, 3],
                W1,
                &[3, 4],
                B1,
                &[1, 4],
                W2,
                &[4, 2],
                B2,
                &[1, 2],
            )
            .await
            .unwrap();

            assert_eq!(result, vec![-2.75, 2.75]);
        });
    }

    #[test]
    fn gpu_mlp_forward_tensor_test_is_opt_in_for_gpu_validation() {
        let run_gpu_test = std::env::var("RUST_GPU_RUN_GPU_TESTS").ok().as_deref() == Some("1");
        if !run_gpu_test {
            return;
        }

        pollster::block_on(async {
            let context = GpuContext::new().await.unwrap();
            let x = GpuTensor::from_vec(&context, X, &[1, 3]).unwrap();
            let w1 = GpuTensor::from_vec(&context, W1, &[3, 4]).unwrap();
            let b1 = GpuTensor::from_vec(&context, B1, &[1, 4]).unwrap();
            let w2 = GpuTensor::from_vec(&context, W2, &[4, 2]).unwrap();
            let b2 = GpuTensor::from_vec(&context, B2, &[1, 2]).unwrap();

            let result = gpu_mlp_forward_tensor(&context, &x, &w1, &b1, &w2, &b2)
                .await
                .unwrap();

            assert_eq!(result.shape(), &[1, 2]);
            assert_eq!(result.to_vec(&context).await.unwrap(), vec![-2.75, 2.75]);
        });
    }

    #[test]
    fn gpu_mlp_forward_tiled_tensor_test_is_opt_in_for_gpu_validation() {
        let run_gpu_test = std::env::var("RUST_GPU_RUN_GPU_TESTS").ok().as_deref() == Some("1");
        if !run_gpu_test {
            return;
        }

        pollster::block_on(async {
            let context = GpuContext::new().await.unwrap();
            let x = GpuTensor::from_vec(&context, X, &[1, 3]).unwrap();
            let w1 = GpuTensor::from_vec(&context, W1, &[3, 4]).unwrap();
            let b1 = GpuTensor::from_vec(&context, B1, &[1, 4]).unwrap();
            let w2 = GpuTensor::from_vec(&context, W2, &[4, 2]).unwrap();
            let b2 = GpuTensor::from_vec(&context, B2, &[1, 2]).unwrap();

            let result = gpu_mlp_forward_tiled_tensor(&context, &x, &w1, &b1, &w2, &b2)
                .await
                .unwrap();

            assert_eq!(result.shape(), &[1, 2]);
            assert_eq!(result.to_vec(&context).await.unwrap(), vec![-2.75, 2.75]);
        });
    }
}
