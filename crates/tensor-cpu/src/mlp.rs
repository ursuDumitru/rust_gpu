//! CPU implementation of the small fixed MLP used by the smoke demo.

use anyhow::Result;
use tensor_core::shape::validate_mlp_shapes;

use crate::ops::{cpu_add, cpu_matmul, cpu_relu};

/// Runs the fixed two-layer MLP forward pass on the CPU.
///
/// The computation is `matmul -> add -> relu -> matmul -> add`. All tensors are
/// row-major `f32` slices with explicit shapes.
pub fn cpu_mlp_forward(
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
    validate_mlp_shapes(x_shape, w1_shape, b1_shape, w2_shape, b2_shape)?;

    let h = cpu_matmul(x, x_shape, w1, w1_shape)?;
    let h = cpu_add(&h, b1)?;
    let h = cpu_relu(&h);
    let output = cpu_matmul(&h, b1_shape, w2, w2_shape)?;

    cpu_add(&output, b2)
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
    fn cpu_mlp_forward_computes_fixed_known_output() {
        let result = cpu_mlp_forward(
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
        .unwrap();

        assert_eq!(result, vec![-2.75, 2.75]);
    }
}
