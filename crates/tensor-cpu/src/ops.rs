//! CPU implementations of the basic tensor operations.
//!
//! These functions are deliberately straightforward so they can act as readable
//! references for the GPU kernels.

use anyhow::Result;
use tensor_core::shape::{validate_matmul_shapes, validate_same_len, validate_shape_len};

/// Adds two equal-length vectors on the CPU and returns a new output vector.
pub fn cpu_add(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    validate_same_len(a, b)?;
    Ok(a.iter().zip(b).map(|(left, right)| left + right).collect())
}

/// Applies ReLU on the CPU by replacing each negative value with `0.0`.
pub fn cpu_relu(input: &[f32]) -> Vec<f32> {
    input.iter().map(|value| value.max(0.0)).collect()
}

/// Multiplies two row-major 2D matrices on the CPU.
///
/// The supported shape rule is `[m, k] @ [k, n] -> [m, n]`.
pub fn cpu_matmul(a: &[f32], a_shape: &[usize], b: &[f32], b_shape: &[usize]) -> Result<Vec<f32>> {
    let [m, n] = validate_matmul_shapes(a_shape, b_shape)?;
    validate_shape_len(a_shape, a.len())?;
    validate_shape_len(b_shape, b.len())?;

    let k = a_shape[1];
    let mut output = vec![0.0; m * n];

    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0;
            for inner in 0..k {
                sum += a[row * k + inner] * b[inner * n + col];
            }
            output[row * n + col] = sum;
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_adds_equal_length_vectors() {
        let result = cpu_add(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).unwrap();
        assert_eq!(result, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn cpu_add_accepts_empty_vectors() {
        let result = cpu_add(&[], &[]).unwrap();
        assert_eq!(result, Vec::<f32>::new());
    }

    #[test]
    fn cpu_add_rejects_length_mismatch() {
        let error = cpu_add(&[1.0], &[2.0, 3.0]).unwrap_err();
        assert!(error.to_string().contains("input length mismatch"));
    }

    #[test]
    fn cpu_relu_clamps_negative_values_to_zero() {
        let result = cpu_relu(&[-2.0, -0.5, -10.0]);
        assert_eq!(result, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn cpu_relu_passes_positive_values_through() {
        let result = cpu_relu(&[1.0, 2.5, 10.0]);
        assert_eq!(result, vec![1.0, 2.5, 10.0]);
    }

    #[test]
    fn cpu_relu_leaves_zero_as_zero() {
        let result = cpu_relu(&[-1.0, 0.0, 1.0]);
        assert_eq!(result, vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn cpu_relu_accepts_empty_input() {
        let result = cpu_relu(&[]);
        assert_eq!(result, Vec::<f32>::new());
    }

    #[test]
    fn cpu_matmul_computes_known_2x3_by_3x2_result() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

        let result = cpu_matmul(&a, &[2, 3], &b, &[3, 2]).unwrap();
        assert_eq!(result, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn cpu_matmul_computes_1x1_result() {
        let result = cpu_matmul(&[3.0], &[1, 1], &[4.0], &[1, 1]).unwrap();
        assert_eq!(result, vec![12.0]);
    }

    #[test]
    fn cpu_matmul_rejects_invalid_rank() {
        let error = cpu_matmul(&[1.0, 2.0], &[2], &[3.0, 4.0], &[2, 1]).unwrap_err();
        assert!(error.to_string().contains("rank 2"));
    }

    #[test]
    fn cpu_matmul_rejects_inner_dimension_mismatch() {
        let error = cpu_matmul(&[1.0, 2.0], &[1, 2], &[3.0, 4.0], &[1, 2]).unwrap_err();
        assert!(error.to_string().contains("inner dimension mismatch"));
    }

    #[test]
    fn cpu_matmul_accepts_zero_sized_output() {
        let result = cpu_matmul(&[], &[0, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        assert_eq!(result, Vec::<f32>::new());
    }
}
