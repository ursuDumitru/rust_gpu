//! Shape validation helpers for tensor operations.
//!
//! This module keeps shape rules in one place so CPU and GPU code reject the
//! same invalid inputs.

use anyhow::{Context, Result, bail};

/// Checks that two flat `f32` slices contain the same number of values.
pub fn validate_same_len(left: &[f32], right: &[f32]) -> Result<()> {
    if left.len() != right.len() {
        bail!(
            "input length mismatch: left has {} elements, right has {} elements",
            left.len(),
            right.len()
        );
    }

    Ok(())
}

/// Checks that two tensor shapes are exactly the same.
pub fn validate_same_shape(left: &[usize], right: &[usize]) -> Result<()> {
    if left != right {
        bail!(
            "elementwise shape mismatch: left has shape {:?}, right has shape {:?}",
            left,
            right
        );
    }

    Ok(())
}

/// Checks that a tensor shape describes exactly `len` elements.
pub fn validate_shape_len(shape: &[usize], len: usize) -> Result<()> {
    let expected = element_count(shape)?;
    if expected != len {
        bail!(
            "shape {:?} expects {} elements, got {} elements",
            shape,
            expected,
            len
        );
    }

    Ok(())
}

/// Returns the number of elements described by a tensor shape.
pub fn element_count(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |total, dim| {
        total
            .checked_mul(*dim)
            .context("shape element count overflow")
    })
}

/// Validates 2D matrix multiplication shapes and returns the output shape.
///
/// For row-major matmul, `[m, k] @ [k, n]` produces `[m, n]`.
pub fn validate_matmul_shapes(left: &[usize], right: &[usize]) -> Result<[usize; 2]> {
    if left.len() != 2 {
        bail!("matmul left input must be rank 2, got shape {:?}", left);
    }

    if right.len() != 2 {
        bail!("matmul right input must be rank 2, got shape {:?}", right);
    }

    let m = left[0];
    let k = left[1];
    let right_k = right[0];
    let n = right[1];

    if k != right_k {
        bail!(
            "matmul inner dimension mismatch: left has k={}, right has k={}",
            k,
            right_k
        );
    }

    Ok([m, n])
}

/// Validates the fixed two-layer MLP shape contract used by the demo runtime.
///
/// The expected computation is `x @ w1 + b1`, then ReLU, then `h @ w2 + b2`.
pub fn validate_mlp_shapes(
    x_shape: &[usize],
    w1_shape: &[usize],
    b1_shape: &[usize],
    w2_shape: &[usize],
    b2_shape: &[usize],
) -> Result<[usize; 2]> {
    let hidden_shape = validate_matmul_shapes(x_shape, w1_shape)?;
    validate_exact_shape("first bias", b1_shape, &hidden_shape)?;

    let output_shape = validate_matmul_shapes(&hidden_shape, w2_shape)?;
    validate_exact_shape("second bias", b2_shape, &output_shape)?;

    Ok(output_shape)
}

/// Checks one named shape against the exact expected shape.
fn validate_exact_shape(name: &str, actual: &[usize], expected: &[usize]) -> Result<()> {
    if actual != expected {
        bail!(
            "{} shape mismatch: expected {:?}, got {:?}",
            name,
            expected,
            actual
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_same_len_accepts_matching_lengths() {
        validate_same_len(&[1.0, 2.0], &[3.0, 4.0]).unwrap();
    }

    #[test]
    fn validate_same_len_rejects_mismatched_lengths() {
        let error = validate_same_len(&[1.0], &[]).unwrap_err();
        assert!(error.to_string().contains("input length mismatch"));
    }

    #[test]
    fn validate_shape_accepts_flat_shape() {
        validate_shape_len(&[3], 3).unwrap();
    }

    #[test]
    fn validate_shape_accepts_multidimensional_shape() {
        validate_shape_len(&[2, 3, 4], 24).unwrap();
    }

    #[test]
    fn validate_shape_rejects_product_mismatch() {
        let error = validate_shape_len(&[2, 3], 5).unwrap_err();
        assert!(error.to_string().contains("expects 6 elements"));
    }

    #[test]
    fn validate_shape_accepts_empty_tensor_shape() {
        validate_shape_len(&[0], 0).unwrap();
        validate_shape_len(&[2, 0, 4], 0).unwrap();
    }

    #[test]
    fn validate_shape_treats_empty_shape_as_scalar() {
        validate_shape_len(&[], 1).unwrap();
        let error = validate_shape_len(&[], 0).unwrap_err();
        assert!(error.to_string().contains("expects 1 elements"));
    }

    #[test]
    fn validate_same_shape_rejects_elementwise_shape_mismatch() {
        let error = validate_same_shape(&[2, 3], &[3, 2]).unwrap_err();
        assert!(error.to_string().contains("elementwise shape mismatch"));
    }

    #[test]
    fn validate_matmul_shapes_accepts_compatible_2d_shapes() {
        let output = validate_matmul_shapes(&[2, 3], &[3, 4]).unwrap();
        assert_eq!(output, [2, 4]);
    }

    #[test]
    fn validate_matmul_shapes_rejects_invalid_rank() {
        let error = validate_matmul_shapes(&[2, 3, 4], &[4, 5]).unwrap_err();
        assert!(error.to_string().contains("rank 2"));
    }

    #[test]
    fn validate_matmul_shapes_rejects_inner_dimension_mismatch() {
        let error = validate_matmul_shapes(&[2, 3], &[4, 5]).unwrap_err();
        assert!(error.to_string().contains("inner dimension mismatch"));
    }

    #[test]
    fn validate_mlp_shapes_rejects_incompatible_layer_dimensions() {
        let error = validate_mlp_shapes(&[1, 3], &[4, 4], &[1, 4], &[4, 2], &[1, 2]).unwrap_err();
        assert!(error.to_string().contains("inner dimension mismatch"));
    }

    #[test]
    fn validate_mlp_shapes_rejects_bias_shape_mismatch() {
        let error = validate_mlp_shapes(&[1, 3], &[3, 4], &[4], &[4, 2], &[1, 2]).unwrap_err();
        assert!(error.to_string().contains("first bias shape mismatch"));
    }
}
