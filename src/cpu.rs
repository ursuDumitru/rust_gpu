use anyhow::{Result, bail};

pub fn cpu_add(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    validate_same_len(a, b)?;
    Ok(a.iter().zip(b).map(|(left, right)| left + right).collect())
}

pub(crate) fn validate_same_len(a: &[f32], b: &[f32]) -> Result<()> {
    if a.len() != b.len() {
        bail!(
            "input length mismatch: left has {} elements, right has {} elements",
            a.len(),
            b.len()
        );
    }

    Ok(())
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
    fn validate_same_len_accepts_matching_lengths() {
        validate_same_len(&[1.0, 2.0], &[3.0, 4.0]).unwrap();
    }

    #[test]
    fn validate_same_len_rejects_mismatched_lengths() {
        let error = validate_same_len(&[1.0], &[]).unwrap_err();
        assert!(error.to_string().contains("input length mismatch"));
    }
}
