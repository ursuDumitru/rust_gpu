//! GPU timing result types.
//!
//! Timestamp-query helpers return both the GPU-resident output tensor and the
//! measured GPU time for the compute pass that produced it.

use std::time::Duration;

use crate::GpuTensor;

/// Result of one GPU operation measured with timestamp queries.
pub struct GpuKernelTiming {
    /// The tensor produced by the timed GPU operation.
    pub output: GpuTensor,
    /// Elapsed GPU time in nanoseconds for the timed compute pass.
    pub elapsed_ns: f64,
}

impl GpuKernelTiming {
    /// Converts the measured nanoseconds into a standard `Duration`.
    pub fn elapsed_duration(&self) -> Duration {
        Duration::from_nanos(self.elapsed_ns.round() as u64)
    }
}
