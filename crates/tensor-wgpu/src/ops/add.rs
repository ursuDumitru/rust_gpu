//! GPU vector and tensor addition.
//!
//! The tensor-native API keeps the output on the GPU for chaining.

use anyhow::Result;
use tensor_core::shape::{validate_same_len, validate_same_shape};

use crate::{GpuContext, GpuTensor};

const WORKGROUP_SIZE: u32 = 64;

/// Adds two CPU slices on the GPU and downloads the result back to the CPU.
pub async fn gpu_add(context: &GpuContext, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    validate_same_len(a, b)?;

    let a = GpuTensor::from_vec(context, a, &[a.len()])?;
    let b = GpuTensor::from_vec(context, b, &[b.len()])?;
    let result = gpu_add_tensor(context, &a, &b).await?;

    result.to_vec(context).await
}

/// Adds two GPU-resident tensors and returns a GPU-resident output tensor.
///
/// Both input tensors must have exactly the same shape. No data is read back to
/// the CPU by this function.
pub async fn gpu_add_tensor(
    context: &GpuContext,
    a: &GpuTensor,
    b: &GpuTensor,
) -> Result<GpuTensor> {
    validate_same_shape(a.shape(), b.shape())?;

    let output = GpuTensor::empty_output(context, a.shape())?;
    if output.is_empty() {
        return Ok(output);
    }

    let kernel = &context.kernels.add;
    let bind_group = context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("add bind group"),
            layout: &kernel.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.buffer.as_entire_binding(),
                },
            ],
        });

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("add command encoder"),
        });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("add compute pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&kernel.pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (a.len() as u32).div_ceil(WORKGROUP_SIZE);
        compute_pass.dispatch_workgroups(workgroups, 1, 1);
    }

    context.queue.submit(Some(encoder.finish()));

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_add_tensor_test_is_opt_in_for_gpu_validation() {
        let run_gpu_test = std::env::var("RUST_GPU_RUN_GPU_TESTS").ok().as_deref() == Some("1");
        if !run_gpu_test {
            return;
        }

        pollster::block_on(async {
            let context = GpuContext::new().await.unwrap();
            let a = GpuTensor::from_vec(&context, &[1.0, 2.0, 3.0], &[3]).unwrap();
            let b = GpuTensor::from_vec(&context, &[4.0, 5.0, 6.0], &[3]).unwrap();

            let result = gpu_add_tensor(&context, &a, &b).await.unwrap();
            assert_eq!(result.shape(), &[3]);
            assert_eq!(result.to_vec(&context).await.unwrap(), vec![5.0, 7.0, 9.0]);
        });
    }
}
