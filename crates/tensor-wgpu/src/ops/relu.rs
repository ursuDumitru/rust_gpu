//! GPU ReLU operation.
//!
//! ReLU is a unary elementwise operation, so each output value depends on one
//! input value.

use anyhow::Result;

use crate::{GpuContext, GpuTensor};

const WORKGROUP_SIZE: u32 = 64;

/// Applies ReLU to a CPU slice on the GPU and downloads the result.
pub async fn gpu_relu(context: &GpuContext, input: &[f32]) -> Result<Vec<f32>> {
    let input = GpuTensor::from_vec(context, input, &[input.len()])?;
    let result = gpu_relu_tensor(context, &input).await?;

    result.to_vec(context).await
}

/// Applies ReLU to a GPU-resident tensor and returns a GPU-resident output.
///
/// ReLU maps negative values to `0.0` and leaves zero or positive values as-is.
pub async fn gpu_relu_tensor(context: &GpuContext, input: &GpuTensor) -> Result<GpuTensor> {
    let output = GpuTensor::empty_output(context, input.shape())?;
    if output.is_empty() {
        return Ok(output);
    }

    let kernel = &context.kernels.relu;
    let bind_group = context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("relu bind group"),
            layout: &kernel.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.buffer.as_entire_binding(),
                },
            ],
        });

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("relu command encoder"),
        });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("relu compute pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&kernel.pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (input.len() as u32).div_ceil(WORKGROUP_SIZE);
        compute_pass.dispatch_workgroups(workgroups, 1, 1);
    }

    context.queue.submit(Some(encoder.finish()));

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_relu_tensor_test_is_opt_in_for_gpu_validation() {
        let run_gpu_test = std::env::var("RUST_GPU_RUN_GPU_TESTS").ok().as_deref() == Some("1");
        if !run_gpu_test {
            return;
        }

        pollster::block_on(async {
            let context = GpuContext::new().await.unwrap();
            let input = GpuTensor::from_vec(&context, &[-2.0, 0.0, 3.5], &[3]).unwrap();

            let result = gpu_relu_tensor(&context, &input).await.unwrap();
            assert_eq!(result.shape(), &[3]);
            assert_eq!(result.to_vec(&context).await.unwrap(), vec![0.0, 0.0, 3.5]);
        });
    }
}
