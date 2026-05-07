//! GPU row-major 2D matrix multiplication.
//!
//! The current kernel is correctness-focused and assigns one shader invocation
//! to each output cell.

use anyhow::{Context, Result};
use tensor_core::shape::validate_matmul_shapes;
use wgpu::util::DeviceExt;

use crate::{GpuContext, GpuTensor};

const MATMUL_WORKGROUP_SIZE: u32 = 16;

/// Multiplies two CPU matrices on the GPU and downloads the result.
///
/// The supported shape rule is row-major `[m, k] @ [k, n] -> [m, n]`.
pub async fn gpu_matmul(
    context: &GpuContext,
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
) -> Result<Vec<f32>> {
    let a = GpuTensor::from_vec(context, a, a_shape)?;
    let b = GpuTensor::from_vec(context, b, b_shape)?;
    let result = gpu_matmul_tensor(context, &a, &b).await?;

    result.to_vec(context).await
}

/// Multiplies two GPU-resident row-major matrices.
///
/// This uses the current simple WGSL kernel with one shader invocation per
/// output cell. The result stays on the GPU until `GpuTensor::to_vec` is called.
pub async fn gpu_matmul_tensor(
    context: &GpuContext,
    a: &GpuTensor,
    b: &GpuTensor,
) -> Result<GpuTensor> {
    let output_shape = validate_matmul_shapes(a.shape(), b.shape())?;
    let output = GpuTensor::empty_output(context, &output_shape)?;
    if output.is_empty() {
        return Ok(output);
    }

    let m = output_shape[0];
    let k = a.dim(1).context("matmul left tensor missing k dimension")?;
    let n = output_shape[1];
    let params = [
        u32::try_from(m).context("matmul m dimension does not fit in u32")?,
        u32::try_from(k).context("matmul k dimension does not fit in u32")?,
        u32::try_from(n).context("matmul n dimension does not fit in u32")?,
        0,
    ];

    let params_buffer = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("matmul params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let kernel = &context.kernels.matmul;
    let bind_group = context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul bind group"),
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
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("matmul command encoder"),
        });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("matmul compute pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&kernel.pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroups_x = (n as u32).div_ceil(MATMUL_WORKGROUP_SIZE);
        let workgroups_y = (m as u32).div_ceil(MATMUL_WORKGROUP_SIZE);
        compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    context.queue.submit(Some(encoder.finish()));

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_matmul_tensor_test_is_opt_in_for_gpu_validation() {
        let run_gpu_test = std::env::var("RUST_GPU_RUN_GPU_TESTS").ok().as_deref() == Some("1");
        if !run_gpu_test {
            return;
        }

        pollster::block_on(async {
            let context = GpuContext::new().await.unwrap();
            let a =
                GpuTensor::from_vec(&context, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
            let b =
                GpuTensor::from_vec(&context, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]).unwrap();

            let result = gpu_matmul_tensor(&context, &a, &b).await.unwrap();
            assert_eq!(result.shape(), &[2, 2]);
            assert_eq!(
                result.to_vec(&context).await.unwrap(),
                vec![58.0, 64.0, 139.0, 154.0]
            );
        });
    }
}
