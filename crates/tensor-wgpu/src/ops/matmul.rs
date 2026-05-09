//! GPU row-major 2D matrix multiplication.
//!
//! The naive kernel assigns one shader invocation to each output cell. The tiled
//! kernel reuses chunks of input matrices through workgroup memory.

use std::sync::mpsc;

use anyhow::{Context, Result, bail};
use tensor_core::shape::validate_matmul_shapes;
use wgpu::util::DeviceExt;

use crate::{GpuContext, GpuKernelTiming, GpuTensor, kernels::CachedKernel};

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

/// Multiplies two CPU matrices with the tiled GPU kernel and downloads the result.
///
/// The supported shape rule is row-major `[m, k] @ [k, n] -> [m, n]`.
pub async fn gpu_matmul_tiled(
    context: &GpuContext,
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
) -> Result<Vec<f32>> {
    let a = GpuTensor::from_vec(context, a, a_shape)?;
    let b = GpuTensor::from_vec(context, b, b_shape)?;
    let result = gpu_matmul_tiled_tensor(context, &a, &b).await?;

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
    gpu_matmul_tensor_with_kernel(context, a, b, &context.kernels.matmul, "matmul").await
}

/// Multiplies two GPU-resident row-major matrices with the tiled kernel.
///
/// The result stays on the GPU until `GpuTensor::to_vec` is called.
pub async fn gpu_matmul_tiled_tensor(
    context: &GpuContext,
    a: &GpuTensor,
    b: &GpuTensor,
) -> Result<GpuTensor> {
    gpu_matmul_tensor_with_kernel(context, a, b, &context.kernels.matmul_tiled, "matmul tiled")
        .await
}

/// Multiplies two GPU-resident matrices and measures the compute pass on the GPU.
///
/// The returned timing excludes CPU-side command encoding, queue submission,
/// query readback, and output download. It requires adapter support for
/// `wgpu::Features::TIMESTAMP_QUERY`.
pub async fn gpu_matmul_tensor_timed(
    context: &GpuContext,
    a: &GpuTensor,
    b: &GpuTensor,
) -> Result<GpuKernelTiming> {
    gpu_matmul_tensor_with_kernel_timed(context, a, b, &context.kernels.matmul, "matmul").await
}

/// Multiplies two GPU-resident matrices with the tiled kernel and measures it.
///
/// The measured time is GPU timestamp-query time for the compute pass itself,
/// not host wall-clock time.
pub async fn gpu_matmul_tiled_tensor_timed(
    context: &GpuContext,
    a: &GpuTensor,
    b: &GpuTensor,
) -> Result<GpuKernelTiming> {
    gpu_matmul_tensor_with_kernel_timed(
        context,
        a,
        b,
        &context.kernels.matmul_tiled,
        "matmul tiled",
    )
    .await
}

/// Shared matmul dispatch path used by the naive and tiled kernels.
async fn gpu_matmul_tensor_with_kernel(
    context: &GpuContext,
    a: &GpuTensor,
    b: &GpuTensor,
    kernel: &CachedKernel,
    label: &'static str,
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
            label: Some(&format!("{label} params")),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bind_group = context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label} bind group")),
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
            label: Some(&format!("{label} command encoder")),
        });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&format!("{label} compute pass")),
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

/// Shared timestamp-query dispatch path used by the naive and tiled kernels.
async fn gpu_matmul_tensor_with_kernel_timed(
    context: &GpuContext,
    a: &GpuTensor,
    b: &GpuTensor,
    kernel: &CachedKernel,
    label: &'static str,
) -> Result<GpuKernelTiming> {
    if !context.timestamp_queries_supported() {
        bail!(
            "GPU timestamp queries are not supported by the selected adapter/device; run normal gpu_resident benchmarks instead"
        );
    }

    let output_shape = validate_matmul_shapes(a.shape(), b.shape())?;
    let output = GpuTensor::empty_output(context, &output_shape)?;
    if output.is_empty() {
        return Ok(GpuKernelTiming {
            output,
            elapsed_ns: 0.0,
        });
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
            label: Some(&format!("{label} timed params")),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bind_group = context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label} timed bind group")),
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

    let query_set = context.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some(&format!("{label} timestamp query set")),
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let query_buffer_size = wgpu::BufferAddress::from(wgpu::QUERY_SIZE) * 2;
    let query_resolve_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("{label} timestamp resolve buffer")),
        size: query_buffer_size,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let query_readback_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("{label} timestamp readback buffer")),
        size: query_buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(&format!("{label} timed command encoder")),
        });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&format!("{label} timed compute pass")),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: &query_set,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            }),
        });
        compute_pass.set_pipeline(&kernel.pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroups_x = (n as u32).div_ceil(MATMUL_WORKGROUP_SIZE);
        let workgroups_y = (m as u32).div_ceil(MATMUL_WORKGROUP_SIZE);
        compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    encoder.resolve_query_set(&query_set, 0..2, &query_resolve_buffer, 0);
    encoder.copy_buffer_to_buffer(
        &query_resolve_buffer,
        0,
        &query_readback_buffer,
        0,
        query_buffer_size,
    );
    context.queue.submit(Some(encoder.finish()));

    let timestamps = read_timestamp_pair(context, &query_readback_buffer).await?;
    let elapsed_ticks = timestamps[1]
        .checked_sub(timestamps[0])
        .context("GPU timestamp query wrapped while measuring matmul")?;
    let elapsed_ns = elapsed_ticks as f64 * f64::from(context.queue.get_timestamp_period());

    Ok(GpuKernelTiming { output, elapsed_ns })
}

/// Reads the start and end timestamp values from a mapped query readback buffer.
async fn read_timestamp_pair(context: &GpuContext, buffer: &wgpu::Buffer) -> Result<[u64; 2]> {
    let buffer_slice = buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });

    context
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .context("failed while waiting for GPU timestamp readback")?;

    receiver
        .recv()
        .context("GPU timestamp readback callback was dropped before completion")?
        .context("failed to map GPU timestamp readback buffer")?;

    let mapped = buffer_slice.get_mapped_range();
    let timestamps = bytemuck::cast_slice::<u8, u64>(&mapped);
    let result = [timestamps[0], timestamps[1]];
    drop(mapped);
    buffer.unmap();

    Ok(result)
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

    #[test]
    fn gpu_matmul_tiled_tensor_test_is_opt_in_for_gpu_validation() {
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

            let result = gpu_matmul_tiled_tensor(&context, &a, &b).await.unwrap();
            assert_eq!(result.shape(), &[2, 2]);
            assert_eq!(
                result.to_vec(&context).await.unwrap(),
                vec![58.0, 64.0, 139.0, 154.0]
            );
        });
    }

    #[test]
    fn gpu_matmul_tiled_non_tile_aligned_test_is_opt_in_for_gpu_validation() {
        let run_gpu_test = std::env::var("RUST_GPU_RUN_GPU_TESTS").ok().as_deref() == Some("1");
        if !run_gpu_test {
            return;
        }

        pollster::block_on(async {
            let context = GpuContext::new().await.unwrap();
            let a_shape = [17, 19];
            let b_shape = [19, 23];
            let a = make_test_matrix(17, 19, 0.125);
            let b = make_test_matrix(19, 23, -0.25);
            let expected = cpu_matmul_for_test(&a, &a_shape, &b, &b_shape);
            let a = GpuTensor::from_vec(&context, &a, &a_shape).unwrap();
            let b = GpuTensor::from_vec(&context, &b, &b_shape).unwrap();

            let result = gpu_matmul_tiled_tensor(&context, &a, &b).await.unwrap();
            assert_eq!(result.shape(), &[17, 23]);
            assert_eq!(result.to_vec(&context).await.unwrap(), expected);
        });
    }

    #[test]
    fn gpu_matmul_timed_test_is_opt_in_for_gpu_validation() {
        let run_gpu_test = std::env::var("RUST_GPU_RUN_GPU_TESTS").ok().as_deref() == Some("1");
        if !run_gpu_test {
            return;
        }

        pollster::block_on(async {
            let context = GpuContext::new().await.unwrap();
            if !context.timestamp_queries_supported() {
                return;
            }

            let a =
                GpuTensor::from_vec(&context, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
            let b =
                GpuTensor::from_vec(&context, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]).unwrap();

            let timing = gpu_matmul_tiled_tensor_timed(&context, &a, &b)
                .await
                .unwrap();
            assert!(timing.elapsed_ns >= 0.0);
            assert_eq!(timing.output.shape(), &[2, 2]);
            assert_eq!(
                timing.output.to_vec(&context).await.unwrap(),
                vec![58.0, 64.0, 139.0, 154.0]
            );
        });
    }

    fn make_test_matrix(rows: usize, cols: usize, offset: f32) -> Vec<f32> {
        (0..rows * cols)
            .map(|index| {
                let row = index / cols;
                let col = index % cols;
                ((row % 17) as f32 * 0.03) - ((col % 13) as f32 * 0.02) + offset
            })
            .collect()
    }

    fn cpu_matmul_for_test(
        a: &[f32],
        a_shape: &[usize; 2],
        b: &[f32],
        b_shape: &[usize; 2],
    ) -> Vec<f32> {
        let [m, k] = *a_shape;
        let n = b_shape[1];
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

        output
    }
}
