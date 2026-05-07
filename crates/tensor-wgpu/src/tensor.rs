//! GPU-resident tensor buffers.
//!
//! `GpuTensor` keeps values in a WGPU buffer and keeps shape metadata in Rust
//! for validation and output sizing.

use std::sync::mpsc;

use anyhow::{Context, Result};
use tensor_core::shape::{element_count, validate_shape_len};
use wgpu::util::DeviceExt;

use crate::GpuContext;

/// A tensor whose values live in a GPU storage buffer.
///
/// The shape and length are kept on the CPU side so Rust can validate operation
/// shapes before dispatching WGSL kernels.
#[derive(Debug)]
pub struct GpuTensor {
    pub(crate) buffer: wgpu::Buffer,
    shape: Vec<usize>,
    len: usize,
}

impl GpuTensor {
    /// Uploads `f32` values into a new GPU tensor with the given shape.
    pub fn from_vec(context: &GpuContext, values: &[f32], shape: &[usize]) -> Result<Self> {
        validate_shape_len(shape, values.len())?;

        let buffer = if values.is_empty() {
            context.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("empty tensor"),
                size: buffer_size(0)?,
                usage: tensor_buffer_usage(),
                mapped_at_creation: false,
            })
        } else {
            context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("tensor upload"),
                    contents: bytemuck::cast_slice(values),
                    usage: tensor_buffer_usage(),
                })
        };

        Ok(Self {
            buffer,
            shape: shape.to_vec(),
            len: values.len(),
        })
    }

    /// Creates a GPU tensor for an operation output without initializing values.
    pub fn empty_output(context: &GpuContext, shape: &[usize]) -> Result<Self> {
        let len = element_count(shape)?;
        let buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tensor output"),
            size: buffer_size(len)?,
            usage: tensor_buffer_usage(),
            mapped_at_creation: false,
        });

        Ok(Self {
            buffer,
            shape: shape.to_vec(),
            len,
        })
    }

    /// Downloads the tensor values from the GPU into a CPU `Vec<f32>`.
    pub async fn to_vec(&self, context: &GpuContext) -> Result<Vec<f32>> {
        if self.is_empty() {
            return Ok(Vec::new());
        }

        let byte_len = buffer_size(self.len)?;
        let staging_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tensor readback staging"),
            size: byte_len,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tensor readback command encoder"),
            });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging_buffer, 0, byte_len);
        context.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        context
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("failed while waiting for GPU tensor readback")?;

        receiver
            .recv()
            .context("GPU tensor readback callback was dropped before completion")?
            .context("failed to map GPU tensor readback buffer")?;

        let mapped = buffer_slice.get_mapped_range();
        let result = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging_buffer.unmap();

        Ok(result)
    }

    /// Returns the tensor shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the total number of `f32` elements in the tensor.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the tensor contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of dimensions in the tensor shape.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Returns one dimension from the shape, or `None` if it is out of bounds.
    pub fn dim(&self, index: usize) -> Option<usize> {
        self.shape.get(index).copied()
    }
}

/// Converts an element count into a buffer byte size.
fn buffer_size(len: usize) -> Result<wgpu::BufferAddress> {
    let bytes = len
        .checked_mul(std::mem::size_of::<f32>())
        .context("tensor byte size overflow")?;
    Ok(bytes.max(std::mem::size_of::<f32>()) as wgpu::BufferAddress)
}

/// Returns the buffer usage flags required by tensor storage buffers.
fn tensor_buffer_usage() -> wgpu::BufferUsages {
    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST
}
