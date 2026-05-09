//! Cached WGPU pipeline state for tensor kernels.
//!
//! Shader modules, bind group layouts, pipeline layouts, and compute pipelines
//! are expensive setup objects. This module builds them once when `GpuContext`
//! is created so each tensor operation can reuse the stable pipeline state.

use crate::ops::common::{storage_buffer_entry, uniform_buffer_entry};

/// Reusable pipeline state for all currently supported GPU kernels.
pub(crate) struct KernelCache {
    /// Cached state for elementwise add.
    pub(crate) add: CachedKernel,
    /// Cached state for ReLU.
    pub(crate) relu: CachedKernel,
    /// Cached state for row-major 2D matmul.
    pub(crate) matmul: CachedKernel,
    /// Cached state for row-major 2D tiled matmul.
    pub(crate) matmul_tiled: CachedKernel,
}

impl KernelCache {
    /// Builds all cached kernels for one WGPU device.
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        Self {
            add: CachedKernel::new(
                device,
                "add",
                include_str!("../shaders/add.wgsl"),
                &[
                    storage_buffer_entry(0, true),
                    storage_buffer_entry(1, true),
                    storage_buffer_entry(2, false),
                ],
            ),
            relu: CachedKernel::new(
                device,
                "relu",
                include_str!("../shaders/relu.wgsl"),
                &[
                    storage_buffer_entry(0, true),
                    storage_buffer_entry(1, false),
                ],
            ),
            matmul: CachedKernel::new(
                device,
                "matmul",
                include_str!("../shaders/matmul.wgsl"),
                &[
                    storage_buffer_entry(0, true),
                    storage_buffer_entry(1, true),
                    storage_buffer_entry(2, false),
                    uniform_buffer_entry(3),
                ],
            ),
            matmul_tiled: CachedKernel::new(
                device,
                "matmul tiled",
                include_str!("../shaders/matmul_tiled.wgsl"),
                &[
                    storage_buffer_entry(0, true),
                    storage_buffer_entry(1, true),
                    storage_buffer_entry(2, false),
                    uniform_buffer_entry(3),
                ],
            ),
        }
    }
}

/// Reusable bind group layout and compute pipeline for one WGSL kernel.
pub(crate) struct CachedKernel {
    /// Layout describing the buffers the shader expects.
    pub(crate) bind_group_layout: wgpu::BindGroupLayout,
    /// Compiled compute pipeline that can be reused across dispatches.
    pub(crate) pipeline: wgpu::ComputePipeline,
}

impl CachedKernel {
    /// Creates the reusable WGPU objects for one compute shader.
    fn new(
        device: &wgpu::Device,
        name: &str,
        source: &str,
        bind_group_entries: &[wgpu::BindGroupLayoutEntry],
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{name} shader")),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{name} bind group layout")),
            entries: bind_group_entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{name} pipeline layout")),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{name} pipeline")),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            bind_group_layout,
            pipeline,
        }
    }
}
