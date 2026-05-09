//! WGPU context setup and synchronization.
//!
//! A context selects a GPU adapter, creates a device and queue, and provides the
//! objects needed by tensor operations to create buffers and submit work.

use anyhow::{Context, Result, bail};

use crate::kernels::KernelCache;

/// Owns the WGPU objects needed to submit compute work to one GPU adapter.
pub struct GpuContext {
    /// The top-level WGPU object used to discover adapters.
    pub instance: wgpu::Instance,
    /// The selected physical or logical GPU adapter.
    pub adapter: wgpu::Adapter,
    /// Human-readable details about the selected adapter.
    pub adapter_info: wgpu::AdapterInfo,
    /// The logical device used to create buffers, shaders, and pipelines.
    pub device: wgpu::Device,
    /// The queue used to submit encoded GPU commands.
    pub queue: wgpu::Queue,
    /// Reusable compute pipelines and bind layouts for tensor kernels.
    pub(crate) kernels: KernelCache,
    timestamp_queries_supported: bool,
}

impl GpuContext {
    /// Creates a high-performance GPU context and rejects CPU/software adapters.
    pub async fn new() -> Result<Self> {
        let instance_descriptor = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
        let requested_backends = instance_descriptor.backends;
        let instance = wgpu::Instance::new(instance_descriptor);

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .context("no suitable GPU adapter found; check WSL GPU/Vulkan setup")?;

        let adapter_info = adapter.get_info();
        if adapter_info.device_type == wgpu::DeviceType::Cpu {
            let adapters = describe_adapters(&instance, requested_backends).await;
            bail!(
                "wgpu selected a CPU/software adapter ({}) instead of a GPU; check WSL GPU/Vulkan setup\navailable adapters:\n{}",
                adapter_info.name,
                adapters
            );
        }

        let timestamp_queries_supported =
            adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let required_features = if timestamp_queries_supported {
            wgpu::Features::TIMESTAMP_QUERY
        } else {
            wgpu::Features::empty()
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("rust_gpu device"),
                required_features,
                required_limits: wgpu::Limits::downlevel_defaults(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .context("failed to create wgpu device and queue")?;
        let kernels = KernelCache::new(&device);

        Ok(Self {
            instance,
            adapter,
            adapter_info,
            device,
            queue,
            kernels,
            timestamp_queries_supported,
        })
    }

    /// Blocks until previously submitted GPU work has finished.
    ///
    /// This is mainly useful for benchmarks and explicit synchronization points.
    pub fn wait_idle(&self) -> Result<()> {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("failed while waiting for GPU work to finish")?;

        Ok(())
    }

    /// Returns whether this context can record GPU timestamp queries.
    pub fn timestamp_queries_supported(&self) -> bool {
        self.timestamp_queries_supported
    }
}

/// Builds a short report of adapters visible to WGPU.
async fn describe_adapters(instance: &wgpu::Instance, backends: wgpu::Backends) -> String {
    let adapters = instance.enumerate_adapters(backends).await;

    if adapters.is_empty() {
        return "  none".to_string();
    }

    adapters
        .iter()
        .map(|adapter| {
            let info = adapter.get_info();
            format!(
                "  - {} ({:?}, {:?})",
                info.name, info.backend, info.device_type
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}
