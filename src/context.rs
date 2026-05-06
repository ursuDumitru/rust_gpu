use anyhow::{Context, Result, bail};

pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub adapter_info: wgpu::AdapterInfo,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
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

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("rust_gpu device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .context("failed to create wgpu device and queue")?;

        Ok(Self {
            instance,
            adapter,
            adapter_info,
            device,
            queue,
        })
    }
}

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
