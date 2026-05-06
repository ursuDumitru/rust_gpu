use anyhow::Result;
use rust_gpu::{GpuContext, cpu::cpu_add, ops::gpu_add};

fn main() -> Result<()> {
    env_logger::init();
    pollster::block_on(run())
}

async fn run() -> Result<()> {
    let context = GpuContext::new().await?;
    println!(
        "Using adapter: {} ({:?}, {:?})",
        context.adapter_info.name, context.adapter_info.backend, context.adapter_info.device_type
    );

    let a = vec![1.0_f32, 2.0, 3.0];
    let b = vec![4.0_f32, 5.0, 6.0];
    let expected = cpu_add(&a, &b)?;
    let result = gpu_add(&context, &a, &b).await?;

    println!("a:        {a:?}");
    println!("b:        {b:?}");
    println!("gpu add:  {result:?}");
    println!("expected: {expected:?}");

    assert_eq!(result, expected);
    println!("GPU vector add succeeded.");

    Ok(())
}
