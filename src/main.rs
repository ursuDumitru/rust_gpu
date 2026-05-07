//! Smoke-test binary for the tensor runtime.
//!
//! `cargo run` executes a few tiny CPU-vs-GPU checks so the user can confirm
//! that the selected GPU adapter runs add, ReLU, matmul, and MLP inference.

use anyhow::Result;
use tensor_cpu::{cpu_add, cpu_matmul, cpu_mlp_forward, cpu_relu};
use tensor_wgpu::{GpuContext, gpu_add, gpu_matmul, gpu_mlp_forward, gpu_relu};

/// Starts logging and runs the async GPU smoke test.
fn main() -> Result<()> {
    env_logger::init();
    pollster::block_on(run())
}

/// Runs each demo operation and compares the GPU result with the CPU reference.
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

    let relu_input = vec![-2.0_f32, 0.0, 3.5];
    let relu_expected = cpu_relu(&relu_input);
    let relu_result = gpu_relu(&context, &relu_input).await?;

    println!("relu input:    {relu_input:?}");
    println!("gpu relu:      {relu_result:?}");
    println!("relu expected: {relu_expected:?}");

    assert_eq!(relu_result, relu_expected);
    println!("GPU ReLU succeeded.");

    let matmul_a = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let matmul_b = vec![7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let matmul_a_shape = [2, 3];
    let matmul_b_shape = [3, 2];
    let matmul_expected = cpu_matmul(&matmul_a, &matmul_a_shape, &matmul_b, &matmul_b_shape)?;
    let matmul_result = gpu_matmul(
        &context,
        &matmul_a,
        &matmul_a_shape,
        &matmul_b,
        &matmul_b_shape,
    )
    .await?;

    println!("matmul a shape: {matmul_a_shape:?}");
    println!("matmul b shape: {matmul_b_shape:?}");
    println!("gpu matmul:     {matmul_result:?}");
    println!("matmul expected:{matmul_expected:?}");

    assert_eq!(matmul_result, matmul_expected);
    println!("GPU matmul succeeded.");

    let mlp_x = vec![1.0_f32, -2.0, 3.0];
    let mlp_w1 = vec![
        0.5, -1.0, 2.0, 0.0, //
        1.0, 0.5, -0.5, 2.0, //
        -1.5, 1.0, 0.0, 0.5,
    ];
    let mlp_b1 = vec![0.5_f32, 1.0, -1.0, 0.0];
    let mlp_w2 = vec![
        1.0, -1.0, //
        0.5, 0.25, //
        -2.0, 1.5, //
        1.0, 0.0,
    ];
    let mlp_b2 = vec![0.25_f32, -0.75];
    let mlp_expected = cpu_mlp_forward(
        &mlp_x,
        &[1, 3],
        &mlp_w1,
        &[3, 4],
        &mlp_b1,
        &[1, 4],
        &mlp_w2,
        &[4, 2],
        &mlp_b2,
        &[1, 2],
    )?;
    let mlp_result = gpu_mlp_forward(
        &context,
        &mlp_x,
        &[1, 3],
        &mlp_w1,
        &[3, 4],
        &mlp_b1,
        &[1, 4],
        &mlp_w2,
        &[4, 2],
        &mlp_b2,
        &[1, 2],
    )
    .await?;

    println!("gpu mlp:        {mlp_result:?}");
    println!("mlp expected:   {mlp_expected:?}");

    assert_eq!(mlp_result, mlp_expected);
    println!("GPU MLP forward succeeded.");

    Ok(())
}
