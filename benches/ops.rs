//! Criterion benchmarks for the tensor runtime.
//!
//! This file is compiled into a separate benchmark executable when you run
//! `cargo bench`. It measures CPU reference ops, GPU end-to-end ops, and
//! GPU-resident ops.
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use tensor_cpu::{cpu_add, cpu_matmul, cpu_relu};
use tensor_wgpu::{
    GpuContext, GpuTensor, gpu_add, gpu_add_tensor, gpu_matmul, gpu_matmul_tensor, gpu_relu,
    gpu_relu_tensor,
};

/// Vector sizes used for add and ReLU benchmarks.
///
/// These operations are elementwise, so the input is a one-dimensional list of
/// values.
const ELEMENTWISE_SIZES: &[usize] = &[1_024, 1_048_576];

/// Square matrix sizes used for matmul benchmarks.
///
/// A size of `16` means `16x16 @ 16x16`. These sizes stay modest because the
/// current GPU matmul kernel is intentionally simple.
const MATMUL_SIZES: &[usize] = &[16, 64, 128];

/// Registers all CPU benchmark groups.
///
/// CPU benchmarks always run because they do not require a GPU.
fn bench_cpu_ops(c: &mut Criterion) {
    bench_cpu_add(c);
    bench_cpu_relu(c);
    bench_cpu_matmul(c);
}

/// Registers all GPU benchmark groups when GPU benchmarks are enabled.
///
/// Set `RUST_GPU_RUN_GPU_BENCHES=1` to run these. Without that variable,
/// `cargo bench` skips GPU setup and still runs the CPU benchmarks.
fn bench_gpu_ops(c: &mut Criterion) {
    if std::env::var("RUST_GPU_RUN_GPU_BENCHES").ok().as_deref() != Some("1") {
        return;
    }

    let context = pollster::block_on(GpuContext::new()).expect("failed to create GPU context");

    bench_gpu_add(c, &context);
    bench_gpu_relu(c, &context);
    bench_gpu_matmul(c, &context);
    bench_gpu_resident_add(c, &context);
    bench_gpu_resident_relu(c, &context);
    bench_gpu_resident_matmul(c, &context);
}

/// Measures CPU vector addition.
///
/// The timed part calls `cpu_add` and returns a new vector result.
fn bench_cpu_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu/add");
    for &size in ELEMENTWISE_SIZES {
        let a = make_vector(size, 0.25);
        let b = make_vector(size, -0.5);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = cpu_add(black_box(&a), black_box(&b)).unwrap();
                black_box(result);
            });
        });
    }
    group.finish();
}

/// Measures CPU ReLU.
///
/// ReLU replaces every negative value with zero and keeps every positive value.
fn bench_cpu_relu(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu/relu");
    for &size in ELEMENTWISE_SIZES {
        let input = make_vector(size, -0.75);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = cpu_relu(black_box(&input));
                black_box(result);
            });
        });
    }
    group.finish();
}

/// Measures CPU matrix multiplication.
///
/// The matrices are stored in row-major order, meaning each row is laid out
/// directly after the previous row in memory.
fn bench_cpu_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu/matmul");
    for &size in MATMUL_SIZES {
        let a = make_matrix(size, size, 0.125);
        let b = make_matrix(size, size, -0.25);
        let shape = [size, size];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = cpu_matmul(black_box(&a), &shape, black_box(&b), &shape).unwrap();
                black_box(result);
            });
        });
    }
    group.finish();
}

/// Measures end-to-end GPU addition.
///
/// End-to-end means the timed work includes upload to the GPU, kernel execution,
/// and readback to the CPU.
fn bench_gpu_add(c: &mut Criterion, context: &GpuContext) {
    let mut group = c.benchmark_group("gpu_e2e/add");
    for &size in ELEMENTWISE_SIZES {
        let a = make_vector(size, 0.25);
        let b = make_vector(size, -0.5);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                let result =
                    pollster::block_on(gpu_add(context, black_box(&a), black_box(&b))).unwrap();
                black_box(result);
            });
        });
    }
    group.finish();
}

/// Measures end-to-end GPU ReLU.
///
/// This includes input upload, GPU work, and output readback.
fn bench_gpu_relu(c: &mut Criterion, context: &GpuContext) {
    let mut group = c.benchmark_group("gpu_e2e/relu");
    for &size in ELEMENTWISE_SIZES {
        let input = make_vector(size, -0.75);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = pollster::block_on(gpu_relu(context, black_box(&input))).unwrap();
                black_box(result);
            });
        });
    }
    group.finish();
}

/// Measures end-to-end GPU matmul.
///
/// The public convenience API receives CPU slices, so each iteration moves data
/// to the GPU and reads the result back.
fn bench_gpu_matmul(c: &mut Criterion, context: &GpuContext) {
    let mut group = c.benchmark_group("gpu_e2e/matmul");
    for &size in MATMUL_SIZES {
        let a = make_matrix(size, size, 0.125);
        let b = make_matrix(size, size, -0.25);
        let shape = [size, size];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = pollster::block_on(gpu_matmul(
                    context,
                    black_box(&a),
                    &shape,
                    black_box(&b),
                    &shape,
                ))
                .unwrap();
                black_box(result);
            });
        });
    }
    group.finish();
}

/// Measures GPU-resident addition.
///
/// The input tensors are uploaded once before timing starts, so the timed part
/// focuses on the tensor-native GPU operation.
fn bench_gpu_resident_add(c: &mut Criterion, context: &GpuContext) {
    let mut group = c.benchmark_group("gpu_resident/add");
    for &size in ELEMENTWISE_SIZES {
        let a = make_vector(size, 0.25);
        let b = make_vector(size, -0.5);
        let a = GpuTensor::from_vec(context, &a, &[size]).unwrap();
        let b = GpuTensor::from_vec(context, &b, &[size]).unwrap();
        context.wait_idle().unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                // `wait_idle` makes the benchmark wait until the GPU has
                // finished, instead of timing only CPU command submission.
                let result =
                    pollster::block_on(gpu_add_tensor(context, black_box(&a), black_box(&b)))
                        .unwrap();
                context.wait_idle().unwrap();
                black_box(result);
            });
        });
    }
    group.finish();
}

/// Measures GPU-resident ReLU.
///
/// No result is copied back to the CPU inside the timed loop, which keeps
/// transfer cost out of this benchmark.
fn bench_gpu_resident_relu(c: &mut Criterion, context: &GpuContext) {
    let mut group = c.benchmark_group("gpu_resident/relu");
    for &size in ELEMENTWISE_SIZES {
        let input = make_vector(size, -0.75);
        let input = GpuTensor::from_vec(context, &input, &[size]).unwrap();
        context.wait_idle().unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                let result =
                    pollster::block_on(gpu_relu_tensor(context, black_box(&input))).unwrap();
                context.wait_idle().unwrap();
                black_box(result);
            });
        });
    }
    group.finish();
}

/// Measures GPU-resident matmul.
///
/// The matrix data starts on the GPU, and each timed iteration launches the
/// matmul kernel and waits for it to finish.
fn bench_gpu_resident_matmul(c: &mut Criterion, context: &GpuContext) {
    let mut group = c.benchmark_group("gpu_resident/matmul");
    for &size in MATMUL_SIZES {
        let a = make_matrix(size, size, 0.125);
        let b = make_matrix(size, size, -0.25);
        let shape = [size, size];
        let a = GpuTensor::from_vec(context, &a, &shape).unwrap();
        let b = GpuTensor::from_vec(context, &b, &shape).unwrap();
        context.wait_idle().unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                let result =
                    pollster::block_on(gpu_matmul_tensor(context, black_box(&a), black_box(&b)))
                        .unwrap();
                context.wait_idle().unwrap();
                black_box(result);
            });
        });
    }
    group.finish();
}

/// Creates deterministic vector data for benchmarks.
///
/// Using a formula instead of random numbers keeps benchmark input stable and
/// avoids adding a random-number crate.
fn make_vector(len: usize, offset: f32) -> Vec<f32> {
    (0..len)
        .map(|index| ((index % 97) as f32 * 0.01) + offset)
        .collect()
}

/// Creates deterministic row-major matrix data for benchmarks.
///
/// The returned vector has `rows * cols` values, with each row placed directly
/// after the previous row.
fn make_matrix(rows: usize, cols: usize, offset: f32) -> Vec<f32> {
    (0..rows * cols)
        .map(|index| {
            let row = index / cols;
            let col = index % cols;
            ((row % 17) as f32 * 0.03) - ((col % 13) as f32 * 0.02) + offset
        })
        .collect()
}

// These macros tell Criterion which benchmark functions to run and generate the
// benchmark executable entry point.
criterion_group!(benches, bench_cpu_ops, bench_gpu_ops);
criterion_main!(benches);
