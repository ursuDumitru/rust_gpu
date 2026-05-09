//! Criterion benchmarks for the tensor runtime.
//!
//! This file is compiled into a separate benchmark executable when you run
//! `cargo bench`. It measures CPU reference ops, GPU end-to-end ops, and
//! GPU-resident ops.
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use tensor_cpu::{cpu_add, cpu_matmul, cpu_mlp_forward, cpu_relu};
use tensor_wgpu::{
    GpuContext, GpuTensor, gpu_add, gpu_add_tensor, gpu_matmul, gpu_matmul_tensor,
    gpu_matmul_tensor_timed, gpu_matmul_tiled, gpu_matmul_tiled_tensor,
    gpu_matmul_tiled_tensor_timed, gpu_mlp_forward, gpu_mlp_forward_tensor, gpu_mlp_forward_tiled,
    gpu_mlp_forward_tiled_tensor, gpu_relu, gpu_relu_tensor,
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

/// Larger square matrix sizes used for targeted GPU scaling benchmarks.
///
/// These are kept out of the default matmul groups so broad `cargo bench` runs
/// stay reasonable. Use a Criterion filter containing `matmul_large` or
/// `matmul_tiled_large` to run them.
const LARGE_MATMUL_SIZES: &[usize] = &[256, 512, 1024];

/// Larger MLP cases used to find when GPU-resident inference becomes useful.
const GENERATED_MLP_CASES: &[GeneratedMlpCase] = &[
    GeneratedMlpCase {
        name: "small",
        batch: 1,
        input: 128,
        hidden: 256,
        output: 64,
    },
    GeneratedMlpCase {
        name: "medium",
        batch: 1,
        input: 512,
        hidden: 512,
        output: 128,
    },
    GeneratedMlpCase {
        name: "batch_32",
        batch: 32,
        input: 128,
        hidden: 256,
        output: 64,
    },
];

/// Fixed MLP input used by tests, the demo, and MLP benchmarks.
const MLP_X: &[f32] = &[1.0, -2.0, 3.0];
/// First fixed MLP weight matrix with shape `[3, 4]`.
const MLP_W1: &[f32] = &[
    0.5, -1.0, 2.0, 0.0, //
    1.0, 0.5, -0.5, 2.0, //
    -1.5, 1.0, 0.0, 0.5,
];
/// First fixed MLP bias with shape `[1, 4]`.
const MLP_B1: &[f32] = &[0.5, 1.0, -1.0, 0.0];
/// Second fixed MLP weight matrix with shape `[4, 2]`.
const MLP_W2: &[f32] = &[
    1.0, -1.0, //
    0.5, 0.25, //
    -2.0, 1.5, //
    1.0, 0.0,
];
/// Second fixed MLP bias with shape `[1, 2]`.
const MLP_B2: &[f32] = &[0.25, -0.75];

/// Operation families selected for this benchmark run.
struct BenchSelection {
    add: bool,
    relu: bool,
    matmul: bool,
    mlp: bool,
}

impl BenchSelection {
    /// Decides which benchmark families run from env vars or Criterion filters.
    fn from_env() -> Self {
        if let Some(raw_filter) = std::env::var("RUST_GPU_BENCH_FILTER")
            .ok()
            .filter(|value| !value.trim().is_empty())
        {
            return Self::from_filter_text(&raw_filter);
        }

        criterion_filter_arg()
            .and_then(|filter| Self::from_criterion_filter(&filter))
            .unwrap_or_else(Self::all)
    }

    /// Parses the comma-separated `RUST_GPU_BENCH_FILTER` value.
    fn from_filter_text(raw_filter: &str) -> Self {
        let mut selection = Self::none();
        for raw_part in raw_filter.split(',') {
            let part = raw_part.trim().to_ascii_lowercase();
            match part.as_str() {
                "" => {}
                "all" => return Self::all(),
                "add" => selection.add = true,
                "relu" => selection.relu = true,
                "matmul" => selection.matmul = true,
                "mlp" => selection.mlp = true,
                other => {
                    panic!(
                        "unknown RUST_GPU_BENCH_FILTER value `{other}`; valid values are add, relu, matmul, mlp, all"
                    );
                }
            }
        }

        if selection.any() {
            selection
        } else {
            Self::all()
        }
    }

    /// Infers the operation family from Criterion's positional filter.
    fn from_criterion_filter(filter: &str) -> Option<Self> {
        let filter = filter.to_ascii_lowercase();
        let mut selection = Self::none();

        if filter.contains("add") {
            selection.add = true;
        }
        if filter.contains("relu") {
            selection.relu = true;
        }
        if filter.contains("matmul") {
            selection.matmul = true;
        }
        if filter.contains("mlp") {
            selection.mlp = true;
        }

        selection.any().then_some(selection)
    }

    /// Selects every benchmark family.
    fn all() -> Self {
        Self {
            add: true,
            relu: true,
            matmul: true,
            mlp: true,
        }
    }

    /// Selects no benchmark families before filter parsing fills them in.
    fn none() -> Self {
        Self {
            add: false,
            relu: false,
            matmul: false,
            mlp: false,
        }
    }

    /// Returns whether at least one benchmark family is selected.
    fn any(&self) -> bool {
        self.add || self.relu || self.matmul || self.mlp
    }
}

/// Shape description for one generated two-layer MLP benchmark case.
struct GeneratedMlpCase {
    name: &'static str,
    batch: usize,
    input: usize,
    hidden: usize,
    output: usize,
}

/// CPU-owned tensors and shapes for one generated MLP benchmark case.
struct GeneratedMlpData {
    x: Vec<f32>,
    x_shape: [usize; 2],
    w1: Vec<f32>,
    w1_shape: [usize; 2],
    b1: Vec<f32>,
    b1_shape: [usize; 2],
    w2: Vec<f32>,
    w2_shape: [usize; 2],
    b2: Vec<f32>,
    b2_shape: [usize; 2],
}

/// Registers all CPU benchmark groups.
///
/// CPU benchmarks always run because they do not require a GPU.
fn bench_cpu_ops(c: &mut Criterion) {
    let selection = BenchSelection::from_env();
    if selection.add {
        bench_cpu_add(c);
    }
    if selection.relu {
        bench_cpu_relu(c);
    }
    if selection.matmul {
        bench_cpu_matmul(c);
    }
    if selection.mlp {
        bench_cpu_mlp(c);
    }
}

/// Registers all GPU benchmark groups when GPU benchmarks are enabled.
///
/// Set `RUST_GPU_RUN_GPU_BENCHES=1` to run all GPU benchmarks, or pass a
/// Criterion filter containing `gpu_` to run matching GPU benchmarks.
fn bench_gpu_ops(c: &mut Criterion) {
    let selection = BenchSelection::from_env();
    if !gpu_benches_enabled() {
        return;
    }

    let context = pollster::block_on(GpuContext::new()).expect("failed to create GPU context");

    if selection.add {
        bench_gpu_add(c, &context);
        bench_gpu_resident_add(c, &context);
    }
    if selection.relu {
        bench_gpu_relu(c, &context);
        bench_gpu_resident_relu(c, &context);
    }
    if selection.matmul {
        bench_gpu_matmul(c, &context);
        bench_gpu_matmul_tiled(c, &context);
        bench_gpu_resident_matmul(c, &context);
        bench_gpu_resident_matmul_tiled(c, &context);
        if large_matmul_benches_enabled() {
            bench_gpu_resident_matmul_large(c, &context);
            bench_gpu_resident_matmul_tiled_large(c, &context);
        }
        if timestamp_matmul_benches_enabled() {
            bench_gpu_timestamp_matmul_large(c, &context);
            bench_gpu_timestamp_matmul_tiled_large(c, &context);
        }
    }
    if selection.mlp {
        bench_gpu_mlp(c, &context);
        bench_gpu_mlp_tiled(c, &context);
        bench_gpu_resident_mlp(c, &context);
        bench_gpu_resident_mlp_tiled(c, &context);
    }
}

/// Returns whether this run should create a GPU context for GPU benchmarks.
fn gpu_benches_enabled() -> bool {
    std::env::var("RUST_GPU_RUN_GPU_BENCHES").ok().as_deref() == Some("1")
        || criterion_filter_requests_gpu()
}

/// Detects Criterion filters such as `gpu_resident/mlp`.
fn criterion_filter_requests_gpu() -> bool {
    criterion_filter_arg().is_some_and(|arg| arg.contains("gpu_"))
}

/// Returns whether targeted large matmul benchmark groups should be registered.
fn large_matmul_benches_enabled() -> bool {
    criterion_filter_arg()
        .is_some_and(|arg| arg.contains("matmul_large") || arg.contains("matmul_tiled_large"))
}

/// Returns whether GPU timestamp-query matmul groups should be registered.
fn timestamp_matmul_benches_enabled() -> bool {
    criterion_filter_arg().is_some_and(|arg| arg.contains("gpu_timestamp"))
}

/// Returns Criterion's positional benchmark filter argument, when present.
fn criterion_filter_arg() -> Option<String> {
    std::env::args().skip(1).find(|arg| !arg.starts_with('-'))
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

/// Measures the full fixed CPU MLP forward pass.
fn bench_cpu_mlp(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu/mlp");
    group.bench_function("fixed", |bencher| {
        bencher.iter(|| {
            let result = cpu_mlp_forward(
                black_box(MLP_X),
                &[1, 3],
                black_box(MLP_W1),
                &[3, 4],
                black_box(MLP_B1),
                &[1, 4],
                black_box(MLP_W2),
                &[4, 2],
                black_box(MLP_B2),
                &[1, 2],
            )
            .unwrap();
            black_box(result);
        });
    });
    group.finish();

    let mut group = c.benchmark_group("cpu/mlp/generated");
    for case in GENERATED_MLP_CASES {
        let data = make_generated_mlp_data(case);
        group.bench_function(case.name, |bencher| {
            bencher.iter(|| {
                let result = cpu_mlp_forward(
                    black_box(&data.x),
                    &data.x_shape,
                    black_box(&data.w1),
                    &data.w1_shape,
                    black_box(&data.b1),
                    &data.b1_shape,
                    black_box(&data.w2),
                    &data.w2_shape,
                    black_box(&data.b2),
                    &data.b2_shape,
                )
                .unwrap();
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

/// Measures end-to-end GPU matmul using the tiled kernel.
fn bench_gpu_matmul_tiled(c: &mut Criterion, context: &GpuContext) {
    let mut group = c.benchmark_group("gpu_e2e/matmul_tiled");
    for &size in MATMUL_SIZES {
        let a = make_matrix(size, size, 0.125);
        let b = make_matrix(size, size, -0.25);
        let shape = [size, size];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = pollster::block_on(gpu_matmul_tiled(
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

/// Measures end-to-end GPU MLP forward.
///
/// This includes uploading input, weights, and biases, running all kernels, and
/// reading the final output back to the CPU.
fn bench_gpu_mlp(c: &mut Criterion, context: &GpuContext) {
    let mut group = c.benchmark_group("gpu_e2e/mlp");
    group.bench_function("fixed", |bencher| {
        bencher.iter(|| {
            let result = pollster::block_on(gpu_mlp_forward(
                context,
                black_box(MLP_X),
                &[1, 3],
                black_box(MLP_W1),
                &[3, 4],
                black_box(MLP_B1),
                &[1, 4],
                black_box(MLP_W2),
                &[4, 2],
                black_box(MLP_B2),
                &[1, 2],
            ))
            .unwrap();
            black_box(result);
        });
    });
    group.finish();

    let mut group = c.benchmark_group("gpu_e2e/mlp/generated");
    for case in GENERATED_MLP_CASES {
        let data = make_generated_mlp_data(case);
        group.bench_function(case.name, |bencher| {
            bencher.iter(|| {
                let result = pollster::block_on(gpu_mlp_forward(
                    context,
                    black_box(&data.x),
                    &data.x_shape,
                    black_box(&data.w1),
                    &data.w1_shape,
                    black_box(&data.b1),
                    &data.b1_shape,
                    black_box(&data.w2),
                    &data.w2_shape,
                    black_box(&data.b2),
                    &data.b2_shape,
                ))
                .unwrap();
                black_box(result);
            });
        });
    }
    group.finish();
}

/// Measures end-to-end GPU MLP forward using tiled matmul kernels.
fn bench_gpu_mlp_tiled(c: &mut Criterion, context: &GpuContext) {
    let mut group = c.benchmark_group("gpu_e2e/mlp_tiled/generated");
    for case in GENERATED_MLP_CASES {
        let data = make_generated_mlp_data(case);
        group.bench_function(case.name, |bencher| {
            bencher.iter(|| {
                let result = pollster::block_on(gpu_mlp_forward_tiled(
                    context,
                    black_box(&data.x),
                    &data.x_shape,
                    black_box(&data.w1),
                    &data.w1_shape,
                    black_box(&data.b1),
                    &data.b1_shape,
                    black_box(&data.w2),
                    &data.w2_shape,
                    black_box(&data.b2),
                    &data.b2_shape,
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

/// Measures GPU-resident matmul using the tiled kernel.
fn bench_gpu_resident_matmul_tiled(c: &mut Criterion, context: &GpuContext) {
    let mut group = c.benchmark_group("gpu_resident/matmul_tiled");
    for &size in MATMUL_SIZES {
        let a = make_matrix(size, size, 0.125);
        let b = make_matrix(size, size, -0.25);
        let shape = [size, size];
        let a = GpuTensor::from_vec(context, &a, &shape).unwrap();
        let b = GpuTensor::from_vec(context, &b, &shape).unwrap();
        context.wait_idle().unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = pollster::block_on(gpu_matmul_tiled_tensor(
                    context,
                    black_box(&a),
                    black_box(&b),
                ))
                .unwrap();
                context.wait_idle().unwrap();
                black_box(result);
            });
        });
    }
    group.finish();
}

/// Measures larger GPU-resident matmul cases with the naive kernel.
fn bench_gpu_resident_matmul_large(c: &mut Criterion, context: &GpuContext) {
    let mut group = c.benchmark_group("gpu_resident/matmul_large");
    for &size in LARGE_MATMUL_SIZES {
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

/// Measures larger GPU-resident matmul cases with the tiled kernel.
fn bench_gpu_resident_matmul_tiled_large(c: &mut Criterion, context: &GpuContext) {
    let mut group = c.benchmark_group("gpu_resident/matmul_tiled_large");
    for &size in LARGE_MATMUL_SIZES {
        let a = make_matrix(size, size, 0.125);
        let b = make_matrix(size, size, -0.25);
        let shape = [size, size];
        let a = GpuTensor::from_vec(context, &a, &shape).unwrap();
        let b = GpuTensor::from_vec(context, &b, &shape).unwrap();
        context.wait_idle().unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = pollster::block_on(gpu_matmul_tiled_tensor(
                    context,
                    black_box(&a),
                    black_box(&b),
                ))
                .unwrap();
                context.wait_idle().unwrap();
                black_box(result);
            });
        });
    }
    group.finish();
}

/// Measures larger naive matmul cases using GPU timestamp queries.
///
/// Criterion normally measures host wall-clock time. This group returns the
/// timestamp-query delta from the GPU instead, so command encoding, queue
/// submission, synchronization, and query readback are not part of the reported
/// duration.
fn bench_gpu_timestamp_matmul_large(c: &mut Criterion, context: &GpuContext) {
    if !context.timestamp_queries_supported() {
        eprintln!("skipping gpu_timestamp/matmul_large: timestamp queries are unsupported");
        return;
    }

    let mut group = c.benchmark_group("gpu_timestamp/matmul_large");
    for &size in LARGE_MATMUL_SIZES {
        let a = make_matrix(size, size, 0.125);
        let b = make_matrix(size, size, -0.25);
        let shape = [size, size];
        let a = GpuTensor::from_vec(context, &a, &shape).unwrap();
        let b = GpuTensor::from_vec(context, &b, &shape).unwrap();
        context.wait_idle().unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter_custom(|iters| {
                let mut total = std::time::Duration::ZERO;
                for _ in 0..iters {
                    let timing = pollster::block_on(gpu_matmul_tensor_timed(
                        context,
                        black_box(&a),
                        black_box(&b),
                    ))
                    .unwrap();
                    total += timing.elapsed_duration();
                    black_box(timing.output);
                }
                total
            });
        });
    }
    group.finish();
}

/// Measures larger tiled matmul cases using GPU timestamp queries.
fn bench_gpu_timestamp_matmul_tiled_large(c: &mut Criterion, context: &GpuContext) {
    if !context.timestamp_queries_supported() {
        eprintln!("skipping gpu_timestamp/matmul_tiled_large: timestamp queries are unsupported");
        return;
    }

    let mut group = c.benchmark_group("gpu_timestamp/matmul_tiled_large");
    for &size in LARGE_MATMUL_SIZES {
        let a = make_matrix(size, size, 0.125);
        let b = make_matrix(size, size, -0.25);
        let shape = [size, size];
        let a = GpuTensor::from_vec(context, &a, &shape).unwrap();
        let b = GpuTensor::from_vec(context, &b, &shape).unwrap();
        context.wait_idle().unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter_custom(|iters| {
                let mut total = std::time::Duration::ZERO;
                for _ in 0..iters {
                    let timing = pollster::block_on(gpu_matmul_tiled_tensor_timed(
                        context,
                        black_box(&a),
                        black_box(&b),
                    ))
                    .unwrap();
                    total += timing.elapsed_duration();
                    black_box(timing.output);
                }
                total
            });
        });
    }
    group.finish();
}

/// Measures GPU-resident MLP forward.
///
/// The fixed input, weights, and biases are uploaded once before timing starts.
/// The timed loop runs the multi-op forward pass without CPU readback.
fn bench_gpu_resident_mlp(c: &mut Criterion, context: &GpuContext) {
    let x = GpuTensor::from_vec(context, MLP_X, &[1, 3]).unwrap();
    let w1 = GpuTensor::from_vec(context, MLP_W1, &[3, 4]).unwrap();
    let b1 = GpuTensor::from_vec(context, MLP_B1, &[1, 4]).unwrap();
    let w2 = GpuTensor::from_vec(context, MLP_W2, &[4, 2]).unwrap();
    let b2 = GpuTensor::from_vec(context, MLP_B2, &[1, 2]).unwrap();
    context.wait_idle().unwrap();

    let mut group = c.benchmark_group("gpu_resident/mlp");
    group.bench_function("fixed", |bencher| {
        bencher.iter(|| {
            let result = pollster::block_on(gpu_mlp_forward_tensor(
                context,
                black_box(&x),
                black_box(&w1),
                black_box(&b1),
                black_box(&w2),
                black_box(&b2),
            ))
            .unwrap();
            context.wait_idle().unwrap();
            black_box(result);
        });
    });
    group.finish();

    let mut group = c.benchmark_group("gpu_resident/mlp/generated");
    for case in GENERATED_MLP_CASES {
        let data = make_generated_mlp_data(case);
        let x = GpuTensor::from_vec(context, &data.x, &data.x_shape).unwrap();
        let w1 = GpuTensor::from_vec(context, &data.w1, &data.w1_shape).unwrap();
        let b1 = GpuTensor::from_vec(context, &data.b1, &data.b1_shape).unwrap();
        let w2 = GpuTensor::from_vec(context, &data.w2, &data.w2_shape).unwrap();
        let b2 = GpuTensor::from_vec(context, &data.b2, &data.b2_shape).unwrap();
        context.wait_idle().unwrap();

        group.bench_function(case.name, |bencher| {
            bencher.iter(|| {
                let result = pollster::block_on(gpu_mlp_forward_tensor(
                    context,
                    black_box(&x),
                    black_box(&w1),
                    black_box(&b1),
                    black_box(&w2),
                    black_box(&b2),
                ))
                .unwrap();
                context.wait_idle().unwrap();
                black_box(result);
            });
        });
    }
    group.finish();
}

/// Measures GPU-resident MLP forward using tiled matmul kernels.
fn bench_gpu_resident_mlp_tiled(c: &mut Criterion, context: &GpuContext) {
    let mut group = c.benchmark_group("gpu_resident/mlp_tiled/generated");
    for case in GENERATED_MLP_CASES {
        let data = make_generated_mlp_data(case);
        let x = GpuTensor::from_vec(context, &data.x, &data.x_shape).unwrap();
        let w1 = GpuTensor::from_vec(context, &data.w1, &data.w1_shape).unwrap();
        let b1 = GpuTensor::from_vec(context, &data.b1, &data.b1_shape).unwrap();
        let w2 = GpuTensor::from_vec(context, &data.w2, &data.w2_shape).unwrap();
        let b2 = GpuTensor::from_vec(context, &data.b2, &data.b2_shape).unwrap();
        context.wait_idle().unwrap();

        group.bench_function(case.name, |bencher| {
            bencher.iter(|| {
                let result = pollster::block_on(gpu_mlp_forward_tiled_tensor(
                    context,
                    black_box(&x),
                    black_box(&w1),
                    black_box(&b1),
                    black_box(&w2),
                    black_box(&b2),
                ))
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

/// Creates deterministic tensors for one generated two-layer MLP case.
fn make_generated_mlp_data(case: &GeneratedMlpCase) -> GeneratedMlpData {
    GeneratedMlpData {
        x: make_matrix(case.batch, case.input, 0.05),
        x_shape: [case.batch, case.input],
        w1: make_matrix(case.input, case.hidden, -0.02),
        w1_shape: [case.input, case.hidden],
        b1: make_repeated_bias(case.batch, case.hidden, 0.01),
        b1_shape: [case.batch, case.hidden],
        w2: make_matrix(case.hidden, case.output, 0.03),
        w2_shape: [case.hidden, case.output],
        b2: make_repeated_bias(case.batch, case.output, -0.04),
        b2_shape: [case.batch, case.output],
    }
}

/// Creates exact-shape bias data by repeating one bias row for every batch row.
fn make_repeated_bias(rows: usize, cols: usize, offset: f32) -> Vec<f32> {
    (0..rows * cols)
        .map(|index| {
            let col = index % cols;
            ((col % 19) as f32 * 0.005) + offset
        })
        .collect()
}

// These macros tell Criterion which benchmark functions to run and generate the
// benchmark executable entry point.
criterion_group!(benches, bench_cpu_ops, bench_gpu_ops);
criterion_main!(benches);
