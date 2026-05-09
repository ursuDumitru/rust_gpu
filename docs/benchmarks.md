# Benchmark Notes

This file records what the benchmark harness measures, how to run selected
benchmarks, and what the current results tell us about the runtime.

## How To Run

Run the whole CPU benchmark set:

```powershell
cargo bench
```

Run one benchmark family by using Criterion's filter argument:

```powershell
cargo bench --bench ops -- cpu/mlp
cargo bench --bench ops -- gpu_resident/matmul
cargo bench --bench ops -- gpu_resident/matmul_tiled_large
cargo bench --bench ops -- gpu_timestamp/matmul_tiled_large
```

GPU benchmarks are normally skipped unless they are requested. They run when one
of these is true:

- `RUST_GPU_RUN_GPU_BENCHES=1` is set.
- The Criterion filter contains `gpu_`, for example `gpu_resident/matmul`.

The older family filter is still supported:

```powershell
$env:RUST_GPU_BENCH_FILTER="mlp"
cargo bench
```

Clear the environment variables in PowerShell with:

```powershell
Remove-Item Env:RUST_GPU_BENCH_FILTER
Remove-Item Env:RUST_GPU_RUN_GPU_BENCHES
```

## Benchmark Families

`cpu/*` benchmarks measure the Rust CPU reference implementation.

`gpu_e2e/*` benchmarks measure the public convenience GPU APIs. These include:

- CPU-to-GPU upload
- GPU command encoding and dispatch
- waiting for completion
- GPU-to-CPU readback

`gpu_resident/*` benchmarks upload inputs before timing starts. The timed loop
runs tensor-native GPU operations and waits for the GPU to finish, but does not
read the result back to the CPU. This is closer to how real inference should be
measured when tensors stay on the GPU across multiple operations.

`gpu_timestamp/*` benchmarks use GPU timestamp queries. Criterion still drives
the benchmark loop, but the reported duration comes from timestamps written in
the GPU command stream around the compute pass. These numbers exclude Rust
function-call overhead, WGPU command encoding, queue submission, synchronization,
query readback, and CPU/GPU transfer. They require adapter support for
`wgpu::Features::TIMESTAMP_QUERY`.

Run timestamp-query matmul benchmarks directly:

```powershell
cargo bench --bench ops -- gpu_timestamp/matmul_large
cargo bench --bench ops -- gpu_timestamp/matmul_tiled_large
```

## Current Matmul Results

These results were measured on Windows with an NVIDIA GeForce RTX 5070.

Naive GPU-resident matmul:

| Size | Time |
| --- | ---: |
| `256x256` | `98.9 us` |
| `512x512` | `328.7 us` |
| `1024x1024` | `2.00 ms` |

Tiled GPU-resident matmul:

| Size | Time |
| --- | ---: |
| `256x256` | `90.2 us` |
| `512x512` | `223.6 us` |
| `1024x1024` | `1.28 ms` |

Approximate tiled speedup:

| Size | Speedup |
| --- | ---: |
| `256x256` | `1.10x` |
| `512x512` | `1.47x` |
| `1024x1024` | `1.56x` |

## What This Means

The tiled kernel is faster because it reduces repeated global memory reads.

The naive matmul kernel assigns one GPU thread to each output element. Each
thread independently reads one row of `A` and one column of `B` from global GPU
memory. Neighboring threads need many of the same values, but the naive kernel
does not share them.

The tiled matmul kernel makes each workgroup load small chunks of `A` and `B`
into workgroup memory. Threads in the same workgroup then reuse those cached
values while computing a block of output elements. This lowers global memory
traffic and improves scaling as matrices get larger.

The improvement is small at `256x256` because fixed overhead still matters. It
becomes clearer at `512x512` and `1024x1024`, where there is enough arithmetic
work for memory reuse to pay off.

## Current Limitations

Most Criterion groups still measure host wall-clock time around the Rust API
call. For GPU-resident benchmarks that includes command encoding, command
submission, and `wait_idle()`, not only shader execution time.

The `gpu_timestamp/*` groups now measure the matmul compute pass on the GPU
command stream. They currently exist for large naive and tiled matmul cases only.
They do not yet cover add, ReLU, or full MLP chains.

Output buffers are still allocated inside each tensor operation. Reusing output
buffers may reduce fixed overhead further, especially for small operations and
multi-op MLP benchmarks.

## Next Measurement Step

Use `gpu_timestamp/*` and `gpu_resident/*` together. The difference between them
is the host-side overhead around a kernel dispatch.

After collecting timestamp-query results, the next implementation optimizations
to compare are:

- reusable output buffers
- larger tiled matmul cases
- fused MLP kernels such as `matmul + bias + relu`
