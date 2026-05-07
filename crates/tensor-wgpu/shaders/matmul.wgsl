struct Data {
    values: array<f32>,
};

struct Params {
    m: u32,
    k: u32,
    n: u32,
    _padding: u32,
};

@group(0) @binding(0)
var<storage, read> a: Data;

@group(0) @binding(1)
var<storage, read> b: Data;

@group(0) @binding(2)
var<storage, read_write> c: Data;

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let col = id.x;
    let row = id.y;

    if (row >= params.m || col >= params.n) {
        return;
    }

    var sum = 0.0;
    for (var inner = 0u; inner < params.k; inner = inner + 1u) {
        let a_index = row * params.k + inner;
        let b_index = inner * params.n + col;
        sum = sum + a.values[a_index] * b.values[b_index];
    }

    c.values[row * params.n + col] = sum;
}
