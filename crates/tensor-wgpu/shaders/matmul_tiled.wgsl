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

var<workgroup> tile_a: array<array<f32, 16>, 16>;
var<workgroup> tile_b: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let col = global_id.x;
    let row = global_id.y;
    let local_col = local_id.x;
    let local_row = local_id.y;

    var sum = 0.0;

    for (var tile_start = 0u; tile_start < params.k; tile_start = tile_start + 16u) {
        let a_col = tile_start + local_col;
        let b_row = tile_start + local_row;

        if (row < params.m && a_col < params.k) {
            tile_a[local_row][local_col] = a.values[row * params.k + a_col];
        } else {
            tile_a[local_row][local_col] = 0.0;
        }

        if (b_row < params.k && col < params.n) {
            tile_b[local_row][local_col] = b.values[b_row * params.n + col];
        } else {
            tile_b[local_row][local_col] = 0.0;
        }

        workgroupBarrier();

        for (var inner = 0u; inner < 16u; inner = inner + 1u) {
            sum = sum + tile_a[local_row][inner] * tile_b[inner][local_col];
        }

        workgroupBarrier();
    }

    if (row < params.m && col < params.n) {
        c.values[row * params.n + col] = sum;
    }
}
