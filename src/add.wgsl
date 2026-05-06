struct Data {
    values: array<f32>,
};

@group(0) @binding(0)
var<storage, read> a: Data;

@group(0) @binding(1)
var<storage, read> b: Data;

@group(0) @binding(2)
var<storage, read_write> c: Data;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;

    if (index >= arrayLength(&c.values)) {
        return;
    }

    c.values[index] = a.values[index] + b.values[index];
}
