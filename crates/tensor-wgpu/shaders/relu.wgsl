struct Data {
    values: array<f32>,
};

@group(0) @binding(0)
var<storage, read> input: Data;

@group(0) @binding(1)
var<storage, read_write> output: Data;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;

    if (index >= arrayLength(&output.values)) {
        return;
    }

    output.values[index] = max(input.values[index], 0.0);
}
