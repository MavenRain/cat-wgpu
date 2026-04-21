// Vector add: c[i] = a[i] + b[i] for i in 0..arrayLength(&a).
//
// `@group(0)` bind-group entries line up with the `BindGroupEntry`
// order in `runtime::launch_vector_add_f32`:
//   binding 0 -> a (read_write)
//   binding 1 -> b (read_write)
//   binding 2 -> c (read_write, output)
// All three buffers carry the same length; the host enforces that
// before the dispatch.

@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(64)
fn vector_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i: u32 = gid.x;
    let n: u32 = arrayLength(&a);
    if (i >= n) {
        return;
    }
    c[i] = a[i] + b[i];
}
