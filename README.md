# cat-wgpu

Categorical host layer for [`wgpu`](https://crates.io/crates/wgpu) compute,
built on [`comp-cat-rs`](../comp-cat-rs).

Adapter and device selection, buffer lifecycle, WGSL shader compilation,
compute pipelines, and dispatches are expressed as `Io<Error, _>` and
`Resource<Error, _>` combinators.  Nothing touches the GPU until the final
`.run()` at the program boundary, keeping the entire pipeline inside the
delay-run catamorphism.

## Why wgpu

`wgpu` is the safe-Rust compute surface: Metal on Apple, Vulkan on Linux,
D3D12 on Windows, WebGPU in the browser.  Dispatches are `&mut self` but
never `unsafe`, so `cat-wgpu` runs end-to-end on an M2 MacBook against the
system Metal driver with no extra infrastructure.

The cost is that we don't reach CUDA-only perf, and downstream consumers
(ZK provers targeting NVIDIA, etc.) get a compatibility layer rather than
native CUDA.  A parallel `cat-cuda` crate can slot in later if a consumer
needs it.

## Vector add

```sh
cargo run --example vector_add
```

Runs on the default adapter reported by `wgpu::Instance` (Metal on macOS,
Vulkan / D3D12 elsewhere).  The WGSL kernel lives at
`kernels/vector_add.wgsl`.

## Naming

- `SlipStream`: our CUDA-style work queue.  In wgpu terms it wraps the
  `Queue` handle.  The rename avoids collision with
  `comp_cat_rs::effect::Stream`, which is the iterative colimit abstraction.
- `Wgsl`: owned WGSL source.  The name makes it obvious which shader
  language we accept.  SPIR-V can be added later under a separate newtype.
- `ShaderModule` / `ComputePipeline`: thin newtypes over the wgpu objects.
- `DeviceMem<T>`: a typed storage buffer.  Construction and drop go through
  `Resource<Error, DeviceMem<T>>` so GPU memory release is always scheduled.

## House rules

Inherits the workspace conventions: no `unwrap`, no `expect`, no
`unreachable!`, no `unsafe`, no `dyn` (except through `comp-cat-rs`),
hand-rolled `Error` enum, combinators over pattern matching on `Option`,
exhaustive matches on enums, `if`/`else` on `bool`, no public struct fields,
dual MIT OR Apache-2.0 license.

A single narrow `let mut` carve-out exists for dispatch-recording in
`src/runtime/backend.rs`, because `wgpu`'s `ComputePass` recording API
is entirely `&mut self`.

## Build

```sh
cargo build
cargo test
RUSTFLAGS="-D warnings" cargo clippy --all-targets
cargo run --example vector_add
```
