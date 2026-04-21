//! Runtime backend: wgpu driver plumbing.
//!
//! The single backend lives in [`backend`] and is re-exported verbatim.
//! The indirection exists so the `let mut` carve-out documented in
//! [`CLAUDE.md`](../../../CLAUDE.md) stays scoped to exactly one file on
//! disk.  A future backend (parallel `cat-cuda`, SPIR-V driver) would land
//! as a sibling file with its own documented carve-out.

pub(crate) mod backend;

pub(crate) use backend::{
    alloc_storage_buffer, build_vector_add_pipeline, compile_wgsl, default_queue,
    download_f32_buffer, launch_vector_add_f32, pick_adapter_and_device, upload_f32_buffer,
};
