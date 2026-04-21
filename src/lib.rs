//! # cat-wgpu
//!
//! Categorical host layer for [`wgpu`] compute.
//!
//! Adapter selection, buffer lifecycle, WGSL shader compilation, compute
//! pipelines, and dispatches are surfaced as [`comp_cat_rs::effect::io::Io`]
//! combinators.  Nothing touches the GPU until the final `run()` at the
//! program boundary, keeping the entire pipeline inside the delay-run
//! catamorphism.
//!
//! On Apple Silicon the active backend is Metal; on Linux, Vulkan; on
//! Windows, D3D12.  All are picked by `wgpu` at runtime from the features
//! enabled in `Cargo.toml`.
//!
//! See [`CLAUDE.md`](https://github.com/MavenRain/cat-wgpu/blob/main/CLAUDE.md)
//! for house rules and the narrow `let mut` carve-out in the dispatch path.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used)]
#![warn(missing_docs)]

pub mod error;
pub mod device;
pub mod memory;
pub mod slipstream;
pub mod kernel;
pub mod launch;
pub mod runtime;

pub use crate::error::{CopyDirection, Error};
pub use crate::device::{Device, DeviceId};
pub use crate::memory::DeviceMem;
pub use crate::slipstream::SlipStream;
pub use crate::kernel::{
    BlockDim, ComputePipeline, GridDim, LaunchConfig, ShaderModule, Wgsl,
};
pub use crate::launch::{
    alloc, build_vector_add, compile, default_slipstream, device_resource,
    download_f32, launch_vector_add_f32, open_device, upload_f32, VECTOR_ADD_ENTRY,
};
