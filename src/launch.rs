//! Public surface: [`Io`] / [`Resource`] combinators over the wgpu
//! runtime.
//!
//! Everything here is lazy.  Composing these combinators builds a
//! description of a GPU pipeline; the description only fires when the
//! caller invokes [`Io::run`] at the program boundary.  That keeps the
//! side-effect ordering visible in the categorical graph and matches
//! the delay-run pattern used throughout the workspace.

use bytemuck::Pod;

use comp_cat_rs::effect::io::Io;
use comp_cat_rs::effect::resource::Resource;

use crate::device::{Device, DeviceId};
use crate::error::Error;
use crate::kernel::{ComputePipeline, LaunchConfig, ShaderModule, Wgsl};
use crate::memory::DeviceMem;
use crate::runtime;
use crate::slipstream::SlipStream;

/// Entry-point name baked into [`build_vector_add`] and the default
/// WGSL kernel shipped with the crate.
pub const VECTOR_ADD_ENTRY: &str = "vector_add";

/// Open the wgpu device at `id`, as an [`Io`].
///
/// The underlying adapter / device / queue handshake runs inside the
/// suspended thunk, so nothing fires until [`Io::run`] is called.
#[must_use]
pub fn open_device(id: DeviceId) -> Io<Error, Device> {
    Io::suspend(move || runtime::pick_adapter_and_device(id))
}

/// Bracketed form of [`open_device`].
///
/// The release action is an [`Io::pure`] `()` because `wgpu::Device`
/// and `wgpu::Queue` are `Arc`-backed handles: tear-down happens when
/// the final clone is dropped.  The [`Resource`] bracket is still
/// useful because it surfaces the acquire / release pairing in the
/// categorical graph, matching how other workspace crates compose
/// device-like resources.
#[must_use]
pub fn device_resource(id: DeviceId) -> Resource<Error, Device> {
    Resource::make(move || open_device(id), |_device| Io::pure(()))
}

/// Clone the device's queue handle into a fresh [`SlipStream`].
///
/// Wrapped in [`Io::pure`] for API uniformity even though the work is
/// just an `Arc` bump.
#[must_use]
pub fn default_slipstream(device: &Device) -> Io<Error, SlipStream> {
    Io::pure(SlipStream::new(runtime::default_queue(device)))
}

/// Allocate a zero-initialised device buffer holding `len` elements of
/// type `T`.
#[must_use]
pub fn alloc<T>(device: &Device, len: usize) -> Io<Error, DeviceMem<T>>
where
    T: Pod + Send + 'static,
{
    let device = device.clone();
    Io::suspend(move || {
        runtime::alloc_storage_buffer::<T>(&device, len)
            .map(|(buffer, n)| DeviceMem::new(buffer, n))
    })
}

/// Upload an `f32` host vector into a new device buffer.
#[must_use]
pub fn upload_f32(device: &Device, data: Vec<f32>) -> Io<Error, DeviceMem<f32>> {
    let device = device.clone();
    Io::suspend(move || {
        let (buffer, n) = runtime::upload_f32_buffer(&device, &data);
        Ok(DeviceMem::new(buffer, n))
    })
}

/// Download an `f32` device buffer back to the host as an owned `Vec`.
#[must_use]
pub fn download_f32(device: &Device, mem: &DeviceMem<f32>) -> Io<Error, Vec<f32>> {
    let device = device.clone();
    let mem = mem.clone();
    Io::suspend(move || runtime::download_f32_buffer(&device, mem.raw(), mem.len()))
}

/// Compile a [`Wgsl`] source string on the device into a
/// [`ShaderModule`].
#[must_use]
pub fn compile(device: &Device, source: Wgsl) -> Io<Error, ShaderModule> {
    let device = device.clone();
    Io::suspend(move || {
        runtime::compile_wgsl(&device, source.as_str()).map(ShaderModule::new)
    })
}

/// Build the vector-add compute pipeline from a previously compiled
/// [`ShaderModule`].
#[must_use]
pub fn build_vector_add(
    device: &Device,
    module: &ShaderModule,
) -> Io<Error, ComputePipeline> {
    let device = device.clone();
    let module = module.clone();
    Io::suspend(move || {
        runtime::build_vector_add_pipeline(&device, module.raw())
            .map(|pipeline| ComputePipeline::new(pipeline, VECTOR_ADD_ENTRY.to_string()))
    })
}

/// Dispatch the vector-add kernel: `c[i] = a[i] + b[i]` for
/// `i in 0..n`.
///
/// All three buffers must agree in length; mismatches surface as
/// [`Error::MismatchedBufferLengths`] without touching the GPU.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn launch_vector_add_f32(
    device: &Device,
    stream: &SlipStream,
    pipeline: &ComputePipeline,
    a: &DeviceMem<f32>,
    b: &DeviceMem<f32>,
    c: &DeviceMem<f32>,
    n: u32,
    cfg: LaunchConfig,
) -> Io<Error, ()> {
    let device = device.clone();
    let stream = stream.clone();
    let pipeline = pipeline.clone();
    let a = a.clone();
    let b = b.clone();
    let c = c.clone();
    Io::suspend(move || {
        runtime::launch_vector_add_f32(
            &device,
            stream.raw(),
            pipeline.raw(),
            (a.raw(), a.len()),
            (b.raw(), b.len()),
            (c.raw(), c.len()),
            n,
            cfg,
        )
    })
}
