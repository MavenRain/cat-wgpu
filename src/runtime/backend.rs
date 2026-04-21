//! wgpu driver plumbing.
//!
//! This is the single file in the crate where `let mut` is permitted, per
//! [`CLAUDE.md`](../../../CLAUDE.md).  The exception is load-bearing for
//! `wgpu`'s compute-recording API (`CommandEncoder::begin_compute_pass`,
//! `ComputePass::set_pipeline`, `ComputePass::dispatch_workgroups`), which
//! is entirely `&mut self`.
//!
//! Every entry point here is a plain synchronous function that returns
//! `Result<_, crate::Error>` and runs wgpu's async machinery through
//! [`pollster::block_on`] at the boundary.  Callers in [`crate::launch`]
//! wrap these in `Io::suspend` so the side effects stay deferred.

use std::borrow::Cow;

use bytemuck::Pod;
use wgpu::util::DeviceExt;

use crate::device::{Device, DeviceId};
use crate::error::{CopyDirection, Error};
use crate::kernel::LaunchConfig;

const VECTOR_ADD_ENTRY: &str = "vector_add";

/// Pick the adapter at ordinal `id.index()` and open a wgpu device +
/// queue pair on it.
pub(crate) fn pick_adapter_and_device(id: DeviceId) -> Result<Device, Error> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
    let idx = usize::try_from(id.index())
        .map_err(|_| Error::InvalidDeviceIndex(id.index()))?;
    let adapter = adapters
        .into_iter()
        .nth(idx)
        .ok_or(Error::InvalidDeviceIndex(id.index()))?;
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("cat-wgpu device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        },
    ))?;
    Ok(Device::new(id, device, queue))
}

/// Clone the device's queue handle to serve as the default
/// [`crate::SlipStream`] endpoint.
pub(crate) fn default_queue(device: &Device) -> wgpu::Queue {
    device.raw_queue().clone()
}

/// Allocate a zero-initialised storage buffer of `len` elements of type
/// `T`.
pub(crate) fn alloc_storage_buffer<T: Pod>(
    device: &Device,
    len: usize,
) -> Result<(wgpu::Buffer, usize), Error> {
    let size_bytes = len
        .checked_mul(core::mem::size_of::<T>())
        .ok_or(Error::BufferAllocationFailed { requested: usize::MAX })?;
    let buffer = device.raw_device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("cat-wgpu storage buffer"),
        size: u64::try_from(size_bytes)
            .map_err(|_| Error::BufferAllocationFailed { requested: size_bytes })?,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    Ok((buffer, len))
}

/// Upload a host `f32` slice to a storage buffer.
pub(crate) fn upload_f32_buffer(device: &Device, data: &[f32]) -> (wgpu::Buffer, usize) {
    let len = data.len();
    let bytes = bytemuck::cast_slice::<f32, u8>(data);
    let buffer = device
        .raw_device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cat-wgpu upload buffer"),
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
    (buffer, len)
}

/// Download a storage buffer back to a host `Vec<f32>`.
///
/// Internally records a copy into a mappable staging buffer, submits,
/// synchronously polls the device to completion, and then copies the
/// mapped range into an owned `Vec`.
pub(crate) fn download_f32_buffer(
    device: &Device,
    source: &wgpu::Buffer,
    len: usize,
) -> Result<Vec<f32>, Error> {
    let size_bytes = len
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or(Error::CopyFailed {
            bytes: usize::MAX,
            direction: CopyDirection::DeviceToHost,
        })?;
    let size_u64 = u64::try_from(size_bytes).map_err(|_| Error::CopyFailed {
        bytes: size_bytes,
        direction: CopyDirection::DeviceToHost,
    })?;
    let staging = device.raw_device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("cat-wgpu download staging"),
        size: size_u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder =
        device
            .raw_device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cat-wgpu download encoder"),
            });
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, size_u64);
    let cb = encoder.finish();
    device.raw_queue().submit([cb]);

    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = sender.send(res);
    });
    device
        .raw_device()
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|_| Error::CopyFailed {
            bytes: size_bytes,
            direction: CopyDirection::DeviceToHost,
        })
        .and_then(|_| {
            receiver.recv().map_err(|_| Error::CopyFailed {
                bytes: size_bytes,
                direction: CopyDirection::DeviceToHost,
            })
        })?
        .map_err(Error::from)?;
    let view = slice.get_mapped_range();
    let out = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
    drop(view);
    staging.unmap();
    Ok(out)
}

/// Compile a WGSL source string into a `wgpu::ShaderModule`.
pub(crate) fn compile_wgsl(
    device: &Device,
    source: &str,
) -> Result<wgpu::ShaderModule, Error> {
    let guard = device
        .raw_device()
        .push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device
        .raw_device()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cat-wgpu shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(source.to_string())),
        });
    let scope = pollster::block_on(guard.pop());
    scope.map_or(Ok(module), |err| {
        Err(Error::ShaderCompileFailed {
            entry: VECTOR_ADD_ENTRY.to_string(),
            diagnostics: err.to_string(),
        })
    })
}

/// Build the compute pipeline for the vector-add kernel.
pub(crate) fn build_vector_add_pipeline(
    device: &Device,
    module: &wgpu::ShaderModule,
) -> Result<wgpu::ComputePipeline, Error> {
    let guard = device
        .raw_device()
        .push_error_scope(wgpu::ErrorFilter::Validation);
    let pipeline =
        device
            .raw_device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("cat-wgpu vector-add pipeline"),
                layout: None,
                module,
                entry_point: Some(VECTOR_ADD_ENTRY),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
    let scope = pollster::block_on(guard.pop());
    scope.map_or(Ok(pipeline), |err| {
        Err(Error::PipelineBuildFailed(err.to_string()))
    })
}

/// Launch the vector-add kernel against three f32 storage buffers.
///
/// `c[i] = a[i] + b[i]` for `i in 0..n`, where `n` is typically the common
/// length of `a`, `b`, and `c`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_vector_add_f32(
    device: &Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    a: (&wgpu::Buffer, usize),
    b: (&wgpu::Buffer, usize),
    c: (&wgpu::Buffer, usize),
    n: u32,
    cfg: LaunchConfig,
) -> Result<(), Error> {
    match () {
        () if a.1 != b.1 || b.1 != c.1 => {
            Err(Error::MismatchedBufferLengths {
                a: a.1,
                b: b.1,
                c: c.1,
            })
        }
        () => launch_vector_add_f32_inner(device, queue, pipeline, a.0, b.0, c.0, n, cfg),
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_vector_add_f32_inner(
    device: &Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    c: &wgpu::Buffer,
    n: u32,
    cfg: LaunchConfig,
) -> Result<(), Error> {
    let layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.raw_device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cat-wgpu vector-add bind group"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: c.as_entire_binding(),
            },
        ],
    });

    let guard = device
        .raw_device()
        .push_error_scope(wgpu::ErrorFilter::Validation);
    let mut encoder =
        device
            .raw_device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cat-wgpu vector-add encoder"),
            });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cat-wgpu vector-add pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let grid = cfg.grid();
        pass.dispatch_workgroups(grid.x(), grid.y(), grid.z());
    }
    let cb = encoder.finish();
    queue.submit([cb]);
    let _ = n;
    let scope = pollster::block_on(guard.pop());
    scope.map_or(Ok(()), |err| Err(Error::DispatchFailed(err.to_string())))
}
