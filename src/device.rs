//! Device identification and handle.
//!
//! A [`DeviceId`] is an ordinal into `wgpu::Instance::enumerate_adapters`.
//! A [`Device`] bundles the id with the `wgpu::Device` and `wgpu::Queue`
//! pair that drives all subsequent operations.

use core::marker::PhantomData;

/// Ordinal index into the list of adapters a `wgpu::Instance` enumerates
/// in its current backend set.
///
/// Construct via [`DeviceId::new`] or `From<u32>`.  Index 0 is always the
/// adapter wgpu would pick for
/// [`wgpu::PowerPreference::HighPerformance`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceId(u32);

impl DeviceId {
    /// Wrap a raw ordinal.
    #[must_use]
    pub fn new(index: u32) -> Self {
        Self(index)
    }

    /// Read the ordinal back out.
    #[must_use]
    pub fn index(self) -> u32 {
        self.0
    }
}

impl From<u32> for DeviceId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl From<DeviceId> for u32 {
    fn from(id: DeviceId) -> Self {
        id.0
    }
}

/// An initialised `wgpu` device + queue pair.
///
/// Cloning is cheap: `wgpu::Device` and `wgpu::Queue` both hold an
/// `Arc` internally, so a `Device` is a handle, not an owning resource.
/// Drop of the last `Device` tears down the wgpu device, which in turn
/// releases all buffers and pipelines it spawned.
#[derive(Clone)]
pub struct Device {
    id: DeviceId,
    handle: wgpu::Device,
    queue: wgpu::Queue,
    _marker: PhantomData<()>,
}

impl Device {
    pub(crate) fn new(id: DeviceId, handle: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self {
            id,
            handle,
            queue,
            _marker: PhantomData,
        }
    }

    /// Ordinal of the adapter this context was opened for.
    #[must_use]
    pub fn id(&self) -> DeviceId {
        self.id
    }

    pub(crate) fn raw_device(&self) -> &wgpu::Device {
        &self.handle
    }

    pub(crate) fn raw_queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}
