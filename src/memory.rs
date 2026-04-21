//! Typed device memory.
//!
//! [`DeviceMem<T>`] is a typed handle to a `wgpu::Buffer` that lives on the
//! active adapter.  The element type `T` must be `bytemuck::Pod` so we can
//! safely reinterpret the host-side `Vec<T>` as `&[u8]` for upload and back
//! again after readback.
//!
//! Release of the underlying buffer is handled by `wgpu::Buffer`'s own
//! `Drop`, which schedules destruction on the next queue flush.  The
//! `Resource<Error, DeviceMem<T>>` bracket in [`crate::launch`] just makes
//! the release point visible in the categorical graph.

use core::marker::PhantomData;

/// Owned typed storage buffer on the active device.
#[derive(Clone)]
pub struct DeviceMem<T> {
    buffer: wgpu::Buffer,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T> DeviceMem<T> {
    pub(crate) fn new(buffer: wgpu::Buffer, len: usize) -> Self {
        Self {
            buffer,
            len,
            _marker: PhantomData,
        }
    }

    /// Element count of the buffer (not byte count).
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer holds zero elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(crate) fn raw(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}
