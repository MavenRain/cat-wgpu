//! Hand-rolled error enum for the cat-wgpu crate.
//!
//! Every variant wraps a specific failure mode at a specific domain
//! boundary.  `Display` and `std::error::Error` are implemented by hand.
//! No `thiserror`, no `anyhow`.

use core::fmt;

/// Direction of a host <-> device memory copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopyDirection {
    /// Host memory into device memory.
    HostToDevice,
    /// Device memory back out to host memory.
    DeviceToHost,
}

impl fmt::Display for CopyDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HostToDevice => f.write_str("host -> device"),
            Self::DeviceToHost => f.write_str("device -> host"),
        }
    }
}

/// Every failure that the cat-wgpu surface can report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// `wgpu::Instance::request_adapter` returned `None`: no backend
    /// matched the request (no GPU, no compatible driver, or the enabled
    /// cargo features rule out every adapter on this host).
    NoAdapter,
    /// The caller requested an adapter index that the instance does not
    /// expose.
    InvalidDeviceIndex(u32),
    /// `Adapter::request_device` failed.  The message is wgpu's own
    /// stringification of the limits or features negotiation error.
    DeviceRequestFailed(String),
    /// A buffer allocation exceeded what the adapter can provide or hit
    /// wgpu's buffer-size validation cap.
    BufferAllocationFailed {
        /// Bytes requested at the failing allocation site.
        requested: usize,
    },
    /// WGSL compilation or module creation failed.
    ShaderCompileFailed {
        /// Entry point the caller was trying to bind.
        entry: String,
        /// Diagnostics collected from the validation error scope.
        diagnostics: String,
    },
    /// Compute pipeline construction failed after the shader loaded.
    PipelineBuildFailed(String),
    /// A compute dispatch surfaced a validation error.
    DispatchFailed(String),
    /// A memory copy failed for a known byte count and direction.
    CopyFailed {
        /// Bytes the failing copy attempted to move.
        bytes: usize,
        /// Direction of the failing copy.
        direction: CopyDirection,
    },
    /// A `Buffer::map_async` callback returned an error instead of a
    /// readable range.
    BufferMapFailed(String),
    /// A launch saw buffers of disagreeing lengths where the kernel
    /// requires them to line up.
    MismatchedBufferLengths {
        /// First operand length (elements).
        a: usize,
        /// Second operand length (elements).
        b: usize,
        /// Output length (elements).
        c: usize,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAdapter => f.write_str("no compatible wgpu adapter available"),
            Self::InvalidDeviceIndex(idx) => {
                write!(f, "invalid adapter index {idx}: not present on this host")
            }
            Self::DeviceRequestFailed(msg) => write!(f, "request_device failed: {msg}"),
            Self::BufferAllocationFailed { requested } => {
                write!(f, "buffer allocation failed: {requested} bytes requested")
            }
            Self::ShaderCompileFailed { entry, diagnostics } => write!(
                f,
                "shader compile failed for entry '{entry}': {diagnostics}"
            ),
            Self::PipelineBuildFailed(msg) => {
                write!(f, "compute pipeline build failed: {msg}")
            }
            Self::DispatchFailed(msg) => write!(f, "compute dispatch failed: {msg}"),
            Self::CopyFailed { bytes, direction } => {
                write!(f, "{direction} copy of {bytes} bytes failed")
            }
            Self::BufferMapFailed(msg) => write!(f, "buffer map_async failed: {msg}"),
            Self::MismatchedBufferLengths { a, b, c } => write!(
                f,
                "buffer length mismatch: a={a}, b={b}, c={c}"
            ),
        }
    }
}

impl std::error::Error for Error {}

impl From<wgpu::RequestDeviceError> for Error {
    fn from(e: wgpu::RequestDeviceError) -> Self {
        Self::DeviceRequestFailed(e.to_string())
    }
}

impl From<wgpu::BufferAsyncError> for Error {
    fn from(e: wgpu::BufferAsyncError) -> Self {
        Self::BufferMapFailed(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::{CopyDirection, Error};

    #[test]
    fn display_covers_every_variant() {
        let cases = [
            Error::NoAdapter,
            Error::InvalidDeviceIndex(3),
            Error::DeviceRequestFailed("features not supported".into()),
            Error::BufferAllocationFailed { requested: 1 << 30 },
            Error::ShaderCompileFailed {
                entry: "vector_add".into(),
                diagnostics: "unknown identifier".into(),
            },
            Error::PipelineBuildFailed("bind group mismatch".into()),
            Error::DispatchFailed("buffer out of bounds".into()),
            Error::CopyFailed { bytes: 64, direction: CopyDirection::HostToDevice },
            Error::CopyFailed { bytes: 64, direction: CopyDirection::DeviceToHost },
            Error::BufferMapFailed("device lost".into()),
            Error::MismatchedBufferLengths { a: 4, b: 4, c: 8 },
        ];
        assert!(cases.iter().all(|e| !format!("{e}").is_empty()));
    }
}
