//! Shaders, pipelines, and launch geometry.

/// Owned WGSL source text.
///
/// The bytes are copied into [`ShaderModule`] when compiled onto a device;
/// the `Wgsl` value is not required to outlive the compile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Wgsl(String);

impl Wgsl {
    /// Wrap a WGSL string.
    #[must_use]
    pub fn new(source: String) -> Self {
        Self(source)
    }

    /// Borrow the underlying source.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Element count in bytes (same as the UTF-8 length).
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the source is zero bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl From<String> for Wgsl {
    fn from(source: String) -> Self {
        Self(source)
    }
}

impl From<&str> for Wgsl {
    fn from(source: &str) -> Self {
        Self(source.to_string())
    }
}

/// A WGSL module compiled onto a specific device.
#[derive(Clone)]
pub struct ShaderModule {
    module: wgpu::ShaderModule,
}

impl ShaderModule {
    pub(crate) fn new(module: wgpu::ShaderModule) -> Self {
        Self { module }
    }

    pub(crate) fn raw(&self) -> &wgpu::ShaderModule {
        &self.module
    }
}

/// A compute pipeline with its bind-group layout pinned, ready to dispatch.
#[derive(Clone)]
pub struct ComputePipeline {
    pipeline: wgpu::ComputePipeline,
    entry: String,
}

impl ComputePipeline {
    pub(crate) fn new(pipeline: wgpu::ComputePipeline, entry: String) -> Self {
        Self { pipeline, entry }
    }

    /// Entry-point name this pipeline dispatches into.
    #[must_use]
    pub fn entry(&self) -> &str {
        &self.entry
    }

    pub(crate) fn raw(&self) -> &wgpu::ComputePipeline {
        &self.pipeline
    }
}

/// Workgroup grid dimensions (how many workgroups to launch per axis).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridDim {
    x: u32,
    y: u32,
    z: u32,
}

impl GridDim {
    /// Construct a 3-D grid.
    #[must_use]
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Construct a 1-D grid (y = z = 1).
    #[must_use]
    pub fn linear(x: u32) -> Self {
        Self { x, y: 1, z: 1 }
    }

    /// Workgroup count along the x axis.
    #[must_use]
    pub fn x(self) -> u32 {
        self.x
    }

    /// Workgroup count along the y axis.
    #[must_use]
    pub fn y(self) -> u32 {
        self.y
    }

    /// Workgroup count along the z axis.
    #[must_use]
    pub fn z(self) -> u32 {
        self.z
    }
}

/// Workgroup size (threads per workgroup per axis).
///
/// Carried as a host-side hint.  The authoritative workgroup size is the
/// `@workgroup_size(...)` attribute on the WGSL entry point; host-side
/// [`BlockDim`] exists so launch logic that ceils grid sizes can reason
/// about it without re-parsing the shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockDim {
    x: u32,
    y: u32,
    z: u32,
}

impl BlockDim {
    /// Construct a 3-D block.
    #[must_use]
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Construct a 1-D block (y = z = 1).
    #[must_use]
    pub fn linear(x: u32) -> Self {
        Self { x, y: 1, z: 1 }
    }

    /// Thread count along the x axis.
    #[must_use]
    pub fn x(self) -> u32 {
        self.x
    }

    /// Thread count along the y axis.
    #[must_use]
    pub fn y(self) -> u32 {
        self.y
    }

    /// Thread count along the z axis.
    #[must_use]
    pub fn z(self) -> u32 {
        self.z
    }
}

/// Host-side description of a launch: grid plus host-mirrored workgroup
/// size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchConfig {
    grid: GridDim,
    block: BlockDim,
}

impl LaunchConfig {
    /// Build an explicit grid + block config.
    #[must_use]
    pub fn new(grid: GridDim, block: BlockDim) -> Self {
        Self { grid, block }
    }

    /// Build a 1-D config that covers `n` elements with `threads_per_block`
    /// threads per workgroup.  The grid ceils `n` up to the next multiple
    /// of `threads_per_block`.
    #[must_use]
    pub fn for_num_elems(n: u32, threads_per_block: u32) -> Self {
        let blocks = match () {
            () if threads_per_block == 0 => 0,
            () => n.div_ceil(threads_per_block),
        };
        Self {
            grid: GridDim::linear(blocks),
            block: BlockDim::linear(threads_per_block),
        }
    }

    /// Grid geometry.
    #[must_use]
    pub fn grid(self) -> GridDim {
        self.grid
    }

    /// Block geometry.
    #[must_use]
    pub fn block(self) -> BlockDim {
        self.block
    }
}
