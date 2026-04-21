//! `SlipStream`: our name for a GPU work queue.
//!
//! In `wgpu` terms a [`SlipStream`] wraps the shared `wgpu::Queue` handle.
//! The rename avoids collision with
//! [`comp_cat_rs::effect::stream::Stream`], which is the iterative colimit
//! abstraction over effectful computations.
//!
//! A [`SlipStream`] is the submission endpoint for all launches: dispatches
//! are recorded into a per-call `wgpu::CommandEncoder` inside the runtime
//! and then submitted to the slipstream's queue.

/// A GPU submission endpoint bound to a specific device.
#[derive(Clone)]
pub struct SlipStream {
    queue: wgpu::Queue,
}

impl SlipStream {
    pub(crate) fn new(queue: wgpu::Queue) -> Self {
        Self { queue }
    }

    pub(crate) fn raw(&self) -> &wgpu::Queue {
        &self.queue
    }
}
