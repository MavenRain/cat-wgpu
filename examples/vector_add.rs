//! End-to-end vector-add example.
//!
//! Composes the [`Io`] combinators from [`cat_wgpu::launch`] into a
//! single pipeline:
//!
//!   1. open adapter 0,
//!   2. clone a [`cat_wgpu::SlipStream`] off its queue,
//!   3. compile the bundled WGSL,
//!   4. build the compute pipeline,
//!   5. upload two host vectors,
//!   6. allocate an output buffer,
//!   7. dispatch the kernel,
//!   8. read the output back to the host.
//!
//! Nothing fires until the `.run()` at the end.
//!
//! Run with:
//!
//! ```text
//! cargo run --example vector_add
//! ```

use std::process::ExitCode;

use cat_wgpu::{
    alloc, build_vector_add, compile, default_slipstream, device_resource, download_f32,
    launch_vector_add_f32, upload_f32, DeviceId, Error, LaunchConfig, Wgsl,
};
use comp_cat_rs::effect::io::Io;

const KERNEL_SRC: &str = include_str!("../kernels/vector_add.wgsl");
const THREADS_PER_BLOCK: u32 = 64;
const N: u32 = 256;
const N_USIZE: usize = 256;

fn run_example() -> Result<(), Error> {
    let a_host: Vec<f32> = (0..N).map(u16_truncate).map(f32::from).collect();
    let b_host: Vec<f32> = (0..N)
        .map(u16_truncate)
        .map(|i| f32::from(i.saturating_mul(2)))
        .collect();
    let expected: Vec<f32> = a_host
        .iter()
        .zip(b_host.iter())
        .map(|(x, y)| x + y)
        .collect();

    let pipeline: Io<Error, Vec<f32>> =
        device_resource(DeviceId::new(0)).use_resource(move |device| {
            let dev_stream = device.clone();
            let dev_compile = device.clone();
            let dev_build = device.clone();
            let dev_upload_a = device.clone();
            let dev_upload_b = device.clone();
            let dev_alloc = device.clone();
            let dev_launch = device.clone();
            let dev_download = device.clone();

            default_slipstream(&dev_stream).flat_map(move |stream| {
                compile(&dev_compile, Wgsl::new(KERNEL_SRC.to_string())).flat_map(
                    move |module| {
                        build_vector_add(&dev_build, &module).flat_map(move |pipe| {
                            upload_f32(&dev_upload_a, a_host.clone()).flat_map(
                                move |a_dev| {
                                    upload_f32(&dev_upload_b, b_host.clone()).flat_map(
                                        move |b_dev| {
                                            alloc::<f32>(&dev_alloc, N_USIZE).flat_map(
                                                move |c_dev| {
                                                    let cfg = LaunchConfig::for_num_elems(
                                                        N,
                                                        THREADS_PER_BLOCK,
                                                    );
                                                    launch_vector_add_f32(
                                                        &dev_launch,
                                                        &stream,
                                                        &pipe,
                                                        &a_dev,
                                                        &b_dev,
                                                        &c_dev,
                                                        N,
                                                        cfg,
                                                    )
                                                    .flat_map(move |()| {
                                                        download_f32(&dev_download, &c_dev)
                                                    })
                                                },
                                            )
                                        },
                                    )
                                },
                            )
                        })
                    },
                )
            })
        });

    pipeline.run().and_then(|result| {
        let matches = result.len() == expected.len()
            && result
                .iter()
                .zip(expected.iter())
                .all(|(got, want)| (got - want).abs() < f32::EPSILON);
        match () {
            () if matches => {
                println!(
                    "cat-wgpu vector_add: {} elements matched host reference",
                    result.len()
                );
                Ok(())
            }
            () => Err(Error::DispatchFailed(format!(
                "result mismatch: got {} elems, want {}",
                result.len(),
                expected.len()
            ))),
        }
    })
}

fn u16_truncate(i: u32) -> u16 {
    u16::try_from(i).unwrap_or(u16::MAX)
}

fn main() -> ExitCode {
    run_example().map_or_else(
        |err| {
            eprintln!("cat-wgpu vector_add failed: {err}");
            ExitCode::from(1)
        },
        |()| ExitCode::SUCCESS,
    )
}
