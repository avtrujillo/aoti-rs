//! Smoke test: load a tiny .pt2 via the runner path and run inference.
//!
//! Set AOTI_RS_TEST_PT2 to the path of a CPU-exported .pt2 (default
//! `/tmp/tiny_model.pt2`) and AOTI_RS_TEST_MODEL_NAME to the model name
//! used at export time (default `tiny_model`).

use aoti_rs::{AOTIModel, AnyAOTIModel, Cpu, DeviceTensor};
use tch::Tensor;

fn pt2_path() -> String {
    std::env::var("AOTI_RS_TEST_PT2").unwrap_or_else(|_| "/tmp/tiny_model.pt2".to_string())
}

fn model_name() -> String {
    std::env::var("AOTI_RS_TEST_MODEL_NAME").unwrap_or_else(|_| "tiny_model".to_string())
}

fn cpu_input() -> DeviceTensor<Cpu> {
    let x = Tensor::randn([2, 4], (tch::Kind::Float, tch::Device::Cpu));
    DeviceTensor::try_new(x).expect("CPU tensor")
}

#[test]
fn loads_and_runs_tiny_model() {
    let path = pt2_path();
    if !std::path::Path::new(&path).exists() {
        eprintln!("skipping: {path} does not exist");
        return;
    }
    let mut model = AOTIModel::<Cpu>::builder(&path)
        .model_name(model_name())
        .build()
        .expect("build");
    let metadata = model.get_metadata().expect("metadata");
    assert_eq!(
        metadata.get("AOTI_DEVICE_KEY").map(String::as_str),
        Some("cpu")
    );

    let outputs = model.run(&[cpu_input()]).expect("run");
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].size(), &[2, 8]);
    assert_eq!(outputs[0].device(), tch::Device::Cpu);

    // boxed_run consumes its inputs so the runtime may reuse their storage.
    let outputs = model.boxed_run(vec![cpu_input()]).expect("boxed_run");
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].size(), &[2, 8]);
}

#[test]
fn any_model_dispatches_to_cpu() {
    let path = pt2_path();
    if !std::path::Path::new(&path).exists() {
        eprintln!("skipping: {path} does not exist");
        return;
    }
    match AnyAOTIModel::load_named(&path, &model_name()).expect("load") {
        AnyAOTIModel::Cpu(mut model) => {
            let outputs = model.run(&[cpu_input()]).expect("run");
            assert_eq!(outputs.len(), 1);
        }
        _ => panic!("expected a CPU model"),
    }
}

// cfg(aoti_cuda) is emitted by build.rs for every target in this package,
// so integration tests can gate on it too.
#[cfg(aoti_cuda)]
#[test]
fn cpu_package_rejected_as_cuda_model() {
    use aoti_rs::{Cuda, Error};

    let path = pt2_path();
    if !std::path::Path::new(&path).exists() {
        eprintln!("skipping: {path} does not exist");
        return;
    }
    match AOTIModel::<Cuda>::builder(&path)
        .model_name(model_name())
        .build()
    {
        Err(Error::ModelDeviceMismatch { expected, found }) => {
            assert_eq!(expected, "cuda");
            assert_eq!(found, "cpu");
        }
        Err(e) => panic!("unexpected error: {e}"),
        Ok(_) => panic!("CPU package must not load as a CUDA model"),
    }
}

#[test]
fn load_metadata_from_package_streams_json() {
    let path = pt2_path();
    if !std::path::Path::new(&path).exists() {
        eprintln!("skipping: {path} does not exist");
        return;
    }
    let metadata = aoti_rs::load_metadata_from_package(&path, &model_name()).expect("metadata");
    assert_eq!(
        metadata.get("AOTI_DEVICE_KEY").map(String::as_str),
        Some("cpu")
    );
}
