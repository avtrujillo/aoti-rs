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

#[test]
fn any_model_converts_to_typed() {
    use aoti_rs::Device;

    // Generic context: a match can't narrow D to a concrete device type,
    // so this is exactly what try_into_typed exists for.
    fn model_from_any<D: Device>(any: AnyAOTIModel) -> Result<AOTIModel<D>, aoti_rs::Error> {
        any.try_into_typed::<D>()
    }

    let path = pt2_path();
    if !std::path::Path::new(&path).exists() {
        eprintln!("skipping: {path} does not exist");
        return;
    }
    let any = AnyAOTIModel::load_named(&path, &model_name()).expect("load");
    let mut model = model_from_any::<Cpu>(any).expect("typed conversion");
    let outputs = model.run(&[cpu_input()]).expect("run");
    assert_eq!(outputs.len(), 1);

    #[cfg(aoti_cuda)]
    {
        use aoti_rs::{Cuda, Error};
        let any = AnyAOTIModel::load_named(&path, &model_name()).expect("load");
        match model_from_any::<Cuda>(any) {
            Err(Error::ModelDeviceMismatch { expected, found }) => {
                assert_eq!(expected, "cuda");
                assert_eq!(found, "cpu");
            }
            Err(e) => panic!("unexpected error: {e}"),
            Ok(_) => panic!("CPU model must not convert to AOTIModel<Cuda>"),
        }
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

/// Extract the fixture into a fresh temp dir, as an export step would ahead
/// of time. `None` (and the test skips) when the fixture is absent.
fn extracted_fixture() -> Option<tempfile::TempDir> {
    let path = pt2_path();
    if !std::path::Path::new(&path).exists() {
        eprintln!("skipping: {path} does not exist");
        return None;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    aoti_rs::extract_package(&path, dir.path()).expect("extract");
    Some(dir)
}

#[test]
fn extracted_dir_loads_in_place_and_matches_package() {
    let Some(dir) = extracted_fixture() else {
        return;
    };
    let mut in_place = AOTIModel::<Cpu>::builder_extracted(dir.path())
        .model_name(model_name())
        .build()
        .expect("build from extracted dir");
    assert_eq!(
        in_place
            .get_metadata()
            .expect("metadata")
            .get("AOTI_DEVICE_KEY")
            .map(String::as_str),
        Some("cpu")
    );

    // Same weights, so the same input must give the same output as the
    // package-loaded model: in place is a different route to one model.
    let mut packaged = AOTIModel::<Cpu>::builder(pt2_path())
        .model_name(model_name())
        .build()
        .expect("build from package");
    let x = Tensor::randn([2, 4], (tch::Kind::Float, tch::Device::Cpu));
    let input = || DeviceTensor::<Cpu>::try_new(x.shallow_clone()).expect("CPU tensor");
    let a = in_place.run(&[input()]).expect("run in place");
    let b = packaged.run(&[input()]).expect("run packaged");
    assert_eq!(a[0].size(), &[2, 8]);
    assert!(a[0].allclose(&b[0], 0.0, 0.0, false), "outputs differ");
}

#[test]
fn any_model_loads_extracted_dir() {
    let Some(dir) = extracted_fixture() else {
        return;
    };
    match AnyAOTIModel::load_extracted_named(dir.path(), &model_name()).expect("load") {
        AnyAOTIModel::Cpu(mut model) => {
            let outputs = model.run(&[cpu_input()]).expect("run");
            assert_eq!(outputs.len(), 1);
        }
        _ => panic!("expected a CPU model"),
    }
}

#[test]
fn metadata_from_extracted_matches_package() {
    let Some(dir) = extracted_fixture() else {
        return;
    };
    let extracted =
        aoti_rs::load_metadata_from_extracted(dir.path(), &model_name()).expect("extracted");
    let packaged =
        aoti_rs::load_metadata_from_package(&pt2_path(), &model_name()).expect("package");
    assert_eq!(extracted, packaged);
}

#[test]
fn missing_extracted_dir_is_invalid_path() {
    match AOTIModel::<Cpu>::load_extracted("/nonexistent/aoti-rs/extracted") {
        Err(aoti_rs::Error::InvalidPath(msg)) => assert!(msg.contains("does not exist"), "{msg}"),
        Err(e) => panic!("unexpected error: {e}"),
        Ok(_) => panic!("a missing directory must not load"),
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
