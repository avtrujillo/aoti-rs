//! Smoke test: load a tiny .pt2 via the runner path and run inference.
//!
//! Set AOTI_RS_TEST_PT2 to the path of a CPU-exported .pt2 (default
//! `/tmp/tiny_model.pt2`) and AOTI_RS_TEST_MODEL_NAME to the model name
//! used at export time (default `tiny_model`).

use aoti_rs::AOTIModel;
use tch::Tensor;

fn pt2_path() -> String {
    std::env::var("AOTI_RS_TEST_PT2").unwrap_or_else(|_| "/tmp/tiny_model.pt2".to_string())
}

fn model_name() -> String {
    std::env::var("AOTI_RS_TEST_MODEL_NAME").unwrap_or_else(|_| "tiny_model".to_string())
}

#[test]
fn loads_and_runs_tiny_model() {
    let path = pt2_path();
    if !std::path::Path::new(&path).exists() {
        eprintln!("skipping: {path} does not exist");
        return;
    }
    let mut model = AOTIModel::builder(&path)
        .model_name(model_name())
        .build()
        .expect("build");
    let metadata = model.get_metadata().expect("metadata");
    assert_eq!(metadata.get("AOTI_DEVICE_KEY").map(String::as_str), Some("cpu"));

    let x = Tensor::randn([2, 4], (tch::Kind::Float, tch::Device::Cpu));
    let outputs = model.run(&[x]).expect("run");
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].size(), &[2, 8]);
}

#[test]
fn load_metadata_from_package_streams_json() {
    let path = pt2_path();
    if !std::path::Path::new(&path).exists() {
        eprintln!("skipping: {path} does not exist");
        return;
    }
    let metadata = AOTIModel::load_metadata_from_package(&path, &model_name()).expect("metadata");
    assert_eq!(metadata.get("AOTI_DEVICE_KEY").map(String::as_str), Some("cpu"));
}
