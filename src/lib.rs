use std::collections::HashMap;
use std::path::{Path, PathBuf};

use tch::Tensor;
use tempfile::TempDir;

#[cxx::bridge(namespace = "aoti_rs")]
mod ffi {
    #[namespace = ""]
    unsafe extern "C++" {
        include!("csrc/cvoid.h");
        type c_void;
    }

    struct TensorPtr {
        ptr: *const c_void,
    }

    struct OwnedTensor {
        ptr: *mut c_void,
    }

    #[namespace = "torch::inductor"]
    unsafe extern "C++" {
        type AOTIModelContainerRunner;
    }

    unsafe extern "C++" {
        include!("csrc/aoti.h");

        fn runner_new(
            model_so_path: &str,
            cubin_dir: &str,
            is_cuda: bool,
            device_index: i8,
            num_runners: usize,
            run_single_threaded: bool,
        ) -> Result<UniquePtr<AOTIModelContainerRunner>>;

        fn runner_run(
            runner: Pin<&mut AOTIModelContainerRunner>,
            inputs: &Vec<TensorPtr>,
        ) -> Result<Vec<OwnedTensor>>;

        fn runner_boxed_run(
            runner: Pin<&mut AOTIModelContainerRunner>,
            inputs: &mut Vec<TensorPtr>,
        ) -> Result<Vec<OwnedTensor>>;

        fn runner_get_call_spec(
            runner: Pin<&mut AOTIModelContainerRunner>,
        ) -> Result<Vec<String>>;

        fn runner_get_constant_fqns(
            runner: Pin<&mut AOTIModelContainerRunner>,
        ) -> Result<Vec<String>>;
    }
}

/// Errors returned by [`AOTIModel`] operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("zip error: {0}")]
    Zip(#[from] zip::result::ZipError),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error(transparent)]
    Ffi(#[from] cxx::Exception),

    #[error("invalid path: {0}")]
    InvalidPath(String),

    #[error("model error: {0}")]
    Model(String),
}

/// Convert a slice of `tch::Tensor` references into `TensorPtr` values for the FFI boundary.
fn tensors_to_ptrs(tensors: &[Tensor]) -> Vec<ffi::TensorPtr> {
    tensors
        .iter()
        .map(|t| ffi::TensorPtr {
            ptr: t.as_ptr() as *const ffi::c_void,
        })
        .collect()
}

/// Convert `OwnedTensor` values from C++ into `tch::Tensor` values.
///
/// # Safety
/// Each `OwnedTensor::ptr` must be a valid, heap-allocated `at::Tensor*`
/// created with `new at::Tensor(...)` on the C++ side. Ownership is transferred
/// to the returned `tch::Tensor` which will free the underlying storage on drop.
fn owned_to_tensors(owned: Vec<ffi::OwnedTensor>) -> Vec<Tensor> {
    owned
        .into_iter()
        .map(|ot| unsafe { Tensor::from_ptr(ot.ptr as *mut _) })
        .collect()
}

/// Extract every non-directory entry of a `.pt2` archive into a fresh temp dir,
/// using a Zip64-aware reader.  This bypasses libtorch's miniz-based extractor,
/// which fails on archives whose internal `wrapper.so` pushes the central
/// directory past the 32-bit Zip offset boundary.
fn extract_pt2(pt2_path: &str) -> Result<TempDir, Error> {
    let file = std::fs::File::open(pt2_path)?;
    let mut archive = zip::ZipArchive::new(file)?;
    let dir = tempfile::tempdir()?;
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let name = entry.name().to_string();
        // Skip directory marker entries (the source of the original miniz bug).
        if name.ends_with('/') {
            continue;
        }
        let outpath = match entry.enclosed_name() {
            Some(p) => dir.path().join(p),
            None => dir.path().join(&name),
        };
        if let Some(parent) = outpath.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut out = std::fs::File::create(&outpath)?;
        std::io::copy(&mut entry, &mut out)?;
    }
    Ok(dir)
}

/// Walk `root` recursively and return every file whose name matches `predicate`.
fn find_files<F: Fn(&str) -> bool>(root: &Path, predicate: F) -> std::io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if !dir.is_dir() {
            continue;
        }
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            let ft = entry.file_type()?;
            if ft.is_dir() {
                stack.push(path);
            } else if ft.is_file() {
                let name = entry.file_name();
                if let Some(s) = name.to_str()
                    && predicate(s)
                {
                    out.push(path);
                }
            }
        }
    }
    Ok(out)
}

/// Locate the `wrapper.so` for `model_name` inside an extracted `.pt2` directory.
///
/// The conventional path is `<dir>/<model_name>/data/aotinductor/.../<hash>.wrapper.so`,
/// but we walk the whole `data/aotinductor/` subtree to be robust to layout
/// changes across PyTorch versions.
fn find_wrapper_so(dir: &Path, model_name: &str) -> Result<PathBuf, Error> {
    let aotinductor = dir.join(model_name).join("data").join("aotinductor");
    let search_root = if aotinductor.is_dir() { aotinductor } else { dir.to_path_buf() };

    let mut hits = find_files(&search_root, |n| n.ends_with(".wrapper.so"))?;
    if hits.is_empty() {
        hits = find_files(&search_root, |n| n.ends_with(".so"))?;
    }

    match hits.len() {
        0 => Err(Error::Model(format!(
            "no wrapper.so found under {}",
            search_root.display()
        ))),
        1 => Ok(hits.pop().unwrap()),
        _ => Err(Error::Model(format!(
            "multiple .so candidates under {}: {:?}",
            search_root.display(),
            hits
        ))),
    }
}

/// Parse a `*_metadata.json` (a flat string→string map) into a `HashMap`.
fn parse_metadata_json(bytes: &[u8]) -> Result<HashMap<String, String>, Error> {
    let value: serde_json::Value = serde_json::from_slice(bytes)?;
    let obj = value
        .as_object()
        .ok_or_else(|| Error::Model("metadata JSON is not an object".into()))?;
    let mut map = HashMap::with_capacity(obj.len());
    for (k, v) in obj {
        let s = match v {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        map.insert(k.clone(), s);
    }
    Ok(map)
}

/// Find and read the metadata JSON file inside an extracted `.pt2` directory.
/// Returns an empty map if no `*_metadata.json` is present (some packages
/// don't ship one).
fn read_metadata_from_dir(
    dir: &Path,
    model_name: &str,
) -> Result<HashMap<String, String>, Error> {
    let model_root = dir.join(model_name);
    let search_root = if model_root.is_dir() { model_root } else { dir.to_path_buf() };
    let hits = find_files(&search_root, |n| n.ends_with("_metadata.json") || n == "metadata.json")?;
    let Some(path) = hits.into_iter().next() else {
        return Ok(HashMap::new());
    };
    let bytes = std::fs::read(&path)?;
    parse_metadata_json(&bytes)
}

/// Read metadata directly from a `.pt2` archive without fully extracting it,
/// by streaming a single `*_metadata.json` entry.
fn read_metadata_from_zip(
    pt2_path: &str,
    model_name: &str,
) -> Result<HashMap<String, String>, Error> {
    let file = std::fs::File::open(pt2_path)?;
    let mut archive = zip::ZipArchive::new(file)?;
    let prefix = format!("{model_name}/");
    // Prefer entries scoped to the requested model; fall back to any metadata file.
    let mut chosen: Option<usize> = None;
    let mut fallback: Option<usize> = None;
    for i in 0..archive.len() {
        let entry = archive.by_index(i)?;
        let name = entry.name();
        if name.ends_with('/') {
            continue;
        }
        let is_metadata = name.ends_with("_metadata.json")
            || name.ends_with("/metadata.json")
            || name == "metadata.json";
        if !is_metadata {
            continue;
        }
        if name.starts_with(&prefix) {
            chosen = Some(i);
            break;
        } else if fallback.is_none() {
            fallback = Some(i);
        }
    }
    let Some(idx) = chosen.or(fallback) else {
        return Ok(HashMap::new());
    };
    let mut entry = archive.by_index(idx)?;
    let mut buf = Vec::with_capacity(entry.size() as usize);
    std::io::copy(&mut entry, &mut buf)?;
    parse_metadata_json(&buf)
}

/// Builder for configuring and creating an [`AOTIModel`].
pub struct AOTIModelBuilder {
    path: String,
    model_name: String,
    run_single_threaded: bool,
    num_runners: usize,
    device_index: i8,
}

impl AOTIModelBuilder {
    /// Create a builder for the given `.pt2` model package path.
    pub fn new(model_package_path: impl Into<String>) -> Self {
        Self {
            path: model_package_path.into(),
            model_name: "model".to_string(),
            run_single_threaded: false,
            num_runners: 1,
            device_index: -1,
        }
    }

    /// Set the model name within the package (default: `"model"`).
    pub fn model_name(mut self, name: impl Into<String>) -> Self {
        self.model_name = name.into();
        self
    }

    /// Run in single-threaded mode, avoiding thread synchronization overhead.
    /// Useful when running under CUDA graphs.
    pub fn single_threaded(mut self, single_threaded: bool) -> Self {
        self.run_single_threaded = single_threaded;
        self
    }

    /// Set the number of runner instances (default: 1).
    pub fn num_runners(mut self, n: usize) -> Self {
        self.num_runners = n;
        self
    }

    /// Set the CUDA device index (default: -1 for the current default device).
    pub fn device_index(mut self, idx: i8) -> Self {
        self.device_index = idx;
        self
    }

    /// Build the model, extracting the package and constructing the runner.
    pub fn build(self) -> Result<AOTIModel, Error> {
        let temp_dir = extract_pt2(&self.path)?;
        let so_path = find_wrapper_so(temp_dir.path(), &self.model_name)?;
        let metadata = read_metadata_from_dir(temp_dir.path(), &self.model_name)?;

        let is_cuda = metadata
            .get("AOTI_DEVICE_KEY")
            .map(|v| v == "cuda")
            .unwrap_or(false);

        let cubin_dir = so_path
            .parent()
            .ok_or_else(|| Error::InvalidPath(format!("{} has no parent", so_path.display())))?
            .to_string_lossy()
            .into_owned();
        let so_path_str = so_path
            .to_str()
            .ok_or_else(|| Error::InvalidPath(format!("{} is not valid UTF-8", so_path.display())))?;

        let inner = ffi::runner_new(
            so_path_str,
            &cubin_dir,
            is_cuda,
            self.device_index,
            self.num_runners,
            self.run_single_threaded,
        )?;

        Ok(AOTIModel {
            inner,
            metadata,
            _temp_dir: temp_dir,
        })
    }
}

/// A loaded AOT-compiled PyTorch model, ready for inference.
///
/// Wraps `torch::inductor::AOTIModelContainerRunner` from libtorch.  The `.pt2`
/// archive is extracted in Rust and the resulting `wrapper.so` is handed to the
/// runner directly, bypassing `AOTIModelPackageLoader`'s miniz-based extractor.
///
/// # Example
///
/// ```no_run
/// use aoti_rs::AOTIModel;
/// use tch::Tensor;
///
/// let mut model = AOTIModel::load("model.pt2").unwrap();
/// let input = Tensor::randn([1, 3, 224, 224], (tch::Kind::Float, tch::Device::Cpu));
/// let outputs = model.run(&[input]).unwrap();
/// ```
pub struct AOTIModel {
    inner: cxx::UniquePtr<ffi::AOTIModelContainerRunner>,
    metadata: HashMap<String, String>,
    // The runner mmaps `wrapper.so` and reads `.cubin` kernel files lazily
    // during inference, so the extracted directory must outlive `inner`.
    _temp_dir: TempDir,
}

// Safety: AOTIModelContainerRunner manages its own thread safety via num_runners.
// When num_runners > 1, concurrent run() calls are safe. The user controls this
// via the builder. Single-runner use should be externally synchronized.
unsafe impl Send for AOTIModel {}

impl AOTIModel {
    /// Load a `.pt2` model package with default settings.
    pub fn load(model_package_path: impl Into<String>) -> Result<Self, Error> {
        AOTIModelBuilder::new(model_package_path).build()
    }

    /// Create a builder for more control over loading options.
    pub fn builder(model_package_path: impl Into<String>) -> AOTIModelBuilder {
        AOTIModelBuilder::new(model_package_path)
    }

    /// Run inference on the given input tensors.
    ///
    /// Input tensors must match the shapes, dtypes, and device used during
    /// model export.
    pub fn run(&mut self, inputs: &[Tensor]) -> Result<Vec<Tensor>, Error> {
        let ptrs = tensors_to_ptrs(inputs);
        let owned = ffi::runner_run(self.inner.pin_mut(), &ptrs)?;
        Ok(owned_to_tensors(owned))
    }

    /// Run inference, allowing the runtime to take ownership of input tensors
    /// for potential in-place optimization.
    pub fn boxed_run(&mut self, inputs: &[Tensor]) -> Result<Vec<Tensor>, Error> {
        let mut ptrs = tensors_to_ptrs(inputs);
        let owned = ffi::runner_boxed_run(self.inner.pin_mut(), &mut ptrs)?;
        Ok(owned_to_tensors(owned))
    }

    /// Get model metadata as a key-value map.
    ///
    /// Typical keys include `"AOTI_DEVICE_KEY"` indicating the target device.
    pub fn get_metadata(&self) -> Result<HashMap<String, String>, Error> {
        Ok(self.metadata.clone())
    }

    /// Get the call specification strings for the model.
    pub fn get_call_spec(&mut self) -> Result<Vec<String>, Error> {
        Ok(ffi::runner_get_call_spec(self.inner.pin_mut())?)
    }

    /// Get the fully qualified names of all constants in the model.
    pub fn get_constant_fqns(&mut self) -> Result<Vec<String>, Error> {
        Ok(ffi::runner_get_constant_fqns(self.inner.pin_mut())?)
    }

    /// Load metadata from a model package without fully loading the model.
    ///
    /// This is a static method that doesn't require an [`AOTIModel`] instance.
    /// It streams just the metadata JSON entry from the zip, so it's cheap
    /// even on multi-GB packages.
    pub fn load_metadata_from_package(
        model_package_path: &str,
        model_name: &str,
    ) -> Result<HashMap<String, String>, Error> {
        read_metadata_from_zip(model_package_path, model_name)
    }
}
