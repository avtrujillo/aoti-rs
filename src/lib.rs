//! Safe Rust bindings to PyTorch's AOT Inductor (AOTI) model runtime.
//!
//! # Device safety
//!
//! Models and tensors are tagged at the type level with the device kind they
//! live on ([`Cpu`] or [`Cuda`]). An `AOTIModel<Cuda>` only accepts
//! [`DeviceTensor<Cuda>`] inputs (and produces `DeviceTensor<Cuda>` outputs),
//! so handing a RAM-resident tensor to a model that expects VRAM pointers is
//! a *compile-time* error:
//!
//! ```compile_fail
//! use aoti_rs::{AOTIModel, Cpu, Cuda, DeviceTensor};
//!
//! fn run_on_gpu(model: &mut AOTIModel<Cuda>, input: DeviceTensor<Cpu>) {
//!     // ERROR: expected `DeviceTensor<Cuda>`, found `DeviceTensor<Cpu>`
//!     model.run(&[input]).unwrap();
//! }
//! ```
//!
//! Runtime checks only happen at the boundaries where untyped values enter:
//!
//! - [`DeviceTensor::try_new`] verifies the actual placement of a
//!   `tch::Tensor` once; from then on the device is carried by the type.
//! - [`AOTIModelBuilder::build`] verifies the package's `AOTI_DEVICE_KEY`
//!   metadata against the requested device type and fails with
//!   [`Error::ModelDeviceMismatch`] if they disagree.
//!
//! When the crate is built against a libtorch without CUDA support (or with
//! `AOTI_RS_NO_CUDA=1`), the CUDA model-loading APIs are compiled out
//! entirely (`cfg(aoti_cuda)` is unset), so loading a CUDA model in a
//! CPU-only build is also a compile-time error instead of a C++ exception.
//!
//! If the target device of a package is only known at runtime, use
//! [`AnyAOTIModel`], which inspects the package metadata and dispatches to
//! the right typed model.

use std::collections::HashMap;
use std::marker::PhantomData;
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

        fn runner_get_call_spec(runner: Pin<&mut AOTIModelContainerRunner>) -> Result<Vec<String>>;

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

    #[error("model package targets device '{found}' but was loaded as a {expected} model")]
    ModelDeviceMismatch {
        expected: &'static str,
        found: String,
    },

    #[error("tensor resides on {found:?} but a {expected} tensor was required")]
    TensorDeviceMismatch {
        expected: &'static str,
        found: tch::Device,
    },
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::Cpu {}
    impl Sealed for super::Cuda {}
}

/// Type-level marker for the device kind a model or tensor lives on.
///
/// This trait is sealed: the only implementors are [`Cpu`] and [`Cuda`],
/// matching the two runner kinds libtorch's AOTI runtime provides.
pub trait Device: sealed::Sealed + 'static {
    /// Device key as it appears in `.pt2` metadata (`AOTI_DEVICE_KEY`).
    const KEY: &'static str;
    /// Whether this device kind is CUDA.
    const IS_CUDA: bool;

    /// Whether a runtime `tch::Device` belongs to this device kind.
    fn matches(device: tch::Device) -> bool;

    /// Extract the typed model from an [`AnyAOTIModel`] if its variant is
    /// `Self`. Implementation detail of [`AnyAOTIModel::try_into_typed`];
    /// call that instead.
    #[doc(hidden)]
    fn model_from_any(any: AnyAOTIModel) -> Result<AOTIModel<Self>, Error>
    where
        Self: Sized;
}

/// Marker type for CPU (host RAM) placement.
pub struct Cpu;

/// Marker type for CUDA (device VRAM) placement.
///
/// The device *kind* is tracked at compile time; the CUDA device *index* is
/// still a runtime property (see [`AOTIModelBuilder::device_index`]) and is
/// validated by libtorch.
pub struct Cuda;

impl Device for Cpu {
    const KEY: &'static str = "cpu";
    const IS_CUDA: bool = false;

    fn matches(device: tch::Device) -> bool {
        device == tch::Device::Cpu
    }

    fn model_from_any(any: AnyAOTIModel) -> Result<AOTIModel<Cpu>, Error> {
        match any {
            AnyAOTIModel::Cpu(model) => Ok(model),
            #[cfg(aoti_cuda)]
            AnyAOTIModel::Cuda(_) => Err(Error::ModelDeviceMismatch {
                expected: Cpu::KEY,
                found: Cuda::KEY.to_string(),
            }),
        }
    }
}

impl Device for Cuda {
    const KEY: &'static str = "cuda";
    const IS_CUDA: bool = true;

    fn matches(device: tch::Device) -> bool {
        matches!(device, tch::Device::Cuda(_))
    }

    fn model_from_any(any: AnyAOTIModel) -> Result<AOTIModel<Cuda>, Error> {
        match any {
            #[cfg(aoti_cuda)]
            AnyAOTIModel::Cuda(model) => Ok(model),
            AnyAOTIModel::Cpu(_) => Err(Error::ModelDeviceMismatch {
                expected: Cuda::KEY,
                found: Cpu::KEY.to_string(),
            }),
        }
    }
}

/// The device kind (`"cpu"`, `"cuda"`) of a metadata device key such as
/// `"cuda"` or `"cuda:1"`.
fn device_kind(key: &str) -> &str {
    key.split(':').next().unwrap_or(key)
}

/// A `tch::Tensor` whose placement on device kind `D` has been verified.
///
/// This is the only input type accepted by [`AOTIModel::run`] and
/// [`AOTIModel::boxed_run`], which makes the device of every tensor crossing
/// the FFI boundary part of its compile-time type. The placement is checked
/// once, at construction; the wrapper then dereferences to `&tch::Tensor`
/// for read access. No `&mut Tensor` access is exposed, so the placement
/// cannot be invalidated after construction.
pub struct DeviceTensor<D: Device> {
    tensor: Tensor,
    _device: PhantomData<D>,
}

impl<D: Device> DeviceTensor<D> {
    /// Wrap `tensor`, verifying that it actually resides on device kind `D`.
    ///
    /// Returns [`Error::TensorDeviceMismatch`] otherwise.
    pub fn try_new(tensor: Tensor) -> Result<Self, Error> {
        let found = tensor.device();
        if D::matches(found) {
            Ok(Self {
                tensor,
                _device: PhantomData,
            })
        } else {
            Err(Error::TensorDeviceMismatch {
                expected: D::KEY,
                found,
            })
        }
    }

    /// Wrap every tensor in `tensors`, verifying each one's placement.
    pub fn try_new_all(tensors: Vec<Tensor>) -> Result<Vec<Self>, Error> {
        tensors.into_iter().map(Self::try_new).collect()
    }

    /// Wrap `tensor` without verifying its placement.
    ///
    /// # Safety
    /// The caller must guarantee that `tensor` resides on device kind `D`.
    /// Passing a wrongly-placed tensor to a model defeats the type-level
    /// device guarantee and leads to undefined behavior inside the AOTI
    /// runtime (e.g. a RAM pointer dereferenced as a VRAM pointer).
    pub unsafe fn new_unchecked(tensor: Tensor) -> Self {
        debug_assert!(
            D::matches(tensor.device()),
            "DeviceTensor::<{}>::new_unchecked called with a tensor on {:?}",
            D::KEY,
            tensor.device()
        );
        Self {
            tensor,
            _device: PhantomData,
        }
    }

    /// Unwrap into the underlying `tch::Tensor`.
    pub fn into_inner(self) -> Tensor {
        self.tensor
    }

    /// Copy (if necessary) to host RAM, returning a CPU-typed tensor.
    pub fn to_cpu(&self) -> DeviceTensor<Cpu> {
        DeviceTensor {
            tensor: self.tensor.to_device(tch::Device::Cpu),
            _device: PhantomData,
        }
    }

    /// Copy (if necessary) to VRAM on CUDA device `index`, returning a
    /// CUDA-typed tensor.
    #[cfg(aoti_cuda)]
    pub fn to_cuda(&self, index: usize) -> DeviceTensor<Cuda> {
        DeviceTensor {
            tensor: self.tensor.to_device(tch::Device::Cuda(index)),
            _device: PhantomData,
        }
    }
}

impl<D: Device> std::ops::Deref for DeviceTensor<D> {
    type Target = Tensor;

    fn deref(&self) -> &Tensor {
        &self.tensor
    }
}

impl<D: Device> AsRef<Tensor> for DeviceTensor<D> {
    fn as_ref(&self) -> &Tensor {
        &self.tensor
    }
}

impl<D: Device> TryFrom<Tensor> for DeviceTensor<D> {
    type Error = Error;

    fn try_from(tensor: Tensor) -> Result<Self, Error> {
        Self::try_new(tensor)
    }
}

impl<D: Device> From<DeviceTensor<D>> for Tensor {
    fn from(t: DeviceTensor<D>) -> Tensor {
        t.tensor
    }
}

impl<D: Device> std::fmt::Debug for DeviceTensor<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DeviceTensor<{}>({:?})", D::KEY, self.tensor)
    }
}

/// Convert a slice of device-verified tensors into `TensorPtr` values for the
/// FFI boundary.
fn tensors_to_ptrs<D: Device>(tensors: &[DeviceTensor<D>]) -> Vec<ffi::TensorPtr> {
    tensors
        .iter()
        .map(|t| ffi::TensorPtr {
            ptr: t.tensor.as_ptr() as *const ffi::c_void,
        })
        .collect()
}

/// Convert `OwnedTensor` values from C++ into device-typed `tch::Tensor`s.
///
/// # Safety
/// Each `OwnedTensor::ptr` must be a valid, heap-allocated `at::Tensor*`
/// created with `new at::Tensor(...)` on the C++ side. Ownership is
/// transferred to the returned tensor which will free the underlying storage
/// on drop. The tensors must reside on device kind `D` — for model outputs
/// this holds because AOTI places outputs on the model's own device, which
/// `build()` verified to be `D`.
fn owned_to_tensors<D: Device>(owned: Vec<ffi::OwnedTensor>) -> Vec<DeviceTensor<D>> {
    owned
        .into_iter()
        .map(|ot| {
            let tensor = unsafe { Tensor::from_ptr(ot.ptr as *mut _) };
            debug_assert!(
                D::matches(tensor.device()),
                "AOTI returned an output on {:?} from a {} model",
                tensor.device(),
                D::KEY
            );
            DeviceTensor {
                tensor,
                _device: PhantomData,
            }
        })
        .collect()
}

/// Extract every non-directory entry of a `.pt2` archive into a fresh temp dir.
/// See [`extract_package`].
fn extract_pt2(pt2_path: &str) -> Result<TempDir, Error> {
    let dir = tempfile::tempdir()?;
    extract_package(pt2_path, dir.path())?;
    Ok(dir)
}

/// Extract every non-directory entry of a `.pt2` archive into `dest`, using a
/// Zip64-aware reader.  This bypasses libtorch's miniz-based extractor, which
/// fails on archives whose internal `wrapper.so` pushes the central directory
/// past the 32-bit Zip offset boundary.
///
/// The result is the layout [`AOTIModelBuilder::from_extracted`] loads in
/// place, so a package can be extracted once ahead of time instead of on every
/// [`AOTIModelBuilder::new`] build. Any Zip64-aware extractor produces the same
/// layout (Python's `zipfile.ZipFile.extractall` included); this one exists so
/// Rust callers need nothing else.
///
/// Entries whose names would escape `dest` are rejected rather than written.
pub fn extract_package(pt2_path: impl AsRef<Path>, dest: impl AsRef<Path>) -> Result<(), Error> {
    let dest = dest.as_ref();
    let file = std::fs::File::open(pt2_path)?;
    let mut archive = zip::ZipArchive::new(file)?;
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let name = entry.name().to_string();
        // Skip directory marker entries (the source of the original miniz bug).
        if name.ends_with('/') {
            continue;
        }
        let Some(relative) = entry.enclosed_name() else {
            return Err(Error::InvalidPath(format!(
                "package entry '{name}' would extract outside the destination"
            )));
        };
        let outpath = dest.join(relative);
        if let Some(parent) = outpath.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut out = std::fs::File::create(&outpath)?;
        std::io::copy(&mut entry, &mut out)?;
    }
    Ok(())
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
    let search_root = if aotinductor.is_dir() {
        aotinductor
    } else {
        dir.to_path_buf()
    };

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
fn read_metadata_from_dir(dir: &Path, model_name: &str) -> Result<HashMap<String, String>, Error> {
    let model_root = dir.join(model_name);
    let search_root = if model_root.is_dir() {
        model_root
    } else {
        dir.to_path_buf()
    };
    let hits = find_files(&search_root, |n| {
        n.ends_with("_metadata.json") || n == "metadata.json"
    })?;
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

/// Load metadata from a model package without fully loading the model.
///
/// Streams just the metadata JSON entry from the zip, so it's cheap even on
/// multi-GB packages.
pub fn load_metadata_from_package(
    model_package_path: &str,
    model_name: &str,
) -> Result<HashMap<String, String>, Error> {
    read_metadata_from_zip(model_package_path, model_name)
}

/// Load metadata from an already-extracted package directory (see
/// [`extract_package`]) without loading the model.
pub fn load_metadata_from_extracted(
    extracted_dir: impl AsRef<Path>,
    model_name: &str,
) -> Result<HashMap<String, String>, Error> {
    read_metadata_from_dir(extracted_dir.as_ref(), model_name)
}

/// Where an [`AOTIModelBuilder`] gets the model's files from.
enum ModelSource {
    /// A `.pt2` archive, extracted into a fresh temp dir on every build.
    Package(String),
    /// A directory holding an already-extracted package, loaded in place.
    Extracted(PathBuf),
}

impl std::fmt::Display for ModelSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Package(path) => f.write_str(path),
            Self::Extracted(dir) => write!(f, "{}", dir.display()),
        }
    }
}

/// Builder for configuring and creating an [`AOTIModel`].
///
/// The device kind is part of the builder's type: `device_index` (a
/// CUDA-only concept) is only available on `AOTIModelBuilder<Cuda>`, and
/// `build()` for the CUDA variant only exists when the crate was built with
/// CUDA support (`cfg(aoti_cuda)`).
pub struct AOTIModelBuilder<D: Device> {
    source: ModelSource,
    model_name: String,
    run_single_threaded: bool,
    num_runners: usize,
    device_index: i8,
    _device: PhantomData<D>,
}

impl<D: Device> AOTIModelBuilder<D> {
    /// Create a builder for the given `.pt2` model package path.
    ///
    /// The package is extracted into a fresh temp dir when the model is
    /// built, and the copy lives as long as the model does.
    pub fn new(model_package_path: impl Into<String>) -> Self {
        Self::with_source(ModelSource::Package(model_package_path.into()))
    }

    /// Create a builder for a package that has already been extracted into
    /// `extracted_dir` (by [`extract_package`] or any Zip64-aware extractor).
    ///
    /// The model is loaded in place: nothing is copied, and the runner maps
    /// `wrapper.so` and reads `.cubin` kernels straight from `extracted_dir`.
    /// For a multi-GB package that saves rewriting the whole package on every
    /// load, and the page cache holds one clean copy of it rather than two.
    ///
    /// The directory must therefore stay in place, unmodified, for the
    /// model's lifetime — kernels are read lazily during inference. It may be
    /// read-only, but its filesystem must allow executable mappings (no
    /// `noexec`), since `wrapper.so` is loaded as a shared library.
    pub fn from_extracted(extracted_dir: impl Into<PathBuf>) -> Self {
        Self::with_source(ModelSource::Extracted(extracted_dir.into()))
    }

    fn with_source(source: ModelSource) -> Self {
        Self {
            source,
            model_name: "model".to_string(),
            run_single_threaded: false,
            num_runners: 1,
            device_index: -1,
            _device: PhantomData,
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

    /// Extract the package (unless it already is), validate its device
    /// metadata against `D`, and construct the runner.
    fn build_inner(self) -> Result<AOTIModel<D>, Error> {
        let (root, temp_dir) = match self.source {
            ModelSource::Package(path) => {
                let temp_dir = extract_pt2(&path)?;
                (temp_dir.path().to_path_buf(), Some(temp_dir))
            }
            ModelSource::Extracted(dir) => {
                // Checked up front: otherwise a typo'd path surfaces as the
                // much less obvious "no wrapper.so found".
                if !dir.is_dir() {
                    return Err(Error::InvalidPath(format!(
                        "extracted package directory {} does not exist",
                        dir.display()
                    )));
                }
                (dir, None)
            }
        };
        let so_path = find_wrapper_so(&root, &self.model_name)?;
        let metadata = read_metadata_from_dir(&root, &self.model_name)?;

        // The type parameter decides which runner we construct; the package
        // metadata, when present, must agree with it.
        if let Some(found) = metadata.get("AOTI_DEVICE_KEY")
            && device_kind(found) != D::KEY
        {
            return Err(Error::ModelDeviceMismatch {
                expected: D::KEY,
                found: found.clone(),
            });
        }

        let cubin_dir = so_path
            .parent()
            .ok_or_else(|| Error::InvalidPath(format!("{} has no parent", so_path.display())))?
            .to_string_lossy()
            .into_owned();
        let so_path_str = so_path.to_str().ok_or_else(|| {
            Error::InvalidPath(format!("{} is not valid UTF-8", so_path.display()))
        })?;

        let inner = ffi::runner_new(
            so_path_str,
            &cubin_dir,
            D::IS_CUDA,
            self.device_index,
            self.num_runners,
            self.run_single_threaded,
        )?;

        Ok(AOTIModel {
            inner,
            metadata,
            _temp_dir: temp_dir,
            _device: PhantomData,
        })
    }
}

impl AOTIModelBuilder<Cpu> {
    /// Build the model, extracting the package and constructing the CPU runner.
    pub fn build(self) -> Result<AOTIModel<Cpu>, Error> {
        self.build_inner()
    }
}

#[cfg(aoti_cuda)]
impl AOTIModelBuilder<Cuda> {
    /// Set the CUDA device index (default: -1 for the current default device).
    pub fn device_index(mut self, idx: i8) -> Self {
        self.device_index = idx;
        self
    }

    /// Build the model, extracting the package and constructing the CUDA runner.
    pub fn build(self) -> Result<AOTIModel<Cuda>, Error> {
        self.build_inner()
    }
}

/// A loaded AOT-compiled PyTorch model, ready for inference on device kind `D`.
///
/// Wraps `torch::inductor::AOTIModelContainerRunner` from libtorch.  The `.pt2`
/// archive is extracted in Rust and the resulting `wrapper.so` is handed to the
/// runner directly, bypassing `AOTIModelPackageLoader`'s miniz-based extractor.
///
/// # Example
///
/// ```no_run
/// use aoti_rs::{AOTIModel, Cpu, DeviceTensor};
/// use tch::Tensor;
///
/// let mut model = AOTIModel::<Cpu>::load("model.pt2").unwrap();
/// let input = Tensor::randn([1, 3, 224, 224], (tch::Kind::Float, tch::Device::Cpu));
/// let input = DeviceTensor::<Cpu>::try_new(input).unwrap();
/// let outputs = model.run(&[input]).unwrap();
/// ```
pub struct AOTIModel<D: Device> {
    inner: cxx::UniquePtr<ffi::AOTIModelContainerRunner>,
    metadata: HashMap<String, String>,
    // The runner mmaps `wrapper.so` and reads `.cubin` kernel files lazily
    // during inference, so the extracted directory must outlive `inner`.
    // `None` when loaded in place via `from_extracted`, where keeping the
    // directory alive is the caller's job.
    _temp_dir: Option<TempDir>,
    _device: PhantomData<D>,
}

// Safety: AOTIModelContainerRunner manages its own thread safety via num_runners.
// When num_runners > 1, concurrent run() calls are safe. The user controls this
// via the builder. Single-runner use should be externally synchronized.
unsafe impl<D: Device> Send for AOTIModel<D> {}

impl AOTIModel<Cpu> {
    /// Load a `.pt2` model package targeting the CPU with default settings.
    pub fn load(model_package_path: impl Into<String>) -> Result<Self, Error> {
        AOTIModelBuilder::<Cpu>::new(model_package_path).build()
    }

    /// Load an already-extracted package in place, targeting the CPU with
    /// default settings. See [`AOTIModelBuilder::from_extracted`].
    pub fn load_extracted(extracted_dir: impl Into<PathBuf>) -> Result<Self, Error> {
        AOTIModelBuilder::<Cpu>::from_extracted(extracted_dir).build()
    }
}

#[cfg(aoti_cuda)]
impl AOTIModel<Cuda> {
    /// Load a `.pt2` model package targeting CUDA with default settings.
    pub fn load(model_package_path: impl Into<String>) -> Result<Self, Error> {
        AOTIModelBuilder::<Cuda>::new(model_package_path).build()
    }

    /// Load an already-extracted package in place, targeting CUDA with
    /// default settings. See [`AOTIModelBuilder::from_extracted`].
    pub fn load_extracted(extracted_dir: impl Into<PathBuf>) -> Result<Self, Error> {
        AOTIModelBuilder::<Cuda>::from_extracted(extracted_dir).build()
    }
}

impl<D: Device> AOTIModel<D> {
    /// Create a builder for more control over loading options.
    pub fn builder(model_package_path: impl Into<String>) -> AOTIModelBuilder<D> {
        AOTIModelBuilder::new(model_package_path)
    }

    /// Create a builder for an already-extracted package, loaded in place.
    /// See [`AOTIModelBuilder::from_extracted`].
    pub fn builder_extracted(extracted_dir: impl Into<PathBuf>) -> AOTIModelBuilder<D> {
        AOTIModelBuilder::from_extracted(extracted_dir)
    }

    /// Run inference on the given input tensors.
    ///
    /// Device placement is enforced at compile time by [`DeviceTensor<D>`];
    /// shapes and dtypes must match the model export and are checked at
    /// runtime by the AOTI runtime. Outputs are returned on the model's
    /// device, carrying the same type-level tag.
    pub fn run(&mut self, inputs: &[DeviceTensor<D>]) -> Result<Vec<DeviceTensor<D>>, Error> {
        let ptrs = tensors_to_ptrs(inputs);
        let owned = ffi::runner_run(self.inner.pin_mut(), &ptrs)?;
        Ok(owned_to_tensors(owned))
    }

    /// Run inference, transferring ownership of the input tensors to the
    /// runtime so it can reuse their storage for in-place optimization.
    ///
    /// Taking the inputs by value is what makes the optimization possible:
    /// the runtime may only steal a tensor's buffer when nothing else
    /// references it, which the type system guarantees here.
    pub fn boxed_run(
        &mut self,
        inputs: Vec<DeviceTensor<D>>,
    ) -> Result<Vec<DeviceTensor<D>>, Error> {
        let mut ptrs = tensors_to_ptrs(&inputs);
        let owned = ffi::runner_boxed_run(self.inner.pin_mut(), &mut ptrs)?;
        // The C++ side moved out of the input tensors; `inputs` now holds
        // empty shells that must stay alive until the call returns.
        drop(inputs);
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
}

/// A model whose device kind is determined at runtime from package metadata.
///
/// Use this when the target device of a `.pt2` package isn't known until
/// runtime. Matching on the variant recovers a fully device-typed
/// [`AOTIModel`], so all subsequent tensor traffic is still checked at
/// compile time.
#[non_exhaustive]
pub enum AnyAOTIModel {
    Cpu(AOTIModel<Cpu>),
    #[cfg(aoti_cuda)]
    Cuda(AOTIModel<Cuda>),
}

impl AnyAOTIModel {
    /// Load a `.pt2` package, dispatching on its `AOTI_DEVICE_KEY` metadata
    /// (missing metadata is treated as CPU). Uses the default model name
    /// `"model"`.
    pub fn load(model_package_path: &str) -> Result<Self, Error> {
        Self::load_named(model_package_path, "model")
    }

    /// Load a `.pt2` package by model name, dispatching on its
    /// `AOTI_DEVICE_KEY` metadata (missing metadata is treated as CPU).
    pub fn load_named(model_package_path: &str, model_name: &str) -> Result<Self, Error> {
        let metadata = read_metadata_from_zip(model_package_path, model_name)?;
        Self::dispatch(
            ModelSource::Package(model_package_path.to_string()),
            &metadata,
            model_name,
        )
    }

    /// Load an already-extracted package in place, dispatching on its
    /// `AOTI_DEVICE_KEY` metadata (missing metadata is treated as CPU). Uses
    /// the default model name `"model"`. See
    /// [`AOTIModelBuilder::from_extracted`] for what in-place loading requires
    /// of the directory.
    pub fn load_extracted(extracted_dir: impl Into<PathBuf>) -> Result<Self, Error> {
        Self::load_extracted_named(extracted_dir, "model")
    }

    /// Load an already-extracted package in place by model name, dispatching
    /// on its `AOTI_DEVICE_KEY` metadata (missing metadata is treated as CPU).
    pub fn load_extracted_named(
        extracted_dir: impl Into<PathBuf>,
        model_name: &str,
    ) -> Result<Self, Error> {
        let dir = extracted_dir.into();
        let metadata = read_metadata_from_dir(&dir, model_name)?;
        Self::dispatch(ModelSource::Extracted(dir), &metadata, model_name)
    }

    /// Build the typed model `metadata` says `source` targets.
    fn dispatch(
        source: ModelSource,
        metadata: &HashMap<String, String>,
        model_name: &str,
    ) -> Result<Self, Error> {
        let is_cuda = metadata
            .get("AOTI_DEVICE_KEY")
            .map(|v| device_kind(v) == Cuda::KEY)
            .unwrap_or(false);

        if is_cuda {
            #[cfg(aoti_cuda)]
            {
                Ok(Self::Cuda(
                    AOTIModelBuilder::<Cuda>::with_source(source)
                        .model_name(model_name)
                        .build()?,
                ))
            }
            #[cfg(not(aoti_cuda))]
            {
                Err(Error::Model(format!(
                    "'{source}' targets CUDA but aoti-rs was built without CUDA support"
                )))
            }
        } else {
            Ok(Self::Cpu(
                AOTIModelBuilder::<Cpu>::with_source(source)
                    .model_name(model_name)
                    .build()?,
            ))
        }
    }

    /// Extract the typed model, requiring it to be on device kind `D`.
    ///
    /// Returns [`Error::ModelDeviceMismatch`] if the loaded model is on a
    /// different device. Unlike matching on the enum directly, this works in
    /// code that is generic over [`Device`] (a `match` cannot narrow a type
    /// parameter to a concrete type) and is portable across CPU-only and
    /// CUDA-enabled builds (the `Cuda` variant only exists in the latter).
    ///
    /// ```no_run
    /// use aoti_rs::{AnyAOTIModel, AOTIModel, Device};
    ///
    /// fn model_from_any<D: Device>(any: AnyAOTIModel) -> Result<AOTIModel<D>, aoti_rs::Error> {
    ///     any.try_into_typed::<D>()
    /// }
    /// ```
    pub fn try_into_typed<D: Device>(self) -> Result<AOTIModel<D>, Error> {
        D::model_from_any(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_tensor_wraps_as_cpu() {
        let t = Tensor::zeros([2, 2], (tch::Kind::Float, tch::Device::Cpu));
        let dt = DeviceTensor::<Cpu>::try_new(t).expect("CPU tensor should wrap as Cpu");
        assert_eq!(dt.size(), &[2, 2]);
        assert_eq!(dt.into_inner().device(), tch::Device::Cpu);
    }

    #[test]
    fn cpu_tensor_rejected_as_cuda() {
        let t = Tensor::zeros([2, 2], (tch::Kind::Float, tch::Device::Cpu));
        match DeviceTensor::<Cuda>::try_new(t) {
            Err(Error::TensorDeviceMismatch { expected, found }) => {
                assert_eq!(expected, "cuda");
                assert_eq!(found, tch::Device::Cpu);
            }
            Ok(_) => panic!("CPU tensor must not wrap as Cuda"),
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    fn device_kind_strips_index() {
        assert_eq!(device_kind("cuda"), "cuda");
        assert_eq!(device_kind("cuda:1"), "cuda");
        assert_eq!(device_kind("cpu"), "cpu");
    }
}
