use std::collections::HashMap;
use std::pin::Pin;

use tch::Tensor;

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

    struct MetadataEntry {
        key: String,
        value: String,
    }

    #[namespace = "torch::inductor"]
    unsafe extern "C++" {
        type AOTIModelPackageLoader;
    }

    unsafe extern "C++" {
        include!("csrc/aoti.h");

        fn loader_new(
            model_package_path: &str,
            model_name: &str,
            run_single_threaded: bool,
            num_runners: usize,
            device_index: i8,
        ) -> Result<UniquePtr<AOTIModelPackageLoader>>;

        fn loader_run(
            loader: Pin<&mut AOTIModelPackageLoader>,
            inputs: &Vec<TensorPtr>,
        ) -> Result<Vec<OwnedTensor>>;

        fn loader_boxed_run(
            loader: Pin<&mut AOTIModelPackageLoader>,
            inputs: &mut Vec<TensorPtr>,
        ) -> Result<Vec<OwnedTensor>>;

        fn loader_get_metadata(
            loader: Pin<&mut AOTIModelPackageLoader>,
        ) -> Result<Vec<MetadataEntry>>;

        fn loader_get_call_spec(loader: Pin<&mut AOTIModelPackageLoader>) -> Result<Vec<String>>;

        fn loader_get_constant_fqns(
            loader: Pin<&mut AOTIModelPackageLoader>,
        ) -> Result<Vec<String>>;

        fn loader_load_metadata_from_package(
            model_package_path: &str,
            model_name: &str,
        ) -> Result<Vec<MetadataEntry>>;
    }
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

fn entries_to_map(entries: Vec<ffi::MetadataEntry>) -> HashMap<String, String> {
    entries.into_iter().map(|e| (e.key, e.value)).collect()
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

    /// Build the model, loading the package and returning an [`AOTIModel`].
    pub fn build(self) -> Result<AOTIModel, cxx::Exception> {
        let inner = ffi::loader_new(
            &self.path,
            &self.model_name,
            self.run_single_threaded,
            self.num_runners,
            self.device_index,
        )?;
        Ok(AOTIModel { inner })
    }
}

/// A loaded AOT-compiled PyTorch model, ready for inference.
///
/// Wraps `torch::inductor::AOTIModelPackageLoader` from libtorch.
///
/// # Example
///
/// ```no_run
/// use aoti_rs::AOTIModel;
/// use tch::Tensor;
///
/// let model = AOTIModel::load("model.pt2").unwrap();
/// let input = Tensor::randn([1, 3, 224, 224], (tch::Kind::Float, tch::Device::Cpu));
/// let outputs = model.run(&[input]).unwrap();
/// ```
pub struct AOTIModel {
    inner: cxx::UniquePtr<ffi::AOTIModelPackageLoader>,
}

// Safety: AOTIModelPackageLoader manages its own thread safety via num_runners.
// When num_runners > 1, concurrent run() calls are safe. The user controls this
// via the builder. Single-runner use should be externally synchronized.
unsafe impl Send for AOTIModel {}

impl AOTIModel {
    /// Load a `.pt2` model package with default settings.
    pub fn load(model_package_path: impl Into<String>) -> Result<Self, cxx::Exception> {
        AOTIModelBuilder::new(model_package_path).build()
    }

    /// Create a builder for more control over loading options.
    pub fn builder(model_package_path: impl Into<String>) -> AOTIModelBuilder {
        AOTIModelBuilder::new(model_package_path)
    }

    /// Obtain a `Pin<&mut AOTIModelPackageLoader>` from the inner `UnsafeCell`.
    ///
    /// # Safety
    /// Caller must ensure no aliasing mutable references exist.
    fn pin_inner(&mut self) -> Option<Pin<&mut ffi::AOTIModelPackageLoader>> {
        self.inner.as_mut()
    }

    fn try_pin_inner(&mut self) -> Result<Pin<&mut ffi::AOTIModelPackageLoader>, AOTIModelError> {
        self.pin_inner().ok_or(
            AOTIModelError::InnerNone
        )
    }

    /// Run inference on the given input tensors.
    ///
    /// Input tensors must match the shapes, dtypes, and device used during
    /// model export.
    pub fn run(&mut self, inputs: &[Tensor]) -> Result<Vec<Tensor>, AOTIModelError> {
        let ptrs = tensors_to_ptrs(inputs);
        let owned = ffi::loader_run(
            self.try_pin_inner()?, &ptrs
        )?;
        Ok(owned_to_tensors(owned))
    }

    /// Run inference, allowing the runtime to take ownership of input tensors
    /// for potential in-place optimization.
    pub fn boxed_run(&mut self, inputs: &[Tensor]) -> Result<Vec<Tensor>, AOTIModelError> {
        let mut ptrs = tensors_to_ptrs(inputs);
        let owned = ffi::loader_boxed_run(
            self.try_pin_inner()?,
            &mut ptrs
        )?;
        Ok(owned_to_tensors(owned))
    }

    /// Get model metadata as a key-value map.
    ///
    /// Typical keys include `"AOTI_DEVICE_KEY"` indicating the target device.
    pub fn get_metadata(&mut self) -> Result<HashMap<String, String>, AOTIModelError> {
        let entries = ffi::loader_get_metadata(
            self.try_pin_inner()?
        )?;
        Ok(entries_to_map(entries))
    }

    /// Get the call specification strings for the model.
    pub fn get_call_spec(&mut self) -> Result<Vec<String>, AOTIModelError> {
        Ok(ffi::loader_get_call_spec(self.try_pin_inner()?)?)
    }

    /// Get the fully qualified names of all constants in the model.
    pub fn get_constant_fqns(&mut self) -> Result<Vec<String>, AOTIModelError> {
        Ok(ffi::loader_get_constant_fqns(self.try_pin_inner()?)?)
    }

    /// Load metadata from a model package without fully loading the model.
    ///
    /// This is a static method that doesn't require an [`AOTIModel`] instance.
    pub fn load_metadata_from_package(
        model_package_path: &str,
        model_name: &str,
    ) -> Result<HashMap<String, String>, cxx::Exception> {
        let entries = ffi::loader_load_metadata_from_package(model_package_path, model_name)?;
        Ok(entries_to_map(entries))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AOTIModelError {
    #[error("AOTIModel's inner field was empty")]
    InnerNone,
    #[error("CXX exception: {0}")]
    Cxx(#[from] cxx::Exception)
}