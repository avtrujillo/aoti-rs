# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`aoti-rs` is a Rust crate providing safe bindings to PyTorch's AOT Inductor (AOTI) model runtime. It allows loading and running `.pt2` model packages (exported via `torch.export` + `torch.compile`) from Rust.

## Build Commands

```bash
# Build (requires libtorch/PyTorch available)
cargo build

# Run tests
cargo test

# Run a single test
cargo test <test_name>

# Lint
cargo clippy

# Format
cargo fmt
```

## libtorch Discovery (Build Requirement)

The build script ([build.rs](build.rs)) locates libtorch in this priority order:
1. `LIBTORCH` env var (path to libtorch root)
2. `DEP_TCH_LIBTORCH_LIB` (exported by `torch-sys` automatically)
3. Python: `python3 -c "import torch; print(torch.__file__)"` — use `LIBTORCH_USE_PYTORCH=1` with `torch-sys` to activate this path

Override individual paths with `LIBTORCH_INCLUDE` and `LIBTORCH_LIB`.

## Architecture

The crate bridges Rust ↔ C++ using the `cxx` crate:

```
src/lib.rs          — Rust public API + cxx::bridge FFI declarations
csrc/aoti.h         — C++ function signatures for cxx bridge
csrc/aoti.cc        — C++ implementation wrapping torch::inductor::AOTIModelPackageLoader
csrc/cvoid.h        — Trivial header: `using c_void = void` (needed by cxx for opaque void*)
build.rs            — Locates libtorch, compiles csrc/aoti.cc, links torch/torch_cpu/c10
```

### Compile-time device safety

Device placement (RAM vs VRAM) is tracked in the type system via a sealed
`Device` trait with `Cpu`/`Cuda` marker types:

- `DeviceTensor<D>` wraps a `tch::Tensor` whose placement was verified once at
  construction (`try_new`); after that the device is carried by the type. It
  derefs to `&Tensor` (read-only, so the invariant can't be broken).
- `AOTIModel<D>` / `AOTIModelBuilder<D>` are parameterized by device. `build()`
  cross-checks the package's `AOTI_DEVICE_KEY` metadata against `D`
  (`Error::ModelDeviceMismatch`); `run`/`boxed_run` accept and return only
  `DeviceTensor<D>`, so passing a RAM tensor to a CUDA model is a compile error.
- build.rs emits `cfg(aoti_cuda)` when libtorch ships `libtorch_cuda.so` (unless
  `AOTI_RS_NO_CUDA=1`). CUDA-only APIs (`AOTIModelBuilder::<Cuda>::build`,
  `device_index`, `AOTIModel::<Cuda>::load`, `DeviceTensor::to_cuda`,
  `AnyAOTIModel::Cuda`) only exist under that cfg, so loading a CUDA model in a
  CPU-only build is also a compile error. The cfg applies to this package's
  tests too.
- `AnyAOTIModel` handles packages whose device is only known at runtime by
  reading the metadata and dispatching to a typed `AOTIModel<D>` variant.

### Data flow at the FFI boundary

- **Input tensors**: `DeviceTensor<D>` → raw `*const at::Tensor` pointer wrapped in `TensorPtr` struct
- **Output tensors**: C++ heap-allocates each `at::Tensor*`, passes pointer back as `OwnedTensor`; Rust takes ownership via `Tensor::from_ptr` and re-tags it as `DeviceTensor<D>`
- **`boxed_run`**: Rust passes inputs by value; C++ moves out of each `at::Tensor` (keeping use count at 1 so the runtime can reuse buffers in place) and Rust drops the empty shells after the call
- **Metadata**: parsed in Rust from the package's `*_metadata.json`

### Public Rust API

- `AOTIModel::<Cpu>::load(path)` / `AOTIModel::<Cuda>::load(path)` — quick load with defaults
- `AOTIModel::<D>::builder(path)` — returns `AOTIModelBuilder<D>` for configuring `model_name`, `num_runners`, `single_threaded`, and (CUDA only) `device_index`
- `AOTIModel::run(&[DeviceTensor<D>])` — runs inference, returns `Vec<DeviceTensor<D>>`
- `AOTIModel::boxed_run(Vec<DeviceTensor<D>>)` — run giving the runtime ownership of inputs (enables in-place optimization)
- `AOTIModel::get_metadata()`, `get_call_spec()`, `get_constant_fqns()` — introspection
- `AnyAOTIModel::load(path)` / `load_named(path, name)` — runtime device dispatch
- `load_metadata_from_package(path, name)` — free function, reads metadata without fully loading

### Key cxx bridge constraints

- `AOTIModelPackageLoader` is `Pin<&mut ...>` in Rust because cxx requires `Pin` for non-const C++ methods
- `c_void` is declared via `csrc/cvoid.h` because cxx cannot directly use Rust's `std::ffi::c_void`
- The bridge namespace is `aoti_rs`; `AOTIModelPackageLoader` uses `#[namespace = "torch::inductor"]`

## Dependencies

- `cxx` — Rust/C++ interop
- `tch` / `torch-sys` — Rust bindings to libtorch (pinned to `=0.24.0`)
- `thiserror` — error type derivation
- `dlpk` — dynamic library helpers
