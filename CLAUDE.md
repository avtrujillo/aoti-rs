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

### Data flow at the FFI boundary

- **Input tensors**: `tch::Tensor` → raw `*const at::Tensor` pointer wrapped in `TensorPtr` struct
- **Output tensors**: C++ heap-allocates each `at::Tensor*`, passes pointer back as `OwnedTensor`; Rust takes ownership via `Tensor::from_ptr`
- **Metadata**: returned as `Vec<MetadataEntry>` (key/value pairs), converted to `HashMap<String, String>` on the Rust side

### Public Rust API

- `AOTIModel::load(path)` — quick load with defaults
- `AOTIModel::builder(path)` — returns `AOTIModelBuilder` for configuring `model_name`, `num_runners`, `single_threaded`, `device_index`
- `AOTIModel::run(&[Tensor])` — runs inference
- `AOTIModel::boxed_run(&[Tensor])` — run allowing runtime to take ownership of inputs (potential in-place optimization)
- `AOTIModel::get_metadata()`, `get_call_spec()`, `get_constant_fqns()` — introspection
- `AOTIModel::load_metadata_from_package(path, name)` — static method, reads metadata without fully loading

### Key cxx bridge constraints

- `AOTIModelPackageLoader` is `Pin<&mut ...>` in Rust because cxx requires `Pin` for non-const C++ methods
- `c_void` is declared via `csrc/cvoid.h` because cxx cannot directly use Rust's `std::ffi::c_void`
- The bridge namespace is `aoti_rs`; `AOTIModelPackageLoader` uses `#[namespace = "torch::inductor"]`

## Dependencies

- `cxx` — Rust/C++ interop
- `tch` / `torch-sys` — Rust bindings to libtorch (pinned to `=0.22.0`)
- `thiserror` — error type derivation
- `dlpk` — dynamic library helpers
