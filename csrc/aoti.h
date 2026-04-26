#pragma once

#include "rust/cxx.h"
#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>
#include <memory>
#include <cstdint>

// Shared structs (TensorPtr, OwnedTensor) are defined by the cxx code
// generator from the Rust bridge declaration.  We only need forward
// declarations here so the function signatures below compile.
namespace aoti_rs {

struct TensorPtr;
struct OwnedTensor;

// Construct an AOTIModelContainerRunner{Cpu,Cuda} from a pre-extracted
// wrapper.so.  The .pt2 archive is extracted in Rust with a Zip64-aware
// reader, bypassing the miniz-based extractor in AOTIModelPackageLoader
// which fails on archives whose internal wrapper.so pushes the central
// directory past the 32-bit Zip offset boundary.
std::unique_ptr<torch::inductor::AOTIModelContainerRunner> runner_new(
    rust::Str model_so_path,
    rust::Str cubin_dir,
    bool is_cuda,
    int8_t device_index,
    size_t num_runners,
    bool run_single_threaded);

rust::Vec<OwnedTensor> runner_run(
    torch::inductor::AOTIModelContainerRunner& runner,
    const rust::Vec<TensorPtr>& inputs);

rust::Vec<OwnedTensor> runner_boxed_run(
    torch::inductor::AOTIModelContainerRunner& runner,
    rust::Vec<TensorPtr>& inputs);

rust::Vec<rust::String> runner_get_call_spec(
    torch::inductor::AOTIModelContainerRunner& runner);

rust::Vec<rust::String> runner_get_constant_fqns(
    torch::inductor::AOTIModelContainerRunner& runner);

} // namespace aoti_rs
