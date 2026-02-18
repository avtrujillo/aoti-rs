#pragma once

#include "rust/cxx.h"
#include <memory>
#include <cstdint>

namespace torch::inductor {
class AOTIModelPackageLoader;
}

namespace aoti_rs {

struct TensorPtr {
    size_t ptr;
};

struct OwnedTensor {
    size_t ptr;
};

struct MetadataEntry {
    rust::String key;
    rust::String value;
};

std::unique_ptr<torch::inductor::AOTIModelPackageLoader> loader_new(
    rust::Str model_package_path,
    rust::Str model_name,
    bool run_single_threaded,
    size_t num_runners,
    int8_t device_index);

rust::Vec<OwnedTensor> loader_run(
    torch::inductor::AOTIModelPackageLoader& loader,
    const rust::Vec<TensorPtr>& inputs);

rust::Vec<OwnedTensor> loader_boxed_run(
    torch::inductor::AOTIModelPackageLoader& loader,
    rust::Vec<TensorPtr>& inputs);

rust::Vec<MetadataEntry> loader_get_metadata(
    torch::inductor::AOTIModelPackageLoader& loader);

rust::Vec<rust::String> loader_get_call_spec(
    torch::inductor::AOTIModelPackageLoader& loader);

rust::Vec<rust::String> loader_get_constant_fqns(
    torch::inductor::AOTIModelPackageLoader& loader);

rust::Vec<MetadataEntry> loader_load_metadata_from_package(
    rust::Str model_package_path,
    rust::Str model_name);

} // namespace aoti_rs
