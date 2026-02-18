#include "aoti.h"
#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <string>
#include <vector>

namespace aoti_rs {

std::unique_ptr<torch::inductor::AOTIModelPackageLoader> loader_new(
    rust::Str model_package_path,
    rust::Str model_name,
    bool run_single_threaded,
    size_t num_runners,
    int8_t device_index) {
    return std::make_unique<torch::inductor::AOTIModelPackageLoader>(
        std::string(model_package_path),
        std::string(model_name),
        run_single_threaded,
        num_runners,
        static_cast<c10::DeviceIndex>(device_index));
}

rust::Vec<OwnedTensor> loader_run(
    torch::inductor::AOTIModelPackageLoader& loader,
    const rust::Vec<TensorPtr>& inputs) {
    std::vector<at::Tensor> cpp_inputs;
    cpp_inputs.reserve(inputs.size());
    for (const auto& t : inputs) {
        // The pointer is a const at::Tensor* from tch-rs (via torch-sys).
        // tch::Tensor is a repr(C) wrapper around *mut C_tensor, which is at::Tensor.
        const at::Tensor* tensor_ptr = reinterpret_cast<const at::Tensor*>(static_cast<uintptr_t>(t.ptr));
        cpp_inputs.push_back(*tensor_ptr);
    }

    std::vector<at::Tensor> outputs = loader.run(cpp_inputs);

    rust::Vec<OwnedTensor> result;
    result.reserve(outputs.size());
    for (auto& out : outputs) {
        // Heap-allocate each output tensor so Rust can own it via a raw pointer.
        at::Tensor* heap_tensor = new at::Tensor(std::move(out));
        OwnedTensor ot;
        ot.ptr = reinterpret_cast<size_t>(heap_tensor);
        result.push_back(ot);
    }
    return result;
}

rust::Vec<OwnedTensor> loader_boxed_run(
    torch::inductor::AOTIModelPackageLoader& loader,
    rust::Vec<TensorPtr>& inputs) {
    std::vector<at::Tensor> cpp_inputs;
    cpp_inputs.reserve(inputs.size());
    for (const auto& t : inputs) {
        const at::Tensor* tensor_ptr = reinterpret_cast<const at::Tensor*>(static_cast<uintptr_t>(t.ptr));
        cpp_inputs.push_back(*tensor_ptr);
    }

    std::vector<at::Tensor> outputs = loader.boxed_run(std::move(cpp_inputs));

    rust::Vec<OwnedTensor> result;
    result.reserve(outputs.size());
    for (auto& out : outputs) {
        at::Tensor* heap_tensor = new at::Tensor(std::move(out));
        OwnedTensor ot;
        ot.ptr = reinterpret_cast<size_t>(heap_tensor);
        result.push_back(ot);
    }
    return result;
}

rust::Vec<MetadataEntry> loader_get_metadata(
    torch::inductor::AOTIModelPackageLoader& loader) {
    auto metadata = loader.get_metadata();
    rust::Vec<MetadataEntry> result;
    result.reserve(metadata.size());
    for (const auto& [k, v] : metadata) {
        MetadataEntry entry;
        entry.key = rust::String(k);
        entry.value = rust::String(v);
        result.push_back(std::move(entry));
    }
    return result;
}

rust::Vec<rust::String> loader_get_call_spec(
    torch::inductor::AOTIModelPackageLoader& loader) {
    auto specs = loader.get_call_spec();
    rust::Vec<rust::String> result;
    result.reserve(specs.size());
    for (const auto& s : specs) {
        result.push_back(rust::String(s));
    }
    return result;
}

rust::Vec<rust::String> loader_get_constant_fqns(
    torch::inductor::AOTIModelPackageLoader& loader) {
    auto fqns = loader.get_constant_fqns();
    rust::Vec<rust::String> result;
    result.reserve(fqns.size());
    for (const auto& s : fqns) {
        result.push_back(rust::String(s));
    }
    return result;
}

rust::Vec<MetadataEntry> loader_load_metadata_from_package(
    rust::Str model_package_path,
    rust::Str model_name) {
    auto metadata = torch::inductor::AOTIModelPackageLoader::load_metadata_from_package(
        std::string(model_package_path),
        std::string(model_name));
    rust::Vec<MetadataEntry> result;
    result.reserve(metadata.size());
    for (const auto& [k, v] : metadata) {
        MetadataEntry entry;
        entry.key = rust::String(k);
        entry.value = rust::String(v);
        result.push_back(std::move(entry));
    }
    return result;
}

} // namespace aoti_rs
