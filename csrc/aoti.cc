#include "aoti-rs/src/lib.rs.h"
#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>
#ifdef USE_CUDA
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>
#endif
#include <stdexcept>
#include <string>
#include <vector>

namespace aoti_rs {

std::unique_ptr<torch::inductor::AOTIModelContainerRunner> runner_new(
    rust::Str model_so_path,
    rust::Str cubin_dir,
    bool is_cuda,
    int8_t device_index,
    size_t num_runners,
    bool run_single_threaded) {
    std::string so_path(model_so_path);
    std::string cubin(cubin_dir);

    if (is_cuda) {
#ifdef USE_CUDA
        std::string device_str = "cuda";
        if (device_index >= 0) {
            device_str += ":" + std::to_string(static_cast<int>(device_index));
        }
        return std::unique_ptr<torch::inductor::AOTIModelContainerRunner>(
            new torch::inductor::AOTIModelContainerRunnerCuda(
                so_path,
                num_runners,
                device_str,
                cubin,
                run_single_threaded));
#else
        throw std::runtime_error(
            "aoti-rs was built without CUDA support; cannot construct "
            "AOTIModelContainerRunnerCuda");
#endif
    }

    // CPU: AOTIModelContainerRunnerCpu takes (model_so_path, num_models,
    // run_single_threaded) in PyTorch 2.9.  cubin_dir / device_index aren't
    // meaningful on CPU.
    (void)cubin;
    (void)device_index;
    return std::unique_ptr<torch::inductor::AOTIModelContainerRunner>(
        new torch::inductor::AOTIModelContainerRunnerCpu(
            so_path, num_runners, run_single_threaded));
}

rust::Vec<OwnedTensor> runner_run(
    torch::inductor::AOTIModelContainerRunner& runner,
    const rust::Vec<TensorPtr>& inputs) {
    std::vector<at::Tensor> cpp_inputs;
    cpp_inputs.reserve(inputs.size());
    for (const auto& t : inputs) {
        // The pointer is a const at::Tensor* from tch-rs (via torch-sys).
        // tch::Tensor is a repr(C) wrapper around *mut C_tensor, which is at::Tensor.
        const at::Tensor* tensor_ptr = reinterpret_cast<const at::Tensor*>(t.ptr);
        cpp_inputs.push_back(*tensor_ptr);
    }

    std::vector<at::Tensor> outputs = runner.run(cpp_inputs);

    rust::Vec<OwnedTensor> result;
    result.reserve(outputs.size());
    for (auto& out : outputs) {
        // Heap-allocate each output tensor so Rust can own it via a raw pointer.
        at::Tensor* heap_tensor = new at::Tensor(std::move(out));
        OwnedTensor ot;
        ot.ptr = static_cast<void*>(heap_tensor);
        result.push_back(ot);
    }
    return result;
}

rust::Vec<OwnedTensor> runner_boxed_run(
    torch::inductor::AOTIModelContainerRunner& runner,
    rust::Vec<TensorPtr>& inputs) {
    std::vector<at::Tensor> cpp_inputs;
    cpp_inputs.reserve(inputs.size());
    for (const auto& t : inputs) {
        const at::Tensor* tensor_ptr = reinterpret_cast<const at::Tensor*>(t.ptr);
        cpp_inputs.push_back(*tensor_ptr);
    }

    std::vector<at::Tensor> outputs = runner.boxed_run(std::move(cpp_inputs));

    rust::Vec<OwnedTensor> result;
    result.reserve(outputs.size());
    for (auto& out : outputs) {
        at::Tensor* heap_tensor = new at::Tensor(std::move(out));
        OwnedTensor ot;
        ot.ptr = static_cast<void*>(heap_tensor);
        result.push_back(ot);
    }
    return result;
}

rust::Vec<rust::String> runner_get_call_spec(
    torch::inductor::AOTIModelContainerRunner& runner) {
    auto specs = runner.get_call_spec();
    rust::Vec<rust::String> result;
    result.reserve(specs.size());
    for (const auto& s : specs) {
        result.push_back(rust::String(s));
    }
    return result;
}

rust::Vec<rust::String> runner_get_constant_fqns(
    torch::inductor::AOTIModelContainerRunner& runner) {
    auto fqns = runner.getConstantNamesToOriginalFQNs();
    rust::Vec<rust::String> result;
    result.reserve(fqns.size());
    for (const auto& kv : fqns) {
        result.push_back(rust::String(kv.second));
    }
    return result;
}

} // namespace aoti_rs
