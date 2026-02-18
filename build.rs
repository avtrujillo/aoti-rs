use std::env;
use std::path::PathBuf;
use std::process::Command;

/// Try to find the torch package root via Python (same as LIBTORCH_USE_PYTORCH in torch-sys).
fn find_torch_from_python() -> Option<PathBuf> {
    let output = Command::new("python3")
        .args(["-c", "import torch; print(torch.__file__)"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let torch_init = String::from_utf8(output.stdout).ok()?.trim().to_string();
    // torch.__file__ is e.g. /path/to/site-packages/torch/__init__.py
    PathBuf::from(torch_init).parent().map(|p| p.to_path_buf())
}

fn main() {
    // Locate libtorch. We try, in order:
    // 1. Explicit LIBTORCH / LIBTORCH_INCLUDE / LIBTORCH_LIB env vars
    // 2. DEP_TCH_LIBTORCH_LIB exported by torch-sys (derive include from ../include)
    // 3. LIBTORCH_USE_PYTORCH: find torch via Python import
    let libtorch = env::var("LIBTORCH").map(PathBuf::from).ok();

    // Try to get the lib dir from torch-sys or derive from LIBTORCH
    let dep_lib = env::var("DEP_TCH_LIBTORCH_LIB").map(PathBuf::from).ok();

    // If we have DEP_TCH_LIBTORCH_LIB, derive the torch root (lib dir's parent)
    let torch_root_from_dep = dep_lib.as_ref().and_then(|lib| lib.parent().map(|p| p.to_path_buf()));

    // As a last resort, find torch via Python
    let torch_from_python = find_torch_from_python();

    // Resolve the effective torch root directory
    let torch_root = libtorch
        .as_ref()
        .or(torch_root_from_dep.as_ref())
        .or(torch_from_python.as_ref());

    let libtorch_include = env::var("LIBTORCH_INCLUDE")
        .map(PathBuf::from)
        .ok()
        .or_else(|| torch_root.map(|p| p.join("include")));

    let libtorch_lib = env::var("LIBTORCH_LIB")
        .map(PathBuf::from)
        .ok()
        .or(dep_lib)
        .or_else(|| torch_root.map(|p| p.join("lib")));

    let include_dirs: Vec<PathBuf> = if let Some(ref inc) = libtorch_include {
        // libtorch ships headers under include/ and include/torch/csrc/api/include/
        let api_include = inc.join("torch").join("csrc").join("api").join("include");
        let mut dirs = vec![inc.clone()];
        if api_include.exists() {
            dirs.push(api_include);
        }
        dirs
    } else {
        panic!(
            "Cannot find libtorch include directory. \
             Set the LIBTORCH environment variable to your libtorch installation, \
             set LIBTORCH_INCLUDE directly, or ensure PyTorch is installed in Python."
        );
    };

    let lib_dir = libtorch_lib.unwrap_or_else(|| {
        panic!(
            "Cannot find libtorch lib directory. \
             Set the LIBTORCH environment variable to your libtorch installation, \
             set LIBTORCH_LIB directly, or ensure PyTorch is installed in Python."
        );
    });

    // Build the C++ bridge via cxx
    let mut build = cxx_build::bridge("src/lib.rs");
    build
        .file("csrc/aoti.cc")
        .std("c++17")
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-missing-field-initializers");

    for dir in &include_dirs {
        build.include(dir);
    }

    // Also include the crate root so #include "csrc/aoti.h" resolves
    // when referenced from the generated cxx code.
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    build.include(&manifest_dir);

    build.compile("aoti_rs_bridge");

    // Link against libtorch libraries
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=torch");
    println!("cargo:rustc-link-lib=dylib=torch_cpu");
    println!("cargo:rustc-link-lib=dylib=c10");

    // Optionally link CUDA libs if available
    let torch_cuda = lib_dir.join("libtorch_cuda.so");
    if torch_cuda.exists() {
        println!("cargo:rustc-link-lib=dylib=torch_cuda");
    }

    // Set rpath so the dynamic linker can find libtorch at runtime
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());

    // Rerun if sources change
    println!("cargo:rerun-if-changed=csrc/aoti.h");
    println!("cargo:rerun-if-changed=csrc/aoti.cc");
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rerun-if-env-changed=LIBTORCH_INCLUDE");
    println!("cargo:rerun-if-env-changed=LIBTORCH_LIB");
    println!("cargo:rerun-if-env-changed=LIBTORCH_USE_PYTORCH");
}
