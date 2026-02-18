use std::env;
use std::path::PathBuf;

fn main() {
    // Locate libtorch. We check the same environment variables that torch-sys uses,
    // plus DEP_TCH_LIBTORCH_LIB which torch-sys exports to dependent crates.
    let libtorch = env::var("LIBTORCH")
        .map(PathBuf::from)
        .ok();

    let libtorch_include = env::var("LIBTORCH_INCLUDE")
        .map(PathBuf::from)
        .or_else(|_| libtorch.as_ref()
            .map(|p| p.join("include"))
            .ok_or(()))
        .ok();

    let libtorch_lib = env::var("LIBTORCH_LIB")
        .map(PathBuf::from)
        .or_else(|_| env::var("DEP_TCH_LIBTORCH_LIB").map(PathBuf::from))
        .or_else(|_| libtorch.as_ref()
            .map(|p| p.join("lib"))
            .ok_or(()))
        .ok();

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
             or set LIBTORCH_INCLUDE directly."
        );
    };

    let lib_dir = libtorch_lib.unwrap_or_else(|| {
        panic!(
            "Cannot find libtorch lib directory. \
             Set the LIBTORCH environment variable to your libtorch installation, \
             or set LIBTORCH_LIB directly."
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
}
