use std::env;
use std::path::PathBuf;
use std::process::Command;

/// Remove an existing symlink (or file) at `dst`, then create a new symlink
/// pointing to `src`.  Errors are intentionally ignored – IDE symlinks are
/// best-effort and must never break the build.
#[cfg(unix)]
fn force_symlink(src: &std::path::Path, dst: &std::path::Path) {
    let _ = std::fs::remove_file(dst);
    let _ = std::os::unix::fs::symlink(src, dst);
}

/// Look for a directory containing `cuda_runtime_api.h`.  Checked in order:
/// `CUDA_HOME` / `CUDA_PATH` / `CUDA_ROOT`, then common system install
/// locations.  We deliberately don't probe pip's `nvidia-*-cuXX` wheels —
/// those split the toolkit across many wheels and finding one header doesn't
/// mean the rest will resolve.
fn find_cuda_include() -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();

    for var in ["CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"] {
        if let Ok(v) = env::var(var) {
            candidates.push(PathBuf::from(&v).join("include"));
        }
    }
    for p in [
        "/usr/local/cuda/include",
        "/opt/cuda/include",
        "/usr/include/cuda",
    ] {
        candidates.push(PathBuf::from(p));
    }

    candidates
        .into_iter()
        .find(|p| p.join("cuda_runtime_api.h").exists())
}

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
    let torch_root_from_dep = dep_lib
        .as_ref()
        .and_then(|lib| lib.parent().map(|p| p.to_path_buf()));

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

    // Detect CUDA support: presence of libtorch_cuda.so in the lib dir.  Set
    // AOTI_RS_NO_CUDA=1 to force a CPU-only build even when libtorch ships
    // with CUDA support (useful for type-checking in environments without
    // CUDA headers installed).
    let force_no_cuda = env::var_os("AOTI_RS_NO_CUDA").is_some();
    let has_cuda = !force_no_cuda && lib_dir.join("libtorch_cuda.so").exists();

    // Expose CUDA availability to the Rust side as cfg(aoti_cuda) so that
    // CUDA-only APIs (AOTIModelBuilder::<Cuda>::build, etc.) are compiled
    // out — not just failing at runtime — when CUDA support is absent.
    println!("cargo::rustc-check-cfg=cfg(aoti_cuda)");
    if has_cuda {
        println!("cargo:rustc-cfg=aoti_cuda");
    }

    // When libtorch is CUDA-enabled, model_container_runner_cuda.h transitively
    // pulls in c10/cuda/CUDAStream.h -> cuda_runtime_api.h, which ships with
    // the CUDA toolkit (not libtorch).  Try to find the toolkit's include dir
    // so the user doesn't have to set CUDA_HOME manually if it's in a standard
    // location.
    let cuda_include = if has_cuda { find_cuda_include() } else { None };

    // Build the C++ bridge via cxx
    let mut build = cxx_build::bridge("src/lib.rs");
    build
        .file("csrc/aoti.cc")
        .std("c++17")
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-missing-field-initializers");

    if has_cuda {
        build.define("USE_CUDA", None);
        if let Some(ref cuda_inc) = cuda_include {
            build.include(cuda_inc);
        } else {
            // Don't fail the build here: the system compiler might still find
            // cuda_runtime_api.h on its default include path (e.g. when CUDA
            // is installed via the OS package manager into /usr/include).
            // If it can't, the cc invocation will fail with a clear "fatal
            // error: cuda_runtime_api.h: No such file or directory" — and
            // the warning below tells the user how to fix it.
            println!(
                "cargo:warning=libtorch ships libtorch_cuda.so but no CUDA \
                 toolkit include dir was found.  If the build fails on \
                 cuda_runtime_api.h, install the CUDA toolkit and set \
                 CUDA_HOME (or CUDA_PATH), or set AOTI_RS_NO_CUDA=1 to \
                 build a CPU-only runner."
            );
        }
    }

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

    if has_cuda {
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
    println!("cargo:rerun-if-env-changed=AOTI_RS_NO_CUDA");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");

    // Create a stable symlink under target/ so that the checked-in .clangd
    // file can reference libtorch headers via a fixed relative path.
    // (cxx-build already creates target/cxxbridge/ with its own symlinks.)
    #[cfg(unix)]
    {
        if let Some(root) = torch_root {
            let target_dir = PathBuf::from(&manifest_dir).join("target");
            force_symlink(root, &target_dir.join("libtorch"));
        }
    }
}
