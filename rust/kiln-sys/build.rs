// Builds libkiln with CMake and links it into the Rust program.
//
// Two ways to get the library:
//  - default: build it from the kiln source tree (the repository this crate sits in, or
//    KILN_SOURCE_DIR) as a static library, so the program needs no libkiln.so at run time.
//    llama.cpp and whisper.cpp are compiled once per target directory; that takes minutes.
//  - KILN_LIB_DIR=<dir>: link a libkiln.so / libkiln.dylib already built there, for example
//    build/host after `cmake --build --preset host`. The program then needs that library at run
//    time (LD_LIBRARY_PATH, or next to the executable).
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=KILN_LIB_DIR");
    println!("cargo:rerun-if-env-changed=KILN_SOURCE_DIR");

    if let Ok(dir) = env::var("KILN_LIB_DIR") {
        println!("cargo:rustc-link-search=native={dir}");
        println!("cargo:rustc-link-lib=dylib=kiln");
        return;
    }

    // Linux, macOS and Android (arm64, with the NDK) are built. MSVC writes .lib archives this
    // build does not look for, and nobody has tried Windows: refuse instead of failing at link time.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    if target_os == "windows" {
        panic!(
            "building libkiln from Cargo is not supported on Windows yet; build it with CMake \
             and set KILN_LIB_DIR to the folder that holds the library"
        );
    }
    println!("cargo:rerun-if-env-changed=ANDROID_NDK_HOME");

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let src = env::var("KILN_SOURCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| manifest.join("../.."));
    let src = src
        .canonicalize()
        .unwrap_or_else(|_| panic!("kiln source not found at {}; set KILN_SOURCE_DIR", src.display()));
    if !src.join("third_party/llama.cpp/CMakeLists.txt").exists() {
        panic!(
            "{}/third_party/llama.cpp is empty: run `git submodule update --init --recursive`",
            src.display()
        );
    }
    for p in [
        "CMakeLists.txt",
        "include",
        "src",
        "cmake",
        "third_party/llama.cpp",
        "third_party/whisper.cpp",
    ] {
        println!("cargo:rerun-if-changed={}", src.join(p).display());
    }

    // Always Release: a Debug llama.cpp is many times slower, and Rust's debug profile would
    // otherwise pick it.
    let mut cfg = cmake::Config::new(&src);
    cfg.profile("Release")
        .define("KILN_BUILD_SHARED", "OFF")
        .define("KILN_BUILD_CLI", "OFF")
        .define("KILN_BUILD_TESTS", "OFF")
        .build_target("kiln");
    if target_os == "android" {
        // the same settings as the CMake preset android-arm64 and the Flutter plugin
        let ndk = android_ndk();
        cfg.define("CMAKE_TOOLCHAIN_FILE", ndk.join("build/cmake/android.toolchain.cmake"))
            .define("ANDROID_ABI", "arm64-v8a")
            .define("ANDROID_PLATFORM", "android-29")
            .define("ANDROID_STL", "c++_static");
    }
    let dst = cfg.build();
    let build = dst.join("build");

    // CMake leaves libkiln.a and the static libraries of llama.cpp, ggml and whisper.cpp spread
    // over its build tree. Link all of them. Rust packs them into this crate's archive, and the
    // linker searches one archive until nothing new is found, so their order does not matter.
    let mut libs = Vec::new();
    find_static_libs(&build, &mut libs);
    if !libs.iter().any(|(_, name)| name == "kiln") {
        panic!("libkiln.a not found under {}", build.display());
    }
    libs.sort();
    for (dir, name) in &libs {
        println!("cargo:rustc-link-search=native={}", dir.display());
        println!("cargo:rustc-link-lib=static={name}");
    }

    // What the C++ code needs from the system.
    let cache = fs::read_to_string(build.join("CMakeCache.txt")).unwrap_or_default();
    match target_os.as_str() {
        "android" => {
            // the NDK's static C++ runtime, and its static OpenMP (the libomp.a CMake was told
            // to use), so the program needs no libc++_shared.so or libomp.so
            let prebuilt = android_ndk().join("toolchains/llvm/prebuilt");
            // one folder per host OS (darwin-x86_64, linux-x86_64); skip files such as .DS_Store
            let host = fs::read_dir(&prebuilt)
                .ok()
                .and_then(|d| d.flatten().map(|e| e.path()).find(|p| p.is_dir()))
                .unwrap_or_else(|| panic!("no prebuilt toolchain in {}", prebuilt.display()));
            // Copy just the two C++ runtime archives: their NDK folder also holds a static libc,
            // libm and libdl, and putting it on the search path links those into the program,
            // which then crashes at start (a static libc in a dynamic executable).
            let ndk_lib = host.join("sysroot/usr/lib/aarch64-linux-android");
            let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("ndk-cxx");
            fs::create_dir_all(&out).unwrap();
            for a in ["libc++_static.a", "libc++abi.a"] {
                fs::copy(ndk_lib.join(a), out.join(a)).unwrap_or_else(|e| panic!("{a}: {e}"));
            }
            println!("cargo:rustc-link-search=native={}", out.display());
            println!("cargo:rustc-link-lib=static=c++_static");
            println!("cargo:rustc-link-lib=static=c++abi");
            // copied for the same reason: its folder holds other runtime libraries too
            if let Some(omp) = cache_value(&cache, "OpenMP_omp_LIBRARY").map(PathBuf::from) {
                fs::copy(&omp, out.join("libomp.a")).unwrap_or_else(|e| panic!("{}: {e}", omp.display()));
                println!("cargo:rustc-link-lib=static=omp");
            }
            for l in ["log", "dl", "m"] {
                println!("cargo:rustc-link-lib=dylib={l}");
            }
        }
        "macos" | "ios" => {
            println!("cargo:rustc-link-lib=dylib=c++");
            for fw in ["Foundation", "Metal", "MetalKit", "Accelerate"] {
                println!("cargo:rustc-link-lib=framework={fw}");
            }
        }
        _ => {
            println!("cargo:rustc-link-lib=dylib=stdc++");
            // ggml's OpenMP: GCC's libgomp on Linux (the names CMake found, e.g. "gomp;pthread")
            if let Some(names) = cache_value(&cache, "OpenMP_CXX_LIB_NAMES") {
                for n in names.split(';').filter(|n| !n.is_empty()) {
                    println!("cargo:rustc-link-lib=dylib={n}");
                }
            }
            println!("cargo:rustc-link-lib=dylib=m");
            println!("cargo:rustc-link-lib=dylib=dl");
        }
    }
}

fn find_static_libs(dir: &Path, out: &mut Vec<(PathBuf, String)>) {
    let Ok(entries) = fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            find_static_libs(&p, out);
        } else if let Some(file) = p.file_name().and_then(|f| f.to_str()) {
            if let Some(name) = file.strip_prefix("lib").and_then(|f| f.strip_suffix(".a")) {
                out.push((dir.to_path_buf(), name.to_string()));
            }
        }
    }
}

fn cache_value<'a>(cache: &'a str, key: &str) -> Option<&'a str> {
    cache
        .lines()
        .find_map(|l| l.strip_prefix(key)?.strip_prefix(':')?.split_once('=').map(|(_, v)| v))
}

/// The NDK: ANDROID_NDK_HOME or ANDROID_NDK_ROOT, else the newest one in the SDK's ndk/ folder
/// (ANDROID_HOME, or Android Studio's default location on macOS and Linux).
fn android_ndk() -> PathBuf {
    for v in ["ANDROID_NDK_HOME", "ANDROID_NDK_ROOT"] {
        if let Ok(p) = env::var(v) {
            return PathBuf::from(p);
        }
    }
    let home = env::var("HOME").unwrap_or_default();
    let sdks = [
        env::var("ANDROID_HOME").map(PathBuf::from).ok(),
        Some(PathBuf::from(&home).join("Library/Android/sdk")),
        Some(PathBuf::from(&home).join("Android/Sdk")),
    ];
    for sdk in sdks.into_iter().flatten() {
        let mut versions: Vec<PathBuf> = fs::read_dir(sdk.join("ndk"))
            .map(|d| d.flatten().map(|e| e.path()).collect())
            .unwrap_or_default();
        versions.sort();
        if let Some(newest) = versions.pop() {
            return newest;
        }
    }
    panic!("Android NDK not found: set ANDROID_NDK_HOME");
}
