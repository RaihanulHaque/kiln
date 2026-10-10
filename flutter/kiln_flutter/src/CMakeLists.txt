# Flutter/AGP entry point. Builds the kiln core from the repo root so there is one source of truth.
cmake_minimum_required(VERSION 3.22)
project(kiln_flutter LANGUAGES C CXX)

set(KILN_BUILD_CLI   OFF CACHE BOOL "" FORCE)
set(KILN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(KILN_GPU         OFF CACHE BOOL "" FORCE)
if(ANDROID)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(GGML_CPU_ARM_ARCH "armv8.2-a+dotprod+fp16" CACHE STRING "" FORCE)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../.. kiln_core)

if(ANDROID)
    # Android 15 16 KB page size
    target_link_options(kiln PRIVATE "-Wl,-z,max-page-size=16384")
endif()
