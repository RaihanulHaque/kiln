# Flutter/AGP entry point. Builds the kiln core.
#
# Two layouts are supported:
#   - inside the kiln monorepo: the core lives at ../../.. (repo root)
#   - published package:        the core was vendored into src/core/ by scripts/vendor_flutter.sh
cmake_minimum_required(VERSION 3.22)
project(kiln_flutter LANGUAGES C CXX)

set(KILN_BUILD_CLI   OFF CACHE BOOL "" FORCE)
set(KILN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(KILN_GPU         OFF CACHE BOOL "" FORCE)
if(ANDROID)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(GGML_CPU_ARM_ARCH "armv8.2-a+dotprod+fp16" CACHE STRING "" FORCE)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/CMakeLists.txt")
    message(STATUS "kiln_flutter: building vendored core (src/core)")
    set(KILN_CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/core")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../../CMakeLists.txt")
    message(STATUS "kiln_flutter: building core from the monorepo root")
    set(KILN_CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
else()
    message(FATAL_ERROR "kiln_flutter: no core found. Run scripts/vendor_flutter.sh in the kiln repo, or build inside the monorepo.")
endif()
add_subdirectory("${KILN_CORE_DIR}" kiln_core)

if(ANDROID)
    target_link_options(kiln PRIVATE "-Wl,-z,max-page-size=16384")   # Android 15 16 KB pages

    # Phones whose processor has no dot-product instruction (Armv8.0, e.g. Snapdragon 680)
    # crash on the default build above with SIGILL. Build the core a second time with the
    # plain Armv8.0 baseline, as libkiln_armv8.so. native.dart loads it on those phones.
    # A separate CMake project (not add_subdirectory) because ggml's target names are global.
    include(ExternalProject)
    set(KILN_ARMV8_BUILD "${CMAKE_CURRENT_BINARY_DIR}/kiln_armv8")
    ExternalProject_Add(kiln_armv8
        SOURCE_DIR "${KILN_CORE_DIR}"
        BINARY_DIR "${KILN_ARMV8_BUILD}"
        CMAKE_ARGS
            -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
            -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
            -DANDROID_ABI=${ANDROID_ABI}
            -DANDROID_PLATFORM=${ANDROID_PLATFORM}
            -DANDROID_STL=${ANDROID_STL}
            -DANDROID_NDK=${ANDROID_NDK}
            -DCMAKE_BUILD_TYPE=Release
            -DKILN_BUILD_CLI=OFF
            -DKILN_BUILD_TESTS=OFF
            -DKILN_GPU=OFF
            -DGGML_NATIVE=OFF
            -DGGML_CPU_ARM_ARCH=armv8-a
        BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target kiln
        INSTALL_COMMAND ${CMAKE_COMMAND} -E copy
            "<BINARY_DIR>/libkiln.so" "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libkiln_armv8.so"
        BUILD_BYPRODUCTS "${KILN_ARMV8_BUILD}/libkiln.so")
    add_dependencies(kiln kiln_armv8)
endif()
