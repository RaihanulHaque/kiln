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
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/core" kiln_core)
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../../CMakeLists.txt")
    message(STATUS "kiln_flutter: building core from the monorepo root")
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../.." kiln_core)
else()
    message(FATAL_ERROR "kiln_flutter: no core found. Run scripts/vendor_flutter.sh in the kiln repo, or build inside the monorepo.")
endif()

if(ANDROID)
    target_link_options(kiln PRIVATE "-Wl,-z,max-page-size=16384")   # Android 15 16 KB pages
endif()
