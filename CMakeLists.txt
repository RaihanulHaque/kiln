cmake_minimum_required(VERSION 3.22)
project(kiln VERSION 0.5.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

option(KILN_BUILD_SHARED "Build libkiln as a shared library" ON)
option(KILN_BUILD_CLI    "Build kiln-cli test harness"      ON)
option(KILN_GPU          "Enable ggml GPU backend (Metal on Apple, Vulkan/OpenCL on Android if set)" ON)

# ---- llama.cpp (pinned submodule) ----
# A plain `git clone` (without --recursive) leaves the submodule empty; fetch it here.
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/CMakeLists.txt")
    find_package(Git QUIET)
    if(GIT_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
        message(STATUS "llama.cpp submodule missing, running: git submodule update --init --recursive")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            RESULT_VARIABLE KILN_SUBMODULE_RESULT)
    endif()
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/CMakeLists.txt")
        message(FATAL_ERROR
            "third_party/llama.cpp is empty. Run:\n"
            "  git submodule update --init --recursive\n"
            "or clone with: git clone --recursive https://github.com/RaihanulHaque/kiln")
    endif()
endif()

set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)   # chat templates (minja) live here
set(LLAMA_BUILD_MTMD     ON  CACHE BOOL "" FORCE)   # image and audio encoders (kiln_embed_media)
set(MTMD_VIDEO           OFF CACHE BOOL "" FORCE)   # video needs an ffmpeg binary; not on phones
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL        OFF CACHE BOOL "" FORCE)
set(LLAMA_SUBPROCESS     OFF CACHE BOOL "" FORCE)
set(LLAMA_LLGUIDANCE     OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL      OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)   # fold ggml+llama into libkiln

if(APPLE)
    set(GGML_METAL ${KILN_GPU} CACHE BOOL "" FORCE)
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
elseif(ANDROID)
    # kiln's Android baseline (docs/CODEBASE.md): without it ggml uses its generic ARM code,
    # several times slower. The preset, the Kotlin library and the Flutter plugin set it too;
    # this covers any other CMake build (Cargo's, for one). A builder can still pass another.
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    if(NOT GGML_CPU_ARM_ARCH)
        set(GGML_CPU_ARM_ARCH "armv8.2-a+dotprod+fp16" CACHE STRING "" FORCE)
    endif()
    # GPU on Android is opt-in: -DKILN_ANDROID_GPU=vulkan|opencl
    set(KILN_ANDROID_GPU "" CACHE STRING "vulkan, opencl, or empty")
    if(KILN_ANDROID_GPU STREQUAL "vulkan")
        set(GGML_VULKAN ON CACHE BOOL "" FORCE)
    elseif(KILN_ANDROID_GPU STREQUAL "opencl")
        set(GGML_OPENCL ON CACHE BOOL "" FORCE)
    endif()
    # OpenMP on Android, linked statically from the NDK's libomp.a: ggml then keeps its worker
    # threads alive between graphs. Without it the whisper.cpp decoders, which compute dozens of
    # tiny graphs per clip, start fresh threads for each one, and on a phone that took seconds.
    option(KILN_ANDROID_OPENMP "OpenMP on Android (off: ggml spawns threads per graph)" ON)
    set(GGML_OPENMP ${KILN_ANDROID_OPENMP} CACHE BOOL "" FORCE)
    if(KILN_ANDROID_OPENMP)
        execute_process(COMMAND ${CMAKE_CXX_COMPILER} --target=${CMAKE_CXX_COMPILER_TARGET} -print-file-name=libomp.a
                        OUTPUT_VARIABLE KILN_LIBOMP_A OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(NOT EXISTS "${KILN_LIBOMP_A}")
            message(FATAL_ERROR "NDK libomp.a not found (clang said: ${KILN_LIBOMP_A})")
        endif()
        set(OpenMP_omp_LIBRARY "${KILN_LIBOMP_A}" CACHE FILEPATH "" FORCE)   # FindOpenMP would pick libomp.so
    endif()
    set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
endif()

add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)

# ---- whisper.cpp (pinned submodule): Whisper and Parakeet speech models. It finds the ggml
# target llama.cpp already made (both pin ggml 0.26.0), so libkiln holds one copy of ggml. ----
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/whisper.cpp/CMakeLists.txt")
    message(FATAL_ERROR "third_party/whisper.cpp is empty. Run: git submodule update --init --recursive")
endif()
set(WHISPER_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(WHISPER_ALL_WARNINGS   OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/whisper.cpp EXCLUDE_FROM_ALL)

# ---- libkiln ----
set(KILN_SOURCES
    src/core/error.cpp
    src/core/registry.cpp
    src/core/kiln.cpp
    src/probe/probe.cpp
    src/backends/llamacpp/llamacpp_backend.cpp
    src/backends/whispercpp/whispercpp_backend.cpp
)

if(KILN_BUILD_SHARED)
    add_library(kiln SHARED ${KILN_SOURCES})
else()
    add_library(kiln STATIC ${KILN_SOURCES})
endif()

target_include_directories(kiln
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE third_party/llama.cpp/include third_party/llama.cpp/ggml/include third_party/llama.cpp/common
            third_party/llama.cpp/tools/mtmd third_party/llama.cpp/vendor third_party/whisper.cpp/include)
target_link_libraries(kiln PRIVATE llama-common mtmd llama whisper parakeet ggml)
if(NOT MSVC)
    # miniaudio.h (compiled into the whisper.cpp backend for reading audio files) casts away const
    set_source_files_properties(src/backends/whispercpp/whispercpp_backend.cpp PROPERTIES COMPILE_OPTIONS "-Wno-cast-qual")
endif()
target_compile_definitions(kiln PRIVATE KILN_BUILD)
if(ANDROID)
    target_link_libraries(kiln PRIVATE log)
    if(KILN_ANDROID_OPENMP)
        target_link_options(kiln PRIVATE -static-openmp)   # with OpenMP_omp_LIBRARY above: libomp.a goes into libkiln.so, nothing extra to ship
    endif()
    # miniaudio.h defines ma_android_sdk_version() without a prior prototype
    set_source_files_properties(src/backends/whispercpp/whispercpp_backend.cpp PROPERTIES COMPILE_OPTIONS "-Wno-cast-qual;-Wno-missing-prototypes")
endif()
set_target_properties(kiln PROPERTIES
    VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER include/kiln/kiln.h)

# ---- CLI + bench ----
if(KILN_BUILD_CLI)
    add_executable(kiln-cli tools/kiln-cli/main.cpp)
    add_executable(kiln-bench tools/kiln-bench/main.cpp)
    target_link_libraries(kiln-cli PRIVATE kiln)
    target_link_libraries(kiln-bench PRIVATE kiln)
    if(ANDROID)
        # let the binaries find libkiln.so next to themselves when pushed via adb
        set_target_properties(kiln-cli kiln-bench PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
    endif()
endif()

include(GNUInstallDirs)
install(TARGETS kiln
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/kiln)

# ---- tests ----
option(KILN_BUILD_TESTS "Build kiln tests" ON)
if(KILN_BUILD_TESTS)
    enable_testing()
    add_executable(kiln-tests tests/test_core.cpp)
    target_link_libraries(kiln-tests PRIVATE kiln)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(kiln-tests PRIVATE -Wno-format-zero-length)   # CHECK(cond, "")
    endif()
    if(ANDROID)
        set_target_properties(kiln-tests PROPERTIES BUILD_RPATH "$ORIGIN")
    endif()
    add_test(NAME core COMMAND kiln-tests ${CMAKE_CURRENT_SOURCE_DIR}/models)
endif()
