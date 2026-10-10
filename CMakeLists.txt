cmake_minimum_required(VERSION 3.22)
project(kiln VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

option(KILN_BUILD_SHARED "Build libkiln as a shared library" ON)
option(KILN_BUILD_CLI    "Build kiln-cli test harness"      ON)
option(KILN_GPU          "Enable ggml GPU backend (Metal on Apple, Vulkan/OpenCL on Android if set)" ON)

# ---- llama.cpp (pinned submodule) ----
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON   OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL      OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)   # fold ggml+llama into libkiln

if(APPLE)
    set(GGML_METAL ${KILN_GPU} CACHE BOOL "" FORCE)
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
elseif(ANDROID)
    # GPU on Android is opt-in: -DKILN_ANDROID_GPU=vulkan|opencl
    set(KILN_ANDROID_GPU "" CACHE STRING "vulkan, opencl, or empty")
    if(KILN_ANDROID_GPU STREQUAL "vulkan")
        set(GGML_VULKAN ON CACHE BOOL "" FORCE)
    elseif(KILN_ANDROID_GPU STREQUAL "opencl")
        set(GGML_OPENCL ON CACHE BOOL "" FORCE)
    endif()
    set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
    set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
endif()

add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)

# ---- libkiln ----
set(KILN_SOURCES
    src/core/error.cpp
    src/core/registry.cpp
    src/core/kiln.cpp
    src/probe/probe.cpp
    src/backends/llamacpp/llamacpp_backend.cpp
)

if(KILN_BUILD_SHARED)
    add_library(kiln SHARED ${KILN_SOURCES})
else()
    add_library(kiln STATIC ${KILN_SOURCES})
endif()

target_include_directories(kiln
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE third_party/llama.cpp/include third_party/llama.cpp/ggml/include)
target_link_libraries(kiln PRIVATE llama ggml)
target_compile_definitions(kiln PRIVATE KILN_BUILD)
if(ANDROID)
    target_link_libraries(kiln PRIVATE log)
endif()
set_target_properties(kiln PROPERTIES
    VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER include/kiln/kiln.h)

# ---- CLI ----
if(KILN_BUILD_CLI)
    add_executable(kiln-cli tools/kiln-cli/main.cpp)
    target_link_libraries(kiln-cli PRIVATE kiln)
    if(ANDROID)
        # let the binary find libkiln.so next to itself when pushed via adb
        set_target_properties(kiln-cli PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
    endif()
endif()

include(GNUInstallDirs)
install(TARGETS kiln
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/kiln)
