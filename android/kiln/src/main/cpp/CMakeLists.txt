# Builds libkiln_jni.so: the JNI glue with the kiln core (llama.cpp, whisper.cpp) linked in
# statically, so an app ships one library. Gradle uses this for Android; the host unit tests
# build the same file for the desktop JVM (see ../../../build.gradle.kts).
cmake_minimum_required(VERSION 3.22)
project(kiln_jni LANGUAGES C CXX)

set(KILN_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(KILN_BUILD_CLI    OFF CACHE BOOL "" FORCE)
set(KILN_BUILD_TESTS  OFF CACHE BOOL "" FORCE)
set(KILN_GPU          OFF CACHE BOOL "" FORCE)
if(ANDROID)
    # the same baseline as the Flutter plugin: every arm64 phone kiln supports has dotprod
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(GGML_CPU_ARM_ARCH "armv8.2-a+dotprod+fp16" CACHE STRING "" FORCE)
endif()

set(KILN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../.." CACHE PATH "kiln repository root")
add_subdirectory("${KILN_ROOT}" kiln_core EXCLUDE_FROM_ALL)

add_library(kiln_jni SHARED kiln_jni.cpp)
target_link_libraries(kiln_jni PRIVATE kiln)
if(ANDROID)
    target_link_libraries(kiln_jni PRIVATE log)
    target_link_options(kiln_jni PRIVATE "-Wl,-z,max-page-size=16384")   # Android 15 16 KB pages
    if(KILN_ANDROID_OPENMP)
        # libomp.a into libkiln_jni.so, nothing extra to ship (as for libkiln.so in the root CMakeLists)
        target_link_options(kiln_jni PRIVATE -static-openmp)
    endif()
else()
    find_package(JNI REQUIRED)
    target_include_directories(kiln_jni PRIVATE ${JNI_INCLUDE_DIRS})
endif()
