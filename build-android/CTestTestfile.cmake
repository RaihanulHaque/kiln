# CMake generated Testfile for 
# Source directory: /Users/rahi/Code/kiln
# Build directory: /Users/rahi/Code/kiln/build-android
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test("core" "/Users/rahi/Code/kiln/build-android/kiln-tests" "/Users/rahi/Code/kiln/models")
set_tests_properties("core" PROPERTIES  _BACKTRACE_TRIPLES "/Users/rahi/Code/kiln/CMakeLists.txt;155;add_test;/Users/rahi/Code/kiln/CMakeLists.txt;0;")
subdirs("third_party/llama.cpp")
subdirs("third_party/whisper.cpp")
