# nes_cpp: C++ eikonal kernels for NES (factored fast marching and NES inference).
#
# The same file serves two builds:
#   * C++ tests, from CLion or the command line (run from the repository root):
#       cmake -S NES/cpp -B NES/cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Debug
#       cmake --build NES/cpp/build
#       ctest --test-dir NES/cpp/build --output-on-failure
#   * the Python module `nes_cpp`, when pip runs CMake through scikit-build-core:
#       pip install ./NES/cpp
# scikit-build-core sets the variable SKBUILD. A pip build makes the module and skips the tests.

cmake_minimum_required(VERSION 3.24)
project(nes_cpp LANGUAGES CXX)

# C++20 for every target, without compiler extensions (-std=c++20, not -std=gnu++20).
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags for our own targets only; fetched dependencies keep their own flags.
if(MSVC)
    set(NES_CPP_FLAGS /W4 /utf-8)  # /utf-8: read sources as UTF-8 (comments use math symbols)
else()
    set(NES_CPP_FLAGS -Wall -Wextra -Wpedantic)
endif()

# ---- Dependencies -------------------------------------------------------------------------------
include(FetchContent)

# Eigen 3.4 (header-only): dense matrices for the NES forward pass.
# SOURCE_SUBDIR names a folder that does not exist, so CMake only downloads the headers and does
# not configure Eigen's own large build. We then describe the headers as a target ourselves.
FetchContent_Declare(eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.1/eigen-3.4.1.tar.gz
    URL_HASH SHA256=b93c667d1b69265cdb4d9f30ec21f8facbbe8b307cf34c0b9942834c6d4fdbe2
    SOURCE_SUBDIR no-cmake-build)
FetchContent_MakeAvailable(eigen)
add_library(Eigen3::Eigen INTERFACE IMPORTED)  # same name as Eigen's own CMake package
target_include_directories(Eigen3::Eigen INTERFACE ${eigen_SOURCE_DIR})

# OpenMP is optional: without it, everything runs on one thread.
find_package(OpenMP)

# ---- The C++ kernels, as one static library ---------------------------------------------------
add_library(nes_cpp_core STATIC src/fmm.cpp src/nes_model.cpp src/nes_train.cpp)
target_include_directories(nes_cpp_core PUBLIC include) # users write #include "nes_cpp/fmm.hpp"
target_compile_options(nes_cpp_core PRIVATE ${NES_CPP_FLAGS})
# PUBLIC: nes_model.hpp includes Eigen, so code using it needs Eigen too. Parallelism comes from our
# own loop over batches; Eigen's internal OpenMP is switched off everywhere (same setting in every
# file that includes Eigen).
target_link_libraries(nes_cpp_core PUBLIC Eigen3::Eigen)
target_compile_definitions(nes_cpp_core PUBLIC EIGEN_DONT_PARALLELIZE)
# The Python module is a shared library that links this static one, so on Linux its code must be
# position independent (-fPIC).
set_target_properties(nes_cpp_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(OpenMP_CXX_FOUND)
    # PRIVATE: only our sources get the OpenMP flag; programs linking the library still get the
    # OpenMP runtime.
    target_link_libraries(nes_cpp_core PRIVATE OpenMP::OpenMP_CXX)
endif()

# ---- Python module (pip install) or C++ tests (CLion, command line) -----------------------------
if(SKBUILD)
    set(PYBIND11_FINDPYTHON ON)  # find the Python that runs pip with CMake's FindPython
    find_package(pybind11 CONFIG REQUIRED)  # pip installs pybind11 first (see pyproject.toml)

    pybind11_add_module(nes_cpp bindings/module.cpp)
    target_link_libraries(nes_cpp PRIVATE nes_cpp_core)
    target_compile_options(nes_cpp PRIVATE ${NES_CPP_FLAGS})
    # The version is written once, in pyproject.toml; scikit-build-core passes it to CMake.
    target_compile_definitions(nes_cpp PRIVATE NES_CPP_VERSION="${SKBUILD_PROJECT_VERSION}")
    install(TARGETS nes_cpp LIBRARY DESTINATION .)  # top level of site-packages
else()
    # Catch2 v3: unit tests. Catch2WithMain supplies main(), so test files hold only tests.
    FetchContent_Declare(catch2
        URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.16.1.tar.gz
        URL_HASH SHA256=d16bd1b8b2364918bd472cb216394acb93bde6784890df0d91bccfb3f28d777d)
    FetchContent_MakeAvailable(catch2)

    add_executable(tests tests/cpp/test_build.cpp tests/cpp/test_fmm.cpp
                         tests/cpp/test_nes_model.cpp tests/cpp/test_nes_train.cpp)
    target_link_libraries(tests PRIVATE nes_cpp_core Catch2::Catch2WithMain)
    target_include_directories(tests PRIVATE src) # tests also check internal headers in src/
    if(OpenMP_CXX_FOUND)
        target_link_libraries(tests PRIVATE OpenMP::OpenMP_CXX)
    endif()
    target_compile_options(tests PRIVATE ${NES_CPP_FLAGS})

    enable_testing()
    add_test(NAME tests COMMAND tests)  # one CTest entry that runs every Catch2 test case
endif()
