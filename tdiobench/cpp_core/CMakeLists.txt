cmake_minimum_required(VERSION 3.16)
project(eTIOBench_CPP VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find dependencies
find_package(Threads REQUIRED)

# Try to find pybind11 in different ways
find_package(pybind11 QUIET CONFIG)
if(NOT pybind11_FOUND)
    find_package(pybind11 QUIET)
endif()
if(NOT pybind11_FOUND)
    # Try finding via Python
    execute_process(
        COMMAND python3 -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(PYBIND11_CMAKE_DIR)
        set(pybind11_DIR "${PYBIND11_CMAKE_DIR}")
        find_package(pybind11 QUIET CONFIG)
    endif()
endif()

# Include directories
include_directories(common)

# Source files
set(SOURCES
    src/statistical_analyzer.cpp
    src/data_processor.cpp
    src/time_series_collector.cpp
    common/simd_utils.cpp
    common/threading_utils.cpp
)

# Create the library
add_library(etiobench_core STATIC ${SOURCES})

# Set compiler flags
target_compile_features(etiobench_core PUBLIC cxx_std_17)
target_compile_options(etiobench_core PRIVATE
    $<$<CONFIG:Release>:-O3>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra>
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra>
)

# Link libraries
target_link_libraries(etiobench_core PUBLIC Threads::Threads)

# Include directories for users of this library
target_include_directories(etiobench_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/common>
)

# Build Python bindings if pybind11 is available
if(pybind11_FOUND)
    add_subdirectory(python_bindings)
    message(STATUS "Building Python bindings")
else()
    message(WARNING "pybind11 not found - Python bindings will not be built")
endif()

# Print build info
message(STATUS "Build configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  pybind11: ${pybind11_FOUND}")
