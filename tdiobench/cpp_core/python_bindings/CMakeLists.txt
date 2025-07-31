cmake_minimum_required(VERSION 3.16)

# Find pybind11
find_package(pybind11 REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common)

# Core source files to include directly
set(CORE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/statistical_analyzer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/data_processor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/time_series_collector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/simd_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/threading_utils.cpp
)

# Python bindings source files
set(BINDING_SOURCES
    module.cpp
    statistical_analyzer_binding.cpp
    data_processor_binding.cpp
    time_series_collector_binding.cpp
)

# Create Python module with both core and binding sources
pybind11_add_module(etiobench_cpp ${BINDING_SOURCES} ${CORE_SOURCES})

# Link with threading
find_package(Threads REQUIRED)
target_link_libraries(etiobench_cpp PRIVATE Threads::Threads)

# Compiler-specific optimizations
target_compile_definitions(etiobench_cpp PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_compile_features(etiobench_cpp PRIVATE cxx_std_17)

# Set properties for Python module
set_target_properties(etiobench_cpp PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    INTERPROCEDURAL_OPTIMIZATION TRUE
)
