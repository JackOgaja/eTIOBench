#!/usr/bin/env python3
"""
Setup script for eTIOBench C++ extensions.
"""

import os
import sys
from pathlib import Path
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, Extension

# Check if we have the required dependencies
try:
    import pybind11
    import numpy
except ImportError as e:
    print(f"Required dependency missing: {e}")
    print("Please install: pip install pybind11 numpy")
    sys.exit(1)

# Custom build_ext class to control output directory
class CustomBuildExt(build_ext):
    def build_extension(self, ext):
        # Create lib directory if it doesn't exist
        lib_dir = os.path.join(os.path.dirname(__file__), 'lib')
        os.makedirs(lib_dir, exist_ok=True)
        
        # Call the parent build method
        super().build_extension(ext)
        
        # Move the built extension to the lib directory
        built_ext = self.get_ext_fullpath(ext.name)
        target_path = os.path.join(lib_dir, os.path.basename(built_ext))
        
        if os.path.exists(built_ext):
            import shutil
            shutil.move(built_ext, target_path)
            print(f"Extension moved to: {target_path}")

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "etiobench_cpp",
        [
            "src/statistical_analyzer.cpp",
            "src/data_processor.cpp",
            "src/time_series_collector.cpp",
            "common/simd_utils.cpp",
            "common/threading_utils.cpp",
            "python_bindings/statistical_analyzer_binding.cpp",
            "python_bindings/data_processor_binding.cpp",
            "python_bindings/time_series_collector_binding.cpp",
            "python_bindings/module.cpp"
        ],
        include_dirs=[
            "include",
            "common",
            pybind11.get_cmake_dir() + "/../../../include",
            numpy.get_include()
        ],
        cxx_std=17,
        define_macros=[("VERSION_INFO", '"dev"')],
    ),
]

setup(
    name="etiobench_cpp",
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt},
    zip_safe=False,
    python_requires=">=3.7",
)
