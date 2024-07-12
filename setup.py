from glob import glob
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

import subprocess
import os, sys
import re
import platform

__version__ = "0.0.1"

def find_cuda_version():
    try:
        output = subprocess.check_output(["nvcc", "--version"]).decode()
        for line in output.split('\n'):
            if "release" in line:
                version = line.split("release")[-1].strip().split(",")[0]
                return version
    except Exception as e:
        print(f"ERROR: Could not find CUDA version: {e}")
        sys.exit(1)

if platform.system() == "Darwin":
    try:
        include_dirs = [subprocess.check_output(["xcrun", "--show-sdk-path"], stderr=subprocess.STDOUT).decode().strip() + "/System/Library/Frameworks/Metal.framework/Headers"]
        if not os.path.exists(include_dirs[0]):
            raise Exception()
        
        ext_modules = [
            Pybind11Extension(
                "grand._gmetal",
                sorted(glob("grand/gmetal/**/*.cpp", recursive=True)),
                include_dirs=include_dirs,
                extra_link_args=["Metal"],
            ),
        ]
    except Exception as e:
        print(f"ERROR: Grand requires Xcode command line tools to be installed.")
        sys.exit(1)

if platform.system() == "Windows":
    try:
        cuda_version = find_cuda_version()
        cuda_include_dir = f"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v{cuda_version}/include"
        cuda_lib_dir = f"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v{cuda_version}/lib/x64"
        
        include_dirs = [cuda_include_dir]
        library_dirs = [cuda_lib_dir]
        
        ext_modules = [
            Pybind11Extension(
                "grand._gcuda",
                sorted(glob("grand/gcuda/**/*.cpp", recursive=True)),
                include_dirs=include_dirs,
                library_dirs=library_dirs,
                libraries=["cudart", "cublas", "cufft"],
                extra_link_args=[],
            ),
        ]
    except Exception as e:
        print(f"ERROR: Failed to set up CUDA directories: {e}")
        sys.exit(1)

if platform.system() == "Linux":
    pass

setup(
    name="grand",
    version=__version__,
    author="Adam L.",
    author_email="adamloeckle@gmail.com",
    url="https://github.com/adamloec/grand",
    description="Definitely a tensor framework.",
    long_description="",
    packages=["grand"],
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    setup_requires=[
        "pybind11",
    ],
    install_requires=[
    ],
    zip_safe=False,
    python_requires=">=3.7",
)