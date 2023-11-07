from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import subprocess, os
import re
import platform

__version__ = "0.0.1"

# Finds the currently installed version of CUDA, if no CUDA is installed raises install exception.
def get_cuda():
    try:
        # Run nvcc to check if it's installed
        result = subprocess.run(["nvcc", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Check if nvcc was found
        if result.returncode != 0:
            raise EnvironmentError("CUDA is not installed on your system. Please install CUDA v11.*.")

        # Parse the version information to get the CUDA version
        version_info = result.stdout
        match = re.search(r"release (\d+\.\d+)", version_info)
        if match:
            cuda_version = match.group(1)
        else:
            raise EnvironmentError("Failed to extract CUDA version from nvcc output.")

        if float(cuda_version) <= 11.0:
            raise EnvironmentError("CUDA version 11.* or 12.* required. Please upgrade your current CUDA installation.")
        return cuda_version
    except Exception as e:
        raise EnvironmentError(f"Error while checking CUDA installation: {e}")

cuda_version = get_cuda()

if platform.system() == "Windows":
    cuda_include_dir = f"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v{cuda_version}/include"
else:
    cuda_include_dir = f"/usr/local/cuda-{cuda_version}/include"

ext_modules = [
    Pybind11Extension(
        "grand.gcuda",
        ["gcuda/ops.cpp"],
        define_macros=[("VERSION_INFO", __version__)],
        include_dirs=[cuda_include_dir]
    ),
]

setup(
    name="grand",
    version=__version__,
    author="Adam L.",
    author_email="adamloeckle@gmail.com",
    url="https://github.com/adamloec/grand",
    description="Definitely a tensor framework.",
    long_description="",
    packages=["grand"],
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    cmdclass={"build_ext": build_ext},
    install_requires=[
        "numpy>=1.18.0",
        "pybind11",
    ],
    zip_safe=False,
    python_requires=">=3.7",
)