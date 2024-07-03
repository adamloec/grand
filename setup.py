from glob import glob
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

import subprocess, os
import re
import platform

__version__ = "0.0.1"

sdk_path = subprocess.check_output(["xcrun", "--show-sdk-path"]).decode().strip()


ext_modules = [
    Pybind11Extension(
        "grand._gmetal",
        sorted(glob("grand/gmetal/**/*.cpp", recursive=True)),
        include_dirs=[f"{sdk_path}/System/Library/Frameworks/Metal.framework/Headers"],
        extra_compile_args=["-fobjc-arc", "-x", "objective-c++"],
        extra_link_args=["-framework", "Metal"],
        libraries=[],
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