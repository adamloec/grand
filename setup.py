from glob import glob
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

import subprocess
import os, sys
import re
import platform

__version__ = "0.0.1"

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

if platform.system() == "win32":
    pass

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