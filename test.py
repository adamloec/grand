import subprocess
import sys, platform

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

print(find_cuda_version())

print(platform.system())