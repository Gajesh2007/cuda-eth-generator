#!/bin/bash

# Exit on error
set -e

echo "Installing required packages..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git

# Check if CUDA is installed
if ! command -v nvcc &> /dev/null; then
    echo "CUDA not found. Please install CUDA Toolkit 11.0 or higher."
    echo "You can download it from: https://developer.nvidia.com/cuda-downloads"
    echo "Or install via package manager:"
    echo "sudo apt-get install nvidia-cuda-toolkit"
    exit 1
fi

# Check CUDA version
CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
REQUIRED_VERSION="11.0"

if (( $(echo "$CUDA_VERSION < $REQUIRED_VERSION" | bc -l) )); then
    echo "CUDA version $CUDA_VERSION is installed, but version $REQUIRED_VERSION or higher is required."
    exit 1
fi

echo "Environment check passed. You can now build the project:"
echo "mkdir build"
echo "cd build"
echo "cmake .."
echo "make -j$(nproc)" 