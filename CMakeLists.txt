cmake_minimum_required(VERSION 3.18)
project(cuda-eth-cracker CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Set CUDA architectures based on your GPU
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)  # Adjust these based on your GPU architecture

# Add source files
set(SOURCES
    src/main.cpp
    src/gpu/eth_kernel.cu
    src/gpu/gpu_manager.cpp
    src/crypto/secp256k1.cu
    src/crypto/keccak256.cu
)

# Create executable
add_executable(eth_cracker ${SOURCES})

# Set CUDA specific flags
set_target_properties(eth_cracker PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Include directories
target_include_directories(eth_cracker PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link libraries
target_link_libraries(eth_cracker PRIVATE
    cuda
    cudart
) 