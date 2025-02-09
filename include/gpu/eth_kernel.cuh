#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <string>
#include <stdint.h>

namespace eth_cracker {

// Ethereum address is 20 bytes
constexpr int ETH_ADDR_LEN = 20;
// Private key is 32 bytes
constexpr int PRIVKEY_LEN = 32;
// Uncompressed public key is 65 bytes (0x04 + 32 bytes X + 32 bytes Y)
constexpr int PUBKEY_LEN = 65;

// GPU kernel configuration
constexpr int BLOCK_SIZE = 256;  // Threads per block
constexpr int KEYS_PER_THREAD = 256;  // Each thread processes this many keys before updating counter
constexpr int MAX_BLOCKS_PER_SM = 16;  // Maximum blocks per streaming multiprocessor

// Structure to hold a found match
struct FoundMatch {
    uint8_t privateKey[PRIVKEY_LEN];
    uint8_t address[ETH_ADDR_LEN];
    volatile int found;  // Using int instead of bool for atomic operations
};

// CUDA kernel declarations
extern "C" {
    // Initialize GPU RNG state
    __global__ void initRNG(unsigned int seed, curandState* states);
    
    // Main kernel for generating and checking addresses
    __global__ void generateAndCheckAddresses(
        curandState* states,
        const char* targetPattern,
        int patternLength,
        bool isFullAddress,
        FoundMatch* result,
        uint64_t* keysChecked
    );
    
    // Helper function to convert address to hex string for pattern matching
    __device__ void addressToHex(const uint8_t* address, char* hexString);
}

// Forward declarations
struct GPUDevice;

// Host-side function declarations
bool initializeKernel(GPUDevice* device, dim3 blocks, dim3 threads);
bool launchKernel(
    GPUDevice* device,
    dim3 blocks,
    dim3 threads,
    const std::string& targetPattern,
    bool isFullAddress,
    FoundMatch* result,
    uint64_t* keysChecked
);

} // namespace eth_cracker 