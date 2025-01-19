#include "gpu/eth_kernel.cuh"
#include "crypto/secp256k1.cuh"
#include "crypto/keccak256.cuh"
#include <curand_kernel.h>

namespace eth_cracker {

// Device memory pointers
namespace {
    curandState* d_rngStates = nullptr;
    char* d_targetPattern = nullptr;
    FoundMatch* d_result = nullptr;
    uint64_t* d_keysChecked = nullptr;
}

// Convert a byte to two hex characters
__device__ void byteToHex(uint8_t byte, char* out) {
    static const char hex[] = "0123456789abcdef";
    out[0] = hex[byte >> 4];
    out[1] = hex[byte & 0x0f];
}

__device__ void addressToHex(const uint8_t* address, char* hexString) {
    hexString[0] = '0';
    hexString[1] = 'x';
    
    for (int i = 0; i < ETH_ADDR_LEN; i++) {
        byteToHex(address[i], &hexString[2 + i * 2]);
    }
    hexString[42] = '\0';
}

__global__ void initRNG(unsigned int seed, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void generateAndCheckAddresses(
    curandState* states,
    const char* targetPattern,
    int patternLength,
    bool isFullAddress,
    FoundMatch* result,
    uint64_t* keysChecked
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = states[idx];
    
    // Generate private key
    uint8_t privateKey[PRIVKEY_LEN];
    for (int i = 0; i < PRIVKEY_LEN; i++) {
        privateKey[i] = curand(&localState) & 0xFF;
    }
    
    // Generate public key
    uint8_t publicKey[PUBKEY_LEN];
    crypto::privateToPublic(privateKey, publicKey);
    
    // Hash public key to get address
    uint8_t hash[32];
    crypto::keccak256PublicKey(publicKey + 1, hash);  // Skip the 0x04 prefix
    
    // Take last 20 bytes as Ethereum address
    uint8_t address[ETH_ADDR_LEN];
    for (int i = 0; i < ETH_ADDR_LEN; i++) {
        address[i] = hash[i + 12];
    }
    
    // Convert address to hex for pattern matching
    char hexAddress[43];  // 0x + 40 chars + null terminator
    addressToHex(address, hexAddress);
    
    // Check if address matches pattern
    bool found = false;
    if (isFullAddress) {
        found = true;
        for (int i = 0; i < 42 && found; i++) {
            if (hexAddress[i] != targetPattern[i]) {
                found = false;
            }
        }
    } else {
        // Check if pattern appears at start of address (after 0x)
        found = true;
        for (int i = 0; i < patternLength && found; i++) {
            if (hexAddress[i + 2] != targetPattern[i]) {
                found = false;
            }
        }
    }
    
    // If found, store result
    if (found) {
        result->found = true;
        for (int i = 0; i < PRIVKEY_LEN; i++) {
            result->privateKey[i] = privateKey[i];
        }
        for (int i = 0; i < ETH_ADDR_LEN; i++) {
            result->address[i] = address[i];
        }
    }
    
    // Update RNG state and key counter
    states[idx] = localState;
    atomicAdd((unsigned long long int*)keysChecked, 1ULL);
}

// Host-side initialization
bool initializeKernel(int deviceId, dim3 blocks, dim3 threads) {
    cudaError_t error;
    
    error = cudaSetDevice(deviceId);
    if (error != cudaSuccess) return false;
    
    // Allocate memory for RNG states
    size_t numThreads = blocks.x * threads.x;
    error = cudaMalloc(&d_rngStates, numThreads * sizeof(curandState));
    if (error != cudaSuccess) return false;
    
    // Allocate memory for target pattern (max 42 chars for full address)
    error = cudaMalloc(&d_targetPattern, 43 * sizeof(char));
    if (error != cudaSuccess) return false;
    
    // Allocate memory for result
    error = cudaMalloc(&d_result, sizeof(FoundMatch));
    if (error != cudaSuccess) return false;
    
    // Allocate memory for key counter
    error = cudaMalloc(&d_keysChecked, sizeof(uint64_t));
    if (error != cudaSuccess) return false;
    
    // Initialize RNG states
    unsigned int seed = static_cast<unsigned int>(time(nullptr));
    initRNG<<<blocks, threads>>>(seed, d_rngStates);
    
    error = cudaGetLastError();
    if (error != cudaSuccess) return false;
    
    return true;
}

// Host-side kernel launch
bool launchKernel(
    int deviceId,
    dim3 blocks,
    dim3 threads,
    const std::string& targetPattern,
    bool isFullAddress,
    FoundMatch* result,
    uint64_t* keysChecked
) {
    cudaError_t error = cudaSetDevice(deviceId);
    if (error != cudaSuccess) return false;
    
    // Copy target pattern to device
    error = cudaMemcpy(d_targetPattern, targetPattern.c_str(),
                      targetPattern.length() + 1, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) return false;
    
    // Reset result and counter
    FoundMatch hostResult = {0};
    error = cudaMemcpy(d_result, &hostResult, sizeof(FoundMatch),
                      cudaMemcpyHostToDevice);
    if (error != cudaSuccess) return false;
    
    uint64_t zero = 0;
    error = cudaMemcpy(d_keysChecked, &zero, sizeof(uint64_t),
                      cudaMemcpyHostToDevice);
    if (error != cudaSuccess) return false;
    
    // Launch kernel
    generateAndCheckAddresses<<<blocks, threads>>>(
        d_rngStates,
        d_targetPattern,
        targetPattern.length(),
        isFullAddress,
        d_result,
        d_keysChecked
    );
    
    error = cudaGetLastError();
    if (error != cudaSuccess) return false;
    
    return true;
}

} // namespace eth_cracker 