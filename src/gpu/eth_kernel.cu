#include "gpu/eth_kernel.cuh"
#include "crypto/secp256k1.cuh"
#include "crypto/keccak256.cuh"
#include "gpu/gpu_manager.hpp"
#include <curand_kernel.h>
#include <iostream>

namespace eth_cracker {

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
    uint64_t localKeysChecked = 0;
    
    // Each thread processes multiple keys
    for (int batch = 0; batch < KEYS_PER_THREAD && !result->found; batch++) {
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
        
        localKeysChecked++;
        
        // If found, store result atomically
        if (found) {
            // Use atomic operation on an int
            if (atomicCAS((int*)&result->found, 0, 1) == 0) {
                // Copy results only if we won the race
                for (int i = 0; i < PRIVKEY_LEN; i++) {
                    result->privateKey[i] = privateKey[i];
                }
                for (int i = 0; i < ETH_ADDR_LEN; i++) {
                    result->address[i] = address[i];
                }
                __threadfence();  // Ensure result is visible to host
            }
            break;  // Exit loop if found
        }
    }
    
    // Update RNG state and key counter
    states[idx] = localState;
    atomicAdd((unsigned long long int*)keysChecked, localKeysChecked);
}

// Host-side initialization
bool initializeKernel(GPUDevice* device, dim3 blocks, dim3 threads) {
    if (!device) {
        std::cerr << "Invalid device pointer" << std::endl;
        return false;
    }

    std::cout << "Initializing kernel for device " << device->deviceId << std::endl;
    
    cudaError_t error;
    
    error = cudaSetDevice(device->deviceId);
    if (error != cudaSuccess) {
        std::cerr << "Failed to set device: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Initialize RNG states
    std::cout << "Initializing RNG states" << std::endl;
    unsigned int seed = static_cast<unsigned int>(time(nullptr)) + device->deviceId;  // Different seed per device
    initRNG<<<blocks, threads, 0, device->stream>>>(seed, static_cast<curandState*>(device->d_rngStates));
    
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "Failed to initialize RNG: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    error = cudaStreamSynchronize(device->stream);
    if (error != cudaSuccess) {
        std::cerr << "Failed to synchronize after RNG init: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    std::cout << "Device initialized successfully" << std::endl;
    return true;
}

// Host-side kernel launch
bool launchKernel(
    GPUDevice* device,
    dim3 blocks,
    dim3 threads,
    const std::string& targetPattern,
    bool isFullAddress,
    FoundMatch* result,
    uint64_t* keysChecked
) {
    if (!device) {
        std::cerr << "Invalid device pointer" << std::endl;
        return false;
    }

    cudaError_t error = cudaSetDevice(device->deviceId);
    if (error != cudaSuccess) {
        std::cerr << "Failed to set device: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    std::cout << "Launching kernel on device " << device->deviceId << std::endl;
    std::cout << "Target pattern: '" << targetPattern << "' (length: " << targetPattern.length() << ")" << std::endl;
    
    // Launch kernel with proper error checking
    generateAndCheckAddresses<<<blocks, threads, 0, device->stream>>>(
        static_cast<curandState*>(device->d_rngStates),
        static_cast<const char*>(device->d_targetPattern),
        targetPattern.length(),
        isFullAddress,
        static_cast<FoundMatch*>(device->d_result),
        static_cast<uint64_t*>(device->d_keysChecked)
    );
    
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "Failed to launch kernel: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Don't wait for kernel completion - it will run asynchronously
    return true;
}

} // namespace eth_cracker 