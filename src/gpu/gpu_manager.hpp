#pragma once

#include <vector>
#include <memory>
#include <string>
#include <cuda_runtime.h>
#include <chrono>
#include "gpu/eth_kernel.cuh"

namespace eth_cracker {

struct GPUDevice {
    int deviceId;
    cudaDeviceProp properties;
    size_t totalMemory;
    size_t freeMemory;
    std::unique_ptr<FoundMatch> result;
    std::unique_ptr<uint64_t> keysChecked;
    
    // Device memory pointers
    void* d_rngStates;
    void* d_targetPattern;
    void* d_result;
    void* d_keysChecked;
    
    // Device context
    cudaStream_t stream;

    GPUDevice() : deviceId(-1), d_rngStates(nullptr), d_targetPattern(nullptr), d_result(nullptr), d_keysChecked(nullptr), stream(nullptr) {}
    
    ~GPUDevice() {
        if (deviceId >= 0) {
            cudaSetDevice(deviceId);  // Set device context before cleanup
            if (d_rngStates) cudaFree(d_rngStates);
            if (d_targetPattern) cudaFree(d_targetPattern);
            if (d_result) cudaFree(d_result);
            if (d_keysChecked) cudaFree(d_keysChecked);
            if (stream) cudaStreamDestroy(stream);
        }
    }
    
    // Delete copy constructor and assignment
    GPUDevice(const GPUDevice&) = delete;
    GPUDevice& operator=(const GPUDevice&) = delete;
    
    // Add move constructor and assignment
    GPUDevice(GPUDevice&& other) noexcept 
        : deviceId(other.deviceId)
        , properties(other.properties)
        , totalMemory(other.totalMemory)
        , freeMemory(other.freeMemory)
        , result(std::move(other.result))
        , keysChecked(std::move(other.keysChecked))
        , d_rngStates(other.d_rngStates)
        , d_targetPattern(other.d_targetPattern)
        , d_result(other.d_result)
        , d_keysChecked(other.d_keysChecked)
        , stream(other.stream)
    {
        other.deviceId = -1;
        other.d_rngStates = nullptr;
        other.d_targetPattern = nullptr;
        other.d_result = nullptr;
        other.d_keysChecked = nullptr;
        other.stream = nullptr;
    }
    
    GPUDevice& operator=(GPUDevice&& other) noexcept {
        if (this != &other) {
            // Clean up current resources
            if (deviceId >= 0) {
                cudaSetDevice(deviceId);
                if (d_rngStates) cudaFree(d_rngStates);
                if (d_targetPattern) cudaFree(d_targetPattern);
                if (d_result) cudaFree(d_result);
                if (d_keysChecked) cudaFree(d_keysChecked);
                if (stream) cudaStreamDestroy(stream);
            }
            
            // Move resources from other
            deviceId = other.deviceId;
            properties = other.properties;
            totalMemory = other.totalMemory;
            freeMemory = other.freeMemory;
            result = std::move(other.result);
            keysChecked = std::move(other.keysChecked);
            d_rngStates = other.d_rngStates;
            d_targetPattern = other.d_targetPattern;
            d_result = other.d_result;
            d_keysChecked = other.d_keysChecked;
            stream = other.stream;
            
            // Clear other's pointers
            other.deviceId = -1;
            other.d_rngStates = nullptr;
            other.d_targetPattern = nullptr;
            other.d_result = nullptr;
            other.d_keysChecked = nullptr;
            other.stream = nullptr;
        }
        return *this;
    }
};

class GPUManager {
public:
    GPUManager();
    ~GPUManager();

    // Initialize all available GPUs
    bool initializeGPUs();
    
    // Start the cracking process
    bool startCracking(const std::string& targetPattern, bool isFullAddress);
    
    // Stop all GPU operations
    void stopCracking();
    
    // Get current statistics
    double getKeysPerSecond();
    uint64_t getTotalKeysChecked() const;

private:
    std::vector<GPUDevice> devices_;
    bool isRunning_;
    uint64_t totalKeysChecked_;
    mutable uint64_t lastKeysChecked_;
    mutable std::chrono::steady_clock::time_point lastCheckTime_;
    
    // GPU kernel configuration
    static constexpr int BLOCK_SIZE = 256;  // Threads per block
    static constexpr int KEYS_PER_THREAD = 1024;  // Each thread processes this many keys before updating counter
    static constexpr int MAX_BLOCKS_PER_SM = 32;  // Maximum blocks per streaming multiprocessor
    
    // Initialize a single GPU
    bool initializeDevice(int deviceId);
    
    // Calculate optimal kernel launch parameters for a device
    void calculateLaunchParams(const GPUDevice& device, dim3& blocks, dim3& threads);
};

} // namespace eth_cracker 