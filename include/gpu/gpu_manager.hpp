#pragma once

#include <vector>
#include <memory>
#include <string>
#include <cuda_runtime.h>

namespace eth_cracker {

struct GPUDevice {
    int deviceId;
    cudaDeviceProp properties;
    size_t totalMemory;
    size_t freeMemory;
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
    double getKeysPerSecond() const;
    uint64_t getTotalKeysChecked() const;

private:
    std::vector<GPUDevice> devices_;
    bool isRunning_;
    
    // GPU kernel configuration
    static constexpr int BLOCK_SIZE = 256;
    static constexpr int KEYS_PER_THREAD = 16;
    
    // Initialize a single GPU
    bool initializeDevice(int deviceId);
    
    // Calculate optimal kernel launch parameters for a device
    void calculateLaunchParams(const GPUDevice& device, dim3& blocks, dim3& threads);
};

} // namespace eth_cracker 