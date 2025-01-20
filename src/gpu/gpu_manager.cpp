#include "gpu/gpu_manager.hpp"
#include "gpu/eth_kernel.cuh"
#include "crypto/secp256k1.cuh"
#include "crypto/keccak256.cuh"
#include <stdexcept>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>

namespace eth_cracker {

GPUManager::GPUManager() : isRunning_(false), totalKeysChecked_(0), lastCheckTime_(std::chrono::steady_clock::now()), lastKeysChecked_(0) {}

GPUManager::~GPUManager() {
    stopCracking();
}

bool GPUManager::initializeGPUs() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        std::cerr << "Failed to get CUDA device count: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found" << std::endl;
        return false;
    }
    
    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;
    
    // Initialize each device
    for (int i = 0; i < deviceCount; ++i) {
        if (!initializeDevice(i)) {
            std::cerr << "Failed to initialize device " << i << std::endl;
            return false;
        }
    }
    
    return true;
}

bool GPUManager::initializeDevice(int deviceId) {
    cudaError_t error = cudaSetDevice(deviceId);
    if (error != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    GPUDevice device;
    device.deviceId = deviceId;
    
    error = cudaGetDeviceProperties(&device.properties, deviceId);
    if (error != cudaSuccess) {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    std::cout << "Initializing device " << deviceId << ": " << device.properties.name << std::endl;
    
    size_t free, total;
    error = cudaMemGetInfo(&free, &total);
    if (error != cudaSuccess) {
        std::cerr << "Failed to get device memory info: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    device.totalMemory = total;
    device.freeMemory = free;

    // Create CUDA stream
    error = cudaStreamCreate(&device.stream);
    if (error != cudaSuccess) {
        std::cerr << "Failed to create CUDA stream: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Calculate launch parameters first to know memory requirements
    dim3 blocks, threads;
    calculateLaunchParams(device, blocks, threads);
    
    // Allocate device memory
    size_t rngStatesSize = blocks.x * threads.x * sizeof(curandState);
    error = cudaMalloc(&device.d_rngStates, rngStatesSize);
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate RNG states: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    error = cudaMalloc(&device.d_result, sizeof(FoundMatch));
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate result buffer: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    error = cudaMalloc(&device.d_keysChecked, sizeof(uint64_t));
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate keys checked counter: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Initialize device-specific constants and kernels
    try {
        crypto::initializeSecp256k1Constants();
        crypto::initializeKeccakConstants();
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize crypto constants: " << e.what() << std::endl;
        return false;
    }
    
    std::cout << "Using " << blocks.x << " blocks with " << threads.x << " threads each" << std::endl;
    
    // Add device to vector first so we can get a stable pointer
    devices_.emplace_back(std::move(device));
    
    // Initialize kernel with pointer to the device we just added
    if (!initializeKernel(&devices_.back(), blocks, threads)) {
        std::cerr << "Failed to initialize kernel for device " << deviceId << std::endl;
        devices_.pop_back();
        return false;
    }
    
    return true;
}

void GPUManager::calculateLaunchParams(const GPUDevice& device, dim3& blocks, dim3& threads) {
    threads.x = BLOCK_SIZE;
    threads.y = 1;
    threads.z = 1;
    
    // Calculate optimal number of blocks based on device capabilities
    int multiProcessorCount = device.properties.multiProcessorCount;
    
    // Use maximum number of blocks per SM that the device supports
    int maxBlocksPerSM = std::min(MAX_BLOCKS_PER_SM, device.properties.maxBlocksPerMultiProcessor);
    
    // Calculate total blocks based on SM count and blocks per SM
    blocks.x = multiProcessorCount * maxBlocksPerSM;
    blocks.y = 1;
    blocks.z = 1;
    
    std::cout << "Device " << device.deviceId << " configuration:" << std::endl;
    std::cout << "  SMs: " << multiProcessorCount << std::endl;
    std::cout << "  Blocks per SM: " << maxBlocksPerSM << std::endl;
    std::cout << "  Total blocks: " << blocks.x << std::endl;
    std::cout << "  Threads per block: " << threads.x << std::endl;
    std::cout << "  Total threads: " << blocks.x * threads.x << std::endl;
    std::cout << "  Keys per thread: " << KEYS_PER_THREAD << std::endl;
    std::cout << "  Total keys per iteration: " << blocks.x * threads.x * KEYS_PER_THREAD << std::endl;
}

bool GPUManager::startCracking(const std::string& targetPattern, bool isFullAddress) {
    if (isRunning_) {
        std::cerr << "Cracking process already running" << std::endl;
        return false;
    }
    
    if (devices_.empty()) {
        std::cerr << "No GPU devices initialized" << std::endl;
        return false;
    }
    
    isRunning_ = true;
    totalKeysChecked_ = 0;
    lastKeysChecked_ = 0;
    lastCheckTime_ = std::chrono::steady_clock::now();
    
    // Launch kernel on each GPU
    for (auto& device : devices_) {
        // Set device context
        cudaError_t error = cudaSetDevice(device.deviceId);
        if (error != cudaSuccess) {
            std::cerr << "Failed to set device " << device.deviceId << ": " << cudaGetErrorString(error) << std::endl;
            stopCracking();
            return false;
        }
        
        // Allocate and copy target pattern
        error = cudaMalloc(&device.d_targetPattern, targetPattern.length() + 1);
        if (error != cudaSuccess) {
            std::cerr << "Failed to allocate target pattern buffer: " << cudaGetErrorString(error) << std::endl;
            stopCracking();
            return false;
        }
        
        error = cudaMemcpyAsync(device.d_targetPattern, targetPattern.c_str(),
                              targetPattern.length() + 1, cudaMemcpyHostToDevice, device.stream);
        if (error != cudaSuccess) {
            std::cerr << "Failed to copy target pattern: " << cudaGetErrorString(error) << std::endl;
            stopCracking();
            return false;
        }
        
        // Initialize result and counter
        error = cudaMemsetAsync(device.d_result, 0, sizeof(FoundMatch), device.stream);
        if (error != cudaSuccess) {
            std::cerr << "Failed to initialize result buffer: " << cudaGetErrorString(error) << std::endl;
            stopCracking();
            return false;
        }
        
        error = cudaMemsetAsync(device.d_keysChecked, 0, sizeof(uint64_t), device.stream);
        if (error != cudaSuccess) {
            std::cerr << "Failed to initialize counter: " << cudaGetErrorString(error) << std::endl;
            stopCracking();
            return false;
        }
        
        // Ensure all initialization is complete
        error = cudaStreamSynchronize(device.stream);
        if (error != cudaSuccess) {
            std::cerr << "Failed to synchronize device " << device.deviceId << ": " << cudaGetErrorString(error) << std::endl;
            stopCracking();
            return false;
        }
        
        dim3 blocks, threads;
        calculateLaunchParams(device, blocks, threads);
        
        device.result = std::make_unique<FoundMatch>();
        device.keysChecked = std::make_unique<uint64_t>(0);
        
        // Launch kernel
        if (!launchKernel(&device, blocks, threads, targetPattern, 
                         isFullAddress, device.result.get(), device.keysChecked.get())) {
            std::cerr << "Failed to launch kernel on device " << device.deviceId << std::endl;
            stopCracking();
            return false;
        }
    }
    
    return true;
}

void GPUManager::stopCracking() {
    if (!isRunning_) {
        return;
    }
    
    isRunning_ = false;
    
    // Cleanup and synchronize all devices
    for (auto& device : devices_) {
        cudaSetDevice(device.deviceId);
        cudaStreamSynchronize(device.stream);
        
        // Free target pattern buffer
        if (device.d_targetPattern) {
            cudaFree(device.d_targetPattern);
            device.d_targetPattern = nullptr;
        }
    }
}

double GPUManager::getKeysPerSecond() {
    if (!isRunning_) {
        return 0.0;
    }

    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - lastCheckTime_);
    if (duration.count() == 0) return 0.0;
    
    // Get current counters from all devices
    std::vector<uint64_t> deviceCounts;
    uint64_t currentTotal = 0;
    bool matchFound = false;
    FoundMatch foundMatch = {0};
    int foundDeviceId = -1;
    
    for (auto& device : devices_) {
        if (device.keysChecked) {
            cudaError_t error = cudaSetDevice(device.deviceId);
            if (error != cudaSuccess) {
                std::cerr << "Failed to set device " << device.deviceId << ": " << cudaGetErrorString(error) << std::endl;
                continue;
            }
            
            // Get current count
            error = cudaMemcpyAsync(device.keysChecked.get(), device.d_keysChecked,
                                  sizeof(uint64_t), cudaMemcpyDeviceToHost, device.stream);
            if (error != cudaSuccess) {
                std::cerr << "Failed to copy counter from device " << device.deviceId << ": " << cudaGetErrorString(error) << std::endl;
                continue;
            }
            
            error = cudaStreamSynchronize(device.stream);
            if (error != cudaSuccess) {
                std::cerr << "Failed to synchronize device " << device.deviceId << ": " << cudaGetErrorString(error) << std::endl;
                continue;
            }
            
            deviceCounts.push_back(*device.keysChecked);
            currentTotal += *device.keysChecked;
            
            // Check for match
            FoundMatch hostResult = {0};
            error = cudaMemcpyAsync(&hostResult, device.d_result,
                                  sizeof(FoundMatch), cudaMemcpyDeviceToHost, device.stream);
            if (error == cudaSuccess) {
                error = cudaStreamSynchronize(device.stream);
                if (error == cudaSuccess && hostResult.found && !matchFound) {
                    matchFound = true;
                    foundMatch = hostResult;
                    foundDeviceId = device.deviceId;
                }
            }
        }
    }
    
    if (matchFound) {
        std::cout << "\nFound match on GPU " << foundDeviceId << "!" << std::endl;
        std::cout << "Private key: ";
        for (int i = 0; i < PRIVKEY_LEN; i++) {
            std::cout << std::hex << std::setw(2) << std::setfill('0') 
                     << static_cast<int>(foundMatch.privateKey[i]);
        }
        std::cout << std::endl;
        
        std::cout << "Address: 0x";
        for (int i = 0; i < ETH_ADDR_LEN; i++) {
            std::cout << std::hex << std::setw(2) << std::setfill('0') 
                     << static_cast<int>(foundMatch.address[i]);
        }
        std::cout << std::dec << std::endl;
        
        stopCracking();
        return 0.0;
    }
    
    double rate = static_cast<double>(currentTotal - lastKeysChecked_) / duration.count();
    
    // Update last values for next calculation
    lastKeysChecked_ = currentTotal;
    lastCheckTime_ = now;
    
    // Calculate total keyspace and probability
    const uint256_t TOTAL_KEYSPACE("115792089237316195423570985008687907852837564279074904382605163141518161494337");
    uint256_t keysChecked = currentTotal;
    double probability = (double)keysChecked / (double)TOTAL_KEYSPACE * 100.0;
    uint256_t keysRemaining = TOTAL_KEYSPACE - keysChecked;
    
    // Calculate estimated time remaining at current rate
    double yearsRemaining = (double)keysRemaining / rate / 31536000.0; // seconds in a year
    
    // Print statistics
    std::cout << "\033[2K\r";  // Clear line and return cursor to start
    
    // Per-device statistics
    for (size_t i = 0; i < devices_.size(); i++) {
        if (i < deviceCounts.size()) {
            std::cout << "GPU " << devices_[i].deviceId << ": " 
                     << std::fixed << std::setprecision(2) 
                     << static_cast<double>(deviceCounts[i]) / 1000000.0 << "M keys | ";
        }
    }
    
    // Overall statistics
    std::cout << std::fixed << std::setprecision(2)
              << "Rate: " << rate / 1000000.0 << "M keys/s | "
              << "Total: " << currentTotal / 1000000.0 << "M keys | "
              << "Probability: " << std::scientific << std::setprecision(6) << probability << "% | "
              << "Est. time: " << std::fixed << std::setprecision(2) << yearsRemaining << " years"
              << std::flush;
    
    return rate;
}

uint64_t GPUManager::getTotalKeysChecked() const {
    uint64_t total = 0;
    for (const auto& device : devices_) {
        if (device.keysChecked) {
            total += *device.keysChecked;
        }
    }
    return total;
}

} // namespace eth_cracker 