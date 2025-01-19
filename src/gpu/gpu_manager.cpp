#include "gpu/gpu_manager.hpp"
#include "gpu/eth_kernel.cuh"
#include "crypto/secp256k1.cuh"
#include "crypto/keccak256.cuh"
#include <stdexcept>
#include <chrono>

namespace eth_cracker {

GPUManager::GPUManager() : isRunning_(false) {}

GPUManager::~GPUManager() {
    stopCracking();
}

bool GPUManager::initializeGPUs() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to get CUDA device count: " + 
                               std::string(cudaGetErrorString(error)));
    }
    
    if (deviceCount == 0) {
        throw std::runtime_error("No CUDA devices found");
    }
    
    // Initialize each device
    for (int i = 0; i < deviceCount; ++i) {
        if (!initializeDevice(i)) {
            return false;
        }
    }
    
    return true;
}

bool GPUManager::initializeDevice(int deviceId) {
    cudaError_t error = cudaSetDevice(deviceId);
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to set CUDA device: " + 
                               std::string(cudaGetErrorString(error)));
    }
    
    GPUDevice device;
    device.deviceId = deviceId;
    
    error = cudaGetDeviceProperties(&device.properties, deviceId);
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to get device properties: " + 
                               std::string(cudaGetErrorString(error)));
    }
    
    size_t free, total;
    error = cudaMemGetInfo(&free, &total);
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to get device memory info: " + 
                               std::string(cudaGetErrorString(error)));
    }
    
    device.totalMemory = total;
    device.freeMemory = free;
    
    devices_.push_back(device);
    
    // Initialize device-specific constants and kernels
    crypto::initializeSecp256k1Constants();
    crypto::initializeKeccakConstants();
    
    dim3 blocks, threads;
    calculateLaunchParams(device, blocks, threads);
    if (!initializeKernel(deviceId, blocks, threads)) {
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
    int maxBlocksPerSM = device.properties.maxBlocksPerMultiProcessor;
    
    blocks.x = multiProcessorCount * maxBlocksPerSM;
    blocks.y = 1;
    blocks.z = 1;
}

bool GPUManager::startCracking(const std::string& targetPattern, bool isFullAddress) {
    if (isRunning_) {
        return false;
    }
    
    isRunning_ = true;
    
    // Launch kernel on each GPU
    for (const auto& device : devices_) {
        dim3 blocks, threads;
        calculateLaunchParams(device, blocks, threads);
        
        if (!launchKernel(device.deviceId, blocks, threads, targetPattern, 
                         isFullAddress, nullptr, nullptr)) {
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
    for (const auto& device : devices_) {
        cudaSetDevice(device.deviceId);
        cudaDeviceSynchronize();
    }
}

double GPUManager::getKeysPerSecond() const {
    // TODO: Implement actual performance monitoring
    return 0.0;
}

uint64_t GPUManager::getTotalKeysChecked() const {
    // TODO: Implement actual key counting
    return 0;
}

} // namespace eth_cracker 