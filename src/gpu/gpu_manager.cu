#include "gpu/gpu_manager.hpp"
#include "gpu/eth_kernel.cuh"
#include "crypto/secp256k1.cuh"
#include "crypto/keccak256.cuh"
#include <stdexcept>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <thread>

namespace eth_cracker {

GPUManager::GPUManager() : isRunning_(false), totalKeysChecked_(0), lastCheckTime_(std::chrono::steady_clock::now()), lastKeysChecked_(0), lastRate_(0.0) {}

GPUManager::~GPUManager() {
    stopCracking();
}

bool GPUManager::initialize() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        std::cerr << "Failed to get device count: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found" << std::endl;
        return false;
    }
    
    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;
    
    // Initialize each device
    for (int i = 0; i < deviceCount; i++) {
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
        std::cerr << "Failed to set device " << deviceId << ": "
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    GPUDevice device;
    device.deviceId = deviceId;
    
    // Create two separate streams
    error = cudaStreamCreate(&device.computeStream);
    if (error != cudaSuccess) {
        std::cerr << "Failed to create compute stream: "
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }
    error = cudaStreamCreate(&device.copyStream);
    if (error != cudaSuccess) {
        std::cerr << "Failed to create copy stream: "
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
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
    
    // Calculate launch parameters
    dim3 blocks, threads;
    calculateLaunchParams(device, blocks, threads);
    
    // Allocate RNG states
    size_t numThreads = blocks.x * threads.x;
    error = cudaMalloc(&device.d_rngStates, numThreads * sizeof(curandState));
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate RNG states: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Initialize kernel (RNG) on computeStream, not the old device.stream
    if (!initializeKernel(&device, blocks, threads)) {
        std::cerr << "Failed to initialize kernel" << std::endl;
        return false;
    }
    
    std::cout << "Using " << blocks.x << " blocks with " << threads.x << " threads each" << std::endl;
    
    devices_.push_back(std::move(device));
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

bool GPUManager::startCracking(const std::string& targetPattern) {
    if (devices_.empty()) {
        std::cerr << "No devices initialized" << std::endl;
        return false;
    }
    isRunning_ = true;

    // For each device:
    for (auto& device : devices_) {
        cudaSetDevice(device.deviceId);

        // Allocate result / pattern / counters on device
        cudaMalloc(&device.d_result, sizeof(FoundMatch));
        cudaMalloc(&device.d_keysChecked, sizeof(uint64_t));
        cudaMalloc(&device.d_targetPattern, targetPattern.length() + 1);

        // Initialize them on, say, computeStream or copyStream, but do a single
        // synchronization on computeStream so the kernel can rely on them:
        cudaMemsetAsync(device.d_result, 0, sizeof(FoundMatch), device.computeStream);
        cudaMemsetAsync(device.d_keysChecked, 0, sizeof(uint64_t), device.computeStream);
        cudaMemcpyAsync(device.d_targetPattern, targetPattern.c_str(), 
                       targetPattern.length() + 1, cudaMemcpyHostToDevice, device.computeStream);
        cudaStreamSynchronize(device.computeStream); // Wait for init

        // Launch the infinite kernel on computeStream:
        dim3 blocks, threads;
        calculateLaunchParams(device, blocks, threads);
        if (!launchKernel(
                &device, blocks, threads, targetPattern,
                targetPattern.length() == 40,
                (FoundMatch*)device.d_result,
                (uint64_t*)device.d_keysChecked
        )) {
            std::cerr << "Failed to launch kernel on device " << device.deviceId << std::endl;
            stopCracking();
            return false;
        }
    }

    // Main loop:
    auto lastUpdate = std::chrono::steady_clock::now();
    uint64_t lastTotal = 0;

    while (isRunning_) {
        bool matchFound = false;
        GPUDevice* matchDevice = nullptr;

        // Check each device for "found" flag, but use copyStream for the memcpy
        for (auto& device : devices_) {
            bool foundOnHost = false;
            cudaSetDevice(device.deviceId);
            // Asynchronously copy the "found" flag from device.d_result
            cudaMemcpyAsync(
                &foundOnHost,
                (const void*)&((FoundMatch*)device.d_result)->found,
                sizeof(bool),
                cudaMemcpyDeviceToHost,
                device.copyStream
            );
            // Now synchronize copyStream only
            cudaStreamSynchronize(device.copyStream);

            if (foundOnHost) {
                matchFound = true;
                matchDevice = &device;
                break;
            }
        }

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastUpdate);
        if (elapsed.count() >= 1000) {
            // Read keysChecked from each device asynchronously on copyStream
            uint64_t currentTotal = 0;
            for (auto& device : devices_) {
                uint64_t deviceCountHost = 0;
                cudaSetDevice(device.deviceId);
                cudaMemcpyAsync(
                    &deviceCountHost,
                    (uint64_t*)device.d_keysChecked,
                    sizeof(uint64_t),
                    cudaMemcpyDeviceToHost,
                    device.copyStream
                );
                cudaStreamSynchronize(device.copyStream);
                currentTotal += deviceCountHost;
            }
            // Now you can calculate rate, probability, etc. without blocking the kernel.

            double rate = (currentTotal - lastTotal) * 1000.0 / elapsed.count();
            
            // Calculate probability - use string representation for better precision
            const std::string TOTAL_KEYSPACE = "115792089237316195423570985008687907852837564279074904382605163141518161494337";
            double probability = (currentTotal * 100.0) / std::stod(TOTAL_KEYSPACE);
            
            // Calculate estimated time remaining at current rate (in years)
            double remainingKeys = std::stod(TOTAL_KEYSPACE) - currentTotal;
            double yearsRemaining = remainingKeys / rate / 31536000.0; // seconds in a year
            
            // Clear line and print stats
            std::cout << "\r\033[K" // Clear line
                      << std::fixed << std::setprecision(2)
                      << "Rate: " << rate / 1000000.0 << "M keys/s | "
                      << "Total: " << currentTotal / 1000000.0 << "M keys | "
                      << std::scientific << std::setprecision(6)
                      << "Probability: " << probability << "% | "
                      << std::fixed << std::setprecision(2)
                      << "Est. time: " << yearsRemaining << " years" << std::flush;
            
            lastUpdate = now;
            lastTotal = currentTotal;
        }

        if (matchFound && matchDevice) {
            // Copy full result from matchDevice->d_result (again, on copyStream)
            FoundMatch hostMatch;
            cudaMemcpyAsync(
                &hostMatch,
                matchDevice->d_result,
                sizeof(FoundMatch),
                cudaMemcpyDeviceToHost,
                matchDevice->copyStream
            );
            cudaStreamSynchronize(matchDevice->copyStream);
            
            std::cout << "\nFound match on GPU " << matchDevice->deviceId << "!" << std::endl;
            std::cout << "Private key: ";
            for (int i = 0; i < PRIVKEY_LEN; i++) {
                std::cout << std::hex << std::setw(2) << std::setfill('0') 
                         << static_cast<int>(hostMatch.privateKey[i]);
            }
            std::cout << std::endl;
            
            std::cout << "Address: 0x";
            for (int i = 0; i < ETH_ADDR_LEN; i++) {
                std::cout << std::hex << std::setw(2) << std::setfill('0') 
                         << static_cast<int>(hostMatch.address[i]);
            }
            std::cout << std::dec << std::endl;
            
            stopCracking();
            return true;
        }

        // Small sleep to avoid busy-wait
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
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
        cudaStreamSynchronize(device.computeStream);
        cudaStreamSynchronize(device.copyStream);
        
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
            uint64_t currentCount = 0;
            error = cudaMemcpyAsync(&currentCount, device.d_keysChecked,
                                  sizeof(uint64_t), cudaMemcpyDeviceToHost, device.copyStream);
            if (error != cudaSuccess) {
                std::cerr << "Failed to copy counter from device " << device.deviceId << ": " << cudaGetErrorString(error) << std::endl;
                continue;
            }
            
            error = cudaStreamSynchronize(device.copyStream);
            if (error != cudaSuccess) {
                std::cerr << "Failed to synchronize device " << device.deviceId << ": " << cudaGetErrorString(error) << std::endl;
                continue;
            }
            
            deviceCounts.push_back(currentCount);
            currentTotal += currentCount;
            
            // Check for match
            FoundMatch hostResult = {0};
            error = cudaMemcpyAsync(&hostResult, device.d_result,
                                  sizeof(FoundMatch), cudaMemcpyDeviceToHost, device.copyStream);
            if (error == cudaSuccess) {
                error = cudaStreamSynchronize(device.copyStream);
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
    
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastCheckTime_);
    if (duration.count() == 0) return lastRate_; // Return last known rate if duration is too small
    
    // Calculate rate in keys per second
    double rate = static_cast<double>(currentTotal - lastKeysChecked_) / (duration.count() / 1000.0);
    
    // Update last values for next calculation
    lastKeysChecked_ = currentTotal;
    lastCheckTime_ = now;
    lastRate_ = rate; // Store current rate
    
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

bool GPUManager::launchKernel(GPUDevice* device, dim3 blocks, dim3 threads,
                            const std::string& targetPattern, bool isFullAddress,
                            FoundMatch* result, uint64_t* keysChecked) {
    cudaError_t error = cudaSetDevice(device->deviceId);
    if (error != cudaSuccess) {
        std::cerr << "Failed to set device " << device->deviceId << ": " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Launch kernel
    generateAndCheckAddresses<<<blocks, threads, 0, device->computeStream>>>(
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
    
    return true;
}

bool GPUManager::initializeKernel(GPUDevice* device, dim3 blocks, dim3 threads) {
    cudaError_t error = cudaSetDevice(device->deviceId);
    if (error != cudaSuccess) {
        std::cerr << "Failed to set device " << device->deviceId << ": " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Initialize RNG states with different seeds for each thread
    unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
    seed += device->deviceId * 1000000;  // Make sure each GPU gets different seeds
    
    initRNG<<<blocks, threads, 0, device->computeStream>>>(
        seed,
        static_cast<curandState*>(device->d_rngStates)
    );
    
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "Failed to initialize RNG states: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    error = cudaStreamSynchronize(device->computeStream);
    if (error != cudaSuccess) {
        std::cerr << "Failed to synchronize after RNG init: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    return true;
}

} // namespace eth_cracker 