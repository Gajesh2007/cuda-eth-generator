#pragma once

#include <vector>
#include <memory>
#include <string>
#include <cuda_runtime.h>
#include <chrono>
#include "gpu/eth_kernel.cuh"

namespace eth_cracker {

// Simple 256-bit unsigned integer class
class uint256_t {
public:
    uint256_t() : data_{0} {}
    
    explicit uint256_t(const std::string& str) {
        // Parse hex or decimal string
        if (str.substr(0, 2) == "0x") {
            fromHexString(str.substr(2));
        } else {
            fromDecString(str);
        }
    }
    
    uint256_t(uint64_t value) : data_{0} {
        data_[0] = value;
    }
    
    uint256_t operator-(const uint256_t& other) const {
        uint256_t result;
        uint64_t borrow = 0;
        
        for (int i = 0; i < 4; i++) {
            uint64_t diff = data_[i] - other.data_[i] - borrow;
            result.data_[i] = diff;
            borrow = (diff > data_[i]) ? 1 : 0;
        }
        
        return result;
    }
    
    operator double() const {
        // Convert to double (with loss of precision)
        double result = 0.0;
        double factor = 1.0;
        
        for (int i = 0; i < 4; i++) {
            result += data_[i] * factor;
            factor *= 18446744073709551616.0; // 2^64
        }
        
        return result;
    }
    
private:
    void fromHexString(const std::string& hex) {
        // Parse hex string into 256-bit number
        for (int i = 0; i < 4; i++) {
            data_[i] = 0;
        }
        
        int digits = (hex.length() + 15) / 16;
        for (int i = 0; i < digits && i < 4; i++) {
            int start = std::max(0, (int)hex.length() - (i + 1) * 16);
            int len = std::min(16, (int)hex.length() - i * 16);
            std::string part = hex.substr(start, len);
            data_[i] = strtoull(part.c_str(), nullptr, 16);
        }
    }
    
    void fromDecString(const std::string& dec) {
        // Parse decimal string into 256-bit number
        uint256_t result;
        uint256_t ten(10);
        
        for (char c : dec) {
            if (c >= '0' && c <= '9') {
                // result = result * 10 + (c - '0')
                uint256_t digit(c - '0');
                uint256_t temp = result;
                result = result * ten + digit;
            }
        }
        
        *this = result;
    }
    
    uint256_t operator*(const uint256_t& other) const {
        uint256_t result;
        
        // Simple schoolbook multiplication
        for (int i = 0; i < 4; i++) {
            uint64_t carry = 0;
            for (int j = 0; j < 4 - i; j++) {
                __uint128_t prod = (__uint128_t)data_[i] * other.data_[j] + result.data_[i + j] + carry;
                result.data_[i + j] = (uint64_t)prod;
                carry = (uint64_t)(prod >> 64);
            }
        }
        
        return result;
    }
    
    uint256_t operator+(const uint256_t& other) const {
        uint256_t result;
        uint64_t carry = 0;
        
        for (int i = 0; i < 4; i++) {
            uint64_t sum = data_[i] + other.data_[i] + carry;
            result.data_[i] = sum;
            carry = (sum < data_[i]) ? 1 : 0;
        }
        
        return result;
    }
    
    uint64_t data_[4];  // Little-endian representation
};

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
    
    // Delete copy constructor and assignment
    GPUManager(const GPUManager&) = delete;
    GPUManager& operator=(const GPUManager&) = delete;
    
    bool initialize();  // Initialize all available GPUs
    bool startCracking(const std::string& targetPattern, bool isFullAddress = false);
    void stopCracking();
    double getKeysPerSecond();
    uint64_t getTotalKeysChecked() const;
    
private:
    bool isRunning_;
    std::vector<GPUDevice> devices_;
    uint64_t totalKeysChecked_;
    std::chrono::steady_clock::time_point lastCheckTime_;
    uint64_t lastKeysChecked_;
    double lastRate_;  // Store last calculated rate
    
    bool initializeDevice(int deviceId);  // Initialize a single GPU device
    bool initializeKernel(GPUDevice* device, dim3 blocks, dim3 threads);  // Initialize kernel and RNG states
    void calculateLaunchParams(const GPUDevice& device, dim3& blocks, dim3& threads);
    bool launchKernel(GPUDevice* device, dim3 blocks, dim3 threads, 
                     const std::string& targetPattern, bool isFullAddress,
                     FoundMatch* result, uint64_t* keysChecked);
};

} // namespace eth_cracker 