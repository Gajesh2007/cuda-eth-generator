#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <csignal>
#include "gpu/gpu_manager.hpp"

namespace {
    volatile sig_atomic_t g_running = 1;
    
    void signalHandler(int) {
        g_running = 0;
    }
}

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " <options>\n"
              << "Options:\n"
              << "  -p, --pattern <pattern>    Target pattern to search for (e.g., '0x1234')\n"
              << "  -f, --full-address         Search for exact address match\n"
              << "  -h, --help                 Show this help message\n";
}

int main(int argc, char* argv[]) {
    std::string targetPattern;
    bool isFullAddress = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "-p" || arg == "--pattern") {
            if (i + 1 < argc) {
                targetPattern = argv[++i];
                // Remove 0x prefix if present
                if (targetPattern.substr(0, 2) == "0x") {
                    targetPattern = targetPattern.substr(2);
                }
            } else {
                std::cerr << "Error: Pattern argument missing\n";
                return 1;
            }
        } else if (arg == "-f" || arg == "--full-address") {
            isFullAddress = true;
        }
    }
    
    if (targetPattern.empty()) {
        std::cerr << "Error: No target pattern specified\n";
        printUsage(argv[0]);
        return 1;
    }
    
    // Setup signal handler for graceful shutdown
    signal(SIGINT, signalHandler);
    
    try {
        eth_cracker::GPUManager manager;
        
        // Initialize GPUs
        if (!manager.initialize()) {
            std::cerr << "Failed to initialize GPUs" << std::endl;
            return 1;
        }
        
        std::cout << "Starting address search for pattern: " << targetPattern << "\n";
        // Start cracking process
        if (!manager.startCracking(targetPattern)) {
            std::cerr << "Failed to start cracking process" << std::endl;
            return 1;
        }
        
        // Wait for interrupt
        while (g_running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        std::cout << "\nStopping...\n";
        manager.stopCracking();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
} 