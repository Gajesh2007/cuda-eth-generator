#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace eth_cracker {
namespace crypto {

// Keccak-256 state is 5x5 array of 64-bit words
struct KeccakState {
    uint64_t state[25];
};

// Constants
constexpr int KECCAK_ROUNDS = 24;
constexpr int KECCAK_RATE = 1088;  // Rate for Keccak-256 (1600 - 512)

// Device constants for Keccak round constants (declared here, defined in .cu file)
extern __constant__ uint64_t KECCAK_ROUND_CONSTANTS[24];

// Core Keccak-f[1600] permutation
__device__ void keccakF1600(KeccakState* state);

// Absorb data into the sponge
__device__ void keccakAbsorb(
    KeccakState* state,
    const uint8_t* data,
    size_t length
);

// Squeeze output from the sponge
__device__ void keccakSqueeze(
    KeccakState* state,
    uint8_t* output,
    size_t length
);

// Main Keccak-256 hash function
__device__ void keccak256(
    const uint8_t* input,
    size_t inputLength,
    uint8_t* output  // Must be 32 bytes
);

// Specialized version for Ethereum address generation
// Takes 64-byte public key (without 0x04 prefix) and produces 32-byte hash
__device__ void keccak256PublicKey(
    const uint8_t* publicKey,
    uint8_t* output
);

// Host initialization
void initializeKeccakConstants();

} // namespace crypto
} // namespace eth_cracker 