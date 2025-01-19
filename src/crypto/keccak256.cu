#include "crypto/keccak256.cuh"

namespace eth_cracker {
namespace crypto {

// Keccak-f[1600] round constants
__constant__ uint64_t KECCAK_ROUND_CONSTANTS[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL,
    0x8000000080008000ULL, 0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008AULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

// Rotation offsets for rho step
__constant__ int KECCAK_RHO_OFFSETS[24] = {
     1,  3,  6, 10, 15, 21,
    28, 36, 45, 55,  2, 14,
    27, 41, 56,  8, 25, 43,
    62, 18, 39, 61, 20, 44
};

// Lookup table for pi step
__constant__ int KECCAK_PI_LOOKUP[24] = {
    10,  7, 11, 17, 18, 3,
     5, 16,  8, 21, 24, 4,
    15, 23, 19, 13, 12, 2,
    20, 14, 22,  9,  6, 1
};

__device__ uint64_t rotateLeft(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

__device__ void keccakF1600(KeccakState* state) {
    uint64_t B[25], C[5], D[5];
    
    for (int round = 0; round < KECCAK_ROUNDS; round++) {
        // Theta step
        for (int x = 0; x < 5; x++) {
            C[x] = state->state[x] ^ state->state[x + 5] ^ state->state[x + 10] ^
                   state->state[x + 15] ^ state->state[x + 20];
        }
        
        for (int x = 0; x < 5; x++) {
            D[x] = C[(x + 4) % 5] ^ rotateLeft(C[(x + 1) % 5], 1);
            for (int y = 0; y < 5; y++) {
                state->state[x + 5 * y] ^= D[x];
            }
        }
        
        // Rho and Pi steps
        uint64_t last = state->state[1];
        for (int i = 0; i < 24; i++) {
            int x = KECCAK_PI_LOOKUP[i];
            B[i] = rotateLeft(state->state[x], KECCAK_RHO_OFFSETS[i]);
        }
        for (int i = 0; i < 24; i++) {
            state->state[KECCAK_PI_LOOKUP[i]] = B[i];
        }
        
        // Chi step
        for (int y = 0; y < 5; y++) {
            for (int x = 0; x < 5; x++) {
                B[x] = state->state[x + 5 * y];
            }
            for (int x = 0; x < 5; x++) {
                state->state[x + 5 * y] = B[x] ^
                    ((~B[(x + 1) % 5]) & B[(x + 2) % 5]);
            }
        }
        
        // Iota step
        state->state[0] ^= KECCAK_ROUND_CONSTANTS[round];
    }
}

__device__ void keccakAbsorb(KeccakState* state, const uint8_t* data, size_t length) {
    size_t blockSize = KECCAK_RATE / 8;
    size_t offset = 0;
    
    while (length >= blockSize) {
        for (size_t i = 0; i < blockSize / 8; i++) {
            uint64_t lane = 0;
            for (int j = 0; j < 8; j++) {
                lane |= ((uint64_t)data[offset + i * 8 + j] << (8 * j));
            }
            state->state[i] ^= lane;
        }
        
        keccakF1600(state);
        offset += blockSize;
        length -= blockSize;
    }
    
    // Handle remaining data
    if (length > 0) {
        for (size_t i = 0; i < length; i++) {
            state->state[i / 8] ^= ((uint64_t)data[offset + i] << (8 * (i % 8)));
        }
    }
}

__device__ void keccakSqueeze(KeccakState* state, uint8_t* output, size_t length) {
    size_t blockSize = KECCAK_RATE / 8;
    size_t offset = 0;
    
    while (length >= blockSize) {
        for (size_t i = 0; i < blockSize / 8; i++) {
            uint64_t lane = state->state[i];
            for (int j = 0; j < 8; j++) {
                output[offset + i * 8 + j] = (lane >> (8 * j)) & 0xFF;
            }
        }
        
        keccakF1600(state);
        offset += blockSize;
        length -= blockSize;
    }
    
    // Handle remaining output
    if (length > 0) {
        for (size_t i = 0; i < length; i++) {
            output[offset + i] = (state->state[i / 8] >> (8 * (i % 8))) & 0xFF;
        }
    }
}

__device__ void keccak256(const uint8_t* input, size_t inputLength, uint8_t* output) {
    KeccakState state = {0};
    
    // Absorb input
    keccakAbsorb(&state, input, inputLength);
    
    // Pad with 0x01 || 0x00* || 0x80
    state.state[inputLength / 8] ^= ((uint64_t)0x01 << (8 * (inputLength % 8)));
    state.state[(KECCAK_RATE / 64) - 1] ^= (uint64_t)0x80 << 56;
    
    // Final permutation
    keccakF1600(&state);
    
    // Squeeze output
    for (int i = 0; i < 4; i++) {
        uint64_t lane = state.state[i];
        for (int j = 0; j < 8; j++) {
            output[i * 8 + j] = (lane >> (8 * j)) & 0xFF;
        }
    }
}

__device__ void keccak256PublicKey(const uint8_t* publicKey, uint8_t* output) {
    // For Ethereum addresses, we hash the 64-byte public key (without 0x04 prefix)
    keccak256(publicKey, 64, output);
}

void initializeKeccakConstants() {
    // Constants are already initialized in __constant__ memory
}

} // namespace crypto
} // namespace eth_cracker 