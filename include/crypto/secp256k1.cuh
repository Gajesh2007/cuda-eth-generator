#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace eth_cracker {
namespace crypto {

// Field element for secp256k1 curve
struct FieldElement {
    uint32_t words[8];  // 256-bit number stored as 8 32-bit words
};

// Point on secp256k1 curve
struct CurvePoint {
    FieldElement x;
    FieldElement y;
    FieldElement z;  // For projective coordinates
    bool infinity;
};

// Constants for secp256k1 (declared here, defined in .cu file)
extern __constant__ FieldElement SECP256K1_P;  // Prime field modulus
extern __constant__ FieldElement SECP256K1_A;  // Curve parameter a
extern __constant__ FieldElement SECP256K1_B;  // Curve parameter b
extern __constant__ CurvePoint SECP256K1_G;    // Generator point

// Device functions for field arithmetic
__device__ void fieldAdd(FieldElement* result, const FieldElement* a, const FieldElement* b);
__device__ void fieldSub(FieldElement* result, const FieldElement* a, const FieldElement* b);
__device__ void fieldMul(FieldElement* result, const FieldElement* a, const FieldElement* b);
__device__ void fieldInv(FieldElement* result, const FieldElement* a);

// Montgomery form conversions
__device__ void toMontgomery(FieldElement* result, const FieldElement* a);
__device__ void fromMontgomery(FieldElement* result, const FieldElement* a);

// Point arithmetic on the curve
__device__ void pointDouble(CurvePoint* result, const CurvePoint* p);
__device__ void pointAdd(CurvePoint* result, const CurvePoint* p, const CurvePoint* q);
__device__ void pointMultiply(CurvePoint* result, const FieldElement* scalar, const CurvePoint* p);

// Convert private key to public key
__device__ void privateToPublic(
    const uint8_t* privateKey,
    uint8_t* publicKey  // 65 bytes: 0x04 || x || y
);

// Host functions for initialization
void initializeSecp256k1Constants();

} // namespace crypto
} // namespace eth_cracker 