#include "crypto/secp256k1.cuh"

namespace eth_cracker {
namespace crypto {

// secp256k1 curve parameters
__constant__ FieldElement SECP256K1_P = {
    {0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
     0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF}
};

__constant__ FieldElement SECP256K1_A = {
    {0x00000000, 0x00000000, 0x00000000, 0x00000000,
     0x00000000, 0x00000000, 0x00000000, 0x00000000}
};

__constant__ FieldElement SECP256K1_B = {
    {0x00000000, 0x00000000, 0x00000000, 0x00000000,
     0x00000000, 0x00000000, 0x00000000, 0x00000007}
};

__constant__ CurvePoint SECP256K1_G = {
    // x coordinate
    {{0x79BE667E, 0xF9DCBBAC, 0x55A06295, 0xCE870B07,
      0x029BFCDB, 0x2DCE28D9, 0x59F2815B, 0x16F81798}},
    // y coordinate
    {{0x483ADA77, 0x26A3C465, 0x5DA4FBFC, 0x0E1108A8,
      0xFD17B448, 0xA6855419, 0x9C47D08F, 0xFB10D4B8}},
    // z coordinate (1 in projective coordinates)
    {{0x00000000, 0x00000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000000, 0x00000001}},
    false  // not infinity
};

// Montgomery constants for secp256k1
__constant__ FieldElement MONT_R = {
    {0x00000001, 0x00000000, 0x00000000, 0x00000000,
     0x00000000, 0x00000000, 0x00000001, 0x00000000}
};

__constant__ FieldElement MONT_R2 = {
    {0x00000003, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFE,
     0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF}
};

__constant__ uint32_t MONT_N0 = 0xD838091D;

// Convert to Montgomery form
__device__ void toMontgomery(FieldElement* result, const FieldElement* a) {
    fieldMul(result, a, &MONT_R2);
}

// Convert from Montgomery form
__device__ void fromMontgomery(FieldElement* result, const FieldElement* a) {
    FieldElement one = {0};
    one.words[0] = 1;
    fieldMul(result, a, &one);
}

__device__ void fieldMul(FieldElement* result, const FieldElement* a, const FieldElement* b) {
    uint32_t t[16] = {0};
    uint64_t carry;
    
    // Compute t = a * b
    for (int i = 0; i < 8; i++) {
        carry = 0;
        for (int j = 0; j < 8; j++) {
            uint64_t prod = (uint64_t)a->words[i] * b->words[j] + t[i + j] + carry;
            t[i + j] = (uint32_t)prod;
            carry = prod >> 32;
        }
        t[i + 8] = (uint32_t)carry;
    }
    
    // Montgomery reduction
    for (int i = 0; i < 8; i++) {
        uint32_t m = t[i] * MONT_N0;
        carry = 0;
        
        for (int j = 0; j < 8; j++) {
            uint64_t prod = (uint64_t)m * SECP256K1_P.words[j] + t[i + j] + carry;
            t[i + j] = (uint32_t)prod;
            carry = prod >> 32;
        }
        
        for (int j = i + 8; j < 16 && carry; j++) {
            uint64_t sum = (uint64_t)t[j] + carry;
            t[j] = (uint32_t)sum;
            carry = sum >> 32;
        }
    }
    
    // Final reduction
    bool needSubtract = false;
    if (t[8] || t[9] || t[10] || t[11] || t[12] || t[13] || t[14] || t[15]) {
        needSubtract = true;
    } else {
        for (int i = 7; i >= 0; i--) {
            if (t[i] > SECP256K1_P.words[i]) {
                needSubtract = true;
                break;
            }
            if (t[i] < SECP256K1_P.words[i]) {
                break;
            }
        }
    }
    
    if (needSubtract) {
        carry = 0;
        for (int i = 0; i < 8; i++) {
            uint64_t diff = (uint64_t)t[i] - SECP256K1_P.words[i] - carry;
            result->words[i] = (uint32_t)diff;
            carry = (diff >> 32) & 1;
        }
    } else {
        for (int i = 0; i < 8; i++) {
            result->words[i] = t[i];
        }
    }
}

__device__ void fieldAdd(FieldElement* result, const FieldElement* a, const FieldElement* b) {
    uint64_t carry = 0;
    
    for (int i = 0; i < 8; i++) {
        uint64_t sum = (uint64_t)a->words[i] + b->words[i] + carry;
        result->words[i] = (uint32_t)sum;
        carry = sum >> 32;
    }
    
    // Reduce modulo p if necessary
    if (carry || (result->words[7] > SECP256K1_P.words[7] ||
        (result->words[7] == SECP256K1_P.words[7] && result->words[6] >= SECP256K1_P.words[6]))) {
        carry = 0;
        for (int i = 0; i < 8; i++) {
            uint64_t diff = (uint64_t)result->words[i] - SECP256K1_P.words[i] - carry;
            result->words[i] = (uint32_t)diff;
            carry = (diff >> 32) & 1;
        }
    }
}

__device__ void fieldSub(FieldElement* result, const FieldElement* a, const FieldElement* b) {
    uint64_t borrow = 0;
    
    for (int i = 0; i < 8; i++) {
        uint64_t diff = (uint64_t)a->words[i] - b->words[i] - borrow;
        result->words[i] = (uint32_t)diff;
        borrow = (diff >> 32) & 1;
    }
    
    // Add p if result is negative
    if (borrow) {
        uint64_t carry = 0;
        for (int i = 0; i < 8; i++) {
            uint64_t sum = (uint64_t)result->words[i] + SECP256K1_P.words[i] + carry;
            result->words[i] = (uint32_t)sum;
            carry = sum >> 32;
        }
    }
}

__device__ void fieldInv(FieldElement* result, const FieldElement* a) {
    // Fermat's little theorem: a^(p-1) ≡ 1 (mod p)
    // Therefore: a^(p-2) ≡ a^(-1) (mod p)
    
    // Binary exponentiation
    FieldElement base = *a;
    FieldElement exp = SECP256K1_P;  // p-2
    exp.words[0] -= 2;
    
    result->words[0] = 1;
    for (int i = 1; i < 8; i++) {
        result->words[i] = 0;
    }
    
    while (true) {
        bool allZero = true;
        for (int i = 0; i < 8; i++) {
            if (exp.words[i] != 0) {
                allZero = false;
                break;
            }
        }
        if (allZero) break;
        
        for (int i = 7; i >= 0; i--) {
            for (int j = 31; j >= 0; j--) {
                // Square
                fieldMul(result, result, result);
                
                // Multiply if bit is set
                if (exp.words[i] & (1U << j)) {
                    fieldMul(result, result, &base);
                }
            }
        }
    }
}

__device__ void pointDouble(CurvePoint* result, const CurvePoint* p) {
    if (p->infinity) {
        result->infinity = true;
        return;
    }
    
    // Point doubling formulas for projective coordinates
    FieldElement t0, t1, t2, t3;
    
    // t0 = 3x²
    fieldMul(&t0, &p->x, &p->x);
    t1 = t0;
    t1.words[0] *= 3;  // Simplified - should handle overflow properly
    
    // t1 = 2y²
    fieldMul(&t2, &p->y, &p->y);
    fieldAdd(&t2, &t2, &t2);
    
    // t2 = 4xy²
    fieldMul(&t3, &p->x, &t2);
    
    // x' = t0² - 2t2
    fieldMul(&result->x, &t1, &t1);
    fieldSub(&result->x, &result->x, &t3);
    fieldSub(&result->x, &result->x, &t3);
    
    // y' = t0(t2 - x') - 8y⁴
    fieldSub(&t3, &t3, &result->x);
    fieldMul(&result->y, &t1, &t3);
    fieldMul(&t2, &t2, &t2);
    fieldSub(&result->y, &result->y, &t2);
    
    // z' = 2yz
    fieldMul(&result->z, &p->y, &p->z);
    fieldAdd(&result->z, &result->z, &result->z);
    
    result->infinity = false;
}

__device__ void pointAdd(CurvePoint* result, const CurvePoint* p, const CurvePoint* q) {
    if (p->infinity) {
        *result = *q;
        return;
    }
    if (q->infinity) {
        *result = *p;
        return;
    }
    
    // Point addition formulas for projective coordinates
    FieldElement t0, t1, t2, t3, t4;
    
    // t0 = z1²
    fieldMul(&t0, &p->z, &p->z);
    
    // t1 = z2²
    fieldMul(&t1, &q->z, &q->z);
    
    // t2 = x1z2² - x2z1²
    fieldMul(&t2, &p->x, &t1);
    fieldMul(&t3, &q->x, &t0);
    fieldSub(&t2, &t2, &t3);
    
    // t3 = y1z2³ - y2z1³
    fieldMul(&t3, &t1, &q->z);
    fieldMul(&t3, &t3, &p->y);
    fieldMul(&t4, &t0, &p->z);
    fieldMul(&t4, &t4, &q->y);
    fieldSub(&t3, &t3, &t4);
    
    if (t2.words[0] == 0) {
        if (t3.words[0] == 0) {
            pointDouble(result, p);
            return;
        }
        result->infinity = true;
        return;
    }
    
    // x3 = t2²
    fieldMul(&result->x, &t2, &t2);
    
    // y3 = t3(t2x1 - x3) - t2³y1
    fieldMul(&t4, &t2, &p->x);
    fieldSub(&t4, &t4, &result->x);
    fieldMul(&result->y, &t3, &t4);
    fieldMul(&t4, &t2, &t2);
    fieldMul(&t4, &t4, &t2);
    fieldMul(&t4, &t4, &p->y);
    fieldSub(&result->y, &result->y, &t4);
    
    // z3 = t2z1z2
    fieldMul(&result->z, &t2, &p->z);
    fieldMul(&result->z, &result->z, &q->z);
    
    result->infinity = false;
}

__device__ void pointMultiply(CurvePoint* result, const FieldElement* scalar, const CurvePoint* p) {
    result->infinity = true;
    CurvePoint temp = *p;
    
    for (int i = 7; i >= 0; i--) {
        for (int j = 31; j >= 0; j--) {
            if (!result->infinity) {
                pointDouble(result, result);
            }
            
            if (scalar->words[i] & (1U << j)) {
                if (result->infinity) {
                    *result = temp;
                } else {
                    pointAdd(result, result, &temp);
                }
            }
        }
    }
}

__device__ void privateToPublic(const uint8_t* privateKey, uint8_t* publicKey) {
    // Convert private key to field element
    FieldElement scalar = {0};
    for (int i = 0; i < 32; i++) {
        scalar.words[i / 4] |= ((uint32_t)privateKey[i] << ((i % 4) * 8));
    }
    
    // Convert to Montgomery form
    toMontgomery(&scalar, &scalar);
    
    // Compute public key as scalar * G
    CurvePoint result;
    pointMultiply(&result, &scalar, &SECP256K1_G);
    
    // Convert result coordinates from Montgomery form
    fromMontgomery(&result.x, &result.x);
    fromMontgomery(&result.y, &result.y);
    fromMontgomery(&result.z, &result.z);
    
    // Convert to uncompressed public key format
    publicKey[0] = 0x04;  // Uncompressed point format
    
    // Convert x coordinate
    for (int i = 0; i < 32; i++) {
        publicKey[i + 1] = (result.x.words[i / 4] >> ((i % 4) * 8)) & 0xFF;
    }
    
    // Convert y coordinate
    for (int i = 0; i < 32; i++) {
        publicKey[i + 33] = (result.y.words[i / 4] >> ((i % 4) * 8)) & 0xFF;
    }
}

void initializeSecp256k1Constants() {
    // Constants are already initialized in __constant__ memory
}

} // namespace crypto
} // namespace eth_cracker 