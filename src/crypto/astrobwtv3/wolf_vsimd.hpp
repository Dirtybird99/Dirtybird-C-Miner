#pragma once
#include <immintrin.h>
#include <cstdint>
#include <openssl/rc4.h>
#include <openssl/sha.h>
#include "astroworker.h"
#include "rc4_avx512.hpp"

namespace wolf_vsimd {

struct Batch32 {
    alignas(32) __m256i chunks[288]; // 256 + 32 padding for p2 > 255
    uint64_t lhash[32];
    uint64_t prev_lhash[32];
    uint16_t tries[32];
    uint8_t op[32];
    uint8_t p1[32], p2[32];
    bool active[32];
    uint32_t active_mask;
    
    // Per-miner state
    RC4_KEY key[32];
#if USE_FAST_RC4
    rc4_avx512::FastRc4 fast_rc4_key[32];
#elif USE_CRYPTOGAMS_RC4_DUAL
    rc4_cryptogams::CryptogamsRc4 cryptogams_rc4[32];
#endif
    uint8_t lp1[32], lp2[32], cc[32];
    int fc[32];
    
    // We don't need full workerData for each, just the buffers
    uint8_t sData[32][ASTRO_SCRATCH_SIZE];
    uint32_t data_len[32];
    int32_t sa[32][277*256+1]; // Full SA buffer
    
    SHA256_CTX sha256[32];
    ucstk::Salsa20 salsa20[32];
    uint8_t salsaInput[32][256];
    
    workerData* workers[32];
};

void wolfCompute_Batch32(Batch32& b);
void AstroBWTv3_vsimd(const uint8_t inputs[32][MINIBLOCK_SIZE], int inputLen, uint8_t outputs[32][32]);

} // namespace wolf_vsimd