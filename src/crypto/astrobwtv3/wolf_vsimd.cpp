/**
 * wolf_vsimd.cpp - 32-way Vertical SIMD Wolf Compute
 * 
 * Processes 32 independent hashes in parallel using AVX2 vertical SIMD.
 */

#include <immintrin.h>
#include <cstdint>
#include <algorithm>
#include <cstring>
#include <vector>
#include "astrobwtv3.h"
#include "astroworker.h"
#include "fnv1a.h"
#include "xxhash64.h"
#include <highwayhash/sip_hash.h>
#include "simd_util.hpp"
#include "rc4_avx512.hpp"
#include "spsa_state.hpp"

#ifdef USE_DLUNA_RADIX_SA
  #include "dluna_radix_sa.h"
  #define SA_FUNCTION_LOCAL dluna_radix_sa::radix_sort_sa
#else
  #include "libsais.h"
  #define SA_FUNCTION_LOCAL(T, SA, n, bA, bB) libsais(T, SA, n, 0, nullptr)
#endif

extern void hashSHA256(SHA256_CTX &sha256, const byte *input, byte *digest, unsigned long inputSize);

namespace wolf_vsimd {

static const __m256i vec_97 = _mm256_set1_epi8(97);
static const __m256i vec_3 = _mm256_set1_epi8(3);
static const __m256i vec_ff = _mm256_set1_epi32(-1);

struct Batch32 {
    // Keep a 32-byte tail so the wolf loop can mirror bytes 0..31 at 256..287.
    alignas(32) __m256i chunks[288];
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
    
    // Templates for SPSA
    int templateIdx[32];
    templateMarker astroTemplate[32][256];
    
    // We don't need full workerData for each, just the buffers
    uint8_t sData[32][ASTRO_SCRATCH_SIZE];
    uint32_t data_len[32];
    int32_t sa[32][277*256+1]; // Full SA buffer
    
    SHA256_CTX sha256[32];
    ucstk::Salsa20 salsa20[32];
    uint8_t salsaInput[32][256];
    
    workerData* workers[32];
};

static inline __m256i gather_pos2(const Batch32& b) {
    alignas(32) uint8_t vals[32] = {0};
    for (int i = 0; i < 32; i++) {
        if (b.active[i]) {
            vals[i] = ((uint8_t*)&b.chunks[b.p2[i]])[i];
        }
    }
    return _mm256_load_si256((__m256i*)vals);
}

void wolfCompute_Batch32(Batch32& b) {
    while (b.active_mask != 0) {
        uint32_t executed_this_round = 0;

        // 1. Update tries and switcher
        for (int m = 0; m < 32; m++) {
            if (!b.active[m]) continue;
            
            uint8_t c_255 = ((uint8_t*)&b.chunks[255])[m];
            if (b.tries[m] >= 260 + 16 || (c_255 >= 0xf0 && b.tries[m] >= 260)) {
                b.active[m] = false;
                b.active_mask &= ~(1u << m);
                continue;
            }
            
            executed_this_round |= (1u << m);
            b.tries[m]++;
            uint32_t rs = b.prev_lhash[m] ^ b.lhash[m] ^ b.tries[m];
            b.op[m] = static_cast<uint8_t>(rs);
            uint8_t p1 = static_cast<uint8_t>(rs >> 8);
            uint8_t p2 = static_cast<uint8_t>(rs >> 16);
            if (p1 > p2) std::swap(p1, p2);
            if (p2 - p1 > 32) p2 = p1 + ((p2 - p1) & 0x1f);
            
            b.lp1[m] = std::min(b.lp1[m], p1);
            b.lp2[m] = std::max(b.lp2[m], p2);
            b.p1[m] = p1; b.p2[m] = p2;

            if (b.op[m] >= 254) {
                alignas(32) uint8_t horizontal_chunk[256];
                for (int i=0; i<256; i++) {
                    horizontal_chunk[i] = ((uint8_t*)&b.chunks[i])[m];
                }
#if USE_FAST_RC4
                rc4_avx512::fast_rc4_set_key_dual(b.fast_rc4_key[m], &b.key[m], 256, horizontal_chunk);
#elif USE_CRYPTOGAMS_RC4_DUAL
                b.cryptogams_rc4[m].set_key(horizontal_chunk, 256);
                RC4_set_key(&b.key[m], 256, horizontal_chunk);
#else
                RC4_set_key(&b.key[m], 256, horizontal_chunk);
#endif
            }
        }

        // 2. Precompute execution masks
        uint32_t range_mask[288] = {0};
        for (int m = 0; m < 32; m++) {
            if (b.active[m]) {
                for (int i = b.p1[m]; i < b.p2[m]; i++) {
                    range_mask[i] |= (1u << m);
                }
            }
        }

        // Replicate first 32 bytes to the end to simulate AstroBWTv3 wrap-around behavior
        // BUT ONLY for tries > 1, because for tries == 1, sData[256+] is zero.
        for (int i = 0; i < 32; i++) {
            for (int m = 0; m < 32; m++) {
                if (b.active[m]) {
                    ((uint8_t*)&b.chunks[256 + i])[m] = (b.tries[m] == 1) ? 0 : ((uint8_t*)&b.chunks[i])[m];
                }
            }
        }

        __m256i pos2_vals = gather_pos2(b);

        // 3. Dispatch opcodes (SIMD execution)
        for (int op = 0; op < 256; op++) {
            uint32_t op_mask = 0;
            for (int m = 0; m < 32; m++) {
                if (b.active[m] && b.op[m] == op) op_mask |= (1u << m);
            }
            if (op_mask == 0) continue;

            for (int i = 0; i < 288; i++) {
                uint32_t exec_mask = op_mask & range_mask[i];
                if (exec_mask == 0) continue;

                alignas(32) uint8_t byte_mask_arr[32];
                for(int m = 0; m < 32; m++) byte_mask_arr[m] = ((exec_mask >> m) & 1) ? 0xFF : 0x00;
                __m256i v_op_mask = _mm256_load_si256((__m256i*)byte_mask_arr);
                
                __m256i val = b.chunks[i];

                #include "wolf_vsimd_ops.hpp"
                
                b.chunks[i] = _mm256_blendv_epi8(b.chunks[i], val, v_op_mask);
            }
        }

        // Handle special op0 swap
        for (int m = 0; m < 32; m++) {
            if (b.active[m] && b.op[m] == 0) {
                if ((b.p2[m] - b.p1[m]) % 2 == 1) {
                    uint8_t t1 = ((uint8_t*)&b.chunks[b.p1[m]])[m];
                    uint8_t t2 = ((uint8_t*)&b.chunks[b.p2[m]])[m];
                    ((uint8_t*)&b.chunks[b.p1[m]])[m] = reverse8(t2);
                    ((uint8_t*)&b.chunks[b.p2[m]])[m] = reverse8(t1);
                }
            }
        }

        // 4. Update hashes and check termination
        for (int m = 0; m < 32; m++) {
            if (!b.active[m]) continue;
            
            int p1 = b.p1[m];
            int p2 = b.p2[m];
            int A = ((uint8_t*)&b.chunks[p1])[m] - ((uint8_t*)&b.chunks[p2])[m];
            A = (256 + (A % 256)) % 256;

            if (A <= 0x40) {
                alignas(32) uint8_t horizontal_chunk[256];
                for (int i=0; i<256; i++) horizontal_chunk[i] = ((uint8_t*)&b.chunks[i])[m];

                if (A < 0x30) {
                    if (A < 0x10) {
                        b.prev_lhash[m] += b.lhash[m]; 
                        b.lhash[m] = XXHash64::hash(horizontal_chunk, p2, 0);
                    }
                    if (A < 0x20) {
                        b.prev_lhash[m] += b.lhash[m]; 
                        b.lhash[m] = hash_64_fnv1a(horizontal_chunk, p2);
                    }
                    b.prev_lhash[m] += b.lhash[m];
                    HH_ALIGNAS(16) const highwayhash::HH_U64 sk[2] = {(uint64_t)b.tries[m], b.prev_lhash[m]};
                    b.lhash[m] = highwayhash::SipHash(sk, (char*)horizontal_chunk, p2);
                }

#if USE_FAST_RC4
                rc4_avx512::fast_rc4_dual(b.fast_rc4_key[m], &b.key[m], 256, horizontal_chunk, horizontal_chunk);
#elif USE_CRYPTOGAMS_RC4_DUAL
                b.cryptogams_rc4[m].apply_keystream_256(horizontal_chunk);
#else
                RC4(&b.key[m], 256, horizontal_chunk, horizontal_chunk);
#endif
                
                b.astroTemplate[m][b.templateIdx[m]] = templateMarker{(uint8_t)(b.cc[m] > 1 ? b.lp1[m] : 0), (uint8_t)(b.cc[m] > 1 ? b.lp2[m] : 255), 0, 0, (uint16_t)((b.fc[m] << 7) | b.cc[m])};
                b.templateIdx[m] += (b.tries[m] > 1);
                
                b.fc[m] = b.tries[m] - 1; b.lp1[m] = 255; b.lp2[m] = 0; b.cc[m] = 1;

                for (int i=0; i<256; i++) ((uint8_t*)&b.chunks[i])[m] = horizontal_chunk[i];
            } else {
                b.cc[m]++;
            }

            uint8_t c_p1 = ((uint8_t*)&b.chunks[p1])[m];
            uint8_t c_p2 = ((uint8_t*)&b.chunks[p2])[m];
            ((uint8_t*)&b.chunks[255])[m] ^= c_p1 ^ c_p2;

            uint8_t c_255 = ((uint8_t*)&b.chunks[255])[m];
            if (b.tries[m] > 260 + 16 || (c_255 >= 0xf0 && b.tries[m] > 260)) {
                b.active[m] = false;
                b.active_mask &= ~(1u << m);
            }
        }

        // Store history for SA phase
        for (int m = 0; m < 32; m++) {
            if (executed_this_round & (1u << m)) {
                uint8_t* dst = &b.sData[m][(b.tries[m] - 1) * 256];
                for (int i = 0; i < 256; i++) dst[i] = ((uint8_t*)&b.chunks[i])[m];
            }
        }
    }
}

static thread_local Batch32* tls_batch = nullptr;

void AstroBWTv3_vsimd(const uint8_t inputs[32][MINIBLOCK_SIZE], int inputLen, uint8_t outputs[32][32]) {
    if (!tls_batch) {
        tls_batch = static_cast<Batch32*>(_mm_malloc(sizeof(Batch32), 32));
        new (tls_batch) Batch32();
    }
    Batch32& b = *tls_batch;
    
    b.active_mask = 0;
    memset(b.active, 0, sizeof(b.active));
    memset(b.tries, 0, sizeof(b.tries));
    memset(b.templateIdx, 0, sizeof(b.templateIdx));
    memset(b.cc, 0, sizeof(b.cc));
    memset(b.fc, 0, sizeof(b.fc));
    // chunks and sData will be overwritten during prep phase

    // 1. Prep Phase
    for (int m = 0; m < 32; m++) {
        b.active[m] = true;
        b.active_mask |= (1u << m);
        b.tries[m] = 0;
        b.fc[m] = 0;
        b.cc[m] = 1;
        b.lp1[m] = 255;
        b.lp2[m] = 0;
        
        alignas(32) uint8_t scratch[384] = {0};
        hashSHA256(b.sha256[m], inputs[m], &scratch[320], inputLen);
        
        b.salsa20[m].setKey(&scratch[320]);
        b.salsa20[m].setIv(&scratch[256]);
        b.salsa20[m].processBytes(b.salsaInput[m], scratch, 256);
        
        RC4_set_key(&b.key[m], 256, scratch);
        RC4(&b.key[m], 256, scratch, scratch);
        
        b.lhash[m] = hash_64_fnv1a_256(scratch);
        b.prev_lhash[m] = b.lhash[m];
        
        memcpy(&b.sData[m][0], scratch, 256);
        
        for (int i=0; i<256; i++) {
            ((uint8_t*)&b.chunks[i])[m] = scratch[i];
        }
    }
    
    // 2. Wolf Compute
    wolfCompute_Batch32(b);

    // 3. Extract and SA Phase
    static thread_local workerData* dummyWorker = nullptr;
    if (!dummyWorker) {
        dummyWorker = static_cast<workerData*>(_mm_malloc(sizeof(workerData), 32));
        if (dummyWorker) new (dummyWorker) workerData();
    }
    if (!dummyWorker) return; // Should not happen
    
    for (int m = 0; m < 32; m++) {
        b.data_len[m] = (b.tries[m] - 4) * 256 + (((uint64_t)b.sData[m][(b.tries[m]-1)*256 + 253] << 8 | b.sData[m][(b.tries[m]-1)*256 + 254]) & 0x3ff);
        memset(&b.sData[m][b.data_len[m]], 0, 16);

        memset(dummyWorker, 0, sizeof(workerData)); // zero-init POD part
        dummyWorker->templateIdx = b.templateIdx[m];
        memcpy(dummyWorker->astroTemplate, b.astroTemplate[m], sizeof(templateMarker) * b.templateIdx[m]);

        if (spsa::SPSA_Integrated(b.sData[m], b.data_len[m], *dummyWorker, outputs[m])) {
            continue;
        }

        SA_FUNCTION_LOCAL(b.sData[m], b.sa[m], b.data_len[m], nullptr, nullptr);
        hashSHA256(b.sha256[m], reinterpret_cast<byte*>(b.sa[m]), outputs[m], b.data_len[m] * 4);
    }
}

} // namespace wolf_vsimd
