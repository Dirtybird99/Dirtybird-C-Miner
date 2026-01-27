/**
 * Optimized RC4 Implementation for AstroBWT
 *
 * Key optimizations:
 * 1. SIMD-accelerated identity initialization (AVX2)
 * 2. 16x unrolled key scheduling with prefetching
 * 3. 16x unrolled PRGA with software pipelining
 * 4. Cache-aligned S-box (64-byte alignment)
 * 5. Branch-free swap operations
 *
 * Note: RC4 has inherent data dependencies that prevent true parallelization
 * of the KSA/PRGA, but we can optimize memory access patterns and reduce
 * loop overhead through unrolling.
 */

#pragma once

#include <cstdint>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace rc4_opt {

/**
 * Optimized RC4 state
 * - 64-byte aligned S-box for optimal cache line usage
 * - i, j stored separately for better register allocation
 */
class alignas(64) OptimizedRc4 {
public:
    alignas(64) uint8_t s[256];  // S-box - fits in 4 cache lines
    uint8_t i;
    uint8_t j;
    uint8_t pad[62];  // Pad to cache line boundary

    OptimizedRc4() : i(0), j(0) {
        identity_init();
    }

    /**
     * Initialize S-box to identity permutation using SIMD
     */
    void identity_init() {
#if defined(__AVX2__)
        // AVX2: Initialize 32 bytes at a time
        alignas(32) static const uint8_t iota[32] = {
            0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
        };
        __m256i base = _mm256_load_si256((__m256i*)iota);
        __m256i offset = _mm256_setzero_si256();
        __m256i inc = _mm256_set1_epi8(32);

        for (int k = 0; k < 256; k += 32) {
            __m256i val = _mm256_add_epi8(base, offset);
            _mm256_store_si256((__m256i*)&s[k], val);
            offset = _mm256_add_epi8(offset, inc);
        }
#elif defined(__SSE2__)
        // SSE2: Initialize 16 bytes at a time
        alignas(16) static const uint8_t iota[16] = {
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        };
        __m128i base = _mm_load_si128((__m128i*)iota);
        __m128i offset = _mm_setzero_si128();
        __m128i inc = _mm_set1_epi8(16);

        for (int k = 0; k < 256; k += 16) {
            __m128i val = _mm_add_epi8(base, offset);
            _mm_store_si128((__m128i*)&s[k], val);
            offset = _mm_add_epi8(offset, inc);
        }
#else
        // Scalar fallback
        for (int k = 0; k < 256; k++) {
            s[k] = static_cast<uint8_t>(k);
        }
#endif
        i = 0;
        j = 0;
    }

    /**
     * Optimized Key Scheduling Algorithm (KSA)
     * - 16x unrolled with prefetching
     * - Software pipelining for better ILP
     *
     * @param key Pointer to key data
     * @param key_len Length of key (typically 256 for AstroBWT)
     */
    void set_key(const uint8_t* key, size_t key_len) {
        identity_init();

        uint8_t jval = 0;

        // Prefetch key data
        __builtin_prefetch(key, 0, 3);
        __builtin_prefetch(key + 64, 0, 3);
        __builtin_prefetch(key + 128, 0, 3);
        __builtin_prefetch(key + 192, 0, 3);

        if (key_len == 256) {
            // Special case: 256-byte key (most common in AstroBWT)
            // Full unroll - no modulo needed

            // 16x unrolled KSA
            for (int k = 0; k < 256; k += 16) {
                // Prefetch next cache line of S-box
                __builtin_prefetch(&s[k + 64], 1, 3);

                // Round 0
                jval = static_cast<uint8_t>(jval + s[k+0] + key[k+0]);
                uint8_t tmp0 = s[k+0]; s[k+0] = s[jval]; s[jval] = tmp0;

                // Round 1
                jval = static_cast<uint8_t>(jval + s[k+1] + key[k+1]);
                uint8_t tmp1 = s[k+1]; s[k+1] = s[jval]; s[jval] = tmp1;

                // Round 2
                jval = static_cast<uint8_t>(jval + s[k+2] + key[k+2]);
                uint8_t tmp2 = s[k+2]; s[k+2] = s[jval]; s[jval] = tmp2;

                // Round 3
                jval = static_cast<uint8_t>(jval + s[k+3] + key[k+3]);
                uint8_t tmp3 = s[k+3]; s[k+3] = s[jval]; s[jval] = tmp3;

                // Round 4
                jval = static_cast<uint8_t>(jval + s[k+4] + key[k+4]);
                uint8_t tmp4 = s[k+4]; s[k+4] = s[jval]; s[jval] = tmp4;

                // Round 5
                jval = static_cast<uint8_t>(jval + s[k+5] + key[k+5]);
                uint8_t tmp5 = s[k+5]; s[k+5] = s[jval]; s[jval] = tmp5;

                // Round 6
                jval = static_cast<uint8_t>(jval + s[k+6] + key[k+6]);
                uint8_t tmp6 = s[k+6]; s[k+6] = s[jval]; s[jval] = tmp6;

                // Round 7
                jval = static_cast<uint8_t>(jval + s[k+7] + key[k+7]);
                uint8_t tmp7 = s[k+7]; s[k+7] = s[jval]; s[jval] = tmp7;

                // Round 8
                jval = static_cast<uint8_t>(jval + s[k+8] + key[k+8]);
                uint8_t tmp8 = s[k+8]; s[k+8] = s[jval]; s[jval] = tmp8;

                // Round 9
                jval = static_cast<uint8_t>(jval + s[k+9] + key[k+9]);
                uint8_t tmp9 = s[k+9]; s[k+9] = s[jval]; s[jval] = tmp9;

                // Round 10
                jval = static_cast<uint8_t>(jval + s[k+10] + key[k+10]);
                uint8_t tmp10 = s[k+10]; s[k+10] = s[jval]; s[jval] = tmp10;

                // Round 11
                jval = static_cast<uint8_t>(jval + s[k+11] + key[k+11]);
                uint8_t tmp11 = s[k+11]; s[k+11] = s[jval]; s[jval] = tmp11;

                // Round 12
                jval = static_cast<uint8_t>(jval + s[k+12] + key[k+12]);
                uint8_t tmp12 = s[k+12]; s[k+12] = s[jval]; s[jval] = tmp12;

                // Round 13
                jval = static_cast<uint8_t>(jval + s[k+13] + key[k+13]);
                uint8_t tmp13 = s[k+13]; s[k+13] = s[jval]; s[jval] = tmp13;

                // Round 14
                jval = static_cast<uint8_t>(jval + s[k+14] + key[k+14]);
                uint8_t tmp14 = s[k+14]; s[k+14] = s[jval]; s[jval] = tmp14;

                // Round 15
                jval = static_cast<uint8_t>(jval + s[k+15] + key[k+15]);
                uint8_t tmp15 = s[k+15]; s[k+15] = s[jval]; s[jval] = tmp15;
            }
        } else {
            // General case with modulo
            for (int k = 0; k < 256; k++) {
                jval = static_cast<uint8_t>(jval + s[k] + key[k % key_len]);
                uint8_t tmp = s[k];
                s[k] = s[jval];
                s[jval] = tmp;
            }
        }

        i = 0;
        j = 0;
    }

    /**
     * Optimized PRGA - Apply keystream XOR to 256-byte buffer
     * - 16x unrolled for reduced loop overhead
     * - Software pipelining for better instruction-level parallelism
     * - Immediate XOR after each round (avoids store buffer saturation)
     *
     * @param data Buffer to XOR with keystream (256 bytes, in-place)
     */
    void apply_keystream_256(uint8_t* data) {
        uint8_t ii = i;
        uint8_t jj = j;

        // Process 16 bytes at a time (16 rounds per iteration)
        for (int base = 0; base < 256; base += 16) {
            // Prefetch next cache line of output
            __builtin_prefetch(&data[base + 64], 1, 3);

            // Define a macro for one PRGA round with immediate XOR
            #define PRGA_ROUND(off) do { \
                ii++; \
                uint8_t s_i = s[ii]; \
                jj = static_cast<uint8_t>(jj + s_i); \
                uint8_t s_j = s[jj]; \
                s[ii] = s_j; \
                s[jj] = s_i; \
                data[base + off] ^= s[static_cast<uint8_t>(s_i + s_j)]; \
            } while(0)

            PRGA_ROUND(0);
            PRGA_ROUND(1);
            PRGA_ROUND(2);
            PRGA_ROUND(3);
            PRGA_ROUND(4);
            PRGA_ROUND(5);
            PRGA_ROUND(6);
            PRGA_ROUND(7);
            PRGA_ROUND(8);
            PRGA_ROUND(9);
            PRGA_ROUND(10);
            PRGA_ROUND(11);
            PRGA_ROUND(12);
            PRGA_ROUND(13);
            PRGA_ROUND(14);
            PRGA_ROUND(15);

            #undef PRGA_ROUND
        }

        i = ii;
        j = jj;
    }

    /**
     * Generic keystream application for any length
     *
     * @param data Buffer to XOR with keystream
     * @param len Length of buffer
     */
    void apply_keystream(uint8_t* data, size_t len) {
        if (len == 256) {
            apply_keystream_256(data);
            return;
        }

        uint8_t ii = i;
        uint8_t jj = j;

        // 8x unrolled for general case
        size_t chunks = len / 8;
        for (size_t c = 0; c < chunks; c++) {
            size_t base = c * 8;

            #define PRGA_ROUND_GEN(off) do { \
                ii++; \
                uint8_t s_i = s[ii]; \
                jj = static_cast<uint8_t>(jj + s_i); \
                uint8_t s_j = s[jj]; \
                s[ii] = s_j; \
                s[jj] = s_i; \
                data[base + off] ^= s[static_cast<uint8_t>(s_i + s_j)]; \
            } while(0)

            PRGA_ROUND_GEN(0);
            PRGA_ROUND_GEN(1);
            PRGA_ROUND_GEN(2);
            PRGA_ROUND_GEN(3);
            PRGA_ROUND_GEN(4);
            PRGA_ROUND_GEN(5);
            PRGA_ROUND_GEN(6);
            PRGA_ROUND_GEN(7);

            #undef PRGA_ROUND_GEN
        }

        // Handle remaining bytes
        size_t start = chunks * 8;
        for (size_t k = start; k < len; k++) {
            ii++;
            uint8_t s_i = s[ii];
            jj = static_cast<uint8_t>(jj + s_i);
            uint8_t s_j = s[jj];
            s[ii] = s_j;
            s[jj] = s_i;
            data[k] ^= s[static_cast<uint8_t>(s_i + s_j)];
        }

        i = ii;
        j = jj;
    }

    /**
     * Reset state for reuse
     */
    void reset() {
        identity_init();
    }
};

/**
 * Standalone function interface (compatible with OpenSSL RC4)
 */
inline void rc4_opt_set_key(OptimizedRc4& ctx, size_t len, const uint8_t* key) {
    ctx.set_key(key, len);
}

inline void rc4_opt_crypt(OptimizedRc4& ctx, size_t len, const uint8_t* in, uint8_t* out) {
    if (in != out) {
        memcpy(out, in, len);
    }
    ctx.apply_keystream(out, len);
}

// =============================================================================
// Multi-stream RC4 for batch processing (4-way parallel state)
// =============================================================================

/**
 * 4-way parallel RC4 state
 * Useful for processing multiple independent streams simultaneously
 * Note: This doesn't make individual KSA/PRGA faster, but allows
 * better CPU utilization when processing multiple hashes
 */
class alignas(64) Rc4x4 {
public:
    alignas(64) uint8_t s[4][256];  // 4 S-boxes
    uint8_t i[4];
    uint8_t j[4];

    Rc4x4() {
        reset();
    }

    void reset() {
        for (int stream = 0; stream < 4; stream++) {
            for (int k = 0; k < 256; k++) {
                s[stream][k] = static_cast<uint8_t>(k);
            }
            i[stream] = 0;
            j[stream] = 0;
        }
    }

    /**
     * Initialize all 4 streams with different keys
     */
    void set_keys(const uint8_t* key0, const uint8_t* key1,
                  const uint8_t* key2, const uint8_t* key3, size_t key_len) {
        const uint8_t* keys[4] = {key0, key1, key2, key3};

        // Reset S-boxes to identity
        for (int stream = 0; stream < 4; stream++) {
            for (int k = 0; k < 256; k++) {
                s[stream][k] = static_cast<uint8_t>(k);
            }
        }

        // KSA for all 4 streams (interleaved for better cache behavior)
        if (key_len == 256) {
            uint8_t jj[4] = {0, 0, 0, 0};

            for (int k = 0; k < 256; k++) {
                // Process all 4 streams for index k
                for (int stream = 0; stream < 4; stream++) {
                    jj[stream] = static_cast<uint8_t>(jj[stream] + s[stream][k] + keys[stream][k]);
                    uint8_t tmp = s[stream][k];
                    s[stream][k] = s[stream][jj[stream]];
                    s[stream][jj[stream]] = tmp;
                }
            }
        } else {
            // General case
            for (int stream = 0; stream < 4; stream++) {
                uint8_t jval = 0;
                for (int k = 0; k < 256; k++) {
                    jval = static_cast<uint8_t>(jval + s[stream][k] + keys[stream][k % key_len]);
                    uint8_t tmp = s[stream][k];
                    s[stream][k] = s[stream][jval];
                    s[stream][jval] = tmp;
                }
            }
        }

        memset(i, 0, sizeof(i));
        memset(j, 0, sizeof(j));
    }

    /**
     * Apply keystream to all 4 buffers (256 bytes each)
     * Interleaved processing for better cache utilization
     */
    void apply_keystream_256_x4(uint8_t* buf0, uint8_t* buf1,
                                 uint8_t* buf2, uint8_t* buf3) {
        uint8_t* bufs[4] = {buf0, buf1, buf2, buf3};
        uint8_t ii[4], jj[4];

        memcpy(ii, i, 4);
        memcpy(jj, j, 4);

        // Process 4 bytes from each stream per iteration
        for (int base = 0; base < 256; base += 4) {
            for (int off = 0; off < 4; off++) {
                int byte_idx = base + off;

                // Process all 4 streams for this byte index
                for (int stream = 0; stream < 4; stream++) {
                    ii[stream]++;
                    uint8_t idx_i = ii[stream];
                    uint8_t s_i = s[stream][idx_i];
                    jj[stream] = static_cast<uint8_t>(jj[stream] + s_i);
                    uint8_t idx_j = jj[stream];
                    uint8_t s_j = s[stream][idx_j];
                    s[stream][idx_i] = s_j;
                    s[stream][idx_j] = s_i;
                    bufs[stream][byte_idx] ^= s[stream][static_cast<uint8_t>(s_i + s_j)];
                }
            }
        }

        memcpy(i, ii, 4);
        memcpy(j, jj, 4);
    }
};

} // namespace rc4_opt
