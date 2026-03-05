/**
 * AVX-512 16-Way Parallel RC4 Implementation
 *
 * Port of Tritonn's AVX-512 RC4 from Rust to C++.
 */

#include "rc4_avx512.hpp"
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

#endif

namespace rc4_avx512 {

// ============================================================================
// CPU Feature Detection
// ============================================================================

bool avx512_available() {
#if defined(__x86_64__) || defined(_M_X64)
#ifdef _MSC_VER
    int info[4];
    __cpuidex(info, 7, 0);
    // Check for AVX-512F (bit 16 of EBX)
    return (info[1] & (1 << 16)) != 0;
#else
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        // Check for AVX-512F (bit 16 of EBX)
        return (ebx & (1 << 16)) != 0;
    }
    return false;
#endif
#else
    return false;
#endif
}

// ============================================================================
// AVX-512 Implementation (x86-64 only)
// ============================================================================

#if defined(__x86_64__) || defined(_M_X64)

/**
 * AVX-512 optimized keystream application
 *
 * Note: This is a partial AVX-512 implementation. True 16-way parallel
 * RC4 with scatter/gather operations requires careful handling of the
 * S-box updates which have data dependencies.
 *
 * Current approach: Use AVX-512 for XOR operations and gather/scatter
 * where beneficial, fall back to scalar for S-box state updates.
 */
void apply_keystream_16_avx512(
    uint8_t s[16][256],
    uint8_t ii[16],
    uint8_t jj[16],
    uint8_t buffers[16][256]
) {
    // Check for AVX-512 at runtime
    if (!avx512_available()) {
        // Fallback to scalar
        for (int chunk = 0; chunk < 64; chunk++) {
            int base = chunk * 4;

            for (int offset = 0; offset < 4; offset++) {
                int byte_idx = base + offset;

                for (int stream = 0; stream < 16; stream++) {
                    ii[stream]++;
                    uint8_t ival = ii[stream];

                    uint8_t s_i = s[stream][ival];
                    jj[stream] = static_cast<uint8_t>(jj[stream] + s_i);
                    uint8_t jval = jj[stream];

                    uint8_t s_j = s[stream][jval];
                    s[stream][ival] = s_j;
                    s[stream][jval] = s_i;

                    uint8_t k_idx = static_cast<uint8_t>(s_i + s_j);
                    uint8_t k = s[stream][k_idx];

                    buffers[stream][byte_idx] ^= k;
                }
            }
        }
        return;
    }

#if defined(__AVX512F__)
    // AVX-512 optimized path
    // Process 4 bytes at a time across all 16 streams

    alignas(64) uint8_t keystream[16][4];  // 4 bytes per stream

    for (int chunk = 0; chunk < 64; chunk++) {
        int base = chunk * 4;

        // Generate 4 keystream bytes for each of 16 streams
        for (int stream = 0; stream < 16; stream++) {
            for (int b = 0; b < 4; b++) {
                ii[stream]++;
                uint8_t ival = ii[stream];

                uint8_t s_i = s[stream][ival];
                jj[stream] = static_cast<uint8_t>(jj[stream] + s_i);
                uint8_t jval = jj[stream];

                uint8_t s_j = s[stream][jval];
                s[stream][ival] = s_j;
                s[stream][jval] = s_i;

                keystream[stream][b] = s[stream][static_cast<uint8_t>(s_i + s_j)];
            }
        }

        // Use AVX-512 for XOR operation
        // Load 64 bytes (4 bytes x 16 streams) into ZMM register
        __m512i ks = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(keystream));

        // Load corresponding buffer bytes (need to gather from different locations)
        // For simplicity, we use a loop here - true gather would require specific layout
        alignas(64) uint8_t buf_bytes[64];
        for (int stream = 0; stream < 16; stream++) {
            for (int b = 0; b < 4; b++) {
                buf_bytes[stream * 4 + b] = buffers[stream][base + b];
            }
        }

        __m512i buf = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(buf_bytes));
        __m512i result = _mm512_xor_si512(buf, ks);
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(buf_bytes), result);

        // Store back
        for (int stream = 0; stream < 16; stream++) {
            for (int b = 0; b < 4; b++) {
                buffers[stream][base + b] = buf_bytes[stream * 4 + b];
            }
        }
    }
#else
    // Compiler doesn't support AVX-512 intrinsics
    // Fallback to scalar
    for (int chunk = 0; chunk < 64; chunk++) {
        int base = chunk * 4;

        for (int offset = 0; offset < 4; offset++) {
            int byte_idx = base + offset;

            for (int stream = 0; stream < 16; stream++) {
                ii[stream]++;
                uint8_t ival = ii[stream];

                uint8_t s_i = s[stream][ival];
                jj[stream] = static_cast<uint8_t>(jj[stream] + s_i);
                uint8_t jval = jj[stream];

                uint8_t s_j = s[stream][jval];
                s[stream][ival] = s_j;
                s[stream][jval] = s_i;

                uint8_t k_idx = static_cast<uint8_t>(s_i + s_j);
                uint8_t k = s[stream][k_idx];

                buffers[stream][byte_idx] ^= k;
            }
        }
    }
#endif
}

#endif  // x86_64

} // namespace rc4_avx512
