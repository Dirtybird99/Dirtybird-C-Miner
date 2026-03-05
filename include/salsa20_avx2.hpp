/**
 * AVX2-optimized Salsa20 C++ Wrapper
 *
 * Drop-in replacement for ucstk::Salsa20 with SIMD optimization.
 * Uses AVX2 for 2-block parallel processing when available.
 */

#ifndef SALSA20_AVX2_HPP
#define SALSA20_AVX2_HPP

#include <cstdint>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
// On MSYS2/MinGW with Clang, there's a conflict between __cpuid macro in cpuid.h
// and __cpuid function in intrin.h. We avoid this by undefining before including.
#ifdef __clang__
#undef __cpuid
#undef __cpuidex
#endif
#include <immintrin.h>
// Use GCC-style CPUID for GCC, Clang, MinGW; use MSVC style only for pure MSVC
#if defined(_MSC_VER) && !defined(__GNUC__) && !defined(__clang__)
#include <intrin.h>
#define SALSA_USE_MSVC_CPUID 1
#else
// For GCC, Clang, MinGW - use built-in cpuid
#define SALSA_USE_BUILTIN_CPUID 1
#endif
#define HAVE_X86_SIMD_SALSA 1
#else
#define HAVE_X86_SIMD_SALSA 0
#endif

namespace salsa20_avx2 {

// =============================================================================
// CPU Feature Detection
// =============================================================================

inline bool has_avx2() {
    static int cached = -1;
    if (cached >= 0) return cached;

#if HAVE_X86_SIMD_SALSA
#if defined(SALSA_USE_MSVC_CPUID)
    int cpuInfo[4] = {0};
    __cpuid(cpuInfo, 0);
    if (cpuInfo[0] >= 7) {
        __cpuidex(cpuInfo, 7, 0);
        cached = (cpuInfo[1] & (1 << 5)) != 0;
    } else {
        cached = 0;
    }
#elif defined(SALSA_USE_BUILTIN_CPUID)
    // Use inline assembly for CPUID to avoid header conflicts
    unsigned int eax, ebx, ecx, edx;
    // Get max supported leaf
    __asm__ __volatile__("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(0));
    if (eax >= 7) {
        // Check for AVX2 in leaf 7, subleaf 0
        __asm__ __volatile__("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(7), "c"(0));
        cached = (ebx & (1 << 5)) != 0;
    } else {
        cached = 0;
    }
#else
    cached = 0;
#endif
#else
    cached = 0;
#endif
    return cached;
}

// =============================================================================
// Constants
// =============================================================================

static constexpr uint32_t SIGMA[4] = {
    0x61707865, 0x3320646e, 0x79622d32, 0x6b206574
};

// =============================================================================
// Helper Functions
// =============================================================================

inline uint32_t load32_le(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

inline void store32_le(uint8_t* p, uint32_t v) {
    p[0] = static_cast<uint8_t>(v);
    p[1] = static_cast<uint8_t>(v >> 8);
    p[2] = static_cast<uint8_t>(v >> 16);
    p[3] = static_cast<uint8_t>(v >> 24);
}

#define ROTL32(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

// =============================================================================
// Optimized Scalar Core (fully unrolled)
// =============================================================================

inline void salsa20_core_scalar(uint32_t* output, const uint32_t* input) {
    uint32_t x0 = input[0], x1 = input[1], x2 = input[2], x3 = input[3];
    uint32_t x4 = input[4], x5 = input[5], x6 = input[6], x7 = input[7];
    uint32_t x8 = input[8], x9 = input[9], x10 = input[10], x11 = input[11];
    uint32_t x12 = input[12], x13 = input[13], x14 = input[14], x15 = input[15];

    for (int i = 0; i < 10; i++) {
        // Column round
        x4  ^= ROTL32(x0  + x12, 7);  x8  ^= ROTL32(x4  + x0,  9);
        x12 ^= ROTL32(x8  + x4,  13); x0  ^= ROTL32(x12 + x8,  18);
        x9  ^= ROTL32(x5  + x1,  7);  x13 ^= ROTL32(x9  + x5,  9);
        x1  ^= ROTL32(x13 + x9,  13); x5  ^= ROTL32(x1  + x13, 18);
        x14 ^= ROTL32(x10 + x6,  7);  x2  ^= ROTL32(x14 + x10, 9);
        x6  ^= ROTL32(x2  + x14, 13); x10 ^= ROTL32(x6  + x2,  18);
        x3  ^= ROTL32(x15 + x11, 7);  x7  ^= ROTL32(x3  + x15, 9);
        x11 ^= ROTL32(x7  + x3,  13); x15 ^= ROTL32(x11 + x7,  18);

        // Row round
        x1  ^= ROTL32(x0  + x3,  7);  x2  ^= ROTL32(x1  + x0,  9);
        x3  ^= ROTL32(x2  + x1,  13); x0  ^= ROTL32(x3  + x2,  18);
        x6  ^= ROTL32(x5  + x4,  7);  x7  ^= ROTL32(x6  + x5,  9);
        x4  ^= ROTL32(x7  + x6,  13); x5  ^= ROTL32(x4  + x7,  18);
        x11 ^= ROTL32(x10 + x9,  7);  x8  ^= ROTL32(x11 + x10, 9);
        x9  ^= ROTL32(x8  + x11, 13); x10 ^= ROTL32(x9  + x8,  18);
        x12 ^= ROTL32(x15 + x14, 7);  x13 ^= ROTL32(x12 + x15, 9);
        x14 ^= ROTL32(x13 + x12, 13); x15 ^= ROTL32(x14 + x13, 18);
    }

    output[0]  = x0  + input[0];  output[1]  = x1  + input[1];
    output[2]  = x2  + input[2];  output[3]  = x3  + input[3];
    output[4]  = x4  + input[4];  output[5]  = x5  + input[5];
    output[6]  = x6  + input[6];  output[7]  = x7  + input[7];
    output[8]  = x8  + input[8];  output[9]  = x9  + input[9];
    output[10] = x10 + input[10]; output[11] = x11 + input[11];
    output[12] = x12 + input[12]; output[13] = x13 + input[13];
    output[14] = x14 + input[14]; output[15] = x15 + input[15];
}

// =============================================================================
// AVX2 SIMD Core (2-block parallel)
// =============================================================================

#if HAVE_X86_SIMD_SALSA

#define AVX2_ROTL(x, n) _mm256_or_si256(_mm256_slli_epi32(x, n), _mm256_srli_epi32(x, 32 - n))

#define QUARTERROUND_AVX2(a, b, c, d) do { \
    __m256i t; \
    t = _mm256_add_epi32(a, d); b = _mm256_xor_si256(b, AVX2_ROTL(t, 7)); \
    t = _mm256_add_epi32(b, a); c = _mm256_xor_si256(c, AVX2_ROTL(t, 9)); \
    t = _mm256_add_epi32(c, b); d = _mm256_xor_si256(d, AVX2_ROTL(t, 13)); \
    t = _mm256_add_epi32(d, c); a = _mm256_xor_si256(a, AVX2_ROTL(t, 18)); \
} while(0)

#ifdef __GNUC__
__attribute__((target("avx2")))
#endif
inline void salsa20_core_avx2_2blocks(uint32_t* out0, uint32_t* out1,
                                       const uint32_t* in0, const uint32_t* in1) {
    // Load two blocks: low 128 bits from block0, high 128 bits from block1
    __m256i row0 = _mm256_set_m128i(
        _mm_loadu_si128((__m128i*)(in1 + 0)),
        _mm_loadu_si128((__m128i*)(in0 + 0))
    );
    __m256i row1 = _mm256_set_m128i(
        _mm_loadu_si128((__m128i*)(in1 + 4)),
        _mm_loadu_si128((__m128i*)(in0 + 4))
    );
    __m256i row2 = _mm256_set_m128i(
        _mm_loadu_si128((__m128i*)(in1 + 8)),
        _mm_loadu_si128((__m128i*)(in0 + 8))
    );
    __m256i row3 = _mm256_set_m128i(
        _mm_loadu_si128((__m128i*)(in1 + 12)),
        _mm_loadu_si128((__m128i*)(in0 + 12))
    );

    __m256i orig0 = row0, orig1 = row1, orig2 = row2, orig3 = row3;

    for (int i = 0; i < 10; i++) {
        // Column round with shuffles
        row1 = _mm256_shuffle_epi32(row1, _MM_SHUFFLE(0, 3, 2, 1));
        row2 = _mm256_shuffle_epi32(row2, _MM_SHUFFLE(1, 0, 3, 2));
        row3 = _mm256_shuffle_epi32(row3, _MM_SHUFFLE(2, 1, 0, 3));

        QUARTERROUND_AVX2(row0, row1, row2, row3);

        row1 = _mm256_shuffle_epi32(row1, _MM_SHUFFLE(2, 1, 0, 3));
        row2 = _mm256_shuffle_epi32(row2, _MM_SHUFFLE(1, 0, 3, 2));
        row3 = _mm256_shuffle_epi32(row3, _MM_SHUFFLE(0, 3, 2, 1));

        // Row round with shuffles
        row1 = _mm256_shuffle_epi32(row1, _MM_SHUFFLE(0, 3, 2, 1));
        row2 = _mm256_shuffle_epi32(row2, _MM_SHUFFLE(1, 0, 3, 2));
        row3 = _mm256_shuffle_epi32(row3, _MM_SHUFFLE(2, 1, 0, 3));

        QUARTERROUND_AVX2(row0, row1, row2, row3);

        row1 = _mm256_shuffle_epi32(row1, _MM_SHUFFLE(2, 1, 0, 3));
        row2 = _mm256_shuffle_epi32(row2, _MM_SHUFFLE(1, 0, 3, 2));
        row3 = _mm256_shuffle_epi32(row3, _MM_SHUFFLE(0, 3, 2, 1));
    }

    row0 = _mm256_add_epi32(row0, orig0);
    row1 = _mm256_add_epi32(row1, orig1);
    row2 = _mm256_add_epi32(row2, orig2);
    row3 = _mm256_add_epi32(row3, orig3);

    // Store results
    _mm_storeu_si128((__m128i*)(out0 + 0), _mm256_castsi256_si128(row0));
    _mm_storeu_si128((__m128i*)(out0 + 4), _mm256_castsi256_si128(row1));
    _mm_storeu_si128((__m128i*)(out0 + 8), _mm256_castsi256_si128(row2));
    _mm_storeu_si128((__m128i*)(out0 + 12), _mm256_castsi256_si128(row3));

    _mm_storeu_si128((__m128i*)(out1 + 0), _mm256_extracti128_si256(row0, 1));
    _mm_storeu_si128((__m128i*)(out1 + 4), _mm256_extracti128_si256(row1, 1));
    _mm_storeu_si128((__m128i*)(out1 + 8), _mm256_extracti128_si256(row2, 1));
    _mm_storeu_si128((__m128i*)(out1 + 12), _mm256_extracti128_si256(row3, 1));
}

#undef AVX2_ROTL
#undef QUARTERROUND_AVX2

#endif // HAVE_X86_SIMD_SALSA

// =============================================================================
// Salsa20 Class - Drop-in replacement for ucstk::Salsa20
// =============================================================================

class Salsa20 {
public:
    enum : size_t {
        VECTOR_SIZE = 16,
        BLOCK_SIZE = 64,
        KEY_SIZE = 32,
        IV_SIZE = 8
    };

    Salsa20(const uint8_t* key = nullptr) {
        std::memset(state_, 0, sizeof(state_));
        if (key) setKey(key);
    }

    void setKey(const uint8_t* key) {
        if (!key) return;

        state_[0]  = SIGMA[0];
        state_[1]  = load32_le(&key[0]);
        state_[2]  = load32_le(&key[4]);
        state_[3]  = load32_le(&key[8]);
        state_[4]  = load32_le(&key[12]);
        state_[5]  = SIGMA[1];
        state_[6]  = 0;  // IV low
        state_[7]  = 0;  // IV high
        state_[8]  = 0;  // Counter low
        state_[9]  = 0;  // Counter high
        state_[10] = SIGMA[2];
        state_[11] = load32_le(&key[16]);
        state_[12] = load32_le(&key[20]);
        state_[13] = load32_le(&key[24]);
        state_[14] = load32_le(&key[28]);
        state_[15] = SIGMA[3];
    }

    void setIv(const uint8_t* iv) {
        if (iv) {
            state_[6] = load32_le(&iv[0]);
            state_[7] = load32_le(&iv[4]);
        } else {
            state_[6] = 0;
            state_[7] = 0;
        }
        state_[8] = 0;
        state_[9] = 0;
    }

    void generateKeyStream(uint8_t output[BLOCK_SIZE]) {
        uint32_t keystream[16];
        salsa20_core_scalar(keystream, state_);

        for (int i = 0; i < 16; i++) {
            store32_le(&output[i * 4], keystream[i]);
        }

        // Increment counter
        state_[8]++;
        if (state_[8] == 0) state_[9]++;
    }

    void processBytes(const uint8_t* input, uint8_t* output, size_t numBytes) {
        // Special case for 256 bytes (common in AstroBWT)
        if (numBytes == 256) {
            processBytes256(input, output);
            return;
        }

        uint8_t keyStream[BLOCK_SIZE];
        while (numBytes > 0) {
            generateKeyStream(keyStream);
            size_t chunk = (numBytes < BLOCK_SIZE) ? numBytes : BLOCK_SIZE;
            for (size_t i = 0; i < chunk; i++) {
                output[i] = input[i] ^ keyStream[i];
            }
            input += chunk;
            output += chunk;
            numBytes -= chunk;
        }
    }

private:
    uint32_t state_[16];

    /**
     * Optimized 256-byte processing using AVX2 when available
     */
    void processBytes256(const uint8_t* input, uint8_t* output) {
#if HAVE_X86_SIMD_SALSA
        if (has_avx2()) {
            processBytes256_avx2(input, output);
            return;
        }
#endif
        // Scalar fallback
        uint8_t keyStream[BLOCK_SIZE];
        for (int block = 0; block < 4; block++) {
            generateKeyStream(keyStream);
            for (int i = 0; i < BLOCK_SIZE; i++) {
                output[block * BLOCK_SIZE + i] =
                    input[block * BLOCK_SIZE + i] ^ keyStream[i];
            }
        }
    }

#if HAVE_X86_SIMD_SALSA
#ifdef __GNUC__
    __attribute__((target("avx2")))
#endif
    void processBytes256_avx2(const uint8_t* input, uint8_t* output) {
        // Process 4 blocks using AVX2 2-block parallel
        uint32_t state0[16], state1[16], state2[16], state3[16];
        uint32_t ks0[16], ks1[16], ks2[16], ks3[16];

        // Set up states for blocks 0-3
        std::memcpy(state0, state_, sizeof(state0));
        std::memcpy(state1, state_, sizeof(state1));
        std::memcpy(state2, state_, sizeof(state2));
        std::memcpy(state3, state_, sizeof(state3));

        state1[8] = state0[8] + 1;
        state1[9] = state0[9] + (state1[8] == 0 ? 1 : 0);

        state2[8] = state1[8] + 1;
        state2[9] = state1[9] + (state2[8] == 0 ? 1 : 0);

        state3[8] = state2[8] + 1;
        state3[9] = state2[9] + (state3[8] == 0 ? 1 : 0);

        // Process blocks 0,1 and 2,3 in parallel
        salsa20_core_avx2_2blocks(ks0, ks1, state0, state1);
        salsa20_core_avx2_2blocks(ks2, ks3, state2, state3);

        // XOR with input using AVX2
        const __m256i* in = (const __m256i*)input;
        __m256i* out = (__m256i*)output;

        // Block 0 (bytes 0-63)
        __m256i ks_vec = _mm256_loadu_si256((__m256i*)ks0);
        __m256i in_vec = _mm256_loadu_si256(in);
        _mm256_storeu_si256(out, _mm256_xor_si256(in_vec, ks_vec));

        ks_vec = _mm256_loadu_si256((__m256i*)((uint8_t*)ks0 + 32));
        in_vec = _mm256_loadu_si256(in + 1);
        _mm256_storeu_si256(out + 1, _mm256_xor_si256(in_vec, ks_vec));

        // Block 1 (bytes 64-127)
        ks_vec = _mm256_loadu_si256((__m256i*)ks1);
        in_vec = _mm256_loadu_si256(in + 2);
        _mm256_storeu_si256(out + 2, _mm256_xor_si256(in_vec, ks_vec));

        ks_vec = _mm256_loadu_si256((__m256i*)((uint8_t*)ks1 + 32));
        in_vec = _mm256_loadu_si256(in + 3);
        _mm256_storeu_si256(out + 3, _mm256_xor_si256(in_vec, ks_vec));

        // Block 2 (bytes 128-191)
        ks_vec = _mm256_loadu_si256((__m256i*)ks2);
        in_vec = _mm256_loadu_si256(in + 4);
        _mm256_storeu_si256(out + 4, _mm256_xor_si256(in_vec, ks_vec));

        ks_vec = _mm256_loadu_si256((__m256i*)((uint8_t*)ks2 + 32));
        in_vec = _mm256_loadu_si256(in + 5);
        _mm256_storeu_si256(out + 5, _mm256_xor_si256(in_vec, ks_vec));

        // Block 3 (bytes 192-255)
        ks_vec = _mm256_loadu_si256((__m256i*)ks3);
        in_vec = _mm256_loadu_si256(in + 6);
        _mm256_storeu_si256(out + 6, _mm256_xor_si256(in_vec, ks_vec));

        ks_vec = _mm256_loadu_si256((__m256i*)((uint8_t*)ks3 + 32));
        in_vec = _mm256_loadu_si256(in + 7);
        _mm256_storeu_si256(out + 7, _mm256_xor_si256(in_vec, ks_vec));

        // Update counter
        state_[8] = state3[8] + 1;
        state_[9] = state3[9] + (state_[8] == 0 ? 1 : 0);
    }
#endif
};

#undef ROTL32

} // namespace salsa20_avx2

#endif // SALSA20_AVX2_HPP
