/**
 * SHA-256 Override: SHA-NI hardware-accelerated replacement for OpenSSL SHA256
 *
 * This file provides SHA256_Init/Update/Final with the same ABI as OpenSSL,
 * but uses Intel SHA-NI instructions for ~3-5x faster hashing.
 *
 * The SPSA precompiled library calls SHA256_Init/Update/Final (C linkage).
 * By defining them here in the main binary, the linker resolves to these
 * implementations before reaching the OpenSSL static library.
 *
 * This eliminates OpenSSL's dispatch overhead (~80ns per hash) and uses
 * hardware SHA-NI directly.
 */

#include <cstdint>
#include <cstring>
#include <algorithm>
#include <atomic>
#include <chrono>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

// Match OpenSSL's SHA256_CTX layout exactly
// From openssl/sha.h:
//   unsigned int h[8], Nl, Nh;
//   union { unsigned int d[16]; unsigned char p[64]; } u;
//   unsigned int num, md_len;
#include <openssl/sha.h>
#include "sha256_override_telemetry.hpp"

extern bool g_spsa_sha_profile;
extern bool g_spsa_sha_pair;
extern bool g_spsa_sha_zeroize;

// SHA-256 initial hash values
static const uint32_t SHA256_H0[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
};

// SHA-256 round constants
alignas(16) static const uint32_t SHA256_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

static std::atomic<uint64_t> g_sha_init_calls{0};
static std::atomic<uint64_t> g_sha_update_calls{0};
static std::atomic<uint64_t> g_sha_final_calls{0};
static std::atomic<uint64_t> g_sha_update_bytes{0};
static std::atomic<uint64_t> g_sha_update_ns{0};
static std::atomic<uint64_t> g_sha_final_ns{0};
static std::atomic<uint64_t> g_sha_final_zeroize_ns{0};
static std::atomic<uint64_t> g_sha_update_len_le32_calls{0};
static std::atomic<uint64_t> g_sha_update_len_33_48_calls{0};
static std::atomic<uint64_t> g_sha_update_len_49_64_calls{0};
static std::atomic<uint64_t> g_sha_update_len_65_128_calls{0};
static std::atomic<uint64_t> g_sha_update_len_129_512_calls{0};
static std::atomic<uint64_t> g_sha_update_len_gt512_calls{0};
static std::atomic<uint64_t> g_sha_update_flush_blocks{0};
static std::atomic<uint64_t> g_sha_update_direct_blocks{0};
static std::atomic<uint64_t> g_sha_pair_attempt_calls{0};
static std::atomic<uint64_t> g_sha_pair_success_calls{0};
static std::atomic<uint64_t> g_sha_pair_fallback_calls{0};
static std::atomic<uint64_t> g_sha_pair_blocks{0};
static std::atomic<uint64_t> g_sha_final_blocks{0};

namespace {

struct alignas(64) Sha256PairSlot {
    std::atomic<uint32_t> phase{0};
    uint32_t* state_words{nullptr};
    const uint8_t* blocks{nullptr};
    size_t block_count{0};
};

enum Sha256PairPhase : uint32_t {
    SHA256_PAIR_IDLE = 0,
    SHA256_PAIR_RESERVED = 1,
    SHA256_PAIR_WAITING = 2,
    SHA256_PAIR_CLAIMED = 3,
    SHA256_PAIR_DONE = 4,
};

static Sha256PairSlot g_sha_pair_slot;
static constexpr size_t SHA256_PAIR_MIN_BLOCKS = 64;   // 4 KiB minimum direct stream
static constexpr uint32_t SHA256_PAIR_SPIN_LIMIT = 512;

}  // namespace

static inline uint64_t sha_now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

static inline void sha_record_update_len_bucket(size_t len) {
    if (len <= 32) {
        g_sha_update_len_le32_calls.fetch_add(1, std::memory_order_relaxed);
    } else if (len <= 48) {
        g_sha_update_len_33_48_calls.fetch_add(1, std::memory_order_relaxed);
    } else if (len <= 64) {
        g_sha_update_len_49_64_calls.fetch_add(1, std::memory_order_relaxed);
    } else if (len <= 128) {
        g_sha_update_len_65_128_calls.fetch_add(1, std::memory_order_relaxed);
    } else if (len <= 512) {
        g_sha_update_len_129_512_calls.fetch_add(1, std::memory_order_relaxed);
    } else {
        g_sha_update_len_gt512_calls.fetch_add(1, std::memory_order_relaxed);
    }
}

void resetSha256OverrideTelemetry() {
    g_sha_init_calls.store(0, std::memory_order_relaxed);
    g_sha_update_calls.store(0, std::memory_order_relaxed);
    g_sha_final_calls.store(0, std::memory_order_relaxed);
    g_sha_update_bytes.store(0, std::memory_order_relaxed);
    g_sha_update_ns.store(0, std::memory_order_relaxed);
    g_sha_final_ns.store(0, std::memory_order_relaxed);
    g_sha_final_zeroize_ns.store(0, std::memory_order_relaxed);
    g_sha_update_len_le32_calls.store(0, std::memory_order_relaxed);
    g_sha_update_len_33_48_calls.store(0, std::memory_order_relaxed);
    g_sha_update_len_49_64_calls.store(0, std::memory_order_relaxed);
    g_sha_update_len_65_128_calls.store(0, std::memory_order_relaxed);
    g_sha_update_len_129_512_calls.store(0, std::memory_order_relaxed);
    g_sha_update_len_gt512_calls.store(0, std::memory_order_relaxed);
    g_sha_update_flush_blocks.store(0, std::memory_order_relaxed);
    g_sha_update_direct_blocks.store(0, std::memory_order_relaxed);
    g_sha_pair_attempt_calls.store(0, std::memory_order_relaxed);
    g_sha_pair_success_calls.store(0, std::memory_order_relaxed);
    g_sha_pair_fallback_calls.store(0, std::memory_order_relaxed);
    g_sha_pair_blocks.store(0, std::memory_order_relaxed);
    g_sha_final_blocks.store(0, std::memory_order_relaxed);
}

Sha256OverrideTelemetrySnapshot getSha256OverrideTelemetrySnapshot() {
    Sha256OverrideTelemetrySnapshot snapshot{};
    snapshot.init_calls = g_sha_init_calls.load(std::memory_order_relaxed);
    snapshot.update_calls = g_sha_update_calls.load(std::memory_order_relaxed);
    snapshot.final_calls = g_sha_final_calls.load(std::memory_order_relaxed);
    snapshot.update_bytes = g_sha_update_bytes.load(std::memory_order_relaxed);
    snapshot.update_ns = g_sha_update_ns.load(std::memory_order_relaxed);
    snapshot.final_ns = g_sha_final_ns.load(std::memory_order_relaxed);
    snapshot.final_zeroize_ns = g_sha_final_zeroize_ns.load(std::memory_order_relaxed);
    snapshot.update_len_le32_calls = g_sha_update_len_le32_calls.load(std::memory_order_relaxed);
    snapshot.update_len_33_48_calls = g_sha_update_len_33_48_calls.load(std::memory_order_relaxed);
    snapshot.update_len_49_64_calls = g_sha_update_len_49_64_calls.load(std::memory_order_relaxed);
    snapshot.update_len_65_128_calls = g_sha_update_len_65_128_calls.load(std::memory_order_relaxed);
    snapshot.update_len_129_512_calls = g_sha_update_len_129_512_calls.load(std::memory_order_relaxed);
    snapshot.update_len_gt512_calls = g_sha_update_len_gt512_calls.load(std::memory_order_relaxed);
    snapshot.update_flush_blocks = g_sha_update_flush_blocks.load(std::memory_order_relaxed);
    snapshot.update_direct_blocks = g_sha_update_direct_blocks.load(std::memory_order_relaxed);
    snapshot.pair_attempt_calls = g_sha_pair_attempt_calls.load(std::memory_order_relaxed);
    snapshot.pair_success_calls = g_sha_pair_success_calls.load(std::memory_order_relaxed);
    snapshot.pair_fallback_calls = g_sha_pair_fallback_calls.load(std::memory_order_relaxed);
    snapshot.pair_blocks = g_sha_pair_blocks.load(std::memory_order_relaxed);
    snapshot.final_blocks = g_sha_final_blocks.load(std::memory_order_relaxed);
    return snapshot;
}

// ============================================================================
// SHA-NI block processing (copied from sha256_spsa.cpp, proven correct)
// ============================================================================

#if defined(__x86_64__) || defined(_M_X64)

static bool g_sha_ni_available = false;
static bool g_sha_ni_checked = false;

static bool check_sha_ni() {
#ifdef _MSC_VER
    int info[4];
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 29)) != 0;
#else
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return (ebx & (1 << 29)) != 0;
    }
    return false;
#endif
}

#if defined(__GNUC__) && !defined(_MSC_VER)
__attribute__((target("sha,sse4.1,ssse3")))
#endif
static inline void sha256_state_to_ni(const uint32_t state[8], __m128i& state0, __m128i& state1) {
    state0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(state));
    state1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(state + 4));

    __m128i tmp = _mm_shuffle_epi32(state0, 0xB1);
    __m128i tmp2 = _mm_shuffle_epi32(state1, 0x1B);
    state0 = _mm_alignr_epi8(tmp, tmp2, 8);
    state1 = _mm_blend_epi16(tmp2, tmp, 0xF0);
}

#if defined(__GNUC__) && !defined(_MSC_VER)
__attribute__((target("sha,sse4.1,ssse3")))
#endif
static inline void sha256_state_from_ni(uint32_t state[8], __m128i state0, __m128i state1) {
    __m128i tmp = _mm_shuffle_epi32(state0, 0x1B);
    state1 = _mm_shuffle_epi32(state1, 0xB1);
    state0 = _mm_blend_epi16(tmp, state1, 0xF0);
    state1 = _mm_alignr_epi8(state1, tmp, 8);

    _mm_storeu_si128(reinterpret_cast<__m128i*>(state), state0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(state + 4), state1);
}

#if defined(__GNUC__) && !defined(_MSC_VER)
__attribute__((target("sha,sse4.1,ssse3")))
#endif
static inline void sha256_block_ni_core(__m128i& state0, __m128i& state1, const uint8_t* block, __m128i shuf_mask) {
    __m128i abef_save = state0;
    __m128i cdgh_save = state1;
    const __m128i* k_ptr = reinterpret_cast<const __m128i*>(SHA256_K);

    __m128i msg0 = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(block)), shuf_mask);
    __m128i msg1 = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(block + 16)), shuf_mask);
    __m128i msg2 = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(block + 32)), shuf_mask);
    __m128i msg3 = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(block + 48)), shuf_mask);

    __m128i msg_tmp;

    // Rounds 0-3
    msg_tmp = _mm_add_epi32(msg0, _mm_loadu_si128(k_ptr));
    state1 = _mm_sha256rnds2_epu32(state1, state0, msg_tmp);
    msg_tmp = _mm_shuffle_epi32(msg_tmp, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, msg_tmp);

    // Rounds 4-7
    msg_tmp = _mm_add_epi32(msg1, _mm_loadu_si128(k_ptr + 1));
    state1 = _mm_sha256rnds2_epu32(state1, state0, msg_tmp);
    msg_tmp = _mm_shuffle_epi32(msg_tmp, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, msg_tmp);
    msg0 = _mm_sha256msg1_epu32(msg0, msg1);

    // Rounds 8-11
    msg_tmp = _mm_add_epi32(msg2, _mm_loadu_si128(k_ptr + 2));
    state1 = _mm_sha256rnds2_epu32(state1, state0, msg_tmp);
    msg_tmp = _mm_shuffle_epi32(msg_tmp, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, msg_tmp);
    msg1 = _mm_sha256msg1_epu32(msg1, msg2);

    // Rounds 12-15
    msg_tmp = _mm_add_epi32(msg3, _mm_loadu_si128(k_ptr + 3));
    state1 = _mm_sha256rnds2_epu32(state1, state0, msg_tmp);
    msg_tmp = _mm_shuffle_epi32(msg_tmp, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, msg_tmp);
    msg0 = _mm_add_epi32(msg0, _mm_alignr_epi8(msg3, msg2, 4));
    msg0 = _mm_sha256msg2_epu32(msg0, msg3);
    msg2 = _mm_sha256msg1_epu32(msg2, msg3);

    // Rounds 16-19
    msg_tmp = _mm_add_epi32(msg0, _mm_loadu_si128(k_ptr + 4));
    state1 = _mm_sha256rnds2_epu32(state1, state0, msg_tmp);
    msg_tmp = _mm_shuffle_epi32(msg_tmp, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, msg_tmp);
    msg1 = _mm_add_epi32(msg1, _mm_alignr_epi8(msg0, msg3, 4));
    msg1 = _mm_sha256msg2_epu32(msg1, msg0);
    msg3 = _mm_sha256msg1_epu32(msg3, msg0);

    // Rounds 20-23
    msg_tmp = _mm_add_epi32(msg1, _mm_loadu_si128(k_ptr + 5));
    state1 = _mm_sha256rnds2_epu32(state1, state0, msg_tmp);
    msg_tmp = _mm_shuffle_epi32(msg_tmp, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, msg_tmp);
    msg2 = _mm_add_epi32(msg2, _mm_alignr_epi8(msg1, msg0, 4));
    msg2 = _mm_sha256msg2_epu32(msg2, msg1);
    msg0 = _mm_sha256msg1_epu32(msg0, msg1);

    // Rounds 24-27
    msg_tmp = _mm_add_epi32(msg2, _mm_loadu_si128(k_ptr + 6));
    state1 = _mm_sha256rnds2_epu32(state1, state0, msg_tmp);
    msg_tmp = _mm_shuffle_epi32(msg_tmp, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, msg_tmp);
    msg3 = _mm_add_epi32(msg3, _mm_alignr_epi8(msg2, msg1, 4));
    msg3 = _mm_sha256msg2_epu32(msg3, msg2);
    msg1 = _mm_sha256msg1_epu32(msg1, msg2);

    // Rounds 28-31
    msg_tmp = _mm_add_epi32(msg3, _mm_loadu_si128(k_ptr + 7));
    state1 = _mm_sha256rnds2_epu32(state1, state0, msg_tmp);
    msg_tmp = _mm_shuffle_epi32(msg_tmp, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, msg_tmp);
    msg0 = _mm_add_epi32(msg0, _mm_alignr_epi8(msg3, msg2, 4));
    msg0 = _mm_sha256msg2_epu32(msg0, msg3);
    msg2 = _mm_sha256msg1_epu32(msg2, msg3);

    // Rounds 32-35
    msg_tmp = _mm_add_epi32(msg0, _mm_loadu_si128(k_ptr + 8));
    state1 = _mm_sha256rnds2_epu32(state1, state0, msg_tmp);
    msg_tmp = _mm_shuffle_epi32(msg_tmp, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, msg_tmp);
    msg1 = _mm_add_epi32(msg1, _mm_alignr_epi8(msg0, msg3, 4));
    msg1 = _mm_sha256msg2_epu32(msg1, msg0);
    msg3 = _mm_sha256msg1_epu32(msg3, msg0);

    // Rounds 36-39
    msg_tmp = _mm_add_epi32(msg1, _mm_loadu_si128(k_ptr + 9));
    state1 = _mm_sha256rnds2_epu32(state1, state0, msg_tmp);
    msg_tmp = _mm_shuffle_epi32(msg_tmp, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, msg_tmp);
    msg2 = _mm_add_epi32(msg2, _mm_alignr_epi8(msg1, msg0, 4));
    msg2 = _mm_sha256msg2_epu32(msg2, msg1);
    msg0 = _mm_sha256msg1_epu32(msg0, msg1);

    // Rounds 40-43
    msg_tmp = _mm_add_epi32(msg2, _mm_loadu_si128(k_ptr + 10));
    state1 = _mm_sha256rnds2_epu32(state1, state0, msg_tmp);
    msg_tmp = _mm_shuffle_epi32(msg_tmp, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, msg_tmp);
    msg3 = _mm_add_epi32(msg3, _mm_alignr_epi8(msg2, msg1, 4));
    msg3 = _mm_sha256msg2_epu32(msg3, msg2);
    msg1 = _mm_sha256msg1_epu32(msg1, msg2);

    // Rounds 44-47
    msg_tmp = _mm_add_epi32(msg3, _mm_loadu_si128(k_ptr + 11));
    state1 = _mm_sha256rnds2_epu32(state1, state0, msg_tmp);
    msg_tmp = _mm_shuffle_epi32(msg_tmp, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, msg_tmp);
    msg0 = _mm_add_epi32(msg0, _mm_alignr_epi8(msg3, msg2, 4));
    msg0 = _mm_sha256msg2_epu32(msg0, msg3);
    msg2 = _mm_sha256msg1_epu32(msg2, msg3);

    // Rounds 48-51
    msg_tmp = _mm_add_epi32(msg0, _mm_loadu_si128(k_ptr + 12));
    state1 = _mm_sha256rnds2_epu32(state1, state0, msg_tmp);
    msg_tmp = _mm_shuffle_epi32(msg_tmp, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, msg_tmp);
    msg1 = _mm_add_epi32(msg1, _mm_alignr_epi8(msg0, msg3, 4));
    msg1 = _mm_sha256msg2_epu32(msg1, msg0);
    msg3 = _mm_sha256msg1_epu32(msg3, msg0);

    // Rounds 52-55
    msg_tmp = _mm_add_epi32(msg1, _mm_loadu_si128(k_ptr + 13));
    state1 = _mm_sha256rnds2_epu32(state1, state0, msg_tmp);
    msg_tmp = _mm_shuffle_epi32(msg_tmp, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, msg_tmp);
    msg2 = _mm_add_epi32(msg2, _mm_alignr_epi8(msg1, msg0, 4));
    msg2 = _mm_sha256msg2_epu32(msg2, msg1);

    // Rounds 56-59
    msg_tmp = _mm_add_epi32(msg2, _mm_loadu_si128(k_ptr + 14));
    state1 = _mm_sha256rnds2_epu32(state1, state0, msg_tmp);
    msg_tmp = _mm_shuffle_epi32(msg_tmp, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, msg_tmp);
    msg3 = _mm_add_epi32(msg3, _mm_alignr_epi8(msg2, msg1, 4));
    msg3 = _mm_sha256msg2_epu32(msg3, msg2);

    // Rounds 60-63
    msg_tmp = _mm_add_epi32(msg3, _mm_loadu_si128(k_ptr + 15));
    state1 = _mm_sha256rnds2_epu32(state1, state0, msg_tmp);
    msg_tmp = _mm_shuffle_epi32(msg_tmp, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, msg_tmp);

    state0 = _mm_add_epi32(state0, abef_save);
    state1 = _mm_add_epi32(state1, cdgh_save);
}

#if defined(__GNUC__) && !defined(_MSC_VER)
__attribute__((target("sha,sse4.1,ssse3")))
#endif
static void sha256_block_ni(uint32_t state[8], const uint8_t* block) {
    const __m128i shuf_mask = _mm_set_epi64x(0x0c0d0e0f08090a0bLL, 0x0405060700010203LL);
    __m128i state0;
    __m128i state1;
    sha256_state_to_ni(state, state0, state1);
    sha256_block_ni_core(state0, state1, block, shuf_mask);
    sha256_state_from_ni(state, state0, state1);
}

#if defined(__GNUC__) && !defined(_MSC_VER)
__attribute__((target("sha,sse4.1,ssse3")))
#endif
static void sha256_blocks_ni(uint32_t state[8], const uint8_t* blocks, size_t block_count) {
    if (block_count == 0) {
        return;
    }

    static constexpr size_t kPrefetchBlocksAhead = 8;
    const __m128i shuf_mask = _mm_set_epi64x(0x0c0d0e0f08090a0bLL, 0x0405060700010203LL);
    __m128i state0;
    __m128i state1;
    sha256_state_to_ni(state, state0, state1);
    size_t block_idx = 0;
    for (; block_idx + 1 < block_count; block_idx += 2) {
        const size_t pf0 = block_idx + kPrefetchBlocksAhead;
        const size_t pf1 = pf0 + 1;
        if (pf1 < block_count) {
            _mm_prefetch(reinterpret_cast<const char*>(blocks + pf0 * 64), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(blocks + pf1 * 64), _MM_HINT_T0);
        } else if (pf0 < block_count) {
            _mm_prefetch(reinterpret_cast<const char*>(blocks + pf0 * 64), _MM_HINT_T0);
        }
        sha256_block_ni_core(state0, state1, blocks + block_idx * 64, shuf_mask);
        sha256_block_ni_core(state0, state1, blocks + (block_idx + 1) * 64, shuf_mask);
    }
    for (; block_idx < block_count; ++block_idx) {
        const size_t pf = block_idx + kPrefetchBlocksAhead;
        if (pf < block_count) {
            _mm_prefetch(reinterpret_cast<const char*>(blocks + pf * 64), _MM_HINT_T0);
        }
        sha256_block_ni_core(state0, state1, blocks + block_idx * 64, shuf_mask);
    }
    sha256_state_from_ni(state, state0, state1);
}

#if defined(__GNUC__) && !defined(_MSC_VER)
__attribute__((target("sha,sse4.1,ssse3")))
#endif
static void sha256_2way_block_ni_core(
    __m128i& state0_a, __m128i& state1_a, const uint8_t* block_a,
    __m128i& state0_b, __m128i& state1_b, const uint8_t* block_b,
    __m128i shuf_mask);

#if defined(__GNUC__) && !defined(_MSC_VER)
__attribute__((target("sha,sse4.1,ssse3")))
#endif
static void sha256_blocks_2way_ni(
    uint32_t state_a[8], const uint8_t* blocks_a, size_t block_count_a,
    uint32_t state_b[8], const uint8_t* blocks_b, size_t block_count_b)
{
    if (block_count_a == 0 && block_count_b == 0) {
        return;
    }

    const __m128i shuf_mask = _mm_set_epi64x(0x0c0d0e0f08090a0bLL, 0x0405060700010203LL);
    __m128i state0_a;
    __m128i state1_a;
    __m128i state0_b;
    __m128i state1_b;
    sha256_state_to_ni(state_a, state0_a, state1_a);
    sha256_state_to_ni(state_b, state0_b, state1_b);

    const size_t common_blocks = std::min(block_count_a, block_count_b);
    for (size_t block_idx = 0; block_idx < common_blocks; ++block_idx) {
        sha256_2way_block_ni_core(
            state0_a, state1_a, blocks_a + block_idx * 64,
            state0_b, state1_b, blocks_b + block_idx * 64,
            shuf_mask);
    }
    for (size_t block_idx = common_blocks; block_idx < block_count_a; ++block_idx) {
        sha256_block_ni_core(state0_a, state1_a, blocks_a + block_idx * 64, shuf_mask);
    }
    for (size_t block_idx = common_blocks; block_idx < block_count_b; ++block_idx) {
        sha256_block_ni_core(state0_b, state1_b, blocks_b + block_idx * 64, shuf_mask);
    }

    sha256_state_from_ni(state_a, state0_a, state1_a);
    sha256_state_from_ni(state_b, state0_b, state1_b);
}

// ============================================================================
// 2-WAY SHA256-NI: Process TWO independent hash contexts simultaneously
// ============================================================================
//
// Ported from DeroLuna's FUN_14047f2a0 (Ghidra reverse engineering).
// Interleaves SHA-NI rounds from two independent SHA256 computations to
// hide the ~5-cycle latency of sha256rnds2 (throughput: 1/cycle on Raptor Lake).
//
// Use case: When two independent SHA256_Init/Update/Final sequences are
// available (e.g., two nonces' initial hashes or two SA output hashes),
// calling this function processes both ~40% faster than sequential calls.
//
// Register allocation (matching DeroLuna):
//   Stream A state: state0_a (ABEF), state1_a (CDGH)
//   Stream B state: state0_b (ABEF), state1_b (CDGH)
//   Stream A messages: msg0_a..msg3_a
//   Stream B messages: msg0_b..msg3_b
//

#if defined(__GNUC__) && !defined(_MSC_VER)
__attribute__((target("sha,sse4.1,ssse3")))
#endif
static void sha256_2way_block_ni_core(
    __m128i& state0_a, __m128i& state1_a, const uint8_t* block_a,
    __m128i& state0_b, __m128i& state1_b, const uint8_t* block_b,
    __m128i shuf_mask)
{
    const __m128i* k_ptr = reinterpret_cast<const __m128i*>(SHA256_K);

    __m128i abef_save_a = state0_a;
    __m128i cdgh_save_a = state1_a;
    __m128i abef_save_b = state0_b;
    __m128i cdgh_save_b = state1_b;

    // Load and byte-swap message blocks for both streams
    __m128i msg0_a = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(block_a)),      shuf_mask);
    __m128i msg0_b = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(block_b)),      shuf_mask);
    __m128i msg1_a = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(block_a + 16)), shuf_mask);
    __m128i msg1_b = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(block_b + 16)), shuf_mask);
    __m128i msg2_a = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(block_a + 32)), shuf_mask);
    __m128i msg2_b = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(block_b + 32)), shuf_mask);
    __m128i msg3_a = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(block_a + 48)), shuf_mask);
    __m128i msg3_b = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(block_b + 48)), shuf_mask);

    __m128i tmp_a, tmp_b;

    // Rounds 0-3: interleaved A then B
    tmp_a = _mm_add_epi32(msg0_a, _mm_loadu_si128(k_ptr));
    state1_a = _mm_sha256rnds2_epu32(state1_a, state0_a, tmp_a);
    tmp_b = _mm_add_epi32(msg0_b, _mm_loadu_si128(k_ptr));
    state1_b = _mm_sha256rnds2_epu32(state1_b, state0_b, tmp_b);
    tmp_a = _mm_shuffle_epi32(tmp_a, 0x0E);
    state0_a = _mm_sha256rnds2_epu32(state0_a, state1_a, tmp_a);
    tmp_b = _mm_shuffle_epi32(tmp_b, 0x0E);
    state0_b = _mm_sha256rnds2_epu32(state0_b, state1_b, tmp_b);

    // Rounds 4-7
    tmp_a = _mm_add_epi32(msg1_a, _mm_loadu_si128(k_ptr + 1));
    state1_a = _mm_sha256rnds2_epu32(state1_a, state0_a, tmp_a);
    tmp_b = _mm_add_epi32(msg1_b, _mm_loadu_si128(k_ptr + 1));
    state1_b = _mm_sha256rnds2_epu32(state1_b, state0_b, tmp_b);
    tmp_a = _mm_shuffle_epi32(tmp_a, 0x0E);
    state0_a = _mm_sha256rnds2_epu32(state0_a, state1_a, tmp_a);
    tmp_b = _mm_shuffle_epi32(tmp_b, 0x0E);
    state0_b = _mm_sha256rnds2_epu32(state0_b, state1_b, tmp_b);
    msg0_a = _mm_sha256msg1_epu32(msg0_a, msg1_a);
    msg0_b = _mm_sha256msg1_epu32(msg0_b, msg1_b);

    // Rounds 8-11
    tmp_a = _mm_add_epi32(msg2_a, _mm_loadu_si128(k_ptr + 2));
    state1_a = _mm_sha256rnds2_epu32(state1_a, state0_a, tmp_a);
    tmp_b = _mm_add_epi32(msg2_b, _mm_loadu_si128(k_ptr + 2));
    state1_b = _mm_sha256rnds2_epu32(state1_b, state0_b, tmp_b);
    tmp_a = _mm_shuffle_epi32(tmp_a, 0x0E);
    state0_a = _mm_sha256rnds2_epu32(state0_a, state1_a, tmp_a);
    tmp_b = _mm_shuffle_epi32(tmp_b, 0x0E);
    state0_b = _mm_sha256rnds2_epu32(state0_b, state1_b, tmp_b);
    msg1_a = _mm_sha256msg1_epu32(msg1_a, msg2_a);
    msg1_b = _mm_sha256msg1_epu32(msg1_b, msg2_b);

    // Rounds 12-15
    tmp_a = _mm_add_epi32(msg3_a, _mm_loadu_si128(k_ptr + 3));
    state1_a = _mm_sha256rnds2_epu32(state1_a, state0_a, tmp_a);
    tmp_b = _mm_add_epi32(msg3_b, _mm_loadu_si128(k_ptr + 3));
    state1_b = _mm_sha256rnds2_epu32(state1_b, state0_b, tmp_b);
    tmp_a = _mm_shuffle_epi32(tmp_a, 0x0E);
    state0_a = _mm_sha256rnds2_epu32(state0_a, state1_a, tmp_a);
    tmp_b = _mm_shuffle_epi32(tmp_b, 0x0E);
    state0_b = _mm_sha256rnds2_epu32(state0_b, state1_b, tmp_b);
    msg0_a = _mm_add_epi32(msg0_a, _mm_alignr_epi8(msg3_a, msg2_a, 4));
    msg0_a = _mm_sha256msg2_epu32(msg0_a, msg3_a);
    msg0_b = _mm_add_epi32(msg0_b, _mm_alignr_epi8(msg3_b, msg2_b, 4));
    msg0_b = _mm_sha256msg2_epu32(msg0_b, msg3_b);
    msg2_a = _mm_sha256msg1_epu32(msg2_a, msg3_a);
    msg2_b = _mm_sha256msg1_epu32(msg2_b, msg3_b);

    // Rounds 16-19
    tmp_a = _mm_add_epi32(msg0_a, _mm_loadu_si128(k_ptr + 4));
    state1_a = _mm_sha256rnds2_epu32(state1_a, state0_a, tmp_a);
    tmp_b = _mm_add_epi32(msg0_b, _mm_loadu_si128(k_ptr + 4));
    state1_b = _mm_sha256rnds2_epu32(state1_b, state0_b, tmp_b);
    tmp_a = _mm_shuffle_epi32(tmp_a, 0x0E);
    state0_a = _mm_sha256rnds2_epu32(state0_a, state1_a, tmp_a);
    tmp_b = _mm_shuffle_epi32(tmp_b, 0x0E);
    state0_b = _mm_sha256rnds2_epu32(state0_b, state1_b, tmp_b);
    msg1_a = _mm_add_epi32(msg1_a, _mm_alignr_epi8(msg0_a, msg3_a, 4));
    msg1_a = _mm_sha256msg2_epu32(msg1_a, msg0_a);
    msg1_b = _mm_add_epi32(msg1_b, _mm_alignr_epi8(msg0_b, msg3_b, 4));
    msg1_b = _mm_sha256msg2_epu32(msg1_b, msg0_b);
    msg3_a = _mm_sha256msg1_epu32(msg3_a, msg0_a);
    msg3_b = _mm_sha256msg1_epu32(msg3_b, msg0_b);

    // Rounds 20-23
    tmp_a = _mm_add_epi32(msg1_a, _mm_loadu_si128(k_ptr + 5));
    state1_a = _mm_sha256rnds2_epu32(state1_a, state0_a, tmp_a);
    tmp_b = _mm_add_epi32(msg1_b, _mm_loadu_si128(k_ptr + 5));
    state1_b = _mm_sha256rnds2_epu32(state1_b, state0_b, tmp_b);
    tmp_a = _mm_shuffle_epi32(tmp_a, 0x0E);
    state0_a = _mm_sha256rnds2_epu32(state0_a, state1_a, tmp_a);
    tmp_b = _mm_shuffle_epi32(tmp_b, 0x0E);
    state0_b = _mm_sha256rnds2_epu32(state0_b, state1_b, tmp_b);
    msg2_a = _mm_add_epi32(msg2_a, _mm_alignr_epi8(msg1_a, msg0_a, 4));
    msg2_a = _mm_sha256msg2_epu32(msg2_a, msg1_a);
    msg2_b = _mm_add_epi32(msg2_b, _mm_alignr_epi8(msg1_b, msg0_b, 4));
    msg2_b = _mm_sha256msg2_epu32(msg2_b, msg1_b);
    msg0_a = _mm_sha256msg1_epu32(msg0_a, msg1_a);
    msg0_b = _mm_sha256msg1_epu32(msg0_b, msg1_b);

    // Rounds 24-27
    tmp_a = _mm_add_epi32(msg2_a, _mm_loadu_si128(k_ptr + 6));
    state1_a = _mm_sha256rnds2_epu32(state1_a, state0_a, tmp_a);
    tmp_b = _mm_add_epi32(msg2_b, _mm_loadu_si128(k_ptr + 6));
    state1_b = _mm_sha256rnds2_epu32(state1_b, state0_b, tmp_b);
    tmp_a = _mm_shuffle_epi32(tmp_a, 0x0E);
    state0_a = _mm_sha256rnds2_epu32(state0_a, state1_a, tmp_a);
    tmp_b = _mm_shuffle_epi32(tmp_b, 0x0E);
    state0_b = _mm_sha256rnds2_epu32(state0_b, state1_b, tmp_b);
    msg3_a = _mm_add_epi32(msg3_a, _mm_alignr_epi8(msg2_a, msg1_a, 4));
    msg3_a = _mm_sha256msg2_epu32(msg3_a, msg2_a);
    msg3_b = _mm_add_epi32(msg3_b, _mm_alignr_epi8(msg2_b, msg1_b, 4));
    msg3_b = _mm_sha256msg2_epu32(msg3_b, msg2_b);
    msg1_a = _mm_sha256msg1_epu32(msg1_a, msg2_a);
    msg1_b = _mm_sha256msg1_epu32(msg1_b, msg2_b);

    // Rounds 28-31
    tmp_a = _mm_add_epi32(msg3_a, _mm_loadu_si128(k_ptr + 7));
    state1_a = _mm_sha256rnds2_epu32(state1_a, state0_a, tmp_a);
    tmp_b = _mm_add_epi32(msg3_b, _mm_loadu_si128(k_ptr + 7));
    state1_b = _mm_sha256rnds2_epu32(state1_b, state0_b, tmp_b);
    tmp_a = _mm_shuffle_epi32(tmp_a, 0x0E);
    state0_a = _mm_sha256rnds2_epu32(state0_a, state1_a, tmp_a);
    tmp_b = _mm_shuffle_epi32(tmp_b, 0x0E);
    state0_b = _mm_sha256rnds2_epu32(state0_b, state1_b, tmp_b);
    msg0_a = _mm_add_epi32(msg0_a, _mm_alignr_epi8(msg3_a, msg2_a, 4));
    msg0_a = _mm_sha256msg2_epu32(msg0_a, msg3_a);
    msg0_b = _mm_add_epi32(msg0_b, _mm_alignr_epi8(msg3_b, msg2_b, 4));
    msg0_b = _mm_sha256msg2_epu32(msg0_b, msg3_b);
    msg2_a = _mm_sha256msg1_epu32(msg2_a, msg3_a);
    msg2_b = _mm_sha256msg1_epu32(msg2_b, msg3_b);

    // Rounds 32-35
    tmp_a = _mm_add_epi32(msg0_a, _mm_loadu_si128(k_ptr + 8));
    state1_a = _mm_sha256rnds2_epu32(state1_a, state0_a, tmp_a);
    tmp_b = _mm_add_epi32(msg0_b, _mm_loadu_si128(k_ptr + 8));
    state1_b = _mm_sha256rnds2_epu32(state1_b, state0_b, tmp_b);
    tmp_a = _mm_shuffle_epi32(tmp_a, 0x0E);
    state0_a = _mm_sha256rnds2_epu32(state0_a, state1_a, tmp_a);
    tmp_b = _mm_shuffle_epi32(tmp_b, 0x0E);
    state0_b = _mm_sha256rnds2_epu32(state0_b, state1_b, tmp_b);
    msg1_a = _mm_add_epi32(msg1_a, _mm_alignr_epi8(msg0_a, msg3_a, 4));
    msg1_a = _mm_sha256msg2_epu32(msg1_a, msg0_a);
    msg1_b = _mm_add_epi32(msg1_b, _mm_alignr_epi8(msg0_b, msg3_b, 4));
    msg1_b = _mm_sha256msg2_epu32(msg1_b, msg0_b);
    msg3_a = _mm_sha256msg1_epu32(msg3_a, msg0_a);
    msg3_b = _mm_sha256msg1_epu32(msg3_b, msg0_b);

    // Rounds 36-39
    tmp_a = _mm_add_epi32(msg1_a, _mm_loadu_si128(k_ptr + 9));
    state1_a = _mm_sha256rnds2_epu32(state1_a, state0_a, tmp_a);
    tmp_b = _mm_add_epi32(msg1_b, _mm_loadu_si128(k_ptr + 9));
    state1_b = _mm_sha256rnds2_epu32(state1_b, state0_b, tmp_b);
    tmp_a = _mm_shuffle_epi32(tmp_a, 0x0E);
    state0_a = _mm_sha256rnds2_epu32(state0_a, state1_a, tmp_a);
    tmp_b = _mm_shuffle_epi32(tmp_b, 0x0E);
    state0_b = _mm_sha256rnds2_epu32(state0_b, state1_b, tmp_b);
    msg2_a = _mm_add_epi32(msg2_a, _mm_alignr_epi8(msg1_a, msg0_a, 4));
    msg2_a = _mm_sha256msg2_epu32(msg2_a, msg1_a);
    msg2_b = _mm_add_epi32(msg2_b, _mm_alignr_epi8(msg1_b, msg0_b, 4));
    msg2_b = _mm_sha256msg2_epu32(msg2_b, msg1_b);
    msg0_a = _mm_sha256msg1_epu32(msg0_a, msg1_a);
    msg0_b = _mm_sha256msg1_epu32(msg0_b, msg1_b);

    // Rounds 40-43
    tmp_a = _mm_add_epi32(msg2_a, _mm_loadu_si128(k_ptr + 10));
    state1_a = _mm_sha256rnds2_epu32(state1_a, state0_a, tmp_a);
    tmp_b = _mm_add_epi32(msg2_b, _mm_loadu_si128(k_ptr + 10));
    state1_b = _mm_sha256rnds2_epu32(state1_b, state0_b, tmp_b);
    tmp_a = _mm_shuffle_epi32(tmp_a, 0x0E);
    state0_a = _mm_sha256rnds2_epu32(state0_a, state1_a, tmp_a);
    tmp_b = _mm_shuffle_epi32(tmp_b, 0x0E);
    state0_b = _mm_sha256rnds2_epu32(state0_b, state1_b, tmp_b);
    msg3_a = _mm_add_epi32(msg3_a, _mm_alignr_epi8(msg2_a, msg1_a, 4));
    msg3_a = _mm_sha256msg2_epu32(msg3_a, msg2_a);
    msg3_b = _mm_add_epi32(msg3_b, _mm_alignr_epi8(msg2_b, msg1_b, 4));
    msg3_b = _mm_sha256msg2_epu32(msg3_b, msg2_b);
    msg1_a = _mm_sha256msg1_epu32(msg1_a, msg2_a);
    msg1_b = _mm_sha256msg1_epu32(msg1_b, msg2_b);

    // Rounds 44-47
    tmp_a = _mm_add_epi32(msg3_a, _mm_loadu_si128(k_ptr + 11));
    state1_a = _mm_sha256rnds2_epu32(state1_a, state0_a, tmp_a);
    tmp_b = _mm_add_epi32(msg3_b, _mm_loadu_si128(k_ptr + 11));
    state1_b = _mm_sha256rnds2_epu32(state1_b, state0_b, tmp_b);
    tmp_a = _mm_shuffle_epi32(tmp_a, 0x0E);
    state0_a = _mm_sha256rnds2_epu32(state0_a, state1_a, tmp_a);
    tmp_b = _mm_shuffle_epi32(tmp_b, 0x0E);
    state0_b = _mm_sha256rnds2_epu32(state0_b, state1_b, tmp_b);
    msg0_a = _mm_add_epi32(msg0_a, _mm_alignr_epi8(msg3_a, msg2_a, 4));
    msg0_a = _mm_sha256msg2_epu32(msg0_a, msg3_a);
    msg0_b = _mm_add_epi32(msg0_b, _mm_alignr_epi8(msg3_b, msg2_b, 4));
    msg0_b = _mm_sha256msg2_epu32(msg0_b, msg3_b);
    msg2_a = _mm_sha256msg1_epu32(msg2_a, msg3_a);
    msg2_b = _mm_sha256msg1_epu32(msg2_b, msg3_b);

    // Rounds 48-51
    tmp_a = _mm_add_epi32(msg0_a, _mm_loadu_si128(k_ptr + 12));
    state1_a = _mm_sha256rnds2_epu32(state1_a, state0_a, tmp_a);
    tmp_b = _mm_add_epi32(msg0_b, _mm_loadu_si128(k_ptr + 12));
    state1_b = _mm_sha256rnds2_epu32(state1_b, state0_b, tmp_b);
    tmp_a = _mm_shuffle_epi32(tmp_a, 0x0E);
    state0_a = _mm_sha256rnds2_epu32(state0_a, state1_a, tmp_a);
    tmp_b = _mm_shuffle_epi32(tmp_b, 0x0E);
    state0_b = _mm_sha256rnds2_epu32(state0_b, state1_b, tmp_b);
    msg1_a = _mm_add_epi32(msg1_a, _mm_alignr_epi8(msg0_a, msg3_a, 4));
    msg1_a = _mm_sha256msg2_epu32(msg1_a, msg0_a);
    msg1_b = _mm_add_epi32(msg1_b, _mm_alignr_epi8(msg0_b, msg3_b, 4));
    msg1_b = _mm_sha256msg2_epu32(msg1_b, msg0_b);
    msg3_a = _mm_sha256msg1_epu32(msg3_a, msg0_a);
    msg3_b = _mm_sha256msg1_epu32(msg3_b, msg0_b);

    // Rounds 52-55
    tmp_a = _mm_add_epi32(msg1_a, _mm_loadu_si128(k_ptr + 13));
    state1_a = _mm_sha256rnds2_epu32(state1_a, state0_a, tmp_a);
    tmp_b = _mm_add_epi32(msg1_b, _mm_loadu_si128(k_ptr + 13));
    state1_b = _mm_sha256rnds2_epu32(state1_b, state0_b, tmp_b);
    tmp_a = _mm_shuffle_epi32(tmp_a, 0x0E);
    state0_a = _mm_sha256rnds2_epu32(state0_a, state1_a, tmp_a);
    tmp_b = _mm_shuffle_epi32(tmp_b, 0x0E);
    state0_b = _mm_sha256rnds2_epu32(state0_b, state1_b, tmp_b);
    msg2_a = _mm_add_epi32(msg2_a, _mm_alignr_epi8(msg1_a, msg0_a, 4));
    msg2_a = _mm_sha256msg2_epu32(msg2_a, msg1_a);
    msg2_b = _mm_add_epi32(msg2_b, _mm_alignr_epi8(msg1_b, msg0_b, 4));
    msg2_b = _mm_sha256msg2_epu32(msg2_b, msg1_b);

    // Rounds 56-59
    tmp_a = _mm_add_epi32(msg2_a, _mm_loadu_si128(k_ptr + 14));
    state1_a = _mm_sha256rnds2_epu32(state1_a, state0_a, tmp_a);
    tmp_b = _mm_add_epi32(msg2_b, _mm_loadu_si128(k_ptr + 14));
    state1_b = _mm_sha256rnds2_epu32(state1_b, state0_b, tmp_b);
    tmp_a = _mm_shuffle_epi32(tmp_a, 0x0E);
    state0_a = _mm_sha256rnds2_epu32(state0_a, state1_a, tmp_a);
    tmp_b = _mm_shuffle_epi32(tmp_b, 0x0E);
    state0_b = _mm_sha256rnds2_epu32(state0_b, state1_b, tmp_b);
    msg3_a = _mm_add_epi32(msg3_a, _mm_alignr_epi8(msg2_a, msg1_a, 4));
    msg3_a = _mm_sha256msg2_epu32(msg3_a, msg2_a);
    msg3_b = _mm_add_epi32(msg3_b, _mm_alignr_epi8(msg2_b, msg1_b, 4));
    msg3_b = _mm_sha256msg2_epu32(msg3_b, msg2_b);

    // Rounds 60-63
    tmp_a = _mm_add_epi32(msg3_a, _mm_loadu_si128(k_ptr + 15));
    state1_a = _mm_sha256rnds2_epu32(state1_a, state0_a, tmp_a);
    tmp_b = _mm_add_epi32(msg3_b, _mm_loadu_si128(k_ptr + 15));
    state1_b = _mm_sha256rnds2_epu32(state1_b, state0_b, tmp_b);
    tmp_a = _mm_shuffle_epi32(tmp_a, 0x0E);
    state0_a = _mm_sha256rnds2_epu32(state0_a, state1_a, tmp_a);
    tmp_b = _mm_shuffle_epi32(tmp_b, 0x0E);
    state0_b = _mm_sha256rnds2_epu32(state0_b, state1_b, tmp_b);

    // Add saved state
    state0_a = _mm_add_epi32(state0_a, abef_save_a);
    state1_a = _mm_add_epi32(state1_a, cdgh_save_a);
    state0_b = _mm_add_epi32(state0_b, abef_save_b);
    state1_b = _mm_add_epi32(state1_b, cdgh_save_b);
}

// Public API: Hash two independent messages in parallel
// Both messages are processed with SHA256 from scratch (Init+Update+Final).
// Approximately 40% faster than two sequential hashSHA256 calls.
#if defined(__GNUC__) && !defined(_MSC_VER)
__attribute__((target("sha,sse4.1,ssse3")))
#endif
extern "C" void sha256_2way_ni(
    const uint8_t* data_a, size_t len_a, uint8_t digest_a[32],
    const uint8_t* data_b, size_t len_b, uint8_t digest_b[32])
{
    // We implement full SHA256 for both messages using 2-way interleaving.
    // This handles Init, Update (with padding), and Final for both streams.

    const __m128i shuf_mask = _mm_set_epi64x(0x0c0d0e0f08090a0bLL, 0x0405060700010203LL);

    // Initialize both states to SHA256 initial values
    __m128i state0_a, state1_a, state0_b, state1_b;
    sha256_state_to_ni(const_cast<uint32_t*>(SHA256_H0), state0_a, state1_a);
    sha256_state_to_ni(const_cast<uint32_t*>(SHA256_H0), state0_b, state1_b);

    // Process full blocks from both messages in parallel
    size_t blocks_a = len_a / 64;
    size_t blocks_b = len_b / 64;
    size_t common_blocks = (blocks_a < blocks_b) ? blocks_a : blocks_b;

    // Process common blocks using 2-way
    for (size_t i = 0; i < common_blocks; ++i) {
        sha256_2way_block_ni_core(
            state0_a, state1_a, data_a + i * 64,
            state0_b, state1_b, data_b + i * 64,
            shuf_mask);
    }

    // Process remaining blocks for stream A (single-way)
    for (size_t i = common_blocks; i < blocks_a; ++i) {
        sha256_block_ni_core(state0_a, state1_a, data_a + i * 64, shuf_mask);
    }

    // Process remaining blocks for stream B (single-way)
    for (size_t i = common_blocks; i < blocks_b; ++i) {
        sha256_block_ni_core(state0_b, state1_b, data_b + i * 64, shuf_mask);
    }

    // Finalize both messages (pad + length + final block)
    // Build padding blocks for A and B
    uint8_t pad_a[128] = {0};  // Max 2 final blocks
    uint8_t pad_b[128] = {0};
    size_t rem_a = len_a - blocks_a * 64;
    size_t rem_b = len_b - blocks_b * 64;

    // Copy remaining bytes
    if (rem_a > 0) memcpy(pad_a, data_a + blocks_a * 64, rem_a);
    if (rem_b > 0) memcpy(pad_b, data_b + blocks_b * 64, rem_b);

    // Append 0x80
    pad_a[rem_a] = 0x80;
    pad_b[rem_b] = 0x80;

    // Calculate final block counts
    size_t final_blocks_a = (rem_a < 56) ? 1 : 2;
    size_t final_blocks_b = (rem_b < 56) ? 1 : 2;

    // Append bit length (big-endian 64-bit) at the end of the last block
    uint64_t bit_len_a = (uint64_t)len_a * 8;
    uint64_t bit_len_b = (uint64_t)len_b * 8;
    size_t len_offset_a = final_blocks_a * 64 - 8;
    size_t len_offset_b = final_blocks_b * 64 - 8;
    for (int i = 0; i < 8; i++) {
        pad_a[len_offset_a + i] = (uint8_t)(bit_len_a >> (56 - i * 8));
        pad_b[len_offset_b + i] = (uint8_t)(bit_len_b >> (56 - i * 8));
    }

    // Process final blocks using 2-way where possible
    size_t common_final = (final_blocks_a < final_blocks_b) ? final_blocks_a : final_blocks_b;
    for (size_t i = 0; i < common_final; ++i) {
        sha256_2way_block_ni_core(
            state0_a, state1_a, pad_a + i * 64,
            state0_b, state1_b, pad_b + i * 64,
            shuf_mask);
    }
    // Process remaining final blocks single-way
    for (size_t i = common_final; i < final_blocks_a; ++i) {
        sha256_block_ni_core(state0_a, state1_a, pad_a + i * 64, shuf_mask);
    }
    for (size_t i = common_final; i < final_blocks_b; ++i) {
        sha256_block_ni_core(state0_b, state1_b, pad_b + i * 64, shuf_mask);
    }

    // Extract final hash values
    uint32_t state_out_a[8], state_out_b[8];
    sha256_state_from_ni(state_out_a, state0_a, state1_a);
    sha256_state_from_ni(state_out_b, state0_b, state1_b);

    // Convert to big-endian output
    for (int i = 0; i < 8; i++) {
        digest_a[i*4]   = (uint8_t)(state_out_a[i] >> 24);
        digest_a[i*4+1] = (uint8_t)(state_out_a[i] >> 16);
        digest_a[i*4+2] = (uint8_t)(state_out_a[i] >> 8);
        digest_a[i*4+3] = (uint8_t)(state_out_a[i]);
        digest_b[i*4]   = (uint8_t)(state_out_b[i] >> 24);
        digest_b[i*4+1] = (uint8_t)(state_out_b[i] >> 16);
        digest_b[i*4+2] = (uint8_t)(state_out_b[i] >> 8);
        digest_b[i*4+3] = (uint8_t)(state_out_b[i]);
    }
}

#endif  // x86_64

// ============================================================================
// Software fallback
// ============================================================================

static inline uint32_t rotr32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

static void sha256_block_soft(uint32_t state[8], const uint8_t* block) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)block[i*4] << 24) | ((uint32_t)block[i*4+1] << 16)
             | ((uint32_t)block[i*4+2] << 8) | (uint32_t)block[i*4+3];
    }
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr32(w[i-15], 7) ^ rotr32(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = rotr32(w[i-2], 17) ^ rotr32(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }

    uint32_t a=state[0], b=state[1], c=state[2], d=state[3];
    uint32_t e=state[4], f=state[5], g=state[6], h=state[7];

    for (int i = 0; i < 64; i++) {
        uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = h + S1 + ch + SHA256_K[i] + w[i];
        uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + maj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }

    state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
    state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
}

// ============================================================================
// Dispatch: SHA-NI if available, else software
// ============================================================================

static void sha256_process_block(uint32_t state[8], const uint8_t* block) {
#if defined(__x86_64__) || defined(_M_X64)
    if (!g_sha_ni_checked) {
        g_sha_ni_available = check_sha_ni();
        g_sha_ni_checked = true;
    }
    if (g_sha_ni_available) {
        sha256_block_ni(state, block);
        return;
    }
#endif
    sha256_block_soft(state, block);
}

static void sha256_process_blocks(uint32_t state[8], const uint8_t* blocks, size_t block_count) {
    if (block_count == 0) {
        return;
    }
#if defined(__x86_64__) || defined(_M_X64)
    if (!g_sha_ni_checked) {
        g_sha_ni_available = check_sha_ni();
        g_sha_ni_checked = true;
    }
    if (g_sha_ni_available) {
        sha256_blocks_ni(state, blocks, block_count);
        return;
    }
#endif
    for (size_t block_idx = 0; block_idx < block_count; ++block_idx) {
        sha256_block_soft(state, blocks + block_idx * 64);
    }
}

static bool sha256_try_pair_blocks(uint32_t state[8], const uint8_t* blocks, size_t block_count, bool profile) {
#if defined(__x86_64__) || defined(_M_X64)
    if (!g_spsa_sha_pair || block_count < SHA256_PAIR_MIN_BLOCKS) {
        return false;
    }

    if (!g_sha_ni_checked) {
        g_sha_ni_available = check_sha_ni();
        g_sha_ni_checked = true;
    }
    if (!g_sha_ni_available) {
        return false;
    }

    if (profile) {
        g_sha_pair_attempt_calls.fetch_add(1, std::memory_order_relaxed);
    }

    Sha256PairSlot& slot = g_sha_pair_slot;
    uint32_t expected = SHA256_PAIR_WAITING;
    if (slot.phase.compare_exchange_strong(expected, SHA256_PAIR_CLAIMED, std::memory_order_acq_rel, std::memory_order_relaxed)) {
        uint32_t* const partner_state = slot.state_words;
        const uint8_t* const partner_blocks = slot.blocks;
        const size_t partner_block_count = slot.block_count;
        if (partner_state != nullptr && partner_blocks != nullptr && partner_block_count > 0) {
            sha256_blocks_2way_ni(state, blocks, block_count, partner_state, partner_blocks, partner_block_count);
            if (profile) {
                g_sha_pair_success_calls.fetch_add(1, std::memory_order_relaxed);
                g_sha_pair_blocks.fetch_add(static_cast<uint64_t>(block_count), std::memory_order_relaxed);
            }
            slot.phase.store(SHA256_PAIR_DONE, std::memory_order_release);
            return true;
        }

        slot.phase.store(SHA256_PAIR_IDLE, std::memory_order_release);
        if (profile) {
            g_sha_pair_fallback_calls.fetch_add(1, std::memory_order_relaxed);
        }
        return false;
    }

    expected = SHA256_PAIR_IDLE;
    if (!slot.phase.compare_exchange_strong(expected, SHA256_PAIR_RESERVED, std::memory_order_acq_rel, std::memory_order_relaxed)) {
        if (profile) {
            g_sha_pair_fallback_calls.fetch_add(1, std::memory_order_relaxed);
        }
        return false;
    }

    slot.state_words = state;
    slot.blocks = blocks;
    slot.block_count = block_count;
    slot.phase.store(SHA256_PAIR_WAITING, std::memory_order_release);

    for (uint32_t spins = 0; spins < SHA256_PAIR_SPIN_LIMIT; ++spins) {
        const uint32_t phase = slot.phase.load(std::memory_order_acquire);
        if (phase == SHA256_PAIR_DONE) {
            slot.phase.store(SHA256_PAIR_IDLE, std::memory_order_release);
            if (profile) {
                g_sha_pair_success_calls.fetch_add(1, std::memory_order_relaxed);
                g_sha_pair_blocks.fetch_add(static_cast<uint64_t>(block_count), std::memory_order_relaxed);
            }
            return true;
        }
        _mm_pause();
    }

    expected = SHA256_PAIR_WAITING;
    if (slot.phase.compare_exchange_strong(expected, SHA256_PAIR_IDLE, std::memory_order_acq_rel, std::memory_order_relaxed)) {
        if (profile) {
            g_sha_pair_fallback_calls.fetch_add(1, std::memory_order_relaxed);
        }
        return false;
    }

    while (slot.phase.load(std::memory_order_acquire) != SHA256_PAIR_DONE) {
        _mm_pause();
    }
    slot.phase.store(SHA256_PAIR_IDLE, std::memory_order_release);
    if (profile) {
        g_sha_pair_success_calls.fetch_add(1, std::memory_order_relaxed);
        g_sha_pair_blocks.fetch_add(static_cast<uint64_t>(block_count), std::memory_order_relaxed);
    }
    return true;
#else
    (void)state;
    (void)blocks;
    (void)block_count;
    (void)profile;
    return false;
#endif
}

// ============================================================================
// OpenSSL-compatible API (C linkage to match SPSA library expectations)
// ============================================================================

#if defined(DIRTYBIRD_SHA256_LINK_WRAP)
#define DIRTYBIRD_SHA256_OVERRIDE_SYMBOL(name) __wrap_##name
#else
#define DIRTYBIRD_SHA256_OVERRIDE_SYMBOL(name) name
#endif

extern "C" {

// Access the SHA256_CTX block buffer as bytes
// OpenSSL uses SHA_LONG data[SHA_LBLOCK] = uint32_t[16] = 64 bytes
#define CTX_BUF(c) ((uint8_t*)(c)->data)

int DIRTYBIRD_SHA256_OVERRIDE_SYMBOL(SHA256_Init)(SHA256_CTX *c) {
    if (g_spsa_sha_profile) {
        g_sha_init_calls.fetch_add(1, std::memory_order_relaxed);
    }
    memcpy(c->h, SHA256_H0, sizeof(SHA256_H0));
    c->Nl = 0;
    c->Nh = 0;
    c->num = 0;
    c->md_len = SHA256_DIGEST_LENGTH;
    return 1;
}

int DIRTYBIRD_SHA256_OVERRIDE_SYMBOL(SHA256_Update)(SHA256_CTX *c, const void *data_, size_t len) {
    const bool profile = g_spsa_sha_profile;
    const size_t input_len = len;
    uint64_t update_start_ns = 0;
    size_t flush_blocks = 0;
    size_t direct_blocks = 0;
    if (profile) {
        update_start_ns = sha_now_ns();
        g_sha_update_calls.fetch_add(1, std::memory_order_relaxed);
        g_sha_update_bytes.fetch_add(static_cast<uint64_t>(input_len), std::memory_order_relaxed);
        sha_record_update_len_bucket(input_len);
    }

    const uint8_t *data = (const uint8_t*)data_;
    uint8_t *buf = CTX_BUF(c);

    // Update bit count
    uint32_t l = c->Nl + (uint32_t)(len << 3);
    if (l < c->Nl) c->Nh++;
    c->Nh += (uint32_t)(len >> 29);
    c->Nl = l;

    // If we have buffered data, try to complete a block
    if (c->num > 0) {
        uint32_t need = 64 - c->num;
        if (len < need) {
            memcpy(buf + c->num, data, len);
            c->num += (uint32_t)len;
            if (profile) {
                g_sha_update_ns.fetch_add(sha_now_ns() - update_start_ns, std::memory_order_relaxed);
            }
            return 1;
        }
        memcpy(buf + c->num, data, need);
        sha256_process_block(c->h, buf);
        flush_blocks++;
        data += need;
        len -= need;
        c->num = 0;
    }

    // Process full blocks directly from input
    if (len >= 64) {
        const size_t full_blocks = len / 64;
        const bool paired = (c->num == 0) && sha256_try_pair_blocks(c->h, data, full_blocks, profile);
        if (!paired) {
            sha256_process_blocks(c->h, data, full_blocks);
        }
        direct_blocks += full_blocks;
        const size_t consumed = full_blocks * 64;
        data += consumed;
        len -= consumed;
    }

    // Buffer remaining bytes
    if (len > 0) {
        memcpy(buf, data, len);
        c->num = (uint32_t)len;
    }

    if (profile) {
        g_sha_update_flush_blocks.fetch_add(static_cast<uint64_t>(flush_blocks), std::memory_order_relaxed);
        g_sha_update_direct_blocks.fetch_add(static_cast<uint64_t>(direct_blocks), std::memory_order_relaxed);
        g_sha_update_ns.fetch_add(sha_now_ns() - update_start_ns, std::memory_order_relaxed);
    }

    return 1;
}

int DIRTYBIRD_SHA256_OVERRIDE_SYMBOL(SHA256_Final)(unsigned char *md, SHA256_CTX *c) {
    const bool profile = g_spsa_sha_profile;
    uint64_t final_start_ns = 0;
    size_t final_blocks = 0;
    if (profile) {
        final_start_ns = sha_now_ns();
        g_sha_final_calls.fetch_add(1, std::memory_order_relaxed);
    }

    uint8_t *buf = CTX_BUF(c);

    // Pad message
    uint8_t *p = buf + c->num;
    *p++ = 0x80;

    uint32_t pad_len = 64 - c->num - 1;
    if (pad_len < 8) {
        // Need two blocks
        memset(p, 0, pad_len);
        sha256_process_block(c->h, buf);
        final_blocks++;
        memset(buf, 0, 56);
    } else {
        memset(p, 0, pad_len - 8);
    }

    // Append length in bits (big-endian, 64-bit)
    buf[56] = (uint8_t)(c->Nh >> 24);
    buf[57] = (uint8_t)(c->Nh >> 16);
    buf[58] = (uint8_t)(c->Nh >> 8);
    buf[59] = (uint8_t)(c->Nh);
    buf[60] = (uint8_t)(c->Nl >> 24);
    buf[61] = (uint8_t)(c->Nl >> 16);
    buf[62] = (uint8_t)(c->Nl >> 8);
    buf[63] = (uint8_t)(c->Nl);

    sha256_process_block(c->h, buf);
    final_blocks++;

    // Output hash (big-endian)
    for (int i = 0; i < 8; i++) {
        md[i*4]   = (uint8_t)(c->h[i] >> 24);
        md[i*4+1] = (uint8_t)(c->h[i] >> 16);
        md[i*4+2] = (uint8_t)(c->h[i] >> 8);
        md[i*4+3] = (uint8_t)(c->h[i]);
    }

    uint64_t zeroize_ns = 0;
    if (g_spsa_sha_zeroize) {
        uint64_t zero_start = 0;
        if (profile) {
            zero_start = sha_now_ns();
        }
        memset(c, 0, sizeof(*c));
        if (profile) {
            zeroize_ns = sha_now_ns() - zero_start;
        }
    }

    if (profile) {
        g_sha_final_blocks.fetch_add(static_cast<uint64_t>(final_blocks), std::memory_order_relaxed);
        g_sha_final_zeroize_ns.fetch_add(zeroize_ns, std::memory_order_relaxed);
        g_sha_final_ns.fetch_add(sha_now_ns() - final_start_ns, std::memory_order_relaxed);
    }

    return 1;
}

#undef CTX_BUF
#undef DIRTYBIRD_SHA256_OVERRIDE_SYMBOL

}  // extern "C"
