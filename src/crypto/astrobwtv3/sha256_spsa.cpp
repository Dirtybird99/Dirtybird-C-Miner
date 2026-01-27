/**
 * SHA-256 with On-Demand SPSA Decompression - Implementation
 *
 * Port of Tritonn's SPSA SHA-256 optimization from Rust to C++.
 * Decompresses SA entries on-the-fly during SHA-256 hashing.
 *
 * Key optimizations:
 * - Prefetch-ahead: Prefetch next block's data while processing current block
 * - Double-buffering: Use two block buffers to overlap decompression and hashing
 * - Stamp prefetching: Speculatively prefetch stamp data for stamp references
 */

#include "sha256_spsa.hpp"
#include <cstring>
#include <algorithm>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

#endif

namespace sha256_spsa {

// ============================================================================
// CPU Feature Detection
// ============================================================================

bool avx2_available() {
#if defined(__x86_64__) || defined(_M_X64)
#ifdef _MSC_VER
    int info[4];
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5)) != 0;  // AVX2 bit in EBX
#else
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return (ebx & (1 << 5)) != 0;  // AVX2 bit in EBX
    }
    return false;
#endif
#else
    return false;
#endif
}

bool sha_ni_available() {
#if defined(__x86_64__) || defined(_M_X64)
#ifdef _MSC_VER
    int info[4];
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 29)) != 0;  // SHA bit in EBX
#else
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return (ebx & (1 << 29)) != 0;  // SHA bit in EBX
    }
    return false;
#endif
#else
    return false;
#endif
}

// ============================================================================
// Compiler hints for bounds assumptions
// ============================================================================

#if defined(__GNUC__) || defined(__clang__)
#define ASSUME(cond) do { if (!(cond)) __builtin_unreachable(); } while(0)
#define ALWAYS_INLINE __attribute__((always_inline)) inline
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#elif defined(_MSC_VER)
#define ASSUME(cond) __assume(cond)
#define ALWAYS_INLINE __forceinline
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#else
#define ASSUME(cond) ((void)0)
#define ALWAYS_INLINE inline
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#endif

// ============================================================================
// Prefetch helpers for hiding decompression latency
// ============================================================================
//
// Key insight: SHA-256 block processing takes ~200 cycles (SHA-NI).
// Decompression (fill_block_from_*) takes ~160 cycles.
// Memory latency from L2/L3 cache is ~100-200 cycles.
//
// By prefetching next block's data while processing current block,
// we can hide most of the memory latency, effectively overlapping:
//   - Current block SHA-256 processing (~200 cycles)
//   - Next block data prefetch (memory latency hidden)
//
// Strategy:
// 1. Prefetch first block's data before entering the loop
// 2. Use double-buffering: fill block_b while processing block_a
// 3. Prefetch next+1 block's SA entries during current block processing
// 4. Speculatively prefetch stamp data that may be needed

#if defined(__x86_64__) || defined(_M_X64)

// Entries per SHA-256 block (16 x 4-byte entries = 64 bytes)
static constexpr size_t ENTRIES_PER_BLOCK = 16;

// Prefetch distance: how many blocks ahead to prefetch
// 2 blocks ahead gives prefetch time to complete during current processing
static constexpr size_t PREFETCH_BLOCKS_AHEAD = 2;

// Prefetch reduced_sa entries for a future block
// uint32_t entries: 16 entries = 64 bytes = 1 cache line
static ALWAYS_INLINE void prefetch_reduced_sa_entries(
    const uint32_t* __restrict sa_data,
    size_t start_entry,
    size_t sa_len
) {
    if (LIKELY(start_entry < sa_len)) {
        // Prefetch first cache line (entries 0-15 = 64 bytes)
        _mm_prefetch(reinterpret_cast<const char*>(sa_data + start_entry), _MM_HINT_T0);
        // Prefetch second cache line for potential misalignment
        if (start_entry + 8 < sa_len) {
            _mm_prefetch(reinterpret_cast<const char*>(sa_data + start_entry + 8), _MM_HINT_T0);
        }
    }
}

// Prefetch compressed_sa entries for a future block
// SaEntry is larger (~24 bytes), need more cache line prefetches
static ALWAYS_INLINE void prefetch_compressed_sa_entries(
    const SaEntry* __restrict sa_data,
    size_t start_entry,
    size_t sa_len
) {
    if (LIKELY(start_entry < sa_len)) {
        // SaEntry ~24 bytes, 16 entries = ~384 bytes = 6 cache lines
        _mm_prefetch(reinterpret_cast<const char*>(sa_data + start_entry), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(sa_data + start_entry + 3), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(sa_data + start_entry + 6), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(sa_data + start_entry + 9), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(sa_data + start_entry + 12), _MM_HINT_T0);
        if (start_entry + 15 < sa_len) {
            _mm_prefetch(reinterpret_cast<const char*>(sa_data + start_entry + 15), _MM_HINT_T0);
        }
    }
}

// Speculatively prefetch stamp data for entries that might be stamp references
// Sample entries from the upcoming block and prefetch their stamps
static ALWAYS_INLINE void prefetch_stamps_for_block(
    const uint32_t* __restrict sa_data,
    const spsa::Stamp* __restrict stamp_data,
    size_t start_entry,
    size_t num_entries,
    size_t sa_len,
    size_t stamp_count
) {
    if (UNLIKELY(start_entry >= sa_len || stamp_count == 0)) return;

    // Sample 4 entries (every 4th) to balance prefetch overhead vs hit rate
    // Most entries in a block often reference the same or nearby stamps
    const size_t sample_stride = 4;
    const size_t max_samples = 4;

    for (size_t s = 0; s < max_samples; s++) {
        size_t i = s * sample_stride;
        size_t idx = start_entry + i;
        if (idx >= sa_len || i >= num_entries) break;

        const uint32_t entry = sa_data[idx];
        if (spsa::is_merge_stamp_ref(entry)) {
            uint16_t stamp_id = spsa::decode_merge_stamp_id(entry);
            if (LIKELY(stamp_id < stamp_count)) {
                // Prefetch stamp structure (start_chunk, chunk_count, pos1, pos2)
                _mm_prefetch(reinterpret_cast<const char*>(&stamp_data[stamp_id]), _MM_HINT_T0);
            }
        }
    }
}

#endif // x86_64

// ============================================================================
// Helper: Fill 64-byte block from compressed SA
// ============================================================================

// SA construction guarantees all indices are valid - no bounds checking needed
static ALWAYS_INLINE void fill_block_from_compressed_sa(
    uint8_t* __restrict block,
    const SaEntry* __restrict data,
    size_t start_entry,
    size_t num_entries
) {
    for (size_t i = 0; i < num_entries; i++) {
        const size_t entry_idx = start_entry + i;
        const size_t offset = i * 4;
        int32_t value = data[entry_idx].original_value();
        memcpy(block + offset, &value, sizeof(int32_t));
    }
}

// ============================================================================
// Helper: Fill 64-byte block from reduced SA
// ============================================================================

// SA construction guarantees all indices are valid - no bounds checking needed
static ALWAYS_INLINE void fill_block_from_reduced_sa(
    uint8_t* __restrict block,
    const uint32_t* __restrict sa_data,
    const spsa::Stamp* __restrict stamp_data,
    size_t stamp_count,
    size_t start_entry,
    size_t num_entries
) {
    for (size_t i = 0; i < num_entries; i++) {
        const size_t entry_idx = start_entry + i;
        const size_t offset = i * 4;
        const uint32_t entry = sa_data[entry_idx];

        // Inline merge_entry_to_global_pos logic to avoid function call overhead
        int32_t value;
        if (spsa::is_merge_stamp_ref(entry)) {
            uint16_t stamp_id = spsa::decode_merge_stamp_id(entry);
            uint16_t relative_pos = spsa::decode_merge_relative_pos(entry);
            ASSUME(stamp_id < stamp_count);
            const spsa::Stamp& stamp = stamp_data[stamp_id];
            value = static_cast<int32_t>(stamp.start_chunk * 256 + relative_pos);
        } else {
            value = static_cast<int32_t>(entry);
        }
        memcpy(block + offset, &value, sizeof(int32_t));
    }
}

// Vector wrapper versions for compatibility
static ALWAYS_INLINE void fill_block_from_compressed_sa_vec(
    uint8_t* __restrict block,
    const std::vector<SaEntry>& compressed_sa,
    size_t start_entry,
    size_t num_entries,
    size_t sa_len
) {
    ASSUME(start_entry + num_entries <= sa_len);
    ASSUME(start_entry + num_entries <= compressed_sa.size());
    fill_block_from_compressed_sa(block, compressed_sa.data(), start_entry, num_entries);
}

static ALWAYS_INLINE void fill_block_from_reduced_sa_vec(
    uint8_t* __restrict block,
    const std::vector<uint32_t>& reduced_sa,
    const std::vector<spsa::Stamp>& stamps,
    size_t start_entry,
    size_t num_entries,
    size_t sa_len
) {
    ASSUME(start_entry + num_entries <= sa_len);
    ASSUME(start_entry + num_entries <= reduced_sa.size());
    fill_block_from_reduced_sa(block, reduced_sa.data(), stamps.data(),
                                stamps.size(), start_entry, num_entries);
}

// ============================================================================
// Software SHA-256 Implementation
// ============================================================================

// Right rotate - critical hot path function
static ALWAYS_INLINE uint32_t rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

// SHA-256 functions - all critical hot path
static ALWAYS_INLINE uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

static ALWAYS_INLINE uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

static ALWAYS_INLINE uint32_t sigma0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

static ALWAYS_INLINE uint32_t sigma1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

static ALWAYS_INLINE uint32_t gamma0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

static ALWAYS_INLINE uint32_t gamma1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

// Process one 64-byte block (software)
static void sha256_process_block_soft(uint32_t* state, const uint8_t* block) {
    uint32_t w[64];

    // Prepare message schedule
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)block[i * 4] << 24) |
               ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) |
               ((uint32_t)block[i * 4 + 3]);
    }

    for (int i = 16; i < 64; i++) {
        w[i] = gamma1(w[i - 2]) + w[i - 7] + gamma0(w[i - 15]) + w[i - 16];
    }

    // Initialize working variables
    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];

    // Main loop
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + sigma1(e) + ch(e, f, g) + K[i] + w[i];
        uint32_t t2 = sigma0(a) + maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    // Update state
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

void sha256_compressed_soft(
    const std::vector<SaEntry>& compressed_sa,
    size_t sa_len,
    size_t total_bytes,
    uint8_t* output
) {
    uint32_t state[8];
    memcpy(state, H, sizeof(H));

    uint8_t block[64];
    size_t full_blocks = total_bytes / 64;
    size_t entries_per_block = 16;

    // Process full blocks
    for (size_t block_idx = 0; block_idx < full_blocks; block_idx++) {
        size_t start_entry = block_idx * entries_per_block;
        fill_block_from_compressed_sa_vec(block, compressed_sa, start_entry, entries_per_block, sa_len);
        sha256_process_block_soft(state, block);
    }

    // Handle padding
    size_t remaining_bytes = total_bytes % 64;
    uint64_t orig_len_bits = static_cast<uint64_t>(total_bytes) * 8;

    uint8_t final_blocks[128];
    memset(final_blocks, 0, 128);

    if (remaining_bytes > 0) {
        size_t start_entry = full_blocks * entries_per_block;
        size_t remaining_entries = (remaining_bytes + 3) / 4;
        fill_block_from_compressed_sa_vec(final_blocks, compressed_sa, start_entry, remaining_entries, sa_len);
    }

    // Add padding bit
    final_blocks[remaining_bytes] = 0x80;

    // Determine number of final blocks
    int num_final_blocks = (remaining_bytes >= 56) ? 2 : 1;
    int len_offset = num_final_blocks * 64 - 8;

    // Write length in big-endian
    for (int i = 0; i < 8; i++) {
        final_blocks[len_offset + i] = (orig_len_bits >> (56 - i * 8)) & 0xFF;
    }

    // Process final blocks
    for (int i = 0; i < num_final_blocks; i++) {
        sha256_process_block_soft(state, final_blocks + i * 64);
    }

    // Write output in big-endian
    for (int i = 0; i < 8; i++) {
        output[i * 4] = (state[i] >> 24) & 0xFF;
        output[i * 4 + 1] = (state[i] >> 16) & 0xFF;
        output[i * 4 + 2] = (state[i] >> 8) & 0xFF;
        output[i * 4 + 3] = state[i] & 0xFF;
    }
}

void sha256_reduced_sa_soft(
    const std::vector<uint32_t>& reduced_sa,
    const std::vector<spsa::Stamp>& stamps,
    size_t total_bytes,
    uint8_t* output
) {
    uint32_t state[8];
    memcpy(state, H, sizeof(H));

    size_t sa_len = reduced_sa.size();
    uint8_t block[64];
    size_t full_blocks = total_bytes / 64;
    size_t entries_per_block = 16;

    // Process full blocks
    for (size_t block_idx = 0; block_idx < full_blocks; block_idx++) {
        size_t start_entry = block_idx * entries_per_block;
        fill_block_from_reduced_sa_vec(block, reduced_sa, stamps, start_entry, entries_per_block, sa_len);
        sha256_process_block_soft(state, block);
    }

    // Handle padding
    size_t remaining_bytes = total_bytes % 64;
    uint64_t orig_len_bits = static_cast<uint64_t>(total_bytes) * 8;

    uint8_t final_blocks[128];
    memset(final_blocks, 0, 128);

    if (remaining_bytes > 0) {
        size_t start_entry = full_blocks * entries_per_block;
        size_t remaining_entries = (remaining_bytes + 3) / 4;
        fill_block_from_reduced_sa_vec(final_blocks, reduced_sa, stamps, start_entry, remaining_entries, sa_len);
    }

    final_blocks[remaining_bytes] = 0x80;

    int num_final_blocks = (remaining_bytes >= 56) ? 2 : 1;
    int len_offset = num_final_blocks * 64 - 8;

    for (int i = 0; i < 8; i++) {
        final_blocks[len_offset + i] = (orig_len_bits >> (56 - i * 8)) & 0xFF;
    }

    for (int i = 0; i < num_final_blocks; i++) {
        sha256_process_block_soft(state, final_blocks + i * 64);
    }

    for (int i = 0; i < 8; i++) {
        output[i * 4] = (state[i] >> 24) & 0xFF;
        output[i * 4 + 1] = (state[i] >> 16) & 0xFF;
        output[i * 4 + 2] = (state[i] >> 8) & 0xFF;
        output[i * 4 + 3] = state[i] & 0xFF;
    }
}

// ============================================================================
// SHA-NI Implementation (x86_64 only) with Prefetch-Ahead Optimization
// ============================================================================

#if defined(__x86_64__) || defined(_M_X64)

// Process one 64-byte block using SHA-NI (~200 cycles)
#if defined(__GNUC__) && !defined(_MSC_VER)
__attribute__((target("sha,sse4.1,ssse3")))
#endif
static void sha256_process_block_ni(
    __m128i* state0,
    __m128i* state1,
    const uint8_t* block_ptr,
    __m128i shuf_mask
) {
    __m128i abef_save = *state0;
    __m128i cdgh_save = *state1;

    const __m128i* k_ptr = reinterpret_cast<const __m128i*>(K);

    // Load and byte-swap message words
    __m128i msg0 = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(block_ptr)), shuf_mask);
    __m128i msg1 = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(block_ptr + 16)), shuf_mask);
    __m128i msg2 = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(block_ptr + 32)), shuf_mask);
    __m128i msg3 = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(block_ptr + 48)), shuf_mask);

    __m128i msg_tmp;

    // Rounds 0-3
    msg_tmp = _mm_add_epi32(msg0, _mm_loadu_si128(k_ptr));
    *state1 = _mm_sha256rnds2_epu32(*state1, *state0, msg_tmp);
    msg_tmp = _mm_shuffle_epi32(msg_tmp, 0x0E);
    *state0 = _mm_sha256rnds2_epu32(*state0, *state1, msg_tmp);

    // Rounds 4-7
    msg_tmp = _mm_add_epi32(msg1, _mm_loadu_si128(k_ptr + 1));
    *state1 = _mm_sha256rnds2_epu32(*state1, *state0, msg_tmp);
    msg_tmp = _mm_shuffle_epi32(msg_tmp, 0x0E);
    *state0 = _mm_sha256rnds2_epu32(*state0, *state1, msg_tmp);
    msg0 = _mm_sha256msg1_epu32(msg0, msg1);

    // Rounds 8-11
    msg_tmp = _mm_add_epi32(msg2, _mm_loadu_si128(k_ptr + 2));
    *state1 = _mm_sha256rnds2_epu32(*state1, *state0, msg_tmp);
    msg_tmp = _mm_shuffle_epi32(msg_tmp, 0x0E);
    *state0 = _mm_sha256rnds2_epu32(*state0, *state1, msg_tmp);
    msg1 = _mm_sha256msg1_epu32(msg1, msg2);

    // Rounds 12-15
    msg_tmp = _mm_add_epi32(msg3, _mm_loadu_si128(k_ptr + 3));
    *state1 = _mm_sha256rnds2_epu32(*state1, *state0, msg_tmp);
    msg_tmp = _mm_shuffle_epi32(msg_tmp, 0x0E);
    *state0 = _mm_sha256rnds2_epu32(*state0, *state1, msg_tmp);
    msg0 = _mm_add_epi32(msg0, _mm_alignr_epi8(msg3, msg2, 4));
    msg0 = _mm_sha256msg2_epu32(msg0, msg3);
    msg2 = _mm_sha256msg1_epu32(msg2, msg3);

    // Rounds 16-19
    msg_tmp = _mm_add_epi32(msg0, _mm_loadu_si128(k_ptr + 4));
    *state1 = _mm_sha256rnds2_epu32(*state1, *state0, msg_tmp);
    msg_tmp = _mm_shuffle_epi32(msg_tmp, 0x0E);
    *state0 = _mm_sha256rnds2_epu32(*state0, *state1, msg_tmp);
    msg1 = _mm_add_epi32(msg1, _mm_alignr_epi8(msg0, msg3, 4));
    msg1 = _mm_sha256msg2_epu32(msg1, msg0);
    msg3 = _mm_sha256msg1_epu32(msg3, msg0);

    // Rounds 20-23
    msg_tmp = _mm_add_epi32(msg1, _mm_loadu_si128(k_ptr + 5));
    *state1 = _mm_sha256rnds2_epu32(*state1, *state0, msg_tmp);
    msg_tmp = _mm_shuffle_epi32(msg_tmp, 0x0E);
    *state0 = _mm_sha256rnds2_epu32(*state0, *state1, msg_tmp);
    msg2 = _mm_add_epi32(msg2, _mm_alignr_epi8(msg1, msg0, 4));
    msg2 = _mm_sha256msg2_epu32(msg2, msg1);
    msg0 = _mm_sha256msg1_epu32(msg0, msg1);

    // Rounds 24-27
    msg_tmp = _mm_add_epi32(msg2, _mm_loadu_si128(k_ptr + 6));
    *state1 = _mm_sha256rnds2_epu32(*state1, *state0, msg_tmp);
    msg_tmp = _mm_shuffle_epi32(msg_tmp, 0x0E);
    *state0 = _mm_sha256rnds2_epu32(*state0, *state1, msg_tmp);
    msg3 = _mm_add_epi32(msg3, _mm_alignr_epi8(msg2, msg1, 4));
    msg3 = _mm_sha256msg2_epu32(msg3, msg2);
    msg1 = _mm_sha256msg1_epu32(msg1, msg2);

    // Rounds 28-31
    msg_tmp = _mm_add_epi32(msg3, _mm_loadu_si128(k_ptr + 7));
    *state1 = _mm_sha256rnds2_epu32(*state1, *state0, msg_tmp);
    msg_tmp = _mm_shuffle_epi32(msg_tmp, 0x0E);
    *state0 = _mm_sha256rnds2_epu32(*state0, *state1, msg_tmp);
    msg0 = _mm_add_epi32(msg0, _mm_alignr_epi8(msg3, msg2, 4));
    msg0 = _mm_sha256msg2_epu32(msg0, msg3);
    msg2 = _mm_sha256msg1_epu32(msg2, msg3);

    // Rounds 32-35
    msg_tmp = _mm_add_epi32(msg0, _mm_loadu_si128(k_ptr + 8));
    *state1 = _mm_sha256rnds2_epu32(*state1, *state0, msg_tmp);
    msg_tmp = _mm_shuffle_epi32(msg_tmp, 0x0E);
    *state0 = _mm_sha256rnds2_epu32(*state0, *state1, msg_tmp);
    msg1 = _mm_add_epi32(msg1, _mm_alignr_epi8(msg0, msg3, 4));
    msg1 = _mm_sha256msg2_epu32(msg1, msg0);
    msg3 = _mm_sha256msg1_epu32(msg3, msg0);

    // Rounds 36-39
    msg_tmp = _mm_add_epi32(msg1, _mm_loadu_si128(k_ptr + 9));
    *state1 = _mm_sha256rnds2_epu32(*state1, *state0, msg_tmp);
    msg_tmp = _mm_shuffle_epi32(msg_tmp, 0x0E);
    *state0 = _mm_sha256rnds2_epu32(*state0, *state1, msg_tmp);
    msg2 = _mm_add_epi32(msg2, _mm_alignr_epi8(msg1, msg0, 4));
    msg2 = _mm_sha256msg2_epu32(msg2, msg1);
    msg0 = _mm_sha256msg1_epu32(msg0, msg1);

    // Rounds 40-43
    msg_tmp = _mm_add_epi32(msg2, _mm_loadu_si128(k_ptr + 10));
    *state1 = _mm_sha256rnds2_epu32(*state1, *state0, msg_tmp);
    msg_tmp = _mm_shuffle_epi32(msg_tmp, 0x0E);
    *state0 = _mm_sha256rnds2_epu32(*state0, *state1, msg_tmp);
    msg3 = _mm_add_epi32(msg3, _mm_alignr_epi8(msg2, msg1, 4));
    msg3 = _mm_sha256msg2_epu32(msg3, msg2);
    msg1 = _mm_sha256msg1_epu32(msg1, msg2);

    // Rounds 44-47
    msg_tmp = _mm_add_epi32(msg3, _mm_loadu_si128(k_ptr + 11));
    *state1 = _mm_sha256rnds2_epu32(*state1, *state0, msg_tmp);
    msg_tmp = _mm_shuffle_epi32(msg_tmp, 0x0E);
    *state0 = _mm_sha256rnds2_epu32(*state0, *state1, msg_tmp);
    msg0 = _mm_add_epi32(msg0, _mm_alignr_epi8(msg3, msg2, 4));
    msg0 = _mm_sha256msg2_epu32(msg0, msg3);
    msg2 = _mm_sha256msg1_epu32(msg2, msg3);

    // Rounds 48-51
    msg_tmp = _mm_add_epi32(msg0, _mm_loadu_si128(k_ptr + 12));
    *state1 = _mm_sha256rnds2_epu32(*state1, *state0, msg_tmp);
    msg_tmp = _mm_shuffle_epi32(msg_tmp, 0x0E);
    *state0 = _mm_sha256rnds2_epu32(*state0, *state1, msg_tmp);
    msg1 = _mm_add_epi32(msg1, _mm_alignr_epi8(msg0, msg3, 4));
    msg1 = _mm_sha256msg2_epu32(msg1, msg0);
    msg3 = _mm_sha256msg1_epu32(msg3, msg0);

    // Rounds 52-55
    msg_tmp = _mm_add_epi32(msg1, _mm_loadu_si128(k_ptr + 13));
    *state1 = _mm_sha256rnds2_epu32(*state1, *state0, msg_tmp);
    msg_tmp = _mm_shuffle_epi32(msg_tmp, 0x0E);
    *state0 = _mm_sha256rnds2_epu32(*state0, *state1, msg_tmp);
    msg2 = _mm_add_epi32(msg2, _mm_alignr_epi8(msg1, msg0, 4));
    msg2 = _mm_sha256msg2_epu32(msg2, msg1);

    // Rounds 56-59
    msg_tmp = _mm_add_epi32(msg2, _mm_loadu_si128(k_ptr + 14));
    *state1 = _mm_sha256rnds2_epu32(*state1, *state0, msg_tmp);
    msg_tmp = _mm_shuffle_epi32(msg_tmp, 0x0E);
    *state0 = _mm_sha256rnds2_epu32(*state0, *state1, msg_tmp);
    msg3 = _mm_add_epi32(msg3, _mm_alignr_epi8(msg2, msg1, 4));
    msg3 = _mm_sha256msg2_epu32(msg3, msg2);

    // Rounds 60-63
    msg_tmp = _mm_add_epi32(msg3, _mm_loadu_si128(k_ptr + 15));
    *state1 = _mm_sha256rnds2_epu32(*state1, *state0, msg_tmp);
    msg_tmp = _mm_shuffle_epi32(msg_tmp, 0x0E);
    *state0 = _mm_sha256rnds2_epu32(*state0, *state1, msg_tmp);

    // Add to previous state
    *state0 = _mm_add_epi32(*state0, abef_save);
    *state1 = _mm_add_epi32(*state1, cdgh_save);
}

/**
 * SHA-256 with compressed SA using prefetch-ahead optimization
 *
 * Optimization strategy:
 * 1. Prefetch first block's data before loop
 * 2. Use double-buffering: process block_a while filling block_b
 * 3. Prefetch 2 blocks ahead to hide memory latency
 */
#if defined(__GNUC__) && !defined(_MSC_VER)
__attribute__((target("sha,sse4.1,ssse3")))
#endif
void sha256_compressed_ni(
    const std::vector<SaEntry>& compressed_sa,
    size_t sa_len,
    size_t total_bytes,
    uint8_t* output
) {
    // Byte swap mask for SHA-256 (big-endian words)
    __m128i shuf_mask = _mm_set_epi64x(0x0c0d0e0f08090a0bLL, 0x0405060700010203LL);

    // Initialize state (ABCD, EFGH format)
    __m128i state0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(H));
    __m128i state1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(H + 4));

    // Rearrange to SHA-NI format
    __m128i tmp = _mm_shuffle_epi32(state0, 0xB1);
    __m128i tmp2 = _mm_shuffle_epi32(state1, 0x1B);
    state0 = _mm_alignr_epi8(tmp, tmp2, 8);
    state1 = _mm_blend_epi16(tmp2, tmp, 0xF0);

    // Double-buffered block storage (cache-line aligned)
    alignas(64) uint8_t block_a[64];
    alignas(64) uint8_t block_b[64];
    uint8_t* current_block = block_a;
    uint8_t* next_block = block_b;

    const size_t full_blocks = total_bytes / 64;
    const SaEntry* sa_data = compressed_sa.data();

    // Prefetch first 2 blocks' data before entering loop
    if (full_blocks > 0) {
        prefetch_compressed_sa_entries(sa_data, 0, sa_len);
    }
    if (full_blocks > 1) {
        prefetch_compressed_sa_entries(sa_data, ENTRIES_PER_BLOCK, sa_len);
    }

    // Fill first block if we have any
    if (full_blocks > 0) {
        fill_block_from_compressed_sa(current_block, sa_data, 0, ENTRIES_PER_BLOCK);
    }

    // Process full blocks with prefetch-ahead and double-buffering
    for (size_t block_idx = 0; block_idx < full_blocks; block_idx++) {
        // Prefetch data for block_idx + 2 (2 blocks ahead)
        size_t prefetch_block = block_idx + PREFETCH_BLOCKS_AHEAD;
        if (prefetch_block < full_blocks) {
            size_t prefetch_entry = prefetch_block * ENTRIES_PER_BLOCK;
            prefetch_compressed_sa_entries(sa_data, prefetch_entry, sa_len);
        }

        // Fill next block while we process current
        // (next iteration will use this as current_block)
        if (block_idx + 1 < full_blocks) {
            size_t next_start = (block_idx + 1) * ENTRIES_PER_BLOCK;
            fill_block_from_compressed_sa(next_block, sa_data, next_start, ENTRIES_PER_BLOCK);
        }

        // Process current block (SHA-256 ~200 cycles)
        sha256_process_block_ni(&state0, &state1, current_block, shuf_mask);

        // Swap buffers for next iteration
        uint8_t* temp = current_block;
        current_block = next_block;
        next_block = temp;
    }

    // Handle padding
    size_t remaining_bytes = total_bytes % 64;
    uint64_t orig_len_bits = static_cast<uint64_t>(total_bytes) * 8;

    alignas(64) uint8_t final_blocks[128];
    memset(final_blocks, 0, 128);

    if (remaining_bytes > 0) {
        size_t start_entry = full_blocks * ENTRIES_PER_BLOCK;
        size_t remaining_entries = (remaining_bytes + 3) / 4;
        fill_block_from_compressed_sa(final_blocks, sa_data, start_entry, remaining_entries);
    }

    final_blocks[remaining_bytes] = 0x80;

    int num_final_blocks = (remaining_bytes >= 56) ? 2 : 1;
    int len_offset = num_final_blocks * 64 - 8;

    // Write length in big-endian
    for (int i = 0; i < 8; i++) {
        final_blocks[len_offset + i] = (orig_len_bits >> (56 - i * 8)) & 0xFF;
    }

    // Process final blocks
    for (int i = 0; i < num_final_blocks; i++) {
        sha256_process_block_ni(&state0, &state1, final_blocks + i * 64, shuf_mask);
    }

    // Extract final hash
    tmp = _mm_shuffle_epi32(state0, 0x1B);
    state1 = _mm_shuffle_epi32(state1, 0xB1);
    state0 = _mm_blend_epi16(tmp, state1, 0xF0);
    state1 = _mm_alignr_epi8(state1, tmp, 8);

    state0 = _mm_shuffle_epi8(state0, shuf_mask);
    state1 = _mm_shuffle_epi8(state1, shuf_mask);

    _mm_storeu_si128(reinterpret_cast<__m128i*>(output), state0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(output + 16), state1);
}

/**
 * SHA-256 with reduced SA using prefetch-ahead optimization
 *
 * This is the more commonly used path with SPSA.
 *
 * Optimization strategy:
 * 1. Prefetch first 2 blocks' SA entries before loop
 * 2. Use double-buffering: process block_a while filling block_b
 * 3. Prefetch SA entries 2 blocks ahead during processing
 * 4. Speculatively prefetch stamp data for upcoming blocks
 */
#if defined(__GNUC__) && !defined(_MSC_VER)
__attribute__((target("sha,sse4.1,ssse3")))
#endif
void sha256_reduced_sa_ni(
    const std::vector<uint32_t>& reduced_sa,
    const std::vector<spsa::Stamp>& stamps,
    size_t total_bytes,
    uint8_t* output
) {
    __m128i shuf_mask = _mm_set_epi64x(0x0c0d0e0f08090a0bLL, 0x0405060700010203LL);

    __m128i state0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(H));
    __m128i state1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(H + 4));

    __m128i tmp = _mm_shuffle_epi32(state0, 0xB1);
    __m128i tmp2 = _mm_shuffle_epi32(state1, 0x1B);
    state0 = _mm_alignr_epi8(tmp, tmp2, 8);
    state1 = _mm_blend_epi16(tmp2, tmp, 0xF0);

    const size_t sa_len = reduced_sa.size();
    const size_t full_blocks = total_bytes / 64;
    const uint32_t* sa_data = reduced_sa.data();
    const spsa::Stamp* stamp_data = stamps.data();
    const size_t stamp_count = stamps.size();

    // Double-buffered block storage (cache-line aligned)
    alignas(64) uint8_t block_a[64];
    alignas(64) uint8_t block_b[64];
    uint8_t* current_block = block_a;
    uint8_t* next_block = block_b;

    // Prefetch first 2 blocks' SA entries and stamps before entering loop
    if (full_blocks > 0) {
        prefetch_reduced_sa_entries(sa_data, 0, sa_len);
        prefetch_stamps_for_block(sa_data, stamp_data, 0, ENTRIES_PER_BLOCK, sa_len, stamp_count);
    }
    if (full_blocks > 1) {
        prefetch_reduced_sa_entries(sa_data, ENTRIES_PER_BLOCK, sa_len);
        prefetch_stamps_for_block(sa_data, stamp_data, ENTRIES_PER_BLOCK, ENTRIES_PER_BLOCK, sa_len, stamp_count);
    }

    // Fill first block if we have any
    if (full_blocks > 0) {
        fill_block_from_reduced_sa(current_block, sa_data, stamp_data, stamp_count, 0, ENTRIES_PER_BLOCK);
    }

    // Process full blocks with prefetch-ahead and double-buffering
    for (size_t block_idx = 0; block_idx < full_blocks; block_idx++) {
        // Prefetch SA entries and stamps for block_idx + 2 (2 blocks ahead)
        size_t prefetch_block = block_idx + PREFETCH_BLOCKS_AHEAD;
        if (prefetch_block < full_blocks) {
            size_t prefetch_entry = prefetch_block * ENTRIES_PER_BLOCK;
            prefetch_reduced_sa_entries(sa_data, prefetch_entry, sa_len);
            prefetch_stamps_for_block(sa_data, stamp_data, prefetch_entry, ENTRIES_PER_BLOCK, sa_len, stamp_count);
        }

        // Fill next block while we process current
        // (next iteration will use this as current_block)
        if (block_idx + 1 < full_blocks) {
            size_t next_start = (block_idx + 1) * ENTRIES_PER_BLOCK;
            fill_block_from_reduced_sa(next_block, sa_data, stamp_data, stamp_count, next_start, ENTRIES_PER_BLOCK);
        }

        // Process current block (SHA-256 ~200 cycles)
        sha256_process_block_ni(&state0, &state1, current_block, shuf_mask);

        // Swap buffers for next iteration
        uint8_t* temp = current_block;
        current_block = next_block;
        next_block = temp;
    }

    // Handle padding
    size_t remaining_bytes = total_bytes % 64;
    uint64_t orig_len_bits = static_cast<uint64_t>(total_bytes) * 8;

    alignas(64) uint8_t final_blocks[128];
    memset(final_blocks, 0, 128);

    if (remaining_bytes > 0) {
        size_t start_entry = full_blocks * ENTRIES_PER_BLOCK;
        size_t remaining_entries = (remaining_bytes + 3) / 4;
        fill_block_from_reduced_sa(final_blocks, sa_data, stamp_data, stamp_count, start_entry, remaining_entries);
    }

    final_blocks[remaining_bytes] = 0x80;

    int num_final_blocks = (remaining_bytes >= 56) ? 2 : 1;
    int len_offset = num_final_blocks * 64 - 8;

    for (int i = 0; i < 8; i++) {
        final_blocks[len_offset + i] = (orig_len_bits >> (56 - i * 8)) & 0xFF;
    }

    for (int i = 0; i < num_final_blocks; i++) {
        sha256_process_block_ni(&state0, &state1, final_blocks + i * 64, shuf_mask);
    }

    tmp = _mm_shuffle_epi32(state0, 0x1B);
    state1 = _mm_shuffle_epi32(state1, 0xB1);
    state0 = _mm_blend_epi16(tmp, state1, 0xF0);
    state1 = _mm_alignr_epi8(state1, tmp, 8);

    state0 = _mm_shuffle_epi8(state0, shuf_mask);
    state1 = _mm_shuffle_epi8(state1, shuf_mask);

    _mm_storeu_si128(reinterpret_cast<__m128i*>(output), state0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(output + 16), state1);
}

#endif  // x86_64

// ============================================================================
// Public API
// ============================================================================

void sha256_compressed(
    const std::vector<SaEntry>& compressed_sa,
    const StampStorage& stamps,
    size_t sa_len,
    uint8_t* output
) {
    size_t total_bytes = sa_len * 4;

#if defined(__x86_64__) || defined(_M_X64)
    if (sha_ni_available()) {
        sha256_compressed_ni(compressed_sa, sa_len, total_bytes, output);
        return;
    }
#endif

    sha256_compressed_soft(compressed_sa, sa_len, total_bytes, output);
}

void sha256_reduced_sa(
    const std::vector<uint32_t>& reduced_sa,
    const std::vector<spsa::Stamp>& stamps,
    uint8_t* output
) {
    size_t total_bytes = reduced_sa.size() * 4;

#if defined(__x86_64__) || defined(_M_X64)
    if (sha_ni_available()) {
        sha256_reduced_sa_ni(reduced_sa, stamps, total_bytes, output);
        return;
    }
#endif

    sha256_reduced_sa_soft(reduced_sa, stamps, total_bytes, output);
}

} // namespace sha256_spsa
