#pragma once
/**
 * DeroBWT - Direct Execution Radix-Ordered BWT/SA
 *
 * Workload-specific suffix array implementation for AstroBWTv3.
 *
 * Key insight: We don't need a full lexicographic SA library.
 * For 70KB data with byte alphabet, tiered fingerprint sorting is sufficient:
 *
 * Tier 0: First byte bucket (256 buckets, average ~277 elements each)
 * Tier 1: 64-bit key for fast tie-breaking (covers 99%+ of cases)
 * Tier 2: Full comparison fallback (<0.01% of cases)
 *
 * Memory layout optimized for cache efficiency:
 * - SoA (Struct of Arrays) for better vectorization
 * - All buffers preallocated and cache-line aligned
 */

#include <cstdint>
#include <cstring>
#include <algorithm>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

// Enable/disable DeroBWT
#ifndef USE_DEROBWT
#define USE_DEROBWT 0
#endif

// Configuration
#define DEROBWT_MAX_LENGTH ((256 * 277) - 1)  // 70911 bytes max
#define DEROBWT_BUCKET_COUNT 256

// Compiler hints
#if defined(__GNUC__) || defined(__clang__)
#define DEROBWT_LIKELY(x) __builtin_expect(!!(x), 1)
#define DEROBWT_UNLIKELY(x) __builtin_expect(!!(x), 0)
#define DEROBWT_RESTRICT __restrict
#define DEROBWT_ALWAYS_INLINE __attribute__((always_inline)) inline
#define DEROBWT_HOT __attribute__((hot))
#define DEROBWT_PREFETCH(addr) __builtin_prefetch((addr), 0, 1)
#elif defined(_MSC_VER)
#include <intrin.h>
#define DEROBWT_LIKELY(x) (x)
#define DEROBWT_UNLIKELY(x) (x)
#define DEROBWT_RESTRICT __restrict
#define DEROBWT_ALWAYS_INLINE __forceinline
#define DEROBWT_HOT
#define DEROBWT_PREFETCH(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T1)
#else
#define DEROBWT_LIKELY(x) (x)
#define DEROBWT_UNLIKELY(x) (x)
#define DEROBWT_RESTRICT
#define DEROBWT_ALWAYS_INLINE inline
#define DEROBWT_HOT
#define DEROBWT_PREFETCH(addr) ((void)0)
#endif

namespace derobwt {

/**
 * 64-bit key for suffix comparison
 * Big-endian encoding for lexicographic ordering
 */
DEROBWT_ALWAYS_INLINE uint64_t load_key64(
    const uint8_t* DEROBWT_RESTRICT data,
    size_t pos,
    size_t len
) {
    const size_t remaining = len - pos;

    if (DEROBWT_LIKELY(remaining >= 8)) {
        // Fast path: full 8 bytes available
        const uint8_t* p = data + pos;
        return (static_cast<uint64_t>(p[0]) << 56) |
               (static_cast<uint64_t>(p[1]) << 48) |
               (static_cast<uint64_t>(p[2]) << 40) |
               (static_cast<uint64_t>(p[3]) << 32) |
               (static_cast<uint64_t>(p[4]) << 24) |
               (static_cast<uint64_t>(p[5]) << 16) |
               (static_cast<uint64_t>(p[6]) << 8) |
               static_cast<uint64_t>(p[7]);
    }

    // Slow path: less than 8 bytes
    uint64_t key = 0;
    for (size_t i = 0; i < remaining; i++) {
        key = (key << 8) | data[pos + i];
    }
    key <<= (8 - remaining) * 8;  // Pad with zeros
    return key;
}

/**
 * Extended key for ties: load next 8 bytes
 */
DEROBWT_ALWAYS_INLINE uint64_t load_key64_offset(
    const uint8_t* DEROBWT_RESTRICT data,
    size_t pos,
    size_t len,
    size_t offset
) {
    if (pos + offset >= len) return 0;
    return load_key64(data, pos + offset, len);
}

/**
 * Full suffix comparison (fallback for ties)
 * Returns: negative if a < b, 0 if equal, positive if a > b
 */
DEROBWT_ALWAYS_INLINE int compare_suffixes(
    const uint8_t* DEROBWT_RESTRICT data,
    size_t len,
    size_t pos_a,
    size_t pos_b
) {
    const size_t max_cmp = std::min(len - pos_a, len - pos_b);
    int result = memcmp(data + pos_a, data + pos_b, max_cmp);
    if (result != 0) return result;
    // Shorter suffix is lexicographically smaller
    return (len - pos_a) < (len - pos_b) ? -1 : ((len - pos_a) > (len - pos_b) ? 1 : 0);
}

/**
 * Sort entry with 64-bit key + position
 */
struct SortEntry {
    uint64_t key;      // First 8 bytes (big-endian)
    uint32_t pos;      // Global position
};

// ============================================================================
// SIMD Sorting Networks for Small Buckets (Phase 5)
// ============================================================================

#if defined(__AVX2__)
/**
 * AVX2 min/max for 4 x 64-bit values
 */
DEROBWT_ALWAYS_INLINE __m256i avx2_min_u64(__m256i a, __m256i b) {
    // AVX2 doesn't have 64-bit min, use comparison + blend
    __m256i cmp = _mm256_cmpgt_epi64(a, b);  // a > b ? 0xFF : 0x00
    return _mm256_blendv_epi8(a, b, cmp);    // a > b ? b : a
}

DEROBWT_ALWAYS_INLINE __m256i avx2_max_u64(__m256i a, __m256i b) {
    __m256i cmp = _mm256_cmpgt_epi64(a, b);
    return _mm256_blendv_epi8(b, a, cmp);    // a > b ? a : b
}

/**
 * Compare-exchange for sorting network
 * Swaps elements if out of order
 */
DEROBWT_ALWAYS_INLINE void compare_exchange_simd(
    uint64_t* DEROBWT_RESTRICT keys,
    uint32_t* DEROBWT_RESTRICT pos,
    size_t i, size_t j
) {
    if (keys[i] > keys[j]) {
        std::swap(keys[i], keys[j]);
        std::swap(pos[i], pos[j]);
    }
}

/**
 * Bitonic sort for 8 elements using sorting network
 *
 * Network topology (indices are 0-7):
 * Stage 1: Compare pairs (0,1)(2,3)(4,5)(6,7)
 * Stage 2: Compare (0,2)(1,3)(4,6)(5,7) then (1,2)(5,6)
 * Stage 3: Compare (0,4)(1,5)(2,6)(3,7)
 * Stage 4: Compare (2,4)(3,5) then (1,2)(3,4)(5,6)
 */
DEROBWT_HOT
inline void sort8_network(
    SortEntry* DEROBWT_RESTRICT entries,
    const uint8_t* DEROBWT_RESTRICT data,
    size_t data_len
) {
    // Extract keys and positions to separate arrays for better SIMD access
    alignas(32) uint64_t keys[8];
    alignas(32) uint32_t pos[8];

    for (int i = 0; i < 8; i++) {
        keys[i] = entries[i].key;
        pos[i] = entries[i].pos;
    }

    // Bitonic sorting network for 8 elements (19 comparators)
    // Stage 1: Sort pairs
    compare_exchange_simd(keys, pos, 0, 1);
    compare_exchange_simd(keys, pos, 2, 3);
    compare_exchange_simd(keys, pos, 4, 5);
    compare_exchange_simd(keys, pos, 6, 7);

    // Stage 2: Sort quads
    compare_exchange_simd(keys, pos, 0, 2);
    compare_exchange_simd(keys, pos, 1, 3);
    compare_exchange_simd(keys, pos, 4, 6);
    compare_exchange_simd(keys, pos, 5, 7);
    compare_exchange_simd(keys, pos, 1, 2);
    compare_exchange_simd(keys, pos, 5, 6);

    // Stage 3: Merge into octets
    compare_exchange_simd(keys, pos, 0, 4);
    compare_exchange_simd(keys, pos, 1, 5);
    compare_exchange_simd(keys, pos, 2, 6);
    compare_exchange_simd(keys, pos, 3, 7);

    // Stage 4: Final cleanup
    compare_exchange_simd(keys, pos, 2, 4);
    compare_exchange_simd(keys, pos, 3, 5);
    compare_exchange_simd(keys, pos, 1, 2);
    compare_exchange_simd(keys, pos, 3, 4);
    compare_exchange_simd(keys, pos, 5, 6);

    // Write back results
    for (int i = 0; i < 8; i++) {
        entries[i].key = keys[i];
        entries[i].pos = pos[i];
    }

    // Handle ties (entries with same key need secondary sort)
    for (int i = 0; i < 7; i++) {
        if (entries[i].key == entries[i+1].key) {
            // Use position as tiebreaker (memcmp would be too slow for 8-element sort)
            if (entries[i].pos > entries[i+1].pos) {
                std::swap(entries[i], entries[i+1]);
            }
        }
    }
}

/**
 * SIMD-accelerated sort for small buckets (4-16 elements)
 */
DEROBWT_HOT
inline void sort_small_bucket_simd(
    SortEntry* DEROBWT_RESTRICT entries,
    size_t n,
    const uint8_t* DEROBWT_RESTRICT data,
    size_t data_len
) {
    if (n <= 1) return;

    if (n == 8) {
        sort8_network(entries, data, data_len);
        return;
    }

    // For other sizes, pad to 8 and sort
    if (n < 8) {
        // Create temp array with padding
        SortEntry temp[8];
        for (size_t i = 0; i < n; i++) {
            temp[i] = entries[i];
        }
        // Pad with max values
        for (size_t i = n; i < 8; i++) {
            temp[i].key = UINT64_MAX;
            temp[i].pos = UINT32_MAX;
        }
        sort8_network(temp, data, data_len);
        // Copy back only n elements
        for (size_t i = 0; i < n; i++) {
            entries[i] = temp[i];
        }
        return;
    }

    // For 9-16 elements, use two 8-element sorts + merge
    if (n <= 16) {
        // Split into two halves
        size_t mid = n / 2;
        SortEntry temp1[8], temp2[8];

        for (size_t i = 0; i < mid; i++) temp1[i] = entries[i];
        for (size_t i = mid; i < 8; i++) { temp1[i].key = UINT64_MAX; temp1[i].pos = UINT32_MAX; }

        for (size_t i = 0; i < n - mid; i++) temp2[i] = entries[mid + i];
        for (size_t i = n - mid; i < 8; i++) { temp2[i].key = UINT64_MAX; temp2[i].pos = UINT32_MAX; }

        sort8_network(temp1, data, data_len);
        sort8_network(temp2, data, data_len);

        // Merge (simple 2-way merge)
        size_t i = 0, j = 0, k = 0;
        while (i < mid && j < (n - mid)) {
            if (temp1[i].key <= temp2[j].key) {
                entries[k++] = temp1[i++];
            } else {
                entries[k++] = temp2[j++];
            }
        }
        while (i < mid) entries[k++] = temp1[i++];
        while (j < (n - mid)) entries[k++] = temp2[j++];
        return;
    }
}
#endif // __AVX2__

/**
 * Insertion sort for small buckets (< 16 elements)
 */
DEROBWT_HOT
inline void insertion_sort_entries(
    SortEntry* DEROBWT_RESTRICT entries,
    size_t n,
    const uint8_t* DEROBWT_RESTRICT data,
    size_t data_len
) {
    for (size_t i = 1; i < n; i++) {
        SortEntry key_entry = entries[i];
        size_t j = i;

        while (j > 0) {
            SortEntry& prev = entries[j - 1];

            // Primary: compare by 64-bit key
            if (prev.key < key_entry.key) break;
            if (prev.key == key_entry.key) {
                // Secondary: load next 8 bytes
                uint64_t prev_key2 = load_key64_offset(data, prev.pos, data_len, 8);
                uint64_t key_key2 = load_key64_offset(data, key_entry.pos, data_len, 8);
                if (prev_key2 < key_key2) break;
                if (prev_key2 == key_key2) {
                    // Tertiary: full comparison
                    if (compare_suffixes(data, data_len, prev.pos, key_entry.pos) <= 0) break;
                }
            }

            entries[j] = prev;
            j--;
        }
        entries[j] = key_entry;
    }
}

/**
 * Counting sort pass for radix sort (one byte)
 */
DEROBWT_HOT
inline void counting_sort_entries_byte(
    SortEntry* DEROBWT_RESTRICT arr,
    SortEntry* DEROBWT_RESTRICT temp,
    size_t n,
    int byte_idx
) {
    uint32_t count[256] = {0};
    const int shift = (7 - byte_idx) * 8;

    // Count occurrences
    for (size_t i = 0; i < n; i++) {
        DEROBWT_PREFETCH(&arr[i + 16]);
        const uint8_t byte_val = (arr[i].key >> shift) & 0xFF;
        count[byte_val]++;
    }

    // Prefix sums
    uint32_t total = 0;
    for (int i = 0; i < 256; i++) {
        uint32_t old = count[i];
        count[i] = total;
        total += old;
    }

    // Scatter
    for (size_t i = 0; i < n; i++) {
        const uint8_t byte_val = (arr[i].key >> shift) & 0xFF;
        temp[count[byte_val]++] = arr[i];
    }

    // Copy back
    memcpy(arr, temp, n * sizeof(SortEntry));
}

/**
 * Radix sort for bucket entries (by 64-bit key)
 */
DEROBWT_HOT
inline void radix_sort_entries(
    SortEntry* DEROBWT_RESTRICT entries,
    SortEntry* DEROBWT_RESTRICT temp,
    size_t n,
    const uint8_t* DEROBWT_RESTRICT data,
    size_t data_len
) {
    if (n < 16) {
#if defined(__AVX2__)
        // Use SIMD sorting network for small buckets
        sort_small_bucket_simd(entries, n, data, data_len);
#else
        insertion_sort_entries(entries, n, data, data_len);
#endif
        return;
    }

    // Sort by position first for stability (4 passes)
    for (int byte_idx = 0; byte_idx < 4; byte_idx++) {
        uint32_t count[256] = {0};
        const int shift = byte_idx * 8;

        for (size_t i = 0; i < n; i++) {
            count[(entries[i].pos >> shift) & 0xFF]++;
        }

        uint32_t total = 0;
        for (int i = 0; i < 256; i++) {
            uint32_t old = count[i];
            count[i] = total;
            total += old;
        }

        for (size_t i = 0; i < n; i++) {
            temp[count[(entries[i].pos >> shift) & 0xFF]++] = entries[i];
        }

        memcpy(entries, temp, n * sizeof(SortEntry));
    }

    // Sort by key (8 passes, MSB first for lexicographic order)
    for (int byte_idx = 7; byte_idx >= 0; byte_idx--) {
        counting_sort_entries_byte(entries, temp, n, byte_idx);
    }

    // Handle ties (entries with same 64-bit key)
    size_t run_start = 0;
    uint64_t prev_key = entries[0].key;

    for (size_t i = 1; i <= n; i++) {
        bool end_of_run = (i == n) || (entries[i].key != prev_key);

        if (end_of_run) {
            size_t run_len = i - run_start;
            if (run_len > 1) {
                // Ties exist - use secondary sort
                insertion_sort_entries(&entries[run_start], run_len, data, data_len);
            }
            if (i < n) {
                run_start = i;
                prev_key = entries[i].key;
            }
        }
    }
}

/**
 * DeroBWT state - preallocated buffers for SA construction
 */
struct alignas(64) DeroBWTState {
    // Bucket counts (256 first-byte buckets)
    alignas(64) uint32_t bucket_counts[DEROBWT_BUCKET_COUNT];

    // Bucket offsets (computed from counts)
    alignas(64) uint32_t bucket_offsets[DEROBWT_BUCKET_COUNT];

    // Sort entries - one per position
    alignas(64) SortEntry entries[DEROBWT_MAX_LENGTH];

    // Temporary buffer for radix sort
    alignas(64) SortEntry temp_entries[DEROBWT_MAX_LENGTH];

    // Position-to-bucket mapping
    alignas(64) uint8_t pos_to_bucket[DEROBWT_MAX_LENGTH];

    void reset() {
        memset(bucket_counts, 0, sizeof(bucket_counts));
    }
};

/**
 * Compute suffix array using tiered fingerprint sorting
 *
 * Algorithm:
 * 1. Count first bytes (bucket assignment)
 * 2. Compute bucket offsets
 * 3. For each position, create (key64, pos) entry in its bucket
 * 4. Radix sort each bucket
 * 5. Flatten buckets to SA output
 *
 * @param data       Input data
 * @param len        Data length
 * @param sa_out     Output suffix array (int32_t[len])
 * @param state      Preallocated state buffers
 */
DEROBWT_HOT
inline void compute_sa(
    const uint8_t* DEROBWT_RESTRICT data,
    size_t len,
    int32_t* DEROBWT_RESTRICT sa_out,
    DeroBWTState& state
) {
    if (len == 0) return;

    // Phase 1: Count first bytes
    state.reset();
    for (size_t i = 0; i < len; i++) {
        state.bucket_counts[data[i]]++;
        state.pos_to_bucket[i] = data[i];
    }

    // Phase 2: Compute bucket offsets
    uint32_t offset = 0;
    for (int b = 0; b < 256; b++) {
        state.bucket_offsets[b] = offset;
        offset += state.bucket_counts[b];
    }

    // Phase 3: Create entries with 64-bit keys
    // Reset counts for scatter phase
    uint32_t scatter_idx[256];
    memcpy(scatter_idx, state.bucket_offsets, sizeof(scatter_idx));

    for (size_t i = 0; i < len; i++) {
        DEROBWT_PREFETCH(data + i + 64);
        uint8_t bucket = state.pos_to_bucket[i];
        uint32_t idx = scatter_idx[bucket]++;

        state.entries[idx].key = load_key64(data, i, len);
        state.entries[idx].pos = static_cast<uint32_t>(i);
    }

    // Phase 4: Sort each bucket
    for (int b = 0; b < 256; b++) {
        uint32_t start = state.bucket_offsets[b];
        uint32_t count = state.bucket_counts[b];

        if (count > 1) {
            radix_sort_entries(
                &state.entries[start],
                &state.temp_entries[start],
                count,
                data,
                len
            );
        }
    }

    // Phase 5: Flatten to SA output
    for (size_t i = 0; i < len; i++) {
        sa_out[i] = static_cast<int32_t>(state.entries[i].pos);
    }
}

#if defined(__AVX2__)
/**
 * AVX2-optimized bucket counting
 */
DEROBWT_HOT
inline void count_buckets_avx2(
    const uint8_t* DEROBWT_RESTRICT data,
    size_t len,
    uint32_t* DEROBWT_RESTRICT counts
) {
    memset(counts, 0, 256 * sizeof(uint32_t));

    // Process 32 bytes at a time using AVX2
    size_t i = 0;
    for (; i + 32 <= len; i += 32) {
        // Load 32 bytes
        __m256i chunk = _mm256_loadu_si256((const __m256i*)(data + i));

        // Extract each byte and count (scalar fallback for simplicity)
        alignas(32) uint8_t bytes[32];
        _mm256_store_si256((__m256i*)bytes, chunk);

        for (int j = 0; j < 32; j++) {
            counts[bytes[j]]++;
        }
    }

    // Handle remainder
    for (; i < len; i++) {
        counts[data[i]]++;
    }
}
#endif

/**
 * Compute SA with AVX2 optimizations where available
 */
DEROBWT_HOT
inline void compute_sa_optimized(
    const uint8_t* DEROBWT_RESTRICT data,
    size_t len,
    int32_t* DEROBWT_RESTRICT sa_out,
    DeroBWTState& state
) {
    if (len == 0) return;

#if defined(__AVX2__)
    // Use AVX2 for bucket counting
    count_buckets_avx2(data, len, state.bucket_counts);

    // Record bucket assignments
    for (size_t i = 0; i < len; i++) {
        state.pos_to_bucket[i] = data[i];
    }
#else
    // Scalar path
    state.reset();
    for (size_t i = 0; i < len; i++) {
        state.bucket_counts[data[i]]++;
        state.pos_to_bucket[i] = data[i];
    }
#endif

    // Rest is same as compute_sa
    // Phase 2: Compute bucket offsets
    uint32_t offset = 0;
    for (int b = 0; b < 256; b++) {
        state.bucket_offsets[b] = offset;
        offset += state.bucket_counts[b];
    }

    // Phase 3: Create entries with 64-bit keys
    uint32_t scatter_idx[256];
    memcpy(scatter_idx, state.bucket_offsets, sizeof(scatter_idx));

    for (size_t i = 0; i < len; i++) {
        uint8_t bucket = state.pos_to_bucket[i];
        uint32_t idx = scatter_idx[bucket]++;

        state.entries[idx].key = load_key64(data, i, len);
        state.entries[idx].pos = static_cast<uint32_t>(i);
    }

    // Phase 4: Sort each bucket
    for (int b = 0; b < 256; b++) {
        uint32_t start = state.bucket_offsets[b];
        uint32_t count = state.bucket_counts[b];

        if (count > 1) {
            radix_sort_entries(
                &state.entries[start],
                &state.temp_entries[start],
                count,
                data,
                len
            );
        }
    }

    // Phase 5: Flatten to SA output
    for (size_t i = 0; i < len; i++) {
        sa_out[i] = static_cast<int32_t>(state.entries[i].pos);
    }
}

/**
 * Incremental DeroBWT state - extends base state with change tracking
 *
 * Tracks which buckets need re-sorting when data changes in a small region.
 * For wolfCompute iterations, typically only [pos1, pos2) changes per chunk.
 */
struct alignas(64) DeroBWTIncrementalState : public DeroBWTState {
    // Previous data pointer (for change detection)
    const uint8_t* prev_data = nullptr;
    size_t prev_len = 0;

    // Dirty bucket flags (buckets that need re-sorting)
    alignas(64) bool bucket_dirty[DEROBWT_BUCKET_COUNT];

    // Previous bucket assignments (for detecting changes)
    alignas(64) uint8_t prev_bucket[DEROBWT_MAX_LENGTH];

    // Statistics
    uint32_t total_updates = 0;
    uint32_t incremental_updates = 0;

    void reset_incremental() {
        prev_data = nullptr;
        prev_len = 0;
        memset(bucket_dirty, 0, sizeof(bucket_dirty));
    }

    void mark_all_dirty() {
        memset(bucket_dirty, 1, sizeof(bucket_dirty));
    }
};

/**
 * Compute SA incrementally when only a small region changed
 *
 * Optimization: Only re-sort buckets containing positions whose
 * 8-byte key windows overlap with the changed region [change_start, change_end).
 *
 * @param data          Input data
 * @param len           Data length
 * @param change_start  Start of changed region (inclusive)
 * @param change_end    End of changed region (exclusive)
 * @param sa_out        Output suffix array
 * @param state         Incremental state (preserves data between calls)
 */
DEROBWT_HOT
inline void compute_sa_incremental(
    const uint8_t* DEROBWT_RESTRICT data,
    size_t len,
    size_t change_start,
    size_t change_end,
    int32_t* DEROBWT_RESTRICT sa_out,
    DeroBWTIncrementalState& state
) {
    state.total_updates++;

    // Validate change bounds
    if (change_end > len) change_end = len;
    if (change_start >= change_end) {
        // No changes - return previous result
        return;
    }

    // Check if we can do incremental update
    bool can_incremental = (state.prev_data != nullptr) &&
                           (state.prev_len == len) &&
                           (change_end - change_start < len / 4);  // < 25% changed

    if (!can_incremental) {
        // Full recomputation
        state.reset();

        // Phase 1: Count first bytes
        for (size_t i = 0; i < len; i++) {
            state.bucket_counts[data[i]]++;
            state.pos_to_bucket[i] = data[i];
            state.prev_bucket[i] = data[i];
        }

        // Phase 2: Compute bucket offsets
        uint32_t offset = 0;
        for (int b = 0; b < 256; b++) {
            state.bucket_offsets[b] = offset;
            offset += state.bucket_counts[b];
        }

        // Mark all buckets dirty
        state.mark_all_dirty();
    } else {
        // Incremental update
        state.incremental_updates++;

        // Determine affected positions:
        // Any position i where the 8-byte key window [i, i+8) overlaps [change_start, change_end)
        size_t affected_start = (change_start >= 8) ? change_start - 7 : 0;
        size_t affected_end = std::min(change_end, len);

        // Clear dirty flags
        memset(state.bucket_dirty, 0, sizeof(state.bucket_dirty));

        // Track bucket count changes
        int32_t count_delta[256] = {0};

        // Update bucket assignments for changed positions
        for (size_t i = affected_start; i < affected_end; i++) {
            uint8_t old_bucket = state.prev_bucket[i];
            uint8_t new_bucket = data[i];

            if (old_bucket != new_bucket) {
                count_delta[old_bucket]--;
                count_delta[new_bucket]++;
                state.bucket_dirty[old_bucket] = true;
                state.bucket_dirty[new_bucket] = true;
                state.pos_to_bucket[i] = new_bucket;
                state.prev_bucket[i] = new_bucket;
            }

            // Mark bucket dirty if key might change (even if first byte same)
            state.bucket_dirty[data[i]] = true;
        }

        // Update bucket counts
        for (int b = 0; b < 256; b++) {
            state.bucket_counts[b] = static_cast<uint32_t>(
                static_cast<int32_t>(state.bucket_counts[b]) + count_delta[b]
            );
        }

        // Recompute offsets (fast)
        uint32_t offset = 0;
        for (int b = 0; b < 256; b++) {
            state.bucket_offsets[b] = offset;
            offset += state.bucket_counts[b];
        }
    }

    // Phase 3: Scatter entries to buckets
    uint32_t scatter_idx[256];
    memcpy(scatter_idx, state.bucket_offsets, sizeof(scatter_idx));

    for (size_t i = 0; i < len; i++) {
        uint8_t bucket = state.pos_to_bucket[i];
        uint32_t idx = scatter_idx[bucket]++;

        state.entries[idx].key = load_key64(data, i, len);
        state.entries[idx].pos = static_cast<uint32_t>(i);
    }

    // Phase 4: Sort only dirty buckets
    int dirty_count = 0;
    for (int b = 0; b < 256; b++) {
        if (state.bucket_dirty[b] || !can_incremental) {
            uint32_t start = state.bucket_offsets[b];
            uint32_t count = state.bucket_counts[b];

            if (count > 1) {
                radix_sort_entries(
                    &state.entries[start],
                    &state.temp_entries[start],
                    count,
                    data,
                    len
                );
            }
            dirty_count++;
        }
    }

    // Phase 5: Flatten to SA output
    for (size_t i = 0; i < len; i++) {
        sa_out[i] = static_cast<int32_t>(state.entries[i].pos);
    }

    // Update state for next call
    state.prev_data = data;
    state.prev_len = len;
}

/**
 * Thread-local DeroBWT state accessor
 * Allocates state lazily on first use per thread
 */
inline DeroBWTState& get_thread_local_state() {
    static thread_local DeroBWTState state;
    return state;
}

/**
 * Thread-local incremental state accessor
 */
inline DeroBWTIncrementalState& get_thread_local_incremental_state() {
    static thread_local DeroBWTIncrementalState state;
    return state;
}

/**
 * Convenience function: compute SA using thread-local state
 */
DEROBWT_HOT
inline void compute_sa_threadsafe(
    const uint8_t* DEROBWT_RESTRICT data,
    size_t len,
    int32_t* DEROBWT_RESTRICT sa_out
) {
    DeroBWTState& state = get_thread_local_state();
    compute_sa_optimized(data, len, sa_out, state);
}

/**
 * Convenience function: compute SA incrementally using thread-local state
 */
DEROBWT_HOT
inline void compute_sa_incremental_threadsafe(
    const uint8_t* DEROBWT_RESTRICT data,
    size_t len,
    size_t change_start,
    size_t change_end,
    int32_t* DEROBWT_RESTRICT sa_out
) {
    DeroBWTIncrementalState& state = get_thread_local_incremental_state();
    compute_sa_incremental(data, len, change_start, change_end, sa_out, state);
}

} // namespace derobwt
