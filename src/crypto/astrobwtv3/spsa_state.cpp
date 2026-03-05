/**
 * SPSA (Stamp-based Predictive Suffix Array) Implementation
 *
 * Port of Tritonn's SPSA algorithm from Rust to C++.
 *
 * Key insight: Consecutive 256-byte chunks ("stamps") in the branchy loop
 * typically differ by only 1-2 bytes (~90% of iterations).
 *
 * Instead of building the full 70KB SA every time, we:
 * 1. Group consecutive chunks with same pos1/pos2 values into "stamps"
 * 2. Build mini suffix arrays for each stamp
 * 3. Merge stamp SAs into the final SA
 * 4. Hash directly from compressed representation (on-demand decompression)
 */

#include "spsa_state.hpp"
#include <algorithm>
#include <cstring>

// ============================================================================
// Compiler hints for bounds assumptions, inlining, and branch prediction
// ============================================================================
#if defined(__GNUC__) || defined(__clang__)
#define ASSUME(cond) do { if (!(cond)) __builtin_unreachable(); } while(0)
#define ALWAYS_INLINE __attribute__((always_inline)) inline
#define HOT_FUNCTION __attribute__((hot))
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#define RESTRICT __restrict
#define PREFETCH_READ(addr, locality) __builtin_prefetch((addr), 0, (locality))
#define PREFETCH_WRITE(addr, locality) __builtin_prefetch((addr), 1, (locality))
#elif defined(_MSC_VER)
#include <intrin.h>
#define ASSUME(cond) __assume(cond)
#define ALWAYS_INLINE __forceinline
#define HOT_FUNCTION
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#define RESTRICT __restrict
#define PREFETCH_READ(addr, locality) _mm_prefetch((const char*)(addr), _MM_HINT_T0)
#define PREFETCH_WRITE(addr, locality) _mm_prefetch((const char*)(addr), _MM_HINT_T0)
#else
#define ASSUME(cond) ((void)0)
#define ALWAYS_INLINE inline
#define HOT_FUNCTION
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#define RESTRICT
#define PREFETCH_READ(addr, locality) ((void)0)
#define PREFETCH_WRITE(addr, locality) ((void)0)
#endif

namespace spsa {

// ============================================================================
// Radix Sort Implementation (O(n) complexity for u32)
// ============================================================================

// Counting sort helper for radix sort (one pass per byte)
// OPTIMIZED: cached sizes, raw pointer access, prefetch, __restrict
static HOT_FUNCTION void counting_sort_byte(
    uint32_t* RESTRICT arr,
    uint32_t* RESTRICT temp,
    size_t len,
    int byte_idx
) {
    uint32_t count[256] = {0};

    // Cache shift value and length to avoid repeated calculations
    const int shift = byte_idx * 8;
    const size_t n = len;
    ASSUME(n > 0);

    // Count occurrences with prefetch for better memory access patterns
    for (size_t i = 0; i < n; i++) {
        // Prefetch 16 entries ahead for L2 cache
        if (LIKELY(i + 16 < n)) {
            PREFETCH_READ(&arr[i + 16], 1);  // L2 hint
        }
        const uint8_t byte_val = (arr[i] >> shift) & 0xFF;
        count[byte_val]++;
    }

    // Compute cumulative counts (prefix sums)
    uint32_t total = 0;
    for (int i = 0; i < 256; i++) {
        const uint32_t old_count = count[i];
        count[i] = total;
        total += old_count;
    }

    // Place elements in sorted order
    for (size_t i = 0; i < n; i++) {
        const uint8_t byte_val = (arr[i] >> shift) & 0xFF;
        temp[count[byte_val]++] = arr[i];
    }

    // Copy back to original array
    memcpy(arr, temp, n * sizeof(uint32_t));
}

// LSD Radix sort for u32 values
// Sorts by least significant byte first, working up to most significant
void radix_sort_u32(uint32_t* arr, size_t len) {
    if (len < 2) return;

    // For small arrays, use insertion sort (lower overhead)
    if (len < RADSORT_THRESHOLD) {
        insertion_sort_packed(arr, len);
        return;
    }

    // Allocate temporary buffer
    std::vector<uint32_t> temp(len);

    // Sort by each byte, LSB to MSB
    // For packed values (byte_value << 16 | index), we only need bytes 0-2
    // But for general u32, we sort all 4 bytes
    counting_sort_byte(arr, temp.data(), len, 0);
    counting_sort_byte(arr, temp.data(), len, 1);
    counting_sort_byte(arr, temp.data(), len, 2);
    counting_sort_byte(arr, temp.data(), len, 3);
}

// ============================================================================
// SpsaState Implementation
// ============================================================================

void SpsaState::finalize_current_stamp() {
    // This is called internally when transitioning stamps
    // The stamp data is already stored via start_stamp/add_chunk calls
}

/**
 * Build a mini suffix array for a stamp
 *
 * The mini-SA contains ALL suffixes from all chunks in this stamp, sorted.
 * Per Tritonn's adjacency guarantee: "All chunk suffixes within a stamp are
 * GUARANTEED to be found directly adjacent to each other in the final suffix array."
 *
 * OPTIMIZED: Raw pointer access, cached sizes, reduced bounds checks
 */
void SpsaState::build_mini_sa(Stamp& stamp, std::vector<uint32_t>& mini_sa,
                               const uint8_t* data, size_t data_size) {
    if (UNLIKELY(stamp.chunk_count == 0)) {
        mini_sa.clear();
        return;
    }

    // Cache all stamp fields locally to avoid repeated struct access
    const size_t n = stamp.chunk_count;
    const uint8_t pos1 = stamp.pos1;
    const uint8_t pos2 = stamp.pos2;
    const uint32_t start_chunk = stamp.start_chunk;
    const uint8_t* RESTRICT pos1_bytes = stamp.pos1_bytes.data();
    const uint8_t* RESTRICT byte255_data = stamp.byte255.data();

    // Create packed sort keys: (byte_value << 16) | chunk_index
    // This allows sorting chunks by their pos1 byte value
    std::vector<uint32_t> s1(n);
    uint32_t* RESTRICT s1_ptr = s1.data();
    for (size_t i = 0; i < n; i++) {
        s1_ptr[i] = (static_cast<uint32_t>(pos1_bytes[i]) << 16) | static_cast<uint32_t>(i);
    }

    // Sort by pos1 byte (primary sort key)
    if (n >= RADSORT_THRESHOLD) {
        radix_sort_u32(s1_ptr, n);
    } else {
        insertion_sort_packed(s1_ptr, n);
    }

    // Also sort by byte255 for secondary ordering
    std::vector<uint32_t> s2(n);
    uint32_t* RESTRICT s2_ptr = s2.data();
    for (size_t i = 0; i < n; i++) {
        s2_ptr[i] = (static_cast<uint32_t>(byte255_data[i]) << 16) | static_cast<uint32_t>(i);
    }

    if (n >= RADSORT_THRESHOLD) {
        radix_sort_u32(s2_ptr, n);
    } else {
        insertion_sort_packed(s2_ptr, n);
    }

    // Build the mini-SA
    // For positions 0..pos1-1: use s1 ordering
    // For positions pos2+1..254: use s2 ordering
    mini_sa.clear();
    mini_sa.reserve(n * 256);

    // Positions before pos1 - use raw pointer access
    for (size_t r = 0; r < pos1; r++) {
        for (size_t j = 0; j < n; j++) {
            const uint32_t chunk_idx = s1_ptr[j] & 0xFFFF;
            const uint32_t global_pos = (start_chunk + chunk_idx) * 256 + static_cast<uint32_t>(r);
            mini_sa.push_back(global_pos);
        }
    }

    // Positions after pos2 (up to 254, excluding 255 which is always modified)
    for (size_t r = static_cast<size_t>(pos2) + 1; r < 255; r++) {
        for (size_t j = 0; j < n; j++) {
            const uint32_t chunk_idx = s2_ptr[j] & 0xFFFF;
            const uint32_t global_pos = (start_chunk + chunk_idx) * 256 + static_cast<uint32_t>(r);
            mini_sa.push_back(global_pos);
        }
    }
}

/**
 * Build mini suffix arrays for all stamps
 */
void SpsaState::build_all_mini_sas(const uint8_t* data, size_t data_size) {
    stamp_sas.clear();
    stamp_sas.reserve(stamps.size());
    first_indexes.clear();
    first_indexes.reserve(stamps.size());

    for (auto& stamp : stamps) {
        std::vector<uint32_t> mini_sa;
        build_mini_sa(stamp, mini_sa, data, data_size);

        // First index = start position of this stamp's first chunk
        uint32_t first_idx = stamp.start_chunk * 256;
        first_indexes.push_back(first_idx);
        stamp_sas.push_back(std::move(mini_sa));
    }
}

/**
 * Get global position from a merge entry (OPTIMIZED)
 *
 * Merge entries can be either:
 * - Direct positions (high bit clear): the value is the global position
 * - Stamp references (high bit set): encoded stamp_id and relative position
 *
 * Optimizations:
 * - Branchless execution using conditional masking
 * - Shift instead of multiply (<< 8 vs * 256)
 * - No bounds check (SA construction guarantees validity)
 */
static ALWAYS_INLINE size_t get_merge_global_position(uint32_t encoded, const Stamp* RESTRICT stamps_ptr) {
    // Extract fields
    const uint16_t stamp_id = (encoded >> 16) & 0x01FF;
    const uint16_t relative_pos = encoded & 0xFFFF;

    // Stamp path: shift instead of multiply (1 cycle vs 3 cycles)
    const size_t stamp_result = (static_cast<size_t>(stamps_ptr[stamp_id].start_chunk) << 8) + relative_pos;

    // Direct path: mask off high bit
    const size_t direct_result = static_cast<size_t>(encoded & 0x7FFFFFFF);

    // Branchless select using high bit
    const bool is_stamp = (encoded & 0x80000000) != 0;
    return is_stamp ? stamp_result : direct_result;
}

// Overload for vector compatibility (used in merge_stamps)
static ALWAYS_INLINE size_t get_merge_global_position(uint32_t encoded, const std::vector<Stamp>& stamps) {
    return get_merge_global_position(encoded, stamps.data());
}

/**
 * Fast 8-byte key extraction for radix sort
 * Extracts first 8 bytes of suffix as a 64-bit key for fast comparison
 * OPTIMIZED: ALWAYS_INLINE, unrolled loop for common case
 */
static ALWAYS_INLINE uint64_t extract_sort_key_8bytes(size_t pos, const uint8_t* RESTRICT data, size_t data_size) {
    const size_t remaining = data_size - pos;

    // Fast path: 8+ bytes remaining (common case)
    if (LIKELY(remaining >= 8)) {
        // Load 8 bytes and convert to big-endian key
        uint64_t key = 0;
        const uint8_t* RESTRICT p = data + pos;
        key = (static_cast<uint64_t>(p[0]) << 56) |
              (static_cast<uint64_t>(p[1]) << 48) |
              (static_cast<uint64_t>(p[2]) << 40) |
              (static_cast<uint64_t>(p[3]) << 32) |
              (static_cast<uint64_t>(p[4]) << 24) |
              (static_cast<uint64_t>(p[5]) << 16) |
              (static_cast<uint64_t>(p[6]) << 8) |
              static_cast<uint64_t>(p[7]);
        return key;
    }

    // Slow path: less than 8 bytes
    uint64_t key = 0;
    const size_t to_copy = remaining;

    // Build key from available bytes (big-endian for lexicographic ordering)
    for (size_t i = 0; i < to_copy; i++) {
        key = (key << 8) | data[pos + i];
    }
    // Pad with zeros (shorter suffixes come first)
    key <<= (8 - to_copy) * 8;

    return key;
}

/**
 * Sort entry with precomputed key for radix sort
 */
struct SortEntry {
    uint64_t key;      // First 8 bytes of suffix (big-endian)
    uint32_t pos;      // Global position (for tiebreaking)
    uint32_t encoded;  // Original encoded entry
};

/**
 * Counting sort for SortEntry by one byte of the key
 * OPTIMIZED: cached size, raw pointer access, __restrict hints
 */
static HOT_FUNCTION void counting_sort_entries_byte(
    SortEntry* RESTRICT arr,
    SortEntry* RESTRICT temp,
    size_t len,
    int byte_idx
) {
    uint32_t count[256] = {0};

    // Cache shift value and length
    const int shift = (7 - byte_idx) * 8;
    const size_t n = len;
    ASSUME(n > 0);

    // Count occurrences with prefetch for better memory access patterns
    for (size_t i = 0; i < n; i++) {
        // Prefetch 16 entries ahead for L2 cache
        if (LIKELY(i + 16 < n)) {
            PREFETCH_READ(&arr[i + 16], 1);  // L2 hint
        }
        const uint8_t byte_val = (arr[i].key >> shift) & 0xFF;
        count[byte_val]++;
    }

    // Compute cumulative counts
    uint32_t total = 0;
    for (int i = 0; i < 256; i++) {
        const uint32_t old_count = count[i];
        count[i] = total;
        total += old_count;
    }

    // Place elements in sorted order
    for (size_t i = 0; i < n; i++) {
        const uint8_t byte_val = (arr[i].key >> shift) & 0xFF;
        temp[count[byte_val]++] = arr[i];
    }

    // Copy back
    memcpy(arr, temp, n * sizeof(SortEntry));
}

/**
 * LSD Radix sort for SortEntry array
 * Sorts by 8-byte key, then by position for stability
 * OPTIMIZED: cached size, raw pointer access, __restrict hints
 */
static HOT_FUNCTION void radix_sort_entries(SortEntry* RESTRICT arr, size_t len) {
    if (UNLIKELY(len < 2)) return;

    const size_t n = len;
    ASSUME(n >= 2);

    // For small arrays, use insertion sort
    if (n < 32) {
        for (size_t i = 1; i < n; i++) {
            const SortEntry key_entry = arr[i];
            size_t j = i;
            while (j > 0 && (arr[j-1].key > key_entry.key ||
                            (arr[j-1].key == key_entry.key && arr[j-1].pos > key_entry.pos))) {
                arr[j] = arr[j-1];
                j--;
            }
            arr[j] = key_entry;
        }
        return;
    }

    std::vector<SortEntry> temp(n);
    SortEntry* RESTRICT temp_ptr = temp.data();

    // Sort by position first (LSB for stability)
    // Then sort by key bytes from LSB to MSB

    // Sort by position (4 bytes)
    for (int byte_idx = 0; byte_idx < 4; byte_idx++) {
        uint32_t count[256] = {0};
        const int shift = byte_idx * 8;

        for (size_t i = 0; i < n; i++) {
            const uint8_t byte_val = (arr[i].pos >> shift) & 0xFF;
            count[byte_val]++;
        }

        uint32_t total = 0;
        for (int i = 0; i < 256; i++) {
            const uint32_t old_count = count[i];
            count[i] = total;
            total += old_count;
        }

        for (size_t i = 0; i < n; i++) {
            const uint8_t byte_val = (arr[i].pos >> shift) & 0xFF;
            temp_ptr[count[byte_val]++] = arr[i];
        }

        memcpy(arr, temp_ptr, n * sizeof(SortEntry));
    }

    // Sort by key (8 bytes, LSB to MSB)
    for (int byte_idx = 7; byte_idx >= 0; byte_idx--) {
        counting_sort_entries_byte(arr, temp_ptr, n, byte_idx);
    }
}

/**
 * Merge all stamp mini-SAs into a reduced suffix array
 *
 * OPTIMIZED Algorithm:
 * 1. Create buckets (256 total, one per first byte)
 * 2. For each stamp, insert POSITION-LEVEL entries (one per position, not per chunk)
 *    - This reduces entries from O(stamps × chunks × positions) to O(stamps × positions)
 * 3. Add modified bytes as direct entries
 * 4. Sort entries within each bucket (lexicographic comparison)
 * 5. Flatten buckets, EXPANDING position-level entries to all chunks
 *
 * Key insight: Within a stamp, all chunks have IDENTICAL data at stable positions.
 * We only need to sort position-level representatives, then expand during flattening.
 *
 * Output:
 * Returns `reduced_sa` where each entry is either:
 * - Stamp reference (high bit set) + stamp_id + relative_pos
 * - Direct suffix index (for modified bytes)
 */
void SpsaState::merge_stamps(const uint8_t* data, size_t data_size) {
    reduced_sa.clear();

    if (stamps.empty()) {
        return;
    }

    // Phase 1: Initialize buckets (256 total, one per first byte)
    std::vector<std::vector<uint32_t>> buckets(256);

    // Estimate entries for pre-allocation (now much smaller due to position-level entries)
    size_t total_positions = 0;
    size_t total_chunks = 0;
    for (const auto& s : stamps) {
        size_t num_modified_in_range = (s.pos1 == s.pos2) ? 1 : 2;
        size_t stable_positions = 255 - num_modified_in_range;
        total_positions += stable_positions;  // One entry per position per stamp
        total_chunks += s.chunk_count;
    }
    size_t estimated_bucket_entries = total_positions + modified_bytes.size();
    size_t avg_per_bucket = estimated_bucket_entries / 256 + 1;

    for (auto& bucket : buckets) {
        bucket.reserve(avg_per_bucket);
    }

    // Phase 2: Add POSITION-LEVEL entries for stable positions
    // Each entry represents ALL chunks at this position within the stamp
    // This is the key optimization: O(stamps × positions) instead of O(stamps × chunks × positions)
    for (size_t stamp_id = 0; stamp_id < stamps.size(); stamp_id++) {
        const Stamp& stamp = stamps[stamp_id];
        if (stamp.chunk_count == 0) continue;

        size_t pos1 = stamp.pos1;
        size_t pos2 = stamp.pos2;

        // Use first chunk as representative for comparison (all chunks identical at stable positions)
        size_t first_chunk_start = static_cast<size_t>(stamp.start_chunk) * 256;

        // Add ONE entry per stable position (not per chunk!)
        for (size_t rel_pos = 0; rel_pos < 255; rel_pos++) {
            if (rel_pos == pos1 || rel_pos == pos2) {
                continue;
            }

            size_t global_pos = first_chunk_start + rel_pos;
            if (global_pos >= data_size) break;

            uint8_t first_byte = data[global_pos];

            // Position-level entry: includes MERGE_POSITION_FLAG to indicate expansion needed
            // Encodes: stamp_id and rel_pos (no chunk_offset since all chunks are identical)
            uint32_t encoded = MERGE_STAMP_MARKER
                | MERGE_POSITION_FLAG  // Flag indicating this expands to multiple suffixes
                | ((static_cast<uint32_t>(stamp_id) << MERGE_STAMP_ID_SHIFT) & MERGE_STAMP_ID_MASK)
                | (static_cast<uint32_t>(rel_pos) & MERGE_POS_MASK);
            buckets[first_byte].push_back(encoded);
        }
    }

    // Phase 3: Add modified bytes as DIRECT entries (high bit clear)
    for (const auto& modified : modified_bytes) {
        size_t global_pos = modified.global_pos;
        if (global_pos >= data_size) continue;

        uint8_t first_byte = data[global_pos];
        // Direct entry: high bit clear, value is global position
        buckets[first_byte].push_back(modified.global_pos);
    }

    // Phase 4: Sort each bucket using radix sort on first 8 bytes
    // Now sorting O(stamps × positions) entries instead of O(stamps × chunks × positions)
    // OPTIMIZED: Cache stamps pointer, use raw pointer access, removed redundant bounds checks
    const Stamp* RESTRICT stamps_ptr = stamps.data();
    const size_t stamps_count = stamps.size();

    for (auto& bucket : buckets) {
        const size_t bucket_size = bucket.size();
        if (bucket_size > 1) {
            // Cache bucket data pointer
            const uint32_t* RESTRICT bucket_data = bucket.data();

            // Build sort entries with precomputed keys
            std::vector<SortEntry> entries(bucket_size);
            SortEntry* RESTRICT entries_ptr = entries.data();

            for (size_t i = 0; i < bucket_size; i++) {
                // Prefetch next entry's data position (24 ahead for NTA streaming)
                if (LIKELY(i + 24 < bucket_size)) {
                    const uint32_t next_encoded = bucket_data[i + 24];
                    size_t next_pos;
                    if (is_position_level_entry(next_encoded)) {
                        const uint16_t next_stamp_id = decode_merge_stamp_id(next_encoded);
                        const uint16_t next_rel_pos = decode_merge_relative_pos(next_encoded);
                        // SA construction guarantees valid stamp_id
                        ASSUME(next_stamp_id < stamps_count);
                        next_pos = (static_cast<size_t>(stamps_ptr[next_stamp_id].start_chunk) << 8) + next_rel_pos;
                    } else {
                        next_pos = get_merge_global_position(next_encoded, stamps_ptr);
                    }
                    if (LIKELY(next_pos < data_size)) {
                        PREFETCH_READ(&data[next_pos], 0);  // NTA hint for streaming
                    }
                }

                const uint32_t encoded = bucket_data[i];
                size_t pos;

                if (is_position_level_entry(encoded)) {
                    // Position-level: use first chunk's position for comparison
                    const uint16_t stamp_id = decode_merge_stamp_id(encoded);
                    const uint16_t rel_pos = decode_merge_relative_pos(encoded);
                    // SA construction guarantees valid stamp_id - no bounds check needed
                    ASSUME(stamp_id < stamps_count);
                    pos = (static_cast<size_t>(stamps_ptr[stamp_id].start_chunk) << 8) + rel_pos;
                } else {
                    pos = get_merge_global_position(encoded, stamps_ptr);
                }

                entries_ptr[i].key = extract_sort_key_8bytes(pos, data, data_size);
                entries_ptr[i].pos = static_cast<uint32_t>(pos);
                entries_ptr[i].encoded = encoded;
            }

            // Radix sort by key, then by position
            radix_sort_entries(entries_ptr, bucket_size);

            // Write back sorted encoded values - use raw pointer
            uint32_t* RESTRICT bucket_out = bucket.data();
            for (size_t i = 0; i < bucket_size; i++) {
                bucket_out[i] = entries_ptr[i].encoded;
            }
        }
    }

    // Phase 5: Flatten buckets into reduced_sa, EXPANDING position-level entries
    // Final size = total_chunks * stable_positions + modified_bytes
    // OPTIMIZED: Cache stamp pointer, removed bounds checks (SA construction guarantees validity)
    size_t final_size = 0;
    for (size_t i = 0; i < stamps_count; i++) {
        const size_t num_modified_in_range = (stamps_ptr[i].pos1 == stamps_ptr[i].pos2) ? 1 : 2;
        const size_t stable_per_chunk = 255 - num_modified_in_range;
        final_size += stable_per_chunk * stamps_ptr[i].chunk_count;
    }
    final_size += modified_bytes.size();
    reduced_sa.reserve(final_size);

    for (const auto& bucket : buckets) {
        const size_t bucket_len = bucket.size();
        const uint32_t* RESTRICT bucket_ptr = bucket.data();

        for (size_t idx = 0; idx < bucket_len; idx++) {
            const uint32_t entry = bucket_ptr[idx];

            if (is_position_level_entry(entry)) {
                // Expand position-level entry to all chunks in the stamp
                const uint16_t stamp_id = decode_merge_stamp_id(entry);
                const uint16_t rel_pos = decode_merge_relative_pos(entry);

                // SA construction guarantees valid stamp_id - no bounds check needed
                ASSUME(stamp_id < stamps_count);
                const Stamp& stamp = stamps_ptr[stamp_id];
                const uint16_t chunk_count = stamp.chunk_count;

                // Pre-compute base encoded value (without chunk offset)
                const uint32_t base_encoded = MERGE_STAMP_MARKER
                    | ((static_cast<uint32_t>(stamp_id) << MERGE_STAMP_ID_SHIFT) & MERGE_STAMP_ID_MASK);
                const uint32_t rel_pos_val = static_cast<uint32_t>(rel_pos);

                // Emit one entry per chunk (all have identical suffix at this position)
                for (uint16_t chunk_offset = 0; chunk_offset < chunk_count; chunk_offset++) {
                    // Convert to chunk-level entry (without MERGE_POSITION_FLAG)
                    const uint32_t combined_pos = (static_cast<uint32_t>(chunk_offset) << 8) + rel_pos_val;
                    const uint32_t expanded = base_encoded | (combined_pos & MERGE_POS_MASK);
                    reduced_sa.push_back(expanded);
                }
            } else {
                // Direct entry or already chunk-level - pass through
                reduced_sa.push_back(entry);
            }
        }
    }
}

/**
 * Convert merge entry to global position (for SHA-256 hashing)
 *
 * This is the compatibility wrapper that delegates to the optimized inline version.
 * For hot paths, use merge_entry_to_global_pos_fast() directly with stamps.data().
 */
int32_t merge_entry_to_global_pos(uint32_t entry, const std::vector<Stamp>& stamps) {
    // Delegate to optimized branchless version
    // The fast version assumes valid stamp_id (guaranteed by SA construction)
    return merge_entry_to_global_pos_fast(entry, stamps.data());
}

} // namespace spsa
