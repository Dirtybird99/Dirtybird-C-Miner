#pragma once
/**
 * Incremental Suffix Array Updates
 *
 * Port of Tritonn's incremental SA algorithm from Rust to C++.
 *
 * When only 1-2 bytes change between iterations, we can update the SA
 * incrementally in O(n) instead of rebuilding in O(n log n).
 *
 * This exploits the pos2 in {0,1} constraint in AstroBWT where 90%+
 * of iterations only modify 1-2 bytes.
 *
 * ## Algorithm Overview
 *
 * For a 256-byte block with suffix array SA[256], when byte at position `p`
 * changes from `old_val` to `new_val`:
 *
 * 1. **Affected Suffixes**: Only suffixes starting at positions 0..=p are affected
 *    (suffixes starting after p don't include the changed byte)
 *
 * 2. **Bucket Movement**: The suffix at position p moves from bucket[old_val]
 *    to bucket[new_val]. All other affected suffixes may need reordering
 *    within their buckets based on the new comparison results.
 *
 * 3. **Incremental Update**: Rather than full rebuild, we:
 *    - Remove affected suffixes from their current positions
 *    - Reinsert them in correct order using binary search within buckets
 *
 * ## AstroBWT Specific Optimization
 *
 * In AstroBWT's branchy loop:
 * - `pos2 = (random_switcher >> 16) & 0xFF` where random_switcher = prev_lhash ^ lhash ^ tries
 * - At loop start: prev_lhash == lhash, so random_switcher = tries (1-277)
 * - For tries 1-255: pos2 = 0
 * - For tries 256-277: pos2 = 1
 *
 * This means ~90% of iterations modify only byte at position 0 or 1!
 */

#include <cstdint>
#include <cstring>

namespace sa_incremental {

// ============================================================================
// Update Result Type
// ============================================================================

enum class UpdateResult {
    Success,              // Update was successful
    NoChange,            // No change needed (old_byte == new_byte)
    FallbackRecommended  // Too many affected suffixes, fallback to full rebuild
};

// ============================================================================
// Quick Check Functions
// ============================================================================

/// Check if we can use single-byte fast-path.
/// This occurs when pos1 == pos2 (only one byte is modified).
inline bool can_use_single_byte_update(uint8_t pos1, uint8_t pos2) {
    return pos1 == pos2;
}

/// Check if we can use two-byte fast-path.
/// This occurs when exactly two consecutive bytes are modified.
inline bool can_use_two_byte_update(uint8_t pos1, uint8_t pos2) {
    return pos2 == pos1 + 1;
}

// ============================================================================
// Suffix Comparison Functions
// ============================================================================

/**
 * Compare two suffixes lexicographically within a 256-byte circular buffer.
 *
 * For AstroBWT's 256-byte chunks, suffixes wrap around, so we compare
 * up to 256 bytes using modular arithmetic for position calculation.
 *
 * @param data The 256-byte data array
 * @param a Starting position of first suffix
 * @param b Starting position of second suffix
 * @return Negative if suffix[a] < suffix[b], positive if suffix[a] > suffix[b], 0 if equal
 */
inline int compare_suffixes_circular(const uint8_t* data, size_t a, size_t b) {
    if (a == b) {
        return 0;
    }

    // Compare byte-by-byte with wraparound
    for (size_t i = 0; i < 256; i++) {
        size_t pos_a = (a + i) & 0xFF;  // Equivalent to (a + i) % 256 but faster
        size_t pos_b = (b + i) & 0xFF;

        if (data[pos_a] < data[pos_b]) return -1;
        if (data[pos_a] > data[pos_b]) return 1;
    }

    return 0;  // All 256 bytes are equal
}

/**
 * Compare two suffixes lexicographically (linear buffer version).
 *
 * @param a Starting position of first suffix
 * @param b Starting position of second suffix
 * @param data The 256-byte data array
 * @return Negative if suffix[a] < suffix[b], positive if suffix[a] > suffix[b], 0 if equal
 */
inline int compare_suffixes(size_t a, size_t b, const uint8_t* data) {
    if (a == b) {
        return 0;
    }

    size_t len_a = 256 - a;
    size_t len_b = 256 - b;
    size_t min_len = (len_a < len_b) ? len_a : len_b;

    // Compare byte by byte
    for (size_t i = 0; i < min_len; i++) {
        if (data[a + i] < data[b + i]) return -1;
        if (data[a + i] > data[b + i]) return 1;
    }

    // One is prefix of the other - shorter comes first
    if (len_a < len_b) return -1;
    if (len_a > len_b) return 1;
    return 0;
}

// ============================================================================
// Binary Search for Insertion Point
// ============================================================================

/**
 * Find insertion point for suffix[pos] in sorted SA using binary search.
 *
 * @param sa The suffix array (sorted)
 * @param sa_len Current number of valid entries in SA
 * @param pos Position of the suffix to insert
 * @param data The 256-byte data array
 * @return Index where the suffix should be inserted
 */
inline size_t find_insertion_point(
    const int32_t* sa,
    size_t sa_len,
    uint8_t pos,
    const uint8_t* data
) {
    if (sa_len == 0) {
        return 0;
    }

    size_t lo = 0;
    size_t hi = sa_len;

    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        int cmp = compare_suffixes_circular(data, pos, sa[mid]);

        if (cmp < 0) {
            hi = mid;
        } else if (cmp > 0) {
            lo = mid + 1;
        } else {
            return mid;  // Found exact match
        }
    }

    return lo;
}

// ============================================================================
// Array Insertion and Removal
// ============================================================================

/**
 * Insert value at position, shifting elements right.
 */
inline void insert_at_fast(int32_t* arr, size_t pos, int32_t value, size_t len) {
    if (pos < len) {
        // Use memmove for efficient memory move (handles overlapping regions)
        memmove(arr + pos + 1, arr + pos, (len - pos) * sizeof(int32_t));
    }
    arr[pos] = value;
}

/**
 * Remove the element at the specified position, shifting elements left.
 *
 * @return The removed value
 */
inline int32_t remove_at_fast(int32_t* arr, size_t pos, size_t len) {
    int32_t removed = arr[pos];

    if (pos + 1 < len) {
        memmove(arr + pos, arr + pos + 1, (len - pos - 1) * sizeof(int32_t));
    }

    return removed;
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Find the position of a suffix in the SA (linear search by value).
 */
inline int find_suffix_position(const int32_t* sa, size_t sa_len, uint8_t suffix_pos) {
    for (size_t i = 0; i < sa_len; i++) {
        if (sa[i] == static_cast<int32_t>(suffix_pos)) {
            return static_cast<int>(i);
        }
    }
    return -1;  // Not found
}

/**
 * Verify that a suffix array is correctly sorted (circular buffer version).
 */
inline bool verify_sa_sorted_circular(const int32_t* sa, size_t sa_len, const uint8_t* data) {
    if (sa_len <= 1) {
        return true;
    }

    for (size_t i = 0; i < sa_len - 1; i++) {
        int cmp = compare_suffixes_circular(data, sa[i], sa[i + 1]);
        if (cmp > 0) {
            return false;
        }
    }

    return true;
}

// ============================================================================
// Core Update Functions
// ============================================================================

/**
 * Update a suffix array when a single byte changes (circular buffer version).
 *
 * This is the core incremental update operation:
 * 1. Find the suffix that was affected by the byte change
 * 2. Remove it from its current position
 * 3. Find its new correct position using binary search
 * 4. Insert it there
 */
void update_sa_single_byte_circular(
    int32_t* sa,
    const uint8_t* data,
    uint8_t changed_pos
);

/**
 * Update suffix array when only one byte changed at position `changed_pos`.
 *
 * @param prev_sa Previous suffix array
 * @param new_sa Output suffix array
 * @param changed_pos Position that changed (0-255)
 * @param new_byte New value at changed_pos
 * @param old_byte Old value at changed_pos
 * @param data Current 256-byte chunk (with new_byte already written)
 * @return UpdateResult indicating success or if fallback is recommended
 */
UpdateResult update_sa_single_byte(
    const int32_t* prev_sa,
    int32_t* new_sa,
    size_t changed_pos,
    uint8_t new_byte,
    uint8_t old_byte,
    const uint8_t* data
);

/**
 * Ultra-fast update when only position 0 changes (pos2=0 in AstroBWT).
 *
 * This is a common case! When pos2=0, only suffix[0] changes its
 * first byte. All other suffixes maintain their relative order.
 *
 * Complexity: O(256) for the shift, O(log 256) for the search = O(256) total
 */
UpdateResult update_sa_pos0_change(
    const int32_t* prev_sa,
    int32_t* new_sa,
    uint8_t new_byte,
    uint8_t old_byte,
    const uint8_t* data
);

/**
 * Update SA for two consecutive byte changes at pos1 and pos2 = pos1 + 1.
 *
 * Similar to single-byte update but with two changed positions.
 * Suffixes at positions 0..=pos2 are affected.
 */
UpdateResult update_sa_two_byte(
    const int32_t* prev_sa,
    int32_t* new_sa,
    size_t pos1,
    size_t pos2,
    const uint8_t* data
);

// ============================================================================
// SIMD-Accelerated Suffix Comparison (AVX2)
// ============================================================================

#if defined(__x86_64__) || defined(_M_X64)
/**
 * Compare two suffixes with SIMD acceleration when available.
 *
 * Uses AVX2 to compare 32 bytes at a time for faster suffix comparison.
 * Falls back to scalar comparison for wraparound cases and when AVX2 is unavailable.
 */
int compare_suffixes_simd(const uint8_t* data, size_t a, size_t b);

/**
 * Find insertion point using SIMD-accelerated suffix comparison.
 */
size_t find_insertion_point_simd(
    const int32_t* sa,
    size_t sa_len,
    uint8_t pos,
    const uint8_t* data
);
#endif

} // namespace sa_incremental
