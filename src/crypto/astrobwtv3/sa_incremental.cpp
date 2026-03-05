/**
 * Incremental Suffix Array Updates - Implementation
 *
 * Port of Tritonn's incremental SA algorithm from Rust to C++.
 */

#include "sa_incremental.hpp"
#include <algorithm>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

#endif

namespace sa_incremental {

// ============================================================================
// Internal Helper Functions
// ============================================================================

/**
 * Insertion sort for a small list of suffix indices.
 */
static void insertion_sort_suffixes(int32_t* indices, size_t len, const uint8_t* data) {
    for (size_t i = 1; i < len; i++) {
        int32_t key = indices[i];
        size_t j = i;

        while (j > 0 && compare_suffixes(indices[j - 1], key, data) > 0) {
            indices[j] = indices[j - 1];
            j--;
        }
        indices[j] = key;
    }
}

/**
 * Merge two sorted suffix lists into the output SA.
 */
static void merge_sorted_suffix_lists(
    const int32_t* affected,
    size_t affected_len,
    const int32_t* unaffected,
    size_t unaffected_len,
    int32_t* output,
    const uint8_t* data
) {
    size_t i = 0;  // affected index
    size_t j = 0;  // unaffected index
    size_t k = 0;  // output index

    while (i < affected_len && j < unaffected_len) {
        int cmp = compare_suffixes(affected[i], unaffected[j], data);
        if (cmp <= 0) {
            output[k] = affected[i];
            i++;
        } else {
            output[k] = unaffected[j];
            j++;
        }
        k++;
    }

    // Copy remaining elements
    while (i < affected_len) {
        output[k] = affected[i];
        i++;
        k++;
    }

    while (j < unaffected_len) {
        output[k] = unaffected[j];
        j++;
        k++;
    }
}

/**
 * Find where suffix[0] should be inserted in the SA.
 */
static size_t find_insertion_point_for_suffix0(
    const int32_t* prev_sa,
    size_t old_rank,
    const uint8_t* data
) {
    for (size_t i = 0; i < 256; i++) {
        if (i == old_rank) {
            continue;  // Skip suffix[0]'s old position
        }

        size_t other_suffix = prev_sa[i];
        int cmp = compare_suffixes(0, other_suffix, data);

        if (cmp <= 0) {
            // suffix[0] should come before or at this position
            return (i < old_rank) ? i : i - 1;
        }
    }

    // suffix[0] is the largest - goes at the end
    return 255;
}

/**
 * Optimized update when only the last byte (position 255) changed.
 */
static UpdateResult update_sa_last_byte_change(
    const int32_t* prev_sa,
    int32_t* new_sa,
    const uint8_t* data
) {
    // Find where suffix[255] currently is in prev_sa
    size_t old_rank = 0;
    for (size_t i = 0; i < 256; i++) {
        if (prev_sa[i] == 255) {
            old_rank = i;
            break;
        }
    }

    // Copy all elements first
    memcpy(new_sa, prev_sa, 256 * sizeof(int32_t));

    // Find new rank for suffix[255]
    uint8_t new_byte = data[255];
    size_t new_rank = 0;

    // suffix[255] is just the single byte `new_byte`
    // Find where it should be placed
    for (size_t i = 0; i < 256; i++) {
        if (i == old_rank) {
            continue;
        }

        size_t other_suffix = prev_sa[i];
        uint8_t first_byte_other = data[other_suffix];

        // Compare single-byte suffix with first byte of other suffix
        if (new_byte < first_byte_other) {
            // Found insertion point
            new_rank = (i < old_rank) ? i : i - 1;
            break;
        } else if (new_byte == first_byte_other) {
            // Equal first bytes - shorter suffix comes first
            if (other_suffix < 255) {
                // suffix[255] is shorter, comes first
                new_rank = (i < old_rank) ? i : i - 1;
                break;
            }
        }

        // Update potential insertion point
        new_rank = (i < old_rank) ? i + 1 : i;
    }

    // Shift and insert
    if (new_rank < old_rank) {
        // Move left
        memcpy(new_sa, prev_sa, new_rank * sizeof(int32_t));
        new_sa[new_rank] = 255;
        memcpy(new_sa + new_rank + 1, prev_sa + new_rank, (old_rank - new_rank) * sizeof(int32_t));
        memcpy(new_sa + old_rank + 1, prev_sa + old_rank + 1, (255 - old_rank) * sizeof(int32_t));
    } else if (new_rank > old_rank) {
        // Move right
        memcpy(new_sa, prev_sa, old_rank * sizeof(int32_t));
        memcpy(new_sa + old_rank, prev_sa + old_rank + 1, (new_rank - old_rank) * sizeof(int32_t));
        new_sa[new_rank] = 255;
        memcpy(new_sa + new_rank + 1, prev_sa + new_rank + 1, (255 - new_rank) * sizeof(int32_t));
    }

    return UpdateResult::Success;
}

/**
 * Main algorithm: Partitioned update
 */
static UpdateResult update_sa_partitioned(
    const int32_t* prev_sa,
    int32_t* new_sa,
    size_t changed_pos,
    const uint8_t* data
) {
    // Step 1: Partition suffixes into affected and unaffected
    size_t num_affected = changed_pos + 1;

    // Collect affected and unaffected suffixes in their previous SA order
    int32_t affected_indices[256];
    size_t affected_count = 0;

    int32_t unaffected_indices[256];
    size_t unaffected_count = 0;

    // Scan prev_sa once to partition
    for (size_t i = 0; i < 256; i++) {
        int32_t suffix_idx = prev_sa[i];
        if (static_cast<size_t>(suffix_idx) <= changed_pos) {
            affected_indices[affected_count++] = suffix_idx;
        } else {
            unaffected_indices[unaffected_count++] = suffix_idx;
        }
    }

    // Step 2: Re-sort the affected suffixes
    insertion_sort_suffixes(affected_indices, affected_count, data);

    // Step 3: Merge the two sorted sequences back into new_sa
    merge_sorted_suffix_lists(
        affected_indices, affected_count,
        unaffected_indices, unaffected_count,
        new_sa, data
    );

    return UpdateResult::Success;
}

// ============================================================================
// Public API Implementation
// ============================================================================

void update_sa_single_byte_circular(
    int32_t* sa,
    const uint8_t* data,
    uint8_t changed_pos
) {
    // Find current position of the changed suffix
    int current_idx = find_suffix_position(sa, 256, changed_pos);
    if (current_idx < 0) return;  // Should not happen

    // Remove the suffix from its current position
    int32_t suffix_val = remove_at_fast(sa, current_idx, 256);

    // Find the new insertion point (now we have 255 valid entries)
    size_t new_idx = find_insertion_point(sa, 255, changed_pos, data);

    // Insert at the new position
    insert_at_fast(sa, new_idx, suffix_val, 255);
}

UpdateResult update_sa_single_byte(
    const int32_t* prev_sa,
    int32_t* new_sa,
    size_t changed_pos,
    uint8_t new_byte,
    uint8_t old_byte,
    const uint8_t* data
) {
    // Trivial case: no change
    if (new_byte == old_byte) {
        memcpy(new_sa, prev_sa, 256 * sizeof(int32_t));
        return UpdateResult::NoChange;
    }

    // Special case: change at position 255 (last byte)
    if (changed_pos == 255) {
        return update_sa_last_byte_change(prev_sa, new_sa, data);
    }

    // General case: use the partitioned update algorithm
    return update_sa_partitioned(prev_sa, new_sa, changed_pos, data);
}

UpdateResult update_sa_pos0_change(
    const int32_t* prev_sa,
    int32_t* new_sa,
    uint8_t new_byte,
    uint8_t old_byte,
    const uint8_t* data
) {
    if (new_byte == old_byte) {
        memcpy(new_sa, prev_sa, 256 * sizeof(int32_t));
        return UpdateResult::NoChange;
    }

    // Step 1: Find where suffix[0] is in prev_sa
    size_t old_rank = 0;
    for (size_t i = 0; i < 256; i++) {
        if (prev_sa[i] == 0) {
            old_rank = i;
            break;
        }
    }

    // Step 2: Find new position for suffix[0]
    size_t new_rank = find_insertion_point_for_suffix0(prev_sa, old_rank, data);

    // Step 3: Build new_sa by shifting elements
    if (new_rank == old_rank) {
        // No change in position - just copy
        memcpy(new_sa, prev_sa, 256 * sizeof(int32_t));
    } else if (new_rank < old_rank) {
        // suffix[0] moves left (became smaller)
        memcpy(new_sa, prev_sa, new_rank * sizeof(int32_t));
        new_sa[new_rank] = 0;
        memcpy(new_sa + new_rank + 1, prev_sa + new_rank, (old_rank - new_rank) * sizeof(int32_t));
        memcpy(new_sa + old_rank + 1, prev_sa + old_rank + 1, (255 - old_rank) * sizeof(int32_t));
    } else {
        // suffix[0] moves right (became larger)
        memcpy(new_sa, prev_sa, old_rank * sizeof(int32_t));
        memcpy(new_sa + old_rank, prev_sa + old_rank + 1, (new_rank - old_rank) * sizeof(int32_t));
        new_sa[new_rank] = 0;
        memcpy(new_sa + new_rank + 1, prev_sa + new_rank + 1, (255 - new_rank) * sizeof(int32_t));
    }

    return UpdateResult::Success;
}

UpdateResult update_sa_two_byte(
    const int32_t* prev_sa,
    int32_t* new_sa,
    size_t pos1,
    size_t pos2,
    const uint8_t* data
) {
    // Two-byte update: suffixes at positions 0..=pos2 are affected
    // Use the same partitioned update algorithm
    return update_sa_partitioned(prev_sa, new_sa, pos2, data);
}

// ============================================================================
// SIMD-Accelerated Suffix Comparison (AVX2)
// ============================================================================

#if defined(__x86_64__) || defined(_M_X64)

// Check for AVX2 support
static bool avx2_available() {
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
}

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((target("avx2")))
#endif
static int compare_suffixes_avx2_impl(const uint8_t* data, size_t a, size_t b) {
    size_t offset = 0;

    // Compare 32 bytes at a time while possible without wraparound
    while (offset + 32 <= 256) {
        size_t pos_a = (a + offset) & 0xFF;
        size_t pos_b = (b + offset) & 0xFF;

        // Check if we can do a contiguous 32-byte comparison without wraparound
        if (pos_a + 32 <= 256 && pos_b + 32 <= 256) {
            __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + pos_a));
            __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + pos_b));

            // Compare for equality
            __m256i cmp = _mm256_cmpeq_epi8(va, vb);
            uint32_t mask = static_cast<uint32_t>(_mm256_movemask_epi8(cmp));

            if (mask != 0xFFFFFFFF) {
                // Found a difference - find the first differing byte
                uint32_t first_diff;
#ifdef _MSC_VER
                unsigned long idx;
                _BitScanForward(&idx, ~mask);
                first_diff = idx;
#else
                first_diff = __builtin_ctz(~mask);
#endif
                uint8_t byte_a = data[pos_a + first_diff];
                uint8_t byte_b = data[pos_b + first_diff];
                return static_cast<int>(byte_a) - static_cast<int>(byte_b);
            }

            offset += 32;
        } else {
            // Wraparound case - fall back to scalar
            break;
        }
    }

    // Scalar comparison for remaining bytes (with wraparound)
    while (offset < 256) {
        size_t pos_a = (a + offset) & 0xFF;
        size_t pos_b = (b + offset) & 0xFF;

        if (data[pos_a] < data[pos_b]) return -1;
        if (data[pos_a] > data[pos_b]) return 1;
        offset++;
    }

    return 0;
}

int compare_suffixes_simd(const uint8_t* data, size_t a, size_t b) {
    if (a == b) {
        return 0;
    }

    // Only use SIMD path if AVX2 is available and both suffixes don't wrap in first 32 bytes
    static bool has_avx2 = avx2_available();
    if (has_avx2 && a + 32 <= 256 && b + 32 <= 256) {
        return compare_suffixes_avx2_impl(data, a, b);
    }

    // Fallback to scalar for wraparound cases
    return compare_suffixes_circular(data, a, b);
}

size_t find_insertion_point_simd(
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
        int cmp = compare_suffixes_simd(data, pos, sa[mid]);

        if (cmp < 0) {
            hi = mid;
        } else if (cmp > 0) {
            lo = mid + 1;
        } else {
            return mid;
        }
    }

    return lo;
}

#endif  // x86_64

} // namespace sa_incremental
