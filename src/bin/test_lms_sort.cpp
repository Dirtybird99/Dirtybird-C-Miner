/**
 * test_lms_sort.cpp - Test LMS comparison and sorting specifically
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

extern "C" {
#include "astrobwtv3/divsufsort.h"
}

// ============================================================================
// Suffix type classification (from custom_sa_70kb.cpp)
// ============================================================================

#define TYPE_BITVEC_SIZE ((72 * 1024 + 63) / 64)

struct TypeBitVector {
    uint64_t bits[TYPE_BITVEC_SIZE];
};

static inline void type_set(TypeBitVector *types, size_t pos, int is_s) {
    size_t word = pos / 64;
    size_t bit = pos % 64;
    if (is_s) {
        types->bits[word] |= (1ULL << bit);
    } else {
        types->bits[word] &= ~(1ULL << bit);
    }
}

static inline int type_is_s(const TypeBitVector *types, size_t pos) {
    size_t word = pos / 64;
    size_t bit = pos % 64;
    return (types->bits[word] >> bit) & 1;
}

static inline int type_is_lms(const TypeBitVector *types, size_t pos) {
    if (pos == 0) return 0;
    return type_is_s(types, pos) && !type_is_s(types, pos - 1);
}

// ============================================================================
// LMS comparison (from custom_sa_70kb.cpp)
// ============================================================================

static int compare_lms_substrings_verbose(
    const uint8_t *T,
    const TypeBitVector *types,
    saidx_t i,
    saidx_t j,
    saidx_t n,
    bool verbose
) {
    saidx_t orig_i = i, orig_j = j;

    if (verbose) {
        printf("  Comparing LMS[%d] vs LMS[%d]:\n", i, j);
    }

    int iterations = 0;
    while (i < n && j < n) {
        if (T[i] != T[j]) {
            if (verbose) {
                printf("    Diff at offset %d: T[%d]=%02x vs T[%d]=%02x\n",
                       iterations, i, T[i], j, T[j]);
            }
            return (int)T[i] - (int)T[j];
        }

        // Check for LMS boundaries (skip starting positions)
        int i_lms = (i > orig_i) && type_is_lms(types, i);
        int j_lms = (j > orig_j) && type_is_lms(types, j);

        if (i_lms && j_lms) {
            if (verbose) {
                printf("    Both hit LMS at offset %d, tie-break: %d - %d = %d\n",
                       iterations, orig_j, orig_i, orig_j - orig_i);
            }
            return orig_j - orig_i;
        }
        if (i_lms) {
            if (verbose) printf("    i hit LMS at offset %d, i is smaller\n", iterations);
            return -1;
        }
        if (j_lms) {
            if (verbose) printf("    j hit LMS at offset %d, j is smaller\n", iterations);
            return 1;
        }

        i++;
        j++;
        iterations++;
    }

    if (verbose) {
        printf("    Reached end of string after %d iterations\n", iterations);
        printf("    i=%d (>= n=%d? %s), j=%d (>= n=%d? %s)\n",
               i, n, i >= n ? "yes" : "no", j, n, j >= n ? "yes" : "no");
    }
    return (i >= n) ? -1 : 1;
}

int main() {
    printf("=== LMS Sort Test ===\n\n");

    const size_t len = 70000;
    uint8_t* data = (uint8_t*)malloc(len);
    TypeBitVector types;
    memset(&types, 0, sizeof(types));

    // Sequential pattern
    for (size_t i = 0; i < len; i++) {
        data[i] = (uint8_t)(i * 17 + 31);
    }

    // Classify suffixes
    type_set(&types, len - 1, 1);  // Last is S-type
    for (ssize_t i = len - 2; i >= 0; i--) {
        if (data[i] < data[i + 1]) {
            type_set(&types, i, 1);  // S-type
        } else if (data[i] > data[i + 1]) {
            type_set(&types, i, 0);  // L-type
        } else {
            type_set(&types, i, type_is_s(&types, i + 1));
        }
    }

    // Find all LMS positions
    std::vector<saidx_t> lms_positions;
    for (size_t i = 0; i < len; i++) {
        if (type_is_lms(&types, i)) {
            lms_positions.push_back(i);
        }
    }
    printf("Found %zu LMS positions\n", lms_positions.size());

    // Find positions 104 and 69992
    saidx_t idx_104 = -1, idx_69992 = -1;
    for (size_t i = 0; i < lms_positions.size(); i++) {
        if (lms_positions[i] == 104) idx_104 = i;
        if (lms_positions[i] == 69992) idx_69992 = i;
    }
    printf("Position 104 is LMS? %s (index %d)\n", idx_104 >= 0 ? "YES" : "no", idx_104);
    printf("Position 69992 is LMS? %s (index %d)\n", idx_69992 >= 0 ? "YES" : "no", idx_69992);

    // Test the comparison
    printf("\n=== Direct LMS Comparison ===\n");
    int cmp = compare_lms_substrings_verbose(data, &types, 104, 69992, len, true);
    printf("Result: compare(104, 69992) = %d\n", cmp);
    printf("Interpretation: %s\n", cmp < 0 ? "104 < 69992 (104 should come first)" :
                                   cmp > 0 ? "104 > 69992 (69992 should come first)" :
                                             "104 == 69992");

    // Test with arguments swapped
    printf("\n=== Swapped Comparison ===\n");
    cmp = compare_lms_substrings_verbose(data, &types, 69992, 104, len, true);
    printf("Result: compare(69992, 104) = %d\n", cmp);
    printf("Interpretation: %s\n", cmp < 0 ? "69992 < 104 (69992 should come first)" :
                                   cmp > 0 ? "69992 > 104 (104 should come first)" :
                                             "69992 == 104");

    // Test sorting just these two
    printf("\n=== Sorting Test ===\n");
    saidx_t test_arr[2] = {104, 69992};
    printf("Before sort: [%d, %d]\n", test_arr[0], test_arr[1]);

    // Simple bubble sort with our comparator
    if (compare_lms_substrings_verbose(data, &types, test_arr[0], test_arr[1], len, false) > 0) {
        std::swap(test_arr[0], test_arr[1]);
    }
    printf("After sort:  [%d, %d]\n", test_arr[0], test_arr[1]);
    printf("Expected:    [69992, 104] (shorter suffix first)\n");

    free(data);
    printf("\n=== Done ===\n");
    return 0;
}
