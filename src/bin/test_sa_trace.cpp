/**
 * test_sa_trace.cpp - Trace specific position handling in custom_sa_70kb
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

extern "C" {
#include "astrobwtv3/divsufsort.h"
#include "astrobwtv3/custom_sa_70kb.h"
}

// Bitvector helper to check suffix types
bool is_s_type(const uint64_t* bits, size_t pos) {
    return (bits[pos / 64] >> (pos % 64)) & 1;
}

bool is_lms(const uint64_t* bits, size_t pos, size_t n) {
    if (pos == 0) return false;
    return is_s_type(bits, pos) && !is_s_type(bits, pos - 1);
}

int main() {
    printf("=== SA Position Trace Test ===\n\n");

    const size_t len = 70000;
    uint8_t* data = (uint8_t*)malloc(len);
    int32_t* sa = (int32_t*)malloc((len + 1) * sizeof(int32_t));
    int32_t* sa_ref = (int32_t*)malloc((len + 1) * sizeof(int32_t));
    saidx_t* bucket_A = (saidx_t*)malloc(256 * sizeof(saidx_t));
    saidx_t* bucket_B = (saidx_t*)malloc(65536 * sizeof(saidx_t));

    // Sequential pattern
    for (size_t i = 0; i < len; i++) {
        data[i] = (uint8_t)(i * 17 + 31);
    }

    // Positions we want to trace
    const int trace_positions[] = {104, 105, 106, 107, 108, 109, 110, 69992, 69993, 69994, 69995, 69996, 69997, 69998, 69999};
    const int num_trace = sizeof(trace_positions) / sizeof(trace_positions[0]);

    // Show suffix types manually by analyzing the data
    printf("=== Suffix Type Analysis ===\n\n");
    printf("Data pattern: sequential (i*17+31), len=%zu\n", len);
    printf("Note: 104 mod 256 = 104, 69992 mod 256 = %d\n\n", 69992 % 256);

    // Classify suffixes manually (S-type vs L-type)
    // L-type: suffix[i] > suffix[i+1]
    // S-type: suffix[i] < suffix[i+1]
    // When equal, inherit type from i+1
    uint8_t* types = (uint8_t*)calloc(len, 1);  // 0=L, 1=S
    types[len - 1] = 1;  // Last suffix is S-type by convention
    for (ssize_t i = len - 2; i >= 0; i--) {
        if (data[i] < data[i + 1]) {
            types[i] = 1;  // S-type
        } else if (data[i] > data[i + 1]) {
            types[i] = 0;  // L-type
        } else {
            types[i] = types[i + 1];  // Equal: inherit
        }
    }

    printf("Traced positions:\n");
    printf("%-8s %-8s %-8s %-12s\n", "Pos", "Type", "LMS?", "First 8 bytes");
    for (int i = 0; i < num_trace; i++) {
        int pos = trace_positions[i];
        if (pos < 0 || pos >= (int)len) continue;

        char type_str[4] = "L";
        if (types[pos]) strcpy(type_str, "S");

        bool is_lms_pos = (pos > 0 && types[pos] && !types[pos - 1]);
        if (pos == 0) is_lms_pos = false;

        printf("%-8d %-8s %-8s ", pos, type_str, is_lms_pos ? "YES" : "no");
        for (int j = 0; j < 8 && pos + j < (int)len; j++) {
            printf("%02x ", data[pos + j]);
        }
        printf("\n");
    }

    // Run divsufsort as reference
    printf("\n=== Running divsufsort ===\n");
    memset(bucket_A, 0, 256 * sizeof(saidx_t));
    memset(bucket_B, 0, 65536 * sizeof(saidx_t));
    divsufsort(data, sa_ref, len, bucket_A, bucket_B);

    // Find where traced positions appear in divsufsort result
    printf("\nPositions in divsufsort result:\n");
    printf("%-8s %-12s\n", "Pos", "SA index");
    for (int i = 0; i < num_trace; i++) {
        int pos = trace_positions[i];
        for (size_t j = 0; j < len; j++) {
            if (sa_ref[j] == pos) {
                printf("%-8d %-12zu\n", pos, j);
                break;
            }
        }
    }

    // Run custom_sa_70kb
    printf("\n=== Running custom_sa_70kb ===\n");
    memset(bucket_A, 0, 256 * sizeof(saidx_t));
    memset(bucket_B, 0, 65536 * sizeof(saidx_t));
    custom_sa_70kb(data, sa, len, bucket_A, bucket_B);

    // Find where traced positions appear in custom_sa result
    printf("\nPositions in custom_sa_70kb result:\n");
    printf("%-8s %-12s\n", "Pos", "SA index");
    for (int i = 0; i < num_trace; i++) {
        int pos = trace_positions[i];
        for (size_t j = 0; j < len; j++) {
            if (sa[j] == pos) {
                printf("%-8d %-12zu\n", pos, j);
                break;
            }
        }
    }

    // Compare adjacent pairs for ordering issues
    printf("\n=== Critical Pairs Analysis ===\n");
    int pairs[][2] = {{104, 69992}, {105, 69993}, {106, 69994}, {107, 69995}};
    for (int p = 0; p < 4; p++) {
        int pos_a = pairs[p][0];
        int pos_b = pairs[p][1];

        // Find their SA indices
        size_t idx_a_ref = 0, idx_b_ref = 0;
        size_t idx_a = 0, idx_b = 0;
        for (size_t j = 0; j < len; j++) {
            if (sa_ref[j] == pos_a) idx_a_ref = j;
            if (sa_ref[j] == pos_b) idx_b_ref = j;
            if (sa[j] == pos_a) idx_a = j;
            if (sa[j] == pos_b) idx_b = j;
        }

        printf("Pair (%d, %d):\n", pos_a, pos_b);
        printf("  divsufsort: pos %d @ SA[%zu], pos %d @ SA[%zu] -> %d comes first\n",
               pos_a, idx_a_ref, pos_b, idx_b_ref,
               idx_a_ref < idx_b_ref ? pos_a : pos_b);
        printf("  custom_sa:  pos %d @ SA[%zu], pos %d @ SA[%zu] -> %d comes first\n",
               pos_a, idx_a, pos_b, idx_b,
               idx_a < idx_b ? pos_a : pos_b);
        printf("  Expected: %d should come first (shorter suffix = lexicographically smaller)\n\n",
               pos_b);  // Higher position = shorter suffix = should come first
    }

    free(data);
    free(sa);
    free(sa_ref);
    free(bucket_A);
    free(bucket_B);
    free(types);

    printf("=== Done ===\n");
    return 0;
}
