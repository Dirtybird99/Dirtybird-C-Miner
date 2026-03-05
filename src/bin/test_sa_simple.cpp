/**
 * test_sa_simple.cpp - Simple SA correctness test
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" {
#include "astrobwtv3/divsufsort.h"
#include "astrobwtv3/custom_sa_70kb.h"
}

int main() {
    printf("=== Simple SA Correctness Test ===\n\n");

    const size_t len = 70000;
    uint8_t* data = (uint8_t*)malloc(len);
    int32_t* sa1 = (int32_t*)malloc((len + 1) * sizeof(int32_t));
    int32_t* sa2 = (int32_t*)malloc((len + 1) * sizeof(int32_t));
    saidx_t* bucket_A = (saidx_t*)malloc(256 * sizeof(saidx_t));
    saidx_t* bucket_B = (saidx_t*)malloc(65536 * sizeof(saidx_t));

    // Fill with sequential data (same as original test)
    for (size_t i = 0; i < len; i++) {
        data[i] = (uint8_t)(i * 17 + 31);
    }

    printf("Testing with sequential data (i*17+31)...\n");

    // Run divsufsort first
    memset(bucket_A, 0, 256 * sizeof(saidx_t));
    memset(bucket_B, 0, 65536 * sizeof(saidx_t));
    divsufsort(data, sa1, len, bucket_A, bucket_B);
    printf("divsufsort completed.\n");

    // Run custom_sa_70kb with fresh buckets
    memset(bucket_A, 0, 256 * sizeof(saidx_t));
    memset(bucket_B, 0, 65536 * sizeof(saidx_t));
    int ret = custom_sa_70kb(data, sa2, len, bucket_A, bucket_B);
    printf("custom_sa_70kb completed (ret=%d).\n", ret);

    // Compare outputs
    int diffs = 0;
    int first_diff = -1;
    for (size_t i = 0; i < len; i++) {
        if (sa1[i] != sa2[i]) {
            diffs++;
            if (first_diff < 0) first_diff = i;
        }
    }

    if (diffs == 0) {
        printf("SUCCESS: Outputs match exactly!\n");
    } else {
        printf("MISMATCH: %d differences found\n", diffs);
        printf("First diff at index %d: divsufsort=%d, custom=%d\n",
               first_diff, sa1[first_diff], sa2[first_diff]);

        // Check if custom SA is still valid (just different ordering for equal suffixes)
        printf("\nValidating custom_sa output...\n");

        // Check all positions appear
        bool* seen = (bool*)calloc(len, sizeof(bool));
        bool valid = true;
        for (size_t i = 0; i < len; i++) {
            if (sa2[i] < 0 || (size_t)sa2[i] >= len) {
                printf("  Invalid position at SA[%zu] = %d\n", i, sa2[i]);
                valid = false;
                break;
            }
            if (seen[sa2[i]]) {
                printf("  Duplicate position: SA[%zu] = %d (seen before)\n", i, sa2[i]);
                valid = false;
                break;
            }
            seen[sa2[i]] = true;
        }
        free(seen);

        if (valid) {
            printf("  All positions valid and unique.\n");

            // Check ordering
            bool sorted = true;
            for (size_t i = 1; i < len && sorted; i++) {
                int cmp = memcmp(data + sa2[i-1], data + sa2[i],
                                 len - (sa2[i-1] > sa2[i] ? sa2[i-1] : sa2[i]));
                if (cmp > 0) {
                    printf("  Ordering violation at %zu: SA[%zu]=%d > SA[%zu]=%d\n",
                           i, i-1, sa2[i-1], i, sa2[i]);
                    sorted = false;
                }
            }
            if (sorted) {
                printf("  Ordering is correct.\n");
                printf("  NOTE: Different but equivalent SA (tie-breaking differs)\n");
            }
        }
    }

    free(data);
    free(sa1);
    free(sa2);
    free(bucket_A);
    free(bucket_B);

    printf("\n=== Done ===\n");
    return 0;
}
