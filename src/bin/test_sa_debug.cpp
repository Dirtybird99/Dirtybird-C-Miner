/**
 * test_sa_debug.cpp - Debug SA correctness issue
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" {
#include "astrobwtv3/divsufsort.h"
#include "astrobwtv3/custom_sa_70kb.h"
}

int main() {
    printf("=== SA Debug Test ===\n\n");

    // Test with 70KB size like the real workload
    const size_t len = 70000;
    uint8_t* data = (uint8_t*)malloc(len + 1);
    int32_t* sa1 = (int32_t*)malloc((len + 1) * sizeof(int32_t));
    int32_t* sa2 = (int32_t*)malloc((len + 1) * sizeof(int32_t));
    saidx_t* bucket_A = (saidx_t*)malloc(256 * sizeof(saidx_t));
    saidx_t* bucket_B = (saidx_t*)malloc(65536 * sizeof(saidx_t));

    // Fill with sequential data like original test
    for (size_t i = 0; i < len; i++) {
        data[i] = (uint8_t)(i * 17 + 31);
    }
    printf("Test data: sequential (i*17+31), len=%zu\n\n", len);

    printf("Running divsufsort...\n");
    fflush(stdout);
    memset(bucket_A, 0, 256 * sizeof(saidx_t));
    memset(bucket_B, 0, 65536 * sizeof(saidx_t));
    divsufsort(data, sa1, len, bucket_A, bucket_B);
    printf("divsufsort completed.\n");

    printf("divsufsort result (first 20): ");
    for (size_t i = 0; i < 20 && i < len; i++) {
        printf("%d ", sa1[i]);
    }
    printf("...\n\n");

    printf("Running custom_sa_70kb...\n");
    fflush(stdout);
    memset(bucket_A, 0, 256 * sizeof(saidx_t));
    memset(bucket_B, 0, 65536 * sizeof(saidx_t));
    int ret = custom_sa_70kb(data, sa2, len, bucket_A, bucket_B);
    printf("custom_sa_70kb completed (ret=%d).\n", ret);

    printf("custom_sa result (first 20): ");
    for (size_t i = 0; i < 20 && i < len; i++) {
        printf("%d ", sa2[i]);
    }
    printf("...\n\n");

    // Validate custom SA
    bool* seen = (bool*)calloc(len, sizeof(bool));
    bool valid = true;
    for (size_t i = 0; i < len && valid; i++) {
        if (sa2[i] < 0 || (size_t)sa2[i] >= len) {
            printf("Invalid position at SA[%zu] = %d\n", i, sa2[i]);
            valid = false;
        } else if (seen[sa2[i]]) {
            printf("Duplicate position: SA[%zu] = %d\n", i, sa2[i]);
            valid = false;
        }
        seen[sa2[i]] = true;
    }
    free(seen);

    if (valid) {
        printf("All positions valid and unique.\n");

        // Check ordering
        bool sorted = true;
        for (size_t i = 1; i < len && sorted; i++) {
            size_t max_cmp = len - (sa2[i-1] > sa2[i] ? sa2[i-1] : sa2[i]);
            int cmp = memcmp(data + sa2[i-1], data + sa2[i], max_cmp);
            if (cmp > 0) {
                printf("Ordering violation at %zu: SA[%zu]=%d > SA[%zu]=%d\n",
                       i, i-1, sa2[i-1], i, sa2[i]);
                printf("  suffix[%d] = \"%s\"\n", sa2[i-1], data + sa2[i-1]);
                printf("  suffix[%d] = \"%s\"\n", sa2[i], data + sa2[i]);
                sorted = false;
            }
        }
        if (sorted) {
            printf("Ordering is correct!\n");
        }
    }

    // Compare with divsufsort
    int diffs = 0;
    for (size_t i = 0; i < len; i++) {
        if (sa1[i] != sa2[i]) diffs++;
    }
    printf("\nDifferences from divsufsort: %d\n", diffs);

    free(data);
    free(sa1);
    free(sa2);
    free(bucket_A);
    free(bucket_B);

    printf("\n=== Done ===\n");
    return 0;
}
