/**
 * test_random_debug.cpp - Debug random pattern failures
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

extern "C" {
#include "astrobwtv3/divsufsort.h"
#include "astrobwtv3/custom_sa_70kb.h"
}

int main() {
    printf("=== Random Pattern Debug Test ===\n\n");

    const size_t len = 70000;
    uint8_t* data = (uint8_t*)malloc(len);
    int32_t* sa = (int32_t*)malloc((len + 1) * sizeof(int32_t));
    int32_t* sa_ref = (int32_t*)malloc((len + 1) * sizeof(int32_t));
    saidx_t* bucket_A = (saidx_t*)malloc(256 * sizeof(saidx_t));
    saidx_t* bucket_B = (saidx_t*)malloc(65536 * sizeof(saidx_t));

    // Random pattern (same as test-custom-sa)
    srand(12345);
    for (size_t i = 0; i < len; i++) {
        data[i] = rand() % 256;
    }
    printf("Pattern: Random uniform (seed=12345), len=%zu\n\n", len);

    // Run divsufsort as reference
    printf("Running divsufsort...\n");
    memset(bucket_A, 0, 256 * sizeof(saidx_t));
    memset(bucket_B, 0, 65536 * sizeof(saidx_t));
    divsufsort(data, sa_ref, len, bucket_A, bucket_B);

    // Run custom_sa_70kb
    printf("Running custom_sa_70kb...\n");
    memset(bucket_A, 0, 256 * sizeof(saidx_t));
    memset(bucket_B, 0, 65536 * sizeof(saidx_t));
    custom_sa_70kb(data, sa, len, bucket_A, bucket_B);

    // Find first ordering violation
    printf("\nChecking for ordering violations...\n");
    int violations = 0;
    for (size_t i = 1; i < len && violations < 5; i++) {
        size_t cmp_len = std::min(len - sa[i-1], len - sa[i]);
        int cmp = memcmp(data + sa[i-1], data + sa[i], cmp_len);

        bool is_violation = false;
        const char* violation_type = "";

        if (cmp > 0) {
            is_violation = true;
            violation_type = "ORDERING (cmp > 0)";
        } else if (cmp == 0 && sa[i-1] < sa[i]) {
            is_violation = true;
            violation_type = "TIE-BREAK (equal but pos[i-1] < pos[i])";
        }

        if (is_violation) {
            violations++;
            printf("\n=== Violation #%d at SA[%zu,%zu] ===\n", violations, i-1, i);
            printf("Type: %s\n", violation_type);
            printf("SA[%zu] = %d (suffix starting at pos %d)\n", i-1, sa[i-1], sa[i-1]);
            printf("SA[%zu] = %d (suffix starting at pos %d)\n", i, sa[i], sa[i]);
            printf("memcmp result: %d (compared %zu bytes)\n", cmp, cmp_len);

            // Show suffix content (first 16 bytes)
            printf("T[%d..+16] = ", sa[i-1]);
            for (int k = 0; k < 16 && sa[i-1] + k < (int)len; k++) {
                printf("%02x ", data[sa[i-1] + k]);
            }
            printf("...\n");

            printf("T[%d..+16] = ", sa[i]);
            for (int k = 0; k < 16 && sa[i] + k < (int)len; k++) {
                printf("%02x ", data[sa[i] + k]);
            }
            printf("...\n");

            // Check what divsufsort has at this position
            printf("\nReference (divsufsort) at same positions:\n");
            printf("sa_ref[%zu] = %d\n", i-1, sa_ref[i-1]);
            printf("sa_ref[%zu] = %d\n", i, sa_ref[i]);
        }
    }

    if (violations == 0) {
        printf("No ordering violations found!\n");
    } else {
        printf("\nTotal violations found: %d (showed first 5)\n", violations);
    }

    // Count total differences from divsufsort
    int diffs = 0;
    for (size_t i = 0; i < len; i++) {
        if (sa[i] != sa_ref[i]) diffs++;
    }
    printf("\nTotal differences from divsufsort: %d\n", diffs);

    // Check for duplicates
    int* seen = (int*)calloc(len, sizeof(int));
    int duplicates = 0;
    for (size_t i = 0; i < len; i++) {
        if (sa[i] >= 0 && sa[i] < (int)len) {
            if (seen[sa[i]]) {
                if (duplicates < 5) {
                    printf("Duplicate: position %d appears at SA[%zu] and previously\n", sa[i], i);
                }
                duplicates++;
            }
            seen[sa[i]] = 1;
        } else if (sa[i] == -1) {
            printf("Empty slot at SA[%zu]\n", i);
        }
    }
    if (duplicates > 0) {
        printf("Total duplicates: %d\n", duplicates);
    }

    // Check for missing positions
    int missing = 0;
    for (size_t i = 0; i < len; i++) {
        if (!seen[i]) {
            if (missing < 5) {
                printf("Missing: position %zu not found in SA\n", i);
            }
            missing++;
        }
    }
    if (missing > 0) {
        printf("Total missing: %d\n", missing);
    }

    free(data);
    free(sa);
    free(sa_ref);
    free(bucket_A);
    free(bucket_B);
    free(seen);

    printf("\n=== Done ===\n");
    return 0;
}
