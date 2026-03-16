#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

extern "C" {
#include "astrobwtv3/divsufsort.h"
#include "astrobwtv3/custom_sa_70kb.h"
}

int main() {
    printf("=== SA Small Test ===\n\n");
    
    // Small test first
    const size_t len = 100;
    uint8_t* data = (uint8_t*)malloc(len + 1);
    int32_t* sa1 = (int32_t*)malloc((len + 1) * sizeof(int32_t));
    int32_t* sa2 = (int32_t*)malloc((len + 1) * sizeof(int32_t));
    saidx_t* bucket_A = (saidx_t*)malloc(256 * sizeof(saidx_t));
    saidx_t* bucket_B = (saidx_t*)malloc(65536 * sizeof(saidx_t));
    
    // Random data
    srand(12345);
    for (size_t i = 0; i < len; i++) {
        data[i] = rand() % 256;
    }
    printf("Test 1: Random data len=%zu\n", len);
    
    memset(bucket_A, 0, 256 * sizeof(saidx_t));
    memset(bucket_B, 0, 65536 * sizeof(saidx_t));
    divsufsort(data, sa1, len, bucket_A, bucket_B);
    
    memset(bucket_A, 0, 256 * sizeof(saidx_t));
    memset(bucket_B, 0, 65536 * sizeof(saidx_t));
    custom_sa_70kb(data, sa2, len, bucket_A, bucket_B);
    
    // Compare
    int diffs = 0;
    for (size_t i = 0; i < len; i++) {
        if (sa1[i] != sa2[i]) diffs++;
    }
    
    if (diffs == 0) {
        printf("Test 1 PASSED: outputs match\n\n");
    } else {
        printf("Test 1 FAILED: %d differences\n", diffs);
        printf("divsufsort: ");
        for (int i = 0; i < 20; i++) printf("%d ", sa1[i]);
        printf("...\n");
        printf("custom_sa:  ");
        for (int i = 0; i < 20; i++) printf("%d ", sa2[i]);
        printf("...\n\n");
    }
    
    // Test 2: Larger random data
    const size_t len2 = 1000;
    free(data); free(sa1); free(sa2);
    data = (uint8_t*)malloc(len2 + 1);
    sa1 = (int32_t*)malloc((len2 + 1) * sizeof(int32_t));
    sa2 = (int32_t*)malloc((len2 + 1) * sizeof(int32_t));
    
    srand(54321);
    for (size_t i = 0; i < len2; i++) {
        data[i] = rand() % 256;
    }
    printf("Test 2: Random data len=%zu\n", len2);
    
    memset(bucket_A, 0, 256 * sizeof(saidx_t));
    memset(bucket_B, 0, 65536 * sizeof(saidx_t));
    divsufsort(data, sa1, len2, bucket_A, bucket_B);
    
    memset(bucket_A, 0, 256 * sizeof(saidx_t));
    memset(bucket_B, 0, 65536 * sizeof(saidx_t));
    custom_sa_70kb(data, sa2, len2, bucket_A, bucket_B);
    
    diffs = 0;
    for (size_t i = 0; i < len2; i++) {
        if (sa1[i] != sa2[i]) diffs++;
    }
    
    if (diffs == 0) {
        printf("Test 2 PASSED: outputs match\n\n");
    } else {
        printf("Test 2 FAILED: %d differences\n\n", diffs);
    }
    
    // Test 3: 70000 random
    const size_t len3 = 70000;
    free(data); free(sa1); free(sa2);
    data = (uint8_t*)malloc(len3 + 1);
    sa1 = (int32_t*)malloc((len3 + 1) * sizeof(int32_t));
    sa2 = (int32_t*)malloc((len3 + 1) * sizeof(int32_t));
    
    srand(99999);
    for (size_t i = 0; i < len3; i++) {
        data[i] = rand() % 256;
    }
    printf("Test 3: Random data len=%zu\n", len3);
    
    memset(bucket_A, 0, 256 * sizeof(saidx_t));
    memset(bucket_B, 0, 65536 * sizeof(saidx_t));
    divsufsort(data, sa1, len3, bucket_A, bucket_B);
    
    memset(bucket_A, 0, 256 * sizeof(saidx_t));
    memset(bucket_B, 0, 65536 * sizeof(saidx_t));
    custom_sa_70kb(data, sa2, len3, bucket_A, bucket_B);
    
    diffs = 0;
    for (size_t i = 0; i < len3; i++) {
        if (sa1[i] != sa2[i]) diffs++;
    }
    
    if (diffs == 0) {
        printf("Test 3 PASSED: outputs match\n");
    } else {
        printf("Test 3 FAILED: %d differences\n", diffs);

        // Validate custom SA - check for duplicates
        bool* seen = (bool*)calloc(len3, sizeof(bool));
        bool valid = true;
        for (size_t i = 0; i < len3 && valid; i++) {
            if (sa2[i] < 0 || (size_t)sa2[i] >= len3) {
                printf("Invalid position at SA[%zu] = %d\n", i, sa2[i]);
                valid = false;
            } else if (seen[sa2[i]]) {
                printf("Duplicate position: SA[%zu] = %d\n", i, sa2[i]);
                valid = false;
            }
            seen[sa2[i]] = true;
        }
        free(seen);

        // Check ordering
        if (valid) {
            printf("All positions valid and unique.\n");
            int ordering_errors = 0;
            for (size_t i = 1; i < len3 && ordering_errors < 5; i++) {
                size_t max_cmp = len3 - ((size_t)sa2[i-1] > (size_t)sa2[i] ? sa2[i-1] : sa2[i]);
                int cmp = memcmp(data + sa2[i-1], data + sa2[i], max_cmp);
                if (cmp > 0 || (cmp == 0 && sa2[i-1] < sa2[i])) {
                    printf("Ordering violation at SA[%zu,%zu]: pos %d > pos %d\n",
                           i-1, i, sa2[i-1], sa2[i]);
                    printf("  T[%d..+5] = ", sa2[i-1]);
                    for (int k = 0; k < 5 && sa2[i-1] + k < (int)len3; k++)
                        printf("%02x ", data[sa2[i-1] + k]);
                    printf("\n");
                    printf("  T[%d..+5] = ", sa2[i]);
                    for (int k = 0; k < 5 && sa2[i] + k < (int)len3; k++)
                        printf("%02x ", data[sa2[i] + k]);
                    printf("\n");
                    ordering_errors++;
                }
            }
            if (ordering_errors > 0) {
                printf("Total ordering violations: %d+\n", ordering_errors);
            } else {
                printf("Ordering is correct!\n");
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
