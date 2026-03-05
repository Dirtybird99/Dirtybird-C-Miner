/**
 * test_chain_trace.cpp - Trace the induction chains for positions 105 and 69993
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

extern "C" {
#include "astrobwtv3/divsufsort.h"
}

int main() {
    printf("=== Chain Trace Test ===\n\n");

    const size_t len = 70000;
    uint8_t* data = (uint8_t*)malloc(len);
    int32_t* sa_ref = (int32_t*)malloc((len + 1) * sizeof(int32_t));
    saidx_t* bucket_A = (saidx_t*)malloc(256 * sizeof(saidx_t));
    saidx_t* bucket_B = (saidx_t*)malloc(65536 * sizeof(saidx_t));

    // Sequential pattern
    for (size_t i = 0; i < len; i++) {
        data[i] = (uint8_t)(i * 17 + 31);
    }

    // Classify suffixes
    uint8_t* types = (uint8_t*)calloc(len, 1);  // 0=L, 1=S
    types[len - 1] = 1;  // Last is S-type
    for (ssize_t i = len - 2; i >= 0; i--) {
        if (data[i] < data[i + 1]) types[i] = 1;
        else if (data[i] > data[i + 1]) types[i] = 0;
        else types[i] = types[i + 1];
    }

    printf("=== Chain from 105 (tracing backwards to L-type origin) ===\n");
    printf("%-8s %-8s %-8s %-8s\n", "Pos", "Byte", "Type", "LMS?");
    int pos = 105;
    while (pos < (int)len && types[pos] == 1) {  // While S-type
        bool is_lms = (pos > 0 && types[pos] && !types[pos - 1]);
        printf("%-8d %-8d %-8s %-8s\n", pos, data[pos], "S", is_lms ? "YES" : "no");
        pos++;
    }
    if (pos < (int)len) {
        printf("%-8d %-8d %-8s %-8s  <- L-type origin\n", pos, data[pos], "L", "no");
    }

    printf("\n=== Chain from 69993 (tracing backwards to sentinel) ===\n");
    printf("%-8s %-8s %-8s %-8s\n", "Pos", "Byte", "Type", "LMS?");
    pos = 69993;
    while (pos < (int)len) {
        bool is_lms = (pos > 0 && types[pos] && !types[pos - 1]);
        printf("%-8d %-8d %-8s %-8s%s\n", pos, data[pos],
               types[pos] ? "S" : "L", is_lms ? "YES" : "no",
               pos == (int)len - 1 ? "  <- sentinel" : "");
        if (!types[pos]) break;  // Hit L-type
        pos++;
    }

    printf("\n=== Bucket analysis for chain endpoints ===\n");
    printf("Position 105 is in bucket %d (T[105]=%d)\n", data[105], data[105]);
    printf("Position 69993 is in bucket %d (T[69993]=%d)\n", data[69993], data[69993]);

    // Find L-type origin for 105's chain
    pos = 105;
    while (pos < (int)len && types[pos]) pos++;
    printf("\nL-type origin for 105's chain: position %d (bucket %d)\n", pos, data[pos]);

    printf("\nSentinel position 69999 is in bucket %d\n", data[69999]);

    // Check if origin buckets are the same
    printf("\n=== Key positions in bucket 126 (shared bucket) ===\n");
    for (int p = 0; p < (int)len; p++) {
        if (data[p] == 126) {
            bool is_lms = (p > 0 && types[p] && !types[p - 1]);
            bool is_sentinel = (p == (int)len - 1);
            if (p >= 105 && p <= 125) {
                printf("Position %d: type=%s, LMS=%s%s\n", p,
                       types[p] ? "S" : "L", is_lms ? "YES" : "no",
                       is_sentinel ? " <- SENTINEL" : "");
            } else if (p == 69999) {
                printf("Position %d: type=%s, LMS=%s%s\n", p,
                       types[p] ? "S" : "L", is_lms ? "YES" : "no",
                       " <- SENTINEL");
            }
        }
    }

    // Run divsufsort for reference
    memset(bucket_A, 0, 256 * sizeof(saidx_t));
    memset(bucket_B, 0, 65536 * sizeof(saidx_t));
    divsufsort(data, sa_ref, len, bucket_A, bucket_B);

    // Find positions of interest in divsufsort result
    printf("\n=== Positions in divsufsort result ===\n");
    int positions_of_interest[] = {105, 106, 107, 108, 109, 110, 111, 118, 69993, 69994, 69995, 69996, 69997, 69998, 69999};
    for (int p : positions_of_interest) {
        for (size_t i = 0; i < len; i++) {
            if (sa_ref[i] == p) {
                printf("SA[%5zu] = %d\n", i, p);
                break;
            }
        }
    }

    // Show bucket 24 contents from divsufsort
    printf("\n=== Bucket 24 contents (where 105 and 69993 both reside) ===\n");
    int count = 0;
    bool found_105 = false, found_69993 = false;
    for (size_t i = 0; i < len && count < 50; i++) {
        if (data[sa_ref[i]] == 24) {  // First byte is 24
            bool show = false;
            if (sa_ref[i] == 105) { found_105 = true; show = true; }
            if (sa_ref[i] == 69993) { found_69993 = true; show = true; }
            if ((found_105 || found_69993) && count < 20) show = true;
            if (!found_105 && !found_69993 && count < 5) show = true;

            if (show) {
                printf("SA[%5zu] = %d\n", i, sa_ref[i]);
            }
            count++;
        }
    }

    free(data);
    free(sa_ref);
    free(bucket_A);
    free(bucket_B);
    free(types);

    printf("\n=== Analysis ===\n");
    printf("The problem: positions 105 and 69993 share bucket 24.\n");
    printf("During S-induction, whichever chain is processed FIRST places its suffix\n");
    printf("at a HIGHER index (since cursor decrements after each placement).\n");
    printf("\n");
    printf("69993 should be at LOWER index (comes first lexicographically).\n");
    printf("So 69993 should be placed SECOND (at lower cursor position).\n");
    printf("So 69994 should be processed AFTER 106.\n");
    printf("\n");
    printf("Both chains converge through the same buckets (109, 92, 75, 58, 41, 24).\n");
    printf("The chain from sentinel (69999) starts in bucket 126.\n");
    printf("The chain from L-type (118) also has position 111 in bucket 126.\n");
    printf("\n");
    printf("In bucket 126, if sentinel 69999 is at TAIL (placed as pseudo-LMS),\n");
    printf("it's processed FIRST (right-to-left scan), placing its chain first.\n");
    printf("This causes 69993 to be at HIGHER index than 105 - WRONG!\n");

    printf("\n=== Done ===\n");
    return 0;
}
