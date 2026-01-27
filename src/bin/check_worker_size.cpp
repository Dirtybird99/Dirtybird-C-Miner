/**
 * Check workerData struct size and member offsets
 */

#include <cstdio>
#include <cstddef>

// Temporarily define USE_FAST_RC4 to check both layouts
#define USE_FAST_RC4 1
#define USE_ASTRO_SPSA 1
#include "astroworker.h"

int main() {
    printf("=== workerData Layout Analysis ===\n\n");
    printf("sizeof(workerData): %zu bytes\n", sizeof(workerData));
    printf("sizeof(RC4_KEY): %zu bytes\n", sizeof(RC4_KEY));

#if USE_FAST_RC4
    printf("sizeof(rc4_avx512::FastRc4): %zu bytes\n", sizeof(rc4_avx512::FastRc4));
#endif

    workerData w;

    // Print offsets of key members
    printf("\nMember offsets:\n");
    printf("  aarchFixup:     %zu\n", offsetof(workerData, aarchFixup));
    printf("  opt:            %zu\n", offsetof(workerData, opt));
    printf("  sha256:         %zu\n", offsetof(workerData, sha256));
    printf("  salsa20:        %zu\n", offsetof(workerData, salsa20));
    printf("  key:            %zu\n", offsetof(workerData, key));
#if USE_FAST_RC4
    printf("  fast_rc4_key:   %zu\n", offsetof(workerData, fast_rc4_key));
#endif
    printf("  salsaInput:     %zu\n", offsetof(workerData, salsaInput));
    printf("  op:             %zu\n", offsetof(workerData, op));
    printf("  chunk:          %zu\n", offsetof(workerData, chunk));
    printf("  prev_chunk:     %zu\n", offsetof(workerData, prev_chunk));
    printf("  sData:          %zu\n", offsetof(workerData, sData));
    printf("  sa:             %zu\n", offsetof(workerData, sa));

    // Check alignments
    printf("\nAlignment checks:\n");
    printf("  &w.sData %% 64 = %zu\n", (size_t)w.sData % 64);
    printf("  &w.sa %% 64 = %zu\n", (size_t)w.sa % 64);
#if USE_FAST_RC4
    printf("  &w.fast_rc4_key %% 64 = %zu\n", (size_t)&w.fast_rc4_key % 64);
#endif

    return 0;
}
