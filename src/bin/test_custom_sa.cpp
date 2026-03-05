/**
 * test_custom_sa.cpp - Benchmark SA implementations for 70KB inputs
 *
 * Compares:
 * - divsufsort (original TNN miner implementation)
 * - custom_sa_70kb (our SA-IS implementation)
 * - libsais 2.10.4 (state-of-the-art SA library)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <vector>
#include <random>

extern "C" {
#include "astrobwtv3/divsufsort.h"

#ifdef USE_CUSTOM_SA
#include "astrobwtv3/custom_sa_70kb.h"
#endif

#ifdef USE_LIBSAIS
#include "libsais.h"
#endif
}

using Clock = std::chrono::steady_clock;
using Duration = std::chrono::duration<double, std::micro>;

// Verify suffix array is correct
bool verify_sa(const uint8_t* data, const int32_t* sa, size_t len) {
    // Check all positions appear exactly once
    std::vector<bool> seen(len, false);
    for (size_t i = 0; i < len; i++) {
        if (sa[i] < 0 || (size_t)sa[i] >= len) return false;
        if (seen[sa[i]]) return false;
        seen[sa[i]] = true;
    }

    // Check lexicographic ordering
    // When suffixes are equal, the shorter suffix (higher position) should come first
    for (size_t i = 1; i < len; i++) {
        int cmp = memcmp(data + sa[i-1], data + sa[i],
                         std::min(len - sa[i-1], len - sa[i]));
        if (cmp > 0) return false;
        // If equal, sa[i-1] should be >= sa[i] (shorter/higher position first)
        if (cmp == 0 && sa[i-1] < sa[i]) return false;
    }
    return true;
}

int main() {
    printf("=== SA Implementation Comparison (70KB) ===\n\n");

#ifdef USE_CUSTOM_SA
    printf("custom_sa_70kb: ENABLED\n");
#else
    printf("custom_sa_70kb: DISABLED\n");
#endif

#ifdef USE_LIBSAIS
    printf("libsais 2.10.4: ENABLED\n");
#else
    printf("libsais 2.10.4: DISABLED\n");
#endif

    printf("\n");

    // Allocate test data
    const size_t len = 70000;
    uint8_t* data = (uint8_t*)malloc(len);
    int32_t* sa = (int32_t*)malloc((len + 1) * sizeof(int32_t));
    int32_t* sa_ref = (int32_t*)malloc((len + 1) * sizeof(int32_t));
    saidx_t* bucket_A = (saidx_t*)malloc(256 * sizeof(saidx_t));
    saidx_t* bucket_B = (saidx_t*)malloc(65536 * sizeof(saidx_t));

    // libsais context for reuse (like mining workload)
#ifdef USE_LIBSAIS
    void* sais_context = libsais_create_ctx();
    if (!sais_context) {
        printf("ERROR: Failed to create libsais context\n");
        return 1;
    }
#endif

    // Test parameters
    const int WARMUP = 50;
    const int ITERATIONS = 200;

    // Test with different data patterns
    struct TestCase {
        const char* name;
        void (*fill)(uint8_t* data, size_t len);
    } tests[] = {
        {"Sequential (i*17+31)", [](uint8_t* d, size_t len) {
            for (size_t i = 0; i < len; i++) d[i] = (uint8_t)(i * 17 + 31);
        }},
        {"Random uniform", [](uint8_t* d, size_t len) {
            std::mt19937 rng(12345);
            std::uniform_int_distribution<> dist(0, 255);
            for (size_t i = 0; i < len; i++) d[i] = (uint8_t)dist(rng);
        }},
        {"Repetitive (low entropy)", [](uint8_t* d, size_t len) {
            for (size_t i = 0; i < len; i++) d[i] = (uint8_t)(i % 16);
        }},
        {"AstroBWT-like (256-byte blocks)", [](uint8_t* d, size_t len) {
            std::mt19937 rng(54321);
            std::uniform_int_distribution<> dist(0, 255);
            // Fill with blocks that differ by 1-2 bytes (like wolf compute)
            for (size_t i = 0; i < len; i += 256) {
                size_t block_len = std::min((size_t)256, len - i);
                for (size_t j = 0; j < block_len; j++) {
                    d[i + j] = (uint8_t)dist(rng);
                }
                // Copy to next block with small changes (simulates SPSA)
                if (i + 256 < len) {
                    memcpy(d + i + 256, d + i, std::min((size_t)256, len - i - 256));
                    // Modify 1-2 bytes
                    d[i + 256 + (dist(rng) % 256)] = (uint8_t)dist(rng);
                }
            }
        }},
    };

    printf("Benchmarking %d iterations after %d warmup...\n\n", ITERATIONS, WARMUP);
    printf("%-30s %12s %12s %12s %12s\n",
           "Test Case", "divsufsort", "custom_sa", "libsais", "Winner");
    printf("%-30s %12s %12s %12s %12s\n",
           "------------------------------", "------------", "------------", "------------", "------------");

    for (const auto& test : tests) {
        test.fill(data, len);

        double divsufsort_us = 0;
        double custom_us = 0;
        double libsais_us = 0;

        // Warmup and benchmark divsufsort
        for (int i = 0; i < WARMUP; i++) {
            divsufsort(data, sa, len, bucket_A, bucket_B);
        }
        auto start = Clock::now();
        for (int i = 0; i < ITERATIONS; i++) {
            divsufsort(data, sa, len, bucket_A, bucket_B);
        }
        divsufsort_us = Duration(Clock::now() - start).count() / ITERATIONS;

        // Store reference for validation
        memcpy(sa_ref, sa, len * sizeof(int32_t));

#ifdef USE_CUSTOM_SA
        // Warmup and benchmark custom_sa_70kb
        for (int i = 0; i < WARMUP; i++) {
            custom_sa_70kb(data, sa, len, bucket_A, bucket_B);
        }
        start = Clock::now();
        for (int i = 0; i < ITERATIONS; i++) {
            custom_sa_70kb(data, sa, len, bucket_A, bucket_B);
        }
        custom_us = Duration(Clock::now() - start).count() / ITERATIONS;

        // Verify correctness
        if (memcmp(sa, sa_ref, len * sizeof(int32_t)) != 0) {
            printf("WARNING: custom_sa_70kb differs from divsufsort for '%s'\n", test.name);
            // Still verify it's a valid SA
            if (!verify_sa(data, sa, len)) {
                printf("ERROR: custom_sa_70kb produced invalid SA!\n");
            }
        }
#endif

#ifdef USE_LIBSAIS
        // Warmup and benchmark libsais with context (simulates mining reuse)
        for (int i = 0; i < WARMUP; i++) {
            libsais_ctx(sais_context, data, sa, len, 0, NULL);
        }
        start = Clock::now();
        for (int i = 0; i < ITERATIONS; i++) {
            libsais_ctx(sais_context, data, sa, len, 0, NULL);
        }
        libsais_us = Duration(Clock::now() - start).count() / ITERATIONS;

        // Verify correctness
        if (memcmp(sa, sa_ref, len * sizeof(int32_t)) != 0) {
            // libsais may produce different but still valid SA
            if (!verify_sa(data, sa, len)) {
                printf("ERROR: libsais produced invalid SA!\n");
            }
        }
#endif

        // Determine winner
        const char* winner = "divsufsort";
        double best = divsufsort_us;
#ifdef USE_CUSTOM_SA
        if (custom_us < best) { best = custom_us; winner = "custom_sa"; }
#endif
#ifdef USE_LIBSAIS
        if (libsais_us < best) { best = libsais_us; winner = "libsais"; }
#endif

        printf("%-30s %10.2f us %10.2f us %10.2f us %12s\n",
               test.name,
               divsufsort_us,
#ifdef USE_CUSTOM_SA
               custom_us,
#else
               0.0,
#endif
#ifdef USE_LIBSAIS
               libsais_us,
#else
               0.0,
#endif
               winner);
    }

    printf("\n=== Summary ===\n");

    // Run one more comprehensive test for summary stats
    std::mt19937 rng(99999);
    std::uniform_int_distribution<> dist(0, 255);
    for (size_t i = 0; i < len; i++) data[i] = (uint8_t)dist(rng);

    // 500 iterations for final timing
    const int FINAL_ITERATIONS = 500;

    for (int i = 0; i < WARMUP; i++) {
        divsufsort(data, sa, len, bucket_A, bucket_B);
    }
    auto start = Clock::now();
    for (int i = 0; i < FINAL_ITERATIONS; i++) {
        divsufsort(data, sa, len, bucket_A, bucket_B);
    }
    double divsufsort_final = Duration(Clock::now() - start).count() / FINAL_ITERATIONS;

#ifdef USE_CUSTOM_SA
    for (int i = 0; i < WARMUP; i++) {
        custom_sa_70kb(data, sa, len, bucket_A, bucket_B);
    }
    start = Clock::now();
    for (int i = 0; i < FINAL_ITERATIONS; i++) {
        custom_sa_70kb(data, sa, len, bucket_A, bucket_B);
    }
    double custom_final = Duration(Clock::now() - start).count() / FINAL_ITERATIONS;
#endif

#ifdef USE_LIBSAIS
    for (int i = 0; i < WARMUP; i++) {
        libsais_ctx(sais_context, data, sa, len, 0, NULL);
    }
    start = Clock::now();
    for (int i = 0; i < FINAL_ITERATIONS; i++) {
        libsais_ctx(sais_context, data, sa, len, 0, NULL);
    }
    double libsais_final = Duration(Clock::now() - start).count() / FINAL_ITERATIONS;
#endif

    printf("\nFinal benchmark (%d iterations, random data):\n", FINAL_ITERATIONS);
    printf("  divsufsort:     %.2f us\n", divsufsort_final);
#ifdef USE_CUSTOM_SA
    printf("  custom_sa_70kb: %.2f us (%.2fx vs divsufsort)\n",
           custom_final, divsufsort_final / custom_final);
#endif
#ifdef USE_LIBSAIS
    printf("  libsais:        %.2f us (%.2fx vs divsufsort)\n",
           libsais_final, divsufsort_final / libsais_final);
#endif

#if defined(USE_CUSTOM_SA) && defined(USE_LIBSAIS)
    printf("\n  custom_sa vs libsais: %.2fx\n", libsais_final / custom_final);
#endif

    // Cleanup
    free(data);
    free(sa);
    free(sa_ref);
    free(bucket_A);
    free(bucket_B);
#ifdef USE_LIBSAIS
    libsais_free_ctx(sais_context);
#endif

    printf("\n=== Done ===\n");
    return 0;
}
