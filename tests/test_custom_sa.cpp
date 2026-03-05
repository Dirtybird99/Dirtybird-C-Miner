/**
 * test_custom_sa.cpp - Unit tests and benchmarks for custom_sa_70kb
 *
 * Tests:
 * 1. Correctness: Byte-by-byte comparison with divsufsort
 * 2. Edge cases: All-zeros, all-same, increasing, decreasing, random
 * 3. Validation: Verify suffix array is valid
 * 4. Size range: Test across 65536 to 71168 bytes
 * 5. Performance: Benchmark optimized vs baseline
 *
 * Usage:
 *   # Compile:
 *   g++ -O3 -march=native -mavx2 -I../src/crypto/astrobwtv3 \
 *       test_custom_sa.cpp \
 *       ../src/crypto/astrobwtv3/custom_sa_70kb.cpp \
 *       ../src/crypto/astrobwtv3/divsufsort.c \
 *       ../src/crypto/astrobwtv3/sssort.c \
 *       ../src/crypto/astrobwtv3/trsort.c \
 *       -o test_custom_sa
 *
 *   # Run all tests:
 *   ./test_custom_sa
 *
 *   # Run benchmark only:
 *   ./test_custom_sa --benchmark
 *
 *   # Run with validation:
 *   ./test_custom_sa --validate
 *
 * Copyright (c) 2025 DERO Miner Project
 * License: MIT
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <random>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>

extern "C" {
    #include "divsufsort.h"
    #include "custom_sa_70kb.h"
}

/* Test sizes to validate */
static const saidx_t TEST_SIZES[] = {
    2,          /* Minimum */
    10,
    256,
    1024,
    4096,
    16384,
    32768,
    65536,      /* 64KB */
    69371,      /* Common AstroBWT size */
    70000,
    70911,      /* Max AstroBWT size */
    71168,      /* Slightly over */
    72 * 1024   /* 72KB max for custom SA */
};
static const int NUM_TEST_SIZES = sizeof(TEST_SIZES) / sizeof(TEST_SIZES[0]);

/* Statistics */
static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

/* Command line options */
static bool opt_benchmark_only = false;
static bool opt_validate_only = false;
static int opt_benchmark_iterations = 1000;

/* Macro for assertions */
#define ASSERT(cond, msg) do { \
    tests_run++; \
    if (!(cond)) { \
        printf("FAIL: %s\n", msg); \
        tests_failed++; \
        return false; \
    } else { \
        tests_passed++; \
    } \
} while(0)

#define ASSERT_QUIET(cond) do { \
    tests_run++; \
    if (!(cond)) { \
        tests_failed++; \
        return false; \
    } else { \
        tests_passed++; \
    } \
} while(0)

/* Bucket arrays */
static saidx_t bucket_A[256];
static saidx_t bucket_B[256 * 256];

/* ========================================================================== */
/* Test Generators                                                            */
/* ========================================================================== */

static void generate_zeros(sauchar_t *data, saidx_t n) {
    memset(data, 0, n);
}

static void generate_same(sauchar_t *data, saidx_t n, sauchar_t value) {
    memset(data, value, n);
}

static void generate_increasing(sauchar_t *data, saidx_t n) {
    for (saidx_t i = 0; i < n; i++) {
        data[i] = (sauchar_t)(i & 0xFF);
    }
}

static void generate_decreasing(sauchar_t *data, saidx_t n) {
    for (saidx_t i = 0; i < n; i++) {
        data[i] = (sauchar_t)((n - 1 - i) & 0xFF);
    }
}

static void generate_random(sauchar_t *data, saidx_t n, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(0, 255);
    for (saidx_t i = 0; i < n; i++) {
        data[i] = (sauchar_t)dist(gen);
    }
}

static void generate_binary(sauchar_t *data, saidx_t n, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(0, 1);
    for (saidx_t i = 0; i < n; i++) {
        data[i] = (sauchar_t)dist(gen);
    }
}

static void generate_dna(sauchar_t *data, saidx_t n, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(0, 3);
    const char bases[] = {'A', 'C', 'G', 'T'};
    for (saidx_t i = 0; i < n; i++) {
        data[i] = (sauchar_t)bases[dist(gen)];
    }
}

static void generate_repetitive(sauchar_t *data, saidx_t n, uint32_t seed) {
    /* Generate data with many repeated patterns (stress test for LMS) */
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(0, 15);
    for (saidx_t i = 0; i < n; i++) {
        data[i] = (sauchar_t)dist(gen);
    }
}

/* ========================================================================== */
/* Test Cases                                                                 */
/* ========================================================================== */

static bool test_matches_divsufsort(const sauchar_t *data, saidx_t n, const char *desc) {
    std::vector<saidx_t> sa_custom(n);
    std::vector<saidx_t> sa_ref(n);

    /* Run custom SA */
    saint_t ret_custom = custom_sa_70kb(data, sa_custom.data(), n, bucket_A, bucket_B);
    ASSERT(ret_custom == 0, "custom_sa_70kb returned error");

    /* Run divsufsort reference */
    saint_t ret_ref = divsufsort(data, sa_ref.data(), n, bucket_A, bucket_B);
    ASSERT(ret_ref == 0, "divsufsort returned error");

    /* Compare byte-by-byte */
    bool match = (memcmp(sa_custom.data(), sa_ref.data(), n * sizeof(saidx_t)) == 0);

    if (!match) {
        /* Find first mismatch */
        for (saidx_t i = 0; i < n; i++) {
            if (sa_custom[i] != sa_ref[i]) {
                printf("  Mismatch at index %d: custom=%d, divsufsort=%d\n",
                       i, sa_custom[i], sa_ref[i]);
                break;
            }
        }
    }

    ASSERT(match, desc);
    return true;
}

static bool test_validation(const sauchar_t *data, saidx_t n) {
    std::vector<saidx_t> sa(n);

    saint_t ret = custom_sa_70kb(data, sa.data(), n, bucket_A, bucket_B);
    ASSERT(ret == 0, "custom_sa_70kb returned error");

    bool valid = custom_sa_validate(data, sa.data(), n);
    ASSERT(valid, "Suffix array validation failed");

    return true;
}

static bool test_edge_cases() {
    printf("Testing edge cases...\n");

    /* Empty (n=0) */
    {
        saidx_t sa[1] = {-1};
        saint_t ret = custom_sa_70kb(nullptr, sa, 0, bucket_A, bucket_B);
        ASSERT(ret == 0, "n=0 should succeed");
    }

    /* Single byte (n=1) */
    {
        sauchar_t data[1] = {42};
        saidx_t sa[1] = {-1};
        saint_t ret = custom_sa_70kb(data, sa, 1, bucket_A, bucket_B);
        ASSERT(ret == 0, "n=1 should succeed");
        ASSERT(sa[0] == 0, "n=1 SA[0] should be 0");
    }

    /* Two bytes, same (n=2) */
    {
        sauchar_t data[2] = {42, 42};
        saidx_t sa[2] = {-1, -1};
        saint_t ret = custom_sa_70kb(data, sa, 2, bucket_A, bucket_B);
        ASSERT(ret == 0, "n=2 same should succeed");
    }

    /* Two bytes, different (n=2) */
    {
        sauchar_t data[2] = {1, 2};
        saidx_t sa[2] = {-1, -1};
        saint_t ret = custom_sa_70kb(data, sa, 2, bucket_A, bucket_B);
        ASSERT(ret == 0, "n=2 different should succeed");
        ASSERT(sa[0] == 0, "n=2 ascending SA[0] should be 0");
        ASSERT(sa[1] == 1, "n=2 ascending SA[1] should be 1");
    }

    /* Two bytes, reverse (n=2) */
    {
        sauchar_t data[2] = {2, 1};
        saidx_t sa[2] = {-1, -1};
        saint_t ret = custom_sa_70kb(data, sa, 2, bucket_A, bucket_B);
        ASSERT(ret == 0, "n=2 descending should succeed");
        ASSERT(sa[0] == 1, "n=2 descending SA[0] should be 1");
        ASSERT(sa[1] == 0, "n=2 descending SA[1] should be 0");
    }

    /* Invalid parameters */
    {
        sauchar_t data[10];
        saidx_t sa[10];

        saint_t ret = custom_sa_70kb(nullptr, sa, 10, bucket_A, bucket_B);
        ASSERT(ret == -1, "nullptr T should return -1");

        ret = custom_sa_70kb(data, nullptr, 10, bucket_A, bucket_B);
        ASSERT(ret == -1, "nullptr SA should return -1");

        ret = custom_sa_70kb(data, sa, -1, bucket_A, bucket_B);
        ASSERT(ret == -1, "negative n should return -1");

        ret = custom_sa_70kb(data, sa, 10, nullptr, bucket_B);
        ASSERT(ret == -2, "nullptr bucket_A should return -2");

        ret = custom_sa_70kb(data, sa, 10, bucket_A, nullptr);
        ASSERT(ret == -2, "nullptr bucket_B should return -2");
    }

    printf("  Edge cases: PASSED\n");
    return true;
}

static bool test_size_range() {
    printf("Testing size range...\n");

    for (int i = 0; i < NUM_TEST_SIZES; i++) {
        saidx_t n = TEST_SIZES[i];
        printf("  Size %d... ", n);
        fflush(stdout);

        std::vector<sauchar_t> data(n);
        generate_random(data.data(), n, 12345 + i);

        if (!test_matches_divsufsort(data.data(), n, "size range test")) {
            printf("FAILED\n");
            return false;
        }
        printf("OK\n");
    }

    return true;
}

static bool test_patterns() {
    printf("Testing data patterns...\n");

    const saidx_t n = 70000;
    std::vector<sauchar_t> data(n);

    /* All zeros */
    printf("  All zeros... ");
    fflush(stdout);
    generate_zeros(data.data(), n);
    if (!test_matches_divsufsort(data.data(), n, "all zeros")) {
        printf("FAILED\n");
        return false;
    }
    printf("OK\n");

    /* All same (non-zero) */
    printf("  All 0x42... ");
    fflush(stdout);
    generate_same(data.data(), n, 0x42);
    if (!test_matches_divsufsort(data.data(), n, "all same")) {
        printf("FAILED\n");
        return false;
    }
    printf("OK\n");

    /* Increasing */
    printf("  Increasing... ");
    fflush(stdout);
    generate_increasing(data.data(), n);
    if (!test_matches_divsufsort(data.data(), n, "increasing")) {
        printf("FAILED\n");
        return false;
    }
    printf("OK\n");

    /* Decreasing */
    printf("  Decreasing... ");
    fflush(stdout);
    generate_decreasing(data.data(), n);
    if (!test_matches_divsufsort(data.data(), n, "decreasing")) {
        printf("FAILED\n");
        return false;
    }
    printf("OK\n");

    /* Binary alphabet */
    printf("  Binary alphabet... ");
    fflush(stdout);
    generate_binary(data.data(), n, 54321);
    if (!test_matches_divsufsort(data.data(), n, "binary")) {
        printf("FAILED\n");
        return false;
    }
    printf("OK\n");

    /* DNA alphabet */
    printf("  DNA alphabet... ");
    fflush(stdout);
    generate_dna(data.data(), n, 98765);
    if (!test_matches_divsufsort(data.data(), n, "DNA")) {
        printf("FAILED\n");
        return false;
    }
    printf("OK\n");

    /* Repetitive (stress test for LMS) */
    printf("  Repetitive (16-char alphabet)... ");
    fflush(stdout);
    generate_repetitive(data.data(), n, 11111);
    if (!test_matches_divsufsort(data.data(), n, "repetitive")) {
        printf("FAILED\n");
        return false;
    }
    printf("OK\n");

    /* Random */
    for (int seed = 1; seed <= 5; seed++) {
        printf("  Random (seed %d)... ", seed);
        fflush(stdout);
        generate_random(data.data(), n, seed);
        if (!test_matches_divsufsort(data.data(), n, "random")) {
            printf("FAILED\n");
            return false;
        }
        printf("OK\n");
    }

    return true;
}

static bool test_astrobwt_sizes() {
    printf("Testing AstroBWT-specific sizes...\n");

    /* Common AstroBWT output sizes */
    const saidx_t astro_sizes[] = {
        65536,      /* Minimum possible */
        69371,      /* Common */
        70000,      /* Common */
        70500,      /* Common */
        70911       /* Maximum */
    };

    for (int i = 0; i < 5; i++) {
        saidx_t n = astro_sizes[i];
        printf("  AstroBWT size %d... ", n);
        fflush(stdout);

        std::vector<sauchar_t> data(n);
        generate_random(data.data(), n, 1000 + i);

        if (!test_matches_divsufsort(data.data(), n, "AstroBWT size")) {
            printf("FAILED\n");
            return false;
        }
        printf("OK\n");
    }

    return true;
}

/* ========================================================================== */
/* Extended Benchmark Suite                                                   */
/* ========================================================================== */

struct BenchmarkResult {
    double mean_us;
    double stddev_us;
    double min_us;
    double max_us;
    double median_us;
};

static BenchmarkResult run_benchmark(
    void (*sa_func)(const sauchar_t*, saidx_t*, saidx_t, saidx_t*, saidx_t*),
    const sauchar_t* data,
    saidx_t n,
    int iterations
) {
    std::vector<double> times;
    times.reserve(iterations);

    std::vector<saidx_t> sa(n);

    /* Warmup */
    for (int i = 0; i < 10; i++) {
        sa_func(data, sa.data(), n, bucket_A, bucket_B);
    }

    /* Benchmark */
    for (int i = 0; i < iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        sa_func(data, sa.data(), n, bucket_A, bucket_B);
        auto end = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(end - start).count();
        times.push_back(us);
    }

    /* Calculate statistics */
    std::sort(times.begin(), times.end());

    double sum = 0;
    for (double t : times) sum += t;
    double mean = sum / times.size();

    double sq_sum = 0;
    for (double t : times) sq_sum += (t - mean) * (t - mean);
    double stddev = std::sqrt(sq_sum / times.size());

    return {
        mean,
        stddev,
        times.front(),
        times.back(),
        times[times.size() / 2]
    };
}

/* Wrapper for custom_sa_70kb */
static void custom_sa_wrapper(const sauchar_t* T, saidx_t* SA, saidx_t n, saidx_t* bA, saidx_t* bB) {
    custom_sa_70kb(T, SA, n, bA, bB);
}

/* Wrapper for divsufsort */
static void divsufsort_wrapper(const sauchar_t* T, saidx_t* SA, saidx_t n, saidx_t* bA, saidx_t* bB) {
    divsufsort(T, SA, n, bA, bB);
}

static bool test_performance_comparison() {
    printf("\n=== Performance Benchmark ===\n\n");

    const saidx_t test_sizes[] = {65536, 70000, 70911};
    const int iterations = opt_benchmark_iterations;

    printf("Build configuration:\n");
#if ENABLE_SA_AVX2
    printf("  AVX2:         ENABLED\n");
#else
    printf("  AVX2:         DISABLED\n");
#endif
#if ENABLE_SA_PREFETCH
    printf("  Prefetch:     ENABLED (SA=%d, Text=%d, Bucket=%d)\n",
           CUSTOM_SA_PREFETCH_DISTANCE, CUSTOM_TEXT_PREFETCH_DISTANCE, CUSTOM_BUCKET_PREFETCH_DISTANCE);
#else
    printf("  Prefetch:     DISABLED\n");
#endif
#if ENABLE_SA_8WAY_UNROLL
    printf("  8-way unroll: ENABLED\n");
#else
    printf("  8-way unroll: DISABLED\n");
#endif
#if ENABLE_SA_BRANCHLESS
    printf("  Branchless:   ENABLED\n");
#else
    printf("  Branchless:   DISABLED\n");
#endif
    printf("\n");

    printf("Iterations: %d\n\n", iterations);
    printf("%-10s %-15s %-15s %-10s %-10s\n",
           "Size", "custom_sa", "divsufsort", "Speedup", "Improv");
    printf("%-10s %-15s %-15s %-10s %-10s\n",
           "----", "---------", "----------", "-------", "------");

    for (saidx_t n : test_sizes) {
        std::vector<sauchar_t> data(n);
        generate_random(data.data(), n, 42);

        auto custom_result = run_benchmark(custom_sa_wrapper, data.data(), n, iterations);
        auto divsufsort_result = run_benchmark(divsufsort_wrapper, data.data(), n, iterations);

        double speedup = divsufsort_result.mean_us / custom_result.mean_us;
        double improvement = (speedup - 1.0) * 100.0;

        printf("%-10d %-15.2f %-15.2f %-10.3fx %+.1f%%\n",
               n,
               custom_result.mean_us,
               divsufsort_result.mean_us,
               speedup,
               improvement);
    }

    printf("\n");

    /* Detailed benchmark for primary size (70000) */
    printf("Detailed results for n=70000:\n");
    const saidx_t n = 70000;
    std::vector<sauchar_t> data(n);
    generate_random(data.data(), n, 42);

    auto custom_result = run_benchmark(custom_sa_wrapper, data.data(), n, iterations);
    auto divsufsort_result = run_benchmark(divsufsort_wrapper, data.data(), n, iterations);

    printf("\n  custom_sa_70kb:\n");
    printf("    Mean:   %.2f us\n", custom_result.mean_us);
    printf("    Stddev: %.2f us\n", custom_result.stddev_us);
    printf("    Min:    %.2f us\n", custom_result.min_us);
    printf("    Max:    %.2f us\n", custom_result.max_us);
    printf("    Median: %.2f us\n", custom_result.median_us);

    printf("\n  divsufsort:\n");
    printf("    Mean:   %.2f us\n", divsufsort_result.mean_us);
    printf("    Stddev: %.2f us\n", divsufsort_result.stddev_us);
    printf("    Min:    %.2f us\n", divsufsort_result.min_us);
    printf("    Max:    %.2f us\n", divsufsort_result.max_us);
    printf("    Median: %.2f us\n", divsufsort_result.median_us);

    double speedup = divsufsort_result.mean_us / custom_result.mean_us;
    printf("\n  Overall speedup: %.3fx (%+.1f%%)\n", speedup, (speedup - 1.0) * 100.0);

    /* Throughput calculation */
    double custom_throughput = (n / 1024.0) / (custom_result.mean_us / 1e6);
    double divsufsort_throughput = (n / 1024.0) / (divsufsort_result.mean_us / 1e6);
    printf("\n  Throughput:\n");
    printf("    custom_sa_70kb: %.2f KB/s\n", custom_throughput);
    printf("    divsufsort:     %.2f KB/s\n", divsufsort_throughput);

    return true;
}

static bool test_pattern_performance() {
    printf("\n=== Pattern-Specific Performance ===\n\n");

    const saidx_t n = 70000;
    const int iterations = opt_benchmark_iterations / 10;  /* Fewer iterations per pattern */

    struct PatternTest {
        const char* name;
        void (*generator)(sauchar_t*, saidx_t, uint32_t);
        uint32_t seed;
    };

    /* Custom generator wrappers */
    auto gen_zeros = [](sauchar_t* d, saidx_t n, uint32_t) { generate_zeros(d, n); };
    auto gen_increasing = [](sauchar_t* d, saidx_t n, uint32_t) { generate_increasing(d, n); };
    auto gen_random = [](sauchar_t* d, saidx_t n, uint32_t s) { generate_random(d, n, s); };
    auto gen_binary = [](sauchar_t* d, saidx_t n, uint32_t s) { generate_binary(d, n, s); };
    auto gen_dna = [](sauchar_t* d, saidx_t n, uint32_t s) { generate_dna(d, n, s); };
    auto gen_repetitive = [](sauchar_t* d, saidx_t n, uint32_t s) { generate_repetitive(d, n, s); };

    printf("%-20s %-12s %-12s %-10s\n", "Pattern", "custom_sa", "divsufsort", "Speedup");
    printf("%-20s %-12s %-12s %-10s\n", "-------", "---------", "----------", "-------");

    std::vector<sauchar_t> data(n);

    /* All zeros */
    generate_zeros(data.data(), n);
    auto r1 = run_benchmark(custom_sa_wrapper, data.data(), n, iterations);
    auto r2 = run_benchmark(divsufsort_wrapper, data.data(), n, iterations);
    printf("%-20s %-12.2f %-12.2f %.3fx\n", "All zeros", r1.mean_us, r2.mean_us, r2.mean_us/r1.mean_us);

    /* Increasing */
    generate_increasing(data.data(), n);
    r1 = run_benchmark(custom_sa_wrapper, data.data(), n, iterations);
    r2 = run_benchmark(divsufsort_wrapper, data.data(), n, iterations);
    printf("%-20s %-12.2f %-12.2f %.3fx\n", "Increasing", r1.mean_us, r2.mean_us, r2.mean_us/r1.mean_us);

    /* Binary */
    generate_binary(data.data(), n, 42);
    r1 = run_benchmark(custom_sa_wrapper, data.data(), n, iterations);
    r2 = run_benchmark(divsufsort_wrapper, data.data(), n, iterations);
    printf("%-20s %-12.2f %-12.2f %.3fx\n", "Binary (2-char)", r1.mean_us, r2.mean_us, r2.mean_us/r1.mean_us);

    /* DNA */
    generate_dna(data.data(), n, 42);
    r1 = run_benchmark(custom_sa_wrapper, data.data(), n, iterations);
    r2 = run_benchmark(divsufsort_wrapper, data.data(), n, iterations);
    printf("%-20s %-12.2f %-12.2f %.3fx\n", "DNA (4-char)", r1.mean_us, r2.mean_us, r2.mean_us/r1.mean_us);

    /* Repetitive */
    generate_repetitive(data.data(), n, 42);
    r1 = run_benchmark(custom_sa_wrapper, data.data(), n, iterations);
    r2 = run_benchmark(divsufsort_wrapper, data.data(), n, iterations);
    printf("%-20s %-12.2f %-12.2f %.3fx\n", "Repetitive (16-char)", r1.mean_us, r2.mean_us, r2.mean_us/r1.mean_us);

    /* Random */
    generate_random(data.data(), n, 42);
    r1 = run_benchmark(custom_sa_wrapper, data.data(), n, iterations);
    r2 = run_benchmark(divsufsort_wrapper, data.data(), n, iterations);
    printf("%-20s %-12.2f %-12.2f %.3fx\n", "Random (256-char)", r1.mean_us, r2.mean_us, r2.mean_us/r1.mean_us);

    printf("\n");
    return true;
}

/* ========================================================================== */
/* Main Test Driver                                                           */
/* ========================================================================== */

static void print_usage(const char* program) {
    printf("Usage: %s [options]\n", program);
    printf("\nOptions:\n");
    printf("  --benchmark        Run benchmark only (skip correctness tests)\n");
    printf("  --validate         Run validation/correctness tests only\n");
    printf("  --iterations N     Set benchmark iterations (default: 1000)\n");
    printf("  --help             Show this help message\n");
}

int main(int argc, char *argv[]) {
    /* Parse command line arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--benchmark") == 0) {
            opt_benchmark_only = true;
        } else if (strcmp(argv[i], "--validate") == 0) {
            opt_validate_only = true;
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            opt_benchmark_iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    printf("=== Custom SA 70KB Unit Tests ===\n\n");

    bool all_passed = true;

    if (!opt_benchmark_only) {
        /* Run correctness tests */
        all_passed &= test_edge_cases();
        all_passed &= test_size_range();
        all_passed &= test_patterns();
        all_passed &= test_astrobwt_sizes();
    }

    if (!opt_validate_only) {
        /* Run performance benchmarks */
        all_passed &= test_performance_comparison();
        all_passed &= test_pattern_performance();
    }

    /* Summary */
    if (!opt_benchmark_only) {
        printf("\n=== Test Summary ===\n");
        printf("Tests run:    %d\n", tests_run);
        printf("Tests passed: %d\n", tests_passed);
        printf("Tests failed: %d\n", tests_failed);

        if (all_passed && tests_failed == 0) {
            printf("\nAll tests PASSED!\n");
        } else {
            printf("\nSome tests FAILED!\n");
        }
    }

    return (all_passed && tests_failed == 0) ? 0 : 1;
}
