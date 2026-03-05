/**
 * Benchmark: Scalar vs SIMD Salsa20 implementations
 *
 * Compares the performance of:
 * 1. ucstk::Salsa20 (current scalar implementation)
 * 2. simd::Salsa20 (new SIMD implementation)
 *
 * Build:
 *   cd build-optimized
 *   cmake --build . --target bench-salsa20-simd
 *
 * Or manually:
 *   clang++ -O3 -mavx2 -I../include -o bench_salsa20_simd.exe \
 *       src/bin/bench_salsa20_simd.cpp src/crypto/salsa20_simd.c
 */

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>

// Include both implementations
#include "Salsa20.h"      // Original scalar
#include "salsa20_simd.h" // New SIMD

// Number of iterations for benchmarking
constexpr int WARMUP_ITERS = 1000;
constexpr int BENCH_ITERS = 100000;
constexpr int BYTES_PER_ITER = 256; // Same as AstroBWT usage

// Test key and IV
static const uint8_t TEST_KEY[32] = {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
    0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f
};

static const uint8_t TEST_IV[8] = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};

// Benchmark result structure
struct BenchResult {
    const char* name;
    double ns_per_iter;
    double mb_per_sec;
    double speedup;
};

// Benchmark the original scalar implementation
double bench_scalar(int iterations) {
    ucstk::Salsa20 salsa(TEST_KEY);
    salsa.setIv(TEST_IV);

    uint8_t input[256] = {0};
    uint8_t output[256];

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        salsa.setIv(TEST_IV); // Reset IV each iteration (like AstroBWT)
        salsa.processBytes(input, output, BYTES_PER_ITER);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    return (double)duration.count() / iterations;
}

// Benchmark the new SIMD implementation
double bench_simd(int iterations) {
    uint8_t input[256] = {0};
    uint8_t output[256];

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        salsa20_simd_process(TEST_KEY, TEST_IV, input, output, BYTES_PER_ITER);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    return (double)duration.count() / iterations;
}

// Benchmark SIMD C++ wrapper class
double bench_simd_class(int iterations) {
    simd::Salsa20 salsa(TEST_KEY);
    salsa.setIv(TEST_IV);

    uint8_t input[256] = {0};
    uint8_t output[256];

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        salsa.setIv(TEST_IV);
        salsa.processBytes(input, output, BYTES_PER_ITER);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    return (double)duration.count() / iterations;
}

// Verify correctness - both implementations should produce same output
bool verify_correctness() {
    uint8_t input[256] = {0};
    uint8_t output_scalar[256];
    uint8_t output_simd[256];

    // Run scalar
    ucstk::Salsa20 scalar_salsa(TEST_KEY);
    scalar_salsa.setIv(TEST_IV);
    scalar_salsa.processBytes(input, output_scalar, 256);

    // Run SIMD
    salsa20_simd_process(TEST_KEY, TEST_IV, input, output_simd, 256);

    // Compare
    bool match = memcmp(output_scalar, output_simd, 256) == 0;

    if (!match) {
        printf("MISMATCH! First differing bytes:\n");
        for (int i = 0; i < 256; i++) {
            if (output_scalar[i] != output_simd[i]) {
                printf("  [%d] scalar=0x%02x simd=0x%02x\n",
                       i, output_scalar[i], output_simd[i]);
                if (i > 10) break;
            }
        }
    }

    return match;
}

int main() {
    printf("======================================================================\n");
    printf("SALSA20 SIMD BENCHMARK\n");
    printf("======================================================================\n\n");

    // Initialize SIMD detection
    salsa20_simd_init();
    printf("SIMD Implementation: %s\n", salsa20_simd_impl_name());
    printf("Bytes per iteration: %d\n", BYTES_PER_ITER);
    printf("Benchmark iterations: %d\n\n", BENCH_ITERS);

    // Verify correctness first
    printf("Verifying correctness... ");
    if (verify_correctness()) {
        printf("PASS\n\n");
    } else {
        printf("FAIL\n");
        printf("ERROR: SIMD output does not match scalar output!\n");
        return 1;
    }

    // Warmup
    printf("Warming up (%d iterations)...\n", WARMUP_ITERS);
    bench_scalar(WARMUP_ITERS);
    bench_simd(WARMUP_ITERS);
    bench_simd_class(WARMUP_ITERS);

    // Run benchmarks
    printf("Running benchmarks...\n\n");

    std::vector<BenchResult> results;

    // Scalar
    double scalar_ns = bench_scalar(BENCH_ITERS);
    results.push_back({
        "Scalar (ucstk::Salsa20)",
        scalar_ns,
        (BYTES_PER_ITER / scalar_ns) * 1000.0, // MB/s
        1.0
    });

    // SIMD C API
    double simd_ns = bench_simd(BENCH_ITERS);
    results.push_back({
        "SIMD (C API)",
        simd_ns,
        (BYTES_PER_ITER / simd_ns) * 1000.0,
        scalar_ns / simd_ns
    });

    // SIMD C++ wrapper
    double simd_class_ns = bench_simd_class(BENCH_ITERS);
    results.push_back({
        "SIMD (C++ class)",
        simd_class_ns,
        (BYTES_PER_ITER / simd_class_ns) * 1000.0,
        scalar_ns / simd_class_ns
    });

    // Print results
    printf("%-25s %12s %12s %10s\n", "Implementation", "ns/iter", "MB/s", "Speedup");
    printf("------------------------------------------------------------\n");

    for (const auto& r : results) {
        printf("%-25s %12.1f %12.1f %9.2fx\n",
               r.name, r.ns_per_iter, r.mb_per_sec, r.speedup);
    }

    printf("\n");

    // Summary
    if (results[1].speedup > 1.0) {
        printf("SIMD is %.1f%% faster than scalar\n",
               (results[1].speedup - 1.0) * 100.0);
    } else {
        printf("WARNING: SIMD is slower than scalar!\n");
    }

    // Estimate impact on full hash
    // Salsa20 is ~5% of total AstroBWT time
    double salsa_fraction = 0.05;
    double hash_speedup = 1.0 / (1.0 - salsa_fraction + salsa_fraction / results[1].speedup);
    printf("Estimated AstroBWT impact: +%.2f%% hashrate\n",
           (hash_speedup - 1.0) * 100.0);

    return 0;
}
