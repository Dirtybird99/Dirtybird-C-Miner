/**
 * Suffix Array Algorithm Comparison Benchmark
 *
 * Compares libsais vs divsufsort for AstroBWTv3 workload:
 * - Fixed ~70KB input size (post-Salsa20 random data)
 * - Measures SA construction time only
 * - Tests different fs (free space) parameters for libsais
 */

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <vector>

// libsais
extern "C" {
#include "libsais.h"
}

// divsufsort
extern "C" {
#include "divsufsort.h"
}

// Test parameters
constexpr int WARMUP_ITERATIONS = 100;
constexpr int BENCHMARK_ITERATIONS = 1000;
constexpr int DATA_SIZE = 70 * 1024;  // ~70KB like AstroBWTv3

// Generate random data (simulates post-Salsa20 output)
void generate_random_data(uint8_t* data, size_t len) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    for (size_t i = 0; i < len; i++) {
        data[i] = static_cast<uint8_t>(dis(gen));
    }
}

// Benchmark libsais with context reuse
double bench_libsais_ctx(const uint8_t* data, int32_t* sa, int n, int fs, int iterations) {
    void* ctx = libsais_create_ctx();
    if (!ctx) {
        fprintf(stderr, "Failed to create libsais context\n");
        return -1;
    }

    // Warmup
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        libsais_ctx(ctx, data, sa, n, fs, nullptr);
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        libsais_ctx(ctx, data, sa, n, fs, nullptr);
    }
    auto end = std::chrono::high_resolution_clock::now();

    libsais_free_ctx(ctx);

    return std::chrono::duration<double, std::micro>(end - start).count() / iterations;
}

// Benchmark libsais without context (fresh allocation each time)
double bench_libsais_no_ctx(const uint8_t* data, int32_t* sa, int n, int fs, int iterations) {
    // Warmup
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        libsais(data, sa, n, fs, nullptr);
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        libsais(data, sa, n, fs, nullptr);
    }
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::micro>(end - start).count() / iterations;
}

// Benchmark divsufsort (note: dirtybird's modified divsufsort needs bucket arrays)
double bench_divsufsort(const uint8_t* data, int32_t* sa, int n, int iterations) {
    // Allocate bucket arrays (required by dirtybird's modified divsufsort)
    int32_t bucket_A[256];
    int32_t* bucket_B = new int32_t[256 * 256];

    // Warmup
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        divsufsort(data, sa, n, bucket_A, bucket_B);
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        divsufsort(data, sa, n, bucket_A, bucket_B);
    }
    auto end = std::chrono::high_resolution_clock::now();

    delete[] bucket_B;
    return std::chrono::duration<double, std::micro>(end - start).count() / iterations;
}

// Verify SA correctness
bool verify_sa(const uint8_t* data, const int32_t* sa, int n) {
    // Check that SA is a permutation of [0, n)
    std::vector<bool> seen(n, false);
    for (int i = 0; i < n; i++) {
        if (sa[i] < 0 || sa[i] >= n) return false;
        if (seen[sa[i]]) return false;
        seen[sa[i]] = true;
    }

    // Check lexicographic order
    for (int i = 1; i < n; i++) {
        int cmp = memcmp(data + sa[i-1], data + sa[i],
                         std::min(n - sa[i-1], n - sa[i]));
        if (cmp > 0) return false;
        if (cmp == 0 && sa[i-1] > sa[i]) return false;
    }

    return true;
}

int main() {
    printf("=== Suffix Array Algorithm Comparison ===\n");
    printf("Data size: %d bytes (~%.1f KB)\n", DATA_SIZE, DATA_SIZE / 1024.0);
    printf("Warmup: %d iterations\n", WARMUP_ITERATIONS);
    printf("Benchmark: %d iterations\n\n", BENCHMARK_ITERATIONS);

    // Allocate buffers
    uint8_t* data = new uint8_t[DATA_SIZE];
    int32_t* sa = new int32_t[DATA_SIZE + 6144];  // Extra space for fs parameter testing

    // Generate random data
    generate_random_data(data, DATA_SIZE);

    printf("Testing with random data (simulates post-Salsa20)...\n\n");

    // Test divsufsort
    printf("1. divsufsort:\n");
    double divsufsort_time = bench_divsufsort(data, sa, DATA_SIZE, BENCHMARK_ITERATIONS);
    bool divsufsort_correct = verify_sa(data, sa, DATA_SIZE);
    printf("   Time: %.2f us/call\n", divsufsort_time);
    printf("   Correct: %s\n", divsufsort_correct ? "YES" : "NO");
    printf("   Throughput: %.2f MB/s\n\n", (DATA_SIZE / 1024.0 / 1024.0) / (divsufsort_time / 1e6));

    // Test libsais with context, fs=0
    printf("2. libsais (ctx, fs=0):\n");
    double libsais_ctx_fs0 = bench_libsais_ctx(data, sa, DATA_SIZE, 0, BENCHMARK_ITERATIONS);
    bool libsais_correct = verify_sa(data, sa, DATA_SIZE);
    printf("   Time: %.2f us/call\n", libsais_ctx_fs0);
    printf("   Correct: %s\n", libsais_correct ? "YES" : "NO");
    printf("   Throughput: %.2f MB/s\n", (DATA_SIZE / 1024.0 / 1024.0) / (libsais_ctx_fs0 / 1e6));
    printf("   vs divsufsort: %.1f%%\n\n", (divsufsort_time / libsais_ctx_fs0 - 1) * 100);

    // Test libsais with context, fs=256
    printf("3. libsais (ctx, fs=256):\n");
    double libsais_ctx_fs256 = bench_libsais_ctx(data, sa, DATA_SIZE, 256, BENCHMARK_ITERATIONS);
    printf("   Time: %.2f us/call\n", libsais_ctx_fs256);
    printf("   vs fs=0: %.1f%%\n\n", (libsais_ctx_fs0 / libsais_ctx_fs256 - 1) * 100);

    // Test libsais with context, fs=4096 (recommended in docs)
    printf("4. libsais (ctx, fs=4096):\n");
    double libsais_ctx_fs4k = bench_libsais_ctx(data, sa, DATA_SIZE, 4096, BENCHMARK_ITERATIONS);
    printf("   Time: %.2f us/call\n", libsais_ctx_fs4k);
    printf("   vs fs=0: %.1f%%\n\n", (libsais_ctx_fs0 / libsais_ctx_fs4k - 1) * 100);

    // Test libsais with context, fs=6144 (optimal per docs)
    printf("5. libsais (ctx, fs=6144):\n");
    double libsais_ctx_fs6k = bench_libsais_ctx(data, sa, DATA_SIZE, 6144, BENCHMARK_ITERATIONS);
    printf("   Time: %.2f us/call\n", libsais_ctx_fs6k);
    printf("   vs fs=0: %.1f%%\n\n", (libsais_ctx_fs0 / libsais_ctx_fs6k - 1) * 100);

    // Test libsais without context (allocation overhead)
    printf("6. libsais (no ctx, fs=0):\n");
    double libsais_no_ctx = bench_libsais_no_ctx(data, sa, DATA_SIZE, 0, BENCHMARK_ITERATIONS);
    printf("   Time: %.2f us/call\n", libsais_no_ctx);
    printf("   vs with ctx: %.1f%% slower (allocation overhead)\n\n", (libsais_no_ctx / libsais_ctx_fs0 - 1) * 100);

    // Summary
    printf("=== Summary ===\n");
    printf("Best libsais config: ");
    double best_libsais = std::min({libsais_ctx_fs0, libsais_ctx_fs256, libsais_ctx_fs4k, libsais_ctx_fs6k});
    if (best_libsais == libsais_ctx_fs0) printf("fs=0");
    else if (best_libsais == libsais_ctx_fs256) printf("fs=256");
    else if (best_libsais == libsais_ctx_fs4k) printf("fs=4096");
    else printf("fs=6144");
    printf(" at %.2f us/call\n", best_libsais);

    printf("\nOverall winner: ");
    if (best_libsais < divsufsort_time) {
        printf("libsais (%.1f%% faster than divsufsort)\n", (divsufsort_time / best_libsais - 1) * 100);
    } else {
        printf("divsufsort (%.1f%% faster than libsais)\n", (best_libsais / divsufsort_time - 1) * 100);
    }

    // Impact on hash rate estimation
    printf("\n=== Hash Rate Impact Estimation ===\n");
    double sa_time_baseline = best_libsais;
    double total_hash_time_us = 1000.0;  // Assume 1ms per hash (1 KH/s baseline)
    double sa_fraction = 0.05;  // SA is ~5% of hash time
    double sa_time_in_hash = total_hash_time_us * sa_fraction;

    printf("If SA construction is 5%% of hash time:\n");
    printf("  Current SA time in hash: %.2f us\n", sa_time_in_hash);
    printf("  If we save 10%% on SA: %.2f us saved\n", sa_time_in_hash * 0.1);
    printf("  Hash time impact: %.2f%% faster\n", (sa_time_in_hash * 0.1 / total_hash_time_us) * 100);

    delete[] data;
    delete[] sa;

    return 0;
}
