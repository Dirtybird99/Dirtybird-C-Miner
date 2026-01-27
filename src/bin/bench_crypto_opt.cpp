/**
 * Benchmark for optimized RC4 and Salsa20 implementations
 *
 * Tests:
 * 1. Correctness: Verify optimized implementations match reference
 * 2. Performance: Compare optimized vs original implementations
 */

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <random>

// Include implementations
#include "salsa20_simd.h"
#include "Salsa20.h"
#include "rc4_optimized.hpp"
#include "crypto/astrobwtv3/rc4_avx512.hpp"

#include <openssl/rc4.h>

using namespace std::chrono;

// =============================================================================
// Test Utilities
// =============================================================================

void print_hex(const char* label, const uint8_t* data, size_t len) {
    printf("%s: ", label);
    for (size_t i = 0; i < len && i < 32; i++) {
        printf("%02x", data[i]);
    }
    if (len > 32) printf("...");
    printf("\n");
}

bool compare_buffers(const uint8_t* a, const uint8_t* b, size_t len) {
    for (size_t i = 0; i < len; i++) {
        if (a[i] != b[i]) {
            printf("Mismatch at offset %zu: expected %02x, got %02x\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

// =============================================================================
// Salsa20 Correctness Test
// =============================================================================

bool test_salsa20_correctness() {
    printf("\n=== Salsa20 Correctness Test ===\n");

    // Initialize SIMD detection
    salsa20_simd_init();
    printf("SIMD implementation: %s\n", salsa20_simd_impl_name());

    // Test key and IV
    uint8_t key[32], iv[8], input[256], output_ref[256], output_simd[256];

    // Initialize with pseudo-random data
    std::mt19937 rng(12345);
    for (int i = 0; i < 32; i++) key[i] = rng() & 0xFF;
    for (int i = 0; i < 8; i++) iv[i] = rng() & 0xFF;
    for (int i = 0; i < 256; i++) input[i] = rng() & 0xFF;

    // Reference implementation (ucstk::Salsa20)
    ucstk::Salsa20 ref_salsa20(key);
    ref_salsa20.setIv(iv);
    ref_salsa20.processBytes(input, output_ref, 256);

    // SIMD implementation
    salsa20_simd_process(key, iv, input, output_simd, 256);

    // Compare results
    bool success = compare_buffers(output_ref, output_simd, 256);
    if (success) {
        printf("PASS: Salsa20 SIMD matches reference\n");
    } else {
        printf("FAIL: Salsa20 SIMD does not match reference\n");
        print_hex("Reference", output_ref, 256);
        print_hex("SIMD", output_simd, 256);
    }

    return success;
}

// =============================================================================
// RC4 Correctness Test
// =============================================================================

bool test_rc4_correctness() {
    printf("\n=== RC4 Correctness Test ===\n");

    // Test key and data
    uint8_t key[256], data_ref[256], data_opt[256];

    // Initialize with pseudo-random data
    std::mt19937 rng(54321);
    for (int i = 0; i < 256; i++) {
        key[i] = rng() & 0xFF;
        data_ref[i] = rng() & 0xFF;
    }
    memcpy(data_opt, data_ref, 256);

    // Reference implementation (OpenSSL RC4)
    RC4_KEY rc4_key;
    RC4_set_key(&rc4_key, 256, key);
    RC4(&rc4_key, 256, data_ref, data_ref);

    // Optimized implementation
    rc4_opt::OptimizedRc4 opt_rc4;
    opt_rc4.set_key(key, 256);
    opt_rc4.apply_keystream_256(data_opt);

    // Compare results
    bool success = compare_buffers(data_ref, data_opt, 256);
    if (success) {
        printf("PASS: Optimized RC4 matches OpenSSL\n");
    } else {
        printf("FAIL: Optimized RC4 does not match OpenSSL\n");
        print_hex("OpenSSL", data_ref, 256);
        print_hex("Optimized", data_opt, 256);
    }

    return success;
}

// =============================================================================
// Salsa20 Performance Test
// =============================================================================

void bench_salsa20_performance() {
    printf("\n=== Salsa20 Performance Benchmark ===\n");

    const int ITERATIONS = 100000;

    uint8_t key[32], iv[8], input[256], output[256];
    std::mt19937 rng(99999);
    for (int i = 0; i < 32; i++) key[i] = rng() & 0xFF;
    for (int i = 0; i < 8; i++) iv[i] = rng() & 0xFF;
    for (int i = 0; i < 256; i++) input[i] = rng() & 0xFF;

    // Benchmark reference (ucstk::Salsa20)
    {
        auto start = high_resolution_clock::now();
        for (int i = 0; i < ITERATIONS; i++) {
            ucstk::Salsa20 salsa20(key);
            salsa20.setIv(iv);
            salsa20.processBytes(input, output, 256);
        }
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start).count();
        double ops_per_sec = (double)ITERATIONS / (duration / 1e6);
        printf("ucstk::Salsa20:    %.2f ops/sec (%.2f MB/s)\n",
               ops_per_sec, ops_per_sec * 256 / 1e6);
    }

    // Benchmark SIMD
    {
        salsa20_simd_init();
        auto start = high_resolution_clock::now();
        for (int i = 0; i < ITERATIONS; i++) {
            salsa20_simd_process(key, iv, input, output, 256);
        }
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start).count();
        double ops_per_sec = (double)ITERATIONS / (duration / 1e6);
        printf("salsa20_simd:      %.2f ops/sec (%.2f MB/s) [%s]\n",
               ops_per_sec, ops_per_sec * 256 / 1e6, salsa20_simd_impl_name());
    }
}

// =============================================================================
// RC4 Performance Test
// =============================================================================

void bench_rc4_performance() {
    printf("\n=== RC4 Performance Benchmark ===\n");

    const int ITERATIONS = 100000;

    uint8_t key[256], data[256];
    std::mt19937 rng(88888);
    for (int i = 0; i < 256; i++) {
        key[i] = rng() & 0xFF;
        data[i] = rng() & 0xFF;
    }

    // Benchmark OpenSSL RC4
    {
        auto start = high_resolution_clock::now();
        for (int i = 0; i < ITERATIONS; i++) {
            RC4_KEY rc4_key;
            uint8_t tmp[256];
            memcpy(tmp, data, 256);
            RC4_set_key(&rc4_key, 256, key);
            RC4(&rc4_key, 256, tmp, tmp);
        }
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start).count();
        double ops_per_sec = (double)ITERATIONS / (duration / 1e6);
        printf("OpenSSL RC4:       %.2f ops/sec (%.2f MB/s)\n",
               ops_per_sec, ops_per_sec * 256 / 1e6);
    }

    // Benchmark Optimized RC4
    {
        auto start = high_resolution_clock::now();
        for (int i = 0; i < ITERATIONS; i++) {
            rc4_opt::OptimizedRc4 rc4;
            uint8_t tmp[256];
            memcpy(tmp, data, 256);
            rc4.set_key(key, 256);
            rc4.apply_keystream_256(tmp);
        }
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start).count();
        double ops_per_sec = (double)ITERATIONS / (duration / 1e6);
        printf("Optimized RC4:     %.2f ops/sec (%.2f MB/s)\n",
               ops_per_sec, ops_per_sec * 256 / 1e6);
    }

    // Benchmark FastRc4 (from rc4_avx512.hpp)
    {
        auto start = high_resolution_clock::now();
        for (int i = 0; i < ITERATIONS; i++) {
            rc4_avx512::FastRc4 rc4;
            uint8_t tmp[256];
            memcpy(tmp, data, 256);
            rc4.set_key(key, 256);
            rc4.apply_keystream_256(tmp);
        }
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start).count();
        double ops_per_sec = (double)ITERATIONS / (duration / 1e6);
        printf("FastRc4:           %.2f ops/sec (%.2f MB/s)\n",
               ops_per_sec, ops_per_sec * 256 / 1e6);
    }
}

// =============================================================================
// RC4 Key Setup Only Benchmark (for ops 254/255)
// =============================================================================

void bench_rc4_ksa_only() {
    printf("\n=== RC4 Key Setup (KSA) Benchmark ===\n");
    printf("This is the critical path for ops 254/255\n");

    const int ITERATIONS = 200000;

    uint8_t key[256];
    std::mt19937 rng(77777);
    for (int i = 0; i < 256; i++) key[i] = rng() & 0xFF;

    // Benchmark OpenSSL RC4_set_key
    {
        auto start = high_resolution_clock::now();
        for (int i = 0; i < ITERATIONS; i++) {
            RC4_KEY rc4_key;
            RC4_set_key(&rc4_key, 256, key);
        }
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start).count();
        double ops_per_sec = (double)ITERATIONS / (duration / 1e6);
        printf("OpenSSL RC4_set_key:   %.2f ops/sec\n", ops_per_sec);
    }

    // Benchmark Optimized RC4 set_key
    {
        rc4_opt::OptimizedRc4 rc4;
        auto start = high_resolution_clock::now();
        for (int i = 0; i < ITERATIONS; i++) {
            rc4.set_key(key, 256);
        }
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start).count();
        double ops_per_sec = (double)ITERATIONS / (duration / 1e6);
        printf("Optimized set_key:     %.2f ops/sec\n", ops_per_sec);
    }

    // Benchmark FastRc4 set_key
    {
        rc4_avx512::FastRc4 rc4;
        auto start = high_resolution_clock::now();
        for (int i = 0; i < ITERATIONS; i++) {
            rc4.set_key(key, 256);
        }
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start).count();
        double ops_per_sec = (double)ITERATIONS / (duration / 1e6);
        printf("FastRc4 set_key:       %.2f ops/sec\n", ops_per_sec);
    }
}

// =============================================================================
// Main
// =============================================================================

int main() {
    printf("===========================================\n");
    printf("RC4 and Salsa20 Optimization Benchmark\n");
    printf("===========================================\n");

    bool all_passed = true;

    // Correctness tests
    all_passed &= test_salsa20_correctness();
    all_passed &= test_rc4_correctness();

    // Performance benchmarks
    bench_salsa20_performance();
    bench_rc4_performance();
    bench_rc4_ksa_only();

    printf("\n===========================================\n");
    if (all_passed) {
        printf("All correctness tests PASSED\n");
    } else {
        printf("Some correctness tests FAILED\n");
    }
    printf("===========================================\n");

    return all_passed ? 0 : 1;
}
