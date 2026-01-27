#pragma once
/**
 * Tests for Tritonn Optimization Modules
 *
 * This file provides test functions for validating the correctness and
 * measuring the performance of the ported optimization modules:
 * - Branch Table Optimizations (wolfPermute AVX2)
 * - Incremental Suffix Array Updates
 * - AVX-512 16-way RC4
 * - SPSA (Stamp-based Predictive Suffix Array)
 * - On-demand SHA-256 from compressed SA
 */

#include <cstdint>
#include <cstddef>
#include <chrono>
#include <string>
#include <functional>

namespace optimization_tests {

// ============================================================================
// Test Result Structures
// ============================================================================

struct TestResult {
    std::string test_name;
    bool passed;
    std::string message;
    double duration_ms;
};

struct BenchmarkResult {
    std::string benchmark_name;
    size_t iterations;
    double total_ms;
    double avg_us;           // Average microseconds per iteration
    double throughput;       // Operations per second
    std::string unit;
};

// ============================================================================
// Branch Table Tests
// ============================================================================

/**
 * Test that wolf_branch_scalar produces correct results for known inputs.
 * Verifies the 16 operations match expected transformations.
 */
TestResult test_wolf_branch_scalar_correctness();

/**
 * Test that AVX2 wolfPermute matches scalar implementation.
 * Tests various pos1/pos2 ranges and operation codes.
 */
TestResult test_wolf_permute_avx2_matches_scalar();

/**
 * Test CODE_LUT_16 compression is correctly derived from CODE_LUT.
 */
TestResult test_code_lut_compression();

/**
 * Benchmark wolf_permute_avx2 vs scalar.
 */
BenchmarkResult benchmark_wolf_permute(size_t iterations = 100000);

// ============================================================================
// Incremental SA Tests
// ============================================================================

/**
 * Test single-byte SA update produces correct result.
 * Compares against full SA rebuild.
 */
TestResult test_sa_single_byte_update();

/**
 * Test pos0 change optimization produces correct result.
 */
TestResult test_sa_pos0_change();

/**
 * Test two-byte SA update produces correct result.
 */
TestResult test_sa_two_byte_update();

/**
 * Verify SA is correctly sorted after incremental update.
 */
TestResult test_sa_sorted_after_update();

/**
 * Benchmark incremental SA update vs full rebuild.
 */
BenchmarkResult benchmark_sa_incremental(size_t iterations = 10000);

// ============================================================================
// RC4 AVX-512 Tests
// ============================================================================

/**
 * Test RC4 key scheduling produces correct S-box state.
 */
TestResult test_rc4_key_schedule();

/**
 * Test 16-way RC4 keystream matches scalar reference.
 */
TestResult test_rc4_16way_matches_scalar();

/**
 * Test 8-way RC4 wrapper produces correct results.
 */
TestResult test_rc4_8way_wrapper();

/**
 * Benchmark 16-way RC4 vs scalar baseline.
 */
BenchmarkResult benchmark_rc4_16way(size_t iterations = 10000);

// ============================================================================
// SPSA Tests
// ============================================================================

/**
 * Test stamp building produces correct structures.
 */
TestResult test_spsa_stamp_building();

/**
 * Test mini-SA building for stamps.
 */
TestResult test_spsa_mini_sa();

/**
 * Test stamp merging algorithm.
 */
TestResult test_spsa_stamp_merge();

/**
 * Benchmark SPSA vs traditional SA building.
 */
BenchmarkResult benchmark_spsa(size_t iterations = 1000);

// ============================================================================
// SHA-256 SPSA Tests
// ============================================================================

/**
 * Test on-demand SHA-256 from compressed SA matches full expansion.
 */
TestResult test_sha256_compressed_correctness();

/**
 * Test SHA-NI acceleration produces correct results.
 */
TestResult test_sha256_ni_correctness();

/**
 * Benchmark SHA-256 from compressed vs expanded SA.
 */
BenchmarkResult benchmark_sha256_compressed(size_t iterations = 10000);

// ============================================================================
// AstroBWT Test Vectors
// ============================================================================

struct AstroBwtTestVector {
    const char* input_hex;
    size_t input_len;
    const char* expected_hash_hex;
    bool expect_fail;
};

/**
 * Known AstroBWT test vectors for validation.
 */
extern const AstroBwtTestVector ASTROBWT_TEST_VECTORS[];
extern const size_t ASTROBWT_TEST_VECTOR_COUNT;

/**
 * Run all AstroBWT test vectors and verify hashes.
 */
TestResult test_astrobwt_vectors();

// ============================================================================
// Test Runner
// ============================================================================

/**
 * Run all optimization module tests.
 * @param verbose Print detailed output for each test
 * @return Number of failed tests
 */
int run_all_tests(bool verbose = true);

/**
 * Run all benchmarks.
 * @param verbose Print detailed output
 */
void run_all_benchmarks(bool verbose = true);

/**
 * Quick sanity check - runs minimal tests.
 */
bool quick_sanity_check();

} // namespace optimization_tests
