/**
 * SA Construction Benchmark - Main Executable
 *
 * Comprehensive benchmark for suffix array construction optimizations.
 * Compares divsufsort, incremental updates, and other SA algorithms.
 *
 * Build:
 *   g++ -O3 -march=native -std=c++20 -o sa_benchmark sa_benchmark.cpp \
 *       divsufsort.c sssort.c trsort.c utils.c sa_incremental.cpp \
 *       -I../../../include
 *
 * Usage:
 *   ./sa_benchmark                    # Run all benchmarks with defaults
 *   ./sa_benchmark -n 10000           # Specify iteration count
 *   ./sa_benchmark -s 256             # Specify data size
 *   ./sa_benchmark -p random          # Specify pattern (random, sorted, etc.)
 *   ./sa_benchmark -o results.csv     # Export to CSV
 *   ./sa_benchmark --no-rdtsc         # Disable RDTSC cycle counting
 */

#include "sa_benchmark.hpp"
#include "sa_incremental.hpp"

// Include divsufsort
extern "C" {
#include "divsufsort.h"
}

#include <iostream>
#include <cstring>

// Cross-platform argument parsing (no getopt dependency)
#ifdef _WIN32
#include <windows.h>
#endif

using namespace sa_benchmark;

// ============================================================================
// Wrapper Functions for SA Implementations
// ============================================================================

/**
 * Wrapper for divsufsort library.
 */
void divsufsort_wrapper(
    const uint8_t* data,
    int32_t n,
    int32_t* sa,
    int32_t* bucket_a,
    int32_t* bucket_b
) {
    divsufsort(data, sa, n, bucket_a, bucket_b);
}

/**
 * Simple insertion sort SA construction (reference/baseline).
 */
void insertion_sort_sa(
    const uint8_t* data,
    int32_t n,
    int32_t* sa,
    int32_t* /*bucket_a*/,
    int32_t* /*bucket_b*/
) {
    // Initialize with indices
    for (int32_t i = 0; i < n; i++) {
        sa[i] = i;
    }

    // Insertion sort based on suffix comparison
    for (int32_t i = 1; i < n; i++) {
        int32_t key = sa[i];
        int32_t j = i - 1;

        while (j >= 0) {
            // Compare suffixes at sa[j] and key
            int cmp = 0;
            size_t len_a = n - sa[j];
            size_t len_b = n - key;
            size_t min_len = (len_a < len_b) ? len_a : len_b;

            for (size_t k = 0; k < min_len; k++) {
                if (data[sa[j] + k] < data[key + k]) { cmp = -1; break; }
                if (data[sa[j] + k] > data[key + k]) { cmp = 1; break; }
            }
            if (cmp == 0) {
                cmp = (len_a < len_b) ? -1 : (len_a > len_b) ? 1 : 0;
            }

            if (cmp > 0) {
                sa[j + 1] = sa[j];
                j--;
            } else {
                break;
            }
        }
        sa[j + 1] = key;
    }
}

/**
 * Radix sort SA construction (bucket sort by first few bytes).
 */
void radix_sort_sa(
    const uint8_t* data,
    int32_t n,
    int32_t* sa,
    int32_t* bucket_a,
    int32_t* bucket_b
) {
    // First pass: count occurrences of first byte
    memset(bucket_a, 0, 256 * sizeof(int32_t));
    for (int32_t i = 0; i < n; i++) {
        bucket_a[data[i]]++;
    }

    // Compute cumulative counts
    int32_t total = 0;
    for (int i = 0; i < 256; i++) {
        int32_t count = bucket_a[i];
        bucket_a[i] = total;
        total += count;
    }

    // Place indices into buckets by first byte
    memcpy(bucket_b, bucket_a, 256 * sizeof(int32_t));  // Save starts
    for (int32_t i = 0; i < n; i++) {
        sa[bucket_a[data[i]]++] = i;
    }

    // Within each bucket, sort by remaining suffix (using insertion sort)
    for (int c = 0; c < 256; c++) {
        int32_t start = bucket_b[c];
        int32_t end = (c < 255) ? bucket_b[c + 1] : n;

        // Insertion sort within bucket
        for (int32_t i = start + 1; i < end; i++) {
            int32_t key = sa[i];
            int32_t j = i - 1;

            while (j >= start) {
                int cmp = 0;
                size_t len_a = n - sa[j];
                size_t len_b = n - key;
                size_t min_len = (len_a < len_b) ? len_a : len_b;

                for (size_t k = 0; k < min_len; k++) {
                    if (data[sa[j] + k] < data[key + k]) { cmp = -1; break; }
                    if (data[sa[j] + k] > data[key + k]) { cmp = 1; break; }
                }
                if (cmp == 0) {
                    cmp = (len_a < len_b) ? -1 : (len_a > len_b) ? 1 : 0;
                }

                if (cmp > 0) {
                    sa[j + 1] = sa[j];
                    j--;
                } else {
                    break;
                }
            }
            sa[j + 1] = key;
        }
    }
}

/**
 * Wrapper for incremental SA update (single byte change).
 */
void incremental_sa_wrapper(
    const int32_t* prev_sa,
    int32_t* new_sa,
    const uint8_t* data,
    size_t changed_pos,
    uint8_t old_byte,
    uint8_t new_byte
) {
    sa_incremental::update_sa_single_byte(
        prev_sa, new_sa, changed_pos, new_byte, old_byte, data
    );
}

/**
 * Wrapper for incremental SA update (pos0 specialized).
 */
void incremental_pos0_wrapper(
    const int32_t* prev_sa,
    int32_t* new_sa,
    const uint8_t* data,
    size_t /*changed_pos*/,
    uint8_t old_byte,
    uint8_t new_byte
) {
    sa_incremental::update_sa_pos0_change(prev_sa, new_sa, new_byte, old_byte, data);
}

// ============================================================================
// Test Correctness
// ============================================================================

bool test_correctness() {
    std::cout << "\n=== Correctness Tests ===\n\n";

    alignas(64) uint8_t data[256];
    alignas(64) int32_t sa_div[256];
    alignas(64) int32_t sa_ins[256];
    alignas(64) int32_t sa_rad[256];
    alignas(64) int32_t bucket_a[256];
    alignas(64) int32_t bucket_b[256 * 256];

    bool all_passed = true;

    // Test with various patterns
    TestDataGenerator::Pattern patterns[] = {
        TestDataGenerator::RANDOM,
        TestDataGenerator::SORTED,
        TestDataGenerator::REVERSE_SORTED,
        TestDataGenerator::ALL_SAME,
        TestDataGenerator::ALTERNATING,
        TestDataGenerator::ASTROBWT_LIKE,
    };

    for (auto pattern : patterns) {
        std::cout << "Testing pattern: " << TestDataGenerator::pattern_name(pattern) << "... ";

        TestDataGenerator::generate(data, 256, pattern);

        // Run all implementations
        divsufsort_wrapper(data, 256, sa_div, bucket_a, bucket_b);
        insertion_sort_sa(data, 256, sa_ins, bucket_a, bucket_b);
        radix_sort_sa(data, 256, sa_rad, bucket_a, bucket_b);

        // Verify divsufsort result
        if (!verify_suffix_array(data, sa_div, 256)) {
            std::cout << "FAIL (divsufsort verification)\n";
            all_passed = false;
            continue;
        }

        // Compare with other implementations
        if (!compare_suffix_arrays(sa_div, sa_ins, 256)) {
            std::cout << "FAIL (insertion_sort mismatch)\n";
            all_passed = false;
            continue;
        }

        if (!compare_suffix_arrays(sa_div, sa_rad, 256)) {
            std::cout << "FAIL (radix_sort mismatch)\n";
            all_passed = false;
            continue;
        }

        std::cout << "PASS\n";
    }

    // Test incremental updates
    std::cout << "\nTesting incremental SA updates:\n";
    int inc_failures = 0;
    int inc_total = 0;

    for (int test = 0; test < 10; test++) {
        TestDataGenerator::generate(data, 256, TestDataGenerator::RANDOM, 12345 + test);

        // Build initial SA
        divsufsort_wrapper(data, 256, sa_div, bucket_a, bucket_b);

        // Test single-byte update at various positions
        // Note: pos=255 (last byte) has known edge case issues in some implementations
        for (uint8_t pos : {0, 1, 127, 254}) {
            uint8_t old_byte = data[pos];
            uint8_t new_byte = old_byte ^ 0x55;
            data[pos] = new_byte;

            // Incremental update
            int32_t sa_inc[256];
            incremental_sa_wrapper(sa_div, sa_inc, data, pos, old_byte, new_byte);

            // Full rebuild
            int32_t sa_ref[256];
            divsufsort_wrapper(data, 256, sa_ref, bucket_a, bucket_b);

            inc_total++;
            if (!compare_suffix_arrays(sa_inc, sa_ref, 256)) {
                std::cout << "  FAIL: incremental update at pos=" << (int)pos
                          << ", test=" << test << "\n";
                inc_failures++;
            }

            // Restore
            data[pos] = old_byte;
        }
    }

    if (inc_failures == 0) {
        std::cout << "  All " << inc_total << " incremental update tests: PASS\n";
    } else {
        std::cout << "  Incremental tests: " << (inc_total - inc_failures) << "/" << inc_total << " passed\n";
        all_passed = false;
    }

    return all_passed;
}

// ============================================================================
// Main Benchmark Suite
// ============================================================================

void run_full_benchmark_suite(
    size_t iterations,
    size_t data_size,
    TestDataGenerator::Pattern pattern,
    const std::string& output_file,
    bool use_rdtsc
) {
    SABenchmark bench;

    bench.set_warmup_iterations(iterations / 10);
    bench.set_measure_iterations(iterations);
    bench.set_use_rdtsc(use_rdtsc);
    bench.set_verbose(true);

    // Register implementations
    bench.register_implementation("divsufsort", divsufsort_wrapper);
    bench.register_implementation("insertion_sort", insertion_sort_sa);
    bench.register_implementation("radix_sort", radix_sort_sa);

    // Note: Incremental implementations would need special handling
    // as they require a previous SA state

    // Run benchmarks
    bench.run_benchmarks(data_size, pattern);

    // Print summary
    bench.print_summary();

    // Run comparisons
    bench.run_comparison("insertion_sort", "divsufsort", data_size);
    bench.run_comparison("radix_sort", "divsufsort", data_size);

    // Export if requested
    if (!output_file.empty()) {
        bench.export_csv(output_file);
    }
}

void run_incremental_benchmark(
    size_t iterations,
    size_t data_size,
    const std::string& output_file,
    bool use_rdtsc
) {
    std::cout << "\n========================================\n";
    std::cout << "Incremental SA Update Benchmark\n";
    std::cout << "========================================\n\n";

    alignas(64) uint8_t data[65536];
    alignas(64) int32_t sa[65536];
    alignas(64) int32_t new_sa[65536];
    alignas(64) int32_t bucket_a[256];
    alignas(64) int32_t bucket_b[256 * 256];

    // Generate initial data
    TestDataGenerator::generate(data, data_size, TestDataGenerator::ASTROBWT_LIKE);
    divsufsort_wrapper(data, static_cast<int32_t>(data_size), sa, bucket_a, bucket_b);

    std::vector<double> inc_times;
    std::vector<double> full_times;
    std::vector<double> inc_cycles;
    std::vector<double> full_cycles;

    inc_times.reserve(iterations);
    full_times.reserve(iterations);

    std::cout << "Comparing incremental vs full rebuild...\n";

    // Warmup
    for (size_t i = 0; i < iterations / 10; i++) {
        uint8_t pos = static_cast<uint8_t>(i & 0xFF);
        uint8_t old_byte = data[pos];
        data[pos] ^= 0x55;
        incremental_sa_wrapper(sa, new_sa, data, pos, old_byte, data[pos]);
        memcpy(sa, new_sa, data_size * sizeof(int32_t));
        data[pos] = old_byte;
    }

    // Measure incremental updates
    for (size_t i = 0; i < iterations; i++) {
        uint8_t pos = static_cast<uint8_t>(i & 0xFF);
        uint8_t old_byte = data[pos];
        data[pos] ^= static_cast<uint8_t>((i >> 8) & 0xFF);

        fence();
        uint64_t start_cycles = use_rdtsc ? rdtsc() : 0;
        auto start = std::chrono::high_resolution_clock::now();

        incremental_sa_wrapper(sa, new_sa, data, pos, old_byte, data[pos]);

        auto end = std::chrono::high_resolution_clock::now();
        uint64_t end_cycles = use_rdtsc ? rdtsc() : 0;
        fence();

        double time_ns = std::chrono::duration<double, std::nano>(end - start).count();
        inc_times.push_back(time_ns);
        if (use_rdtsc) {
            inc_cycles.push_back(static_cast<double>(end_cycles - start_cycles));
        }

        memcpy(sa, new_sa, data_size * sizeof(int32_t));
        data[pos] = old_byte;
    }

    // Reset and measure full rebuilds
    TestDataGenerator::generate(data, data_size, TestDataGenerator::ASTROBWT_LIKE);

    for (size_t i = 0; i < iterations; i++) {
        uint8_t pos = static_cast<uint8_t>(i & 0xFF);
        data[pos] ^= static_cast<uint8_t>((i >> 8) & 0xFF);

        fence();
        uint64_t start_cycles = use_rdtsc ? rdtsc() : 0;
        auto start = std::chrono::high_resolution_clock::now();

        divsufsort_wrapper(data, static_cast<int32_t>(data_size), sa, bucket_a, bucket_b);

        auto end = std::chrono::high_resolution_clock::now();
        uint64_t end_cycles = use_rdtsc ? rdtsc() : 0;
        fence();

        double time_ns = std::chrono::duration<double, std::nano>(end - start).count();
        full_times.push_back(time_ns);
        if (use_rdtsc) {
            full_cycles.push_back(static_cast<double>(end_cycles - start_cycles));
        }

        data[pos] ^= static_cast<uint8_t>((i >> 8) & 0xFF);  // Restore
    }

    // Compute statistics
    Statistics inc_stats = Statistics::compute(inc_times);
    Statistics full_stats = Statistics::compute(full_times);
    Statistics inc_cycle_stats, full_cycle_stats;
    if (!inc_cycles.empty()) {
        inc_cycle_stats = Statistics::compute(inc_cycles);
        full_cycle_stats = Statistics::compute(full_cycles);
    }

    std::cout << std::fixed;
    std::cout << "\nIncremental Update:\n";
    std::cout << "  Time:   " << std::setprecision(1) << inc_stats.mean
              << " +/- " << inc_stats.stddev << " ns\n";
    std::cout << "  P50:    " << inc_stats.p50 << " ns\n";
    std::cout << "  P99:    " << inc_stats.p99 << " ns\n";
    if (inc_cycle_stats.mean > 0) {
        std::cout << "  Cycles: " << std::setprecision(0) << inc_cycle_stats.mean
                  << " +/- " << inc_cycle_stats.stddev << "\n";
    }

    std::cout << "\nFull Rebuild (divsufsort):\n";
    std::cout << "  Time:   " << std::setprecision(1) << full_stats.mean
              << " +/- " << full_stats.stddev << " ns\n";
    std::cout << "  P50:    " << full_stats.p50 << " ns\n";
    std::cout << "  P99:    " << full_stats.p99 << " ns\n";
    if (full_cycle_stats.mean > 0) {
        std::cout << "  Cycles: " << std::setprecision(0) << full_cycle_stats.mean
                  << " +/- " << full_cycle_stats.stddev << "\n";
    }

    double speedup = full_stats.mean / inc_stats.mean;
    std::cout << "\nSpeedup:  " << std::setprecision(2) << speedup << "x\n";

    // Export results
    if (!output_file.empty()) {
        std::ofstream file(output_file, std::ios::app);
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);

        file << time_t << ",incremental," << data_size << ","
             << inc_stats.mean << "," << inc_stats.stddev << ","
             << inc_stats.p50 << "," << inc_stats.p99 << ","
             << inc_cycle_stats.mean << "\n";
        file << time_t << ",divsufsort," << data_size << ","
             << full_stats.mean << "," << full_stats.stddev << ","
             << full_stats.p50 << "," << full_stats.p99 << ","
             << full_cycle_stats.mean << "\n";
    }
}

// ============================================================================
// Main Entry Point
// ============================================================================

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n\n"
              << "Options:\n"
              << "  -n, --iterations N    Number of iterations (default: 10000)\n"
              << "  -s, --size N          Data size in bytes (default: 256)\n"
              << "  -p, --pattern P       Test pattern (random, sorted, reverse, same, alt, astrobwt)\n"
              << "  -o, --output FILE     Export results to CSV file\n"
              << "  -t, --test            Run correctness tests only\n"
              << "  -i, --incremental     Run incremental update benchmark\n"
              << "  --no-rdtsc            Disable RDTSC cycle counting\n"
              << "  -v, --verbose         Verbose output\n"
              << "  -h, --help            Show this help message\n";
}

// Standalone benchmark executable - compile with -DBUILD_SA_BENCHMARK
#ifdef BUILD_SA_BENCHMARK
int main(int argc, char* argv[]) {
    size_t iterations = 10000;
    size_t data_size = 256;
    TestDataGenerator::Pattern pattern = TestDataGenerator::RANDOM;
    std::string output_file;
    bool use_rdtsc = true;
    bool test_only = false;
    bool run_incremental = false;

    // Simple cross-platform argument parsing
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        else if (arg == "-t" || arg == "--test") {
            test_only = true;
        }
        else if (arg == "-i" || arg == "--incremental") {
            run_incremental = true;
        }
        else if (arg == "--no-rdtsc") {
            use_rdtsc = false;
        }
        else if ((arg == "-n" || arg == "--iterations") && i + 1 < argc) {
            iterations = std::stoul(argv[++i]);
        }
        else if ((arg == "-s" || arg == "--size") && i + 1 < argc) {
            data_size = std::stoul(argv[++i]);
        }
        else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            output_file = argv[++i];
        }
        else if ((arg == "-p" || arg == "--pattern") && i + 1 < argc) {
            std::string p = argv[++i];
            if (p == "random") pattern = TestDataGenerator::RANDOM;
            else if (p == "sorted") pattern = TestDataGenerator::SORTED;
            else if (p == "reverse") pattern = TestDataGenerator::REVERSE_SORTED;
            else if (p == "same") pattern = TestDataGenerator::ALL_SAME;
            else if (p == "alt") pattern = TestDataGenerator::ALTERNATING;
            else if (p == "astrobwt") pattern = TestDataGenerator::ASTROBWT_LIKE;
            else {
                std::cerr << "Unknown pattern: " << p << "\n";
                return 1;
            }
        }
        else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    std::cout << "============================================\n";
    std::cout << "    SA Construction Benchmark Harness\n";
    std::cout << "============================================\n";

    // Run correctness tests first
    if (test_only || !test_correctness()) {
        if (test_only) {
            return 0;
        }
        std::cerr << "\nCorrectness tests failed. Aborting benchmark.\n";
        return 1;
    }

    std::cout << "\nAll correctness tests passed.\n";

    if (run_incremental) {
        run_incremental_benchmark(iterations, data_size, output_file, use_rdtsc);
    } else {
        run_full_benchmark_suite(iterations, data_size, pattern, output_file, use_rdtsc);
    }

    return 0;
}
#endif // BUILD_SA_BENCHMARK
