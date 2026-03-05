#pragma once
/**
 * SA Construction Benchmark Harness
 *
 * Comprehensive microbenchmarking for suffix array construction optimizations.
 * Features:
 *   - High-resolution timing with std::chrono
 *   - RDTSC cycle counting (x86/x64)
 *   - Statistical analysis (mean, stddev, percentiles)
 *   - Hardware performance counters (Linux perf)
 *   - Before/after comparison framework
 *   - CSV output for progress tracking
 *
 * Usage:
 *   SABenchmark bench;
 *   bench.register_implementation("divsufsort", divsufsort_wrapper);
 *   bench.register_implementation("incremental", incremental_wrapper);
 *   bench.run_benchmarks(10000);  // 10000 iterations
 *   bench.print_results();
 *   bench.export_csv("sa_benchmark_results.csv");
 */

#include <cstdint>
#include <cstring>
#include <chrono>
#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#include <cmath>
#include <random>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <x86intrin.h>
#endif

#ifdef __linux__
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>
#include <unistd.h>
#endif

namespace sa_benchmark {

// ============================================================================
// RDTSC Cycle Counter
// ============================================================================

/**
 * Read Time-Stamp Counter for precise cycle counting.
 * Uses RDTSCP which includes a serializing instruction.
 */
inline uint64_t rdtsc() {
#if defined(_MSC_VER)
    unsigned int aux;
    return __rdtscp(&aux);
#elif defined(__GNUC__) || defined(__clang__)
    unsigned int lo, hi;
    __asm__ __volatile__ ("rdtscp" : "=a" (lo), "=d" (hi) :: "rcx");
    return ((uint64_t)hi << 32) | lo;
#else
    return 0;
#endif
}

/**
 * Memory fence to prevent instruction reordering around measurements.
 */
inline void fence() {
#if defined(_MSC_VER)
    _mm_mfence();
#elif defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__ ("mfence" ::: "memory");
#endif
}

/**
 * Prevent the compiler from optimizing away a value.
 */
template <typename T>
inline void do_not_optimize(T const& value) {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "r,m"(value) : "memory");
#else
    volatile auto dummy = value;
    (void)dummy;
#endif
}

// ============================================================================
// Linux Perf Counter Support
// ============================================================================

#ifdef __linux__
class PerfCounter {
public:
    int fd = -1;
    uint64_t type;
    uint64_t config;
    std::string name;

    PerfCounter(const std::string& n, uint64_t t, uint64_t c)
        : type(t), config(c), name(n) {}

    bool open_counter() {
        struct perf_event_attr pe;
        memset(&pe, 0, sizeof(pe));
        pe.type = type;
        pe.size = sizeof(pe);
        pe.config = config;
        pe.disabled = 1;
        pe.exclude_kernel = 1;
        pe.exclude_hv = 1;

        fd = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
        return fd >= 0;
    }

    void start() {
        if (fd >= 0) {
            ioctl(fd, PERF_EVENT_IOC_RESET, 0);
            ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
        }
    }

    uint64_t stop() {
        if (fd >= 0) {
            ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
            uint64_t count = 0;
            read(fd, &count, sizeof(count));
            return count;
        }
        return 0;
    }

    void close_counter() {
        if (fd >= 0) {
            close(fd);
            fd = -1;
        }
    }
};
#endif

// ============================================================================
// Statistics Utilities
// ============================================================================

struct Statistics {
    double mean = 0.0;
    double stddev = 0.0;
    double min_val = 0.0;
    double max_val = 0.0;
    double p50 = 0.0;  // Median
    double p95 = 0.0;
    double p99 = 0.0;
    size_t count = 0;

    static Statistics compute(std::vector<double>& samples) {
        Statistics stats = {};
        if (samples.empty()) return stats;

        stats.count = samples.size();

        // Sort for percentiles
        std::sort(samples.begin(), samples.end());

        stats.min_val = samples.front();
        stats.max_val = samples.back();

        // Mean
        double sum = 0;
        for (double v : samples) sum += v;
        stats.mean = sum / samples.size();

        // Standard deviation
        double sq_sum = 0;
        for (double v : samples) {
            double diff = v - stats.mean;
            sq_sum += diff * diff;
        }
        stats.stddev = std::sqrt(sq_sum / samples.size());

        // Percentiles
        auto percentile = [&samples](double p) -> double {
            double idx = (p / 100.0) * (samples.size() - 1);
            size_t lo = static_cast<size_t>(idx);
            size_t hi = lo + 1;
            if (hi >= samples.size()) hi = samples.size() - 1;
            double frac = idx - lo;
            return samples[lo] * (1.0 - frac) + samples[hi] * frac;
        };

        stats.p50 = percentile(50);
        stats.p95 = percentile(95);
        stats.p99 = percentile(99);

        return stats;
    }
};

// ============================================================================
// Benchmark Result
// ============================================================================

struct BenchmarkResult {
    std::string impl_name;
    std::string test_name;

    // Timing results
    Statistics time_ns{};     // Nanoseconds per operation
    Statistics cycles{};      // CPU cycles per operation

    // Hardware counters (Linux only)
    Statistics cache_misses{};
    Statistics branch_misses{};
    Statistics instructions{};

    // Throughput
    double ops_per_second = 0.0;
    double bytes_per_second = 0.0;

    size_t data_size = 0;
};

// ============================================================================
// SA Implementation Interface
// ============================================================================

/**
 * Function type for SA construction implementations.
 *
 * @param data Input data buffer
 * @param n Length of input data
 * @param sa Output suffix array (pre-allocated)
 * @param bucket_a Working buffer A (256 * sizeof(int32_t) for divsufsort)
 * @param bucket_b Working buffer B (256*256 * sizeof(int32_t) for divsufsort)
 */
using SAConstructFn = std::function<void(
    const uint8_t* data,
    int32_t n,
    int32_t* sa,
    int32_t* bucket_a,
    int32_t* bucket_b
)>;

/**
 * Function type for incremental SA update implementations.
 *
 * @param prev_sa Previous suffix array
 * @param new_sa Output suffix array
 * @param data Current data (with changes applied)
 * @param changed_pos Position that changed
 * @param old_byte Previous byte value
 * @param new_byte New byte value
 */
using SAUpdateFn = std::function<void(
    const int32_t* prev_sa,
    int32_t* new_sa,
    const uint8_t* data,
    size_t changed_pos,
    uint8_t old_byte,
    uint8_t new_byte
)>;

// ============================================================================
// Test Data Generator
// ============================================================================

class TestDataGenerator {
public:
    enum Pattern {
        RANDOM,           // Uniform random
        SORTED,           // Ascending order
        REVERSE_SORTED,   // Descending order
        ALL_SAME,         // All bytes identical
        ALTERNATING,      // Alternating two values
        ASTROBWT_LIKE,    // Simulated AstroBWT output patterns
        WORST_CASE_DSS    // Worst case for divsufsort (repeated patterns)
    };

    static void generate(uint8_t* data, size_t len, Pattern pattern, uint32_t seed = 12345) {
        std::mt19937 gen(seed);
        std::uniform_int_distribution<int> dist(0, 255);

        switch (pattern) {
            case RANDOM:
                for (size_t i = 0; i < len; i++) {
                    data[i] = static_cast<uint8_t>(dist(gen));
                }
                break;

            case SORTED:
                for (size_t i = 0; i < len; i++) {
                    data[i] = static_cast<uint8_t>((i * 256) / len);
                }
                break;

            case REVERSE_SORTED:
                for (size_t i = 0; i < len; i++) {
                    data[i] = static_cast<uint8_t>(255 - (i * 256) / len);
                }
                break;

            case ALL_SAME:
                memset(data, 0x42, len);
                break;

            case ALTERNATING:
                for (size_t i = 0; i < len; i++) {
                    data[i] = (i & 1) ? 0xAA : 0x55;
                }
                break;

            case ASTROBWT_LIKE:
                // Simulates patterns seen in AstroBWT after branching operations
                for (size_t i = 0; i < len; i++) {
                    uint8_t base = static_cast<uint8_t>(dist(gen));
                    // Apply some transformations similar to wolf branching
                    base ^= static_cast<uint8_t>(i & 0xFF);
                    base = static_cast<uint8_t>((base * base) & 0xFF);
                    data[i] = base;
                }
                break;

            case WORST_CASE_DSS:
                // Fibonacci-like pattern that's challenging for SA algorithms
                if (len >= 2) {
                    data[0] = 'a';
                    data[1] = 'b';
                    for (size_t i = 2; i < len; i++) {
                        data[i] = data[i-1] ^ data[i-2];
                    }
                }
                break;
        }
    }

    static const char* pattern_name(Pattern p) {
        switch (p) {
            case RANDOM: return "random";
            case SORTED: return "sorted";
            case REVERSE_SORTED: return "reverse";
            case ALL_SAME: return "same";
            case ALTERNATING: return "alternating";
            case ASTROBWT_LIKE: return "astrobwt";
            case WORST_CASE_DSS: return "worst_case";
            default: return "unknown";
        }
    }
};

// ============================================================================
// Main Benchmark Harness
// ============================================================================

class SABenchmark {
public:
    struct Implementation {
        std::string name;
        SAConstructFn construct_fn;
        SAUpdateFn update_fn;  // Optional: for incremental implementations
        bool is_incremental;
    };

private:
    std::vector<Implementation> implementations_;
    std::vector<BenchmarkResult> results_;

    size_t warmup_iterations_ = 100;
    size_t measure_iterations_ = 10000;
    bool use_rdtsc_ = true;
    bool use_perf_counters_ = true;
    bool verbose_ = true;

public:
    /**
     * Register a full SA construction implementation.
     */
    void register_implementation(const std::string& name, SAConstructFn fn) {
        implementations_.push_back({name, fn, nullptr, false});
    }

    /**
     * Register an incremental SA update implementation.
     */
    void register_incremental(const std::string& name, SAUpdateFn fn) {
        implementations_.push_back({name, nullptr, fn, true});
    }

    void set_warmup_iterations(size_t n) { warmup_iterations_ = n; }
    void set_measure_iterations(size_t n) { measure_iterations_ = n; }
    void set_use_rdtsc(bool v) { use_rdtsc_ = v; }
    void set_use_perf_counters(bool v) { use_perf_counters_ = v; }
    void set_verbose(bool v) { verbose_ = v; }

    /**
     * Run benchmarks on all registered implementations.
     */
    void run_benchmarks(
        size_t data_size = 256,
        TestDataGenerator::Pattern pattern = TestDataGenerator::RANDOM
    ) {
        if (verbose_) {
            std::cout << "\n========================================\n";
            std::cout << "SA Construction Benchmark\n";
            std::cout << "----------------------------------------\n";
            std::cout << "Data size:   " << data_size << " bytes\n";
            std::cout << "Pattern:     " << TestDataGenerator::pattern_name(pattern) << "\n";
            std::cout << "Warmup:      " << warmup_iterations_ << " iterations\n";
            std::cout << "Measure:     " << measure_iterations_ << " iterations\n";
            std::cout << "RDTSC:       " << (use_rdtsc_ ? "enabled" : "disabled") << "\n";
#ifdef __linux__
            std::cout << "Perf ctrs:   " << (use_perf_counters_ ? "enabled" : "disabled") << "\n";
#else
            std::cout << "Perf ctrs:   not available (Linux only)\n";
#endif
            std::cout << "========================================\n\n";
        }

        // Allocate aligned buffers
        alignas(64) uint8_t data[65536];
        alignas(64) int32_t sa[65536];
        alignas(64) int32_t prev_sa[65536];
        alignas(64) int32_t bucket_a[256];
        alignas(64) int32_t bucket_b[256 * 256];

        // Generate test data
        TestDataGenerator::generate(data, data_size, pattern);

        for (const auto& impl : implementations_) {
            BenchmarkResult result;
            result.impl_name = impl.name;
            result.test_name = TestDataGenerator::pattern_name(pattern);
            result.data_size = data_size;

            if (verbose_) {
                std::cout << "Benchmarking: " << impl.name << "\n";
            }

            std::vector<double> time_samples;
            std::vector<double> cycle_samples;
            std::vector<double> cache_miss_samples;
            std::vector<double> branch_miss_samples;
            std::vector<double> instruction_samples;

            time_samples.reserve(measure_iterations_);
            cycle_samples.reserve(measure_iterations_);

#ifdef __linux__
            PerfCounter cache_counter("cache-misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES);
            PerfCounter branch_counter("branch-misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES);
            PerfCounter instr_counter("instructions", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);

            bool perf_ok = false;
            if (use_perf_counters_) {
                perf_ok = cache_counter.open_counter() &&
                          branch_counter.open_counter() &&
                          instr_counter.open_counter();
            }
#endif

            // Warmup phase
            for (size_t i = 0; i < warmup_iterations_; i++) {
                if (impl.is_incremental && impl.update_fn) {
                    // For incremental: modify one byte and update
                    uint8_t pos = static_cast<uint8_t>(i & 0xFF);
                    uint8_t old_byte = data[pos];
                    data[pos] ^= 0x55;
                    impl.update_fn(prev_sa, sa, data, pos, old_byte, data[pos]);
                    memcpy(prev_sa, sa, data_size * sizeof(int32_t));
                    data[pos] = old_byte;
                } else if (impl.construct_fn) {
                    impl.construct_fn(data, static_cast<int32_t>(data_size), sa, bucket_a, bucket_b);
                }
                do_not_optimize(sa[0]);
            }

            // Measurement phase
            for (size_t i = 0; i < measure_iterations_; i++) {
                fence();

#ifdef __linux__
                if (perf_ok) {
                    cache_counter.start();
                    branch_counter.start();
                    instr_counter.start();
                }
#endif

                uint64_t start_cycles = use_rdtsc_ ? rdtsc() : 0;
                auto start_time = std::chrono::high_resolution_clock::now();

                if (impl.is_incremental && impl.update_fn) {
                    uint8_t pos = static_cast<uint8_t>(i & 0xFF);
                    uint8_t old_byte = data[pos];
                    data[pos] ^= static_cast<uint8_t>((i >> 8) & 0xFF);
                    impl.update_fn(prev_sa, sa, data, pos, old_byte, data[pos]);
                    data[pos] = old_byte;
                } else if (impl.construct_fn) {
                    impl.construct_fn(data, static_cast<int32_t>(data_size), sa, bucket_a, bucket_b);
                }

                auto end_time = std::chrono::high_resolution_clock::now();
                uint64_t end_cycles = use_rdtsc_ ? rdtsc() : 0;

#ifdef __linux__
                uint64_t cache_misses = 0, branch_misses = 0, instructions = 0;
                if (perf_ok) {
                    cache_misses = cache_counter.stop();
                    branch_misses = branch_counter.stop();
                    instructions = instr_counter.stop();
                    cache_miss_samples.push_back(static_cast<double>(cache_misses));
                    branch_miss_samples.push_back(static_cast<double>(branch_misses));
                    instruction_samples.push_back(static_cast<double>(instructions));
                }
#endif

                fence();
                do_not_optimize(sa[0]);

                double time_ns = std::chrono::duration<double, std::nano>(end_time - start_time).count();
                time_samples.push_back(time_ns);

                if (use_rdtsc_) {
                    cycle_samples.push_back(static_cast<double>(end_cycles - start_cycles));
                }
            }

#ifdef __linux__
            cache_counter.close_counter();
            branch_counter.close_counter();
            instr_counter.close_counter();
#endif

            // Compute statistics
            result.time_ns = Statistics::compute(time_samples);
            if (!cycle_samples.empty()) {
                result.cycles = Statistics::compute(cycle_samples);
            }
#ifdef __linux__
            if (!cache_miss_samples.empty()) {
                result.cache_misses = Statistics::compute(cache_miss_samples);
            }
            if (!branch_miss_samples.empty()) {
                result.branch_misses = Statistics::compute(branch_miss_samples);
            }
            if (!instruction_samples.empty()) {
                result.instructions = Statistics::compute(instruction_samples);
            }
#endif

            result.ops_per_second = 1e9 / result.time_ns.mean;
            result.bytes_per_second = result.ops_per_second * data_size;

            results_.push_back(result);

            if (verbose_) {
                print_result(result);
            }
        }
    }

    /**
     * Run comparison benchmark between baseline and optimized implementation.
     */
    void run_comparison(
        const std::string& baseline_name,
        const std::string& optimized_name,
        size_t data_size = 256
    ) {
        BenchmarkResult* baseline = nullptr;
        BenchmarkResult* optimized = nullptr;

        for (auto& r : results_) {
            if (r.impl_name == baseline_name) baseline = &r;
            if (r.impl_name == optimized_name) optimized = &r;
        }

        if (!baseline || !optimized) {
            std::cerr << "Error: Could not find implementations for comparison\n";
            return;
        }

        std::cout << "\n========================================\n";
        std::cout << "Comparison: " << baseline_name << " vs " << optimized_name << "\n";
        std::cout << "========================================\n";

        double speedup = baseline->time_ns.mean / optimized->time_ns.mean;
        double cycle_reduction = 0;
        if (baseline->cycles.mean > 0 && optimized->cycles.mean > 0) {
            cycle_reduction = (1.0 - optimized->cycles.mean / baseline->cycles.mean) * 100.0;
        }

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Speedup:            " << speedup << "x\n";
        std::cout << "Time reduction:     " << (1.0 - 1.0/speedup) * 100.0 << "%\n";
        if (cycle_reduction != 0) {
            std::cout << "Cycle reduction:    " << cycle_reduction << "%\n";
        }

        // Statistical significance (simple t-test approximation)
        double se_diff = std::sqrt(
            (baseline->time_ns.stddev * baseline->time_ns.stddev +
             optimized->time_ns.stddev * optimized->time_ns.stddev) /
            static_cast<double>(measure_iterations_)
        );
        double t_stat = (baseline->time_ns.mean - optimized->time_ns.mean) / se_diff;

        std::cout << "T-statistic:        " << t_stat << "\n";
        std::cout << "Significant:        " << (std::abs(t_stat) > 2.576 ? "Yes (p<0.01)" : "No") << "\n";
        std::cout << "========================================\n";
    }

    /**
     * Print results for a single benchmark.
     */
    void print_result(const BenchmarkResult& r) {
        std::cout << std::fixed;
        std::cout << "  Time:      " << std::setprecision(1) << r.time_ns.mean
                  << " +/- " << r.time_ns.stddev << " ns";
        std::cout << "  (p50=" << r.time_ns.p50 << ", p99=" << r.time_ns.p99 << ")\n";

        if (r.cycles.mean > 0) {
            std::cout << "  Cycles:    " << std::setprecision(0) << r.cycles.mean
                      << " +/- " << r.cycles.stddev << "\n";
        }

#ifdef __linux__
        if (r.cache_misses.mean > 0) {
            std::cout << "  Cache miss:" << std::setprecision(1) << r.cache_misses.mean
                      << " +/- " << r.cache_misses.stddev << "\n";
        }
        if (r.branch_misses.mean > 0) {
            std::cout << "  Branch miss:" << std::setprecision(1) << r.branch_misses.mean
                      << " +/- " << r.branch_misses.stddev << "\n";
        }
        if (r.instructions.mean > 0) {
            std::cout << "  Instructions:" << std::setprecision(0) << r.instructions.mean << "\n";
        }
#endif

        std::cout << "  Throughput:" << std::setprecision(0) << r.ops_per_second
                  << " ops/sec (" << r.bytes_per_second / (1024*1024) << " MB/s)\n\n";
    }

    /**
     * Print all results in a summary table.
     */
    void print_summary() {
        std::cout << "\n";
        std::cout << "============================================================================================================\n";
        std::cout << "                                         BENCHMARK SUMMARY\n";
        std::cout << "============================================================================================================\n";
        std::cout << std::left << std::setw(20) << "Implementation"
                  << std::right << std::setw(12) << "Mean (ns)"
                  << std::setw(12) << "Stddev"
                  << std::setw(12) << "P50"
                  << std::setw(12) << "P99"
                  << std::setw(12) << "Cycles"
                  << std::setw(14) << "Ops/sec"
                  << "\n";
        std::cout << "------------------------------------------------------------------------------------------------------------\n";

        for (const auto& r : results_) {
            std::cout << std::left << std::setw(20) << r.impl_name
                      << std::right << std::fixed << std::setprecision(1)
                      << std::setw(12) << r.time_ns.mean
                      << std::setw(12) << r.time_ns.stddev
                      << std::setw(12) << r.time_ns.p50
                      << std::setw(12) << r.time_ns.p99
                      << std::setw(12) << std::setprecision(0) << r.cycles.mean
                      << std::setw(14) << std::setprecision(0) << r.ops_per_second
                      << "\n";
        }
        std::cout << "============================================================================================================\n";
    }

    /**
     * Export results to CSV for tracking progress over time.
     */
    void export_csv(const std::string& filename) {
        std::ofstream file(filename);
        if (!file) {
            std::cerr << "Error: Could not open " << filename << " for writing\n";
            return;
        }

        // Header
        file << "timestamp,implementation,test_pattern,data_size,"
             << "time_mean_ns,time_stddev_ns,time_p50_ns,time_p99_ns,"
             << "cycles_mean,cycles_stddev,"
             << "cache_misses,branch_misses,instructions,"
             << "ops_per_sec,bytes_per_sec\n";

        // Get current timestamp
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);

        for (const auto& r : results_) {
            file << time_t << ","
                 << r.impl_name << ","
                 << r.test_name << ","
                 << r.data_size << ","
                 << std::fixed << std::setprecision(2)
                 << r.time_ns.mean << ","
                 << r.time_ns.stddev << ","
                 << r.time_ns.p50 << ","
                 << r.time_ns.p99 << ","
                 << std::setprecision(0)
                 << r.cycles.mean << ","
                 << r.cycles.stddev << ","
                 << r.cache_misses.mean << ","
                 << r.branch_misses.mean << ","
                 << r.instructions.mean << ","
                 << r.ops_per_second << ","
                 << r.bytes_per_second << "\n";
        }

        if (verbose_) {
            std::cout << "Results exported to: " << filename << "\n";
        }
    }

    /**
     * Clear all results.
     */
    void clear_results() {
        results_.clear();
    }

    /**
     * Get results for programmatic access.
     */
    const std::vector<BenchmarkResult>& get_results() const {
        return results_;
    }
};

// ============================================================================
// Verification Utilities
// ============================================================================

/**
 * Verify that a suffix array is correctly constructed.
 */
inline bool verify_suffix_array(const uint8_t* data, const int32_t* sa, int32_t n) {
    // Check all indices are valid
    std::vector<bool> seen(n, false);
    for (int32_t i = 0; i < n; i++) {
        if (sa[i] < 0 || sa[i] >= n) return false;
        if (seen[sa[i]]) return false;  // Duplicate
        seen[sa[i]] = true;
    }

    // Check lexicographic order
    for (int32_t i = 0; i < n - 1; i++) {
        int32_t a = sa[i];
        int32_t b = sa[i + 1];
        size_t len_a = n - a;
        size_t len_b = n - b;
        size_t min_len = std::min(len_a, len_b);

        int cmp = memcmp(data + a, data + b, min_len);
        if (cmp > 0) return false;
        if (cmp == 0 && len_a > len_b) return false;
    }

    return true;
}

/**
 * Compare two suffix arrays for equality.
 */
inline bool compare_suffix_arrays(const int32_t* sa1, const int32_t* sa2, int32_t n) {
    return memcmp(sa1, sa2, n * sizeof(int32_t)) == 0;
}

} // namespace sa_benchmark
