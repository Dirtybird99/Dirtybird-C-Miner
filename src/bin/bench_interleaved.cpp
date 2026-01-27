/**
 * Benchmark: Interleaved (Two Miners Per Thread) vs Standard
 *
 * Monte Carlo testing to determine if the DeroLuna-style interleaving
 * provides measurable performance improvement.
 */

#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>
#include <thread>

#include <astrobwtv3/astrobwtv3.h>
#include <astrobwtv3/lookupcompute.h>
#include <astrobwtv3/interleaved_miner.hpp>
#include <dirtybird-hugepages.hpp>

// =============================================================================
// Stub definitions for extern symbols normally defined in miner.cpp
// These are needed because benchmark links astrobwtv3.cpp without miner.cpp
// =============================================================================

// From dirtybird-hugepages.hpp - controls error message printing
bool printHugepagesError = true;

// From astrobwtv3.h - SPSA feature flags
bool g_use_spsa = false;  // Disabled for benchmark (no SPSA lib linked)
bool g_verbose_tune = false;

// From astrobwtv3.h - algorithm function table (needed for tuneAstroBWTv3)
// For benchmark, we provide a minimal stub since we don't use tuning
AstroFunc allAstroFuncs[] = {
    {"wolfCompute", wolfCompute},
};
size_t numAstroFuncs = 1;

// Stub SPSA function - returns false (SPSA disabled for benchmark)
// The real implementation is in libastroSPSA library which we don't link
bool SPSA(const uint8_t* data, int dataSize, workerData& ctx) {
    (void)data; (void)dataSize; (void)ctx;
    return false;  // Never use SPSA path
}

// External declarations (detectAVX512 is defined in astrobwtv3.cpp)
extern void detectAVX512();

// Statistics helper
struct BenchStats {
    double mean;
    double stddev;
    double min;
    double max;
    double p5;
    double p95;
    int samples;
};

BenchStats computeStats(std::vector<double>& data) {
    BenchStats stats;
    stats.samples = static_cast<int>(data.size());

    if (data.empty()) {
        stats.mean = stats.stddev = stats.min = stats.max = stats.p5 = stats.p95 = 0;
        return stats;
    }

    std::sort(data.begin(), data.end());

    stats.min = data.front();
    stats.max = data.back();
    stats.p5 = data[static_cast<size_t>(data.size() * 0.05)];
    stats.p95 = data[static_cast<size_t>(data.size() * 0.95)];

    stats.mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();

    double sq_sum = 0;
    for (double v : data) {
        sq_sum += (v - stats.mean) * (v - stats.mean);
    }
    stats.stddev = std::sqrt(sq_sum / data.size());

    return stats;
}

void printStats(const char* name, const BenchStats& stats) {
    printf("  %s:\n", name);
    printf("    Mean:   %.2f H/s\n", stats.mean);
    printf("    StdDev: %.2f (CV: %.1f%%)\n", stats.stddev, 100.0 * stats.stddev / stats.mean);
    printf("    Min:    %.2f H/s\n", stats.min);
    printf("    Max:    %.2f H/s\n", stats.max);
    printf("    P5-P95: %.2f - %.2f H/s\n", stats.p5, stats.p95);
}

/**
 * Benchmark standard sequential hashing
 */
double benchmarkStandard(workerData* worker, uint8_t* inputs, int numHashes) {
    uint8_t output[32];
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numHashes; ++i) {
        AstroBWTv3(inputs + (i % 16) * 48, 48, output, *worker, false);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();
    return numHashes / seconds;
}

/**
 * Benchmark interleaved (two miners per thread)
 */
double benchmarkInterleaved(InterleavedMiner* miner, uint8_t* inputs, int numHashes) {
    uint8_t output_a[32], output_b[32];
    auto start = std::chrono::high_resolution_clock::now();

    // Process pairs
    int pairs = numHashes / 2;
    for (int i = 0; i < pairs; ++i) {
        miner->processInterleaved(
            inputs + ((i * 2) % 16) * 48, 48,
            inputs + ((i * 2 + 1) % 16) * 48, 48,
            output_a, output_b,
            false
        );
    }

    auto end = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();
    return (pairs * 2) / seconds;
}

int main(int argc, char* argv[]) {
    printf("=== Two Miners Per Thread Benchmark ===\n\n");

    // Parse arguments
    int warmup_seconds = 5;
    int test_seconds = 30;
    int num_runs = 5;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup_seconds = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--duration") == 0 && i + 1 < argc) {
            test_seconds = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            num_runs = atoi(argv[++i]);
        }
    }

    printf("Configuration:\n");
    printf("  Warmup:   %d seconds\n", warmup_seconds);
    printf("  Duration: %d seconds per run\n", test_seconds);
    printf("  Runs:     %d\n\n", num_runs);

    // Detect CPU features
    detectAVX512();

    // Generate random inputs
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> dist(0, 255);

    uint8_t inputs[16 * 48];  // 16 different inputs
    for (int i = 0; i < 16 * 48; ++i) {
        inputs[i] = dist(gen);
    }

    // Allocate standard worker
    workerData* worker = static_cast<workerData*>(malloc_huge_pages(sizeof(workerData)));
    if (!worker) {
        worker = static_cast<workerData*>(std::malloc(sizeof(workerData)));
    }
    initWorker(*worker);
    lookupGen(*worker, nullptr, nullptr);

    // Create interleaved miner
    InterleavedMiner interleavedMiner;
    if (!interleavedMiner.initialize()) {
        fprintf(stderr, "Failed to initialize interleaved miner\n");
        return 1;
    }

    // Warmup
    printf("Warming up for %d seconds...\n", warmup_seconds);
    {
        uint8_t output[32];
        auto warmup_end = std::chrono::steady_clock::now() +
                          std::chrono::seconds(warmup_seconds);
        while (std::chrono::steady_clock::now() < warmup_end) {
            AstroBWTv3(inputs, 48, output, *worker, false);
        }
    }
    printf("Warmup complete.\n\n");

    // Collect results
    std::vector<double> standard_rates;
    std::vector<double> interleaved_rates;

    // Run benchmarks
    for (int run = 0; run < num_runs; ++run) {
        printf("Run %d/%d:\n", run + 1, num_runs);

        // Benchmark standard
        {
            int hashes = 0;
            uint8_t output[32];
            auto start = std::chrono::steady_clock::now();
            auto end_time = start + std::chrono::seconds(test_seconds);

            while (std::chrono::steady_clock::now() < end_time) {
                AstroBWTv3(inputs + (hashes % 16) * 48, 48, output, *worker, false);
                hashes++;
            }

            auto end = std::chrono::steady_clock::now();
            double seconds = std::chrono::duration<double>(end - start).count();
            double rate = hashes / seconds;
            standard_rates.push_back(rate);
            printf("  Standard:    %.2f H/s (%d hashes)\n", rate, hashes);
        }

        // Benchmark interleaved
        {
            int pairs = 0;
            uint8_t output_a[32], output_b[32];
            auto start = std::chrono::steady_clock::now();
            auto end_time = start + std::chrono::seconds(test_seconds);

            while (std::chrono::steady_clock::now() < end_time) {
                interleavedMiner.processInterleaved(
                    inputs + ((pairs * 2) % 16) * 48, 48,
                    inputs + ((pairs * 2 + 1) % 16) * 48, 48,
                    output_a, output_b,
                    false
                );
                pairs++;
            }

            auto end = std::chrono::steady_clock::now();
            double seconds = std::chrono::duration<double>(end - start).count();
            double rate = (pairs * 2) / seconds;
            interleaved_rates.push_back(rate);
            printf("  Interleaved: %.2f H/s (%d pairs)\n", rate, pairs);
        }

        printf("\n");
    }

    // Compute and print statistics
    printf("=== Results ===\n\n");

    BenchStats standard_stats = computeStats(standard_rates);
    BenchStats interleaved_stats = computeStats(interleaved_rates);

    printStats("Standard (1 hash/iteration)", standard_stats);
    printf("\n");
    printStats("Interleaved (2 hashes/iteration)", interleaved_stats);

    // Compare
    printf("\n=== Comparison ===\n");
    double improvement = (interleaved_stats.mean - standard_stats.mean) / standard_stats.mean * 100.0;
    printf("  Improvement: %.2f%%\n", improvement);

    if (improvement > 0) {
        printf("  INTERLEAVED IS FASTER\n");
    } else {
        printf("  STANDARD IS FASTER\n");
    }

    // Cleanup
    if (worker) {
        free_huge_pages(worker);
    }

    return 0;
}
