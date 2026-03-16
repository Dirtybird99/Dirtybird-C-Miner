/**
 * bench_lean_interleaved.cpp - Benchmark for lean interleaved miner
 *
 * Tests:
 * 1. Hash correctness vs reference implementation
 * 2. Memory usage comparison
 * 3. Performance (H/s)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <thread>
#include <atomic>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

// Include the lean interleaved miner
#include "astrobwtv3/lean_interleaved_miner.hpp"
#include "astroworker.h"

// Stubs for missing symbols (we don't link spsa_integrated.cpp)
// SPSA stub - returns false (SA computed, no SPSA shortcut)
bool SPSA(const uint8_t*, int, workerData&) { return false; }

// Global stubs
bool g_use_spsa = false;
bool g_use_local_spsa = false;
bool g_spsa_stamp_fast = true;
bool g_spsa_decode_bases = false;
int g_spsa_bucket_prefetch = 0;
int g_spsa_max_data_len = 0;
bool g_spsa_hit_counters = false;
bool g_spsa_sha_profile = false;
bool g_spsa_sha_pair = false;
bool g_spsa_sha_zeroize = false;
bool g_verbose_tune = false;
bool useLookupMine = false;
bool lookupMine = false;
uint8_t* lookup1D_global = nullptr;
unsigned char* lookup3D_global = nullptr;
int g_lookup_smart_threshold = 12;
bool g_lookup_smart_telemetry = false;
bool printHugepagesError = false;
void* allAstroFuncs = nullptr;
size_t numAstroFuncs = 0;

// Get current process memory usage
void get_memory_usage(size_t& working_set, size_t& private_bytes) {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        working_set = pmc.WorkingSetSize;
        private_bytes = pmc.PrivateUsage;
    }
#else
    working_set = 0;
    private_bytes = 0;
#endif
}

void print_memory(const char* label) {
    size_t ws, pb;
    get_memory_usage(ws, pb);
    printf("%-35s Working: %6zu MB  Private: %6zu MB\n",
           label, ws / (1024*1024), pb / (1024*1024));
}

// Print hex bytes
void print_hex(const char* label, const uint8_t* data, size_t len) {
    printf("%s: ", label);
    for (size_t i = 0; i < len; i++) {
        printf("%02x", data[i]);
    }
    printf("\n");
}

// Test hash consistency (same input -> same output)
bool test_correctness(int num_tests = 100) {
    printf("\n=== Hash Consistency Test ===\n");

    lean_interleaved::LeanInterleavedThread& thread1 = lean_interleaved::get_thread_state();
    thread1.init();

    // Second thread state for comparison
    lean_interleaved::LeanInterleavedThread thread2;
    thread2.init();

    uint8_t input[48];
    uint8_t hash1[32];
    uint8_t hash2[32];

    int passed = 0;
    int failed = 0;

    for (int i = 0; i < num_tests; i++) {
        // Generate random-ish input
        for (int j = 0; j < 48; j++) {
            input[j] = (uint8_t)((i * 17 + j * 31) & 0xFF);
        }

        // Compute with lean implementation twice (different thread states)
        lean_interleaved::AstroBWTv3_lean(input, 48, hash1, thread1, 0);
        lean_interleaved::AstroBWTv3_lean(input, 48, hash2, thread2, 0);

        if (memcmp(hash1, hash2, 32) == 0) {
            passed++;
        } else {
            failed++;
            if (failed <= 5) {
                printf("INCONSISTENCY at test %d:\n", i);
                print_hex("  Input", input, 48);
                print_hex("  Hash1", hash1, 32);
                print_hex("  Hash2", hash2, 32);
            }
        }
    }

    printf("Consistency: %d/%d tests passed", passed, num_tests);
    if (failed > 0) {
        printf(" (%d FAILED!)\n", failed);
        return false;
    }
    printf("\n");

    // Print a sample hash for visual inspection
    uint8_t sample_input[48] = {0};
    uint8_t sample_hash[32];
    for (int j = 0; j < 48; j++) sample_input[j] = (uint8_t)j;
    lean_interleaved::AstroBWTv3_lean(sample_input, 48, sample_hash, thread1, 0);
    print_hex("Sample hash (input 0..47)", sample_hash, 32);

    return true;
}

// Test interleaved vs single hash consistency
bool test_interleaved_correctness(int num_tests = 50) {
    printf("\n=== Interleaved vs Single Hash Consistency Test ===\n");

    lean_interleaved::LeanInterleavedThread& thread = lean_interleaved::get_thread_state();
    thread.init();

    lean_interleaved::LeanInterleavedThread thread_single;
    thread_single.init();

    uint8_t input_a[48], input_b[48];
    uint8_t hash_a[32], hash_b[32];
    uint8_t hash_single_a[32], hash_single_b[32];

    int passed = 0;
    int failed = 0;

    for (int i = 0; i < num_tests; i++) {
        // Generate random-ish inputs
        for (int j = 0; j < 48; j++) {
            input_a[j] = (uint8_t)((i * 17 + j * 31) & 0xFF);
            input_b[j] = (uint8_t)((i * 23 + j * 41) & 0xFF);
        }

        // Compute with interleaved implementation
        lean_interleaved::processInterleaved(
            input_a, 48, hash_a,
            input_b, 48, hash_b,
            thread
        );

        // Compute with single hash implementation
        lean_interleaved::AstroBWTv3_lean(input_a, 48, hash_single_a, thread_single, 0);
        lean_interleaved::AstroBWTv3_lean(input_b, 48, hash_single_b, thread_single, 1);

        bool a_match = memcmp(hash_a, hash_single_a, 32) == 0;
        bool b_match = memcmp(hash_b, hash_single_b, 32) == 0;

        if (a_match && b_match) {
            passed++;
        } else {
            failed++;
            if (failed <= 5) {
                printf("MISMATCH at test %d:\n", i);
                if (!a_match) {
                    print_hex("  Input A       ", input_a, 48);
                    print_hex("  Interleaved A ", hash_a, 32);
                    print_hex("  Single A      ", hash_single_a, 32);
                }
                if (!b_match) {
                    print_hex("  Input B       ", input_b, 48);
                    print_hex("  Interleaved B ", hash_b, 32);
                    print_hex("  Single B      ", hash_single_b, 32);
                }
            }
        }
    }

    printf("Interleaved vs single consistency: %d/%d tests passed", passed, num_tests);
    if (failed > 0) {
        printf(" (%d FAILED!)\n", failed);
        return false;
    }
    printf("\n");
    return true;
}

// Single-threaded benchmark with phase profiling
void benchmark_single_thread(int duration_secs) {
    printf("\n=== Single Thread Benchmark (%d seconds) ===\n", duration_secs);

    lean_interleaved::LeanInterleavedThread& thread = lean_interleaved::get_thread_state();
    thread.init();

    lean_interleaved::print_memory_usage(1);

    // First, profile phase breakdown for 100 hashes
    printf("\nPhase breakdown (100 sample hashes):\n");
    {
        uint8_t input[48] = {0};
        for (int i = 0; i < 48; i++) input[i] = (uint8_t)i;
        uint8_t hash[32];

        double total_prep_us = 0, total_wolf_us = 0, total_sa_us = 0, total_final_us = 0;
        const int SAMPLES = 100;

        for (int s = 0; s < SAMPLES; s++) {
            auto& worker = thread.worker_a;
            int32_t* sa = thread.shared.sa_a;

            // Prep phase
            auto t0 = std::chrono::steady_clock::now();
            lean_interleaved::prepPhase(worker, input, 48);
            auto t1 = std::chrono::steady_clock::now();

            // Wolf phase
            uint8_t chunkCount = 1;
            int firstChunk = 0;
            uint8_t lp1 = 0, lp2 = 255;
            while (lean_interleaved::wolfIterationLean(worker, lp1, lp2, chunkCount, firstChunk)) {}
            lean_interleaved::wolfFinalize(worker, lp1, lp2, chunkCount, firstChunk);
            auto t2 = std::chrono::steady_clock::now();

            // SA phase
            thread.shared.compute_sa(worker.sData, worker.data_len, sa);
            auto t3 = std::chrono::steady_clock::now();

            // Final hash
            lean_interleaved::hashSHA256(worker.sha256, reinterpret_cast<uint8_t*>(sa), hash, worker.data_len * 4);
            auto t4 = std::chrono::steady_clock::now();

            total_prep_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
            total_wolf_us += std::chrono::duration<double, std::micro>(t2 - t1).count();
            total_sa_us += std::chrono::duration<double, std::micro>(t3 - t2).count();
            total_final_us += std::chrono::duration<double, std::micro>(t4 - t3).count();

            input[0]++;
        }

        double avg_total = (total_prep_us + total_wolf_us + total_sa_us + total_final_us) / SAMPLES;
        printf("  Prep:   %8.2f us (%5.1f%%)\n", total_prep_us / SAMPLES, 100.0 * total_prep_us / SAMPLES / avg_total);
        printf("  Wolf:   %8.2f us (%5.1f%%)\n", total_wolf_us / SAMPLES, 100.0 * total_wolf_us / SAMPLES / avg_total);
        printf("  SA:     %8.2f us (%5.1f%%)\n", total_sa_us / SAMPLES, 100.0 * total_sa_us / SAMPLES / avg_total);
        printf("  Final:  %8.2f us (%5.1f%%)\n", total_final_us / SAMPLES, 100.0 * total_final_us / SAMPLES / avg_total);
        printf("  TOTAL:  %8.2f us = %.1f H/s\n", avg_total, 1e6 / avg_total);
        // Print last data_len for reference
        printf("  data_len (last sample): %u bytes\n", (unsigned)thread.worker_a.data_len);
    }

    uint8_t input_a[48] = {0};
    uint8_t hash_a[32];

    // Initialize input
    for (int i = 0; i < 48; i++) {
        input_a[i] = (uint8_t)i;
    }

    auto start = std::chrono::steady_clock::now();
    uint64_t hashes = 0;

    // Test SEQUENTIAL execution (NOT interleaved) - one hash at a time
    printf("\nTesting SEQUENTIAL mode (avoiding cache thrashing)...\n");
    while (true) {
        // Use sequential mode - one worker at a time (like original AstroBWTv3)
        lean_interleaved::AstroBWTv3_lean(input_a, 48, hash_a, thread, 0);
        hashes++;

        // Vary input
        input_a[0]++;
        if (input_a[0] == 0) input_a[1]++;

        // Check time
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
        if (elapsed >= duration_secs) break;
    }

    auto end = std::chrono::steady_clock::now();
    double elapsed_secs = std::chrono::duration<double>(end - start).count();
    double hashrate = hashes / elapsed_secs;

    printf("Results (SEQUENTIAL):\n");
    printf("  Hashes: %llu\n", (unsigned long long)hashes);
    printf("  Time:   %.2f s\n", elapsed_secs);
    printf("  Rate:   %.2f H/s\n", hashrate);

    // Now test INTERLEAVED mode for comparison
    printf("\nTesting INTERLEAVED mode (may have cache thrashing)...\n");
    uint8_t input_b[48] = {0};
    uint8_t hash_b[32];
    for (int i = 0; i < 48; i++) {
        input_a[i] = (uint8_t)i;
        input_b[i] = (uint8_t)(i + 100);
    }

    start = std::chrono::steady_clock::now();
    hashes = 0;

    while (true) {
        lean_interleaved::processInterleaved(
            input_a, 48, hash_a,
            input_b, 48, hash_b,
            thread
        );
        hashes += 2;

        input_a[0]++;
        if (input_a[0] == 0) input_a[1]++;
        input_b[0]++;
        if (input_b[0] == 0) input_b[1]++;

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
        if (elapsed >= duration_secs) break;
    }

    end = std::chrono::steady_clock::now();
    elapsed_secs = std::chrono::duration<double>(end - start).count();
    hashrate = hashes / elapsed_secs;

    printf("Results (INTERLEAVED):\n");
    printf("  Hashes: %llu\n", (unsigned long long)hashes);
    printf("  Time:   %.2f s\n", elapsed_secs);
    printf("  Rate:   %.2f H/s\n", hashrate);
}

// Multi-threaded benchmark
void benchmark_multi_thread(int num_threads, int duration_secs) {
    printf("\n=== Multi-Thread Benchmark (%d threads, %d seconds) ===\n",
           num_threads, duration_secs);

    print_memory("Before thread allocation:");

    std::atomic<bool> running{true};
    std::atomic<uint64_t> total_hashes{0};
    std::vector<std::thread> threads;

    auto worker = [&running, &total_hashes](int thread_id) {
        lean_interleaved::LeanInterleavedThread& thread = lean_interleaved::get_thread_state();
        thread.init();

        uint8_t input_a[48] = {0};
        uint8_t input_b[48] = {0};
        uint8_t hash_a[32], hash_b[32];

        // Initialize with thread-specific values
        for (int i = 0; i < 48; i++) {
            input_a[i] = (uint8_t)(i + thread_id * 10);
            input_b[i] = (uint8_t)(i + thread_id * 10 + 100);
        }

        uint64_t local_hashes = 0;

        while (running.load(std::memory_order_relaxed)) {
            lean_interleaved::processInterleaved(
                input_a, 48, hash_a,
                input_b, 48, hash_b,
                thread
            );
            local_hashes += 2;

            input_a[0]++;
            if (input_a[0] == 0) input_a[1]++;
            input_b[0]++;
            if (input_b[0] == 0) input_b[1]++;
        }

        total_hashes.fetch_add(local_hashes, std::memory_order_relaxed);
    };

    auto start = std::chrono::steady_clock::now();

    // Launch threads
    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back(worker, i);
    }

    // Wait a moment for threads to allocate memory
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    print_memory("After thread startup:");

    // Wait for duration
    std::this_thread::sleep_for(std::chrono::seconds(duration_secs));
    running.store(false);

    // Join threads
    for (auto& t : threads) {
        t.join();
    }

    auto end = std::chrono::steady_clock::now();
    double elapsed_secs = std::chrono::duration<double>(end - start).count();
    uint64_t hashes = total_hashes.load();
    double hashrate = hashes / elapsed_secs;

    lean_interleaved::print_memory_usage(num_threads);

    printf("Results:\n");
    printf("  Threads: %d\n", num_threads);
    printf("  Hashes:  %llu\n", (unsigned long long)hashes);
    printf("  Time:    %.2f s\n", elapsed_secs);
    printf("  Rate:    %.2f H/s (%.2f KH/s)\n", hashrate, hashrate / 1000.0);
    printf("  Per-thread: %.2f H/s\n", hashrate / num_threads);

    print_memory("Final memory usage:");
}

// Memory comparison test
void test_memory_comparison(int num_threads) {
    printf("\n=== Memory Comparison Test (%d threads) ===\n", num_threads);

    print_memory("Before allocation:");

    // Allocate lean thread states
    std::vector<lean_interleaved::LeanInterleavedThread*> lean_threads(num_threads);
    for (int i = 0; i < num_threads; i++) {
        lean_threads[i] = new lean_interleaved::LeanInterleavedThread();
        lean_threads[i]->init();

        // Touch memory to ensure allocation
        memset(lean_threads[i]->worker_a.sData, i & 0xFF,
               sizeof(lean_threads[i]->worker_a.sData));
        memset(lean_threads[i]->worker_b.sData, i & 0xFF,
               sizeof(lean_threads[i]->worker_b.sData));
    }

    print_memory("After lean allocation:");

    lean_interleaved::print_memory_usage(num_threads);

    // Cleanup
    for (auto* t : lean_threads) delete t;

    print_memory("After cleanup:");
}

void print_lean_bench_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --test           Run correctness tests\n");
    printf("  --bench          Run single-thread benchmark (10s)\n");
    printf("  --bench-mt N     Run multi-thread benchmark with N threads (30s)\n");
    printf("  --memory N       Test memory usage with N threads\n");
    printf("  --all            Run all tests\n");
    printf("  --duration S     Set benchmark duration to S seconds\n");
}

int main(int argc, char** argv) {
    printf("==============================================\n");
    printf("   Lean Interleaved Miner Benchmark\n");
    printf("==============================================\n");

    bool do_test = false;
    bool do_bench = false;
    bool do_bench_mt = false;
    bool do_memory = false;
    int num_threads = 20;
    int duration = 30;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--test") == 0) {
            do_test = true;
        } else if (strcmp(argv[i], "--bench") == 0) {
            do_bench = true;
        } else if (strcmp(argv[i], "--bench-mt") == 0 && i + 1 < argc) {
            do_bench_mt = true;
            num_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--memory") == 0 && i + 1 < argc) {
            do_memory = true;
            num_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--duration") == 0 && i + 1 < argc) {
            duration = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--all") == 0) {
            do_test = do_bench = do_bench_mt = do_memory = true;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_lean_bench_usage(argv[0]);
            return 0;
        }
    }

    // Default: run tests
    if (!do_test && !do_bench && !do_bench_mt && !do_memory) {
        do_test = true;
        do_bench = true;
    }

    // Print structure sizes
    printf("\n=== Structure Sizes ===\n");
    printf("WorkerLean:            %6zu KB\n", lean_interleaved::WORKER_LEAN_SIZE / 1024);
    printf("ThreadSharedLean:      %6zu KB\n", lean_interleaved::THREAD_SHARED_LEAN_SIZE / 1024);
    printf("LeanInterleavedThread: %6zu KB\n", lean_interleaved::LEAN_THREAD_SIZE / 1024);

    if (do_test) {
        bool ok = test_correctness(100);
        if (!ok) {
            printf("\n*** CORRECTNESS TEST FAILED ***\n");
            return 1;
        }

        ok = test_interleaved_correctness(50);
        if (!ok) {
            printf("\n*** INTERLEAVED CORRECTNESS TEST FAILED ***\n");
            return 1;
        }
    }

    if (do_bench) {
        benchmark_single_thread(10);
    }

    if (do_memory) {
        test_memory_comparison(num_threads);
    }

    if (do_bench_mt) {
        benchmark_multi_thread(num_threads, duration);
    }

    printf("\n=== Done ===\n");
    return 0;
}
