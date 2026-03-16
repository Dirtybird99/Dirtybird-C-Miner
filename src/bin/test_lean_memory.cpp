/**
 * test_lean_memory.cpp - Test memory usage of lean vs original workers
 *
 * Build: Add to CMakeLists.txt as a test binary
 * Run: ./test_lean_memory
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>
#include <atomic>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

#include "worker_lean.hpp"
#include "astroworker.h"
#include "astrobwtv3/astrobwtv3.h"

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
    // Could read from /proc/self/status on Linux
#endif
}

void print_memory(const char* label) {
    size_t ws, pb;
    get_memory_usage(ws, pb);
    printf("%-30s Working: %6zu MB  Private: %6zu MB\n",
           label, ws / (1024*1024), pb / (1024*1024));
}

// Test original workerData memory
void test_original_workers(int num_workers) {
    printf("\n=== Testing Original workerData (num_workers=%d) ===\n", num_workers);
    print_memory("Before allocation:");

    std::vector<workerData*> workers(num_workers);

    for (int i = 0; i < num_workers; i++) {
        workers[i] = new workerData();
        initWorker(*workers[i]);
    }

    print_memory("After allocation:");

    // Touch memory to ensure it's committed
    for (int i = 0; i < num_workers; i++) {
        memset(workers[i]->sData, i & 0xFF, sizeof(workers[i]->sData));
        memset(workers[i]->sa, i & 0xFF, sizeof(workers[i]->sa));
    }

    print_memory("After touching memory:");

    // Compute a hash to trigger DeroBWT TLS allocation
    uint8_t input[48] = {0};
    uint8_t output[32];
    for (int i = 0; i < num_workers; i++) {
        AstroBWTv3(input, 48, output, *workers[i], false);
    }

    print_memory("After computing hashes:");

    // Cleanup
    for (auto* w : workers) delete w;

    print_memory("After cleanup:");
}

// Test lean workers memory
void test_lean_workers(int num_threads) {
    int num_workers = num_threads * 2;  // 2 workers per thread in interleaved mode
    printf("\n=== Testing Lean Workers (threads=%d, workers=%d) ===\n", num_threads, num_workers);
    print_memory("Before allocation:");

    std::vector<lean::LeanMinerThread*> threads(num_threads);

    for (int i = 0; i < num_threads; i++) {
        threads[i] = new lean::LeanMinerThread();
        threads[i]->init();
    }

    print_memory("After allocation:");

    // Touch memory to ensure it's committed
    for (int i = 0; i < num_threads; i++) {
        memset(threads[i]->worker_a.sData, i & 0xFF, sizeof(threads[i]->worker_a.sData));
        memset(threads[i]->worker_b.sData, i & 0xFF, sizeof(threads[i]->worker_b.sData));
        memset(threads[i]->shared.sa_a, i & 0xFF, sizeof(threads[i]->shared.sa_a));
        memset(threads[i]->shared.sa_b, i & 0xFF, sizeof(threads[i]->shared.sa_b));
    }

    print_memory("After touching memory:");

    // Cleanup
    for (auto* t : threads) delete t;

    print_memory("After cleanup:");
}

// Test with actual threading to measure TLS overhead
void test_threaded_original(int num_threads) {
    printf("\n=== Testing Threaded Original (threads=%d) ===\n", num_threads);
    print_memory("Before threads:");

    std::atomic<int> ready{0};
    std::atomic<bool> stop{false};
    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back([&ready, &stop, i]() {
            // Allocate thread-local worker
            workerData* worker = new workerData();
            initWorker(*worker);

            // Compute one hash to trigger TLS allocations
            uint8_t input[48] = {(uint8_t)i};
            uint8_t output[32];
            AstroBWTv3(input, 48, output, *worker, false);

            ready++;

            // Wait for signal to stop
            while (!stop) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }

            delete worker;
        });
    }

    // Wait for all threads to be ready
    while (ready < num_threads) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    print_memory("All threads ready:");

    // Signal stop
    stop = true;

    // Join all threads
    for (auto& t : threads) t.join();

    print_memory("After threads joined:");
}

// Test with actual threading using lean workers
void test_threaded_lean(int num_threads) {
    printf("\n=== Testing Threaded Lean (threads=%d, workers=%d) ===\n", num_threads, num_threads * 2);
    print_memory("Before threads:");

    std::atomic<int> ready{0};
    std::atomic<bool> stop{false};
    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back([&ready, &stop, i]() {
            // Use thread-local lean state
            lean::LeanMinerThread& state = lean::get_lean_thread_state();
            state.init();

            // Touch memory to commit
            memset(state.worker_a.sData, i, sizeof(state.worker_a.sData));
            memset(state.worker_b.sData, i, sizeof(state.worker_b.sData));

            ready++;

            // Wait for signal to stop
            while (!stop) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });
    }

    // Wait for all threads to be ready
    while (ready < num_threads) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    print_memory("All threads ready:");

    // Signal stop
    stop = true;

    // Join all threads
    for (auto& t : threads) t.join();

    print_memory("After threads joined:");
}

// Print size comparison
void print_size_comparison() {
    printf("\n=== Structure Size Comparison ===\n");
    printf("Original workerData:        %8zu bytes (%zu KB)\n",
           sizeof(workerData), sizeof(workerData) / 1024);
    printf("Lean WorkerLite:            %8zu bytes (%zu KB)\n",
           lean::WORKER_LITE_SIZE, lean::WORKER_LITE_SIZE / 1024);
    printf("Lean ThreadSharedState:     %8zu bytes (%zu KB)\n",
           lean::THREAD_SHARED_SIZE, lean::THREAD_SHARED_SIZE / 1024);
    printf("Lean LeanMinerThread:       %8zu bytes (%zu KB)\n",
           lean::LEAN_MINER_THREAD_SIZE, lean::LEAN_MINER_THREAD_SIZE / 1024);

    printf("\nFor 20 threads (interleaved = 40 workers):\n");
    size_t original_mem = 40 * sizeof(workerData);
    size_t lean_mem = 20 * lean::LEAN_MINER_THREAD_SIZE;
    printf("  Original: %zu MB (%zu KB/worker)\n",
           original_mem / (1024*1024), sizeof(workerData) / 1024);
    printf("  Lean:     %zu MB (%zu KB/thread)\n",
           lean_mem / (1024*1024), lean::LEAN_MINER_THREAD_SIZE / 1024);
    printf("  Savings:  %zu MB (%.1fx reduction)\n",
           (original_mem - lean_mem) / (1024*1024),
           (double)original_mem / lean_mem);
}

int main(int argc, char** argv) {
    printf("==============================================\n");
    printf("   Memory Usage Test: Original vs Lean\n");
    printf("==============================================\n");

    print_size_comparison();

    int num_threads = 20;
    if (argc > 1) {
        num_threads = atoi(argv[1]);
    }

    // Static allocation tests
    test_original_workers(num_threads * 2);  // 2 workers per thread
    test_lean_workers(num_threads);

    // Threaded tests (shows TLS overhead)
    test_threaded_original(num_threads);
    test_threaded_lean(num_threads);

    printf("\n=== Summary ===\n");
    printf("The lean architecture should use significantly less memory\n");
    printf("by sharing SA buffers per-thread instead of per-worker.\n");

    return 0;
}
