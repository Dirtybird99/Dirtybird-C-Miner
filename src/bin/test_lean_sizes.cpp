/**
 * test_lean_sizes.cpp - Simple test to verify lean worker memory sizes
 *
 * This test only measures structure sizes, no hash computation needed.
 * Build with minimal dependencies.
 *
 * V2: Uses custom_sa_70kb instead of DeroBWT for much smaller memory footprint.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <atomic>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

// Include the headers to get sizeof
#include "worker_lean.hpp"
#include "astroworker.h"
// custom_sa_70kb.h is already included via worker_lean.hpp

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

int main(int argc, char** argv) {
    printf("==============================================\n");
    printf("   Memory Size Comparison: Original vs Lean\n");
    printf("==============================================\n\n");

    // Print structure sizes
    printf("=== Structure Sizes ===\n\n");

    printf("Original workerData:\n");
    printf("  sizeof(workerData):     %8zu bytes (%6zu KB)\n",
           sizeof(workerData), sizeof(workerData) / 1024);

    printf("\nLean structures:\n");
    printf("  sizeof(WorkerLite):         %8zu bytes (%6zu KB)\n",
           lean::WORKER_LITE_SIZE, lean::WORKER_LITE_SIZE / 1024);
    printf("  sizeof(ThreadSharedState):  %8zu bytes (%6zu KB)\n",
           lean::THREAD_SHARED_SIZE, lean::THREAD_SHARED_SIZE / 1024);
    printf("  sizeof(LeanMinerThread):    %8zu bytes (%6zu KB)\n",
           lean::LEAN_MINER_THREAD_SIZE, lean::LEAN_MINER_THREAD_SIZE / 1024);

    printf("\n=== Component Breakdown ===\n\n");

    printf("Original workerData components (estimated):\n");
    printf("  sData (scratch):     %6zu KB\n", ASTRO_SCRATCH_SIZE / 1024);
    printf("  sa (suffix array):   %6zu KB\n", (277*256+1) * sizeof(int32_t) / 1024);
    printf("  buckets_d:           %6zu KB (SPSA only)\n", 256*256*sizeof(uint16_t) / 1024);
    printf("  bHeads:              %6zu KB (SPSA only)\n", 256*256*sizeof(uint32_t) / 1024);
    printf("  bHeadIdx:            %6zu KB (SPSA only)\n", 256*256*sizeof(uint32_t) / 1024);
    printf("  Other fields:        ~%4d KB\n", 10);

    printf("\nLean WorkerLite components:\n");
    printf("  sData (scratch):     %6zu KB\n", LEAN_SCRATCH_SIZE / 1024);
    printf("  Crypto state:        ~%4d KB (SHA, Salsa, RC4)\n", 1);
    printf("  Working buffers:     ~%4d KB (chunk, prev_chunk)\n", 1);
    printf("  maskTable:           ~%4d KB\n", 1);

    printf("\nLean ThreadSharedState V2 components:\n");
    printf("  sa_a:                %6zu KB\n", (LEAN_MAX_LENGTH+1) * sizeof(int32_t) / 1024);
    printf("  sa_b:                %6zu KB\n", (LEAN_MAX_LENGTH+1) * sizeof(int32_t) / 1024);
    printf("  bucket_A:            %6zu KB\n", 256 * sizeof(saidx_t) / 1024);
    printf("  bucket_B:            %6zu KB\n", 65536 * sizeof(saidx_t) / 1024);
    printf("  SAWorkspace:         %6zu KB\n", sizeof(SAWorkspace) / 1024);

    // Memory scaling comparison
    printf("\n=== Memory Scaling (20 threads, 40 workers) ===\n\n");

    int threads = 20;
    int workers = 40;

    size_t original_per_worker = sizeof(workerData);
    size_t original_total = workers * original_per_worker;

    size_t lean_per_thread = lean::LEAN_MINER_THREAD_SIZE;
    size_t lean_total = threads * lean_per_thread;

    printf("Original architecture:\n");
    printf("  %d workers x %zu KB = %zu MB\n",
           workers, original_per_worker / 1024, original_total / (1024*1024));
    printf("  + DeroBWT TLS per thread: ~1.77 MB x %d = ~%d MB\n",
           threads, (int)(threads * 1.77));
    printf("  TOTAL (estimated): ~%d MB\n",
           (int)(original_total / (1024*1024) + threads * 1.77));

    printf("\nLean architecture:\n");
    printf("  %d threads x %zu KB = %zu MB\n",
           threads, lean_per_thread / 1024, lean_total / (1024*1024));
    printf("  (Each thread has 2 workers + shared SA buffers)\n");

    size_t savings = original_total + (size_t)(threads * 1.77 * 1024 * 1024) - lean_total;
    printf("\nEstimated savings: %zu MB (%.1fx reduction)\n",
           savings / (1024*1024),
           (double)(original_total + threads * 1.77 * 1024 * 1024) / lean_total);

    // Actual memory allocation test
    printf("\n=== Actual Memory Allocation Test ===\n\n");

    print_memory("Before any allocation:");

    // Test 1: Allocate original workers
    printf("\n--- Test 1: Original workerData (40 workers) ---\n");
    {
        std::vector<workerData*> originals(workers);
        for (int i = 0; i < workers; i++) {
            originals[i] = new workerData();
            // Touch memory to commit
            memset(originals[i]->sData, i & 0xFF, sizeof(originals[i]->sData));
            memset(originals[i]->sa, i & 0xFF, sizeof(originals[i]->sa));
        }
        print_memory("After allocating 40 workerData:");

        // Cleanup
        for (auto* w : originals) delete w;
    }
    print_memory("After cleanup:");

    // Test 2: Allocate lean structures
    printf("\n--- Test 2: Lean Workers (20 threads) ---\n");
    {
        std::vector<lean::LeanMinerThread*> leans(threads);
        for (int i = 0; i < threads; i++) {
            leans[i] = new lean::LeanMinerThread();
            leans[i]->init();
            // Touch memory to commit
            memset(leans[i]->worker_a.sData, i & 0xFF, sizeof(leans[i]->worker_a.sData));
            memset(leans[i]->worker_b.sData, i & 0xFF, sizeof(leans[i]->worker_b.sData));
            memset(leans[i]->shared.sa_a, i & 0xFF, sizeof(leans[i]->shared.sa_a));
            memset(leans[i]->shared.sa_b, i & 0xFF, sizeof(leans[i]->shared.sa_b));
        }
        print_memory("After allocating 20 LeanMinerThread:");

        // Cleanup
        for (auto* l : leans) delete l;
    }
    print_memory("After cleanup:");

    // DeroLuna comparison
    printf("\n=== DeroLuna Comparison ===\n\n");
    printf("DeroLuna measured: 10.5 MB constant (any thread count)\n");
    printf("  Per worker (40): 262 KB\n");
    printf("  Per thread (20): 525 KB\n\n");

    printf("Our lean architecture:\n");
    printf("  Per thread: %zu KB\n", lean_per_thread / 1024);
    printf("  Total (20 threads): %zu MB\n", lean_total / (1024*1024));

    if (lean_total / (1024*1024) > 10) {
        printf("\nStill larger than DeroLuna. Possible reasons:\n");
        printf("  1. DeroLuna may use stack allocation for temps\n");
        printf("  2. DeroLuna may share SA buffers across all threads\n");
        printf("  3. DeroLuna may use a different SA algorithm entirely\n");
    }

    printf("\n=== Summary ===\n\n");
    printf("The lean architecture reduces memory by sharing SA buffers\n");
    printf("per-thread instead of duplicating them per-worker.\n");
    printf("This reduces memory from ~%d MB to ~%zu MB for 20 threads.\n",
           (int)(original_total / (1024*1024) + threads * 1.77),
           lean_total / (1024*1024));

    return 0;
}
