#pragma once
/**
 * worker_lean.hpp - Memory-optimized worker architecture
 *
 * Memory comparison (20 threads, 40 workers):
 *   Original: ~70 MB (1.1 MB/worker + 1.7 MB/thread TLS)
 *   Lean V2:  ~20 MB (75 KB/worker + 830 KB/thread shared)
 *
 * Key insight: SA computation buffers can be shared per-thread,
 * not duplicated per-worker. Only the scratchpad (sData) needs
 * to be per-worker.
 *
 * V2 uses custom_sa_70kb (~270 KB workspace) instead of DeroBWT (~2.3 MB).
 */

#ifndef WORKER_LEAN_HPP
#define WORKER_LEAN_HPP

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <openssl/sha.h>
#include <openssl/rc4.h>
#include "Salsa20.h"
#include "rc4_cryptogams.hpp"
#include "astrobwtv3/custom_sa_70kb.h"

// Configuration
#define LEAN_MAX_LENGTH ((256 * 277) - 1)
#define LEAN_SCRATCH_SIZE (LEAN_MAX_LENGTH + 64)

namespace lean {

/**
 * WorkerLite - Minimal per-worker state (~75 KB)
 *
 * Contains ONLY the data that must be unique per worker:
 * - Scratchpad for wolf iterations
 * - Crypto state (SHA, Salsa, RC4)
 * - Working chunk buffers
 *
 * Does NOT contain:
 * - SA output buffer (shared per-thread)
 * - DeroBWT working space (shared per-thread)
 * - SPSA bucket arrays (not needed with DeroBWT)
 */
struct alignas(64) WorkerLite {
    // === Crypto state (~700 bytes) ===
    SHA256_CTX sha256;                          // ~112 bytes
    ucstk::Salsa20 salsa20;                     // ~64 bytes
    RC4_KEY rc4_key;                            // ~264 bytes
    rc4_cryptogams::CryptogamsRc4 rc4_fast;     // ~264 bytes

    // === Working buffers (512 bytes) ===
    alignas(64) uint8_t chunk[256];             // Current iteration chunk
    alignas(64) uint8_t prev_chunk[256];        // Previous iteration chunk

    // === Scratch data (~71 KB) ===
    alignas(64) uint8_t sData[LEAN_SCRATCH_SIZE];

    // === Iteration state (~50 bytes) ===
    uint8_t salsaInput[256];
    uint8_t step_3[256];
    uint8_t op;
    uint8_t A;
    uint8_t pos1;
    uint8_t pos2;
    uint8_t t1;
    uint8_t t2;
    uint32_t data_len;
    uint16_t tries;
    uint64_t random_switcher;
    uint64_t lhash;
    uint64_t prev_lhash;

    // === Mask table for AVX2 (1 KB) ===
    alignas(32) uint8_t maskTable_bytes[32 * 33];

    // Padding for cache alignment
    uint8_t _padding[64];

    // Initialize worker state
    void init() {
        memset(salsaInput, 0, sizeof(salsaInput));
        memset(step_3, 0, sizeof(step_3));
        tries = 0;
        data_len = 0;

        // Initialize mask table for AVX2
        #if defined(__AVX2__)
        for (int i = 0; i < 33; i++) {
            int size = 32 - i;
            uint32_t a = ~(size > 28 ? 0xFFFFFFFF >> (std::max(4-(size - 28), 0)*8) : 0);
            uint32_t b = ~(size > 24 ? 0xFFFFFFFF >> (std::max(4-(size - 24), 0)*8) : 0);
            uint32_t c = ~(size > 20 ? 0xFFFFFFFF >> (std::max(4-(size - 20), 0)*8) : 0);
            uint32_t d = ~(size > 16 ? 0xFFFFFFFF >> (std::max(4-(size - 16), 0)*8) : 0);
            uint32_t e = ~(size > 12 ? 0xFFFFFFFF >> (std::max(4-(size - 12), 0)*8) : 0);
            uint32_t f = ~(size > 8  ? 0xFFFFFFFF >> (std::max(4-(size -  8), 0)*8) : 0);
            uint32_t g = ~(size > 4  ? 0xFFFFFFFF >> (std::max(4-(size -  4), 0)*8) : 0);
            uint32_t h = ~(size > 0  ? 0xFFFFFFFF >> (std::max(4-size, 0)*8) : 0);

            // Byte-swap for little-endian
            auto rev = [](uint32_t x) {
                return ((x >> 24) & 0xFF) | ((x >> 8) & 0xFF00) |
                       ((x << 8) & 0xFF0000) | ((x << 24) & 0xFF000000);
            };

            uint32_t vec[8] = {rev(a), rev(b), rev(c), rev(d), rev(e), rev(f), rev(g), rev(h)};
            memcpy(&maskTable_bytes[32 * i], vec, 32);
        }
        #endif

        // Initialize RC4 fast state
        new (&rc4_fast) rc4_cryptogams::CryptogamsRc4();
    }
};

/**
 * ThreadSharedState - Per-thread buffers shared between workers (~830 KB)
 *
 * Contains:
 * - SA output buffers (one per worker for parallel finalization)
 * - custom_sa_70kb workspace (bucket arrays + type bitvector)
 *
 * V2: Switched from DeroBWT (~2.3 MB) to custom_sa_70kb (~270 KB)
 */
struct alignas(64) ThreadSharedState {
    // SA output buffer - shared between 2 workers (283 KB each)
    // We need 2 because after interleaved iterations complete,
    // we compute SA for both workers
    alignas(64) int32_t sa_a[LEAN_MAX_LENGTH + 1];
    alignas(64) int32_t sa_b[LEAN_MAX_LENGTH + 1];

    // custom_sa_70kb workspace (~270 KB total)
    // bucket_A: 256 entries = 1 KB
    // bucket_B: 65536 entries = 256 KB
    // SAWorkspace: ~13 KB (bucket counts/head/tail + type bitvector)
    alignas(64) saidx_t bucket_A[256];
    alignas(64) saidx_t bucket_B[65536];
    alignas(64) SAWorkspace sa_workspace;

    // Statistics
    uint64_t hashes_computed;
    uint64_t sa_computations;

    void init() {
        hashes_computed = 0;
        sa_computations = 0;
        memset(bucket_A, 0, sizeof(bucket_A));
        memset(bucket_B, 0, sizeof(bucket_B));
    }

    // Compute SA using shared buffers
    void compute_sa(const uint8_t* data, size_t len, int32_t* sa_out) {
        custom_sa_70kb(data, sa_out, static_cast<saidx_t>(len), bucket_A, bucket_B);
        sa_computations++;
    }
};

/**
 * LeanMinerThread - Complete thread state for interleaved mining
 *
 * Memory per thread: ~1.0 MB (V2 with custom_sa_70kb)
 * - 2 x WorkerLite: 150 KB
 * - ThreadSharedState: ~830 KB (SA outputs + SA-IS workspace)
 *
 * vs Original: ~3.5 MB per thread
 * - 2 x workerData: 2.2 MB
 * - DeroBWT TLS: 1.3 MB per thread
 *
 * For 20 threads: ~20 MB total (vs DeroLuna's 10.5 MB)
 */
struct alignas(64) LeanMinerThread {
    // Two workers for interleaved execution
    WorkerLite worker_a;
    WorkerLite worker_b;

    // Shared state (SA buffers, DeroBWT)
    ThreadSharedState shared;

    // Pointers for compatibility with existing code
    uint8_t* chunk_a() { return worker_a.chunk; }
    uint8_t* chunk_b() { return worker_b.chunk; }
    uint8_t* prev_chunk_a() { return worker_a.prev_chunk; }
    uint8_t* prev_chunk_b() { return worker_b.prev_chunk; }

    void init() {
        worker_a.init();
        worker_b.init();
        shared.init();
    }

    // Get SA buffer for worker A/B
    int32_t* sa_for_a() { return shared.sa_a; }
    int32_t* sa_for_b() { return shared.sa_b; }
};

/**
 * Calculate memory usage
 */
constexpr size_t WORKER_LITE_SIZE = sizeof(WorkerLite);
constexpr size_t THREAD_SHARED_SIZE = sizeof(ThreadSharedState);
constexpr size_t LEAN_MINER_THREAD_SIZE = sizeof(LeanMinerThread);

// Compile-time size verification
static_assert(WORKER_LITE_SIZE < 80 * 1024, "WorkerLite should be < 80 KB");
static_assert(LEAN_MINER_THREAD_SIZE < 1200 * 1024, "LeanMinerThread should be < 1.2 MB");

/**
 * Global accessor for thread-local lean miner state
 */
inline LeanMinerThread& get_lean_thread_state() {
    static thread_local LeanMinerThread state;
    return state;
}

/**
 * Print memory usage summary
 */
inline void print_memory_usage(int num_threads) {
    printf("\n=== Lean Worker V2 Memory Usage ===\n");
    printf("WorkerLite size:       %6zu KB\n", WORKER_LITE_SIZE / 1024);
    printf("ThreadSharedState:     %6zu KB\n", THREAD_SHARED_SIZE / 1024);
    printf("LeanMinerThread total: %6zu KB\n", LEAN_MINER_THREAD_SIZE / 1024);
    printf("\nFor %d threads (2 workers each):\n", num_threads);
    printf("  Total workers: %d\n", num_threads * 2);
    printf("  Total memory:  %zu MB\n", (num_threads * LEAN_MINER_THREAD_SIZE) / (1024 * 1024));
    printf("\nComparison:\n");
    printf("  DeroLuna:   10.5 MB constant\n");
    printf("  Original:   ~%d MB (workerData + DeroBWT TLS)\n",
           (num_threads * 2 * 1100 + num_threads * 1770) / 1024);
    printf("  Lean V2:    %zu MB (custom_sa_70kb)\n",
           (num_threads * LEAN_MINER_THREAD_SIZE) / (1024 * 1024));
    printf("===================================\n\n");
}

} // namespace lean

#endif // WORKER_LEAN_HPP
