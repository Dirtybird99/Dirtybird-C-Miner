/**
 * lean_interleaved_miner.hpp - Memory-optimized interleaved miner
 *
 * Combines:
 * - Lean V2 architecture (18 MB for 20 threads vs 78 MB original)
 * - Interleaved execution (DeroLuna-style ILP)
 * - custom_sa_70kb for SA computation
 *
 * Memory per thread: ~1 MB (2 workers + shared SA buffers)
 * vs Original: ~5.5 MB per thread (2 workerData + DeroBWT TLS)
 */

#ifndef LEAN_INTERLEAVED_MINER_HPP
#define LEAN_INTERLEAVED_MINER_HPP

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <chrono>

#include <openssl/sha.h>
#include <openssl/rc4.h>
#include "Salsa20.h"
#include "rc4_cryptogams.hpp"

extern "C" {
#include "divsufsort.h"
#ifdef USE_CUSTOM_SA
#include "custom_sa_70kb.h"
#endif
}

// Hash functions
#include <fnv1a.h>
#include <xxhash64.h>
#include <highwayhash/sip_hash.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

// Include branch_tables header BEFORE opening namespace
#include "branch_tables.hpp"

// Configuration
#define LEAN_MAX_LENGTH ((256 * 277) - 1)
#define LEAN_SCRATCH_SIZE (LEAN_MAX_LENGTH + 64)
#define LEAN_MINPREFLEN 4

namespace lean_interleaved {

/**
 * Template marker for SPSA integration
 */
struct templateMarker {
    uint8_t pos1;
    uint8_t pos2;
    uint16_t reserved1;
    uint16_t reserved2;
    uint16_t chunkInfo;  // (firstChunk << 7) | chunkCount
};

/**
 * WorkerLean - Minimal per-worker state (~75 KB)
 *
 * Designed for interleaved execution where two workers share thread resources.
 */
struct alignas(64) WorkerLean {
    // === Crypto state (~700 bytes) ===
    SHA256_CTX sha256;
    ucstk::Salsa20 salsa20;
    RC4_KEY rc4_key;
    rc4_cryptogams::CryptogamsRc4 rc4_fast;

    // === Scratch data (~71 KB) - main memory consumer ===
    alignas(64) uint8_t sData[LEAN_SCRATCH_SIZE];

    // === Working pointers ===
    uint8_t* chunk;       // Current chunk in sData
    uint8_t* prev_chunk;  // Previous chunk in sData

    // === Iteration state ===
    alignas(32) uint8_t salsaInput[256];
    uint8_t op;
    uint8_t A;
    uint8_t pos1;
    uint8_t pos2;
    uint8_t t1;
    uint8_t t2;
    uint32_t data_len;
    uint16_t tries;
    uint16_t templateIdx;
    uint64_t random_switcher;
    uint64_t lhash;
    uint64_t prev_lhash;
    bool isSame;

    // === Template tracking for SPSA ===
    alignas(64) templateMarker astroTemplate[280];

    // === Mask table for AVX2 (1 KB) ===
    alignas(32) uint8_t maskTable[32 * 33];

    void init() {
        tries = 0;
        templateIdx = 0;
        data_len = 0;
        isSame = false;
        chunk = sData;
        prev_chunk = sData;

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

            auto rev = [](uint32_t x) {
                return ((x >> 24) & 0xFF) | ((x >> 8) & 0xFF00) |
                       ((x << 8) & 0xFF0000) | ((x << 24) & 0xFF000000);
            };

            uint32_t vec[8] = {rev(a), rev(b), rev(c), rev(d), rev(e), rev(f), rev(g), rev(h)};
            memcpy(&maskTable[32 * i], vec, 32);
        }
        #endif

        // Initialize RC4 fast state
        new (&rc4_fast) rc4_cryptogams::CryptogamsRc4();
    }
};

/**
 * ThreadSharedLean - Per-thread shared buffers (~830 KB)
 *
 * Contains SA output buffers and custom_sa_70kb workspace.
 */
struct alignas(64) ThreadSharedLean {
    // SA output buffers - one per worker for parallel finalization
    alignas(64) int32_t sa_a[LEAN_MAX_LENGTH + 1];
    alignas(64) int32_t sa_b[LEAN_MAX_LENGTH + 1];

    // custom_sa_70kb workspace
    alignas(64) saidx_t bucket_A[256];
    alignas(64) saidx_t bucket_B[65536];

    // Statistics
    uint64_t hashes_computed;

    void init() {
        hashes_computed = 0;
        memset(bucket_A, 0, sizeof(bucket_A));
        memset(bucket_B, 0, sizeof(bucket_B));
    }

    // Compute SA using custom_sa_70kb (optimized) or divsufsort (fallback)
    void compute_sa(const uint8_t* data, size_t len, int32_t* sa_out) {
#ifdef USE_CUSTOM_SA
        custom_sa_70kb(data, sa_out, static_cast<saidx_t>(len), bucket_A, bucket_B);
#else
        divsufsort(data, sa_out, static_cast<saidx_t>(len), bucket_A, bucket_B);
#endif
    }
};

/**
 * LeanInterleavedThread - Complete thread state (~1 MB)
 *
 * Contains two workers for interleaved execution plus shared SA buffers.
 */
struct alignas(64) LeanInterleavedThread {
    WorkerLean worker_a;
    WorkerLean worker_b;
    ThreadSharedLean shared;

    void init() {
        worker_a.init();
        worker_b.init();
        shared.init();
    }
};

// Size constants
constexpr size_t WORKER_LEAN_SIZE = sizeof(WorkerLean);
constexpr size_t THREAD_SHARED_LEAN_SIZE = sizeof(ThreadSharedLean);
constexpr size_t LEAN_THREAD_SIZE = sizeof(LeanInterleavedThread);

// Compile-time verification
static_assert(WORKER_LEAN_SIZE < 85 * 1024, "WorkerLean should be < 85 KB");
static_assert(LEAN_THREAD_SIZE < 1200 * 1024, "LeanInterleavedThread should be < 1.2 MB");

/**
 * Thread-local accessor
 */
inline LeanInterleavedThread& get_thread_state() {
    static thread_local LeanInterleavedThread state;
    return state;
}

// ============================================================================
// Helper Functions
// ============================================================================

// Rotate left 8-bit
inline uint8_t rl8(uint8_t val, int shift) {
    return (val << shift) | (val >> (8 - shift));
}

// Reverse bits in byte
inline uint8_t reverse8(uint8_t b) {
    b = (b & 0xF0) >> 4 | (b & 0x0F) << 4;
    b = (b & 0xCC) >> 2 | (b & 0x33) << 2;
    b = (b & 0xAA) >> 1 | (b & 0x55) << 1;
    return b;
}

// SHA256 helper
inline void hashSHA256(SHA256_CTX& ctx, const uint8_t* input, uint8_t* output, size_t len) {
    SHA256_Init(&ctx);
    SHA256_Update(&ctx, input, len);
    SHA256_Final(output, &ctx);
}

// AVX2 256-byte copy
inline void memcpy256_avx2(uint8_t* dst, const uint8_t* src) {
    #if defined(__AVX2__)
    _mm256_storeu_si256((__m256i*)&dst[0], _mm256_loadu_si256((const __m256i*)&src[0]));
    _mm256_storeu_si256((__m256i*)&dst[32], _mm256_loadu_si256((const __m256i*)&src[32]));
    _mm256_storeu_si256((__m256i*)&dst[64], _mm256_loadu_si256((const __m256i*)&src[64]));
    _mm256_storeu_si256((__m256i*)&dst[96], _mm256_loadu_si256((const __m256i*)&src[96]));
    _mm256_storeu_si256((__m256i*)&dst[128], _mm256_loadu_si256((const __m256i*)&src[128]));
    _mm256_storeu_si256((__m256i*)&dst[160], _mm256_loadu_si256((const __m256i*)&src[160]));
    _mm256_storeu_si256((__m256i*)&dst[192], _mm256_loadu_si256((const __m256i*)&src[192]));
    _mm256_storeu_si256((__m256i*)&dst[224], _mm256_loadu_si256((const __m256i*)&src[224]));
    #else
    memcpy(dst, src, 256);
    #endif
}

// ============================================================================
// Wolf Permute - Branch Operations
// ============================================================================

/**
 * Cached AVX2 availability flag - initialized once per thread
 * Avoids CPUID on every wolf iteration (was major perf bug)
 */
inline bool has_avx2_cached() {
    static bool cached = ::branch_tables::avx2_available();
    return cached;
}

/**
 * wolfPermute - Apply transformation based on opcode
 *
 * This is the core computation that transforms chunk data.
 * Uses lookup table and SIMD when available.
 */
inline void wolfPermuteLean(const uint8_t* in, uint8_t* out, uint8_t op, uint8_t pos1, uint8_t pos2) {
    #if defined(__AVX2__) || defined(__x86_64__) || defined(_M_X64)
    if (pos2 > pos1 && has_avx2_cached()) {
        ::branch_tables::wolf_permute_avx2(in, out, op, pos1, pos2);
        return;
    }
    #endif
    ::branch_tables::wolf_permute_scalar(in, out, op, pos1, pos2);
}

// ============================================================================
// Core Mining Functions
// ============================================================================

/**
 * prepPhase - Initialize worker for new hash computation
 *
 * Steps 1-4 of AstroBWTv3: SHA256 -> Salsa20 -> RC4
 */
inline void prepPhase(WorkerLean& worker, const uint8_t* input, size_t len) {
    uint8_t scratch[384] = {0};

    // Step 1: SHA256 of input
    hashSHA256(worker.sha256, input, &scratch[320], len);

    // Copy to salsaInput for state tracking
    memcpy(worker.salsaInput, &scratch[320], 32);

    // Step 2: Salsa20 expansion to 256 bytes
    worker.salsa20.setKey(&scratch[320]);
    worker.salsa20.setIv(&scratch[256]);  // IV is zeros
    worker.salsa20.processBytes(scratch, scratch, 256);

    // Step 3: RC4 key setup and encryption
    worker.rc4_fast.set_key(scratch, 256);
    RC4_set_key(&worker.rc4_key, 256, scratch);
    worker.rc4_fast.apply_keystream_256(scratch);

    // Step 4: Initialize worker state
    worker.lhash = hash_64_fnv1a_256_optimized(scratch);
    worker.prev_lhash = worker.lhash;
    worker.tries = 0;
    worker.templateIdx = 0;
    worker.isSame = false;
    worker.pos1 = 0;
    worker.pos2 = 255;
    worker.op = 0;
    worker.data_len = 0;

    // Copy initial data to sData
    memcpy(worker.sData, scratch, 256);

    // Initialize chunk pointers
    worker.chunk = worker.sData;
    worker.prev_chunk = worker.sData;
}

/**
 * wolfIterationLean - Process single wolf iteration
 *
 * Returns true if more iterations needed, false if done.
 */
inline bool wolfIterationLean(
    WorkerLean& worker,
    uint8_t& lp1, uint8_t& lp2,
    uint8_t& chunkCount, int& firstChunk
) {
    worker.tries++;
    worker.random_switcher = worker.prev_lhash ^ worker.lhash ^ worker.tries;

    worker.op = static_cast<uint8_t>(worker.random_switcher);

    uint8_t p1 = static_cast<uint8_t>(worker.random_switcher >> 8);
    uint8_t p2 = static_cast<uint8_t>(worker.random_switcher >> 16);

    if (p1 > p2) std::swap(p1, p2);
    if (p2 - p1 > 32) p2 = p1 + ((p2 - p1) & 0x1f);

    if (worker.tries > 0) {
        lp1 = std::min(lp1, p1);
        lp2 = std::max(lp2, p2);
    }

    if (p1 < worker.pos1 || p2 > worker.pos2) {
        worker.isSame = false;
    }

    worker.pos1 = p1;
    worker.pos2 = p2;

    // Set chunk pointers
    worker.chunk = &worker.sData[(worker.tries - 1) * 256];

    if (worker.tries == 1) {
        worker.prev_chunk = worker.chunk;
    } else {
        worker.prev_chunk = &worker.sData[(worker.tries - 2) * 256];
        memcpy256_avx2(worker.chunk, worker.prev_chunk);
    }

    // Prefetch for next iteration
    #if defined(__x86_64__) || defined(_M_X64)
    _mm_prefetch(reinterpret_cast<const char*>(&worker.prev_chunk[worker.pos1]), _MM_HINT_T0);
    #endif

    // Handle op 253 special case
    if (worker.op == 253) {
        for (int i = worker.pos1; i < worker.pos2; i++) {
            worker.chunk[i] = worker.prev_chunk[i];
        }
        for (int i = worker.pos1; i < worker.pos2; i++) {
            worker.chunk[i] = rl8(worker.chunk[i], 3);
            worker.chunk[i] ^= rl8(worker.chunk[i], 2);
            worker.chunk[i] ^= worker.prev_chunk[worker.pos2];
            worker.chunk[i] = rl8(worker.chunk[i], 3);

            worker.prev_lhash = worker.lhash + worker.prev_lhash;
            worker.lhash = XXHash64::hash(worker.chunk, worker.pos2, 0);
        }
        goto after_permute;
    }

    // Handle op >= 254 (RC4 key reset)
    if (worker.op >= 254) {
        worker.rc4_fast.set_key(worker.prev_chunk, 256);
        RC4_set_key(&worker.rc4_key, 256, worker.prev_chunk);
    }

    // Main permutation
    wolfPermuteLean(worker.prev_chunk, worker.chunk, worker.op, worker.pos1, worker.pos2);

    // Handle op 0 special case
    if (!worker.op) {
        if ((worker.pos2 - worker.pos1) % 2 == 1) {
            worker.t1 = worker.chunk[worker.pos1];
            worker.t2 = worker.chunk[worker.pos2];
            worker.chunk[worker.pos1] = reverse8(worker.t2);
            worker.chunk[worker.pos2] = reverse8(worker.t1);
            worker.isSame = false;
        }
    }

after_permute:
    uint8_t pushPos1 = lp1;
    uint8_t pushPos2 = lp2;

    if (worker.pos1 == worker.pos2) {
        pushPos1 = static_cast<uint8_t>(-1);
        pushPos2 = static_cast<uint8_t>(-1);
    }

    worker.A = (worker.chunk[worker.pos1] - worker.chunk[worker.pos2]);
    worker.A = (256 + (worker.A % 256)) % 256;

    // Hash probability checks
    if (worker.A < 0x10) {
        worker.prev_lhash = worker.lhash + worker.prev_lhash;
        worker.lhash = XXHash64::hash(worker.chunk, worker.pos2, 0);
    }

    if (worker.A < 0x20) {
        worker.prev_lhash = worker.lhash + worker.prev_lhash;
        worker.lhash = hash_64_fnv1a(worker.chunk, worker.pos2);
    }

    if (worker.A < 0x30) {
        worker.prev_lhash = worker.lhash + worker.prev_lhash;
        HH_ALIGNAS(16)
        const highwayhash::HH_U64 key2[2] = {worker.tries, worker.prev_lhash};
        worker.lhash = highwayhash::SipHash(key2, (char*)worker.chunk, worker.pos2);
    }

    if (worker.A <= 0x40) {
        worker.rc4_fast.apply_keystream_256(worker.chunk);
        worker.isSame = false;

        // Template tracking
        if (255 - pushPos2 < LEAN_MINPREFLEN) pushPos2 = 255;
        if (pushPos1 < LEAN_MINPREFLEN) pushPos1 = 0;
        if (pushPos1 == 255) pushPos1 = 0;

        worker.astroTemplate[worker.templateIdx] = templateMarker{
            (uint8_t)(chunkCount > 1 ? pushPos1 : 0),
            (uint8_t)(chunkCount > 1 ? pushPos2 : 255),
            (uint16_t)0,
            (uint16_t)0,
            (uint16_t)((firstChunk << 7) | chunkCount)
        };

        worker.templateIdx += (worker.tries > 1);
        firstChunk = worker.tries - 1;
        lp1 = 255;
        lp2 = 0;
        chunkCount = 1;
    } else {
        chunkCount++;
    }

    worker.chunk[255] = worker.chunk[255] ^ worker.chunk[worker.pos1] ^ worker.chunk[worker.pos2];

    // Check termination condition
    if (worker.tries > 260 + 16 ||
        (worker.chunk[255] >= 0xf0 && worker.tries > 260)) {
        return false;  // Done
    }

    return true;  // Continue
}

/**
 * wolfFinalize - Finalize wolf computation and calculate data length
 */
inline void wolfFinalize(
    WorkerLean& worker,
    uint8_t lp1, uint8_t lp2,
    uint8_t chunkCount, int firstChunk
) {
    if (chunkCount > 0) {
        if (255 - lp2 < LEAN_MINPREFLEN) lp2 = 255;
        if (lp1 < LEAN_MINPREFLEN) lp1 = 0;

        worker.astroTemplate[worker.templateIdx] = templateMarker{
            (uint8_t)(chunkCount > 1 ? lp1 : 0),
            (uint8_t)(chunkCount > 1 ? lp2 : 255),
            (uint16_t)0,
            (uint16_t)0,
            (uint16_t)((firstChunk << 7) | chunkCount)
        };
        worker.templateIdx++;
    }

    worker.data_len = static_cast<uint32_t>(
        (worker.tries - 4) * 256 +
        (((static_cast<uint64_t>(worker.chunk[253]) << 8) |
          static_cast<uint64_t>(worker.chunk[254])) & 0x3ff)
    );
}

/**
 * finalPhase - Compute SA and final hash
 */
inline void finalPhase(
    WorkerLean& worker,
    ThreadSharedLean& shared,
    int32_t* sa_out,
    uint8_t* hash_out
) {
    // Compute suffix array
    shared.compute_sa(worker.sData, worker.data_len, sa_out);

    // Hash the SA output
    hashSHA256(worker.sha256, reinterpret_cast<uint8_t*>(sa_out),
               hash_out, worker.data_len * 4);

    shared.hashes_computed++;
}

// ============================================================================
// Interleaved Execution
// ============================================================================

/**
 * processInterleaved - Compute two hashes with interleaved execution
 *
 * This is the DeroLuna-style optimization: When worker A issues a memory
 * request, work on worker B to hide latency.
 */
inline int processInterleaved(
    const uint8_t* input_a, size_t len_a, uint8_t* hash_a,
    const uint8_t* input_b, size_t len_b, uint8_t* hash_b,
    LeanInterleavedThread& thread
) {
    WorkerLean& wa = thread.worker_a;
    WorkerLean& wb = thread.worker_b;

    // Prep both workers
    prepPhase(wa, input_a, len_a);
    prepPhase(wb, input_b, len_b);

    // State for worker A
    uint8_t chunkCount_a = 1;
    int firstChunk_a = 0;
    uint8_t lp1_a = 0, lp2_a = 255;
    bool done_a = false;

    // State for worker B
    uint8_t chunkCount_b = 1;
    int firstChunk_b = 0;
    uint8_t lp1_b = 0, lp2_b = 255;
    bool done_b = false;

    // Interleaved execution loop
    const int max_iterations = 278;

    for (int it = 0; it < max_iterations && (!done_a || !done_b); ++it) {
        // Process worker A
        if (!done_a) {
            // Prefetch B's data while working on A
            if (!done_b && wb.tries > 0) {
                uint8_t* next_b = &wb.sData[wb.tries * 256];
                #if defined(__x86_64__) || defined(_M_X64)
                _mm_prefetch(reinterpret_cast<const char*>(next_b), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(next_b + 64), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(next_b + 128), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(next_b + 192), _MM_HINT_T0);
                #endif
            }

            if (!wolfIterationLean(wa, lp1_a, lp2_a, chunkCount_a, firstChunk_a)) {
                done_a = true;
                wolfFinalize(wa, lp1_a, lp2_a, chunkCount_a, firstChunk_a);
            }
        }

        // Process worker B
        if (!done_b) {
            // Prefetch A's data while working on B
            if (!done_a && wa.tries > 0) {
                uint8_t* next_a = &wa.sData[wa.tries * 256];
                #if defined(__x86_64__) || defined(_M_X64)
                _mm_prefetch(reinterpret_cast<const char*>(next_a), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(next_a + 64), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(next_a + 128), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(next_a + 192), _MM_HINT_T0);
                #endif
            }

            if (!wolfIterationLean(wb, lp1_b, lp2_b, chunkCount_b, firstChunk_b)) {
                done_b = true;
                wolfFinalize(wb, lp1_b, lp2_b, chunkCount_b, firstChunk_b);
            }
        }
    }

    // Handle incomplete
    if (!done_a) wolfFinalize(wa, lp1_a, lp2_a, chunkCount_a, firstChunk_a);
    if (!done_b) wolfFinalize(wb, lp1_b, lp2_b, chunkCount_b, firstChunk_b);

    // Final phase for both workers
    finalPhase(wa, thread.shared, thread.shared.sa_a, hash_a);
    finalPhase(wb, thread.shared, thread.shared.sa_b, hash_b);

    return 2;  // Two hashes computed
}

/**
 * AstroBWTv3_lean - Single hash computation using lean architecture
 */
inline void AstroBWTv3_lean(
    const uint8_t* input, size_t len,
    uint8_t* output,
    LeanInterleavedThread& thread,
    int worker_idx
) {
    WorkerLean& worker = (worker_idx == 0) ? thread.worker_a : thread.worker_b;
    int32_t* sa = (worker_idx == 0) ? thread.shared.sa_a : thread.shared.sa_b;

    prepPhase(worker, input, len);

    uint8_t chunkCount = 1;
    int firstChunk = 0;
    uint8_t lp1 = 0, lp2 = 255;

    while (wolfIterationLean(worker, lp1, lp2, chunkCount, firstChunk)) {
        // Continue
    }
    wolfFinalize(worker, lp1, lp2, chunkCount, firstChunk);

    finalPhase(worker, thread.shared, sa, output);
}

/**
 * Print memory usage summary
 */
inline void print_memory_usage(int num_threads) {
    printf("\n=== Lean Interleaved Miner Memory Usage ===\n");
    printf("WorkerLean size:           %6zu KB\n", WORKER_LEAN_SIZE / 1024);
    printf("ThreadSharedLean size:     %6zu KB\n", THREAD_SHARED_LEAN_SIZE / 1024);
    printf("LeanInterleavedThread:     %6zu KB\n", LEAN_THREAD_SIZE / 1024);
    printf("\nFor %d threads (2 workers each):\n", num_threads);
    printf("  Total workers: %d\n", num_threads * 2);
    printf("  Total memory:  %zu MB\n", (num_threads * LEAN_THREAD_SIZE) / (1024 * 1024));
    printf("\nComparison:\n");
    printf("  DeroLuna:     10.5 MB constant\n");
    printf("  Original:    ~%zu MB\n", (num_threads * 5500) / 1024);
    printf("  Lean V2:      %zu MB\n", (num_threads * LEAN_THREAD_SIZE) / (1024 * 1024));
    printf("============================================\n\n");
}

} // namespace lean_interleaved

#endif // LEAN_INTERLEAVED_MINER_HPP
