/**
 * Two Miners Per Thread - Interleaved Execution Implementation
 *
 * Key insight from DeroLuna: When one hash is waiting for memory,
 * work on the other hash. This hides L3 cache latency through ILP.
 */

#include "interleaved_miner.hpp"
#include "dirtybird-hugepages.hpp"
#include "astrobwtv3.h"
#include "lookupcompute.h"
#include "rc4_avx512.hpp"

// MINPREFLEN definition (matches astrobwtv3.cpp)
#ifndef MINPREFLEN
#define MINPREFLEN 4
#endif

/* SA_FUNCTION: Dispatch to best available suffix array implementation */
extern "C" {
  #include "divsufsort.h"
  #ifdef USE_CUSTOM_SA
    #include "custom_sa_70kb.h"
  #endif
  #ifdef USE_LIBSAIS
    #include "libsais.h"
  #endif
}

#ifdef USE_DLUNA_RADIX_SA
  #include "dluna_radix_sa.h"
  #define SA_FUNCTION dluna_radix_sa::radix_sort_sa
#elif defined(USE_BUCKET_SA)
  #include "bucket_sa.h"
  #define SA_FUNCTION bucket_sa::bucket_sort_sa
#elif defined(USE_RADIX_SA)
  #include "radix_sa.h"
  #define SA_FUNCTION radix_sa::radix_sort_sa
#elif defined(USE_CUSTOM_SA)
  #define SA_FUNCTION custom_sa_70kb
#elif defined(USE_LIBSAIS)
  static inline int32_t libsais_wrapper_im(const uint8_t* T, int32_t* SA, int32_t n,
                                           int32_t* /*bucket_A*/, int32_t* /*bucket_B*/) {
    static thread_local void* tl_sais_ctx = nullptr;
    if (tl_sais_ctx == nullptr) {
      tl_sais_ctx = libsais_create_ctx();
    }
    return libsais_ctx(tl_sais_ctx, T, SA, n, 0, nullptr);
  }
  #define SA_FUNCTION libsais_wrapper_im
#else
  #define SA_FUNCTION divsufsort
#endif

#include <fnv1a.h>
#include <xxhash64.h>
#include <highwayhash/sip_hash.h>
// Salsa20 included via astrobwtv3.h -> astroworker.h (salsa20_simd.h)
#include <openssl/sha.h>
#include <openssl/rc4.h>

#include <algorithm>
#include <cstring>
#include <bit>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

extern "C" {
#include "divsufsort.h"
}

// External function pointers and globals
extern void (*astroCompFunc)(workerData &worker, bool isTest, int wIndex);
extern bool g_use_spsa;

// Forward declarations from astrobwtv3.cpp
extern void hashSHA256(SHA256_CTX &sha256, const byte *input, byte *digest, unsigned long inputSize);
extern void copyChunkData(workerData &worker, int pos1, int pos2);

// Note: wolfPermute and wolfPermute_avx512 are declared in astrobwtv3.h with uint16_t op
// Note: rl8 is defined as a macro in astrobwtv3.h
// Note: reverse8 is defined as inline function in astrobwtv3.h

#if defined(USE_ASTRO_SPSA)
#include <spsa.hpp>
#endif

// ============================================================================
// InterleavedMiner Implementation
// ============================================================================

InterleavedMiner::InterleavedMiner()
    : worker_a_(nullptr)
    , worker_b_(nullptr)
    , initialized_(false)
{
}

InterleavedMiner::~InterleavedMiner() {
    if (worker_a_) {
        free_huge_pages(worker_a_);
    }
    if (worker_b_) {
        free_huge_pages(worker_b_);
    }
}

bool InterleavedMiner::initialize() {
    if (initialized_) return true;

    // Allocate two worker contexts with huge pages
    worker_a_ = static_cast<workerData*>(malloc_huge_pages(sizeof(workerData)));
    if (!worker_a_) {
        worker_a_ = static_cast<workerData*>(std::malloc(sizeof(workerData)));
    }

    worker_b_ = static_cast<workerData*>(malloc_huge_pages(sizeof(workerData)));
    if (!worker_b_) {
        worker_b_ = static_cast<workerData*>(std::malloc(sizeof(workerData)));
    }

    if (!worker_a_ || !worker_b_) {
        return false;
    }

    // Initialize both workers
    initWorker(*worker_a_);
    initWorker(*worker_b_);
    lookupGen(*worker_a_, nullptr, nullptr);
    lookupGen(*worker_b_, nullptr, nullptr);

    initialized_ = true;
    return true;
}

void InterleavedMiner::prepPhase(workerData& worker, const uint8_t* input, int len) {
    // Step 1: SHA256 of input
    uint8_t scratch[384] = {0};
    hashSHA256(worker.sha256, input, &scratch[320], len);

    // Step 2: Salsa20 expansion
#if USE_SIMD_SALSA20
    // SIMD Salsa20 - AVX2 optimized (~2-5% faster)
    salsa20_simd_process(&scratch[320], &scratch[256], worker.salsaInput, scratch, 256);
#else
    worker.salsa20.setKey(&scratch[320]);
    worker.salsa20.setIv(&scratch[256]); // IV is zeros
    worker.salsa20.processBytes(worker.salsaInput, scratch, 256);
#endif

    // Step 3: RC4 encryption - use fast RC4 when available
#if USE_CRYPTOGAMS_RC4_DUAL
    // Dual-state: Initialize both CRYPTOGAMS (fast) and OpenSSL (SPSA compat)
    worker.cryptogams_rc4[0].set_key(scratch, 256);
    RC4_set_key(&worker.key[0], 256, scratch);
    worker.cryptogams_rc4[0].apply_keystream_256(scratch);
#elif USE_FAST_RC4
    rc4_avx512::fast_rc4_set_key_dual(worker.fast_rc4_key[0], &worker.key[0], 256, scratch);
    rc4_avx512::fast_rc4_dual(worker.fast_rc4_key[0], &worker.key[0], 256, scratch, scratch);
#else
    RC4_set_key(&worker.key[0], 256, scratch);
    RC4(&worker.key[0], 256, scratch, scratch);
#endif

    // Step 4: Initialize worker state (match AstroBWTv3 exactly)
    worker.lhash = hash_64_fnv1a_256_optimized(scratch);
    worker.prev_lhash = worker.lhash;
    worker.tries[0] = 0;
    worker.isSame = false;
    worker.pos1 = 0;
    worker.pos2 = 255;
    worker.op = 0;
    worker.templateIdx = 0;
    worker.data_len = 0;

    // Copy initial data to worker's sData
    std::memcpy(worker.sData, scratch, 256);

    // Initialize chunk pointers for first iteration
    worker.chunk = worker.sData;
    worker.prev_chunk = worker.sData;
}

void InterleavedMiner::finalHashPhase(workerData& worker, uint8_t* output) {
    // Run suffix array construction with SPSA support
    #if defined(USE_ASTRO_SPSA)
      bool alreadySha = g_use_spsa && SPSA(worker.sData, worker.data_len, worker);
      if (alreadySha) {
        memcpy(output, worker.padding, 32);
      } else {
        int32_t *bucketA = nullptr;
        int32_t *bucketB = nullptr;
        getSABucketScratch(worker, bucketA, bucketB);
        SA_FUNCTION(worker.sData, worker.sa, worker.data_len, bucketA, bucketB);
        byte* B = reinterpret_cast<byte*>(worker.sa);
        hashSHA256(worker.sha256, B, output, worker.data_len * 4);
      }
    #else
      int32_t *bucketA = nullptr;
      int32_t *bucketB = nullptr;
      getSABucketScratch(worker, bucketA, bucketB);
      SA_FUNCTION(worker.sData, worker.sa, worker.data_len, bucketA, bucketB);
      byte* B = reinterpret_cast<byte*>(worker.sa);
      hashSHA256(worker.sha256, B, output, worker.data_len * 4);
    #endif
}

int InterleavedMiner::processInterleaved(
    const uint8_t* input_a, int len_a,
    const uint8_t* input_b, int len_b,
    uint8_t* hash_a, uint8_t* hash_b,
    bool useLookup)
{
    if (!initialized_) {
        if (!initialize()) return 0;
    }

    // =========================================================================
    // TWO WORKERS PER THREAD - Memory bandwidth optimization
    //
    // Run two independent hash computations per thread.
    // While one worker waits for memory, the CPU can work on memory
    // requests from the other worker (out-of-order execution benefits).
    //
    // Note: True interleaved execution (wolfComputeInterleaved2) is available
    // but has stability issues. Using sequential execution for now.
    // =========================================================================

    // Process both workers - let OoO execution provide ILP
    AstroBWTv3(const_cast<byte*>(input_a), len_a, hash_a, *worker_a_, false);
    AstroBWTv3(const_cast<byte*>(input_b), len_b, hash_b, *worker_b_, false);

    return 2;
}

// ============================================================================
// Interleaved wolfCompute - The Core Innovation
// ============================================================================

/**
 * Process a single iteration of wolfCompute.
 * Extracted to enable interleaving between two workers.
 *
 * Returns true if worker should continue, false if done.
 */
bool wolfComputeSingleIteration(workerData& worker, int wIndex, int iteration,
                                 uint8_t& lp1, uint8_t& lp2,
                                 uint8_t& chunkCount, int& firstChunk)
{
    worker.tries[wIndex]++;
    worker.random_switcher = worker.prev_lhash ^ worker.lhash ^ worker.tries[wIndex];

    byte prevOp = worker.op;
    worker.op = static_cast<byte>(worker.random_switcher);

    byte p1 = static_cast<byte>(worker.random_switcher >> 8);
    byte p2 = static_cast<byte>(worker.random_switcher >> 16);

    if (p1 > p2) {
        std::swap(p1, p2);
    }

    if (p2 - p1 > 32) {
        p2 = p1 + ((p2 - p1) & 0x1f);
    }

    if (worker.tries[wIndex] > 0) {
        lp1 = std::min(lp1, p1);
        lp2 = std::max(lp2, p2);
    }

    if (p1 < worker.pos1 || p2 > worker.pos2) {
        worker.isSame = false;
    }

    worker.pos1 = p1;
    worker.pos2 = p2;

    worker.chunk = &worker.sData[wIndex * ASTRO_SCRATCH_SIZE + (worker.tries[wIndex] - 1) * 256];

    if (worker.tries[wIndex] == 1) {
        worker.prev_chunk = worker.chunk;
    } else {
        worker.prev_chunk = &worker.sData[wIndex * ASTRO_SCRATCH_SIZE + (worker.tries[wIndex] - 2) * 256];
        // AVX2-optimized 256-byte copy (8x 32-byte stores)
        // NOTE: Only enable when __AVX2__ is defined (not just any x64)
        #if defined(__AVX2__)
        _mm256_storeu_si256((__m256i*)&worker.chunk[0], _mm256_loadu_si256((__m256i*)&worker.prev_chunk[0]));
        _mm256_storeu_si256((__m256i*)&worker.chunk[32], _mm256_loadu_si256((__m256i*)&worker.prev_chunk[32]));
        _mm256_storeu_si256((__m256i*)&worker.chunk[64], _mm256_loadu_si256((__m256i*)&worker.prev_chunk[64]));
        _mm256_storeu_si256((__m256i*)&worker.chunk[96], _mm256_loadu_si256((__m256i*)&worker.prev_chunk[96]));
        _mm256_storeu_si256((__m256i*)&worker.chunk[128], _mm256_loadu_si256((__m256i*)&worker.prev_chunk[128]));
        _mm256_storeu_si256((__m256i*)&worker.chunk[160], _mm256_loadu_si256((__m256i*)&worker.prev_chunk[160]));
        _mm256_storeu_si256((__m256i*)&worker.chunk[192], _mm256_loadu_si256((__m256i*)&worker.prev_chunk[192]));
        _mm256_storeu_si256((__m256i*)&worker.chunk[224], _mm256_loadu_si256((__m256i*)&worker.prev_chunk[224]));
        #else
        memcpy(worker.chunk, worker.prev_chunk, 256);
        #endif
    }

    // Prefetch data into L1 cache before wolfPermute (matches original wolfCompute)
    #if defined(__x86_64__) || defined(_M_X64)
    _mm_prefetch(reinterpret_cast<const char*>(&worker.prev_chunk[worker.pos1]), _MM_HINT_T0);
    if (worker.pos1 <= 224) {  // Bounds check to avoid reading beyond 256 bytes
        _mm_prefetch(reinterpret_cast<const char*>(&worker.prev_chunk[worker.pos1 + 32]), _MM_HINT_T0);
    }
    #endif

    // Handle op 253 special case
    // Use safe scalar copy to avoid buffer overflow when pos1 > 224
    if (worker.op == 253) {
        // Safe copy: only copy bytes within [pos1, pos2)
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
#if USE_CRYPTOGAMS_RC4_DUAL
        worker.cryptogams_rc4[wIndex].set_key(worker.prev_chunk, 256);
        RC4_set_key(&worker.key[wIndex], 256, worker.prev_chunk);  // Keep for SPSA S-box access
#elif USE_FAST_RC4
        rc4_avx512::fast_rc4_set_key_dual(worker.fast_rc4_key[wIndex], &worker.key[wIndex], 256, worker.prev_chunk);
#else
        RC4_set_key(&worker.key[wIndex], 256, worker.prev_chunk);
#endif
    }

    // Main permutation - FMV resolves best version at program load time
    wolfPermute(worker.prev_chunk, worker.chunk, worker.op, worker.pos1, worker.pos2, worker);

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
    if (worker.A < 0x10) { // 6.25%
        worker.prev_lhash = worker.lhash + worker.prev_lhash;
        worker.lhash = XXHash64::hash(worker.chunk, worker.pos2, 0);
    }

    if (worker.A < 0x20) { // 12.5%
        worker.prev_lhash = worker.lhash + worker.prev_lhash;
        worker.lhash = hash_64_fnv1a(worker.chunk, worker.pos2);
    }

    if (worker.A < 0x30) { // 18.75%
        worker.prev_lhash = worker.lhash + worker.prev_lhash;
        HH_ALIGNAS(16)
        const highwayhash::HH_U64 key2[2] = {worker.tries[wIndex], worker.prev_lhash};
        worker.lhash = highwayhash::SipHash(key2, (char*)worker.chunk, worker.pos2);
    }

    if (worker.A <= 0x40) { // 25%
#if USE_CRYPTOGAMS_RC4_DUAL
        worker.cryptogams_rc4[wIndex].apply_keystream_256(worker.chunk);
#elif USE_FAST_RC4
        rc4_avx512::fast_rc4_dual(worker.fast_rc4_key[wIndex], &worker.key[wIndex], 256, worker.chunk, worker.chunk);
#else
        RC4(&worker.key[wIndex], 256, worker.chunk, worker.chunk);
#endif
        worker.isSame = false;

        // Template tracking
        if (255 - pushPos2 < MINPREFLEN) pushPos2 = 255;
        if (pushPos1 < MINPREFLEN) pushPos1 = 0;
        if (pushPos1 == 255) pushPos1 = 0;

        worker.astroTemplate[worker.templateIdx] = templateMarker{
            (uint8_t)(chunkCount > 1 ? pushPos1 : 0),
            (uint8_t)(chunkCount > 1 ? pushPos2 : 255),
            (uint16_t)0,
            (uint16_t)0,
            (uint16_t)((firstChunk << 7) | chunkCount)
        };

        pushPos1 = 0;
        pushPos2 = 255;
        worker.templateIdx += (worker.tries[wIndex] > 1);
        firstChunk = worker.tries[wIndex] - 1;
        lp1 = 255;
        lp2 = 0;
        chunkCount = 1;
    } else {
        chunkCount++;
    }

    worker.chunk[255] = worker.chunk[255] ^ worker.chunk[worker.pos1] ^ worker.chunk[worker.pos2];

    if (255 - pushPos2 < MINPREFLEN) pushPos2 = 255;
    if (pushPos1 < MINPREFLEN) pushPos1 = 0;

    // Check termination condition
    // NOTE: Use worker.chunk[255] which already points to correct location with wIndex offset
    if (worker.tries[wIndex] > 260 + 16 ||
        (worker.chunk[255] >= 0xf0 && worker.tries[wIndex] > 260)) {
        return false; // Done
    }

    return true; // Continue
}

void wolfComputeFinalize(workerData& worker, int wIndex,
                          uint8_t lp1, uint8_t lp2,
                          uint8_t chunkCount, int firstChunk)
{
    if (chunkCount > 0) {
        if (255 - lp2 < MINPREFLEN) lp2 = 255;
        if (lp1 < MINPREFLEN) lp1 = 0;

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
        (worker.tries[wIndex] - 4) * 256 +
        (((static_cast<uint64_t>(worker.chunk[253]) << 8) |
          static_cast<uint64_t>(worker.chunk[254])) & 0x3ff)
    );
}

/**
 * wolfComputeInterleaved2 - The core innovation
 *
 * Process two workers with interleaved iterations.
 * When worker A is waiting for memory, work on worker B.
 */
void wolfComputeInterleaved2(workerData& worker_a, workerData& worker_b, int wIndex)
{
    // State for worker A
    worker_a.templateIdx = 0;
    uint8_t chunkCount_a = 1;
    int firstChunk_a = 0;
    uint8_t lp1_a = 0;
    uint8_t lp2_a = 255;
    worker_a.tries[wIndex] = 0;
    bool done_a = false;

    // State for worker B
    worker_b.templateIdx = 0;
    uint8_t chunkCount_b = 1;
    int firstChunk_b = 0;
    uint8_t lp1_b = 0;
    uint8_t lp2_b = 255;
    worker_b.tries[wIndex] = 0;
    bool done_b = false;

    // =========================================================================
    // INTERLEAVED EXECUTION LOOP
    // This is the key insight from DeroLuna:
    // - Process one iteration of A
    // - Process one iteration of B (while A's memory requests are in flight)
    // - Repeat until both are done
    // =========================================================================

    int max_iterations = 278;

    for (int it = 0; it < max_iterations && (!done_a || !done_b); ++it) {
        // --- Process worker A's iteration ---
        if (!done_a) {
            // Prefetch B's data while we work on A
            if (!done_b && worker_b.tries[wIndex] > 0) {
                byte* next_chunk_b = &worker_b.sData[wIndex * ASTRO_SCRATCH_SIZE + worker_b.tries[wIndex] * 256];
#if defined(__x86_64__) || defined(_M_X64)
                _mm_prefetch(reinterpret_cast<const char*>(next_chunk_b), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(next_chunk_b + 64), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(next_chunk_b + 128), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(next_chunk_b + 192), _MM_HINT_T0);
#endif
            }

            bool continue_a = wolfComputeSingleIteration(
                worker_a, wIndex, it,
                lp1_a, lp2_a, chunkCount_a, firstChunk_a
            );
            if (!continue_a) {
                done_a = true;
                wolfComputeFinalize(worker_a, wIndex, lp1_a, lp2_a, chunkCount_a, firstChunk_a);
            }
        }

        // --- Process worker B's iteration ---
        if (!done_b) {
            // Prefetch A's data while we work on B
            if (!done_a && worker_a.tries[wIndex] > 0) {
                byte* next_chunk_a = &worker_a.sData[wIndex * ASTRO_SCRATCH_SIZE + worker_a.tries[wIndex] * 256];
#if defined(__x86_64__) || defined(_M_X64)
                _mm_prefetch(reinterpret_cast<const char*>(next_chunk_a), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(next_chunk_a + 64), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(next_chunk_a + 128), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(next_chunk_a + 192), _MM_HINT_T0);
#endif
            }

            bool continue_b = wolfComputeSingleIteration(
                worker_b, wIndex, it,
                lp1_b, lp2_b, chunkCount_b, firstChunk_b
            );
            if (!continue_b) {
                done_b = true;
                wolfComputeFinalize(worker_b, wIndex, lp1_b, lp2_b, chunkCount_b, firstChunk_b);
            }
        }
    }

    // Handle case where loop exits before finalization
    if (!done_a) {
        wolfComputeFinalize(worker_a, wIndex, lp1_a, lp2_a, chunkCount_a, firstChunk_a);
    }
    if (!done_b) {
        wolfComputeFinalize(worker_b, wIndex, lp1_b, lp2_b, chunkCount_b, firstChunk_b);
    }
}

// ============================================================================
// Benchmark Function
// ============================================================================

#include <chrono>
#include <random>
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>

double benchmarkInterleaved(int numIterations) {
    std::cout << "=== Interleaved vs Standard Benchmark ===" << std::endl;
    std::cout << "Iterations: " << numIterations << std::endl;
    std::cout << std::endl;

    // Allocate workers
    workerData* standard_worker = static_cast<workerData*>(malloc_huge_pages(sizeof(workerData)));
    if (!standard_worker) {
        standard_worker = static_cast<workerData*>(std::malloc(sizeof(workerData)));
    }
    if (!standard_worker) {
        std::cerr << "Failed to allocate workerData" << std::endl;
        return 0.0;
    }

    initWorker(*standard_worker);
    lookupGen(*standard_worker, nullptr, nullptr);

    // Create interleaved miner
    InterleavedMiner interleavedMiner;
    if (!interleavedMiner.initialize()) {
        std::cerr << "Failed to initialize interleaved miner" << std::endl;
        free_huge_pages(standard_worker);
        return 0.0;
    }

    // Generate random inputs
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> dist(0, 255);

    // 16 different random inputs for variety
    uint8_t inputs[16 * 48];
    for (int i = 0; i < 16 * 48; ++i) {
        inputs[i] = dist(gen);
    }

    uint8_t output[32];
    uint8_t output_a[32], output_b[32];

    // Warmup
    std::cout << "Warming up..." << std::endl;
    for (int i = 0; i < 20; ++i) {
        AstroBWTv3(inputs + (i % 16) * 48, 48, output, *standard_worker, false);
    }
    for (int i = 0; i < 10; ++i) {
        interleavedMiner.processInterleaved(
            inputs + (i * 2 % 16) * 48, 48,
            inputs + ((i * 2 + 1) % 16) * 48, 48,
            output_a, output_b, false
        );
    }
    std::cout << "Warmup complete." << std::endl << std::endl;

    // Benchmark standard sequential
    std::cout << "Benchmarking standard sequential..." << std::endl;
    int standardHashes = numIterations * 2; // Match interleaved count
    auto start_std = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < standardHashes; ++i) {
        AstroBWTv3(inputs + (i % 16) * 48, 48, output, *standard_worker, false);
    }

    auto end_std = std::chrono::high_resolution_clock::now();
    double seconds_std = std::chrono::duration<double>(end_std - start_std).count();
    double rate_std = standardHashes / seconds_std;

    std::cout << "  Standard: " << rate_std << " H/s (" << standardHashes
              << " hashes in " << seconds_std << "s)" << std::endl;

    // Benchmark interleaved
    std::cout << "Benchmarking interleaved..." << std::endl;
    auto start_int = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numIterations; ++i) {
        interleavedMiner.processInterleaved(
            inputs + ((i * 2) % 16) * 48, 48,
            inputs + ((i * 2 + 1) % 16) * 48, 48,
            output_a, output_b, false
        );
    }

    auto end_int = std::chrono::high_resolution_clock::now();
    double seconds_int = std::chrono::duration<double>(end_int - start_int).count();
    double rate_int = (numIterations * 2) / seconds_int;

    std::cout << "  Interleaved: " << rate_int << " H/s (" << (numIterations * 2)
              << " hashes in " << seconds_int << "s)" << std::endl;

    // Calculate improvement
    double improvement = ((rate_int - rate_std) / rate_std) * 100.0;

    std::cout << std::endl << "=== Results ===" << std::endl;
    std::cout << "  Standard:    " << rate_std << " H/s" << std::endl;
    std::cout << "  Interleaved: " << rate_int << " H/s" << std::endl;
    std::cout << "  Improvement: " << (improvement >= 0 ? "+" : "") << improvement << "%" << std::endl;

    if (improvement > 0) {
        std::cout << "  Recommendation: Use --interleaved for better performance" << std::endl;
    } else {
        std::cout << "  Recommendation: Standard mode is faster on this CPU" << std::endl;
    }

    // Cleanup
    free_huge_pages(standard_worker);

    return improvement;
}
