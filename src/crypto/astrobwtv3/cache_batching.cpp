/**
 * Cache-Focused Batching for AstroBWTv3 Suffix Array Construction
 *
 * Implementation of k1's insight:
 * - "do staged batches in the context of one thread"
 * - "i want the suffix array construction code to be 100% in L1"
 * - "pin 1 thread : 1 core for num cores"
 *
 * This implementation separates the AstroBWTv3 pipeline into three stages:
 * 1. Prep Stage: SHA256 -> Salsa20 -> RC4 -> wolfCompute (generates scratch data)
 * 2. SA Stage: divsufsort back-to-back for all buffers (keeps code hot in L1I)
 * 3. Hash Stage: Final SHA256 of all suffix arrays
 */

#include "cache_batching.hpp"
#include "dirtybird-hugepages.hpp"
#include "lookupcompute.h"
#include <fnv1a.h>
#include <chrono>
#include <random>
#include <iostream>
#include <algorithm>

extern "C" {
  #include "divsufsort.h"
  #ifdef USE_CUSTOM_SA
    #include "custom_sa_70kb.h"
  #endif
}

/* SA_FUNCTION: Dispatch to custom_sa_70kb or divsufsort based on compile-time flag */
#ifdef USE_CUSTOM_SA
  #define SA_FUNCTION custom_sa_70kb
#else
  #define SA_FUNCTION divsufsort
#endif

// Platform-specific aligned memory allocation
#ifdef _WIN32
#include <malloc.h>
static inline void* portable_aligned_alloc(size_t alignment, size_t size) {
    return _aligned_malloc(size, alignment);
}
static inline void portable_aligned_free(void* ptr) {
    _aligned_free(ptr);
}
#else
#include <cstdlib>
static inline void* portable_aligned_alloc(size_t alignment, size_t size) {
    return aligned_alloc(alignment, size);
}
static inline void portable_aligned_free(void* ptr) {
    free(ptr);
}
#endif

// External declarations
extern void (*astroCompFunc)(workerData &worker, bool isTest, int wIndex);

// Forward declaration of hashSHA256 (defined in astrobwtv3.cpp)
extern void hashSHA256(SHA256_CTX &sha256, const byte *input, byte *digest, unsigned long inputSize);

// ============================================================================
// CacheFocusedBatcher Implementation
// ============================================================================

CacheFocusedBatcher::CacheFocusedBatcher(int batchSize)
    : batchSize_(std::min(std::max(batchSize, CACHE_BATCH_MIN), CACHE_BATCH_MAX))
    , currentSize_(0)
{
    // Allocate scratch buffers with huge pages for better TLB performance
    size_t totalSize = sizeof(ScratchBuffer) * batchSize_;
    buffers_ = static_cast<ScratchBuffer*>(malloc_huge_pages(totalSize));
    if (!buffers_) {
        // Fallback to regular allocation
        buffers_ = static_cast<ScratchBuffer*>(portable_aligned_alloc(64, totalSize));
    }

    // Initialize validity flags
    for (int i = 0; i < batchSize_; ++i) {
        buffers_[i].valid = false;
    }

    // Allocate shared SA buckets (these are large and can be shared across batch)
    bA_ = static_cast<int32_t*>(malloc_huge_pages(256 * sizeof(int32_t)));
    if (!bA_) {
        bA_ = static_cast<int32_t*>(portable_aligned_alloc(64, 256 * sizeof(int32_t)));
    }

    bB_ = static_cast<int32_t*>(malloc_huge_pages(256 * 256 * sizeof(int32_t)));
    if (!bB_) {
        bB_ = static_cast<int32_t*>(portable_aligned_alloc(64, 256 * 256 * sizeof(int32_t)));
    }
}

CacheFocusedBatcher::~CacheFocusedBatcher() {
    if (buffers_) {
        free_huge_pages(buffers_);
    }
    if (bA_) {
        free_huge_pages(bA_);
    }
    if (bB_) {
        free_huge_pages(bB_);
    }
}

bool CacheFocusedBatcher::addInput(const uint8_t* input, int inputLen) {
    if (currentSize_ >= batchSize_) {
        return true; // Batch is already full
    }

    // Copy input to the next available slot
    ScratchBuffer& buf = buffers_[currentSize_];
    buf.inputLen = std::min(inputLen, 48); // MINIBLOCK_SIZE
    std::memcpy(buf.input, input, buf.inputLen);
    buf.valid = true;
    currentSize_++;

    return currentSize_ >= batchSize_; // Return true if now full
}

void CacheFocusedBatcher::clear() {
    for (int i = 0; i < currentSize_; ++i) {
        buffers_[i].valid = false;
    }
    currentSize_ = 0;
}

int CacheFocusedBatcher::processBatch(workerData& worker, uint8_t* outputHashes) {
    if (currentSize_ == 0) {
        return 0;
    }

    int count = currentSize_;

    // =======================================================================
    // STAGE 1: Prepare all scratch buffers
    // This runs SHA256 + Salsa20 + RC4 + wolfCompute for each input
    // The branching/permutation code has different code paths per hash,
    // so we don't expect as much cache benefit here.
    // =======================================================================
    stage1_PrepareAllBuffers(worker);

    // =======================================================================
    // STAGE 2: Run ALL SA constructions back-to-back
    // THIS IS THE KEY INSIGHT: By running divsufsort N times in a row,
    // the instruction cache stays hot with the SA construction code.
    // L1I cache is typically 32KB, divsufsort code is ~15-20KB.
    // =======================================================================
    stage2_AllSAConstructions(worker);

    // =======================================================================
    // STAGE 3: Compute all final SHA256 hashes
    // The SHA256 code is also kept hot by running all hashes back-to-back.
    // =======================================================================
    stage3_AllFinalHashes(worker);

    // Copy results to output
    for (int i = 0; i < count; ++i) {
        std::memcpy(outputHashes + i * 32, buffers_[i].hash, 32);
    }

    // Clear the batch for reuse
    clear();

    return count;
}

// ============================================================================
// Stage 1: Prepare All Buffers
// ============================================================================

void CacheFocusedBatcher::stage1_PrepareAllBuffers(workerData& worker) {
    for (int i = 0; i < currentSize_; ++i) {
        prepareBuffer(i, worker);
    }
}

void CacheFocusedBatcher::prepareBuffer(int idx, workerData& worker) {
    ScratchBuffer& buf = buffers_[idx];

    // Step 1: SHA256 of input -> key for Salsa20
    uint8_t scratch[384] = {0};
    hashSHA256(worker.sha256, buf.input, &scratch[320], buf.inputLen);

    // Step 2: Initialize Salsa20 with SHA256 output
    worker.salsa20.setKey(&scratch[320]);
    worker.salsa20.setIv(&scratch[256]); // IV is zeros

    // Step 3: Salsa20 keystream -> scratch buffer
    worker.salsa20.processBytes(worker.salsaInput, scratch, 256);

    // Step 4: RC4 encryption of scratch buffer
#if USE_FAST_RC4
    // Sync to OpenSSL key for SPSA compatibility
    rc4_avx512::fast_rc4_set_key_dual(worker.fast_rc4_key[0], &worker.key[0], 256, scratch);
    rc4_avx512::fast_rc4_dual(worker.fast_rc4_key[0], &worker.key[0], 256, scratch, scratch);
#else
    RC4_set_key(&worker.key[0], 256, scratch);
    RC4(&worker.key[0], 256, scratch, scratch);
#endif

    // Step 5: Initialize worker state for branching computation
    worker.lhash = hash_64_fnv1a_256_optimized(scratch);
    worker.prev_lhash = worker.lhash;
    worker.tries[0] = 0;
    worker.isSame = false;

    // Copy initial data to worker's sData (for wolfCompute)
    std::memcpy(worker.sData, scratch, 256);

    // Step 6: Run branching/permutation computation (wolfCompute)
    // This generates the full scratch data buffer (~70KB)
    astroCompFunc(worker, false, 0);

    // Step 7: Copy result to batch buffer
    // worker.data_len is set by wolfCompute
    buf.data_len = worker.data_len;
    std::memcpy(buf.sData, worker.sData, buf.data_len);
}

// ============================================================================
// Stage 2: All SA Constructions (Cache-Hot)
// ============================================================================

void CacheFocusedBatcher::stage2_AllSAConstructions(workerData& worker) {
    // The key insight: by running divsufsort N times in a row,
    // the instruction cache stays hot with the SA construction code.
    // L1I cache is typically 32KB, divsufsort code is ~15-20KB.
    for (int i = 0; i < currentSize_; ++i) {
        constructSA(i, worker);
    }
}

void CacheFocusedBatcher::constructSA(int idx, workerData& worker) {
    ScratchBuffer& buf = buffers_[idx];

    // Run suffix array construction
    // Using shared buckets (bA_, bB_) to minimize memory footprint
    SA_FUNCTION(buf.sData, buf.sa, buf.data_len, bA_, bB_);
}

// ============================================================================
// Stage 3: All Final Hashes
// ============================================================================

void CacheFocusedBatcher::stage3_AllFinalHashes(workerData& worker) {
    for (int i = 0; i < currentSize_; ++i) {
        computeFinalHash(i, worker);
    }
}

void CacheFocusedBatcher::computeFinalHash(int idx, workerData& worker) {
    ScratchBuffer& buf = buffers_[idx];

    // SHA256 of the suffix array (interpreted as bytes)
    byte* B = reinterpret_cast<byte*>(buf.sa);
    hashSHA256(worker.sha256, B, buf.hash, buf.data_len * 4);
}

// ============================================================================
// Cache-Batched Mining Function
// ============================================================================

void mineDero_CacheBatched(int tid) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 255);

    // Allocate worker data with NUMA awareness
    workerData* worker = static_cast<workerData*>(malloc_huge_pages(sizeof(workerData)));
    if (!worker) {
        worker = static_cast<workerData*>(std::malloc(sizeof(workerData)));
    }
    if (!worker) {
        std::cerr << "Failed to allocate workerData for cache-batched miner" << std::endl;
        return;
    }

    initWorker(*worker);
    lookupGen(*worker, nullptr, nullptr);

    // Create cache-focused batcher
    CacheFocusedBatcher batcher(CACHE_BATCH_DEFAULT);

    // Batch input/output buffers
    uint8_t inputBatch[CACHE_BATCH_MAX * 48];  // MINIBLOCK_SIZE each
    uint8_t outputHashes[CACHE_BATCH_MAX * 32];

    byte random_tail[12];
    for (int i = 0; i < 12; ++i) {
        random_tail[i] = static_cast<byte>(dist(gen));
    }

    uint64_t localCount = 0;
    uint32_t nonce = 0;

    // This is a simplified mining loop demonstrating cache-batched processing
    // In production, integrate with the job system from mine_dero.cpp
    while (true) {
        // Fill a batch with inputs
        for (int i = 0; i < batcher.getMaxBatchSize(); ++i) {
            // Generate input for this hash
            byte* input = inputBatch + i * 48;

            // In production: copy from actual job blockhashing_blob
            // Here we generate synthetic input for demonstration
            std::memset(input, 0, 48);
            std::memcpy(input + 36, random_tail, 12);

            // Increment nonce
            nonce++;
            input[43] = static_cast<byte>((nonce >> 24) & 0xFF);
            input[44] = static_cast<byte>((nonce >> 16) & 0xFF);
            input[45] = static_cast<byte>((nonce >> 8) & 0xFF);
            input[46] = static_cast<byte>(nonce & 0xFF);
            input[47] = static_cast<byte>(tid);

            batcher.addInput(input, 48);
        }

        // Process the batch with cache-focused staging
        int hashesComputed = batcher.processBatch(*worker, outputHashes);
        localCount += hashesComputed;

        // In production: check each hash against difficulty target
        // For now, just count hashes
        if (localCount >= 512) {
            // Report hashrate (integrate with counter from mine_dero.cpp)
            localCount = 0;
        }
    }

    free_huge_pages(worker);
}

// ============================================================================
// Benchmark Function
// ============================================================================

int benchmarkCacheBatching(int numIterations) {
    std::cout << "=== Cache Batching Benchmark ===" << std::endl;
    std::cout << "Iterations per batch size: " << numIterations << std::endl;

    // Allocate worker
    workerData* worker = static_cast<workerData*>(malloc_huge_pages(sizeof(workerData)));
    if (!worker) {
        worker = static_cast<workerData*>(std::malloc(sizeof(workerData)));
    }
    if (!worker) {
        std::cerr << "Failed to allocate workerData" << std::endl;
        return CACHE_BATCH_DEFAULT;
    }

    initWorker(*worker);
    lookupGen(*worker, nullptr, nullptr);

    // Generate random inputs
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> dist(0, 255);

    uint8_t inputs[CACHE_BATCH_MAX * 48];
    for (int i = 0; i < CACHE_BATCH_MAX * 48; ++i) {
        inputs[i] = dist(gen);
    }

    uint8_t outputs[CACHE_BATCH_MAX * 32];

    double bestRate = 0.0;
    int bestBatchSize = CACHE_BATCH_DEFAULT;

    // Test batch sizes from MIN to MAX
    for (int batchSize = CACHE_BATCH_MIN; batchSize <= CACHE_BATCH_MAX; batchSize *= 2) {
        CacheFocusedBatcher batcher(batchSize);

        auto start = std::chrono::high_resolution_clock::now();

        int totalHashes = 0;
        for (int iter = 0; iter < numIterations; ++iter) {
            // Fill batch
            for (int i = 0; i < batchSize; ++i) {
                batcher.addInput(inputs + i * 48, 48);
            }

            // Process batch
            totalHashes += batcher.processBatch(*worker, outputs);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double seconds = std::chrono::duration<double>(end - start).count();
        double rate = totalHashes / seconds;

        std::cout << "Batch size " << batchSize << ": "
                  << rate << " H/s (" << totalHashes << " hashes in "
                  << seconds << "s)" << std::endl;

        if (rate > bestRate) {
            bestRate = rate;
            bestBatchSize = batchSize;
        }
    }

    // Also benchmark standard sequential processing for comparison
    std::cout << "\nSequential (no batching) comparison:" << std::endl;
    {
        auto start = std::chrono::high_resolution_clock::now();

        int totalHashes = numIterations * CACHE_BATCH_DEFAULT;
        for (int i = 0; i < totalHashes; ++i) {
            AstroBWTv3(inputs + (i % CACHE_BATCH_MAX) * 48, 48, outputs, *worker, false);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double seconds = std::chrono::duration<double>(end - start).count();
        double rate = totalHashes / seconds;

        std::cout << "Sequential: " << rate << " H/s (" << totalHashes
                  << " hashes in " << seconds << "s)" << std::endl;

        // If sequential is faster, report that
        if (rate > bestRate) {
            std::cout << "\nNote: Sequential processing was faster on this CPU." << std::endl;
            std::cout << "This may indicate cache-batching is not beneficial here." << std::endl;
            bestBatchSize = 1; // Signal to use standard processing
        }
    }

    std::cout << "\nOptimal batch size: " << bestBatchSize << std::endl;
    std::cout << "Best rate: " << bestRate << " H/s" << std::endl;

    free_huge_pages(worker);

    return bestBatchSize;
}
