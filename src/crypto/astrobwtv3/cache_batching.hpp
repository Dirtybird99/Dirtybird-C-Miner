/**
 * Cache-Focused Batching for AstroBWTv3 Suffix Array Construction
 *
 * Based on k1's insight:
 * - "do staged batches in the context of one thread"
 * - "i want the suffix array construction code to be 100% in L1"
 * - "pin 1 thread : 1 core for num cores"
 *
 * Key insight: Keep the SA construction code hot in L1 instruction cache by
 * running multiple SA constructions back-to-back without interleaving other
 * operations that would evict the code from cache.
 *
 * L1 Cache considerations:
 * - Typical L1D: 32KB data cache
 * - Typical L1I: 32KB instruction cache
 * - divsufsort code footprint: ~15-25KB
 * - Goal: Run N SA constructions back-to-back to amortize cache warmup cost
 */

#ifndef CACHE_BATCHING_HPP
#define CACHE_BATCHING_HPP

#include "astroworker.h"
#include "astrobwtv3.h"
#include <cstdint>
#include <cstring>
#include <array>

extern "C" {
    #include "divsufsort.h"
    #include "divsufsort_private.h"
}

/**
 * Optimal batch size analysis:
 *
 * - L1D cache: 32KB, L1I cache: 32KB
 * - sData per hash: ~70KB (variable, 256 * ~270 iterations)
 * - SA buckets (bA): 256 * 4 = 1KB
 * - SA buckets (bB): 256 * 256 * 4 = 256KB (too big for L1, but accessed sequentially)
 * - SA output: ~70KB * 4 = ~280KB per hash
 *
 * The key is keeping the INSTRUCTION CACHE hot, not the data.
 * divsufsort has a working set of ~20KB of code that we want to keep resident.
 *
 * Recommended batch sizes:
 * - 4 hashes: Good balance, ~280KB working memory per batch
 * - 8 hashes: Better instruction cache amortization, ~560KB working memory
 * - 16 hashes: Maximum efficiency, ~1.1MB working memory (L2 bound)
 */
constexpr int CACHE_BATCH_MIN = 4;
constexpr int CACHE_BATCH_DEFAULT = 8;
constexpr int CACHE_BATCH_MAX = 16;

/**
 * Per-hash scratch buffer for staged batch processing.
 * Each ScratchBuffer holds all the intermediate data needed for one hash.
 */
struct alignas(64) ScratchBuffer {
    // Stage 1 outputs: Pre-computed data ready for SA construction
    uint8_t sData[ASTRO_SCRATCH_SIZE];    // Branched/permuted data
    uint32_t data_len;                     // Length of data for this hash

    // Stage 2 outputs: SA construction results
    int32_t sa[277 * 256 + 1];            // Suffix array output

    // Stage 3: Final hash output
    uint8_t hash[32];                      // SHA256 of SA

    // Input data (copied at batch start)
    uint8_t input[48];                     // Original input (MINIBLOCK_SIZE)
    int inputLen;

    // Working state
    bool valid;                            // Whether this slot has valid data
};

/**
 * Cache-Focused Batcher
 *
 * Implements k1's staged batching strategy:
 * 1. Accumulate multiple inputs (avoid context switches)
 * 2. Stage 1: Generate all scratch buffers (SHA256 + Salsa20 + RC4 + branching)
 * 3. Stage 2: Run ALL SA constructions back-to-back (keeps code in L1I)
 * 4. Stage 3: Compute all final SHA256 hashes
 *
 * This approach maximizes instruction cache utilization by ensuring
 * the divsufsort code remains hot throughout the entire batch.
 */
class CacheFocusedBatcher {
public:
    CacheFocusedBatcher(int batchSize = CACHE_BATCH_DEFAULT);
    ~CacheFocusedBatcher();

    // Add an input to the batch, returns true if batch is full
    bool addInput(const uint8_t* input, int inputLen);

    // Process all accumulated inputs (call when batch is full or on demand)
    // Returns number of hashes computed
    int processBatch(workerData& worker, uint8_t* outputHashes);

    // Get current batch fill level
    int getCurrentBatchSize() const { return currentSize_; }
    int getMaxBatchSize() const { return batchSize_; }
    bool isFull() const { return currentSize_ >= batchSize_; }

    // Clear the batch (start fresh)
    void clear();

private:
    // Stage implementations
    void stage1_PrepareAllBuffers(workerData& worker);
    void stage2_AllSAConstructions(workerData& worker);
    void stage3_AllFinalHashes(workerData& worker);

    // Single-hash helpers (called from stage functions)
    void prepareBuffer(int idx, workerData& worker);
    void constructSA(int idx, workerData& worker);
    void computeFinalHash(int idx, workerData& worker);

    ScratchBuffer* buffers_;   // Array of scratch buffers
    int batchSize_;            // Maximum batch size
    int currentSize_;          // Current number of inputs queued

    // Per-worker SA buckets (shared across batch to save memory)
    int32_t* bA_;              // 256 entries
    int32_t* bB_;              // 256*256 entries
};

/**
 * Cache-Batched Mining Function
 *
 * Drop-in replacement for mineDero() that uses cache-focused batching.
 * Designed to be called from the threading code in miner.cpp.
 */
void mineDero_CacheBatched(int tid);

/**
 * Single-threaded benchmark for cache batching
 * Compares:
 * 1. Standard sequential hashing
 * 2. Cache-focused batched hashing (various batch sizes)
 *
 * Returns the optimal batch size for this CPU.
 */
int benchmarkCacheBatching(int numIterations = 100);

/**
 * Memory prefetch hints for optimal cache behavior.
 * Call before processing a batch to warm up cache lines.
 */
inline void prefetchForBatch(const ScratchBuffer* buffers, int count) {
    // Prefetch data that will be needed for SA construction
    for (int i = 0; i < count && i < 4; ++i) {
        __builtin_prefetch(&buffers[i].sData[0], 0, 3);
        __builtin_prefetch(&buffers[i].sa[0], 1, 3);
    }
}

/**
 * Cache line flush for clean state between batches.
 * Optional: helps ensure consistent timing in benchmarks.
 */
inline void flushSACodeFromCache() {
#if defined(__x86_64__)
    // Force a memory fence - actual cache flush would require privileged instructions
    _mm_mfence();
#endif
}

#endif // CACHE_BATCHING_HPP
