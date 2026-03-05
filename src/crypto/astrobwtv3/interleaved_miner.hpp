/**
 * Two Miners Per Thread - Interleaved Execution
 *
 * DeroLuna-style optimization: Run two hash computations within a single thread,
 * interleaving their wolfCompute iterations to hide L3 cache latency.
 *
 * When hash A issues a memory request, instead of stalling, the CPU works on hash B.
 * This provides Instruction Level Parallelism (ILP) within a single thread.
 */

#ifndef INTERLEAVED_MINER_HPP
#define INTERLEAVED_MINER_HPP

#include "astrobwtv3.h"
#include "lookupcompute.h"
#include <cstdint>
#include <cstring>

// Forward declarations
struct workerData;

/**
 * InterleavedMiner - Two miners per thread for ILP
 *
 * This class manages two workerData contexts and interleaves their
 * wolfCompute loop iterations to hide memory latency.
 */
class InterleavedMiner {
public:
    InterleavedMiner();
    ~InterleavedMiner();

    // Initialize both worker contexts
    bool initialize();

    // Process two hashes with interleaved execution
    // Returns number of hashes computed (always 2 on success)
    int processInterleaved(
        const uint8_t* input_a, int len_a,
        const uint8_t* input_b, int len_b,
        uint8_t* hash_a, uint8_t* hash_b,
        bool useLookup = false
    );

    // Get worker contexts for external use
    workerData* getWorkerA() { return worker_a_; }
    workerData* getWorkerB() { return worker_b_; }

private:
    workerData* worker_a_;
    workerData* worker_b_;
    bool initialized_;

    // Internal prep phase (SHA256 -> Salsa20 -> RC4)
    void prepPhase(workerData& worker, const uint8_t* input, int len);

    // Interleaved wolfCompute - the key optimization
    void wolfComputeInterleaved();

    // Final hash phase
    void finalHashPhase(workerData& worker, uint8_t* output);
};

/**
 * wolfComputeInterleaved2 - Interleaved iteration processing
 *
 * Process two workers with interleaved iterations:
 * - Iteration N of worker A
 * - Iteration N of worker B
 * - Memory requests from A are in flight while B executes
 */
void wolfComputeInterleaved2(workerData& worker_a, workerData& worker_b, int wIndex = 0);

/**
 * Process a single iteration of wolfCompute for one worker.
 * Returns true if the worker should continue, false if done.
 */
bool wolfComputeSingleIteration(workerData& worker, int wIndex, int iteration,
                                 uint8_t& lp1, uint8_t& lp2,
                                 uint8_t& chunkCount, int& firstChunk);

/**
 * Finalize wolfCompute after all iterations
 */
void wolfComputeFinalize(workerData& worker, int wIndex,
                          uint8_t lp1, uint8_t lp2,
                          uint8_t chunkCount, int firstChunk);

/**
 * Benchmark interleaved vs standard hashing
 * Returns improvement percentage (positive = interleaved is faster)
 */
double benchmarkInterleaved(int numIterations = 100);

#endif // INTERLEAVED_MINER_HPP
