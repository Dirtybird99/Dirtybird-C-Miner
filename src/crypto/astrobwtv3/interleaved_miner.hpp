/**
 * Two Miners Per Thread - Interleaved Execution
 *
 * DeroLuna-style optimization: Run two hash computations within a single thread,
 * interleaving their wolfCompute iterations to hide L3 cache latency.
 */

#ifndef INTERLEAVED_MINER_HPP
#define INTERLEAVED_MINER_HPP

#include "astrobwtv3.h"
#include <cstdint>
#include <cstring>

// Forward declarations
struct workerData;

class InterleavedMiner {
public:
    InterleavedMiner();
    ~InterleavedMiner();

    bool initialize();

    int processInterleaved(
        const uint8_t* input_a, int len_a,
        const uint8_t* input_b, int len_b,
        uint8_t* hash_a, uint8_t* hash_b,
        int wIndex = 0
    );

    workerData* getWorkerA() { return worker_a_; }
    workerData* getWorkerB() { return worker_b_; }

    void prepPhaseSingle(workerData& worker, const uint8_t* input, int inputLen);
    void wolfComputeInterleaved2(workerData& wa, workerData& wb, int wi);

private:
    workerData* worker_a_;
    workerData* worker_b_;
    bool initialized_;

    void finalHashPhaseInterleaved(uint8_t* hash_a, uint8_t* hash_b);
};

bool wolfComputeSingleIteration(workerData& worker, int wIndex, int iteration,
                                 uint8_t& lp1, uint8_t& lp2,
                                 uint8_t& chunkCount, int& firstChunk);

void wolfComputeFinalize(workerData& worker, int wIndex,
                          uint8_t lp1, uint8_t lp2,
                          uint8_t chunkCount, int firstChunk);

double benchmarkInterleaved(int numIterations = 100);

#endif // INTERLEAVED_MINER_HPP
