/**
 * Cache-Focused Batched DERO Mining
 *
 * Implementation of k1's insight for AstroBWTv3:
 * - "do staged batches in the context of one thread"
 * - "i want the suffix array construction code to be 100% in L1"
 * - "pin 1 thread : 1 core for num cores"
 *
 * This miner uses staged batch processing:
 * 1. Generate N scratch buffers (prep phase)
 * 2. Run N suffix array constructions back-to-back (keeps SA code hot in L1I)
 * 3. Compute N final hashes
 *
 * The key insight is that by running multiple SA constructions consecutively,
 * the instruction cache stays warm with the divsufsort code, reducing I-cache
 * misses and improving overall throughput.
 */

#include "miners.hpp"
#include "numa_optimizer.hpp"
#include <astrobwtv3/astrobwtv3.h>
#include <astrobwtv3/lookupcompute.h>
#include <astrobwtv3/cache_batching.hpp>

#include <array>
#include <atomic>
#include <boost/json.hpp>
#include <boost/chrono.hpp>
#include <boost/thread.hpp>
#include <cstdint>
#include <cstring>
#include <random>
#include <cstdlib>

namespace {

static inline void write_nonce_be(byte* WORK, uint32_t N) {
    WORK[MINIBLOCK_SIZE - 5] = (byte)((N >> 24) & 0xFF);
    WORK[MINIBLOCK_SIZE - 4] = (byte)((N >> 16) & 0xFF);
    WORK[MINIBLOCK_SIZE - 3] = (byte)((N >>  8) & 0xFF);
    WORK[MINIBLOCK_SIZE - 2] = (byte)((N      ) & 0xFF);
}

} // namespace

/**
 * Cache-Batched DERO Mining Function
 *
 * This is a drop-in replacement for mineDero() that uses cache-focused batching.
 * It accumulates multiple inputs before processing them in staged batches,
 * maximizing instruction cache efficiency for the SA construction phase.
 */
void mineDero_Batched(int tid) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 255);

    thread_local std::mt19937 gen_local(rd());
    thread_local std::uniform_real_distribution<double> dev_dist(0, 10000);

    // NUMA-aware worker context
    thread_local workerData* worker = nullptr;
    thread_local uint64_t localCount = 0;

    // Per-thread cache-focused batcher
    thread_local CacheFocusedBatcher* batcher = nullptr;

    // Per-thread mining buffers
    thread_local std::array<byte, MINIBLOCK_SIZE * CACHE_BATCH_MAX> workBatch;
    thread_local std::array<byte, 32 * CACHE_BATCH_MAX> powHashBatch;

    if (!worker) {
        worker = static_cast<workerData*>(NUMAOptimizer::allocateLocal(sizeof(workerData)));
        if (!worker) {
            worker = static_cast<workerData*>(std::malloc(sizeof(workerData)));
        }
        if (!worker) {
            setcolor(RED);
            std::cerr << "Failed to allocate workerData for batched DERO miner" << std::endl << std::flush;
            setcolor(BRIGHT_WHITE);
            return;
        }

        NUMAOptimizer::optimizeMemoryForMining(worker, sizeof(workerData));
        initWorker(*worker);
        lookupGen(*worker, nullptr, nullptr);
        localCount = 0;
    }

    if (!batcher) {
        batcher = new CacheFocusedBatcher(CACHE_BATCH_DEFAULT);
    }

    byte random_tail[12];
    for (int i = 0; i < 12; ++i) {
        random_tail[i] = (byte)dist(gen);
    }

    boost::this_thread::sleep_for(boost::chrono::milliseconds(125));

    int64_t localJobCounter = -1;
    Num cmpDiff;

waitForJob:
    while (!isConnected) {
        CHECK_CLOSE;
        boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
    }

    while (!ABORT_MINER) {
        try {
            boost::json::value myJob;
            int64_t snapshotCounter;
            int64_t diffSnapshot;

            {
                std::scoped_lock<boost::mutex> lockGuard(mutex);
                myJob           = job;
                snapshotCounter = jobCounter;
                diffSnapshot    = difficulty;
            }

            // Parse job template
            std::array<byte, MINIBLOCK_SIZE> workTemplate;
            hexstrToBytes(std::string(myJob.at("blockhashing_blob").as_string()), workTemplate.data());

            std::memcpy(&workTemplate[MINIBLOCK_SIZE - 12], random_tail, 12);
            workTemplate[MINIBLOCK_SIZE - 1] = (byte)tid;

            if ((workTemplate[0] & 0x0F) != 1) {
                std::cerr << "Unknown version, please check for updates: version"
                          << (workTemplate[0] & 0x1F) << std::endl;
                boost::this_thread::sleep_for(boost::chrono::milliseconds(500));
                continue;
            }

            cmpDiff = ConvertDifficultyToBig(diffSnapshot, ALGO_ASTROBWTV3);

            localJobCounter = snapshotCounter;
            uint32_t nonce = 0;

            while (localJobCounter == jobCounter) {
                CHECK_CLOSE;

                const int batchSize = batcher->getMaxBatchSize();

                // ============================================================
                // BATCH ACCUMULATION PHASE
                // Generate batch of inputs with different nonces
                // ============================================================
                batcher->clear();
                uint32_t startNonce = nonce;

                for (int i = 0; i < batchSize && localJobCounter == jobCounter; ++i) {
                    nonce++;
                    byte* work = workBatch.data() + i * MINIBLOCK_SIZE;
                    std::memcpy(work, workTemplate.data(), MINIBLOCK_SIZE);
                    write_nonce_be(work, nonce);
                    batcher->addInput(work, MINIBLOCK_SIZE);
                }

                if (localJobCounter != jobCounter) {
                    break;
                }

                // ============================================================
                // STAGED BATCH PROCESSING
                // This is where the cache-focused optimization happens:
                // 1. Prep all buffers (SHA256 + Salsa20 + RC4 + wolfCompute)
                // 2. Run ALL SA constructions back-to-back (L1I hot)
                // 3. Compute all final hashes
                // ============================================================
                int hashesComputed = batcher->processBatch(*worker, powHashBatch.data());

                // ============================================================
                // SOLUTION CHECKING PHASE
                // Check each hash against the difficulty target
                // ============================================================
                for (int i = 0; i < hashesComputed; ++i) {
                    byte* powHash = powHashBatch.data() + i * 32;
                    byte* WORK = workBatch.data() + i * MINIBLOCK_SIZE;

                    if (++localCount >= 512) {
                        counter.fetch_add(localCount);
                        localCount = 0;
                    }

                    if (CheckHash(powHash, cmpDiff, ALGO_ASTROBWTV3)) {
                        bool submitted = false;
                        while (!submitted) {
                            if (localJobCounter != jobCounter) break;

                            {
                                std::scoped_lock<boost::mutex> submitLock(mutex);
                                if (localJobCounter != jobCounter) break;
                                if (!submitting) {
                                    submitting = true;
                                    setcolor(BRIGHT_YELLOW);
                                    std::cout << "\nThread " << tid << " found a nonce! (batch index " << i << ")\n" << std::flush;
                                    setcolor(BRIGHT_WHITE);
                                    share = {
                                        {"jobid",    myJob.at("jobid").as_string().c_str()},
                                        {"mbl_blob", hexStr(WORK, MINIBLOCK_SIZE).c_str()}
                                    };
                                    data_ready = true;
                                    submitted = true;
                                }
                            }

                            if (!submitted) {
                                boost::this_thread::yield();
                            }
                        }

                        if (!submitted && localJobCounter != jobCounter) break;
                        if (submitted) cv.notify_all();
                    }
                }

                if (!isConnected) break;
            }

            if (localCount) {
                counter.fetch_add(localCount);
                localCount = 0;
            }

            if (!isConnected) break;
        }
        catch (const std::exception& e) {
            setcolor(RED);
            std::cerr << "Error in batched POW Function\n" << e.what() << std::endl << std::flush;
            setcolor(BRIGHT_WHITE);

            localJobCounter = -1;
            if (localCount) {
                counter.fetch_add(localCount);
                localCount = 0;
            }
        }

        if (!isConnected) break;
    }

    goto waitForJob;
}

/**
 * Hybrid Mining Function
 *
 * This variant automatically selects between standard and batched mining
 * based on a quick benchmark at startup. Use this if you're unsure which
 * approach is better for your CPU.
 */
void mineDero_Hybrid(int tid) {
    // Only thread 0 runs the benchmark
    static std::atomic<int> optimalBatchSize{0};
    static std::atomic<bool> benchmarkDone{false};

    if (tid == 1 && !benchmarkDone.load()) {
        // Quick benchmark to determine optimal strategy
        setcolor(CYAN);
        std::cout << "Thread 1 running cache-batching benchmark..." << std::endl;
        setcolor(BRIGHT_WHITE);

        int optimal = benchmarkCacheBatching(10); // Quick benchmark
        optimalBatchSize.store(optimal);
        benchmarkDone.store(true);

        if (optimal <= 1) {
            setcolor(BRIGHT_YELLOW);
            std::cout << "Using standard sequential mining (cache batching not beneficial on this CPU)" << std::endl;
            setcolor(BRIGHT_WHITE);
        } else {
            setcolor(BRIGHT_YELLOW);
            std::cout << "Using cache-batched mining with batch size " << optimal << std::endl;
            setcolor(BRIGHT_WHITE);
        }
    }

    // Wait for benchmark to complete
    while (!benchmarkDone.load()) {
        boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
    }

    // Route to appropriate mining function
    if (optimalBatchSize.load() <= 1) {
        mineDero(tid);  // Standard mining
    } else {
        mineDero_Batched(tid);  // Batched mining
    }
}
