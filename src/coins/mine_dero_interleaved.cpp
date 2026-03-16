/**
 * Two Miners Per Thread - DeroLuna-style Interleaved Mining
 *
 * This implementation runs two hash computations per thread with
 * interleaved wolfCompute iterations to hide L3 cache latency.
 *
 * The key insight: When hash A is waiting for memory, work on hash B.
 */

#include "miners.hpp"
#include "numa_optimizer.hpp"
#include <astrobwtv3/astrobwtv3.h>
#include <astrobwtv3/lookupcompute.h>
#include <astrobwtv3/interleaved_miner.hpp>
#include <dirtybird-hugepages.hpp>
#include <thermal_governor.hpp>

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
 * Interleaved DERO Mining Function
 *
 * Processes two hashes per iteration using DeroLuna-style interleaving.
 * This hides memory latency by working on hash B while hash A's memory
 * requests are in flight.
 */
// External P-core globals (defined in miner.cpp)
#ifdef _WIN32
extern bool g_pcores_only;
extern bool g_no_per_thread_affinity;
extern int g_pcore_count;
extern std::vector<uint32_t> g_pcore_logical_ids;
#endif

void mineDero_Interleaved(int tid) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 255);

#ifdef _WIN32
    // P-core affinity enforcement
    if (!g_no_per_thread_affinity && g_pcores_only && tid > 0 && static_cast<size_t>(tid - 1) < g_pcore_logical_ids.size()) {
        DWORD targetCore = g_pcore_logical_ids[tid - 1];
        DWORD_PTR mask = 1ULL << targetCore;
        SetThreadAffinityMask(GetCurrentThread(), mask);
    }
#endif

    // Per-thread interleaved miner
    thread_local InterleavedMiner* miner = nullptr;
    thread_local uint64_t localCount = 0;
    thread_local uint64_t paceCount = 0;   // Thermal governor pace counter

    // Per-thread mining buffers - two work buffers for interleaved processing
    thread_local std::array<byte, MINIBLOCK_SIZE> work_a;
    thread_local std::array<byte, MINIBLOCK_SIZE> work_b;
    thread_local std::array<byte, 32> powHash_a;
    thread_local std::array<byte, 32> powHash_b;
    thread_local byte target32[32];

    if (!miner) {
        miner = new InterleavedMiner();
        if (!miner->initialize()) {
            setcolor(RED);
            std::cerr << "Failed to initialize interleaved miner for thread " << tid << std::endl << std::flush;
            setcolor(BRIGHT_WHITE);
            return;
        }

        // Optimize memory for mining
        NUMAOptimizer::optimizeMemoryForMining(miner->getWorkerA(), sizeof(workerData));
        NUMAOptimizer::optimizeMemoryForMining(miner->getWorkerB(), sizeof(workerData));

        if (tid == 0 && !beQuiet) {
            setcolor(CYAN);
            printf("Interleaved mining: 2 hashes per thread iteration (DeroLuna-style ILP)\n");
            fflush(stdout);
            setcolor(BRIGHT_WHITE);
        }

        localCount = 0;
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
            NumToTarget32(cmpDiff, target32);

            localJobCounter = snapshotCounter;
            uint32_t nonce = 0;

            while (localJobCounter == jobCounter) {
                CHECK_CLOSE;

                // ============================================================
                // INTERLEAVED HASH COMPUTATION
                // Process two hashes at once with interleaved iterations
                // ============================================================

                // Prepare work_a with nonce
                std::memcpy(work_a.data(), workTemplate.data(), MINIBLOCK_SIZE);
                nonce++;
                write_nonce_be(work_a.data(), nonce);

                // Prepare work_b with next nonce
                std::memcpy(work_b.data(), workTemplate.data(), MINIBLOCK_SIZE);
                nonce++;
                write_nonce_be(work_b.data(), nonce);

                // Process both hashes with interleaved execution
                miner->processInterleaved(
                    work_a.data(), MINIBLOCK_SIZE,
                    work_b.data(), MINIBLOCK_SIZE,
                    powHash_a.data(), powHash_b.data(),
                    false // useLookup
                );

                // Light micro-cooling: single pause reduces core heat slightly
                #if defined(__x86_64__) || defined(_M_X64)
                _mm_pause();
                #endif

                // Update hash count (2 hashes per iteration)
                localCount += 2;
                if (localCount >= 512) {
                    counter.fetch_add(localCount, std::memory_order_relaxed);
                    localCount = 0;
                }

                // Check hash A
                if (CheckHashBytes(powHash_a.data(), target32)) {
                    bool submitted = false;
                    while (!submitted) {
                        if (localJobCounter != jobCounter) break;

                        {
                            std::scoped_lock<boost::mutex> submitLock(mutex);
                            if (localJobCounter != jobCounter) break;
                            if (!submitting) {
                                submitting = true;
                                setcolor(BRIGHT_YELLOW);
                                std::cout << "\nThread " << tid << " found a nonce! (hash A)\n" << std::flush;
                                setcolor(BRIGHT_WHITE);
                                share = {
                                    {"jobid",    myJob.at("jobid").as_string().c_str()},
                                    {"mbl_blob", hexStr(work_a.data(), MINIBLOCK_SIZE).c_str()}
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

                // Check hash B
                if (CheckHashBytes(powHash_b.data(), target32)) {
                    bool submitted = false;
                    while (!submitted) {
                        if (localJobCounter != jobCounter) break;

                        {
                            std::scoped_lock<boost::mutex> submitLock(mutex);
                            if (localJobCounter != jobCounter) break;
                            if (!submitting) {
                                submitting = true;
                                setcolor(BRIGHT_YELLOW);
                                std::cout << "\nThread " << tid << " found a nonce! (hash B)\n" << std::flush;
                                setcolor(BRIGHT_WHITE);
                                share = {
                                    {"jobid",    myJob.at("jobid").as_string().c_str()},
                                    {"mbl_blob", hexStr(work_b.data(), MINIBLOCK_SIZE).c_str()}
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
            std::cerr << "Error in interleaved POW Function\n" << e.what() << std::endl << std::flush;
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
