/**
 * Lock-Free DERO Mining Implementation
 *
 * Optimization #3: Go-style thread coordination
 *
 * Key differences from standard mineDero:
 * 1. Lock-free job counter polling (atomic compare instead of mutex)
 * 2. Local job caching to reduce lock contention
 * 3. Batch hash counting (already in standard, but optimized)
 * 4. Memory prefetching for worker data
 *
 * This mimics Go's goroutine pattern where threads check for work
 * updates using cheap atomic operations rather than locks.
 */

#include "miners.hpp"
#include "numa_optimizer.hpp"
#include "dirtybird-hugepages.hpp"
#include "thermal_governor.hpp"
#include <astrobwtv3/astrobwtv3.h>
#include <astrobwtv3/lookupcompute.h>

#include <array>
#include <atomic>
#include <boost/json.hpp>
#include <boost/chrono.hpp>
#include <boost/thread.hpp>
#include <cstdint>
#include <cstring>
#include <random>

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef DIRTYBIRD_OPENMP
#include <omp.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace {

// Lock-free job snapshot structure
struct alignas(64) JobSnapshot {
    std::array<byte, MINIBLOCK_SIZE> work;
    int64_t counter;
    int64_t diff;
    std::string jobId;
    bool valid;
};

static inline void write_nonce_be(byte* WORK, uint32_t N) {
    WORK[MINIBLOCK_SIZE - 5] = static_cast<byte>((N >> 24) & 0xFF);
    WORK[MINIBLOCK_SIZE - 4] = static_cast<byte>((N >> 16) & 0xFF);
    WORK[MINIBLOCK_SIZE - 3] = static_cast<byte>((N >>  8) & 0xFF);
    WORK[MINIBLOCK_SIZE - 2] = static_cast<byte>((N      ) & 0xFF);
}

// Prefetch worker data for next hash
static inline void prefetch_worker(workerData* worker) {
#if defined(__x86_64__) || defined(_M_X64)
    // Prefetch frequently accessed worker fields
    _mm_prefetch((const char*)worker->sData, _MM_HINT_T0);
    _mm_prefetch((const char*)&worker->sa[0], _MM_HINT_T0);
    _mm_prefetch((const char*)&worker->lhash, _MM_HINT_T0);
#endif
}

} // namespace

// External declarations
#ifdef _WIN32
extern bool g_pcores_only;
extern int g_pcore_count;
extern std::vector<uint32_t> g_pcore_logical_ids;
#endif

/**
 * Lock-Free DERO Mining Function
 *
 * Thread coordination improvements:
 * 1. Atomic job counter check (no mutex for polling)
 * 2. Copy-on-change job snapshot
 * 3. Batch counter updates (512 hashes)
 * 4. Lock-free share flag checking
 */
void mineDero_LockFree(int tid)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 255);

#ifdef _WIN32
    // P-core affinity enforcement
    if (g_pcores_only && tid > 0 && static_cast<size_t>(tid - 1) < g_pcore_logical_ids.size()) {
        DWORD targetCore = g_pcore_logical_ids[tid - 1];
        DWORD_PTR mask = 1ULL << targetCore;
        SetThreadAffinityMask(GetCurrentThread(), mask);
    }
#endif

    // Adaptive: check if this thread was culled before starting
    if (tid > 0 && tid < MAX_MINING_THREADS && thread_stop[tid].load(std::memory_order_relaxed)) {
        return;
    }

    // Thread-local state
    thread_local workerData* worker = nullptr;
    thread_local uint64_t localCount = 0;
    thread_local JobSnapshot localJob;
    thread_local std::array<byte, 32> powHash;
    thread_local Num cmpDiff;
    thread_local std::array<byte, 32> cmpTarget;

    // Initialize worker on first call
    if (!worker) {
        worker = static_cast<workerData*>(malloc_huge_pages(sizeof(workerData)));
        if (!worker) {
            worker = static_cast<workerData*>(std::malloc(sizeof(workerData)));
        }
        if (!worker) {
            setcolor(RED);
            std::cerr << "Failed to allocate workerData" << std::endl << std::flush;
            setcolor(BRIGHT_WHITE);
            return;
        }
        initWorker(*worker);
        lookupGen(*worker, nullptr, nullptr);
        localJob.valid = false;
        localJob.counter = -1;

#ifdef DIRTYBIRD_OPENMP
        // Configure OpenMP threads
        if (tid == 0) {
            int omp_threads = (g_omp_threads > 0) ? g_omp_threads : 2;
            omp_set_num_threads(omp_threads);
            omp_set_dynamic(0);
        }
#endif
    }

    // Random tail for work template
    byte random_tail[12];
    for (int i = 0; i < 12; ++i) {
        random_tail[i] = static_cast<byte>(dist(gen));
    }

    // Initial sleep to stagger thread starts
    boost::this_thread::sleep_for(boost::chrono::milliseconds(125));

waitForJob:
    while (!isConnected) {
        CHECK_CLOSE;
        // Adaptive: check if this thread was culled during warmup
        if (tid > 0 && tid < MAX_MINING_THREADS && thread_stop[tid].load(std::memory_order_relaxed)) {
            if (worker) { free_huge_pages(worker); worker = nullptr; }
            return;
        }
        boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
    }

    while (!ABORT_MINER) {
        try {
            // OPTIMIZATION: Lock-free job counter check
            // Only acquire lock if job actually changed
            int64_t currentCounter = jobCounter;  // volatile read

            if (currentCounter != localJob.counter) {
                // Job changed - need to acquire lock and update snapshot
                {
                    std::scoped_lock<boost::mutex> lock(mutex);
                    localJob.counter = jobCounter;
                    localJob.diff = difficulty;

                    hexstrToBytes(std::string(job.at("blockhashing_blob").as_string()),
                                  localJob.work.data());
                    localJob.jobId = std::string(job.at("jobid").as_string());
                }

                // Apply thread-specific modifications
                std::memcpy(&localJob.work[MINIBLOCK_SIZE - 12], random_tail, 12);
                localJob.work[MINIBLOCK_SIZE - 1] = static_cast<byte>(tid);

                // Validate work version
                if ((localJob.work[0] & 0x0F) != 1) {
                    std::cerr << "Unknown version, please check for updates" << std::endl;
                    boost::this_thread::sleep_for(boost::chrono::milliseconds(500));
                    continue;
                }

                cmpDiff = ConvertDifficultyToBig(localJob.diff, ALGO_ASTROBWTV3);
                NumToTarget32(cmpDiff, cmpTarget.data());
                localJob.valid = true;
            }

            if (!localJob.valid) {
                boost::this_thread::sleep_for(boost::chrono::milliseconds(10));
                continue;
            }

            // Mining loop with lock-free job checking
            uint32_t nonce = 0;
            byte* WORK = localJob.work.data();

            while (true) {
                CHECK_CLOSE;

                // OPTIMIZATION: Lock-free job counter check (cheap volatile read)
                if (jobCounter != localJob.counter) {
                    break;  // Job changed, exit inner loop
                }

                ++nonce;
                write_nonce_be(WORK, nonce);

                // Prefetch for next iteration
                prefetch_worker(worker);

                // Compute hash
                AstroBWTv3(WORK, MINIBLOCK_SIZE, powHash.data(), *worker, useLookupMine);

                // Batch counter update (every 512 hashes)
                if (++localCount >= 512) {
                    counter.fetch_add(localCount, std::memory_order_relaxed);
                    // Per-thread counter for adaptive thread management
                    if (tid > 0 && tid < MAX_MINING_THREADS) {
                        thread_counters[tid].fetch_add(localCount, std::memory_order_relaxed);
                    }
                    localCount = 0;

                    // Adaptive: check if this thread was culled
                    if (tid > 0 && tid < MAX_MINING_THREADS && thread_stop[tid].load(std::memory_order_relaxed)) {
                        if (worker) { free_huge_pages(worker); worker = nullptr; }
                        return;
                    }

                    // Thermal pacing: if governor detects throttling, yield briefly
                    // to prevent cliff-throttling. Cost: one atomic load per 512 hashes.
                    if (thermal::should_thermal_yield()) {
                        thermal::do_thermal_yield();
                    }
                }

                // Check if hash meets difficulty
                if (CheckHashBytes(powHash.data(), cmpTarget.data())) {
                    // OPTIMIZATION: Lock-free check if someone else is submitting
                    while (submitting) {
                        if (jobCounter != localJob.counter) {
                            goto jobChanged;
                        }
                        boost::this_thread::yield();
                    }

                    // Acquire lock only for actual submission
                    {
                        std::scoped_lock<boost::mutex> submitLock(mutex);
                        if (jobCounter != localJob.counter) {
                            goto jobChanged;
                        }

                        submitting = true;
                        setcolor(BRIGHT_YELLOW);
                        std::cout << "\nThread " << tid << " found a nonce!\n" << std::flush;
                        setcolor(BRIGHT_WHITE);

                        share = {
                            {"jobid",    localJob.jobId.c_str()},
                            {"mbl_blob", hexStr(WORK, MINIBLOCK_SIZE).c_str()}
                        };
                        data_ready = true;
                    }
                    cv.notify_all();
                }

                if (!isConnected) break;
            }

jobChanged:
            // Flush any remaining counts
            if (localCount > 0) {
                counter.fetch_add(localCount, std::memory_order_relaxed);
                localCount = 0;
            }

            if (!isConnected) break;
        }
        catch (const std::exception& e) {
            setcolor(RED);
            std::cerr << "Error in POW Function\n" << e.what() << std::endl << std::flush;
            setcolor(BRIGHT_WHITE);

            localJob.valid = false;
            if (localCount > 0) {
                counter.fetch_add(localCount, std::memory_order_relaxed);
                localCount = 0;
            }
        }

        if (!isConnected) break;
    }

    // Adaptive: exit cleanly if this thread was culled
    if (tid > 0 && tid < MAX_MINING_THREADS && thread_stop[tid].load(std::memory_order_relaxed)) {
        if (worker) { free_huge_pages(worker); worker = nullptr; }
        return;
    }

    goto waitForJob;
}
