#include "miners.hpp"
#include "numa_optimizer.hpp"
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
#include <cstdlib>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef DIRTYBIRD_OPENMP
#include <omp.h>
#endif

namespace {

static inline void write_nonce_be(byte* WORK, uint32_t N) {
  WORK[MINIBLOCK_SIZE - 5] = (byte)((N >> 24) & 0xFF);
  WORK[MINIBLOCK_SIZE - 4] = (byte)((N >> 16) & 0xFF);
  WORK[MINIBLOCK_SIZE - 3] = (byte)((N >>  8) & 0xFF);
  WORK[MINIBLOCK_SIZE - 2] = (byte)((N      ) & 0xFF);
}

} // namespace

// External P-core globals (defined in miner.cpp)
#ifdef _WIN32
extern bool g_pcores_only;
extern int g_pcore_count;
extern std::vector<uint32_t> g_pcore_logical_ids;
#endif

void mineDero(int tid)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(0, 255);

#ifdef _WIN32
  // P-core affinity enforcement (safety measure for hybrid CPUs)
  // Re-apply affinity in case the OS migrated the thread
  if (g_pcores_only && tid > 0 && static_cast<size_t>(tid - 1) < g_pcore_logical_ids.size()) {
    DWORD targetCore = g_pcore_logical_ids[tid - 1];  // tid is 1-based
    DWORD_PTR mask = 1ULL << targetCore;
    SetThreadAffinityMask(GetCurrentThread(), mask);
  }
#endif

  thread_local std::mt19937 gen_local(rd());
  // NUMA-aware worker context and counters
  thread_local workerData* worker = nullptr;
  thread_local uint64_t localCount = 0;

  // Per-thread mining buffers
  thread_local std::array<byte, MINIBLOCK_SIZE> work;
  thread_local std::array<byte, 32> powHash;

  if (!worker) {
    worker = static_cast<workerData*>(NUMAOptimizer::allocateLocal(sizeof(workerData)));
    if (!worker) {
      worker = static_cast<workerData*>(std::malloc(sizeof(workerData)));
    }
    if (!worker) {
      setcolor(RED);
      std::cerr << "Failed to allocate workerData for DERO miner" << std::endl << std::flush;
      setcolor(BRIGHT_WHITE);
      return;
    }

    NUMAOptimizer::optimizeMemoryForMining(worker, sizeof(workerData));
    initWorker(*worker);
    lookupGen(*worker, nullptr, nullptr);
    localCount = 0;

#ifdef DIRTYBIRD_OPENMP
    // Configure OpenMP threads for this mining thread
    // With N mining threads, each gets OMP workers for parallel suffix array construction
    // This mirrors TNN/Luna behavior where 20 mining threads create ~41 total OS threads
    if (tid == 0) {  // Only log once from first thread
      int omp_threads;
      if (g_omp_threads > 0) {
        omp_threads = g_omp_threads;
      } else {
        // Auto mode: DeroLuna uses ~2 OMP threads per miner (20T creates ~41 total)
        // Force 2 threads to match that behavior for SPSA parallel SA construction
        omp_threads = 2;  // Fixed: was causing thread starvation with auto calculation
      }
      omp_set_num_threads(omp_threads);
      omp_set_dynamic(0);  // Disable dynamic adjustment for consistent behavior

      if (!beQuiet) {
        setcolor(CYAN);
        printf("OpenMP: %d threads per miner (%d miners x %d OMP = ~%d total threads)\n",
               omp_threads, threads, omp_threads, threads * omp_threads);
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
      }
    }
#endif
  }

  byte random_tail[12];
  for (int i = 0; i < 12; ++i) random_tail[i] = (byte)dist(gen);

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

      hexstrToBytes(std::string(myJob.at("blockhashing_blob").as_string()), work.data());

      std::memcpy(&work[MINIBLOCK_SIZE - 12], random_tail, 12);
      work[MINIBLOCK_SIZE - 1] = (byte)tid;

      if ((work[0] & 0x0F) != 1) {
        std::cerr << "Unknown version, please check for updates: version"
                  << (work[0] & 0x1F) << std::endl;
        boost::this_thread::sleep_for(boost::chrono::milliseconds(500));
        continue;
      }

      cmpDiff = ConvertDifficultyToBig(diffSnapshot, ALGO_ASTROBWTV3);

      localJobCounter = snapshotCounter;
      uint32_t nonce  = 0;

      while (localJobCounter == jobCounter) {
        CHECK_CLOSE;

        ++nonce;
        write_nonce_be(work.data(), nonce);

        AstroBWTv3(work.data(), MINIBLOCK_SIZE, powHash.data(), *worker, useLookupMine);

        if (++localCount >= 512) {
          counter.fetch_add(localCount);
          localCount = 0;
        }

        if (CheckHash(powHash.data(), cmpDiff, ALGO_ASTROBWTV3)) {
          if (!submitting) {
            // Wait if another thread is submitting
            while (submitting && localJobCounter == jobCounter) {
              boost::this_thread::yield();
            }
          }
          if (localJobCounter != jobCounter) break;

          // Protect share submission with mutex
          {
            std::scoped_lock<boost::mutex> submitLock(mutex);
            submitting = true;
            setcolor(BRIGHT_YELLOW);
            std::cout << "\nThread " << tid << " found a nonce!\n" << std::flush;
            setcolor(BRIGHT_WHITE);
            share = {
              {"jobid",    myJob.at("jobid").as_string().c_str()},
              {"mbl_blob", hexStr(work.data(), MINIBLOCK_SIZE).c_str()}
            };
            data_ready = true;
          }
          cv.notify_all();
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
      std::cerr << "Error in POW Function\n" << e.what() << std::endl << std::flush;
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
