/*
 * miner.cpp -- mining thread + share submission
 */

#include "dluna.h"
#include "spsa.hpp"
#include "hugepages.h"
#include "hex.h"
#include "runtime_tune.h"

#include <cstdio>
#include <cstring>

extern bool g_huge_pages_avail;
extern MinerState G;

/* ---- share submission (cold path) ---- */

__attribute__((cold))
static void submit_share(const uint8_t *blob, int blob_len,
                         const std::string &jobId, uint64_t jobEpoch)
{
    /* stale gate 1: job changed while we were hashing */
    if (G.jobEpoch.load(std::memory_order_acquire) != jobEpoch) {
        G.staleDrops.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    std::string hex = hexStr(blob, blob_len);

    std::lock_guard<std::mutex> lk(G.submitMutex);

    /* stale gate 2: job changed during serialization */
    if (G.jobEpoch.load(std::memory_order_acquire) != jobEpoch) {
        G.staleDrops.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    G.submitJobId = jobId;
    G.submitBlob  = std::move(hex);
    G.submitEpoch = jobEpoch;
    G.submitReady.store(true, std::memory_order_release);
}

/* ---- mining thread ---- */

void mine_thread(int tid)
{
    dluna_tune_mining_thread();

    bool huge = false;
    auto *worker = (workerData *)alloc_pinned(sizeof(workerData), &huge);
    if (!worker) {
        fprintf(stderr, "T%d: alloc failed\n", tid);
        return;
    }
    memset(worker, 0, sizeof(workerData));
#if defined(USE_ASTRO_SPSA)
    memcpy(worker->iota8, iota8_g, 256);
#endif
    memset(worker->salsaInput, 0, 256);

    uint8_t localBlob[MINIBLOCK_SIZE];
    uint8_t target[HASH_SIZE];
    uint8_t outputHash[HASH_SIZE];
    uint32_t nonce = 0;
    int64_t localHashCount = 0;

    /* wait for first valid job */
    while (!G.quit.load(std::memory_order_relaxed)) {
        if (G.connected.load(std::memory_order_relaxed) &&
            G.jobEpoch.load(std::memory_order_acquire) > 0 &&
            G.difficulty.load(std::memory_order_relaxed) > 0)
            break;
        dluna_sleep_ms(50);
    }

    while (!G.quit.load(std::memory_order_relaxed)) {
        uint64_t epoch;
        int64_t diff;
        std::string jobId;

        /* snapshot job under lock */
        {
            std::lock_guard<std::mutex> lk(G.jobMutex);
            epoch = G.jobEpoch.load(std::memory_order_relaxed);
            diff  = G.difficulty.load(std::memory_order_relaxed);
            memcpy(localBlob, G.blobBin, MINIBLOCK_SIZE);
            jobId = G.jobId;
        }

        compute_target(diff, target);
        nonce = (uint32_t)tid << 24;

        /* inner hash loop */
        for (;;) {
            if (G.quit.load(std::memory_order_relaxed))
                goto done;

            if ((nonce & 127) == 0 &&
                G.jobEpoch.load(std::memory_order_acquire) != epoch)
                break;

            ++nonce;

            /* big-endian nonce at NONCE_OFFSET */
            localBlob[NONCE_OFFSET + 0] = (uint8_t)(nonce >> 24);
            localBlob[NONCE_OFFSET + 1] = (uint8_t)(nonce >> 16);
            localBlob[NONCE_OFFSET + 2] = (uint8_t)(nonce >> 8);
            localBlob[NONCE_OFFSET + 3] = (uint8_t)(nonce);

            localBlob[THREAD_ID_OFFSET] = (uint8_t)tid;

            dluna_hash(localBlob, MINIBLOCK_SIZE, outputHash, *worker);

            if (check_hash(outputHash, target)) {
                if (g_verbose)
                    log_line("INFO", "T%d found share nonce=%08x", tid, nonce);
                submit_share(localBlob, MINIBLOCK_SIZE, jobId, epoch);
            }

            ++localHashCount;
            if ((localHashCount & 63) == 0) {
                G.totalHashes.fetch_add(64, std::memory_order_relaxed);
                localHashCount = 0;
            }
        }
    }

done:
    /* flush residual */
    if (localHashCount > 0)
        G.totalHashes.fetch_add(localHashCount, std::memory_order_relaxed);

    free_pinned(worker, sizeof(workerData), huge);
}
