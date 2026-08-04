/*
 * miner.cpp -- mining thread + share submission
 */

#include "dluna.h"
#include "spsa.hpp"
#include "hugepages.h"
#include "hex.h"
#include "runtime_tune.h"
#include "cpu_topology.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

extern bool g_huge_pages_avail;
extern MinerState G;

/* ---- share submission (cold path) ---- */

__attribute__((cold))
static void submit_share(const uint8_t *blob, int blob_len,
                         const std::string &jobId, uint64_t jobEpoch)
{
    G.found.fetch_add(1, std::memory_order_relaxed);  /* every find, before any gate */

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

    /* Enqueue rather than overwrite: with 20+ threads, two near-simultaneous
     * finds must both reach the network thread. Counts an eviction if the queue
     * is somehow full, which at depth 32 means the drain has stopped. */
    dluna_submit_enqueue(G, jobId, std::move(hex), jobEpoch);
}

/* ---- mining thread ---- */

/* Job-poll cadence: how often, in nonces, a worker checks for a new job.
 * Default 1 = check every iteration, which is what the derohe reference miner
 * and five of the other readable DERO miners do; they abandon stale work within
 * one hash (~1 ms). Env-tunable via DLUNA_JOB_POLL (rounded up to a power of
 * two) so the cadence can be loosened for A/B. */
static uint32_t job_poll_mask()
{
    static const uint32_t m = []() -> uint32_t {
        const char *e = getenv("DLUNA_JOB_POLL");
        int v = (e && e[0]) ? atoi(e) : 1;
        if (v < 1) v = 1;
        uint32_t p = 1;
        while (p < (uint32_t)v) p <<= 1;
        return p - 1u;
    }();
    return m;
}

/* DLUNA_COUNT_STALE=1 counts hashes computed against an already-advanced
 * epoch. Measurement only -- those hashes cannot yield a valid share. */
static bool count_stale_enabled()
{
    static const bool b = [] {
        const char *e = getenv("DLUNA_COUNT_STALE");
        return e && e[0] == '1';
    }();
    return b;
}

void mine_thread(int tid)
{
    dluna_tune_mining_thread();
    dluna_pin_mining_thread(tid);
    const uint32_t poll_mask = job_poll_mask();
    const bool count_stale = count_stale_enabled();
    int64_t localStale = 0;

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

    /* Second worker for the 2-way SHA path: two nonces in flight, SA stages
     * serial per lane, final SHA-256 batched on interleaved SHA-NI streams.
     * Opt out with DLUNA_NO_SHA_X2=1. */
    static const bool want_x2 = [] {
        const char *e = getenv("DLUNA_NO_SHA_X2");
        if (e && e[0] && strcmp(e, "0") != 0) return false;
        return dluna_hash_x2_available();
    }();
    workerData *worker_b = nullptr;
    bool huge_b = false;
    if (want_x2) {
        worker_b = (workerData *)alloc_pinned(sizeof(workerData), &huge_b);
        if (worker_b) {
            memset(worker_b, 0, sizeof(workerData));
#if defined(USE_ASTRO_SPSA)
            memcpy(worker_b->iota8, iota8_g, 256);
#endif
            memset(worker_b->salsaInput, 0, 256);
        }
    }
    const bool use_x2 = want_x2 && worker_b != nullptr;

    uint8_t localBlob[MINIBLOCK_SIZE];
    uint8_t localBlobB[MINIBLOCK_SIZE];
    uint8_t target[HASH_SIZE];
    uint8_t outputHash[HASH_SIZE];
    uint8_t outputHashB[HASH_SIZE];
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
        if (use_x2)
            memcpy(localBlobB, localBlob, MINIBLOCK_SIZE);

        /* inner hash loop */
        for (;;) {
            if (G.quit.load(std::memory_order_relaxed))
                goto done;

            /* Poll for a new job every poll_mask+1 nonces (default: every
             * iteration, ~1 ms) rather than every 128 (~140 ms at 20T). The
             * work blob changes often, and every hash computed after the epoch
             * advances is on stale work that cannot yield a valid share. The
             * jobEpoch load is an L1 hit -- the line is written only on a job
             * change and this loop already reads G.quit each iteration -- so
             * the tighter poll costs no measurable raw hashrate. Hash output is
             * unchanged. */
            if ((nonce & poll_mask) == 0 &&
                G.jobEpoch.load(std::memory_order_acquire) != epoch)
                break;

            if (count_stale &&
                G.jobEpoch.load(std::memory_order_relaxed) != epoch)
                localStale += use_x2 ? 2 : 1;

            if (use_x2) {
                const uint32_t na = ++nonce;
                const uint32_t nb = ++nonce;

                localBlob[NONCE_OFFSET + 0] = (uint8_t)(na >> 24);
                localBlob[NONCE_OFFSET + 1] = (uint8_t)(na >> 16);
                localBlob[NONCE_OFFSET + 2] = (uint8_t)(na >> 8);
                localBlob[NONCE_OFFSET + 3] = (uint8_t)(na);
                localBlob[THREAD_ID_OFFSET] = (uint8_t)tid;

                localBlobB[NONCE_OFFSET + 0] = (uint8_t)(nb >> 24);
                localBlobB[NONCE_OFFSET + 1] = (uint8_t)(nb >> 16);
                localBlobB[NONCE_OFFSET + 2] = (uint8_t)(nb >> 8);
                localBlobB[NONCE_OFFSET + 3] = (uint8_t)(nb);
                localBlobB[THREAD_ID_OFFSET] = (uint8_t)tid;

                dluna_hash_x2(localBlob, localBlobB, MINIBLOCK_SIZE,
                              outputHash, outputHashB, *worker, *worker_b);

                if (check_hash(outputHash, target)) {
                    if (g_verbose)
                        log_line("INFO", "T%d found share nonce=%08x", tid, na);
                    submit_share(localBlob, MINIBLOCK_SIZE, jobId, epoch);
                }
                if (check_hash(outputHashB, target)) {
                    if (g_verbose)
                        log_line("INFO", "T%d found share nonce=%08x", tid, nb);
                    submit_share(localBlobB, MINIBLOCK_SIZE, jobId, epoch);
                }

                localHashCount += 2;
            } else {
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
            }

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
    if (localStale > 0)
        G.staleHashesObserved.fetch_add(localStale, std::memory_order_relaxed);

    if (worker_b)
        free_pinned(worker_b, sizeof(workerData), huge_b);
    free_pinned(worker, sizeof(workerData), huge);
}
