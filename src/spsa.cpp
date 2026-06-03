/*
 * spsa.cpp -- SPSA bridge
 *
 * Routes to the local C++ SPSA implementation (spsa_local.cpp).
 * The precompiled binary library from Tritonn has a 10% dev fee
 * and DRM anti-tamper. This local port is fee-free.
 */

#include "spsa.hpp"
#include "spsa_state.hpp"
#include "dluna.h"
#include <cstdio>
#include <cstdlib>
#include <atomic>

extern std::string devWallet;
extern std::atomic<int> devFee;
extern std::atomic<bool> ABORT_MINER;

bool SPSA(const uint8_t *data, int dataSize, workerData &worker)
{
    /* 2026-04-29: spsa::SPSA_Integrated is correct (matches libsais byte-for-
     * byte, verified on pow("a") with diff harness) but 3.4x slower as an SA
     * algorithm than libsais. Production path: do nothing here, leave padding
     * zero, dluna_hash falls through to libsais. spsa_local.cpp stays in tree
     * as a proven-correct clean-room SPSA reference; not on the hot path.
     * Set DLUNA_USE_SPSA_LOCAL=1 in env to re-activate (unused now). */
    static const bool use_local = []{ const char* e = std::getenv("DLUNA_USE_SPSA_LOCAL"); return e && e[0] == '1'; }();
    if (use_local) return spsa::SPSA_Integrated(data, dataSize, worker, worker.padding);
    return false;
}

void initSPSA(void)
{
    /* Local SPSA needs no init. */
}
