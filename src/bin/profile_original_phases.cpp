/**
 * profile_original_phases.cpp - Profile phase timings of original AstroBWTv3
 *
 * Direct comparison with lean implementation's phase breakdown
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

#include "astrobwtv3/astrobwtv3.h"
#include "astrobwtv3/lookupcompute.h"
#include "dirtybird-hugepages.hpp"
#include "fnv1a.h"

extern "C" {
#include "astrobwtv3/divsufsort.h"
}

// Stubs
bool SPSA(const uint8_t*, int, workerData&) { return false; }
bool g_use_spsa = false;
bool g_verbose_tune = false;
bool printHugepagesError = false;
AstroFunc allAstroFuncs[] = {{"wolfCompute", nullptr}};
size_t numAstroFuncs = 1;

// External declarations
extern void wolfCompute(workerData &worker, bool isTest, int wIndex);
extern void initWolfPermuteDispatch();
extern void (*astroCompFunc)(workerData &worker, bool isTest, int wIndex);
extern void hashSHA256(SHA256_CTX &sha256, const byte *input, byte *digest, unsigned long inputSize);

using Clock = std::chrono::steady_clock;
using Duration = std::chrono::duration<double, std::micro>;

int main() {
    printf("=== Original AstroBWTv3 Phase Profiler ===\n\n");

    // Initialize
    initWolfPermuteDispatch();
    astroCompFunc = wolfCompute;

    // Allocate worker with huge pages
    workerData* worker = static_cast<workerData*>(malloc_huge_pages(sizeof(workerData)));
    if (!worker) {
        printf("No huge pages, using regular malloc\n");
        worker = static_cast<workerData*>(malloc(sizeof(workerData)));
    }
    initWorker(*worker);
    lookupGen(*worker, nullptr, nullptr);

    // Thread-local bucket arrays (same as original when SPSA disabled at runtime)
    static thread_local int tl_bA[256];
    static thread_local int tl_bB[256*256];

    // Input
    uint8_t input[48];
    for (int i = 0; i < 48; i++) input[i] = (uint8_t)i;
    uint8_t hash[32];
    uint8_t scratch[384] = {0};

    // Warmup
    printf("Warming up...\n");
    for (int i = 0; i < 100; i++) {
        AstroBWTv3(input, 48, hash, *worker, false);
        input[0]++;
    }
    for (int i = 0; i < 48; i++) input[i] = (uint8_t)i;

    // Phase profiling
    const int SAMPLES = 100;
    double total_prep_us = 0, total_wolf_us = 0, total_sa_us = 0, total_final_us = 0;
    uint64_t total_data_len = 0;

    printf("Profiling %d hashes...\n\n", SAMPLES);

    for (int s = 0; s < SAMPLES; s++) {
        memset(scratch, 0, sizeof(scratch));

        // Phase 1: Prep (SHA256 + Salsa20 + RC4)
        auto t0 = Clock::now();
        hashSHA256(worker->sha256, input, &scratch[320], 48);
        worker->salsa20.setKey(&scratch[320]);
        worker->salsa20.setIv(&scratch[256]);
        worker->salsa20.processBytes(worker->salsaInput, scratch, 256);
        RC4_set_key(&worker->key[0], 256, scratch);
        RC4(&worker->key[0], 256, scratch, scratch);
        worker->lhash = hash_64_fnv1a_256(scratch);
        worker->prev_lhash = worker->lhash;
        worker->tries[0] = 0;
        worker->isSame = false;
        memcpy(worker->sData, scratch, 256);
        auto t1 = Clock::now();

        // Phase 2: Wolf compute
        wolfCompute(*worker, false, 0);
        auto t2 = Clock::now();

        // Phase 3: Suffix Array
        divsufsort(worker->sData, worker->sa, worker->data_len, tl_bA, tl_bB);
        auto t3 = Clock::now();

        // Phase 4: Final hash
        uint8_t* B = reinterpret_cast<uint8_t*>(worker->sa);
        hashSHA256(worker->sha256, B, hash, worker->data_len * 4);
        auto t4 = Clock::now();

        total_prep_us += Duration(t1 - t0).count();
        total_wolf_us += Duration(t2 - t1).count();
        total_sa_us += Duration(t3 - t2).count();
        total_final_us += Duration(t4 - t3).count();
        total_data_len += worker->data_len;

        input[0]++;
    }

    double avg_prep = total_prep_us / SAMPLES;
    double avg_wolf = total_wolf_us / SAMPLES;
    double avg_sa = total_sa_us / SAMPLES;
    double avg_final = total_final_us / SAMPLES;
    double avg_total = avg_prep + avg_wolf + avg_sa + avg_final;
    double avg_data_len = (double)total_data_len / SAMPLES;

    printf("Phase breakdown (microseconds per hash):\n");
    printf("  Prep:   %8.2f us (%5.1f%%)\n", avg_prep, 100.0 * avg_prep / avg_total);
    printf("  Wolf:   %8.2f us (%5.1f%%)\n", avg_wolf, 100.0 * avg_wolf / avg_total);
    printf("  SA:     %8.2f us (%5.1f%%)\n", avg_sa, 100.0 * avg_sa / avg_total);
    printf("  Final:  %8.2f us (%5.1f%%)\n", avg_final, 100.0 * avg_final / avg_total);
    printf("  TOTAL:  %8.2f us = %.1f H/s\n", avg_total, 1e6 / avg_total);
    printf("\n  Avg data_len: %.0f bytes\n", avg_data_len);

    // Benchmark for 10 seconds
    printf("\n=== 10 second full benchmark ===\n");
    for (int i = 0; i < 48; i++) input[i] = (uint8_t)i;

    uint64_t hashes = 0;
    auto start = Clock::now();
    while (true) {
        AstroBWTv3(input, 48, hash, *worker, false);
        hashes++;
        input[0]++;
        if (input[0] == 0) input[1]++;

        auto now = Clock::now();
        if (std::chrono::duration<double>(now - start).count() >= 10.0) break;
    }
    auto end = Clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    printf("Rate: %.2f H/s\n", hashes / elapsed);

    free_huge_pages(worker);
    return 0;
}
