/**
 * profile_lean_vs_original.cpp - Profile phase timings of lean vs original
 *
 * Measures time spent in each phase:
 * 1. Prep phase (SHA256 + Salsa20 + RC4)
 * 2. Wolf compute (278 iterations)
 * 3. SA (suffix array)
 * 4. Final SHA256
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

// Include lean implementation
#include "astrobwtv3/lean_interleaved_miner.hpp"

// Include original implementation
#include "astroworker.h"
extern "C" {
#include "divsufsort.h"
}

// SPSA stub
bool SPSA(const uint8_t*, int, workerData&) { return false; }
bool g_use_spsa = false;
bool g_verbose_tune = false;
int numAstroFuncs = 0;
void* allAstroFuncs = nullptr;
bool printHugepagesError = false;

// Global wolfCompute dispatch
std::function<void(workerData &, bool, int)> astroCompFunc = nullptr;

// Timing helpers
using Clock = std::chrono::steady_clock;
using Duration = std::chrono::duration<double, std::micro>;

// External wolfCompute from astrobwtv3.cpp
extern void wolfCompute(workerData &worker, bool isTest, int wIndex);
extern void initWolfPermuteDispatch();

// Profile lean implementation
void profile_lean(int iterations) {
    printf("\n=== Profiling LEAN Implementation (%d iterations) ===\n", iterations);

    lean_interleaved::LeanInterleavedThread& thread = lean_interleaved::get_thread_state();
    thread.init();

    uint8_t input[48];
    for (int i = 0; i < 48; i++) input[i] = (uint8_t)i;

    uint8_t hash[32];

    double total_prep_us = 0;
    double total_wolf_us = 0;
    double total_sa_us = 0;
    double total_final_us = 0;

    for (int iter = 0; iter < iterations; iter++) {
        auto& worker = thread.worker_a;
        int32_t* sa = thread.shared.sa_a;

        // Phase 1: Prep
        auto t0 = Clock::now();
        lean_interleaved::prepPhase(worker, input, 48);
        auto t1 = Clock::now();

        // Phase 2: Wolf
        uint8_t chunkCount = 1;
        int firstChunk = 0;
        uint8_t lp1 = 0, lp2 = 255;
        while (lean_interleaved::wolfIterationLean(worker, lp1, lp2, chunkCount, firstChunk)) {}
        lean_interleaved::wolfFinalize(worker, lp1, lp2, chunkCount, firstChunk);
        auto t2 = Clock::now();

        // Phase 3: SA
        thread.shared.compute_sa(worker.sData, worker.data_len, sa);
        auto t3 = Clock::now();

        // Phase 4: Final hash
        lean_interleaved::hashSHA256(worker.sha256, reinterpret_cast<uint8_t*>(sa), hash, worker.data_len * 4);
        auto t4 = Clock::now();

        total_prep_us += Duration(t1 - t0).count();
        total_wolf_us += Duration(t2 - t1).count();
        total_sa_us += Duration(t3 - t2).count();
        total_final_us += Duration(t4 - t3).count();

        // Vary input
        input[0]++;
    }

    double avg_prep = total_prep_us / iterations;
    double avg_wolf = total_wolf_us / iterations;
    double avg_sa = total_sa_us / iterations;
    double avg_final = total_final_us / iterations;
    double avg_total = avg_prep + avg_wolf + avg_sa + avg_final;

    printf("Phase breakdown (microseconds per hash):\n");
    printf("  Prep:   %8.2f us (%5.1f%%)\n", avg_prep, 100.0 * avg_prep / avg_total);
    printf("  Wolf:   %8.2f us (%5.1f%%)\n", avg_wolf, 100.0 * avg_wolf / avg_total);
    printf("  SA:     %8.2f us (%5.1f%%)\n", avg_sa, 100.0 * avg_sa / avg_total);
    printf("  Final:  %8.2f us (%5.1f%%)\n", avg_final, 100.0 * avg_final / avg_total);
    printf("  TOTAL:  %8.2f us\n", avg_total);
    printf("  Rate:   %.1f H/s\n", 1e6 / avg_total);
}

// Profile original implementation
void profile_original(int iterations) {
    printf("\n=== Profiling ORIGINAL Implementation (%d iterations) ===\n", iterations);

    // Initialize dispatch
    initWolfPermuteDispatch();
    astroCompFunc = wolfCompute;

    // Allocate worker
    workerData worker;
    worker.init();

    uint8_t input[48];
    for (int i = 0; i < 48; i++) input[i] = (uint8_t)i;

    uint8_t hash[32];
    uint8_t scratch[384] = {0};

    double total_prep_us = 0;
    double total_wolf_us = 0;
    double total_sa_us = 0;
    double total_final_us = 0;

    for (int iter = 0; iter < iterations; iter++) {
        memset(scratch, 0, sizeof(scratch));

        // Phase 1: Prep
        auto t0 = Clock::now();
        hashSHA256(worker.sha256, input, &scratch[320], 48);
        worker.salsa20.setKey(&scratch[320]);
        worker.salsa20.setIv(&scratch[256]);
        worker.salsa20.processBytes(worker.salsaInput, scratch, 256);

#if defined(USE_CRYPTOGAMS_RC4_DUAL)
        worker.cryptogams_rc4[0].set_key(scratch, 256);
        RC4_set_key(&worker.key[0], 256, scratch);
        worker.cryptogams_rc4[0].apply_keystream_256(scratch);
#else
        RC4_set_key(&worker.key[0], 256, scratch);
        RC4(&worker.key[0], 256, scratch, scratch);
#endif

        worker.lhash = hash_64_fnv1a_256(scratch);
        worker.prev_lhash = worker.lhash;
        worker.tries[0] = 0;
        worker.isSame = false;
        memcpy(worker.sData, scratch, 256);
        auto t1 = Clock::now();

        // Phase 2: Wolf
        wolfCompute(worker, false, 0);
        auto t2 = Clock::now();

        // Phase 3: SA
        static thread_local int32_t tl_bA[256];
        static thread_local int32_t tl_bB[256 * 256];
        divsufsort(worker.sData, worker.sa, worker.data_len, tl_bA, tl_bB);
        auto t3 = Clock::now();

        // Phase 4: Final hash
        uint8_t* B = reinterpret_cast<uint8_t*>(worker.sa);
        hashSHA256(worker.sha256, B, hash, worker.data_len * 4);
        auto t4 = Clock::now();

        total_prep_us += Duration(t1 - t0).count();
        total_wolf_us += Duration(t2 - t1).count();
        total_sa_us += Duration(t3 - t2).count();
        total_final_us += Duration(t4 - t3).count();

        // Vary input
        input[0]++;
    }

    double avg_prep = total_prep_us / iterations;
    double avg_wolf = total_wolf_us / iterations;
    double avg_sa = total_sa_us / iterations;
    double avg_final = total_final_us / iterations;
    double avg_total = avg_prep + avg_wolf + avg_sa + avg_final;

    printf("Phase breakdown (microseconds per hash):\n");
    printf("  Prep:   %8.2f us (%5.1f%%)\n", avg_prep, 100.0 * avg_prep / avg_total);
    printf("  Wolf:   %8.2f us (%5.1f%%)\n", avg_wolf, 100.0 * avg_wolf / avg_total);
    printf("  SA:     %8.2f us (%5.1f%%)\n", avg_sa, 100.0 * avg_sa / avg_total);
    printf("  Final:  %8.2f us (%5.1f%%)\n", avg_final, 100.0 * avg_final / avg_total);
    printf("  TOTAL:  %8.2f us\n", avg_total);
    printf("  Rate:   %.1f H/s\n", 1e6 / avg_total);
}

int main() {
    printf("===============================================\n");
    printf("   Lean vs Original Phase Profiler\n");
    printf("===============================================\n");

    const int WARMUP = 50;
    const int ITERATIONS = 500;

    printf("\nWarming up lean (%d iterations)...\n", WARMUP);
    {
        lean_interleaved::LeanInterleavedThread& thread = lean_interleaved::get_thread_state();
        thread.init();
        uint8_t input[48] = {0}, hash[32];
        for (int i = 0; i < WARMUP; i++) {
            lean_interleaved::AstroBWTv3_lean(input, 48, hash, thread, 0);
            input[0]++;
        }
    }

    printf("Warming up original (%d iterations)...\n", WARMUP);
    {
        initWolfPermuteDispatch();
        astroCompFunc = wolfCompute;
        workerData worker;
        worker.init();
        uint8_t input[48] = {0}, hash[32], scratch[384] = {0};
        for (int i = 0; i < WARMUP; i++) {
            memset(scratch, 0, sizeof(scratch));
            hashSHA256(worker.sha256, input, &scratch[320], 48);
            worker.salsa20.setKey(&scratch[320]);
            worker.salsa20.setIv(&scratch[256]);
            worker.salsa20.processBytes(worker.salsaInput, scratch, 256);
            RC4_set_key(&worker.key[0], 256, scratch);
            RC4(&worker.key[0], 256, scratch, scratch);
            worker.lhash = hash_64_fnv1a_256(scratch);
            worker.prev_lhash = worker.lhash;
            worker.tries[0] = 0;
            worker.isSame = false;
            memcpy(worker.sData, scratch, 256);
            wolfCompute(worker, false, 0);
            static thread_local int32_t tl_bA[256];
            static thread_local int32_t tl_bB[256 * 256];
            divsufsort(worker.sData, worker.sa, worker.data_len, tl_bA, tl_bB);
            input[0]++;
        }
    }

    profile_lean(ITERATIONS);
    profile_original(ITERATIONS);

    printf("\n=== Done ===\n");
    return 0;
}
