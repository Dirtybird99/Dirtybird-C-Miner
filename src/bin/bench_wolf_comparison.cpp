/**
 * bench_wolf_comparison.cpp - Direct comparison of wolfCompute implementations
 *
 * Isolates the wolf compute phase to find the performance gap
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

// Include lean
#include "astrobwtv3/lean_interleaved_miner.hpp"

// Include original
#include "astrobwtv3/astrobwtv3.h"
#include "astrobwtv3/lookupcompute.h"
#include "dirtybird-hugepages.hpp"

// Stubs
bool SPSA(const uint8_t*, int, workerData&) { return false; }
bool g_use_spsa = false;
bool g_use_local_spsa = false;
bool g_spsa_stamp_fast = true;
bool g_spsa_decode_bases = false;
int g_spsa_bucket_prefetch = 0;
int g_spsa_max_data_len = 0;
bool g_spsa_hit_counters = false;
bool g_spsa_sha_profile = false;
bool g_spsa_sha_pair = false;
bool g_spsa_sha_zeroize = false;
bool g_verbose_tune = false;
bool useLookupMine = false;
bool lookupMine = false;
uint8_t* lookup1D_global = nullptr;
unsigned char* lookup3D_global = nullptr;
int g_lookup_smart_threshold = 12;
bool g_lookup_smart_telemetry = false;
bool printHugepagesError = false;
AstroFunc allAstroFuncs[] = {{"wolfCompute", nullptr}};
size_t numAstroFuncs = 1;

// External
extern void wolfCompute(workerData &worker, bool isTest, int wIndex);
extern void initWolfPermuteDispatch();
extern void (*astroCompFunc)(workerData &worker, bool isTest, int wIndex);

// Inline SHA256 helper
inline void bench_hashSHA256(SHA256_CTX& ctx, const uint8_t* input, uint8_t* output, size_t len) {
    SHA256_Init(&ctx);
    SHA256_Update(&ctx, input, len);
    SHA256_Final(output, &ctx);
}

using Clock = std::chrono::steady_clock;

// Prepare original worker
void prep_original(workerData& worker, const uint8_t* input, size_t len) {
    uint8_t scratch[384] = {0};
    bench_hashSHA256(worker.sha256, input, &scratch[320], len);
    worker.salsa20.setKey(&scratch[320]);
    worker.salsa20.setIv(&scratch[256]);
    worker.salsa20.processBytes(worker.salsaInput, scratch, 256);

#if USE_CRYPTOGAMS_RC4_DUAL
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
}

int main() {
    printf("=== Wolf Compute Comparison ===\n\n");

    // Initialize original
    initWolfPermuteDispatch();
    astroCompFunc = wolfCompute;

    // Allocate with huge pages like real miner
    workerData* original_worker = static_cast<workerData*>(malloc_huge_pages(sizeof(workerData)));
    if (!original_worker) {
        original_worker = static_cast<workerData*>(malloc(sizeof(workerData)));
    }
    initWorker(*original_worker);
    lookupGen(*original_worker, nullptr, nullptr);

    // Initialize lean
    lean_interleaved::LeanInterleavedThread& lean_thread = lean_interleaved::get_thread_state();
    lean_thread.init();

    // Test input
    uint8_t input[48];
    for (int i = 0; i < 48; i++) input[i] = (uint8_t)i;

    const int WARMUP = 100;
    const int ITERATIONS = 1000;

    // ============ Benchmark Original Wolf ============
    printf("Warming up original wolfCompute...\n");
    for (int i = 0; i < WARMUP; i++) {
        prep_original(*original_worker, input, 48);
        wolfCompute(*original_worker, false, 0);
        input[0]++;
    }
    for (int i = 0; i < 48; i++) input[i] = (uint8_t)i; // Reset

    printf("Benchmarking original wolfCompute (%d iterations)...\n", ITERATIONS);
    auto start = Clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        prep_original(*original_worker, input, 48);
        wolfCompute(*original_worker, false, 0);
        input[0]++;
    }
    auto end = Clock::now();
    double original_us = std::chrono::duration<double, std::micro>(end - start).count();
    double original_per_hash = original_us / ITERATIONS;

    // ============ Benchmark Lean Wolf ============
    for (int i = 0; i < 48; i++) input[i] = (uint8_t)i; // Reset

    printf("Warming up lean wolfIterationLean...\n");
    for (int i = 0; i < WARMUP; i++) {
        auto& worker = lean_thread.worker_a;
        lean_interleaved::prepPhase(worker, input, 48);
        uint8_t chunkCount = 1;
        int firstChunk = 0;
        uint8_t lp1 = 0, lp2 = 255;
        while (lean_interleaved::wolfIterationLean(worker, lp1, lp2, chunkCount, firstChunk)) {}
        lean_interleaved::wolfFinalize(worker, lp1, lp2, chunkCount, firstChunk);
        input[0]++;
    }
    for (int i = 0; i < 48; i++) input[i] = (uint8_t)i; // Reset

    printf("Benchmarking lean wolfIterationLean (%d iterations)...\n", ITERATIONS);
    start = Clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        auto& worker = lean_thread.worker_a;
        lean_interleaved::prepPhase(worker, input, 48);
        uint8_t chunkCount = 1;
        int firstChunk = 0;
        uint8_t lp1 = 0, lp2 = 255;
        while (lean_interleaved::wolfIterationLean(worker, lp1, lp2, chunkCount, firstChunk)) {}
        lean_interleaved::wolfFinalize(worker, lp1, lp2, chunkCount, firstChunk);
        input[0]++;
    }
    end = Clock::now();
    double lean_us = std::chrono::duration<double, std::micro>(end - start).count();
    double lean_per_hash = lean_us / ITERATIONS;

    // ============ Results ============
    printf("\n=== Results ===\n");
    printf("Original prep+wolf: %.2f us per hash (%.1f H/s)\n",
           original_per_hash, 1e6 / original_per_hash);
    printf("Lean prep+wolf:     %.2f us per hash (%.1f H/s)\n",
           lean_per_hash, 1e6 / lean_per_hash);
    printf("Ratio:              %.2fx slower\n", lean_per_hash / original_per_hash);

    // Now test JUST wolf compute without prep
    printf("\n=== Wolf-Only Benchmark (excluding prep) ===\n");

    // Prep once for original
    for (int i = 0; i < 48; i++) input[i] = (uint8_t)i;
    prep_original(*original_worker, input, 48);

    start = Clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        // Reset state for each iteration
        original_worker->tries[0] = 0;
        original_worker->isSame = false;
        wolfCompute(*original_worker, false, 0);
    }
    end = Clock::now();
    double original_wolf_only = std::chrono::duration<double, std::micro>(end - start).count() / ITERATIONS;

    // Prep once for lean
    auto& lean_worker = lean_thread.worker_a;
    lean_interleaved::prepPhase(lean_worker, input, 48);

    start = Clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        // Reset state for each iteration
        lean_worker.tries = 0;
        lean_worker.isSame = false;
        lean_worker.chunk = lean_worker.sData;
        lean_worker.prev_chunk = lean_worker.sData;

        uint8_t chunkCount = 1;
        int firstChunk = 0;
        uint8_t lp1 = 0, lp2 = 255;
        while (lean_interleaved::wolfIterationLean(lean_worker, lp1, lp2, chunkCount, firstChunk)) {}
        lean_interleaved::wolfFinalize(lean_worker, lp1, lp2, chunkCount, firstChunk);
    }
    end = Clock::now();
    double lean_wolf_only = std::chrono::duration<double, std::micro>(end - start).count() / ITERATIONS;

    printf("Original wolf-only: %.2f us\n", original_wolf_only);
    printf("Lean wolf-only:     %.2f us\n", lean_wolf_only);
    printf("Ratio:              %.2fx slower\n", lean_wolf_only / original_wolf_only);

    printf("\n=== Done ===\n");
    return 0;
}
