/**
 * Benchmark: Interleaved (Two Miners Per Thread) vs Standard
 */

#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>
#include <thread>

#include <astrobwtv3/astrobwtv3.h>
#include <astrobwtv3/lookupcompute.h>
#include <astrobwtv3/interleaved_miner.hpp>
#include <dirtybird-hugepages.hpp>
#include <lookup_mode.hpp>
#include <lookup_tables.hpp>

#ifdef USE_DLUNA_RADIX_SA
  #include <astrobwtv3/dluna_radix_sa.h>
  #define SA_FUNCTION_LOCAL dluna_radix_sa::radix_sort_sa
#else
  #include <libsais.h>
  #define SA_FUNCTION_LOCAL(T, SA, n, bA, bB) libsais(T, SA, n, 0, nullptr)
#endif

// Stub definitions
bool printHugepagesError = true;
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

AstroFunc allAstroFuncs[] = { {"wolfCompute", wolfCompute} };
size_t numAstroFuncs = 1;

bool SPSA(const uint8_t* data, int dataSize, workerData& ctx) { return false; }

extern void detectAVX512();
extern void initWolfPermuteDispatch();
extern void (*astroCompFunc)(workerData &worker, bool isTest, int wIndex);

struct BenchStats { double mean, stddev, min, max, p5, p95; int samples; };

BenchStats computeStats(std::vector<double>& data) {
    BenchStats stats; stats.samples = data.size();
    if (data.empty()) return stats;
    std::sort(data.begin(), data.end());
    stats.min = data.front(); stats.max = data.back();
    stats.p5 = data[data.size() * 0.05]; stats.p95 = data[data.size() * 0.95];
    stats.mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    double sq_sum = 0;
    for (double v : data) sq_sum += (v - stats.mean) * (v - stats.mean);
    stats.stddev = std::sqrt(sq_sum / data.size());
    return stats;
}

void printStats(const char* name, const BenchStats& stats) {
    printf("  %s:\n    Mean: %.2f H/s\n    StdDev: %.2f (CV: %.1f%%)\n    Min: %.2f H/s\n    Max: %.2f H/s\n", 
           name, stats.mean, stats.stddev, 100.0 * stats.stddev / stats.mean, stats.min, stats.max);
}

int main(int argc, char* argv[]) {
    detectAVX512();
    initWolfLUT();
    initWolfPermuteDispatch();
    lookup1D_global = (uint8_t*)std::malloc(256 * 256);
    lookup_tables::generateTables(lookup1D_global, nullptr);
    astroCompFunc = wolfCompute;

    std::random_device rd; std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> dist(0, 255);
    uint8_t inputs[16 * 48];
    for (int i = 0; i < 16 * 48; ++i) inputs[i] = dist(gen);

    printf("Verifying correctness (Fresh Workers)...\n");
    {
        uint8_t hash_std[32], hash_int_a[32], hash_int_b[32];
        
        workerData* w_std = static_cast<workerData*>(std::malloc(sizeof(workerData)));
        memset(w_std, 0, sizeof(workerData));
        initWorker(*w_std);
        lookupGen(*w_std, nullptr, nullptr);
        AstroBWTv3(inputs, 48, hash_std, *w_std, false);
        uint32_t len_std = w_std->data_len;

        InterleavedMiner interleavedMiner;
        interleavedMiner.initialize();
        memset(interleavedMiner.getWorkerA(), 0, sizeof(workerData));
        initWorker(*interleavedMiner.getWorkerA());
        lookupGen(*interleavedMiner.getWorkerA(), nullptr, nullptr);
        
        interleavedMiner.processInterleaved(inputs, 48, inputs + 48, 48, hash_int_a, hash_int_b, 0);
        uint32_t len_int = interleavedMiner.getWorkerA()->data_len;

        printf("  [Hash A] Standard: len=%u, Interlev: len=%u\n", len_std, len_int);
        
        if (len_std != len_int) {
            printf("CRITICAL ERROR: data_len mismatch!\n");
        } else {
            if (std::memcmp(w_std->sData, interleavedMiner.getWorkerA()->sData, len_std) != 0) {
                printf("CRITICAL ERROR: sData mismatch!\n");
            } else {
                printf("  sData matches 100%% up to len_std.\n");
                if (std::memcmp(w_std->sa, interleavedMiner.getWorkerA()->sa, len_std * 4) != 0) {
                    printf("CRITICAL ERROR: SA mismatch!\n");
                } else {
                    printf("  SA matches 100%%.\n");
                    if (std::memcmp(hash_std, hash_int_a, 32) != 0) {
                        printf("CRITICAL ERROR: Final Hash mismatch!\n");
                    } else {
                        printf("  SUCCESS: Interleaved path matches Standard path.\n");
                    }
                }
            }
        }
        std::free(w_std);
    }

    // Benchmark loop
    std::vector<double> s_rates, i_rates;
    workerData* bench_worker = static_cast<workerData*>(malloc_huge_pages(sizeof(workerData)));
    if (!bench_worker) bench_worker = static_cast<workerData*>(std::malloc(sizeof(workerData)));
    
    InterleavedMiner benchMiner;
    benchMiner.initialize();

    for (int run = 0; run < 3; ++run) {
        auto start = std::chrono::steady_clock::now();
        auto end = start + std::chrono::seconds(5);
        int h = 0;
        while (std::chrono::steady_clock::now() < end) {
            // Zero worker to eliminate ghost data effects on performance/determinism
            memset(bench_worker, 0, sizeof(workerData));
            initWorker(*bench_worker);
            uint8_t out[32]; AstroBWTv3(inputs + (h % 16) * 48, 48, out, *bench_worker, false);
            h++;
        }
        s_rates.push_back(h / 5.0);
        
        start = std::chrono::steady_clock::now();
        end = start + std::chrono::seconds(5);
        int p = 0;
        while (std::chrono::steady_clock::now() < end) {
            memset(benchMiner.getWorkerA(), 0, sizeof(workerData));
            initWorker(*benchMiner.getWorkerA());
            memset(benchMiner.getWorkerB(), 0, sizeof(workerData));
            initWorker(*benchMiner.getWorkerB());
            uint8_t out_a[32], out_b[32];
            benchMiner.processInterleaved(inputs, 48, inputs + 48, 48, out_a, out_b, 0);
            p++;
        }
        i_rates.push_back((p * 2) / 5.0);
    }
    printStats("Standard", computeStats(s_rates));
    printStats("Interleaved", computeStats(i_rates));
    return 0;
}
