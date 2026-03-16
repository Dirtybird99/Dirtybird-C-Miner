/**
 * Benchmark for 32-way vertical SIMD wolfCompute.
 */

#include <iostream>
#include <chrono>
#include <vector>
#include <cstring>
#include <random>
#include <astrobwtv3/astrobwtv3.h>
#include <astroworker.h>
#include <astrobwtv3/wolf_vsimd.hpp>
#include <astrobwtv3/lookupcompute.h>

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
extern void (*astroCompFunc)(workerData &worker, bool isTest, int wIndex);


int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> dist(0, 255);

    uint8_t inputs[32][48];
    uint8_t outputs[32][32];
    for (int i=0; i<32; i++) {
        for (int j=0; j<48; j++) inputs[i][j] = dist(gen);
    }

    auto t0 = std::chrono::steady_clock::now();
    int hashes = 0;
    for (int i=0; i<100; i++) {
        wolf_vsimd::AstroBWTv3_vsimd(inputs, 48, outputs);
        hashes += 32;
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
    printf("Full AstroBWTv3 VSIMD %d hashes: %.2f ms\n", hashes, ms);
    printf("Speed: %.2f H/s\n", (hashes / (ms / 1000.0)));

    printf("\nMeasuring just wolfCompute_Batch32...\n");
    wolf_vsimd::Batch32* batch_ptr = new wolf_vsimd::Batch32();
    wolf_vsimd::Batch32& batch = *batch_ptr;
    memset(&batch, 0, sizeof(batch)); // Zero out POD members
    t0 = std::chrono::steady_clock::now();
    hashes = 0;
    for (int i=0; i<100; i++) {
        for(int m=0; m<32; m++) {
            batch.active[m] = true;
            batch.active_mask |= (1u << m);
            batch.tries[m] = 0;
        }
        wolf_vsimd::wolfCompute_Batch32(batch);
        hashes += 32;
    }
    t1 = std::chrono::steady_clock::now();
    ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
    printf("Just wolfCompute_Batch32 %d hashes: %.2f ms\n", hashes, ms);
    printf("Speed: %.2f H/s\n", (hashes / (ms / 1000.0)));

    return 0;
}
