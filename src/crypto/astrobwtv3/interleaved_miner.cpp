/**
 * Two Miners Per Thread - Interleaved Execution Implementation
 */

#if defined(DIRTYBIRD_OS_ANDROID)

#include "interleaved_miner.hpp"

InterleavedMiner::InterleavedMiner()
    : worker_a_(nullptr), worker_b_(nullptr), initialized_(false) {}

InterleavedMiner::~InterleavedMiner() = default;

bool InterleavedMiner::initialize() {
    return false;
}

int InterleavedMiner::processInterleaved(
    const uint8_t* input_a, int len_a,
    const uint8_t* input_b, int len_b,
    uint8_t* hash_a, uint8_t* hash_b,
    int wIndex
) {
    (void)input_a;
    (void)len_a;
    (void)input_b;
    (void)len_b;
    (void)hash_a;
    (void)hash_b;
    (void)wIndex;
    return 0;
}

void InterleavedMiner::prepPhaseSingle(workerData& worker, const uint8_t* input, int inputLen) {
    (void)worker;
    (void)input;
    (void)inputLen;
}

void InterleavedMiner::wolfComputeInterleaved2(workerData& wa, workerData& wb, int wi) {
    (void)wa;
    (void)wb;
    (void)wi;
}

bool wolfComputeSingleIteration(workerData& worker, int wIndex, int iteration,
                                uint8_t& lp1, uint8_t& lp2,
                                uint8_t& chunkCount, int& firstChunk) {
    (void)worker;
    (void)wIndex;
    (void)iteration;
    (void)lp1;
    (void)lp2;
    (void)chunkCount;
    (void)firstChunk;
    return false;
}

void wolfComputeFinalize(workerData& worker, int wIndex,
                         uint8_t lp1, uint8_t lp2,
                         uint8_t chunkCount, int firstChunk) {
    (void)worker;
    (void)wIndex;
    (void)lp1;
    (void)lp2;
    (void)chunkCount;
    (void)firstChunk;
}

double benchmarkInterleaved(int numIterations) {
    (void)numIterations;
    return 0.0;
}

#else

#include "interleaved_miner.hpp"
#include "dirtybird-hugepages.hpp"
#include "astrobwtv3.h"
#include "astroworker.h"
#include "lookupcompute.h"
#include "lookup_tables.hpp"
#include "rc4_avx512.hpp"
#include "memory_optimized.hpp"
#include "fnv1a.h"
#include <xxhash64.h>
#include <highwayhash/sip_hash.h>
#include <immintrin.h>
#include <openssl/sha.h>
#include <openssl/rc4.h>
#include <cstring>
#include <algorithm>

#ifdef USE_DLUNA_RADIX_SA
  #include "dluna_radix_sa.h"
  #define SA_FUNCTION dluna_radix_sa::radix_sort_sa
#else
  #include "libsais.h"
  #define SA_FUNCTION(T, SA, n, bA, bB) libsais(T, SA, n, 0, nullptr)
#endif

// Global lookup tables and metadata from miner.cpp
extern uint8_t *lookup1D_global;
extern uint8_t g_is_branched[256];
extern uint8_t g_reg_idx[256];
extern uint8_t g_branched_idx[256];

void hashSHA256(SHA256_CTX &sha256, const byte *input, byte *digest, unsigned long inputSize);

InterleavedMiner::InterleavedMiner() 
    : initialized_(false), worker_a_(nullptr), worker_b_(nullptr) {}

InterleavedMiner::~InterleavedMiner() {
    if (worker_a_) free_huge_pages(worker_a_);
    if (worker_b_) free_huge_pages(worker_b_);
}

bool InterleavedMiner::initialize() {
    if (initialized_) return true;
    worker_a_ = static_cast<workerData*>(malloc_huge_pages(sizeof(workerData)));
    if (!worker_a_) worker_a_ = static_cast<workerData*>(std::malloc(sizeof(workerData)));
    worker_b_ = static_cast<workerData*>(malloc_huge_pages(sizeof(workerData)));
    if (!worker_b_) worker_b_ = static_cast<workerData*>(std::malloc(sizeof(workerData)));
    
    if (!worker_a_ || !worker_b_) return false;
    memset(worker_a_, 0, sizeof(workerData));
    memset(worker_b_, 0, sizeof(workerData));
    
    initWorker(*worker_a_);
    initWorker(*worker_b_);
    
    worker_a_->lucky = 0x1337;
    worker_b_->lucky = 0x1337;

    initialized_ = true;
    return true;
}

struct IState { uint8_t lp1, lp2, cc; int fc; bool active; };

namespace astro_branched_zOp {
    typedef void (*OpFunc)(workerData &, __m256i &, __m256i &, int);
    extern OpFunc branchCompute[512];
}

extern bool useLookupMine;

static inline bool wolfIter(workerData& w, int wi, IState& s) {
    uint16_t tries = ++w.tries[wi];
    uint64_t lhash = w.lhash;
    uint64_t prev_lhash = w.prev_lhash;
    uint32_t rs = prev_lhash ^ lhash ^ tries;
    
    byte op = static_cast<byte>(rs);
    uint8_t p1 = static_cast<byte>(rs >> 8);
    uint8_t p2 = static_cast<byte>(rs >> 16);
    if (p1 > p2) std::swap(p1, p2);
    if (p2 - p1 > 32) p2 = p1 + ((p2 - p1) & 0x1f);
    
    s.lp1 = std::min(s.lp1, p1); s.lp2 = std::max(s.lp2, p2);
    if (p1 < w.pos1 || p2 > w.pos2) w.isSame = false;
    w.pos1 = p1; w.pos2 = p2;

    byte* chunk = &w.sData[wi * ASTRO_SCRATCH_SIZE + (tries - 1) * 256];
    w.chunk = chunk;
    if (tries > 1) {
        byte* prev_chunk = &w.sData[wi * ASTRO_SCRATCH_SIZE + (tries - 2) * 256];
        w.prev_chunk = prev_chunk;
        memcpy(chunk, prev_chunk, 256);
    } else w.prev_chunk = chunk;

    if (op == 253) {
        for (int i = p1; i < p2; i++) {
            chunk[i] = rl8(chunk[i], 3);
            chunk[i] ^= rl8(chunk[i], 2);
            chunk[i] ^= w.prev_chunk[p2];
            chunk[i] = rl8(chunk[i], 3);
            prev_lhash += lhash;
            lhash = XXHash64::hash(chunk, p2, 0);
        }
    } else {
        if (op >= 254) {
            RC4_set_key(&w.key[wi], 256, w.prev_chunk);
        }

        if (useLookupMine && !g_is_branched[op]) {
            const uint8_t* lut = &lookup1D_global[static_cast<size_t>(g_reg_idx[op]) * 256];
            for (int i = p1; i < p2; i++) chunk[i] = lut[w.prev_chunk[i]];
            if (!op && ((p2 - p1) % 2 == 1)) {
                uint8_t t1 = chunk[p1], t2 = chunk[p2];
                chunk[p1] = reverse8(t2); chunk[p2] = reverse8(t1);
                w.isSame = false;
            }
        } else {
            __m256i data = _mm256_loadu_si256((__m256i*)&w.prev_chunk[p1]);
            __m256i old = data;
            astro_branched_zOp::branchCompute[op + (256 * w.isSame)](w, data, old, wi);
            if (!op && ((p2 - p1) % 2 == 1)) w.isSame = false;
        }
    }

    int A = (static_cast<int>(chunk[p1]) - static_cast<int>(chunk[p2]));
    A = (256 + (A % 256)) % 256;
    w.A = A;

    {
        const int hash_sel = (A < 0x30) + (A < 0x20) + (A < 0x10);
        if (hash_sel > 0) {
            if (hash_sel >= 3) {
                prev_lhash += lhash; lhash = XXHash64::hash(chunk, p2, 0);
            }
            if (hash_sel >= 2) {
                prev_lhash += lhash; lhash = hash_64_fnv1a(chunk, p2);
            }
            prev_lhash += lhash;
            HH_ALIGNAS(16) const highwayhash::HH_U64 sk[2] = {(uint64_t)tries, prev_lhash};
            lhash = highwayhash::SipHash(sk, (char*)chunk, p2);
        }
    }
    w.lhash = lhash; w.prev_lhash = prev_lhash;

    if (A <= 0x40) {
        RC4(&w.key[wi], 256, chunk, chunk);
        w.isSame = false;
        uint8_t pP1 = s.lp1, pP2 = s.lp2;
        if (p1 == p2) { pP1 = 255; pP2 = 255; }
        if (255 - pP2 < 4) pP2 = 255;
        if (pP1 < 4) pP1 = 0;
        if (pP1 == 255) pP1 = 0;
        w.astroTemplate[w.templateIdx] = templateMarker{(uint8_t)(s.cc > 1 ? pP1 : 0), (uint8_t)(s.cc > 1 ? pP2 : 255), 0, 0, (uint16_t)((s.fc << 7) | s.cc)};
        w.templateIdx += (tries > 1); s.fc = tries - 1; s.lp1 = 255; s.lp2 = 0; s.cc = 1;
    } else s.cc++;

    chunk[255] ^= chunk[p1] ^ chunk[p2];
    return !(tries > 260 + 16 || (chunk[255] >= 0xf0 && tries > 260));
}

void InterleavedMiner::prepPhaseSingle(workerData& worker, const uint8_t* input, int inputLen) {
    uint8_t scratch[384] = {0};
    memset(worker.sData, 0, ASTRO_SCRATCH_SIZE);
    hashSHA256(worker.sha256, input, &scratch[320], inputLen);

#if USE_SIMD_SALSA20
    salsa20_simd_process(&scratch[320], &scratch[256], worker.salsaInput, scratch, 256);
#else
    worker.salsa20.setKey(&scratch[320]);
    worker.salsa20.setIv(&scratch[256]);
    worker.salsa20.processBytes(worker.salsaInput, scratch, 256);
#endif

    RC4_set_key(&worker.key[0], 256, scratch);
    RC4(&worker.key[0], 256, scratch, scratch);

    worker.lhash = hash_64_fnv1a_256(scratch);
    worker.prev_lhash = worker.lhash;
    worker.tries[0] = 0;
    worker.isSame = false;
    std::memcpy(worker.sData, scratch, 256);
}

int InterleavedMiner::processInterleaved(
    const uint8_t* input_a, int len_a,
    const uint8_t* input_b, int len_b,
    uint8_t* hash_a, uint8_t* hash_b,
    int wi) 
{
    prepPhaseSingle(*worker_a_, input_a, len_a);
    prepPhaseSingle(*worker_b_, input_b, len_b);
    
    IState sa = {0, 255, 1, 0, true}, sb = {0, 255, 1, 0, true};
    worker_a_->tries[wi] = 0; worker_b_->tries[wi] = 0;
    worker_a_->templateIdx = 0; worker_b_->templateIdx = 0;

    for (int it = 0; it < 278; ++it) {
        if (sa.active) sa.active = wolfIter(*worker_a_, wi, sa);
        if (sb.active) sb.active = wolfIter(*worker_b_, wi, sb);
        if (!sa.active && !sb.active) break;
    }

    auto fin = [&](workerData& w, uint8_t* h, IState& s) {
        if (s.cc > 0) {
            if (255 - s.lp2 < 8) s.lp2 = 255;
            if (s.lp1 < 8) s.lp1 = 0;
            w.astroTemplate[w.templateIdx++] = templateMarker{(uint8_t)(s.cc > 1 ? s.lp1 : 0), (uint8_t)(s.cc > 1 ? s.lp2 : 255), 0, 0, (uint16_t)((s.fc << 7) | s.cc)};
        }
        w.data_len = (w.tries[wi] - 4) * 256 + (((uint64_t)w.chunk[253] << 8 | w.chunk[254]) & 0x3ff);
        memset(w.sData + w.data_len, 0, 16);
        memset(w.sa, 0, sizeof(w.sa));
        SA_FUNCTION(w.sData, w.sa, w.data_len, nullptr, nullptr);
        hashSHA256(w.sha256, reinterpret_cast<byte*>(w.sa), h, w.data_len * 4);
    };
    fin(*worker_a_, hash_a, sa); fin(*worker_b_, hash_b, sb);
    return 0;
}

void InterleavedMiner::wolfComputeInterleaved2(workerData& wa, workerData& wb, int wi) {}

double benchmarkInterleaved(int numIterations) {
    (void)numIterations;
    return 0.0;
}

#endif
