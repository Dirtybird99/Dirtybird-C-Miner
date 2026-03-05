#pragma once
// radix_sa.h - DeroHE-style two-pass radix sort for suffix array construction
//
// Port of the official DERO Go miner's sort_indices2 algorithm.
// Uses 20-bit radix keys (2 passes of 10-bit counting sort) + insertion sort
// for ties. Produces correct suffix arrays for random data (post-Salsa20+RC4)
// with overwhelming probability.
//
// Thermal advantage: Primarily sequential memory access (counting sort passes)
// vs SA-IS's irregular random-access induction steps. Lower cache pressure =
// less power = less thermal throttling.

#include <cstdint>
#include <cstring>

namespace radix_sa {

static constexpr uint64_t RADIX_BITS = 10;
static constexpr uint64_t RADIX_SIZE = 1ULL << RADIX_BITS;  // 1024
static constexpr uint64_t POS_MASK   = (1ULL << 21) - 1;    // 0x1FFFFF
static constexpr uint64_t KEY_MASK   = ~POS_MASK;            // upper 43 bits

// Big-endian uint64 read (matches Go's BigEndian_Uint64)
static inline uint64_t be_u64(const uint8_t* b) {
    return (uint64_t(b[0]) << 56) | (uint64_t(b[1]) << 48) |
           (uint64_t(b[2]) << 40) | (uint64_t(b[3]) << 32) |
           (uint64_t(b[4]) << 24) | (uint64_t(b[5]) << 16) |
           (uint64_t(b[6]) <<  8) |  uint64_t(b[7]);
}

// Compare two packed indices: upper 43 bits first, then 8 bytes at position+5
static inline bool smaller(const uint8_t* input, uint64_t a, uint64_t b) {
    uint64_t va = a >> 21;
    uint64_t vb = b >> 21;
    if (va < vb) return true;
    if (va > vb) return false;
    // Tiebreak: compare bytes 5..12 of each suffix
    uint64_t da = be_u64(input + (a & POS_MASK) + 5);
    uint64_t db = be_u64(input + (b & POS_MASK) + 5);
    return da < db;
}

// Build suffix array using DeroHE-style 2-pass radix sort
// T: input data, must have >=13 bytes readable padding after T[n-1]
// SA: output suffix array, SA[i] = position of i-th smallest suffix
// n: input length
// bA, bB: unused (compatibility with divsufsort/libsais signature)
static inline int radix_sort_sa(const uint8_t* T, int32_t* SA, int32_t n,
                                int* /*bA*/ = nullptr, int* /*bB*/ = nullptr) {
    // Thread-local temp arrays (~1.1MB total, allocated once per thread)
    static constexpr int MAX_N = 71000;
    thread_local uint64_t indices[MAX_N];
    thread_local uint64_t tmp_indices[MAX_N];

    // Zero padding after data for deterministic reads near end
    // (T points into worker.sData which has 64 bytes padding, but may be uninitialized)
    // We can't modify T (const), but the caller should ensure padding is zeroed.

    uint32_t counters[2][RADIX_SIZE];
    memset(counters, 0, sizeof(counters));

    const int N = n;

    // Phase 1: Histogram (count sort keys for both passes)
    // Optimization: process 3 positions per 8-byte read using bit shifts
    const int loop3 = (N / 3) * 3;
    for (int i = 0; i < loop3; i += 3) {
        uint64_t k0 = be_u64(T + i);
        counters[0][(k0 >> (64 - RADIX_BITS * 2)) & (RADIX_SIZE - 1)]++;
        counters[1][ k0 >> (64 - RADIX_BITS)]++;
        uint64_t k1 = k0 << 8;
        counters[0][(k1 >> (64 - RADIX_BITS * 2)) & (RADIX_SIZE - 1)]++;
        counters[1][ k1 >> (64 - RADIX_BITS)]++;
        uint64_t k2 = k0 << 16;
        counters[0][(k2 >> (64 - RADIX_BITS * 2)) & (RADIX_SIZE - 1)]++;
        counters[1][ k2 >> (64 - RADIX_BITS)]++;
    }
    for (int i = loop3; i < N; i++) {
        uint64_t k = be_u64(T + i);
        counters[0][(k >> (64 - RADIX_BITS * 2)) & (RADIX_SIZE - 1)]++;
        counters[1][ k >> (64 - RADIX_BITS)]++;
    }

    // Phase 2: Prefix sums → end-of-bucket positions (exclusive, Go-style)
    uint32_t prev0 = counters[0][0], prev1 = counters[1][0];
    counters[0][0] = prev0 - 1;
    counters[1][0] = prev1 - 1;
    for (uint64_t i = 1; i < RADIX_SIZE; i++) {
        uint32_t cur0 = counters[0][i] + prev0;
        uint32_t cur1 = counters[1][i] + prev1;
        counters[0][i] = cur0 - 1;
        counters[1][i] = cur1 - 1;
        prev0 = cur0;
        prev1 = cur1;
    }

    // Phase 3: Scatter pass 1 - sort by bits 10-19 (LSD digit)
    // Pack: upper 43 bits of key + lower 21 bits = position
    for (int i = N - 1; i >= 0; i--) {
        uint64_t k = be_u64(T + i);
        uint32_t bucket = (k >> (64 - RADIX_BITS * 2)) & (RADIX_SIZE - 1);
        uint32_t pos = counters[0][bucket];
        counters[0][bucket]--;
        tmp_indices[pos] = (k & KEY_MASK) | (uint64_t)i;
    }

    // Phase 4: Scatter pass 2 - sort by upper 10 bits (MSD digit)
    for (int i = N - 1; i >= 0; i--) {
        uint64_t data = tmp_indices[i];
        uint32_t bucket = (uint32_t)(data >> (64 - RADIX_BITS));
        uint32_t pos = counters[1][bucket];
        counters[1][bucket]--;
        indices[pos] = data;
    }

    // Phase 5: Insertion sort to fix remaining out-of-order elements
    // For random data (~70K elements, 20-bit keys), only ~0.3% of elements
    // need correction, making this phase very fast in practice.
    uint64_t prev_t = indices[0];
    for (int i = 1; i < N; i++) {
        uint64_t t = indices[i];
        if (smaller(T, t, prev_t)) {
            uint64_t t2 = prev_t;
            int j = i - 1;
            for (;;) {
                indices[j + 1] = prev_t;
                j--;
                if (j < 0) break;
                prev_t = indices[j];
                if (!smaller(T, t, prev_t)) break;
            }
            indices[j + 1] = t;
            t = t2;
        }
        prev_t = t;
    }

    // Phase 6: Extract position indices into SA array
    for (int i = 0; i < N; i++) {
        SA[i] = (int32_t)(indices[i] & POS_MASK);
    }

    return 0;
}

} // namespace radix_sa
