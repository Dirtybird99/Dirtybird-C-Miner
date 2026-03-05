#pragma once
// bucket_sa.h - Fast suffix array for random byte inputs (AstroBWT-specific)
//
// Exploits uniform byte distribution from Salsa20+RC4 encryption:
// - 2-byte prefix counting sort into 65536 buckets (256KB working set)
// - For n=70000, ~34% of elements in single-element buckets (trivially sorted)
// - Remaining ~66% in 2-3 element buckets: resolved by short comparisons
// - For random data, byte comparisons resolve ties in ~3 bytes on average
//
// Memory: 256KB thread-local count array + 280KB temp SA = 536KB per thread
// Total for 20 threads: ~10.5MB (fits in 30MB L3)

#include <cstdint>
#include <cstring>
#include <algorithm>

namespace bucket_sa {

static constexpr int MAX_N = 71000;
static constexpr int BUCKETS = 65536;  // 256 * 256

// Compare suffixes starting at positions a and b in text T of length n.
// For random data, this resolves in ~3 bytes on average.
static inline bool suffix_less(const uint8_t* T, int32_t n, int32_t a, int32_t b) {
    // Compare from the start (the 2-byte prefix is already equal within a bucket)
    // We start from offset+2 since the bucket sort already ordered by first 2 bytes
    int32_t pa = a + 2;
    int32_t pb = b + 2;
    int32_t end = n;

    // Fast path: compare 8 bytes at a time
    while (pa + 8 <= end && pb + 8 <= end) {
        uint64_t va = 0, vb = 0;
        memcpy(&va, T + pa, 8);
        memcpy(&vb, T + pb, 8);
        if (va != vb) {
            // Swap to big-endian for correct lexicographic comparison
            va = __builtin_bswap64(va);
            vb = __builtin_bswap64(vb);
            return va < vb;
        }
        pa += 8;
        pb += 8;
    }

    // Byte-by-byte fallback for remaining bytes
    while (pa < end && pb < end) {
        if (T[pa] != T[pb]) return T[pa] < T[pb];
        pa++;
        pb++;
    }
    // Shorter suffix is "less" (sentinel convention)
    return (end - a) < (end - b);  // equivalent to: a > b
}

// Main SA function: 2-byte prefix counting sort + insertion sort for ties
// API compatible with SA_FUNCTION(T, SA, n, bA, bB)
static inline int bucket_sort_sa(const uint8_t* T, int32_t* SA, int32_t n,
                                  int* /*bA*/ = nullptr, int* /*bB*/ = nullptr) {
    if (n <= 0) return 0;
    if (n == 1) { SA[0] = 0; return 0; }

    // Thread-local arrays to avoid allocation
    thread_local uint32_t counts[BUCKETS];
    thread_local uint32_t offsets[BUCKETS];

    // Phase 1: Histogram of 2-byte prefixes
    memset(counts, 0, sizeof(counts));

    // Process all positions except the last one (which has only 1-byte suffix)
    for (int32_t i = 0; i < n - 1; i++) {
        uint32_t key = (uint32_t(T[i]) << 8) | T[i + 1];
        counts[key]++;
    }
    // Last position: use T[n-1] as high byte, 0 as sentinel low byte
    counts[uint32_t(T[n - 1]) << 8]++;

    // Phase 2: Exclusive prefix sum -> bucket start offsets
    uint32_t sum = 0;
    for (int i = 0; i < BUCKETS; i++) {
        offsets[i] = sum;
        sum += counts[i];
    }
    // Save bucket boundaries for the sort phase
    // offsets[i] will be modified during scatter, so copy start positions
    thread_local uint32_t bucket_start[BUCKETS];
    memcpy(bucket_start, offsets, sizeof(offsets));

    // Phase 3: Scatter positions into SA by their 2-byte prefix
    for (int32_t i = 0; i < n - 1; i++) {
        uint32_t key = (uint32_t(T[i]) << 8) | T[i + 1];
        SA[offsets[key]++] = i;
    }
    SA[offsets[uint32_t(T[n - 1]) << 8]++] = n - 1;

    // Phase 4: Sort within each bucket that has >1 element
    // For random data, average bucket size is ~1.07, so most buckets are trivial.
    // Multi-element buckets are sorted by suffix comparison starting at offset+2.
    uint32_t pos = 0;
    for (int b = 0; b < BUCKETS; b++) {
        uint32_t cnt = counts[b];
        if (cnt <= 1) {
            pos += cnt;
            continue;
        }

        uint32_t start = bucket_start[b];

        if (cnt == 2) {
            // Optimize the common case: exactly 2 elements
            if (suffix_less(T, n, SA[start + 1], SA[start])) {
                int32_t tmp = SA[start];
                SA[start] = SA[start + 1];
                SA[start + 1] = tmp;
            }
        } else if (cnt <= 8) {
            // Insertion sort for small buckets (most common: 3-5 elements)
            for (uint32_t i = start + 1; i < start + cnt; i++) {
                int32_t key_val = SA[i];
                int32_t j = (int32_t)i - 1;
                while (j >= (int32_t)start && suffix_less(T, n, key_val, SA[j])) {
                    SA[j + 1] = SA[j];
                    j--;
                }
                SA[j + 1] = key_val;
            }
        } else {
            // std::sort for rare large buckets (>8 elements)
            std::sort(SA + start, SA + start + cnt,
                [T, n](int32_t a, int32_t b) {
                    return suffix_less(T, n, a, b);
                });
        }

        pos += cnt;
    }

    return 0;
}

} // namespace bucket_sa
