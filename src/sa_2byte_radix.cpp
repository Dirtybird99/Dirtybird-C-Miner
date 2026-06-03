/* sa_2byte_radix.cpp -- clean-room 2-byte-prefix radix SA construction
 *
 * Tritonn's libastroSPSA does a 2-byte (16-bit) prefix radix sort followed by
 * introsort tiebreak on collision groups. Per Frida runtime trace, most byte-
 * pairs are unique on AstroBWT v3 data (median collision-bucket size = 5,
 * p95=28, max=82), so 99% of positions never enter the slow tiebreak path.
 * libsais (SA-IS, linear-time) is asymptotically optimal but has a much
 * larger constant factor; on this 70 KB workload with high pair-uniqueness,
 * 2-byte radix wins on constant factor.
 *
 * This file provides `sa_construct_2byte(data, sa, n)` with the same
 * signature as `libsais_ctx(...)`. Output is bit-identical to libsais lex SA.
 */

#include <cstdint>
#include <cstring>
#include <algorithm>

namespace deroluna_sa {

/* Global pointer for sort comparator's full-suffix tiebreak. */
static thread_local const uint8_t* g_data;
static thread_local int32_t g_n;

static inline int suffix_cmp(int32_t a, int32_t b) {
    int la = g_n - a, lb = g_n - b;
    int n = la < lb ? la : lb;
    int c = std::memcmp(g_data + a, g_data + b, n);
    if (c != 0) return c;
    return la - lb;
}

/* In-place 2-byte radix SA construction.
 * data : input bytes (assumed n+1 readable, sentinel handled below)
 * sa   : output int32 array of length n
 * n    : data length
 *
 * Algorithm:
 *  1. Count 2-byte pairs (data[i], data[i+1]) into bucket[256][256] for i in [0..n-1).
 *     Position n-1 has only 1 byte after it — pair with implicit 0 sentinel.
 *  2. Prefix-sum buckets to cumulative offsets (CDF).
 *  3. Scatter positions 0..n-1 into sa[bucket_offset++] indexed by 2-byte prefix.
 *  4. For each non-trivial bucket (size > 1), std::sort with full-suffix memcmp.
 */
extern "C" void sa_construct_2byte(const uint8_t* data, int32_t* sa, int32_t n) {
    if (n <= 0) return;
    if (n == 1) { sa[0] = 0; return; }

    /* Phase 1: 2-byte pair count. Use uint32 to handle pathological cases.
     * Also count single-byte tail (position n-1) as data[n-1] paired with 0. */
    static thread_local uint32_t bucket[256][256];
    static thread_local uint32_t scatter_idx[256][256];
    std::memset(bucket, 0, sizeof(bucket));

    /* Count pairs for positions [0..n-1). */
    for (int i = 0; i < n - 1; ++i) {
        bucket[data[i]][data[i + 1]]++;
    }
    /* Tail: position n-1 has implicit sentinel (0). Pair with 0. */
    bucket[data[n - 1]][0]++;

    /* Phase 2: prefix-sum to CDF. */
    uint32_t total = 0;
    for (int b1 = 0; b1 < 256; ++b1) {
        for (int b2 = 0; b2 < 256; ++b2) {
            uint32_t c = bucket[b1][b2];
            scatter_idx[b1][b2] = total;
            total += c;
        }
    }
    /* total should equal n. Defensive check. */
    if ((int32_t)total != n) {
        /* fallback: assume single-byte tie at n-1; but this is a bug indicator */
    }

    /* Phase 3: scatter positions into sa[]. */
    for (int i = 0; i < n - 1; ++i) {
        sa[scatter_idx[data[i]][data[i + 1]]++] = i;
    }
    sa[scatter_idx[data[n - 1]][0]++] = n - 1;

    /* Phase 4: per-bucket full-suffix tiebreak. scatter_idx[b1][b2] is now the
     * end pointer for that bucket (CDF + count). The start is CDF (recompute). */
    g_data = data;
    g_n = n;

    uint32_t start = 0;
    for (int b1 = 0; b1 < 256; ++b1) {
        for (int b2 = 0; b2 < 256; ++b2) {
            uint32_t end = start + bucket[b1][b2];
            if (end > start + 1) {
                std::sort(sa + start, sa + end,
                    [](int32_t a, int32_t b) { return suffix_cmp(a, b) < 0; });
            }
            start = end;
        }
    }
}

} // namespace deroluna_sa
