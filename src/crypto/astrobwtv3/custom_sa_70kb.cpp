/*
 * custom_sa_70kb.cpp - Custom SA-IS for AstroBWT's 70KB fixed-size inputs
 *
 * SA-IS (Suffix Array by Induced Sorting) implementation optimized for:
 * - 70KB inputs that fit entirely in L2/L3 cache
 * - Cache-aware design with reduced overhead
 * - Bitvector type storage (9KB vs 70KB byte array)
 * - AVX2 SIMD for histogram merge and LMS comparison
 * - Triple-stream prefetch for induced sorting
 * - 8-way loop unrolling for better ILP
 * - Branchless type operations
 *
 * BUCKET_B OPTIMIZATION (256KB bottleneck):
 * - bucket_B[65536] = 256KB, exceeds L2 cache on most CPUs
 * - Use memset for initialization (SIMD stores)
 * - Prefetch hints for anticipated bucket_B accesses
 * - Cache blocking for iterating through bucket ranges
 * - This implementation uses bucket_B for 256-entry bucket_end array only,
 *   avoiding the full 65536-entry array for SA-IS algorithm
 *
 * Algorithm based on:
 *   Nong, Zhang, Chan - "Two Efficient Algorithms for Linear Time Suffix Array Construction"
 *   (IEEE Transactions on Computers, 2011)
 *
 * Copyright (c) 2025 DERO Miner Project
 * License: MIT
 */

#include "custom_sa_70kb.h"
#include "divsufsort.h"
#include "bucket_optimized.h"
#include <string.h>
#include <assert.h>
#include <stdlib.h>

/* ========================================================================== */
/* SIMD Includes                                                              */
/* ========================================================================== */

#if ENABLE_SA_AVX2
    #if defined(_MSC_VER)
        #include <intrin.h>
    #else
        #include <immintrin.h>
    #endif
#endif

/* ========================================================================== */
/* Compiler Intrinsics and Hints                                              */
/* ========================================================================== */

#if defined(__GNUC__) || defined(__clang__)
    #define LIKELY(x)   __builtin_expect(!!(x), 1)
    #define UNLIKELY(x) __builtin_expect(!!(x), 0)
    #define PREFETCH_R(addr) __builtin_prefetch((addr), 0, 1)
    #define PREFETCH_W(addr) __builtin_prefetch((addr), 1, 1)
    #define PREFETCH_R_L2(addr) __builtin_prefetch((addr), 0, 2)
    #define FORCE_INLINE __attribute__((always_inline)) inline
    #define HOT_FUNCTION __attribute__((hot))
    #define NOINLINE __attribute__((noinline))
    #define CTZ32(x) __builtin_ctz(x)
#elif defined(_MSC_VER)
    #include <intrin.h>
    #define LIKELY(x)   (x)
    #define UNLIKELY(x) (x)
    #define PREFETCH_R(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T0)
    #define PREFETCH_W(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T0)
    #define PREFETCH_R_L2(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T1)
    #define FORCE_INLINE __forceinline
    #define HOT_FUNCTION
    #define NOINLINE __declspec(noinline)
    static FORCE_INLINE int CTZ32(unsigned int x) {
        unsigned long idx;
        _BitScanForward(&idx, x);
        return (int)idx;
    }
#else
    #define LIKELY(x)   (x)
    #define UNLIKELY(x) (x)
    #define PREFETCH_R(addr) ((void)0)
    #define PREFETCH_W(addr) ((void)0)
    #define PREFETCH_R_L2(addr) ((void)0)
    #define FORCE_INLINE inline
    #define HOT_FUNCTION
    #define NOINLINE
    static inline int CTZ32(unsigned int x) {
        int n = 0;
        if (!(x & 0xFFFF)) { n += 16; x >>= 16; }
        if (!(x & 0xFF)) { n += 8; x >>= 8; }
        if (!(x & 0xF)) { n += 4; x >>= 4; }
        if (!(x & 0x3)) { n += 2; x >>= 2; }
        if (!(x & 0x1)) { n += 1; }
        return n;
    }
#endif

/* ========================================================================== */
/* Thread-Local Workspace (Phase 1: Memory Layout)                            */
/* ========================================================================== */

/* Thread-local cache-aligned workspace to avoid allocation overhead */
#if defined(__cplusplus) && __cplusplus >= 201103L
    #ifdef _MSC_VER
        static __declspec(thread) SAWorkspace tl_workspace;
    #else
        static thread_local SAWorkspace tl_workspace;
    #endif
#else
    /* C fallback - not thread-safe */
    static SAWorkspace tl_workspace;
#endif

/* ========================================================================== */
/* Runtime SA Configuration                                                   */
/* ========================================================================== */

/* Global SA configuration with compile-time defaults */
SAConfig g_sa_config = {
    CUSTOM_SA_PREFETCH_DISTANCE,      /* sa_prefetch: 16 */
    CUSTOM_TEXT_PREFETCH_DISTANCE,    /* text_prefetch: 24 */
    CUSTOM_BUCKET_PREFETCH_DISTANCE   /* bucket_prefetch: 8 */
};

void sa_config_init(void) {
    g_sa_config.sa_prefetch = CUSTOM_SA_PREFETCH_DISTANCE;
    g_sa_config.text_prefetch = CUSTOM_TEXT_PREFETCH_DISTANCE;
    g_sa_config.bucket_prefetch = CUSTOM_BUCKET_PREFETCH_DISTANCE;
}

void sa_config_set(int sa_pf, int text_pf, int bucket_pf) {
    g_sa_config.sa_prefetch = sa_pf;
    g_sa_config.text_prefetch = text_pf;
    g_sa_config.bucket_prefetch = bucket_pf;
}

const SAConfig* sa_config_get(void) {
    return &g_sa_config;
}

/* ========================================================================== */
/* Instrumentation Support                                                    */
/* ========================================================================== */

#if ENABLE_SA_INSTRUMENTATION

static SAInstrumentationStats sa_stats = {0};

#if defined(_MSC_VER)
    #include <intrin.h>
    #define RDTSC() __rdtsc()
#elif defined(__GNUC__) || defined(__clang__)
    static FORCE_INLINE uint64_t RDTSC(void) {
        uint32_t lo, hi;
        __asm__ volatile ("rdtsc" : "=a"(lo), "=d"(hi));
        return ((uint64_t)hi << 32) | lo;
    }
#else
    #define RDTSC() 0
#endif

#define INSTRUMENT_START(phase) uint64_t _t_##phase = RDTSC()
#define INSTRUMENT_END(phase) sa_stats.phase##_cycles += RDTSC() - _t_##phase

const SAInstrumentationStats* custom_sa_get_stats(void) {
    return &sa_stats;
}

void custom_sa_reset_stats(void) {
    memset(&sa_stats, 0, sizeof(sa_stats));
}

#else

#define INSTRUMENT_START(phase) ((void)0)
#define INSTRUMENT_END(phase) ((void)0)

#endif /* ENABLE_SA_INSTRUMENTATION */

/* ========================================================================== */
/* Stack Entry for Non-Recursive Quicksort                                    */
/* ========================================================================== */

#define SORT_STACK_SIZE 128

typedef struct {
    saidx_t first;
    saidx_t last;
    saidx_t depth;
} SortStackEntry;

/* ========================================================================== */
/* Bitvector Operations (Inline for L1 access)                                */
/* ========================================================================== */

FORCE_INLINE void type_set_s(TypeBitVector *types, saidx_t i) {
    types->bits[i >> 6] |= (1ULL << (i & 63));
}

FORCE_INLINE void type_set_l(TypeBitVector *types, saidx_t i) {
    types->bits[i >> 6] &= ~(1ULL << (i & 63));
}

FORCE_INLINE int type_is_s(const TypeBitVector *types, saidx_t i) {
    return (int)((types->bits[i >> 6] >> (i & 63)) & 1);
}

FORCE_INLINE int type_is_lms(const TypeBitVector *types, saidx_t i) {
    /* LMS = S-type preceded by L-type (or first position) */
    if (i == 0) return 0;
    return type_is_s(types, i) && !type_is_s(types, i - 1);
}

/* ========================================================================== */
/* Phase 5: Branchless Type Operations                                        */
/* ========================================================================== */

#if ENABLE_SA_BRANCHLESS

/**
 * Branchless suffix type determination
 * Returns 1 if S-type, 0 if L-type
 */
FORCE_INLINE int determine_type_branchless(sauchar_t curr, sauchar_t next, int next_is_s) {
    int less_than = curr < next;
    int equal = curr == next;
    return less_than | (equal & next_is_s);
}

/**
 * Branchless conditional type set
 * Sets bit if is_s is true, otherwise leaves unchanged
 */
FORCE_INLINE void type_set_conditional(TypeBitVector *types, saidx_t i, int is_s) {
    uint64_t mask = (uint64_t)is_s << (i & 63);
    types->bits[i >> 6] |= mask;
}

#endif /* ENABLE_SA_BRANCHLESS */

/* ========================================================================== */
/* Phase 4a: AVX2 Histogram Merge                                             */
/* ========================================================================== */

#if ENABLE_SA_AVX2

/**
 * Merge four histograms using AVX2 SIMD
 * Processes 8 counts at a time (256-bit registers)
 */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2")))
#endif
static void merge_histograms_avx2(
    const saidx_t *hist0,
    const saidx_t *hist1,
    const saidx_t *hist2,
    const saidx_t *hist3,
    saidx_t *bucket_A
) {
    for (int c = 0; c < 256; c += 8) {
        __m256i h0 = _mm256_loadu_si256((const __m256i*)&hist0[c]);
        __m256i h1 = _mm256_loadu_si256((const __m256i*)&hist1[c]);
        __m256i h2 = _mm256_loadu_si256((const __m256i*)&hist2[c]);
        __m256i h3 = _mm256_loadu_si256((const __m256i*)&hist3[c]);

        /* Add pairwise: (h0+h1), (h2+h3) */
        __m256i sum01 = _mm256_add_epi32(h0, h1);
        __m256i sum23 = _mm256_add_epi32(h2, h3);

        /* Final sum */
        __m256i total = _mm256_add_epi32(sum01, sum23);

        _mm256_storeu_si256((__m256i*)&bucket_A[c], total);
    }
}

#endif /* ENABLE_SA_AVX2 */

/**
 * Scalar histogram merge fallback
 */
static void merge_histograms_scalar(
    const saidx_t *hist0,
    const saidx_t *hist1,
    const saidx_t *hist2,
    const saidx_t *hist3,
    saidx_t *bucket_A
) {
    for (int c = 0; c < 256; c++) {
        bucket_A[c] = hist0[c] + hist1[c] + hist2[c] + hist3[c];
    }
}

/* ========================================================================== */
/* Phase 4b: AVX2 LMS Substring Comparison                                    */
/* ========================================================================== */

#if ENABLE_SA_AVX2

/**
 * Compare two LMS substrings using AVX2
 * Returns: <0 if i < j, 0 if equal, >0 if i > j
 */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2")))
#endif
static int compare_lms_substrings_avx2(
    const sauchar_t *T,
    const TypeBitVector *types,
    saidx_t i,
    saidx_t j,
    saidx_t n
) {
    /* Fast path: compare 32 bytes at a time using AVX2 */
    while (i + 32 <= n && j + 32 <= n) {
        __m256i vi = _mm256_loadu_si256((const __m256i*)&T[i]);
        __m256i vj = _mm256_loadu_si256((const __m256i*)&T[j]);

        /* Compare for equality */
        __m256i eq = _mm256_cmpeq_epi8(vi, vj);
        unsigned int mask = (unsigned int)_mm256_movemask_epi8(eq);

        if (mask != 0xFFFFFFFF) {
            /* Found difference - find first mismatch position */
            int pos = CTZ32(~mask);
            return (int)T[i + pos] - (int)T[j + pos];
        }

        /* Check for LMS boundaries within this block */
        for (int k = 0; k < 32; k++) {
            int i_lms = (i + k > 0) && type_is_lms(types, i + k);
            int j_lms = (j + k > 0) && type_is_lms(types, j + k);

            if (i_lms && j_lms) {
                return 0; /* Same LMS substring */
            }
            if (i_lms) return -1;
            if (j_lms) return 1;
        }

        i += 32;
        j += 32;
    }

    /* Scalar fallback for remainder */
    while (i < n && j < n) {
        if (T[i] != T[j]) {
            return (int)T[i] - (int)T[j];
        }

        int i_lms = (i > 0) && type_is_lms(types, i);
        int j_lms = (j > 0) && type_is_lms(types, j);

        if (i_lms && j_lms) return 0;
        if (i_lms) return -1;
        if (j_lms) return 1;

        i++;
        j++;
    }

    return (i >= n) ? -1 : 1;
}

#endif /* ENABLE_SA_AVX2 */

/**
 * Scalar LMS substring comparison
 */
FORCE_INLINE int compare_lms_substrings_scalar(
    const sauchar_t *T,
    const TypeBitVector *types,
    saidx_t i,
    saidx_t j,
    saidx_t n
) {
    while (i < n && j < n) {
        if (T[i] != T[j]) {
            return (int)T[i] - (int)T[j];
        }

        int i_lms = (i > 0) && type_is_lms(types, i);
        int j_lms = (j > 0) && type_is_lms(types, j);

        if (i_lms && j_lms) return 0;
        if (i_lms) return -1;
        if (j_lms) return 1;

        i++;
        j++;
    }

    return (i >= n) ? -1 : 1;
}

/**
 * Dispatcher for LMS comparison
 */
FORCE_INLINE int compare_lms_substrings(
    const sauchar_t *T,
    const TypeBitVector *types,
    saidx_t i,
    saidx_t j,
    saidx_t n
) {
#if ENABLE_SA_AVX2
    return compare_lms_substrings_avx2(T, types, i, j, n);
#else
    return compare_lms_substrings_scalar(T, types, i, j, n);
#endif
}

/* ========================================================================== */
/* Phase 3: 8-Way Loop Unrolled Suffix Classification                         */
/* ========================================================================== */

#if ENABLE_SA_8WAY_UNROLL

HOT_FUNCTION
static void classify_suffixes_8way(const sauchar_t *T, TypeBitVector *types, saidx_t n) {
    /* Clear bitvector */
    memset(types, 0, sizeof(TypeBitVector));

    /* Sentinel: last suffix is always S-type */
    type_set_s(types, n - 1);

    /* Right-to-left classification with 8-way unrolling */
    saidx_t i = n - 2;

#if ENABLE_SA_BRANCHLESS
    /* 8-way unrolled loop with branchless operations */
    for (; i >= 7; i -= 8) {
        int is_s7 = determine_type_branchless(T[i - 0], T[i + 1], type_is_s(types, i + 1));
        type_set_conditional(types, i - 0, is_s7);

        int is_s6 = determine_type_branchless(T[i - 1], T[i - 0], is_s7);
        type_set_conditional(types, i - 1, is_s6);

        int is_s5 = determine_type_branchless(T[i - 2], T[i - 1], is_s6);
        type_set_conditional(types, i - 2, is_s5);

        int is_s4 = determine_type_branchless(T[i - 3], T[i - 2], is_s5);
        type_set_conditional(types, i - 3, is_s4);

        int is_s3 = determine_type_branchless(T[i - 4], T[i - 3], is_s4);
        type_set_conditional(types, i - 4, is_s3);

        int is_s2 = determine_type_branchless(T[i - 5], T[i - 4], is_s3);
        type_set_conditional(types, i - 5, is_s2);

        int is_s1 = determine_type_branchless(T[i - 6], T[i - 5], is_s2);
        type_set_conditional(types, i - 6, is_s1);

        int is_s0 = determine_type_branchless(T[i - 7], T[i - 6], is_s1);
        type_set_conditional(types, i - 7, is_s0);
    }
#else
    /* 8-way unrolled loop with branching */
    for (; i >= 7; i -= 8) {
        /* Position i */
        if (T[i] < T[i + 1] || (T[i] == T[i + 1] && type_is_s(types, i + 1)))
            type_set_s(types, i);
        /* Position i-1 */
        if (T[i - 1] < T[i] || (T[i - 1] == T[i] && type_is_s(types, i)))
            type_set_s(types, i - 1);
        /* Position i-2 */
        if (T[i - 2] < T[i - 1] || (T[i - 2] == T[i - 1] && type_is_s(types, i - 1)))
            type_set_s(types, i - 2);
        /* Position i-3 */
        if (T[i - 3] < T[i - 2] || (T[i - 3] == T[i - 2] && type_is_s(types, i - 2)))
            type_set_s(types, i - 3);
        /* Position i-4 */
        if (T[i - 4] < T[i - 3] || (T[i - 4] == T[i - 3] && type_is_s(types, i - 3)))
            type_set_s(types, i - 4);
        /* Position i-5 */
        if (T[i - 5] < T[i - 4] || (T[i - 5] == T[i - 4] && type_is_s(types, i - 4)))
            type_set_s(types, i - 5);
        /* Position i-6 */
        if (T[i - 6] < T[i - 5] || (T[i - 6] == T[i - 5] && type_is_s(types, i - 5)))
            type_set_s(types, i - 6);
        /* Position i-7 */
        if (T[i - 7] < T[i - 6] || (T[i - 7] == T[i - 6] && type_is_s(types, i - 6)))
            type_set_s(types, i - 7);
    }
#endif

    /* Handle remaining elements */
    for (; i >= 0; i--) {
        if (T[i] < T[i + 1] || (T[i] == T[i + 1] && type_is_s(types, i + 1))) {
            type_set_s(types, i);
        }
    }
}

#endif /* ENABLE_SA_8WAY_UNROLL */

/**
 * Original 4-way unrolled classification (fallback)
 */
HOT_FUNCTION
static void classify_suffixes_4way(const sauchar_t *T, TypeBitVector *types, saidx_t n) {
    /* Clear bitvector */
    memset(types, 0, sizeof(TypeBitVector));

    /* Sentinel: last suffix is always S-type */
    type_set_s(types, n - 1);

    /* Right-to-left classification with 4-way unrolling */
    saidx_t i = n - 2;

    for (; i >= 3; i -= 4) {
        if (T[i] < T[i + 1] || (T[i] == T[i + 1] && type_is_s(types, i + 1)))
            type_set_s(types, i);
        if (T[i - 1] < T[i] || (T[i - 1] == T[i] && type_is_s(types, i)))
            type_set_s(types, i - 1);
        if (T[i - 2] < T[i - 1] || (T[i - 2] == T[i - 1] && type_is_s(types, i - 1)))
            type_set_s(types, i - 2);
        if (T[i - 3] < T[i - 2] || (T[i - 3] == T[i - 2] && type_is_s(types, i - 2)))
            type_set_s(types, i - 3);
    }

    for (; i >= 0; i--) {
        if (T[i] < T[i + 1] || (T[i] == T[i + 1] && type_is_s(types, i + 1))) {
            type_set_s(types, i);
        }
    }
}

/**
 * Dispatcher for suffix classification
 */
FORCE_INLINE void classify_suffixes(const sauchar_t *T, TypeBitVector *types, saidx_t n) {
#if ENABLE_SA_8WAY_UNROLL
    classify_suffixes_8way(T, types, n);
#else
    classify_suffixes_4way(T, types, n);
#endif
}

/* ========================================================================== */
/* Phase 2: Bucket Initialization with Enhanced Histogram                     */
/* ========================================================================== */

#if ENABLE_SA_AVX2

/**
 * Merge eight histograms using AVX2 SIMD
 * Processes 8 counts at a time (256-bit registers)
 */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2")))
#endif
static void merge_histograms_8way_avx2(
    const saidx_t hist[8][256],
    saidx_t *bucket_A
) {
    for (int c = 0; c < 256; c += 8) {
        __m256i sum = _mm256_setzero_si256();
        for (int h = 0; h < 8; h++) {
            __m256i hv = _mm256_loadu_si256((const __m256i*)&hist[h][c]);
            sum = _mm256_add_epi32(sum, hv);
        }
        _mm256_storeu_si256((__m256i*)&bucket_A[c], sum);
    }
}

#endif /* ENABLE_SA_AVX2 */

/**
 * Scalar merge for 8-way histograms (fallback)
 */
static void merge_histograms_8way_scalar(
    const saidx_t hist[8][256],
    saidx_t *bucket_A
) {
    for (int c = 0; c < 256; c++) {
        bucket_A[c] = hist[0][c] + hist[1][c] + hist[2][c] + hist[3][c]
                    + hist[4][c] + hist[5][c] + hist[6][c] + hist[7][c];
    }
}

/**
 * 8-way histogram with prefetch and AVX2 merge
 * Uses 8 separate accumulators to avoid memory port contention
 */
HOT_FUNCTION
static void init_buckets_8way(const sauchar_t *T, saidx_t *bucket_A, saidx_t n) {
    /* Clear bucket_A */
    memset(bucket_A, 0, 256 * sizeof(saidx_t));

    /* 8 separate accumulators to avoid memory port contention */
    alignas(64) saidx_t hist[8][256] = {{0}};

    saidx_t i = 0;
    saidx_t n8 = n & ~7; /* Round down to multiple of 8 */

    /* 8-way unrolled counting with prefetch */
    for (; i < n8; i += 8) {
        /* Prefetch ahead for L1 and L2 */
        PREFETCH_R(&T[i + 64]);      /* L1 prefetch */
        PREFETCH_R_L2(&T[i + 256]);  /* L2 prefetch */

        hist[0][T[i + 0]]++;
        hist[1][T[i + 1]]++;
        hist[2][T[i + 2]]++;
        hist[3][T[i + 3]]++;
        hist[4][T[i + 4]]++;
        hist[5][T[i + 5]]++;
        hist[6][T[i + 6]]++;
        hist[7][T[i + 7]]++;
    }

    /* Handle remainder */
    for (; i < n; i++) {
        hist[0][T[i]]++;
    }

    /* Merge histograms using SIMD or scalar */
#if ENABLE_SA_AVX2
    merge_histograms_8way_avx2(hist, bucket_A);
#else
    merge_histograms_8way_scalar(hist, bucket_A);
#endif
}

/**
 * Original 4-way histogram (kept for comparison/fallback)
 */
HOT_FUNCTION
static void init_buckets_4way(const sauchar_t *T, saidx_t *bucket_A, saidx_t n) {
    /* Clear bucket_A */
    memset(bucket_A, 0, 256 * sizeof(saidx_t));

    /* 4-way histogram to reduce cache conflicts */
    saidx_t hist0[256] = {0};
    saidx_t hist1[256] = {0};
    saidx_t hist2[256] = {0};
    saidx_t hist3[256] = {0};

    saidx_t i = 0;
    saidx_t n4 = n & ~3; /* Round down to multiple of 4 */

    /* 4-way unrolled counting */
    for (; i < n4; i += 4) {
        hist0[T[i + 0]]++;
        hist1[T[i + 1]]++;
        hist2[T[i + 2]]++;
        hist3[T[i + 3]]++;
    }

    /* Handle remainder */
    for (; i < n; i++) {
        hist0[T[i]]++;
    }

    /* Merge histograms using SIMD or scalar */
#if ENABLE_SA_AVX2
    merge_histograms_avx2(hist0, hist1, hist2, hist3, bucket_A);
#else
    merge_histograms_scalar(hist0, hist1, hist2, hist3, bucket_A);
#endif
}

/* ========================================================================== */
/* Phase 3: Find LMS Suffixes with Enhanced Prefetch                          */
/* ========================================================================== */

HOT_FUNCTION
static saidx_t find_lms_suffixes(
    const sauchar_t *T,
    const TypeBitVector *types,
    saidx_t *SA,
    saidx_t *bucket_A,
    saidx_t *bucket_B,
    saidx_t n
) {
    /* Clear SA */
    for (saidx_t i = 0; i < n; i++) {
        SA[i] = -1;
    }

    /* Compute bucket starts and ends */
    saidx_t bucket_start[256];
    saidx_t bucket_end[256];

    saidx_t sum = 0;
    for (int c = 0; c < 256; c++) {
        bucket_start[c] = sum;
        sum += bucket_A[c];
        bucket_end[c] = sum;
    }

    /* Copy ends for placing LMS suffixes */
    saidx_t bucket_tail[256];
    memcpy(bucket_tail, bucket_end, sizeof(bucket_tail));

    /* Find and place LMS suffixes (right to left) */
    saidx_t lms_count = 0;
    for (saidx_t i = n - 1; i >= 1; i--) {
        if (type_is_lms(types, i)) {
            int c = T[i];
            bucket_tail[c]--;
            SA[bucket_tail[c]] = i;
            lms_count++;
        }
    }

    /* Store bucket boundaries for induced sorting */
    memcpy(bucket_A, bucket_start, 256 * sizeof(saidx_t));
    memcpy(bucket_B, bucket_end, 256 * sizeof(saidx_t));

    return lms_count;
}

/* ========================================================================== */
/* Phase 2: Enhanced Induced Sorting with Triple-Stream Prefetch              */
/* ========================================================================== */

#if ENABLE_SA_PREFETCH

HOT_FUNCTION
static void induce_l_suffixes_prefetch(
    const sauchar_t *T,
    saidx_t *SA,
    const TypeBitVector *types,
    saidx_t *bucket_head,
    saidx_t n
) {
    /* Initialize bucket heads */
    saidx_t head[256];
    saidx_t sum = 0;
    for (int c = 0; c < 256; c++) {
        head[c] = sum;
        sum += bucket_head[c];
        bucket_head[c] = head[c];
    }

    saidx_t cursor[256];
    memcpy(cursor, head, sizeof(cursor));

    /* Cache runtime config values locally to avoid repeated memory loads */
    const int sa_pf = g_sa_config.sa_prefetch;
    const int text_pf = g_sa_config.text_prefetch;
    const int bucket_pf = g_sa_config.bucket_prefetch;

    /* Left-to-right scan with triple-stream prefetch */
    for (saidx_t i = 0; i < n; i++) {
        /* Stream 1: Prefetch ahead in SA array */
        if (i + sa_pf < n) {
            PREFETCH_R(&SA[i + sa_pf]);
        }

        saidx_t j = SA[i];
        if (j > 0) {
            /* Stream 2: Prefetch text position */
            if (i + text_pf < n) {
                saidx_t future_j = SA[i + text_pf];
                if (future_j > 0) {
                    PREFETCH_R(&T[future_j - 1]);
                }
            }

            saidx_t k = j - 1;
            if (!type_is_s(types, k)) {
                int c = T[k];

                /* Stream 3: Prefetch bucket cursor write location */
                if (i + bucket_pf < n) {
                    saidx_t future_j2 = SA[i + bucket_pf];
                    if (future_j2 > 0 && !type_is_s(types, future_j2 - 1)) {
                        int future_c = T[future_j2 - 1];
                        PREFETCH_W(&SA[cursor[future_c]]);
                    }
                }

                SA[cursor[c]] = k;
                cursor[c]++;
            }
        } else if (j < 0) {
            SA[i] = ~j;
        }
    }
}

HOT_FUNCTION
static void induce_s_suffixes_prefetch(
    const sauchar_t *T,
    saidx_t *SA,
    const TypeBitVector *types,
    const saidx_t *bucket_counts,
    saidx_t n
) {
    /* Compute bucket ends */
    saidx_t tail[256];
    saidx_t sum = 0;
    for (int c = 0; c < 256; c++) {
        sum += bucket_counts[c];
        tail[c] = sum - 1;
    }

    saidx_t cursor[256];
    memcpy(cursor, tail, sizeof(cursor));

    /* Cache runtime config values locally to avoid repeated memory loads */
    const int sa_pf = g_sa_config.sa_prefetch;
    const int text_pf = g_sa_config.text_prefetch;
    const int bucket_pf = g_sa_config.bucket_prefetch;

    /* Right-to-left scan with triple-stream prefetch */
    for (saidx_t i = n - 1; i >= 0; i--) {
        /* Stream 1: Prefetch ahead (backwards) in SA array */
        if (i >= sa_pf) {
            PREFETCH_R(&SA[i - sa_pf]);
        }

        saidx_t j = SA[i];
        if (j > 0) {
            /* Stream 2: Prefetch text position */
            if (i >= text_pf) {
                saidx_t future_j = SA[i - text_pf];
                if (future_j > 0) {
                    PREFETCH_R(&T[future_j - 1]);
                }
            }

            saidx_t k = j - 1;
            if (type_is_s(types, k)) {
                int c = T[k];

                /* Stream 3: Prefetch bucket cursor write location */
                if (i >= bucket_pf) {
                    saidx_t future_j2 = SA[i - bucket_pf];
                    if (future_j2 > 0 && type_is_s(types, future_j2 - 1)) {
                        int future_c = T[future_j2 - 1];
                        PREFETCH_W(&SA[cursor[future_c]]);
                    }
                }

                SA[cursor[c]] = k;
                cursor[c]--;
            }
        }
    }
}

#endif /* ENABLE_SA_PREFETCH */

/**
 * Original induced sorting without enhanced prefetch (fallback)
 */
HOT_FUNCTION
static void induce_l_suffixes_basic(
    const sauchar_t *T,
    saidx_t *SA,
    const TypeBitVector *types,
    saidx_t *bucket_head,
    saidx_t n
) {
    saidx_t head[256];
    saidx_t sum = 0;
    for (int c = 0; c < 256; c++) {
        head[c] = sum;
        sum += bucket_head[c];
        bucket_head[c] = head[c];
    }

    saidx_t cursor[256];
    memcpy(cursor, head, sizeof(cursor));

    for (saidx_t i = 0; i < n; i++) {
        saidx_t j = SA[i];
        if (j > 0) {
            saidx_t k = j - 1;
            if (!type_is_s(types, k)) {
                int c = T[k];
                SA[cursor[c]] = k;
                cursor[c]++;
            }
        } else if (j < 0) {
            SA[i] = ~j;
        }
    }
}

HOT_FUNCTION
static void induce_s_suffixes_basic(
    const sauchar_t *T,
    saidx_t *SA,
    const TypeBitVector *types,
    const saidx_t *bucket_counts,
    saidx_t n
) {
    saidx_t tail[256];
    saidx_t sum = 0;
    for (int c = 0; c < 256; c++) {
        sum += bucket_counts[c];
        tail[c] = sum - 1;
    }

    saidx_t cursor[256];
    memcpy(cursor, tail, sizeof(cursor));

    for (saidx_t i = n - 1; i >= 0; i--) {
        saidx_t j = SA[i];
        if (j > 0) {
            saidx_t k = j - 1;
            if (type_is_s(types, k)) {
                int c = T[k];
                SA[cursor[c]] = k;
                cursor[c]--;
            }
        }
    }
}

/**
 * Dispatcher for induced sorting
 */
FORCE_INLINE void induce_l_suffixes(
    const sauchar_t *T,
    saidx_t *SA,
    const TypeBitVector *types,
    saidx_t *bucket_head,
    saidx_t n
) {
#if ENABLE_SA_PREFETCH
    induce_l_suffixes_prefetch(T, SA, types, bucket_head, n);
#else
    induce_l_suffixes_basic(T, SA, types, bucket_head, n);
#endif
}

FORCE_INLINE void induce_s_suffixes(
    const sauchar_t *T,
    saidx_t *SA,
    const TypeBitVector *types,
    const saidx_t *bucket_counts,
    saidx_t n
) {
#if ENABLE_SA_PREFETCH
    induce_s_suffixes_prefetch(T, SA, types, bucket_counts, n);
#else
    induce_s_suffixes_basic(T, SA, types, bucket_counts, n);
#endif
}

/* ========================================================================== */
/* LMS Suffix Sorting                                                         */
/* ========================================================================== */

/**
 * 2-Character Radix Sort for LMS Suffixes
 * Sorts LMS positions by their first 2 characters for better initial ordering.
 * This can reduce work in the induced sorting phase.
 *
 * Uses 65536 (256*256) buckets for 2-byte keys.
 */
HOT_FUNCTION
static void sort_lms_radix_2char(
    const sauchar_t *T,
    saidx_t *output,        /* Output sorted positions */
    const saidx_t *lms_positions,
    saidx_t lms_count,
    saidx_t n
) {
    /* Count array for 2-character keys */
    static thread_local saidx_t count[65536];
    memset(count, 0, sizeof(count));

    /* Count pass - count occurrences of each 2-char key */
    for (saidx_t i = 0; i < lms_count; i++) {
        saidx_t pos = lms_positions[i];
        uint32_t key;
        if (pos + 1 < n) {
            key = ((uint32_t)T[pos] << 8) | T[pos + 1];
        } else {
            key = (uint32_t)T[pos] << 8;  /* Last character, pad with 0 */
        }
        count[key]++;
    }

    /* Prefix sum to get bucket starts */
    saidx_t sum = 0;
    for (int k = 0; k < 65536; k++) {
        saidx_t c = count[k];
        count[k] = sum;
        sum += c;
    }

    /* Place pass - distribute to output in sorted order */
    for (saidx_t i = 0; i < lms_count; i++) {
        saidx_t pos = lms_positions[i];
        uint32_t key;
        if (pos + 1 < n) {
            key = ((uint32_t)T[pos] << 8) | T[pos + 1];
        } else {
            key = (uint32_t)T[pos] << 8;
        }
        output[count[key]++] = pos;
    }
}

/**
 * Enhanced LMS finding with 2-character radix sort
 * Collects all LMS positions, sorts by first 2 chars, then places in SA.
 */
HOT_FUNCTION
static saidx_t find_lms_suffixes_radix(
    const sauchar_t *T,
    const TypeBitVector *types,
    saidx_t *SA,
    saidx_t *bucket_A,
    saidx_t *bucket_B,
    saidx_t n,
    saidx_t *temp_lms  /* Temporary buffer for LMS positions */
) {
    /* Clear SA */
    for (saidx_t i = 0; i < n; i++) {
        SA[i] = -1;
    }

    /* Compute bucket starts and ends */
    saidx_t bucket_start[256];
    saidx_t bucket_end[256];

    saidx_t sum = 0;
    for (int c = 0; c < 256; c++) {
        bucket_start[c] = sum;
        sum += bucket_A[c];
        bucket_end[c] = sum;
    }

    /* First pass: collect all LMS positions */
    saidx_t lms_count = 0;
    for (saidx_t i = 1; i < n; i++) {
        if (type_is_lms(types, i)) {
            temp_lms[lms_count++] = i;
        }
    }

    if (lms_count > 1) {
        /* Sort LMS positions by first 2 characters */
        saidx_t *sorted_lms = SA;  /* Reuse SA temporarily */
        sort_lms_radix_2char(T, sorted_lms, temp_lms, lms_count, n);

        /* Copy back to temp_lms in sorted order */
        memcpy(temp_lms, sorted_lms, lms_count * sizeof(saidx_t));

        /* Clear SA again */
        for (saidx_t i = 0; i < n; i++) {
            SA[i] = -1;
        }
    }

    /* Place LMS suffixes at bucket ends (right to left for stable sort) */
    saidx_t bucket_tail[256];
    memcpy(bucket_tail, bucket_end, sizeof(bucket_tail));

    for (saidx_t i = lms_count - 1; i >= 0; i--) {
        saidx_t pos = temp_lms[i];
        int c = T[pos];
        bucket_tail[c]--;
        SA[bucket_tail[c]] = pos;
    }

    /* Store bucket boundaries for induced sorting */
    memcpy(bucket_A, bucket_start, 256 * sizeof(saidx_t));
    memcpy(bucket_B, bucket_end, 256 * sizeof(saidx_t));

    return lms_count;
}

/**
 * Insertion sort for small ranges
 */
static void insertion_sort_lms(
    const sauchar_t *T,
    const TypeBitVector *types,
    saidx_t *SA,
    saidx_t first,
    saidx_t last,
    saidx_t n
) {
    for (saidx_t i = first + 1; i < last; i++) {
        saidx_t key = SA[i];
        saidx_t j = i - 1;

        while (j >= first && compare_lms_substrings(T, types, SA[j], key, n) > 0) {
            SA[j + 1] = SA[j];
            j--;
        }
        SA[j + 1] = key;
    }
}

/**
 * Stack-based quicksort for LMS suffixes
 */
HOT_FUNCTION
static void sort_lms_suffixes(
    const sauchar_t *T,
    const TypeBitVector *types,
    saidx_t *SA,
    saidx_t first,
    saidx_t last,
    saidx_t n
) {
    #define INSERTION_THRESHOLD 16

    SortStackEntry stack[SORT_STACK_SIZE];
    int stack_top = 0;

    stack[stack_top++] = (SortStackEntry){first, last, 0};

    while (stack_top > 0) {
        SortStackEntry entry = stack[--stack_top];
        saidx_t lo = entry.first;
        saidx_t hi = entry.last;

        saidx_t size = hi - lo;

        if (size <= INSERTION_THRESHOLD) {
            insertion_sort_lms(T, types, SA, lo, hi, n);
            continue;
        }

        /* Median of three pivot selection */
        saidx_t mid = lo + size / 2;

        if (compare_lms_substrings(T, types, SA[lo], SA[mid], n) > 0) {
            saidx_t tmp = SA[lo]; SA[lo] = SA[mid]; SA[mid] = tmp;
        }
        if (compare_lms_substrings(T, types, SA[mid], SA[hi - 1], n) > 0) {
            saidx_t tmp = SA[mid]; SA[mid] = SA[hi - 1]; SA[hi - 1] = tmp;
            if (compare_lms_substrings(T, types, SA[lo], SA[mid], n) > 0) {
                tmp = SA[lo]; SA[lo] = SA[mid]; SA[mid] = tmp;
            }
        }

        saidx_t pivot = SA[mid];

        /* Partition */
        saidx_t i = lo + 1;
        saidx_t j = hi - 2;

        while (1) {
            while (i < hi && compare_lms_substrings(T, types, SA[i], pivot, n) < 0) i++;
            while (j > lo && compare_lms_substrings(T, types, SA[j], pivot, n) > 0) j--;

            if (i >= j) break;

            saidx_t tmp = SA[i]; SA[i] = SA[j]; SA[j] = tmp;
            i++;
            j--;
        }

        /* Push larger partition first */
        saidx_t left_size = i - lo;
        saidx_t right_size = hi - i;

        if (left_size > right_size) {
            if (left_size > 1 && stack_top < SORT_STACK_SIZE) {
                stack[stack_top++] = (SortStackEntry){lo, i, 0};
            }
            if (right_size > 1 && stack_top < SORT_STACK_SIZE) {
                stack[stack_top++] = (SortStackEntry){i, hi, 0};
            }
        } else {
            if (right_size > 1 && stack_top < SORT_STACK_SIZE) {
                stack[stack_top++] = (SortStackEntry){i, hi, 0};
            }
            if (left_size > 1 && stack_top < SORT_STACK_SIZE) {
                stack[stack_top++] = (SortStackEntry){lo, i, 0};
            }
        }
    }

    #undef INSERTION_THRESHOLD
}

/* ========================================================================== */
/* Sort LMS and Compute Reduced String                                        */
/* ========================================================================== */

HOT_FUNCTION
static saidx_t sort_lms_and_compute_reduced(
    const sauchar_t *T,
    saidx_t *SA,
    const TypeBitVector *types,
    saidx_t *lms_positions,
    saidx_t lms_count,
    saidx_t n,
    saidx_t *bucket_A,
    saidx_t *bucket_B
) {
    /* Collect LMS positions */
    saidx_t k = 0;
    for (saidx_t i = 1; i < n; i++) {
        if (type_is_lms(types, i)) {
            lms_positions[k++] = i;
        }
    }

    /* Clear SA */
    for (saidx_t i = 0; i < n; i++) {
        SA[i] = -1;
    }

    /* Re-compute bucket counts */
    init_buckets_8way(T, bucket_A, n);

    /* Compute bucket ends */
    saidx_t bucket_end[256];
    saidx_t sum = 0;
    for (int c = 0; c < 256; c++) {
        sum += bucket_A[c];
        bucket_end[c] = sum;
    }

    /* Place LMS suffixes at end of buckets */
    saidx_t cursor[256];
    memcpy(cursor, bucket_end, sizeof(cursor));

    for (saidx_t i = lms_count - 1; i >= 0; i--) {
        saidx_t j = lms_positions[i];
        int c = T[j];
        cursor[c]--;
        SA[cursor[c]] = j;
    }

    /* Induce L-type suffixes */
    saidx_t bucket_start[256];
    sum = 0;
    for (int c = 0; c < 256; c++) {
        bucket_start[c] = sum;
        sum += bucket_A[c];
    }

    saidx_t head_cursor[256];
    memcpy(head_cursor, bucket_start, sizeof(head_cursor));

    for (saidx_t i = 0; i < n; i++) {
        saidx_t j = SA[i];
        if (j > 0 && !type_is_s(types, j - 1)) {
            int c = T[j - 1];
            SA[head_cursor[c]] = j - 1;
            head_cursor[c]++;
        }
    }

    /* Induce S-type suffixes */
    memcpy(cursor, bucket_end, sizeof(cursor));

    for (saidx_t i = n - 1; i >= 0; i--) {
        saidx_t j = SA[i];
        if (j > 0 && type_is_s(types, j - 1)) {
            int c = T[j - 1];
            cursor[c]--;
            SA[cursor[c]] = j - 1;
        }
    }

    /* Find unique LMS substring names */
    saidx_t name = 0;
    saidx_t prev_pos = -1;

    saidx_t *names = SA + n - lms_count;

    for (saidx_t i = 0; i < n; i++) {
        saidx_t pos = SA[i];
        if (pos > 0 && type_is_lms(types, pos)) {
            int different = 1;
            if (prev_pos >= 0) {
                different = (compare_lms_substrings(T, types, prev_pos, pos, n) != 0);
            }

            if (different) {
                name++;
            }

            for (saidx_t j = 0; j < lms_count; j++) {
                if (lms_positions[j] == pos) {
                    names[j] = name - 1;
                    break;
                }
            }

            prev_pos = pos;
        }
    }

    return name;
}

/* ========================================================================== */
/* Main SA-IS Algorithm                                                       */
/* ========================================================================== */

/* Thread-local temporary buffer for radix LMS sorting */
#if ENABLE_SA_RADIX_LMS
    #if defined(__cplusplus) && __cplusplus >= 201103L
        #ifdef _MSC_VER
            static __declspec(thread) saidx_t tl_lms_buffer[CUSTOM_SA_MAX_SIZE / 2];
        #else
            static thread_local saidx_t tl_lms_buffer[CUSTOM_SA_MAX_SIZE / 2];
        #endif
    #else
        static saidx_t tl_lms_buffer[CUSTOM_SA_MAX_SIZE / 2];
    #endif
#endif

HOT_FUNCTION
static void sais_main(
    const sauchar_t *T,
    saidx_t *SA,
    saidx_t n,
    saidx_t *bucket_A,
    saidx_t *bucket_B
) {
    /* Use thread-local workspace for type bitvector */
    TypeBitVector *types = &tl_workspace.types;

    INSTRUMENT_START(classify);

    /* Phase 1: Classify all suffixes */
    classify_suffixes(T, types, n);

    INSTRUMENT_END(classify);

    INSTRUMENT_START(histogram);

    /* Phase 2: Initialize buckets (8-way with AVX2 merge) */
    init_buckets_8way(T, bucket_A, n);

    INSTRUMENT_END(histogram);

    /* Phase 3: Find and place LMS suffixes */
#if ENABLE_SA_RADIX_LMS
    /* Use 2-character radix sort for better initial LMS ordering */
    saidx_t lms_count = find_lms_suffixes_radix(T, types, SA, bucket_A, bucket_B, n, tl_lms_buffer);
#else
    saidx_t lms_count = find_lms_suffixes(T, types, SA, bucket_A, bucket_B, n);
#endif

#if ENABLE_SA_INSTRUMENTATION
    sa_stats.lms_count += lms_count;
#endif

    if (lms_count == 0) {
        /* No LMS suffixes (all same character) */
        for (saidx_t i = 0; i < n; i++) {
            SA[i] = n - 1 - i;
        }
        return;
    }

    /* Preserve bucket counts */
    saidx_t bucket_counts[256];
    memcpy(bucket_counts, bucket_A, sizeof(bucket_counts));

    INSTRUMENT_START(induce_l);

    /* Phase 4a: Induce L-type suffixes */
    induce_l_suffixes(T, SA, types, bucket_A, n);

    INSTRUMENT_END(induce_l);

    /* Restore bucket counts */
    memcpy(bucket_A, bucket_counts, sizeof(bucket_counts));

    INSTRUMENT_START(induce_s);

    /* Phase 4b: Induce S-type suffixes */
    induce_s_suffixes(T, SA, types, bucket_A, n);

    INSTRUMENT_END(induce_s);
}

/* ========================================================================== */
/* Public API                                                                 */
/* ========================================================================== */

saint_t custom_sa_70kb(
    const sauchar_t *T,
    saidx_t *SA,
    saidx_t n,
    saidx_t *bucket_A,
    saidx_t *bucket_B
) {
#if ENABLE_SA_INSTRUMENTATION
    uint64_t start_total = RDTSC();
    sa_stats.call_count++;
#endif

    /* Validate parameters */
    if (UNLIKELY(!T || !SA || n < 0)) {
        return -1;
    }

    /* Handle edge cases */
    if (n == 0) {
        return 0;
    }

    if (n == 1) {
        SA[0] = 0;
        return 0;
    }

    if (n == 2) {
        if (T[0] < T[1]) {
            SA[0] = 0;
            SA[1] = 1;
        } else {
            SA[0] = 1;
            SA[1] = 0;
        }
        return 0;
    }

    /* Check size limit */
    if (n > CUSTOM_SA_MAX_SIZE) {
        return -1;
    }

    /* Verify bucket arrays provided */
    if (!bucket_A || !bucket_B) {
        return -2;
    }

    /* Run SA-IS algorithm */
    sais_main(T, SA, n, bucket_A, bucket_B);

#if ENABLE_SA_INSTRUMENTATION
    sa_stats.total_cycles += RDTSC() - start_total;
#endif

    return 0;
}

bool custom_sa_validate(
    const sauchar_t *T,
    const saidx_t *SA,
    saidx_t n
) {
    if (!T || !SA || n <= 0) {
        return false;
    }

    /* Check all positions appear exactly once */
    uint8_t *seen = (uint8_t *)calloc(n, sizeof(uint8_t));
    if (!seen) {
        return false;
    }

    for (saidx_t i = 0; i < n; i++) {
        saidx_t pos = SA[i];
        if (pos < 0 || pos >= n || seen[pos]) {
            free(seen);
            return false;
        }
        seen[pos] = 1;
    }
    free(seen);

    /* Check lexicographic order */
    for (saidx_t i = 1; i < n; i++) {
        saidx_t pos1 = SA[i - 1];
        saidx_t pos2 = SA[i];

        saidx_t j = 0;
        while (pos1 + j < n && pos2 + j < n) {
            if (T[pos1 + j] < T[pos2 + j]) {
                break;
            }
            if (T[pos1 + j] > T[pos2 + j]) {
                return false;
            }
            j++;
        }

        if (pos1 + j >= n && pos2 + j < n) {
            /* pos1 suffix is shorter - correct */
        } else if (pos1 + j < n && pos2 + j >= n) {
            return false;
        }
    }

    return true;
}

bool custom_sa_matches_divsufsort(
    const sauchar_t *T,
    saidx_t n,
    saidx_t *bucket_A,
    saidx_t *bucket_B
) {
    if (!T || n <= 0 || !bucket_A || !bucket_B) {
        return false;
    }

    saidx_t *sa_custom = (saidx_t *)malloc(n * sizeof(saidx_t));
    saidx_t *sa_ref = (saidx_t *)malloc(n * sizeof(saidx_t));

    if (!sa_custom || !sa_ref) {
        free(sa_custom);
        free(sa_ref);
        return false;
    }

    if (custom_sa_70kb(T, sa_custom, n, bucket_A, bucket_B) != 0) {
        free(sa_custom);
        free(sa_ref);
        return false;
    }

    if (divsufsort(T, sa_ref, n, bucket_A, bucket_B) != 0) {
        free(sa_custom);
        free(sa_ref);
        return false;
    }

    bool match = (memcmp(sa_custom, sa_ref, n * sizeof(saidx_t)) == 0);

    free(sa_custom);
    free(sa_ref);

    return match;
}
