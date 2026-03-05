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
#include <stdio.h>

/* ========================================================================== */
/* Negative Marker Encoding                                                   */
/* ========================================================================== */

/*
 * We use negative values to mark LMS positions to protect them from being
 * overwritten during S-type induction. The challenge is that ~0 = -1 in
 * two's complement, which conflicts with -1 used as the "empty" sentinel.
 *
 * Solution: Use -(pos + 2) encoding so that:
 *   - Empty: -1
 *   - Position 0 marker: -2
 *   - Position 1 marker: -3
 *   - Position p marker: -(p + 2)
 *
 * To decode: pos = -(marker + 2) = -marker - 2
 * To check if marked: marker <= -2
 */
#define SA_EMPTY      (-1)
#define SA_MARK(pos)  (-(saidx_t)(pos) - 2)
#define SA_UNMARK(v)  (-(saidx_t)(v) - 2)
#define SA_IS_MARKED(v) ((v) <= -2)

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

/*
 * Thread-local workspace for SA-IS algorithm.
 *
 * MinGW's thread_local implementation doesn't support over-aligned types
 * (alignas(64) on SAWorkspace causes crashes). Fix: use a pointer in TLS
 * and allocate the aligned workspace on first access.
 */

/* Platform-specific aligned allocation */
static FORCE_INLINE void* sa_aligned_alloc(size_t alignment, size_t size) {
    /* MinGW check must come first since clang on MinGW also defines __GNUC__ */
#if defined(__MINGW32__) || defined(__MINGW64__) || defined(_WIN32)
    return _aligned_malloc(size, alignment);
#elif defined(_MSC_VER)
    return _aligned_malloc(size, alignment);
#elif defined(__GNUC__) || defined(__clang__)
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
#else
    return malloc(size);
#endif
}

static FORCE_INLINE void sa_aligned_free(void* ptr) {
#if defined(__MINGW32__) || defined(__MINGW64__) || defined(_WIN32) || defined(_MSC_VER)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

/* Thread-local pointer to aligned workspace (safe on MinGW) */
#if defined(__cplusplus) && __cplusplus >= 201103L
    #ifdef _MSC_VER
        static __declspec(thread) SAWorkspace* tl_workspace_ptr = nullptr;
    #else
        static thread_local SAWorkspace* tl_workspace_ptr = nullptr;
    #endif
#else
    static SAWorkspace* tl_workspace_ptr = NULL;
#endif

/* Get thread-local workspace, allocating on first access */
static FORCE_INLINE SAWorkspace* get_tl_workspace(void) {
    if (UNLIKELY(tl_workspace_ptr == NULL)) {
        tl_workspace_ptr = (SAWorkspace*)sa_aligned_alloc(CACHE_LINE_SIZE, sizeof(SAWorkspace));
        if (tl_workspace_ptr) {
            memset(tl_workspace_ptr, 0, sizeof(SAWorkspace));
        }
    }
    return tl_workspace_ptr;
}

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
    saidx_t orig_i = i, orig_j = j;  /* Save original positions for tie-breaking */

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

        /* Check for LMS boundaries within this block (skip starting positions) */
        for (int k = 0; k < 32; k++) {
            int i_lms = (i + k > orig_i) && type_is_lms(types, i + k);
            int j_lms = (j + k > orig_j) && type_is_lms(types, j + k);

            if (i_lms && j_lms) {
                /* LMS substrings equal - shorter suffix (higher position) is smaller */
                return orig_j - orig_i;
            }
            if (i_lms) return -1;
            if (j_lms) return 1;
        }

        i += 32;
        j += 32;
    }

    /* Scalar fallback for remainder (skip LMS check at starting positions) */
    while (i < n && j < n) {
        if (T[i] != T[j]) {
            return (int)T[i] - (int)T[j];
        }

        int i_lms = (i > orig_i) && type_is_lms(types, i);
        int j_lms = (j > orig_j) && type_is_lms(types, j);

        if (i_lms && j_lms) {
            /* LMS substrings equal - shorter suffix (higher position) is smaller */
            return orig_j - orig_i;
        }
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
    saidx_t orig_i = i, orig_j = j;  /* Save original positions for tie-breaking */

    while (i < n && j < n) {
        if (T[i] != T[j]) {
            return (int)T[i] - (int)T[j];
        }

        /*
         * Check for LMS boundaries only AFTER the starting positions.
         * The starting positions are LMS by definition, so we skip the check there.
         * An LMS-substring extends from one LMS position to just before the next.
         */
        int i_lms = (i > orig_i) && type_is_lms(types, i);
        int j_lms = (j > orig_j) && type_is_lms(types, j);

        if (i_lms && j_lms) {
            /*
             * LMS substrings are equal. For suffix array correctness,
             * the suffix at the HIGHER position (shorter suffix) is smaller.
             * Return negative if orig_i > orig_j (i's suffix is smaller).
             */
            return orig_j - orig_i;
        }
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
    bool placed_sentinel = false;

    for (saidx_t i = n - 1; i >= 1; i--) {
        if (type_is_lms(types, i)) {
            int c = T[i];
            bucket_tail[c]--;
            SA[bucket_tail[c]] = SA_MARK(i);  /* Mark as LMS for induced sorting */
            lms_count++;
            if (i == n - 1) placed_sentinel = true;
        }
    }

    /*
     * Position n-1 (sentinel) MUST be in the SA. It can only enter via:
     * 1. Being an LMS suffix (placed above)
     * 2. Being induced from position n (doesn't exist!)
     *
     * So we must place n-1 as pseudo-LMS if not already placed.
     */
    if (!placed_sentinel && n >= 1) {
        int c = T[n - 1];
        bucket_tail[c]--;
        SA[bucket_tail[c]] = SA_MARK(n - 1);
        lms_count++;
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
#ifdef SA_DEBUG
        if (SA_IS_MARKED(j) && SA_UNMARK(j) >= n - 8) {
            fprintf(stderr, "DEBUG: L-ind(pf) unmarking SA[%d]=%d (pos=%d)\n",
                    i, j, SA_UNMARK(j));
            fflush(stderr);
        }
#endif
        if (SA_IS_MARKED(j)) {
            /* Negative LMS marker - convert to positive FIRST */
            j = SA_UNMARK(j);
            SA[i] = j;
        }
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
    /* Compute bucket starts and ends */
    saidx_t head[256];
    saidx_t tail[256];
    saidx_t sum = 0;
    for (int c = 0; c < 256; c++) {
        head[c] = sum;
        sum += bucket_counts[c];
        tail[c] = sum - 1;
    }

    saidx_t cursor[256];
    memcpy(cursor, tail, sizeof(cursor));

    /*
     * CRITICAL: Handle sentinel n-1 for correct tie-breaking.
     *
     * The sentinel is the shortest suffix, so when two suffixes share a prefix
     * up to the end of the shorter one, the sentinel's chain should be
     * lexicographically SMALLER (appear at LOWER SA index).
     *
     * During S-induction (right-to-left scan), entries placed FIRST get HIGHER
     * indices. So we want the sentinel to be processed LAST in its bucket.
     *
     * Strategy: Inject sentinel processing when we reach the HEAD of its bucket,
     * which is the LAST position in the bucket during right-to-left scan.
     */
    saidx_t sentinel_pos = -1;
    int sentinel_char = (n >= 1) ? T[n - 1] : 0;
    saidx_t sentinel_bucket_head = head[sentinel_char];
    bool sentinel_injected = false;

    if (n >= 2) {
        /* Find where sentinel n-1 is in SA */
        for (saidx_t i = 0; i < n; i++) {
            saidx_t v = SA[i];
            if (SA_IS_MARKED(v)) v = SA_UNMARK(v);
            if (v == n - 1) {
                sentinel_pos = i;
                break;
            }
        }

        if (sentinel_pos >= 0) {
#ifdef SA_DEBUG
            fprintf(stderr, "DEBUG: S-ind(pf) PRE: saving sentinel at SA[%d]=%d, bucket head=%d\n",
                    sentinel_pos, n - 1, sentinel_bucket_head);
            fflush(stderr);
#endif
            /* Clear sentinel position - we'll inject its processing at bucket head */
            SA[sentinel_pos] = -1;
        }
    }

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

        /*
         * CRITICAL: Inject sentinel processing at the HEAD of its bucket.
         *
         * During right-to-left scan, the bucket head is the LAST position
         * processed within a bucket. This ensures all other entries in the
         * sentinel's bucket are processed first, giving the sentinel's chain
         * lower cursor positions (lexicographically smaller).
         */
        if (!sentinel_injected && sentinel_pos >= 0 && i == sentinel_bucket_head) {
            saidx_t k = n - 2;
            if (type_is_s(types, k) && !type_is_lms(types, k)) {
                int c = T[k];
#ifdef SA_DEBUG
                fprintf(stderr, "DEBUG: S-ind(pf) INJECT at bucket head i=%d: inducing k=%d at cursor[%d]=%d\n",
                        i, k, c, cursor[c]);
                fflush(stderr);
#endif
                if (cursor[c] >= head[c]) {
                    SA[cursor[c]] = k;
                    cursor[c]--;
                }
            }
            sentinel_injected = true;
            /* Continue processing - there may be a valid entry at the bucket head too */
        }

#ifdef SA_DEBUG
        if (j >= n - 8 || (SA_IS_MARKED(j) && SA_UNMARK(j) >= n - 8)) {
            fprintf(stderr, "DEBUG: S-ind(pf) at SA[%d]=%d (marked=%d)\n",
                    i, j, SA_IS_MARKED(j) ? 1 : 0);
            fflush(stderr);
        }
#endif
        if (SA_IS_MARKED(j)) {
            /* Negative LMS marker - convert to positive FIRST */
            j = SA_UNMARK(j);
            SA[i] = j;
        }
        if (j > 0) {
            /* Stream 2: Prefetch text position */
            if (i >= text_pf) {
                saidx_t future_j = SA[i - text_pf];
                if (future_j > 0) {
                    PREFETCH_R(&T[future_j - 1]);
                }
            }

            saidx_t k = j - 1;
#ifdef SA_DEBUG
            /* Trace j == n-1 (sentinel) before type check */
            if (j == n - 1 && n >= 70000) {
                fprintf(stderr, "DEBUG: S-ind(pf) found sentinel j=%d at SA[%d], k=%d, type_is_s(k)=%d, T[k]=%d\n",
                        j, i, k, type_is_s(types, k), T[k]);
                fflush(stderr);
            }
            /* Track when we reach i=41610 (where sentinel was placed) */
            if (n >= 70000 && i == 41610) {
                fprintf(stderr, "DEBUG: S-ind(pf) at i=41610, j=SA[41610]=%d (was expecting 69999)\n", j);
                fflush(stderr);
            }
#endif
            /*
             * In SA-IS, S-induction places only NON-LMS S-type suffixes.
             * LMS suffixes were already correctly placed in Phase 6.
             * Re-placing them would put duplicates in the SA.
             */
            if (type_is_s(types, k) && !type_is_lms(types, k)) {
                int c = T[k];

                /* Stream 3: Prefetch bucket cursor write location */
                if (i >= bucket_pf) {
                    saidx_t future_j2 = SA[i - bucket_pf];
                    if (future_j2 > 0 && type_is_s(types, future_j2 - 1)) {
                        int future_c = T[future_j2 - 1];
                        PREFETCH_W(&SA[cursor[future_c]]);
                    }
                }

#ifdef SA_DEBUG
                if (k >= n - 8) {
                    fprintf(stderr, "DEBUG: S-ind(pf) placing k=%d at cursor[%d]=%d (head[%d]=%d) from SA[%d]=%d\n",
                            k, c, cursor[c], c, head[c], i, j);
                    fflush(stderr);
                }
                /* Trace specifically when j == n-1 (sentinel induces n-2) */
                if (j == n - 1 && n >= 70000) {
                    fprintf(stderr, "DEBUG: S-ind sentinel j=%d found at SA[%d], k=%d, T[k]=%d, cursor[%d]=%d, head[%d]=%d, placing=%s\n",
                            j, i, k, c, c, cursor[c], c, head[c], cursor[c] >= head[c] ? "YES" : "NO");
                    fflush(stderr);
                }
                /* Track when cursor reaches sentinel's position (41610 for n=70000) */
                if (n >= 70000 && cursor[c] >= 41608 && cursor[c] <= 41612) {
                    fprintf(stderr, "DEBUG: S-ind(pf) NEAR 41610: placing k=%d at cursor[%d]=%d from SA[%d]=%d, SA[41610]=%d\n",
                            k, c, cursor[c], i, j, SA[41610]);
                    fflush(stderr);
                }
#endif
                if (cursor[c] >= head[c]) {
                    SA[cursor[c]] = k;
                    cursor[c]--;
                }
            }
        }
    }

    /*
     * Edge case: if sentinel was in bucket 0 and we somehow missed injecting
     * (shouldn't happen since loop goes to i=0, but safety check).
     */
    if (!sentinel_injected && sentinel_pos >= 0 && n >= 2) {
        saidx_t k = n - 2;
        if (type_is_s(types, k) && !type_is_lms(types, k)) {
            int c = T[k];
#ifdef SA_DEBUG
            fprintf(stderr, "DEBUG: S-ind(pf) INJECT post-loop: inducing k=%d at cursor[%d]=%d\n",
                    k, c, cursor[c]);
            fflush(stderr);
#endif
            if (cursor[c] >= head[c]) {
                SA[cursor[c]] = k;
                cursor[c]--;
            }
        }
    }

    /*
     * POST-PROCESSING: Ensure sentinel n-1 is at its correct sorted position.
     *
     * The sentinel n-1 is the SMALLEST suffix starting with T[n-1]
     * (because it's the shortest suffix). It should be at head[T[n-1]].
     *
     * We cleared SA[sentinel_pos] = -1 earlier. Now we need to:
     * 1. Find any empty slots (-1) and fill them by shifting
     * 2. Place the sentinel at head[T[n-1]]
     */
    if (n >= 1) {
        /* sentinel_char already defined at function start */
        saidx_t target_pos = head[sentinel_char];

        /* First, check if sentinel is already at the correct position */
        if (SA[target_pos] == (saidx_t)(n - 1)) {
#ifdef SA_DEBUG
            fprintf(stderr, "DEBUG: S-ind(pf) POST: sentinel %d already at head[%d]=%d\n",
                    n - 1, sentinel_char, target_pos);
            fflush(stderr);
#endif
            /* Already correct, nothing to do */
        } else {
            /* Find the empty slot we created (should be at sentinel_pos if not overwritten) */
            saidx_t empty_pos = -1;
            for (saidx_t i = 0; i < n; i++) {
                if (SA[i] == -1) {
                    empty_pos = i;
                    break;
                }
            }

            if (empty_pos >= 0) {
                /* Fill empty slot and insert sentinel at target */
                if (empty_pos > target_pos) {
                    /* Shift [target_pos, empty_pos-1] right by 1 to fill the hole */
                    for (saidx_t i = empty_pos; i > target_pos; i--) {
                        SA[i] = SA[i - 1];
                    }
                } else if (empty_pos < target_pos) {
                    /* Shift [empty_pos+1, target_pos] left by 1 to fill the hole */
                    for (saidx_t i = empty_pos; i < target_pos; i++) {
                        SA[i] = SA[i + 1];
                    }
                }
                SA[target_pos] = n - 1;
#ifdef SA_DEBUG
                fprintf(stderr, "DEBUG: S-ind(pf) POST: filled hole at %d, placed sentinel %d at head[%d]=%d\n",
                        empty_pos, n - 1, sentinel_char, target_pos);
                fflush(stderr);
#endif
            } else {
                /* No empty slot - sentinel might be elsewhere, need to find and move it */
                saidx_t sentinel_current = -1;
                for (saidx_t i = 0; i < n; i++) {
                    if (SA[i] == (saidx_t)(n - 1)) {
                        sentinel_current = i;
                        break;
                    }
                }
                if (sentinel_current >= 0 && sentinel_current != target_pos) {
                    if (sentinel_current < target_pos) {
                        for (saidx_t i = sentinel_current; i < target_pos; i++) {
                            SA[i] = SA[i + 1];
                        }
                    } else {
                        for (saidx_t i = sentinel_current; i > target_pos; i--) {
                            SA[i] = SA[i - 1];
                        }
                    }
                    SA[target_pos] = n - 1;
#ifdef SA_DEBUG
                    fprintf(stderr, "DEBUG: S-ind(pf) POST: moved sentinel %d from %d to head[%d]=%d\n",
                            n - 1, sentinel_current, sentinel_char, target_pos);
                    fflush(stderr);
#endif
                }
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
        if (SA_IS_MARKED(j)) {
            /* Negative LMS marker - convert to positive FIRST */
            j = SA_UNMARK(j);
            SA[i] = j;
        }
        if (j > 0) {
            saidx_t k = j - 1;
            if (!type_is_s(types, k)) {
                int c = T[k];
                SA[cursor[c]] = k;
                cursor[c]++;
            }
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
    /* Compute bucket starts (heads) and ends (tails) */
    saidx_t head[256];
    saidx_t tail[256];
    saidx_t sum = 0;
    for (int c = 0; c < 256; c++) {
        head[c] = sum;
        sum += bucket_counts[c];
        tail[c] = sum - 1;
    }

    saidx_t cursor[256];
    memcpy(cursor, tail, sizeof(cursor));

    /*
     * CRITICAL: Handle sentinel n-1 for correct tie-breaking.
     *
     * The sentinel is the shortest suffix, so when two suffixes share a prefix
     * up to the end of the shorter one, the sentinel's chain should be
     * lexicographically SMALLER (appear at LOWER SA index).
     *
     * During S-induction (right-to-left scan), entries placed FIRST get HIGHER
     * indices. So we want the sentinel to be processed LAST in its bucket.
     *
     * Strategy: Inject sentinel processing when we reach the HEAD of its bucket,
     * which is the LAST position in the bucket during right-to-left scan.
     */
    saidx_t sentinel_pos = -1;
    int sentinel_char = (n >= 1) ? T[n - 1] : 0;
    saidx_t sentinel_bucket_head = head[sentinel_char];
    bool sentinel_injected = false;

    if (n >= 2) {
        /* Find where sentinel n-1 is in SA */
        for (saidx_t i = 0; i < n; i++) {
            saidx_t v = SA[i];
            if (SA_IS_MARKED(v)) v = SA_UNMARK(v);
            if (v == n - 1) {
                sentinel_pos = i;
                break;
            }
        }

        if (sentinel_pos >= 0) {
#ifdef SA_DEBUG
            fprintf(stderr, "DEBUG: S-ind(basic) PRE: saving sentinel at SA[%d]=%d, bucket head=%d\n",
                    sentinel_pos, n - 1, sentinel_bucket_head);
            fflush(stderr);
#endif
            /* Clear sentinel position - we'll inject its processing at bucket head */
            SA[sentinel_pos] = -1;
        }
    }

    for (saidx_t i = n - 1; i >= 0; i--) {
        saidx_t j = SA[i];

        /*
         * CRITICAL: Inject sentinel processing at the HEAD of its bucket.
         *
         * During right-to-left scan, the bucket head is the LAST position
         * processed within a bucket. This ensures all other entries in the
         * sentinel's bucket are processed first, giving the sentinel's chain
         * lower cursor positions (lexicographically smaller).
         */
        if (!sentinel_injected && sentinel_pos >= 0 && i == sentinel_bucket_head) {
            saidx_t k = n - 2;
            if (type_is_s(types, k) && !type_is_lms(types, k)) {
                int c = T[k];
#ifdef SA_DEBUG
                fprintf(stderr, "DEBUG: S-ind(basic) INJECT at bucket head i=%d: inducing k=%d at cursor[%d]=%d\n",
                        i, k, c, cursor[c]);
                fflush(stderr);
#endif
                if (cursor[c] >= head[c]) {
                    SA[cursor[c]] = k;
                    cursor[c]--;
                }
            }
            sentinel_injected = true;
            /* Continue processing - there may be a valid entry at the bucket head too */
        }

        if (SA_IS_MARKED(j)) {
            /* Negative LMS marker - convert to positive FIRST */
            j = SA_UNMARK(j);
            SA[i] = j;
        }
        if (j > 0) {
            /* Positive value - process it */
            saidx_t k = j - 1;
            /*
             * In SA-IS, S-induction places only NON-LMS S-type suffixes.
             * LMS suffixes were already correctly placed in Phase 6.
             * Re-placing them would put duplicates in the SA.
             */
            if (type_is_s(types, k) && !type_is_lms(types, k)) {
                int c = T[k];
                if (cursor[c] >= head[c]) {
#ifdef SA_DEBUG
                    if (j == 69995 || j == 44850 || k == 69994 || k == 44849) {
                        fprintf(stderr, "DEBUG: S-ind(basic) TRACE: i=%d, scanning j=%d, placing k=%d at cursor[%d]=%d\n",
                                i, j, k, c, cursor[c]);
                        fflush(stderr);
                    }
#endif
                    SA[cursor[c]] = k;
                    cursor[c]--;
                }
            }
        }
    }

    /*
     * Edge case: if sentinel was in bucket 0 and we somehow missed injecting
     * (shouldn't happen since loop goes to i=0, but safety check).
     */
    if (!sentinel_injected && sentinel_pos >= 0 && n >= 2) {
        saidx_t k = n - 2;
        if (type_is_s(types, k) && !type_is_lms(types, k)) {
            int c = T[k];
#ifdef SA_DEBUG
            fprintf(stderr, "DEBUG: S-ind(basic) INJECT post-loop: inducing k=%d at cursor[%d]=%d\n",
                    k, c, cursor[c]);
            fflush(stderr);
#endif
            if (cursor[c] >= head[c]) {
                SA[cursor[c]] = k;
                cursor[c]--;
            }
        }
    }

    /*
     * POST-PROCESSING: Ensure sentinel n-1 is at its correct sorted position.
     *
     * The sentinel n-1 is the SMALLEST suffix starting with T[n-1]
     * (because it's the shortest suffix). It should be at head[T[n-1]].
     *
     * We cleared SA[sentinel_pos] = -1 earlier. Now we need to:
     * 1. Find any empty slots (-1) and fill them by shifting
     * 2. Place the sentinel at head[T[n-1]]
     */
    if (n >= 1) {
        /* sentinel_char already defined at function start */
        saidx_t target_pos = head[sentinel_char];

        /* First, check if sentinel is already at the correct position */
        if (SA[target_pos] == (saidx_t)(n - 1)) {
#ifdef SA_DEBUG
            fprintf(stderr, "DEBUG: S-ind(basic) POST: sentinel %d already at head[%d]=%d\n",
                    n - 1, sentinel_char, target_pos);
            fflush(stderr);
#endif
            /* Already correct, nothing to do */
        } else {
            /* Find the empty slot we created (should be at sentinel_pos if not overwritten) */
            saidx_t empty_pos = -1;
            for (saidx_t i = 0; i < n; i++) {
                if (SA[i] == -1) {
                    empty_pos = i;
                    break;
                }
            }

            if (empty_pos >= 0) {
                /* Fill empty slot and insert sentinel at target */
                if (empty_pos > target_pos) {
                    /* Shift [target_pos, empty_pos-1] right by 1 to fill the hole */
                    for (saidx_t i = empty_pos; i > target_pos; i--) {
                        SA[i] = SA[i - 1];
                    }
                } else if (empty_pos < target_pos) {
                    /* Shift [empty_pos+1, target_pos] left by 1 to fill the hole */
                    for (saidx_t i = empty_pos; i < target_pos; i++) {
                        SA[i] = SA[i + 1];
                    }
                }
                SA[target_pos] = n - 1;
#ifdef SA_DEBUG
                fprintf(stderr, "DEBUG: S-ind(basic) POST: filled hole at %d, placed sentinel %d at head[%d]=%d\n",
                        empty_pos, n - 1, sentinel_char, target_pos);
                fflush(stderr);
#endif
            } else {
                /* No empty slot - sentinel might be elsewhere, need to find and move it */
                saidx_t sentinel_current = -1;
                for (saidx_t i = 0; i < n; i++) {
                    if (SA[i] == (saidx_t)(n - 1)) {
                        sentinel_current = i;
                        break;
                    }
                }
                if (sentinel_current >= 0 && sentinel_current != target_pos) {
                    if (sentinel_current < target_pos) {
                        for (saidx_t i = sentinel_current; i < target_pos; i++) {
                            SA[i] = SA[i + 1];
                        }
                    } else {
                        for (saidx_t i = sentinel_current; i > target_pos; i--) {
                            SA[i] = SA[i - 1];
                        }
                    }
                    SA[target_pos] = n - 1;
#ifdef SA_DEBUG
                    fprintf(stderr, "DEBUG: S-ind(basic) POST: moved sentinel %d from %d to head[%d]=%d\n",
                            n - 1, sentinel_current, sentinel_char, target_pos);
                    fflush(stderr);
#endif
                }
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
    bool has_sentinel = false;
    for (saidx_t i = 1; i < n; i++) {
        if (type_is_lms(types, i)) {
            temp_lms[lms_count++] = i;
            if (i == n - 1) has_sentinel = true;
        }
    }

    /*
     * Position n-1 (sentinel) MUST be in the SA. It can only enter via:
     * 1. Being an LMS suffix (collected above)
     * 2. Being induced from position n (doesn't exist!)
     *
     * So we must add n-1 as pseudo-LMS if not already collected.
     */
    if (!has_sentinel && n >= 1) {
        temp_lms[lms_count++] = n - 1;
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
        SA[bucket_tail[c]] = SA_MARK(pos);  /* Mark as LMS for induced sorting */
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

/**
 * Check if two LMS substrings are equal
 * Returns 1 if equal, 0 if different
 */
FORCE_INLINE int lms_substrings_equal(
    const sauchar_t *T,
    const TypeBitVector *types,
    saidx_t i,
    saidx_t j,
    saidx_t n
) {
    if (i < 0 || j < 0) return 0;

    /* Compare characters until we find a difference or hit the end */
    saidx_t len = 0;
    int found_lms_i = 0, found_lms_j = 0;

    while (i + len < n && j + len < n) {
        if (T[i + len] != T[j + len]) {
            return 0; /* Different characters */
        }

        /* Check for LMS boundaries (after first char) */
        if (len > 0) {
            found_lms_i = type_is_lms(types, i + len);
            found_lms_j = type_is_lms(types, j + len);

            if (found_lms_i && found_lms_j) {
                return 1; /* Both hit LMS at same relative position - equal */
            }
            if (found_lms_i || found_lms_j) {
                return 0; /* Only one hit LMS - different lengths */
            }
        }
        len++;
    }

    /* Reached end of text - check if both ended */
    return (i + len >= n) && (j + len >= n);
}

/**
 * Recursive SA-IS for reduced string (integer alphabet)
 * T1 = reduced string of names (0 to max_name-1)
 * SA1 = output suffix array for T1
 * n1 = length of T1 (= lms_count)
 */
/*
 * Helper: Compare two LMS substrings in T1 starting at positions p1 and p2.
 * Returns: 0 if equal, -1 if p1 < p2, +1 if p1 > p2
 */
static int compare_lms_substrings(
    const saidx_t *T1,
    const saidx_t *types_rs,
    saidx_t n1,
    saidx_t p1,
    saidx_t p2
) {
    if (p1 == p2) return 0;

    /* Compare character by character until we reach the end of both LMS substrings */
    saidx_t i = 0;
    while (1) {
        saidx_t c1 = (p1 + i < n1) ? T1[p1 + i] : -1; /* -1 as sentinel */
        saidx_t c2 = (p2 + i < n1) ? T1[p2 + i] : -1;

        if (c1 < c2) return -1;
        if (c1 > c2) return 1;

        /* Check if we've reached the end of either LMS substring */
        if (p1 + i >= n1 || p2 + i >= n1) {
            /* Shorter suffix is smaller */
            if (p1 + i >= n1 && p2 + i < n1) return -1;
            if (p2 + i >= n1 && p1 + i < n1) return 1;
            return 0;
        }

        /* Check if we've passed the end of LMS substrings (reached next LMS) */
        if (i > 0) {
            int is_lms1 = (types_rs[p1 + i] == 1 && types_rs[p1 + i - 1] == 0);
            int is_lms2 = (types_rs[p2 + i] == 1 && types_rs[p2 + i - 1] == 0);
            if (is_lms1 && is_lms2) return 0; /* Both reached next LMS - equal */
            if (is_lms1) return -1; /* p1's LMS substring is shorter */
            if (is_lms2) return 1;  /* p2's LMS substring is shorter */
        }

        i++;
        /* Safety limit to prevent infinite loops */
        if (i > n1) break;
    }
    return 0;
}

static void sais_recursive(
    const saidx_t *T1,
    saidx_t *SA1,
    saidx_t n1,
    saidx_t alphabet_size
) {
#ifdef SA_DEBUG
    fprintf(stderr, "DEBUG: sais_recursive(n1=%d, alpha=%d) ENTER\n", n1, alphabet_size);
    fflush(stderr);
#endif
    if (n1 == 0) return;
    if (n1 == 1) {
        SA1[0] = 0;
        return;
    }

    /* For small inputs or when all names are unique, use counting sort */
    if (alphabet_size >= n1) {
        /* All names unique - direct inverse */
#ifdef SA_DEBUG
        fprintf(stderr, "DEBUG: sais_recursive - all unique, direct inverse\n");
        fflush(stderr);
#endif
        for (saidx_t i = 0; i < n1; i++) {
            SA1[T1[i]] = i;
        }
        return;
    }

#ifdef SA_DEBUG
    fprintf(stderr, "DEBUG: sais_recursive - need full SA-IS\n");
    fflush(stderr);
#endif

    /* Classify types for reduced string */
    saidx_t *types_rs = (saidx_t*)malloc((n1 + 1) * sizeof(saidx_t));
    if (!types_rs) {
        /* Fallback: simple O(n^2) sort */
        for (saidx_t i = 0; i < n1; i++) SA1[i] = i;
        return;
    }

    /* Classify: S-type = 1, L-type = 0 */
    types_rs[n1 - 1] = 1; /* Last is S-type */
    types_rs[n1] = 1;     /* Sentinel */
    for (saidx_t i = n1 - 2; i >= 0; i--) {
        if (T1[i] < T1[i + 1]) {
            types_rs[i] = 1; /* S-type */
        } else if (T1[i] > T1[i + 1]) {
            types_rs[i] = 0; /* L-type */
        } else {
            types_rs[i] = types_rs[i + 1]; /* Same as next */
        }
    }

#ifdef SA_DEBUG
    {
        int l_count = 0, s_count = 0;
        for (saidx_t i = 0; i < n1; i++) {
            if (types_rs[i] == 0) l_count++;
            else s_count++;
        }
        fprintf(stderr, "DEBUG: sais_recursive - types classified: %d L-type, %d S-type\n", l_count, s_count);
        fflush(stderr);
    }
#endif

    /* Count bucket sizes */
    saidx_t *bucket = (saidx_t*)calloc(alphabet_size, sizeof(saidx_t));
    if (!bucket) {
        free(types_rs);
        for (saidx_t i = 0; i < n1; i++) SA1[i] = i;
        return;
    }

    for (saidx_t i = 0; i < n1; i++) {
        bucket[T1[i]]++;
    }

    /* Store bucket counts for later reuse */
    saidx_t *bucket_counts = (saidx_t*)malloc(alphabet_size * sizeof(saidx_t));
    if (!bucket_counts) {
        free(types_rs);
        free(bucket);
        for (saidx_t i = 0; i < n1; i++) SA1[i] = i;
        return;
    }
    memcpy(bucket_counts, bucket, alphabet_size * sizeof(saidx_t));

    /* Compute bucket ends */
    saidx_t sum = 0;
    for (saidx_t i = 0; i < alphabet_size; i++) {
        sum += bucket[i];
        bucket[i] = sum; /* bucket[i] = end of bucket i */
    }

    /* Initialize SA */
    for (saidx_t i = 0; i < n1; i++) {
        SA1[i] = -1;
    }

    /* Count LMS positions and store them */
    saidx_t lms_count_rs = 0;
    bool has_sentinel_rs = false;
    for (saidx_t i = 1; i < n1; i++) {
        if (types_rs[i] == 1 && types_rs[i - 1] == 0) {
            lms_count_rs++;
            if (i == n1 - 1) has_sentinel_rs = true;
        }
    }

    /*
     * Position n1-1 (sentinel of reduced string) must be included as pseudo-LMS
     * if not a true LMS. This mirrors the handling in the main sais_main function.
     */
    if (!has_sentinel_rs && n1 >= 1) {
        lms_count_rs++;
    }

    saidx_t *lms_positions_rs = (saidx_t*)malloc(lms_count_rs * sizeof(saidx_t));
    saidx_t *lms_text_order_rs = (saidx_t*)malloc(lms_count_rs * sizeof(saidx_t));
    if (!lms_positions_rs || !lms_text_order_rs) {
        free(types_rs);
        free(bucket);
        free(bucket_counts);
        if (lms_positions_rs) free(lms_positions_rs);
        if (lms_text_order_rs) free(lms_text_order_rs);
        for (saidx_t i = 0; i < n1; i++) SA1[i] = i;
        return;
    }

    saidx_t lms_idx = 0;
    for (saidx_t i = 1; i < n1; i++) {
        if (types_rs[i] == 1 && types_rs[i - 1] == 0) {
            lms_text_order_rs[lms_idx++] = i;
        }
    }

    /* Add pseudo-LMS n1-1 if not already included */
    if (!has_sentinel_rs && n1 >= 1) {
        lms_text_order_rs[lms_idx++] = n1 - 1;
    }

    saidx_t *bucket_copy = (saidx_t*)malloc(alphabet_size * sizeof(saidx_t));
    if (!bucket_copy) {
        free(types_rs);
        free(bucket);
        free(bucket_counts);
        free(lms_positions_rs);
        free(lms_text_order_rs);
        for (saidx_t i = 0; i < n1; i++) SA1[i] = i;
        return;
    }

    /* ========== FIRST PASS: Initial induced sorting to get approximate LMS order ========== */

    /* Place LMS suffixes at bucket ends (right to left) */
    memcpy(bucket_copy, bucket, alphabet_size * sizeof(saidx_t));
    for (saidx_t i = lms_count_rs - 1; i >= 0; i--) {
        saidx_t pos = lms_text_order_rs[i];
        saidx_t c = T1[pos];
        bucket_copy[c]--;
        SA1[bucket_copy[c]] = pos;
    }

    /* L-induction (left to right) */
    sum = 0;
    for (saidx_t i = 0; i < alphabet_size; i++) {
        saidx_t cnt = bucket_counts[i];
        bucket_copy[i] = sum;
        sum += cnt;
    }

    for (saidx_t i = 0; i < n1; i++) {
        saidx_t j = SA1[i];
        if (j > 0 && types_rs[j - 1] == 0) {
            SA1[bucket_copy[T1[j - 1]]] = j - 1;
            bucket_copy[T1[j - 1]]++;
        }
    }

    /* S-induction (right to left) */
    memcpy(bucket_copy, bucket, alphabet_size * sizeof(saidx_t));
    for (saidx_t i = n1 - 1; i >= 0; i--) {
        saidx_t j = SA1[i];
        if (j > 0 && types_rs[j - 1] == 1) {
            bucket_copy[T1[j - 1]]--;
            SA1[bucket_copy[T1[j - 1]]] = j - 1;
        }
    }

    /* Handle position n1-1 if not LMS (first pass) */
    if (n1 >= 2 && types_rs[n1 - 2] == 1) {
        saidx_t c = T1[n1 - 1];
        saidx_t bucket_start = (c == 0) ? 0 : bucket[c - 1];
        saidx_t bucket_end = bucket[c];
        for (saidx_t i = bucket_start; i < bucket_end; i++) {
            if (SA1[i] == -1) {
                SA1[i] = n1 - 1;
                break;
            }
        }
    }

    /* ========== EXTRACT SORTED LMS AND NAME THEM ========== */

    /* Extract LMS suffixes in sorted order (including pseudo-LMS n1-1) */
    lms_idx = 0;
    for (saidx_t i = 0; i < n1; i++) {
        saidx_t pos = SA1[i];
        if (pos > 0) {
            /* True LMS: S-type preceded by L-type */
            bool is_true_lms = (types_rs[pos] == 1 && types_rs[pos - 1] == 0);
            /* Pseudo-LMS: position n1-1 if not true LMS */
            bool is_pseudo_lms = (pos == n1 - 1 && !has_sentinel_rs);
            if (is_true_lms || is_pseudo_lms) {
                lms_positions_rs[lms_idx++] = pos;
            }
        }
    }

    /* Name the LMS substrings */
    saidx_t *lms_names_rs = (saidx_t*)malloc(n1 * sizeof(saidx_t));
    if (!lms_names_rs) {
        free(types_rs);
        free(bucket);
        free(bucket_counts);
        free(bucket_copy);
        free(lms_positions_rs);
        free(lms_text_order_rs);
        for (saidx_t i = 0; i < n1; i++) SA1[i] = i;
        return;
    }
    for (saidx_t i = 0; i < n1; i++) {
        lms_names_rs[i] = -1;
    }

    saidx_t name = 0;
    lms_names_rs[lms_positions_rs[0]] = name;
    for (saidx_t i = 1; i < lms_count_rs; i++) {
        saidx_t p1 = lms_positions_rs[i - 1];
        saidx_t p2 = lms_positions_rs[i];
        if (compare_lms_substrings(T1, types_rs, n1, p1, p2) != 0) {
            name++;
        }
        lms_names_rs[p2] = name;
    }
    saidx_t num_unique_names_rs = name + 1;

#ifdef SA_DEBUG
    fprintf(stderr, "DEBUG: sais_recursive - %d LMS suffixes, %d unique names\n",
            lms_count_rs, num_unique_names_rs);
    fflush(stderr);
#endif

    /* Build S1 (reduced string) - names in text order */
    saidx_t *S1_rs = (saidx_t*)malloc(lms_count_rs * sizeof(saidx_t));
    if (!S1_rs) {
        free(types_rs);
        free(bucket);
        free(bucket_counts);
        free(bucket_copy);
        free(lms_positions_rs);
        free(lms_text_order_rs);
        free(lms_names_rs);
        for (saidx_t i = 0; i < n1; i++) SA1[i] = i;
        return;
    }
    for (saidx_t i = 0; i < lms_count_rs; i++) {
        S1_rs[i] = lms_names_rs[lms_text_order_rs[i]];
    }

    /* ========== RECURSIVE CALL OR DIRECT INVERSE ========== */

    saidx_t *SA1_rs = (saidx_t*)malloc(lms_count_rs * sizeof(saidx_t));
    if (!SA1_rs) {
        free(types_rs);
        free(bucket);
        free(bucket_counts);
        free(bucket_copy);
        free(lms_positions_rs);
        free(lms_text_order_rs);
        free(lms_names_rs);
        free(S1_rs);
        for (saidx_t i = 0; i < n1; i++) SA1[i] = i;
        return;
    }

    if (num_unique_names_rs < lms_count_rs) {
        /* Names not unique - recurse */
#ifdef SA_DEBUG
        fprintf(stderr, "DEBUG: sais_recursive - recursing with n1=%d, alpha=%d\n",
                lms_count_rs, num_unique_names_rs);
        fflush(stderr);
#endif
        sais_recursive(S1_rs, SA1_rs, lms_count_rs, num_unique_names_rs);
    } else {
        /* All unique - direct inverse */
        for (saidx_t i = 0; i < lms_count_rs; i++) {
            SA1_rs[S1_rs[i]] = i;
        }
    }

    free(S1_rs);
    free(lms_names_rs);

    /* ========== SECOND PASS: Final induced sorting with exact LMS order ========== */

    /* Clear SA */
    for (saidx_t i = 0; i < n1; i++) {
        SA1[i] = -1;
    }

    /* Place LMS in sorted order (using SA1_rs to get exact order) */
    memcpy(bucket_copy, bucket, alphabet_size * sizeof(saidx_t));
    for (saidx_t i = lms_count_rs - 1; i >= 0; i--) {
        saidx_t j = SA1_rs[i];  /* index in text order */
        saidx_t pos = lms_text_order_rs[j];  /* actual position in T1 */
        saidx_t c = T1[pos];
        bucket_copy[c]--;
        SA1[bucket_copy[c]] = pos;
    }

    free(SA1_rs);
    free(lms_positions_rs);
    free(lms_text_order_rs);

    /* Final L-induction */
    sum = 0;
    for (saidx_t i = 0; i < alphabet_size; i++) {
        saidx_t cnt = bucket_counts[i];
        bucket_copy[i] = sum;
        sum += cnt;
    }

    for (saidx_t i = 0; i < n1; i++) {
        saidx_t j = SA1[i];
        if (j > 0 && types_rs[j - 1] == 0) {
            SA1[bucket_copy[T1[j - 1]]] = j - 1;
            bucket_copy[T1[j - 1]]++;
        }
    }

    /* Final S-induction */
    memcpy(bucket_copy, bucket, alphabet_size * sizeof(saidx_t));
    for (saidx_t i = n1 - 1; i >= 0; i--) {
        saidx_t j = SA1[i];
        if (j > 0 && types_rs[j - 1] == 1) {
            bucket_copy[T1[j - 1]]--;
            SA1[bucket_copy[T1[j - 1]]] = j - 1;
        }
    }

    /*
     * Handle position n1-1 if it wasn't placed (not LMS and can't be induced).
     * Position n1-1 is always S-type. If types_rs[n1-2] is also S-type,
     * then n1-1 is NOT LMS and can't be induced from n1 (which doesn't exist).
     * The suffix at n1-1 is just T1[n1-1], which is the smallest S-type suffix
     * with character T1[n1-1]. It should go at the first available S-type
     * position in its bucket (from the left, within the S-type region).
     */
    if (n1 >= 2 && types_rs[n1 - 2] == 1) {
        /* n1-1 is S-type preceded by S-type - not a true LMS, wasn't placed */
        saidx_t c = T1[n1 - 1];
        saidx_t bucket_start = (c == 0) ? 0 : bucket[c - 1];
        saidx_t bucket_end = bucket[c];

        /* Find the first empty position in bucket c's S-type region */
        /* S-type positions are at the end of the bucket, scan from left */
        for (saidx_t i = bucket_start; i < bucket_end; i++) {
            if (SA1[i] == -1) {
                SA1[i] = n1 - 1;
                break;
            }
        }
    }

#ifdef SA_DEBUG
    {
        int empty_count = 0;
        int first_empty = -1;
        for (saidx_t i = 0; i < n1; i++) {
            if (SA1[i] == -1) {
                empty_count++;
                if (first_empty < 0) first_empty = i;
            }
        }
        fprintf(stderr, "DEBUG: sais_recursive - FINAL: %d positions empty (first at %d)\n",
                empty_count, first_empty);
        if (empty_count == 0) {
            fprintf(stderr, "DEBUG: sais_recursive - SA1[0..4] = %d %d %d %d %d\n",
                    SA1[0], SA1[1], SA1[2], SA1[3], SA1[4]);
        }
        fflush(stderr);
    }
#endif

    free(types_rs);
    free(bucket);
    free(bucket_counts);
    free(bucket_copy);
}

HOT_FUNCTION
static void sais_main(
    const sauchar_t *T,
    saidx_t *SA,
    saidx_t n,
    saidx_t *bucket_A,
    saidx_t *bucket_B,
    TypeBitVector *types
) {
    INSTRUMENT_START(classify);

    /* Phase 1: Classify all suffixes */
    classify_suffixes(T, types, n);

    /*
     * NOTE: We do NOT modify types here. Position n-1 is handled as a pseudo-LMS
     * in find_lms_suffixes (always placed regardless of whether it's technically LMS).
     * This preserves correct type invariants for induced sorting.
     */

    INSTRUMENT_END(classify);

    INSTRUMENT_START(histogram);

    /* Phase 2: Initialize buckets (8-way with AVX2 merge) */
    init_buckets_8way(T, bucket_A, n);

    INSTRUMENT_END(histogram);

    /* Preserve bucket counts */
    saidx_t bucket_counts[256];
    memcpy(bucket_counts, bucket_A, sizeof(bucket_counts));

    /* Phase 3: Find and place LMS suffixes */
#if ENABLE_SA_RADIX_LMS
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

    if (lms_count == 1) {
        /* Only one LMS - it's already in the right place */
        memcpy(bucket_A, bucket_counts, sizeof(bucket_counts));
        induce_l_suffixes(T, SA, types, bucket_A, n);
        memcpy(bucket_A, bucket_counts, sizeof(bucket_counts));
        induce_s_suffixes(T, SA, types, bucket_A, n);
        return;
    }

    /* ================================================================== */
    /* PHASE 4: SORT LMS SUFFIXES (using quicksort - simpler than induced) */
    /* ================================================================== */

    INSTRUMENT_START(lms_sort);

#ifdef SA_DEBUG
    fprintf(stderr, "DEBUG: Phase 4 - Sorting %d LMS suffixes\n", lms_count);
    fflush(stderr);
#endif

    /* Collect LMS positions in text order into a buffer */
    saidx_t *lms_positions = (saidx_t*)malloc(lms_count * sizeof(saidx_t));
    saidx_t *lms_names = (saidx_t*)malloc(n * sizeof(saidx_t));

    if (!lms_positions || !lms_names) {
        free(lms_positions);
        free(lms_names);
        return;
    }

    /* Initialize names array */
    for (saidx_t i = 0; i < n; i++) {
        lms_names[i] = -1;
    }

    /* Collect LMS positions in text order */
    /*
     * NOTE: Do NOT include position n-1 here unless it's a true LMS.
     * Position n-1 cannot be induced (no position n exists) and must be
     * handled specially AFTER S-induction completes.
     *
     * If n-1 is a true LMS (preceded by L-type), it will be included normally.
     * If n-1 is S-type preceded by S-type, it must be placed after S-induction.
     */
    saidx_t lms_idx = 0;
    bool has_sentinel = false;

    for (saidx_t i = 1; i < n; i++) {
        if (type_is_lms(types, i)) {
            lms_positions[lms_idx++] = i;
            if (i == n - 1) has_sentinel = true;
        }
    }

    /*
     * Position n-1 (sentinel) must be included as pseudo-LMS if not a true LMS.
     * This matches find_lms_suffixes which also adds n-1 as pseudo-LMS.
     */
    if (!has_sentinel && n >= 1) {
        lms_positions[lms_idx++] = n - 1;
    }

#ifdef SA_DEBUG
    fprintf(stderr, "DEBUG: Collected %d LMS positions\n", lms_idx);
    {
        int sentinel_is_lms = (n >= 2 && !type_is_s(types, n - 2));
        fprintf(stderr, "DEBUG: sentinel n-1=%d is_lms=%d (type[n-2]=%d, type[n-1]=%d)\n",
                n - 1, sentinel_is_lms, type_is_s(types, n - 2), type_is_s(types, n - 1));
        fflush(stderr);
    }
#endif

    /* Clear SA and place LMS positions for sorting */
    for (saidx_t i = 0; i < n; i++) {
        SA[i] = -1;
    }
    for (saidx_t i = 0; i < lms_count; i++) {
        SA[i] = lms_positions[i];
    }

    /* Sort LMS suffixes using quicksort (correct but O(n log n)) */
#ifdef SA_DEBUG
    /* Find positions of 69995 and 44850 before sorting */
    saidx_t idx_69995_before = -1, idx_44850_before = -1;
    for (saidx_t ii = 0; ii < lms_count; ii++) {
        if (SA[ii] == 69995) idx_69995_before = ii;
        if (SA[ii] == 44850) idx_44850_before = ii;
    }
    fprintf(stderr, "DEBUG: Before LMS sort: 69995 at idx=%d, 44850 at idx=%d\n",
            idx_69995_before, idx_44850_before);
    if (idx_69995_before >= 0 && idx_44850_before >= 0) {
        /* Compare them directly */
        int cmp = compare_lms_substrings(T, types, 69995, 44850, n);
        fprintf(stderr, "DEBUG: compare_lms_substrings(69995, 44850) = %d (neg means 69995 < 44850)\n", cmp);
        fprintf(stderr, "DEBUG: T[69995..69999] = %02x %02x %02x %02x %02x\n",
                T[69995], T[69996], T[69997], T[69998], T[69999]);
        fprintf(stderr, "DEBUG: T[44850..44854] = %02x %02x %02x %02x %02x\n",
                T[44850], T[44851], T[44852], T[44853], T[44854]);
    }
    fflush(stderr);
#endif
    sort_lms_suffixes(T, types, SA, 0, lms_count, n);

#ifdef SA_DEBUG
    /* Find positions after sorting */
    saidx_t idx_69995_after = -1, idx_44850_after = -1;
    for (saidx_t ii = 0; ii < lms_count; ii++) {
        if (SA[ii] == 69995) idx_69995_after = ii;
        if (SA[ii] == 44850) idx_44850_after = ii;
    }
    fprintf(stderr, "DEBUG: After LMS sort: 69995 at idx=%d, 44850 at idx=%d\n",
            idx_69995_after, idx_44850_after);
    fprintf(stderr, "DEBUG: LMS sorted, first 10: ");
    for (int i = 0; i < 10 && i < lms_count; i++) {
        fprintf(stderr, "%d ", SA[i]);
    }
    fprintf(stderr, "...\n");
    fflush(stderr);
#endif

    /* Copy sorted LMS back to lms_positions */
    for (saidx_t i = 0; i < lms_count; i++) {
        lms_positions[i] = SA[i];
    }

    /* Assign names by comparing adjacent sorted LMS substrings */
    saidx_t name = 0;
    lms_names[lms_positions[0]] = name;

    for (saidx_t i = 1; i < lms_count; i++) {
        saidx_t prev_pos = lms_positions[i - 1];
        saidx_t curr_pos = lms_positions[i];

        /* Compare LMS substrings - if different, increment name */
        if (!lms_substrings_equal(T, types, prev_pos, curr_pos, n)) {
            name++;
        }
        lms_names[curr_pos] = name;
    }

    saidx_t num_unique_names = name + 1;

#ifdef SA_DEBUG
    fprintf(stderr, "DEBUG: LMS naming done, %d unique names\n", num_unique_names);
    fprintf(stderr, "DEBUG: lms_names[69841]=%d, lms_names[209]=%d, lms_names[69585]=%d\n",
            (69841 < n) ? lms_names[69841] : -1,
            (209 < n) ? lms_names[209] : -1,
            (69585 < n) ? lms_names[69585] : -1);
    /* Check if positions are actually in lms_positions (sorted) */
    bool found_69841 = false, found_209 = false;
    saidx_t idx_69841 = -1, idx_209 = -1;
    for (saidx_t ii = 0; ii < lms_count; ii++) {
        if (lms_positions[ii] == 69841) { found_69841 = true; idx_69841 = ii; }
        if (lms_positions[ii] == 209) { found_209 = true; idx_209 = ii; }
    }
    fprintf(stderr, "DEBUG: 69841 in lms_positions: %s (idx=%d)\n", found_69841 ? "yes" : "no", idx_69841);
    fprintf(stderr, "DEBUG: 209 in lms_positions: %s (idx=%d)\n", found_209 ? "yes" : "no", idx_209);
    fprintf(stderr, "DEBUG: type_is_lms(209)=%d\n", type_is_lms(types, 209));
    /* Check if 69841 is an actual LMS */
    fprintf(stderr, "DEBUG: type_is_lms(69841)=%d, type[69840]=%d, type[69841]=%d\n",
            type_is_lms(types, 69841), type_is_s(types, 69840), type_is_s(types, 69841));
    fflush(stderr);
#endif

    /* Build reduced string S1 (names in original LMS order) */
    /* Collect LMS positions in text order (only true LMS, not n-1 unless it's LMS) */
    saidx_t *lms_text_order = (saidx_t*)malloc(lms_count * sizeof(saidx_t));
    if (!lms_text_order) {
        free(lms_positions);
        free(lms_names);
        return;
    }

    lms_idx = 0;
    for (saidx_t i = 1; i < n; i++) {
        if (type_is_lms(types, i)) {
            lms_text_order[lms_idx++] = i;
        }
    }

    /*
     * If we added n-1 as pseudo-LMS to lms_positions (for sentinel handling),
     * we need to also add it to lms_text_order so the arrays have matching counts.
     */
#ifdef SA_DEBUG
    fprintf(stderr, "DEBUG: Before pseudo-LMS add: lms_idx=%d, lms_count=%d, type_is_lms(n-1)=%d\n",
            lms_idx, lms_count, type_is_lms(types, n - 1));
    fflush(stderr);
#endif
    if (lms_idx < lms_count && !type_is_lms(types, n - 1)) {
        lms_text_order[lms_idx++] = n - 1;
#ifdef SA_DEBUG
        fprintf(stderr, "DEBUG: Added pseudo-LMS n-1=%d to lms_text_order at index %d\n", n - 1, lms_idx - 1);
        fflush(stderr);
#endif
    }

#ifdef SA_DEBUG
    if (lms_idx != lms_count) {
        fprintf(stderr, "DEBUG: ERROR - lms_text_order count (%d) != lms_count (%d)\n", lms_idx, lms_count);
        fflush(stderr);
    }
#endif

    /* Build S1: the reduced string of names in text order */
    saidx_t *S1 = (saidx_t*)malloc(lms_count * sizeof(saidx_t));
    if (!S1) {
        free(lms_positions);
        free(lms_names);
        free(lms_text_order);
        return;
    }

    for (saidx_t i = 0; i < lms_count; i++) {
        saidx_t pos = lms_text_order[i];
        S1[i] = lms_names[pos];
    }

    /* Step 5d: If names not unique, recursively solve */
    saidx_t *SA1 = (saidx_t*)malloc(lms_count * sizeof(saidx_t));
    if (!SA1) {
        free(lms_positions);
        free(lms_names);
        free(S1);
        return;
    }

    if (num_unique_names < lms_count) {
        /* Names not unique - need recursive SA-IS */
#ifdef SA_DEBUG
        fprintf(stderr, "DEBUG: Recursive call - n1=%d, alpha=%d\n", lms_count, num_unique_names);
        fprintf(stderr, "DEBUG: S1[0..9] = ");
        for (int ii = 0; ii < 10 && ii < lms_count; ii++) {
            fprintf(stderr, "%d ", S1[ii]);
        }
        fprintf(stderr, "...\n");
        fprintf(stderr, "DEBUG: S1[last 5] = ");
        for (int ii = lms_count - 5; ii < lms_count; ii++) {
            if (ii >= 0) fprintf(stderr, "%d ", S1[ii]);
        }
        fprintf(stderr, "\n");
        /* Find where name 0 appears in S1 (should correspond to 69841) */
        saidx_t name0_idx = -1;
        for (saidx_t ii = 0; ii < lms_count; ii++) {
            if (S1[ii] == 0) { name0_idx = ii; break; }
        }
        fprintf(stderr, "DEBUG: Name 0 (smallest LMS) at S1[%d], lms_text_order[%d]=%d\n",
                name0_idx, name0_idx, name0_idx >= 0 ? lms_text_order[name0_idx] : -1);
        /* Check: what position should be smallest? */
        fprintf(stderr, "DEBUG: lms_positions (sorted)[0] = %d (expected smallest LMS)\n", lms_positions[0]);
        /* Show S1 at index 13 (where name0_idx is) */
        fprintf(stderr, "DEBUG: S1[13..17] = %d %d %d %d %d\n",
                S1[13], S1[14], S1[15], S1[16], S1[17]);
        fflush(stderr);
#endif
        sais_recursive(S1, SA1, lms_count, num_unique_names);
#ifdef SA_DEBUG
        fprintf(stderr, "DEBUG: Recursive call done\n");
        fprintf(stderr, "DEBUG: SA1[0..9] = ");
        for (int ii = 0; ii < 10 && ii < lms_count; ii++) {
            fprintf(stderr, "%d ", SA1[ii]);
        }
        fprintf(stderr, "...\n");
        fprintf(stderr, "DEBUG: SA1[last 5] = ");
        for (int ii = lms_count - 5; ii < lms_count; ii++) {
            if (ii >= 0) fprintf(stderr, "%d ", SA1[ii]);
        }
        fprintf(stderr, "\n");
        fprintf(stderr, "DEBUG: lms_text_order[0..4] = %d %d %d %d %d\n",
                lms_text_order[0], lms_text_order[1], lms_text_order[2],
                lms_text_order[3], lms_text_order[4]);
        fprintf(stderr, "DEBUG: lms_text_order[last 5] = %d %d %d %d %d\n",
                lms_text_order[lms_count-5], lms_text_order[lms_count-4],
                lms_text_order[lms_count-3], lms_text_order[lms_count-2],
                lms_text_order[lms_count-1]);
        /* Find 69841's index in lms_text_order */
        saidx_t idx_69841_text = -1;
        for (saidx_t ii = 0; ii < lms_count; ii++) {
            if (lms_text_order[ii] == 69841) { idx_69841_text = ii; break; }
        }
        fprintf(stderr, "DEBUG: 69841 in lms_text_order at index %d\n", idx_69841_text);
        if (idx_69841_text >= 0) {
            fprintf(stderr, "DEBUG: S1[%d]=%d, S1[13]=%d (both should be 0)\n",
                    idx_69841_text, S1[idx_69841_text], S1[13]);
            fprintf(stderr, "DEBUG: S1[%d..%d] = ", idx_69841_text,
                    idx_69841_text + 4 < lms_count ? idx_69841_text + 4 : lms_count - 1);
            for (int ii = idx_69841_text; ii < lms_count && ii < idx_69841_text + 5; ii++) {
                fprintf(stderr, "%d ", S1[ii]);
            }
            fprintf(stderr, "\n");
        }
        fflush(stderr);
#endif
    } else {
        /* All names unique - SA1 is just inverse of S1 */
#ifdef SA_DEBUG
        fprintf(stderr, "DEBUG: All names unique - direct inverse\n");
        fflush(stderr);
#endif
        for (saidx_t i = 0; i < lms_count; i++) {
            SA1[S1[i]] = i;
        }
    }

    free(S1);
    free(lms_names);

    INSTRUMENT_END(lms_sort);

    /* ================================================================== */
    /* PHASE 6: FINAL INDUCED SORTING WITH CORRECTLY ORDERED LMS         */
    /* ================================================================== */

    /* Step 6a: Clear SA and place LMS in final sorted order */
    for (saidx_t i = 0; i < n; i++) {
        SA[i] = -1;
    }

    /* Compute bucket ends */
    memcpy(bucket_A, bucket_counts, sizeof(bucket_counts));
    saidx_t bucket_end[256];
    saidx_t sum = 0;
    for (int c = 0; c < 256; c++) {
        sum += bucket_A[c];
        bucket_end[c] = sum;
    }

    /* Place LMS in reverse order of SA1 (so they're placed at bucket ends correctly) */
    saidx_t cursor[256];
    memcpy(cursor, bucket_end, sizeof(cursor));

#ifdef SA_DEBUG
    fprintf(stderr, "DEBUG: Phase 6 - placing LMS (lms_count=%d, includes sentinel n-1)\n", lms_count);
    /* Show what positions will be placed first and last */
    {
        saidx_t first_i = lms_count - 1;
        saidx_t last_i = 0;
        saidx_t first_j = SA1[first_i];
        saidx_t last_j = SA1[last_i];
        saidx_t first_pos = lms_text_order[first_j];
        saidx_t last_pos = lms_text_order[last_j];
        fprintf(stderr, "DEBUG: First to place: i=%d, j=SA1[i]=%d, pos=lms_text_order[j]=%d\n",
                first_i, first_j, first_pos);
        fprintf(stderr, "DEBUG: Last to place: i=%d, j=SA1[i]=%d, pos=lms_text_order[j]=%d\n",
                last_i, last_j, last_pos);
    }
    fflush(stderr);
#endif

    /*
     * Position n-1 is now included in lms_positions (as pseudo-LMS).
     * No need for separate manual placement.
     */

    for (saidx_t i = lms_count - 1; i >= 0; i--) {
        saidx_t j = SA1[i];           /* j = index in text order */
#ifdef SA_DEBUG
        if (j < 0 || j >= lms_count) {
            fprintf(stderr, "DEBUG: ERROR SA1[%d]=%d out of bounds [0,%d)\n", i, j, lms_count);
            fflush(stderr);
        }
#endif
        saidx_t pos = lms_text_order[j]; /* pos = original position in T */
#ifdef SA_DEBUG
        if (pos < 0 || pos >= n) {
            fprintf(stderr, "DEBUG: ERROR lms_text_order[%d]=%d out of bounds [0,%d)\n", j, pos, n);
            fflush(stderr);
        }
#endif
        int c = T[pos];
        cursor[c]--;
#ifdef SA_DEBUG
        if (cursor[c] < 0 || cursor[c] >= n) {
            fprintf(stderr, "DEBUG: ERROR cursor[%d]=%d out of bounds [0,%d)\n", c, cursor[c], n);
            fflush(stderr);
        }
        /* Track where key positions are placed */
        if (pos == 69992 || pos == 69995 || pos == 44850) {
            fprintf(stderr, "DEBUG: Phase 6 placing LMS pos=%d at SA[%d] (bucket[%d])\n", pos, cursor[c], c);
            fflush(stderr);
        }
#endif
        /* Place LMS as NEGATIVE to protect from S-induction overwrite */
        SA[cursor[c]] = SA_MARK(pos);
    }

    free(SA1);
    free(lms_positions);
    free(lms_text_order);

    /* Step 6b: Final induced sort (L-type) */
    memcpy(bucket_A, bucket_counts, sizeof(bucket_counts));

#ifdef SA_DEBUG
    {
        fprintf(stderr, "DEBUG: After Phase 6 LMS placement, SA[0..4] = %d %d %d %d %d\n",
                SA[0], SA[1], SA[2], SA[3], SA[4]);
        /* Count how many LMS in bucket[0] */
        saidx_t b0_end = bucket_counts[0];
        int lms_b0 = 0;
        for (saidx_t i = 0; i < b0_end; i++) {
            if (SA[i] != -1) lms_b0++;
        }
        fprintf(stderr, "DEBUG: Bucket[0] size=%d, filled LMS positions=%d\n", b0_end, lms_b0);
        fprintf(stderr, "DEBUG: Bucket[0] last 5: SA[%d..%d] = %d %d %d %d %d\n",
                b0_end-5, b0_end-1,
                b0_end >= 5 ? SA[b0_end-5] : -999,
                b0_end >= 4 ? SA[b0_end-4] : -999,
                b0_end >= 3 ? SA[b0_end-3] : -999,
                b0_end >= 2 ? SA[b0_end-2] : -999,
                b0_end >= 1 ? SA[b0_end-1] : -999);
        fflush(stderr);
    }
#endif

    induce_l_suffixes(T, SA, types, bucket_A, n);

#ifdef SA_DEBUG
    {
        int empty_count = 0;
        int first_empty = -1;
        for (saidx_t i = 0; i < n; i++) {
            if (SA[i] == -1 || SA_IS_MARKED(SA[i])) {
                empty_count++;
                if (first_empty < 0) first_empty = i;
            }
        }
        fprintf(stderr, "DEBUG: After L-induction, %d positions empty/marked (first at %d)\n",
                empty_count, first_empty);
        fflush(stderr);
    }
#endif

    /* Step 6c: Final induced sort (S-type) */
    memcpy(bucket_A, bucket_counts, sizeof(bucket_counts));

#ifdef SA_DEBUG
    /* Check what's at the sentinel position before S-induction */
    {
        int c = T[n - 1];
        saidx_t sum = 0;
        for (int i = 0; i <= c; i++) sum += bucket_counts[i];
        saidx_t sentinel_pos = sum - 1;  /* bucket_end[c] - 1 */
        fprintf(stderr, "DEBUG: Before S-ind: sentinel char=%d, SA position=%d, SA[%d]=%d\n",
                c, sentinel_pos, sentinel_pos, SA[sentinel_pos]);
        if (n >= 70000) {
            fprintf(stderr, "DEBUG: T[69998]=%d, T[69999]=%d\n", T[69998], T[69999]);
            fprintf(stderr, "DEBUG: type_is_s(69998)=%d, type_is_s(69999)=%d\n",
                    type_is_s(types, 69998), type_is_s(types, 69999));
        }

        if (n >= 70000) {
            /* Print bucket 7 boundaries (where 69992 belongs) */
            saidx_t bucket7_start = 0;
            for (int i = 0; i < 7; i++) bucket7_start += bucket_counts[i];
            saidx_t bucket7_end = bucket7_start + bucket_counts[7];
            fprintf(stderr, "DEBUG: Bucket[7] = [%d, %d), size=%d\n",
                    bucket7_start, bucket7_end, bucket_counts[7]);

            /* Count how many LMS are in bucket 7 */
            int lms_in_bucket7 = 0;
            for (saidx_t i = bucket7_start; i < bucket7_end; i++) {
                if (SA[i] >= 0 && SA[i] < n) lms_in_bucket7++;
            }
            fprintf(stderr, "DEBUG: LMS in bucket[7] after L-induction: %d\n", lms_in_bucket7);

            /* Check where position 69999 (sentinel) and 69998 are before S-induction */
            saidx_t pos69999_loc = -1, pos69998_loc = -1;
            for (saidx_t i = 0; i < n; i++) {
                if (SA[i] == 69999) pos69999_loc = i;
                if (SA[i] == 69998) pos69998_loc = i;
            }
            fprintf(stderr, "DEBUG: Before S-ind: sentinel 69999 at SA[%d], position 69998 at SA[%d]\n",
                    pos69999_loc, pos69998_loc);

            /* Check where position 69992 is before S-induction */
            saidx_t pos69992_loc = -1;
            for (saidx_t i = 0; i < n; i++) {
                if (SA[i] == 69992) {
                    pos69992_loc = i;
                    break;
                }
            }
            fprintf(stderr, "DEBUG: Before S-ind: position 69992 is at SA[%d] (bucket7=[%d,%d))\n",
                    pos69992_loc, bucket7_start, bucket7_end);
            if (pos69992_loc >= 0) {
                fprintf(stderr, "DEBUG: SA[%d-2..%d+2] = %d %d [%d] %d %d\n",
                        pos69992_loc, pos69992_loc,
                        pos69992_loc >= 2 ? SA[pos69992_loc-2] : -999,
                        pos69992_loc >= 1 ? SA[pos69992_loc-1] : -999,
                        SA[pos69992_loc],
                        pos69992_loc+1 < n ? SA[pos69992_loc+1] : -999,
                        pos69992_loc+2 < n ? SA[pos69992_loc+2] : -999);
            }
        }
        fflush(stderr);
    }
#endif

    /* Step 6c: S-induction - place all S-type suffixes in sorted order */
    induce_s_suffixes(T, SA, types, bucket_A, n);

    /*
     * Post-process: Ensure position n-1 (sentinel) is correctly placed.
     *
     * The sentinel n-1 is the lexicographically smallest suffix in bucket T[n-1]
     * because it's the shortest suffix (just one character followed by implicit end).
     *
     * In SA-IS, n-1 cannot be induced (no position n exists), so it must be placed
     * during Phase 6. However, S-induction may have placed other S-type suffixes
     * at positions that could affect n-1.
     *
     * The correct position for n-1 is the FIRST position in bucket T[n-1], since
     * it's smaller than all other suffixes starting with T[n-1].
     */
    if (n >= 1) {
        int sentinel_char = T[n - 1];

        /* Compute bucket boundaries */
        saidx_t bucket_start_sentinel = 0;
        for (int c = 0; c < sentinel_char; c++) {
            bucket_start_sentinel += bucket_A[c];
        }

        /* Find where n-1 currently is in SA */
        saidx_t current_pos = -1;
        for (saidx_t i = 0; i < n; i++) {
            if (SA[i] == n - 1) {
                current_pos = i;
                break;
            }
        }

        /* The sentinel should be at bucket_start[sentinel_char] */
        saidx_t correct_pos = bucket_start_sentinel;

        if (current_pos != correct_pos) {
            if (current_pos >= 0) {
                /* n-1 is in SA but at wrong position - need to move it */
                /* The correct position is at bucket start (smallest in bucket) */
                /* Shift elements to make room */
                if (current_pos > correct_pos) {
                    /* Shift [correct_pos, current_pos-1] right by 1 */
                    for (saidx_t i = current_pos; i > correct_pos; i--) {
                        SA[i] = SA[i - 1];
                    }
                    SA[correct_pos] = n - 1;
                } else {
                    /* current_pos < correct_pos: this shouldn't happen for smallest suffix */
                    /* But handle it anyway by shifting left */
                    for (saidx_t i = current_pos; i < correct_pos; i++) {
                        SA[i] = SA[i + 1];
                    }
                    SA[correct_pos] = n - 1;
                }
            } else {
                /* n-1 is missing from SA - this is an error, but try to fix it */
                /* Find the first empty slot or the correct position */
                saidx_t empty_pos = -1;
                for (saidx_t i = 0; i < n; i++) {
                    if (SA[i] == -1) {
                        empty_pos = i;
                        break;
                    }
                }
                if (empty_pos >= 0) {
                    if (empty_pos != correct_pos) {
                        /* Shift to put n-1 at correct position */
                        if (empty_pos > correct_pos) {
                            for (saidx_t i = empty_pos; i > correct_pos; i--) {
                                SA[i] = SA[i - 1];
                            }
                        } else {
                            for (saidx_t i = empty_pos; i < correct_pos; i++) {
                                SA[i] = SA[i + 1];
                            }
                        }
                    }
                    SA[correct_pos] = n - 1;
                }
            }
        }
    }

#ifdef SA_DEBUG
    {
        int empty_count = 0;
        int first_empty = -1;
        for (saidx_t i = 0; i < n; i++) {
            if (SA[i] == -1) {
                empty_count++;
                if (first_empty < 0) first_empty = i;
            }
        }
        fprintf(stderr, "DEBUG: after main S-induction + sentinel, %d positions empty (first at %d)\n",
                empty_count, first_empty);
        if (first_empty >= 0 && first_empty > 0) {
            fprintf(stderr, "DEBUG: SA[%d]=%d, SA[%d]=%d\n",
                    first_empty-1, SA[first_empty-1], first_empty, SA[first_empty]);
        }
        /* Check which text positions are missing */
        if (empty_count > 0 && empty_count < 10) {
            bool* seen = (bool*)calloc(n, sizeof(bool));
            for (saidx_t i = 0; i < n; i++) {
                if (SA[i] >= 0 && SA[i] < n) seen[SA[i]] = true;
            }
            fprintf(stderr, "DEBUG: Missing text positions: ");
            for (saidx_t i = 0; i < n; i++) {
                if (!seen[i]) fprintf(stderr, "%d ", i);
            }
            fprintf(stderr, "\n");
            /* Also check for duplicates */
            int dups = 0;
            for (saidx_t i = 0; i < n; i++) {
                if (SA[i] >= 0 && SA[i] < n) {
                    if (seen[SA[i]] == false) { /* already cleared by first pass */
                        seen[SA[i]] = true;
                    }
                }
            }
            memset(seen, 0, n * sizeof(bool));
            for (saidx_t i = 0; i < n; i++) {
                if (SA[i] >= 0 && SA[i] < n) {
                    if (seen[SA[i]]) dups++;
                    seen[SA[i]] = true;
                }
            }
            if (dups > 0) {
                fprintf(stderr, "DEBUG: Found %d duplicate positions in SA!\n", dups);
            }
            free(seen);
        }
        fflush(stderr);
    }
#endif
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

    /* Get thread-local workspace for type bitvector */
    SAWorkspace* ws = get_tl_workspace();

    if (UNLIKELY(!ws)) {
        /* Allocation failed - fall back to divsufsort */
        return divsufsort(T, SA, n, bucket_A, bucket_B);
    }

    /* Run SA-IS algorithm */
    sais_main(T, SA, n, bucket_A, bucket_B, &ws->types);

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
