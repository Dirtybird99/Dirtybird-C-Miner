/*
 * bucket_optimized.h - Cache-optimized bucket operations for divsufsort
 *
 * The bucket_B array (256KB for 256x256 character pairs) exceeds L2 cache
 * on most CPUs, making it a significant bottleneck. This header provides
 * optimized functions for bucket initialization and access.
 *
 * OPTIMIZATIONS:
 * 1. SIMD-accelerated initialization using AVX2/SSE
 * 2. Streaming stores to avoid cache pollution
 * 3. Prefetch hints for anticipated access patterns
 * 4. Cache blocking for better locality
 * 5. Coalesced bucket representation for sparse character pairs
 *
 * Copyright (c) 2025 DERO Miner Project
 * License: MIT
 */

#ifndef BUCKET_OPTIMIZED_H
#define BUCKET_OPTIMIZED_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <string.h>
#include "divsufsort.h"

/* ============================================================================
 * Configuration
 * ============================================================================ */

/* Enable AVX2 optimizations if available */
#ifndef ENABLE_BUCKET_AVX2
    #if defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__))
        #define ENABLE_BUCKET_AVX2 1
    #else
        #define ENABLE_BUCKET_AVX2 0
    #endif
#endif

/* Enable streaming stores (non-temporal) for large bucket init */
#ifndef ENABLE_BUCKET_STREAMING_STORES
    #define ENABLE_BUCKET_STREAMING_STORES 1
#endif

/* Bucket sizes */
#define BUCKET_A_ENTRIES 256
#define BUCKET_B_ENTRIES 65536

/* Bucket B memory: 256KB */
#define BUCKET_B_BYTES (BUCKET_B_ENTRIES * sizeof(saidx_t))

/* Cache line size (64 bytes on modern x86) */
#ifndef CACHE_LINE_BYTES
#define CACHE_LINE_BYTES 64
#endif

/* Entries per cache line for saidx_t (int32_t) */
#define ENTRIES_PER_CACHE_LINE (CACHE_LINE_BYTES / sizeof(saidx_t))

/* ============================================================================
 * SIMD Includes
 * ============================================================================ */

#if ENABLE_BUCKET_AVX2
    #if defined(_MSC_VER)
        #include <intrin.h>
    #else
        #include <immintrin.h>
    #endif
#endif

/* ============================================================================
 * Optimized Bucket Initialization
 * ============================================================================ */

/**
 * Initialize bucket_B array to zero using optimized method.
 *
 * For 256KB, uses:
 * - AVX2 streaming stores (non-temporal) when available
 * - Falls back to memset which compilers optimize well
 *
 * @param bucket_B Pointer to bucket_B array (must be 65536 entries)
 */
static inline void bucket_b_init_optimized(saidx_t *bucket_B) {
#if ENABLE_BUCKET_AVX2 && ENABLE_BUCKET_STREAMING_STORES
    /* AVX2 streaming stores - write directly to memory bypassing cache
     * This is optimal for bucket_B since:
     * 1. We're writing all 256KB, polluting cache would evict useful data
     * 2. We don't read the values immediately after writing
     * 3. Non-temporal stores don't require read-for-ownership
     */
    {
        __m256i zero = _mm256_setzero_si256();
        saidx_t *p = bucket_B;
        saidx_t *end = bucket_B + BUCKET_B_ENTRIES;

        /* Process 64 bytes (16 int32_t) per iteration */
        while (p < end) {
            _mm256_stream_si256((__m256i*)(p), zero);
            _mm256_stream_si256((__m256i*)(p + 8), zero);
            p += 16;
        }
        /* Ensure all streaming stores complete before subsequent loads */
        _mm_sfence();
    }
#else
    /* Fallback to memset - compilers optimize this well */
    memset(bucket_B, 0, BUCKET_B_BYTES);
#endif
}

/**
 * Initialize bucket_A array to zero.
 * bucket_A is only 1KB, fits in L1, so simple memset is optimal.
 *
 * @param bucket_A Pointer to bucket_A array (must be 256 entries)
 */
static inline void bucket_a_init_optimized(saidx_t *bucket_A) {
    memset(bucket_A, 0, BUCKET_A_ENTRIES * sizeof(saidx_t));
}

/* ============================================================================
 * Prefetch Utilities for Bucket Access
 * ============================================================================ */

/**
 * Prefetch a bucket_B entry for reading.
 * Use T2 hint (L2 cache) since bucket_B is too large for L1.
 *
 * @param bucket_B Pointer to bucket_B array
 * @param c0 First character of pair
 * @param c1 Second character of pair
 */
static inline void bucket_b_prefetch_read(const saidx_t *bucket_B,
                                          uint8_t c0, uint8_t c1) {
#if defined(__GNUC__) || defined(__clang__)
    /* Use L2 hint since bucket_B is large */
    __builtin_prefetch(&bucket_B[(c1 << 8) | c0], 0, 2);
#elif defined(_MSC_VER)
    _mm_prefetch((const char*)&bucket_B[(c1 << 8) | c0], _MM_HINT_T1);
#endif
}

/**
 * Prefetch a bucket_B entry for writing.
 *
 * @param bucket_B Pointer to bucket_B array
 * @param c0 First character of pair
 * @param c1 Second character of pair
 */
static inline void bucket_b_prefetch_write(saidx_t *bucket_B,
                                           uint8_t c0, uint8_t c1) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(&bucket_B[(c1 << 8) | c0], 1, 2);
#elif defined(_MSC_VER)
    _mm_prefetch((const char*)&bucket_B[(c1 << 8) | c0], _MM_HINT_T1);
#endif
}

/**
 * Prefetch multiple bucket_B entries for a range of second characters.
 * Useful when iterating through buckets with fixed c0.
 *
 * @param bucket_B Pointer to bucket_B array
 * @param c0 First character (fixed)
 * @param c1_start Start of second character range
 * @param c1_end End of second character range
 */
static inline void bucket_b_prefetch_range(const saidx_t *bucket_B,
                                           uint8_t c0,
                                           uint8_t c1_start,
                                           uint8_t c1_end) {
    /* Prefetch cache lines covering the range */
    for (uint8_t c1 = c1_start; c1 <= c1_end; c1 += ENTRIES_PER_CACHE_LINE) {
#if defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(&bucket_B[(c1 << 8) | c0], 0, 2);
#elif defined(_MSC_VER)
        _mm_prefetch((const char*)&bucket_B[(c1 << 8) | c0], _MM_HINT_T1);
#endif
    }
}

/* ============================================================================
 * Cache-Blocked Bucket Operations
 * ============================================================================ */

/**
 * Cache block size for bucket_B operations.
 * Process bucket_B in blocks that fit in L2 cache.
 * 256KB / 4 = 64KB blocks = 16384 entries
 */
#define BUCKET_B_BLOCK_SIZE 16384

/**
 * Process bucket_B in cache-friendly blocks.
 * Callback is invoked for each block with start/end indices.
 *
 * @param bucket_B Pointer to bucket_B array
 * @param callback Function to call for each block
 * @param user_data User data passed to callback
 */
typedef void (*bucket_b_block_callback)(saidx_t *bucket_B,
                                        size_t block_start,
                                        size_t block_end,
                                        void *user_data);

static inline void bucket_b_process_blocked(saidx_t *bucket_B,
                                            bucket_b_block_callback callback,
                                            void *user_data) {
    for (size_t start = 0; start < BUCKET_B_ENTRIES; start += BUCKET_B_BLOCK_SIZE) {
        size_t end = start + BUCKET_B_BLOCK_SIZE;
        if (end > BUCKET_B_ENTRIES) end = BUCKET_B_ENTRIES;
        callback(bucket_B, start, end, user_data);
    }
}

/* ============================================================================
 * Bucket Coalescing for Sparse Access Patterns
 * ============================================================================ */

/**
 * Coalesced bucket entry - stores (c0, c1, value) tuples.
 * For inputs with sparse character distribution, this can be more
 * cache-efficient than the full 256x256 array.
 */
typedef struct {
    uint16_t key;      /* (c1 << 8) | c0 */
    saidx_t value;
} CoalescedBucketEntry;

/**
 * Coalesced bucket structure.
 * Uses sorted array + binary search for sparse access patterns.
 */
typedef struct {
    CoalescedBucketEntry *entries;
    size_t count;
    size_t capacity;
} CoalescedBucket;

/**
 * Initialize coalesced bucket.
 *
 * @param bucket Pointer to coalesced bucket structure
 * @param initial_capacity Initial capacity (number of unique pairs)
 * @return 0 on success, -1 on allocation failure
 */
static inline int coalesced_bucket_init(CoalescedBucket *bucket,
                                        size_t initial_capacity) {
    bucket->entries = (CoalescedBucketEntry*)malloc(
        initial_capacity * sizeof(CoalescedBucketEntry));
    if (!bucket->entries) return -1;
    bucket->count = 0;
    bucket->capacity = initial_capacity;
    return 0;
}

/**
 * Free coalesced bucket.
 *
 * @param bucket Pointer to coalesced bucket structure
 */
static inline void coalesced_bucket_free(CoalescedBucket *bucket) {
    free(bucket->entries);
    bucket->entries = NULL;
    bucket->count = 0;
    bucket->capacity = 0;
}

/**
 * Binary search for key in coalesced bucket.
 *
 * @param bucket Pointer to coalesced bucket
 * @param key Key to search for ((c1 << 8) | c0)
 * @return Index if found, or insertion point (negated - 1) if not found
 */
static inline ssize_t coalesced_bucket_search(const CoalescedBucket *bucket,
                                              uint16_t key) {
    ssize_t lo = 0, hi = (ssize_t)bucket->count - 1;
    while (lo <= hi) {
        ssize_t mid = lo + (hi - lo) / 2;
        if (bucket->entries[mid].key == key) {
            return mid;
        } else if (bucket->entries[mid].key < key) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return -(lo + 1);  /* Insertion point */
}

/* ============================================================================
 * SIMD-Accelerated Bucket Counting
 * ============================================================================ */

#if ENABLE_BUCKET_AVX2

/**
 * Count character occurrences using AVX2.
 * Processes 32 characters at a time.
 *
 * @param T Input text
 * @param n Length of input
 * @param bucket_A Output bucket_A array (256 entries)
 */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2")))
#endif
static inline void bucket_a_count_avx2(const uint8_t *T, size_t n,
                                        saidx_t *bucket_A) {
    /* Use multiple histogram accumulators to reduce memory port contention */
    saidx_t hist0[256] = {0};
    saidx_t hist1[256] = {0};
    saidx_t hist2[256] = {0};
    saidx_t hist3[256] = {0};

    size_t i = 0;
    size_t n4 = n & ~3;

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

    /* Merge histograms using AVX2 */
    for (int c = 0; c < 256; c += 8) {
        __m256i h0 = _mm256_loadu_si256((const __m256i*)&hist0[c]);
        __m256i h1 = _mm256_loadu_si256((const __m256i*)&hist1[c]);
        __m256i h2 = _mm256_loadu_si256((const __m256i*)&hist2[c]);
        __m256i h3 = _mm256_loadu_si256((const __m256i*)&hist3[c]);

        __m256i sum01 = _mm256_add_epi32(h0, h1);
        __m256i sum23 = _mm256_add_epi32(h2, h3);
        __m256i total = _mm256_add_epi32(sum01, sum23);

        _mm256_storeu_si256((__m256i*)&bucket_A[c], total);
    }
}

#endif /* ENABLE_BUCKET_AVX2 */

/* ============================================================================
 * Diagnostic Functions
 * ============================================================================ */

/**
 * Analyze bucket_B sparsity.
 * Returns the number of non-zero entries.
 *
 * @param bucket_B Pointer to bucket_B array
 * @return Number of non-zero entries
 */
static inline size_t bucket_b_count_nonzero(const saidx_t *bucket_B) {
    size_t count = 0;
    for (size_t i = 0; i < BUCKET_B_ENTRIES; i++) {
        if (bucket_B[i] != 0) count++;
    }
    return count;
}

/**
 * Calculate bucket_B utilization percentage.
 *
 * @param bucket_B Pointer to bucket_B array
 * @return Utilization as percentage (0.0 to 100.0)
 */
static inline double bucket_b_utilization(const saidx_t *bucket_B) {
    size_t nonzero = bucket_b_count_nonzero(bucket_B);
    return (100.0 * nonzero) / BUCKET_B_ENTRIES;
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* BUCKET_OPTIMIZED_H */
