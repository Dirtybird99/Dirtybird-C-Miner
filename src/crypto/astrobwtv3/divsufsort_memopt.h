/**
 * Memory-Optimized divsufsort Enhancements
 *
 * This header provides additional prefetch optimizations for divsufsort
 * to reduce cache misses on the 256KB bucket_B array.
 *
 * Key insight: bucket_B is accessed based on character pairs from the text.
 * By prefetching bucket entries ahead of time, we can hide memory latency.
 */

#ifndef DIVSUFSORT_MEMOPT_H
#define DIVSUFSORT_MEMOPT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// BUCKET_B MEMORY ANALYSIS
// ============================================================================
/*
 * bucket_B dimensions: 256 x 256 = 65536 entries
 * Entry size: 4 bytes (int32_t)
 * Total size: 262,144 bytes = 256KB
 *
 * Cache hierarchy impact:
 * - L1D (32KB): Can hold ~12.5% of bucket_B
 * - L2 (256KB): Can hold 100% of bucket_B (barely)
 * - L3 (shared): bucket_B competes with SA and text data
 *
 * Access pattern: Semi-random based on character pairs
 * - Sequential text scan determines (c0, c1) pairs
 * - Bucket access = bucket_B[c1 * 256 + c0]
 *
 * Problem: Random bucket access causes L2 cache misses
 * Solution: Prefetch bucket entries based on text lookahead
 */

// ============================================================================
// PREFETCH CONFIGURATION
// ============================================================================

/**
 * Tunable prefetch distances for bucket_B.
 * These should be adjusted based on:
 * - Memory latency (higher latency = larger distance)
 * - Cache size (smaller cache = smaller distance to avoid thrashing)
 * - Text locality (more random text = less effective prefetch)
 */
#ifndef BUCKET_B_PF_INITIAL
#define BUCKET_B_PF_INITIAL 12  // Distance for initial scan
#endif

#ifndef BUCKET_B_PF_BSTAR
#define BUCKET_B_PF_BSTAR 8     // Distance for B* suffix sorting
#endif

#ifndef BUCKET_B_PF_CONSTRUCT
#define BUCKET_B_PF_CONSTRUCT 16 // Distance for SA construction
#endif

// ============================================================================
// PREFETCH MACROS
// ============================================================================

#if defined(__GNUC__) || defined(__clang__)
    #define PF_BUCKET_READ(addr)  __builtin_prefetch((addr), 0, 2)  // L2 hint
    #define PF_BUCKET_WRITE(addr) __builtin_prefetch((addr), 1, 2)  // Write intent
#elif defined(_MSC_VER)
    #include <intrin.h>
    #define PF_BUCKET_READ(addr)  _mm_prefetch((const char*)(addr), _MM_HINT_T1)
    #define PF_BUCKET_WRITE(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T1)
#else
    #define PF_BUCKET_READ(addr)  ((void)0)
    #define PF_BUCKET_WRITE(addr) ((void)0)
#endif

// ============================================================================
// INLINE PREFETCH FUNCTIONS
// ============================================================================

/**
 * Prefetch bucket_B entry for character pair (c0, c1).
 *
 * @param bucket_B Pointer to bucket_B array
 * @param c0 First character (0-255)
 * @param c1 Second character (0-255)
 */
static inline void prefetch_bucket_b_entry(int* bucket_B, int c0, int c1) {
    // BUCKET_B(c0, c1) = bucket_B[(c1) * 256 + (c0)]
    PF_BUCKET_READ(&bucket_B[c1 * 256 + c0]);
}

/**
 * Prefetch bucket_B entry with write intent.
 */
static inline void prefetch_bucket_b_write(int* bucket_B, int c0, int c1) {
    PF_BUCKET_WRITE(&bucket_B[c1 * 256 + c0]);
}

/**
 * Prefetch multiple bucket_B entries based on text lookahead.
 *
 * @param bucket_B Pointer to bucket_B array
 * @param T Text array
 * @param pos Current position in text
 * @param distance How many positions ahead to prefetch
 */
static inline void prefetch_bucket_b_ahead(int* bucket_B, const unsigned char* T,
                                            int pos, int n, int distance) {
    if (pos + distance + 1 < n) {
        int c0 = T[pos + distance];
        int c1 = T[pos + distance + 1];
        prefetch_bucket_b_entry(bucket_B, c0, c1);
    }
}

// ============================================================================
// BUCKET_B PARTITIONING FOR CACHE EFFICIENCY
// ============================================================================

/**
 * Bucket locality hint: character pairs with similar c1 values
 * access nearby memory in bucket_B.
 *
 * bucket_B memory layout (row-major by c1):
 *   c1=0: [bucket_B[0], bucket_B[1], ..., bucket_B[255]]
 *   c1=1: [bucket_B[256], bucket_B[257], ..., bucket_B[511]]
 *   ...
 *   c1=255: [bucket_B[65280], ..., bucket_B[65535]]
 *
 * Cache optimization: Process text in order to maximize c1 locality.
 * When c1 changes, the entire row (1KB) must be loaded.
 */

/**
 * Calculate the cache line for a bucket_B entry.
 * Useful for detecting cache line conflicts.
 *
 * @return Cache line index (0-4095 for 64-byte lines)
 */
static inline int bucket_b_cache_line(int c0, int c1) {
    int offset = c1 * 256 + c0;  // Entry index
    int byte_offset = offset * sizeof(int);  // Byte offset
    return byte_offset / 64;  // Cache line (64 bytes)
}

// ============================================================================
// SA PREFETCH OPTIMIZATION
// ============================================================================

/**
 * The suffix array (SA) is accessed randomly during induced sorting.
 * Prefetch SA entries based on bucket positions.
 *
 * SA size: 70KB * 4 bytes = 280KB
 * Access pattern: Random, driven by text content
 */

#ifndef SA_PF_DISTANCE
#define SA_PF_DISTANCE 16
#endif

/**
 * Prefetch SA entry at position.
 */
static inline void prefetch_sa(int* SA, int pos) {
    PF_BUCKET_READ(&SA[pos]);
}

/**
 * Prefetch SA entries in a range.
 */
static inline void prefetch_sa_range(int* SA, int start, int count) {
    for (int i = 0; i < count; i += 16) {  // One per cache line
        PF_BUCKET_READ(&SA[start + i]);
    }
}

// ============================================================================
// TEXT PREFETCH OPTIMIZATION
// ============================================================================

/**
 * Text is scanned sequentially in most phases.
 * Prefetch helps when processing jumps (SA[i] - 1).
 */

#ifndef TEXT_PF_DISTANCE
#define TEXT_PF_DISTANCE 24
#endif

/**
 * Prefetch text byte at position.
 */
static inline void prefetch_text(const unsigned char* T, int pos) {
    PF_BUCKET_READ(&T[pos]);
}

// ============================================================================
// MEMORY BANDWIDTH ESTIMATION
// ============================================================================

/*
 * Estimated memory traffic per divsufsort call (70KB input):
 *
 * Phase 1 - Type classification:
 *   - Text scan: 70KB read (sequential)
 *   - bucket_A updates: 1KB write
 *   - bucket_B updates: ~10KB read/write (depends on text)
 *   - SA writes (B* positions): ~5KB
 *
 * Phase 2 - B* suffix sorting:
 *   - SA read/write: ~50KB
 *   - bucket_B read: ~20KB
 *   - Text random reads: ~100KB (multiple comparisons)
 *
 * Phase 3 - Induced sorting:
 *   - SA scan: 280KB read + write
 *   - bucket_B reads: ~50KB
 *   - Text random reads: ~200KB
 *
 * Total estimated: ~800KB - 1MB memory traffic per call
 *
 * Cache miss penalty:
 * - L2 miss: ~20 cycles
 * - L3 miss: ~50-100 cycles
 * - DRAM miss: ~200+ cycles
 *
 * Prefetch can hide ~80% of memory latency when effective.
 */

#ifdef __cplusplus
}
#endif

#endif // DIVSUFSORT_MEMOPT_H
