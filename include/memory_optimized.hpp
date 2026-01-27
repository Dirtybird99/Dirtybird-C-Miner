/**
 * Memory Access Pattern Optimizations for DirtyBird DERO Miner
 *
 * This header provides cache-optimized data structures and memory access patterns
 * to reduce bandwidth and improve cache efficiency.
 *
 * Key Optimizations:
 * 1. Cache-aligned hot data structures
 * 2. Prefetch hints for predictable access patterns
 * 3. Streaming stores for large writes
 * 4. Memory traffic reduction techniques
 *
 * Analysis Summary:
 * - Working set per thread: ~1.2MB
 * - Hot path (wolfCompute): 250-400KB memory traffic per hash
 * - Main bottlenecks: memcpy(256B) per iteration, divsufsort bucket_B (256KB)
 */

#pragma once

#include <cstdint>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
  #include <immintrin.h>
#endif

#if defined(__aarch64__)
  #include <arm_neon.h>
#endif

// ============================================================================
// CACHE LINE CONSTANTS
// ============================================================================

#define CACHE_LINE_SIZE 64
#define L1_CACHE_SIZE   (32 * 1024)     // 32KB typical L1D
#define L2_CACHE_SIZE   (256 * 1024)    // 256KB typical L2
#define L3_CACHE_SIZE   (8 * 1024 * 1024) // 8MB typical L3 slice

// Alignment macros
#define CACHE_ALIGNED __attribute__((aligned(CACHE_LINE_SIZE)))
#define L2_ALIGNED    __attribute__((aligned(128)))

// ============================================================================
// WORKING SET ANALYSIS
// ============================================================================
/*
 * Per-thread memory breakdown for workerData:
 *
 * Hot data (accessed every iteration):
 *   - chunk/prev_chunk pointers: 16 bytes
 *   - pos1, pos2, op, A: 4 bytes
 *   - lhash, prev_lhash: 16 bytes
 *   - random_switcher: 8 bytes
 *   Total hot: ~64 bytes (1 cache line)
 *
 * Warm data (accessed frequently):
 *   - sData[71680]: 70KB (278 chunks * 256 bytes)
 *   - key[]: RC4 key state ~268 bytes
 *   - maskTable_bytes[1056]: 1KB
 *   Total warm: ~72KB
 *
 * Cold data (accessed once per hash):
 *   - sa[70913]: 283KB (suffix array output)
 *   - bA[256], bB[65536]: 257KB (divsufsort workspace)
 *   - SPSA buckets: 640KB (if enabled)
 *   Total cold: ~540KB (or ~1.2MB with SPSA)
 */

// ============================================================================
// MEMORY COPY OPTIMIZATIONS
// ============================================================================

/**
 * Optimized 256-byte copy using streaming stores.
 * Avoids polluting cache when destination won't be read soon.
 *
 * Use when:
 * - Source is already in cache (prev_chunk)
 * - Destination will be modified before being read again
 */
#if defined(__AVX512F__)
static inline void memcpy256_stream(void* __restrict dst, const void* __restrict src) {
    const __m512i* s = (const __m512i*)src;
    __m512i* d = (__m512i*)dst;

    // Non-temporal stores bypass cache
    _mm512_stream_si512(d + 0, _mm512_loadu_si512(s + 0));
    _mm512_stream_si512(d + 1, _mm512_loadu_si512(s + 1));
    _mm512_stream_si512(d + 2, _mm512_loadu_si512(s + 2));
    _mm512_stream_si512(d + 3, _mm512_loadu_si512(s + 3));

    // Ensure stores are visible
    _mm_sfence();
}
#elif defined(__AVX2__)
static inline void memcpy256_stream(void* __restrict dst, const void* __restrict src) {
    const __m256i* s = (const __m256i*)src;
    __m256i* d = (__m256i*)dst;

    _mm256_stream_si256(d + 0, _mm256_loadu_si256(s + 0));
    _mm256_stream_si256(d + 1, _mm256_loadu_si256(s + 1));
    _mm256_stream_si256(d + 2, _mm256_loadu_si256(s + 2));
    _mm256_stream_si256(d + 3, _mm256_loadu_si256(s + 3));
    _mm256_stream_si256(d + 4, _mm256_loadu_si256(s + 4));
    _mm256_stream_si256(d + 5, _mm256_loadu_si256(s + 5));
    _mm256_stream_si256(d + 6, _mm256_loadu_si256(s + 6));
    _mm256_stream_si256(d + 7, _mm256_loadu_si256(s + 7));

    _mm_sfence();
}
#else
static inline void memcpy256_stream(void* dst, const void* src) {
    memcpy(dst, src, 256);
}
#endif

/**
 * Optimized 256-byte copy with cache retention.
 * Uses regular stores so data stays in cache.
 *
 * Use when:
 * - Destination will be read soon (wolfPermute input)
 */
#if defined(__AVX512F__)
static inline void memcpy256_cached(void* __restrict dst, const void* __restrict src) {
    const __m512i* s = (const __m512i*)src;
    __m512i* d = (__m512i*)dst;

    _mm512_storeu_si512(d + 0, _mm512_loadu_si512(s + 0));
    _mm512_storeu_si512(d + 1, _mm512_loadu_si512(s + 1));
    _mm512_storeu_si512(d + 2, _mm512_loadu_si512(s + 2));
    _mm512_storeu_si512(d + 3, _mm512_loadu_si512(s + 3));
}
#elif defined(__AVX2__)
static inline void memcpy256_cached(void* __restrict dst, const void* __restrict src) {
    const __m256i* s = (const __m256i*)src;
    __m256i* d = (__m256i*)dst;

    _mm256_storeu_si256(d + 0, _mm256_loadu_si256(s + 0));
    _mm256_storeu_si256(d + 1, _mm256_loadu_si256(s + 1));
    _mm256_storeu_si256(d + 2, _mm256_loadu_si256(s + 2));
    _mm256_storeu_si256(d + 3, _mm256_loadu_si256(s + 3));
    _mm256_storeu_si256(d + 4, _mm256_loadu_si256(s + 4));
    _mm256_storeu_si256(d + 5, _mm256_loadu_si256(s + 5));
    _mm256_storeu_si256(d + 6, _mm256_loadu_si256(s + 6));
    _mm256_storeu_si256(d + 7, _mm256_loadu_si256(s + 7));
}
#elif defined(__aarch64__)
static inline void memcpy256_cached(void* __restrict dst, const void* __restrict src) {
    const uint8x16_t* s = (const uint8x16_t*)src;
    uint8x16_t* d = (uint8x16_t*)dst;

    for (int i = 0; i < 16; i++) {
        vst1q_u8((uint8_t*)(d + i), vld1q_u8((const uint8_t*)(s + i)));
    }
}
#else
static inline void memcpy256_cached(void* dst, const void* src) {
    memcpy(dst, src, 256);
}
#endif

/**
 * Incremental copy: only copy the range [start, end).
 * Reduces memory traffic when only a small portion changes.
 *
 * Memory savings: (256 - (end - start)) bytes per call
 * At average range of 16 bytes: 240 bytes saved per iteration
 * Over 278 iterations: ~67KB saved per hash
 */
#if defined(__AVX2__)
static inline void memcpy_range_avx2(uint8_t* __restrict dst,
                                      const uint8_t* __restrict src,
                                      uint8_t start, uint8_t end) {
    // Copy everything before start (aligned to 32 bytes)
    int pre_end = start & ~31;
    for (int i = 0; i < pre_end; i += 32) {
        _mm256_storeu_si256((__m256i*)(dst + i), _mm256_loadu_si256((__m256i*)(src + i)));
    }
    // Copy remaining bytes before start
    for (int i = pre_end; i < start; i++) {
        dst[i] = src[i];
    }

    // Copy everything after end (aligned to 32 bytes)
    int post_start = (end + 31) & ~31;
    if (post_start < 256) {
        for (int i = post_start; i < 256; i += 32) {
            _mm256_storeu_si256((__m256i*)(dst + i), _mm256_loadu_si256((__m256i*)(src + i)));
        }
    }
    // Copy remaining bytes after end
    for (int i = end; i < post_start && i < 256; i++) {
        dst[i] = src[i];
    }
}
#endif

// ============================================================================
// PREFETCH UTILITIES
// ============================================================================

/**
 * Prefetch distances tuned for different cache levels
 */
#define PF_L1_DISTANCE  8   // ~8 iterations ahead for L1 (32KB)
#define PF_L2_DISTANCE  24  // ~24 iterations ahead for L2 (256KB)
#define PF_L3_DISTANCE  64  // ~64 iterations ahead for L3

/**
 * Structured prefetch for predictable access patterns.
 * Hint levels:
 *   0 = Non-temporal (don't cache)
 *   1 = T2 (L3 cache)
 *   2 = T1 (L2 cache)
 *   3 = T0 (L1 cache)
 */
#if defined(__GNUC__) || defined(__clang__)
    #define PREFETCH_T0(addr)  __builtin_prefetch((addr), 0, 3)  // L1
    #define PREFETCH_T1(addr)  __builtin_prefetch((addr), 0, 2)  // L2
    #define PREFETCH_T2(addr)  __builtin_prefetch((addr), 0, 1)  // L3
    #define PREFETCH_NTA(addr) __builtin_prefetch((addr), 0, 0)  // Non-temporal
    #define PREFETCH_W(addr)   __builtin_prefetch((addr), 1, 3)  // Write intent
#elif defined(_MSC_VER)
    #include <intrin.h>
    #define PREFETCH_T0(addr)  _mm_prefetch((const char*)(addr), _MM_HINT_T0)
    #define PREFETCH_T1(addr)  _mm_prefetch((const char*)(addr), _MM_HINT_T1)
    #define PREFETCH_T2(addr)  _mm_prefetch((const char*)(addr), _MM_HINT_T2)
    #define PREFETCH_NTA(addr) _mm_prefetch((const char*)(addr), _MM_HINT_NTA)
    #define PREFETCH_W(addr)   _mm_prefetch((const char*)(addr), _MM_HINT_T0)
#else
    #define PREFETCH_T0(addr)  ((void)0)
    #define PREFETCH_T1(addr)  ((void)0)
    #define PREFETCH_T2(addr)  ((void)0)
    #define PREFETCH_NTA(addr) ((void)0)
    #define PREFETCH_W(addr)   ((void)0)
#endif

/**
 * Prefetch a 256-byte chunk into L1 cache.
 * Used before wolfPermute to ensure data is ready.
 */
static inline void prefetch_chunk_L1(const void* addr) {
    const char* p = (const char*)addr;
    PREFETCH_T0(p);
    PREFETCH_T0(p + 64);
    PREFETCH_T0(p + 128);
    PREFETCH_T0(p + 192);
}

/**
 * Prefetch a 256-byte chunk into L2 cache.
 * Used for next iteration's data.
 */
static inline void prefetch_chunk_L2(const void* addr) {
    const char* p = (const char*)addr;
    PREFETCH_T1(p);
    PREFETCH_T1(p + 64);
    PREFETCH_T1(p + 128);
    PREFETCH_T1(p + 192);
}

/**
 * Prefetch with write intent for destination buffer.
 */
static inline void prefetch_chunk_write(void* addr) {
    char* p = (char*)addr;
    PREFETCH_W(p);
    PREFETCH_W(p + 64);
    PREFETCH_W(p + 128);
    PREFETCH_W(p + 192);
}

// ============================================================================
// OPTIMIZED HOT DATA STRUCTURE
// ============================================================================

/**
 * Cache-optimized hot data for wolfCompute inner loop.
 * All frequently-accessed fields in one cache line (64 bytes).
 *
 * This structure can be embedded at the start of workerData
 * to ensure hot data is always cache-aligned.
 */
struct CACHE_ALIGNED HotWorkerData {
    // Frequently read/written (every iteration)
    uint8_t* chunk;           // 8 bytes - current chunk pointer
    uint8_t* prev_chunk;      // 8 bytes - previous chunk pointer
    uint64_t lhash;           // 8 bytes - current hash
    uint64_t prev_lhash;      // 8 bytes - previous hash
    uint64_t random_switcher; // 8 bytes - random state

    // Read every iteration
    uint8_t pos1;             // 1 byte
    uint8_t pos2;             // 1 byte
    uint8_t op;               // 1 byte
    uint8_t A;                // 1 byte

    uint16_t tries;           // 2 bytes - iteration counter
    uint8_t t1, t2;           // 2 bytes - temp swap vars

    // Padding to fill cache line
    uint8_t _pad[16];         // 16 bytes padding

    // Total: exactly 64 bytes (1 cache line)
};

static_assert(sizeof(HotWorkerData) == 64, "HotWorkerData must be exactly one cache line");

// ============================================================================
// MEMORY TRAFFIC REDUCTION
// ============================================================================

/**
 * Optimized wolfCompute iteration with reduced memory traffic.
 *
 * Changes from original:
 * 1. Uses incremental memcpy when range is small
 * 2. Prefetches next iteration's data
 * 3. Uses streaming stores when appropriate
 *
 * Memory traffic reduction:
 * - Original: 256 bytes memcpy every iteration
 * - Optimized: Average ~80 bytes memcpy (considering range size distribution)
 * - Savings: ~50KB per hash
 */

// Threshold for using incremental vs full copy
// If modified range < this, use incremental copy
#define INCREMENTAL_COPY_THRESHOLD 64

/**
 * Decide whether to use incremental or full copy based on range size.
 * Returns true if incremental copy is beneficial.
 */
static inline bool should_use_incremental_copy(uint8_t pos1, uint8_t pos2) {
    // If modified range is small compared to full chunk, use incremental
    return (pos2 - pos1) < INCREMENTAL_COPY_THRESHOLD;
}

// ============================================================================
// DIVSUFSORT MEMORY OPTIMIZATIONS
// ============================================================================

/**
 * bucket_B is 256KB (65536 * 4 bytes), which exceeds L2 cache.
 * Access pattern is semi-random based on text content.
 *
 * Optimization strategies:
 * 1. Prefetch bucket entries based on text lookahead
 * 2. Use software prefetch with appropriate distance
 * 3. Consider bucket locality when possible
 */

// Prefetch distance for bucket_B (tune based on memory latency)
#define BUCKET_B_PF_DISTANCE 8

/**
 * Prefetch bucket_B entry for a character pair.
 * Call this with lookahead characters to hide memory latency.
 */
static inline void prefetch_bucket_b(int* bucket_B, uint8_t c0, uint8_t c1) {
    // bucket_B indexing: BUCKET_B(c0, c1) = bucket_B[(c1) * 256 + (c0)]
    PREFETCH_T1(&bucket_B[c1 * 256 + c0]);
}

/**
 * Batch prefetch multiple bucket_B entries.
 * Useful when processing a sequence of characters.
 */
static inline void prefetch_bucket_b_batch(int* bucket_B,
                                            const uint8_t* text,
                                            int start, int count) {
    for (int i = 0; i < count && start + i + 1 < 70000; i++) {
        uint8_t c0 = text[start + i];
        uint8_t c1 = text[start + i + 1];
        prefetch_bucket_b(bucket_B, c0, c1);
    }
}

// ============================================================================
// MEMORY LAYOUT RECOMMENDATIONS
// ============================================================================

/*
 * Recommended workerData layout for optimal cache performance:
 *
 * Offset 0-63:     HotWorkerData (1 cache line, most frequently accessed)
 * Offset 64-127:   RC4 key state (1 cache line, used every ~4 iterations)
 * Offset 128-1183: maskTable_bytes (aligned, used in AVX2 operations)
 * Offset 1184+:    sData, sa, bA, bB (large arrays, cold data)
 *
 * Key principles:
 * 1. Hot data at start of structure, cache-line aligned
 * 2. Group related data that's accessed together
 * 3. Large arrays at the end to not pollute cache lines
 * 4. Explicit padding to prevent false sharing
 *
 * False sharing prevention:
 * - Each workerData should be allocated on a cache line boundary
 * - malloc_huge_pages already ensures 2MB page alignment
 * - Add 64-byte padding at end if threads share data
 */

// ============================================================================
// EXAMPLE: OPTIMIZED WOLFCOMPUTE ITERATION
// ============================================================================

/*
 * // Pseudocode for optimized iteration:
 *
 * void wolfCompute_optimized(workerData& worker) {
 *     // Prefetch first chunk
 *     prefetch_chunk_L1(&worker.sData[0]);
 *
 *     for (int it = 0; it < 278; ++it) {
 *         // Prefetch next iteration's data while processing current
 *         if (it + 1 < 278) {
 *             prefetch_chunk_L2(&worker.sData[(it + 1) * 256]);
 *         }
 *
 *         // Calculate positions
 *         // ... (existing code)
 *
 *         // Use incremental copy for small changes
 *         if (it > 0) {
 *             if (should_use_incremental_copy(pos1, pos2)) {
 *                 // Copy only unchanged regions
 *                 memcpy_range_avx2(chunk, prev_chunk, pos1, pos2);
 *             } else {
 *                 // Full copy with cache retention
 *                 memcpy256_cached(chunk, prev_chunk);
 *             }
 *         }
 *
 *         // Ensure chunk data is in L1 before wolfPermute
 *         prefetch_chunk_L1(&chunk[pos1]);
 *
 *         // ... rest of iteration
 *     }
 * }
 */

// End of memory_optimized.hpp
