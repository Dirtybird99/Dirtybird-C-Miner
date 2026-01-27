/**
 * DERO Miner Memory Prefetch Strategy
 *
 * This file defines optimized prefetching strategies for the three main
 * phases of AstroBWTv3: Branch Computation, Suffix Array Construction,
 * and SHA-256 Hashing.
 *
 * Memory Access Pattern Analysis:
 * ================================
 *
 * 1. BRANCH COMPUTATION (~40% of hash time)
 *    - Access Pattern: Random 32-byte chunks within 256-byte block
 *    - Positions: pos1 = (tries * 256 + offset) & 0xFF, pos2 derived
 *    - Data Structures: chunk[256], prev_chunk[256], CodeLUT[257], CodeLUT_16[257]
 *    - Cache Behavior: Working set fits in L1 (256 + 256 + 1KB + 514 = ~2KB)
 *
 * 2. SUFFIX ARRAY CONSTRUCTION (~35% of hash time)
 *    - Access Pattern: Bucket-based sorting with random SA access
 *    - Data Structures:
 *      * bucket_A[256] = 1KB (fits in L1)
 *      * bucket_B[65536] = 256KB (exceeds L2, critical to prefetch)
 *      * SA[N] = variable, typically 70K * 4 = 280KB
 *    - Cache Behavior: Bucket B thrashes L2, SA causes L3 pressure
 *
 * 3. SHA-256 HASHING (~25% of hash time)
 *    - Access Pattern: Strictly sequential 64-byte blocks
 *    - Data Structures: SA output as input, 32-byte hash state
 *    - Cache Behavior: Perfect streaming, benefits from streaming stores
 *
 * Prefetch Hints Reference:
 * =========================
 * _MM_HINT_T0  (3): Prefetch to all cache levels (L1+L2+L3) - for imminent use
 * _MM_HINT_T1  (2): Prefetch to L2 and L3 - for near-future use
 * _MM_HINT_T2  (1): Prefetch to L3 only - for later use
 * _MM_HINT_NTA (0): Non-temporal hint - streaming data, minimize cache pollution
 */

#ifndef PREFETCH_STRATEGY_HPP
#define PREFETCH_STRATEGY_HPP

#include <cstdint>
#include <cstddef>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#include <xmmintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

// ============================================================================
// CACHE PARAMETERS (tune for target CPU)
// ============================================================================

namespace prefetch {

// Cache line size (64 bytes on x86-64, 64-128 on ARM)
constexpr size_t CACHE_LINE_SIZE = 64;

// L1 Data Cache: 32-48KB typical, ~4 cycle latency
constexpr size_t L1_SIZE = 32 * 1024;

// L2 Cache: 256KB-1MB typical, ~12 cycle latency
constexpr size_t L2_SIZE = 512 * 1024;

// L3 Cache: 8-32MB typical, ~40 cycle latency
constexpr size_t L3_SIZE = 16 * 1024 * 1024;

// Memory latency in cycles (for prefetch distance calculation)
constexpr int MEMORY_LATENCY_CYCLES = 200;

// Instructions per cycle (IPC) estimate for compute-bound sections
constexpr double ESTIMATED_IPC = 2.0;

// ============================================================================
// PREFETCH DISTANCE CALCULATIONS
// ============================================================================

/**
 * Optimal prefetch distance formula:
 * distance = memory_latency_cycles / (cycles_per_iteration)
 *
 * For branch computation: ~50 cycles/iteration -> distance = 4 iterations
 * For SA construction: ~20 cycles/element -> distance = 10 elements
 * For SHA-256: ~100 cycles/block -> distance = 2 blocks (128 bytes)
 */

// Branch computation: prefetch 4 iterations ahead (4 * 32 = 128 bytes)
constexpr int BRANCH_PREFETCH_DISTANCE = 4;

// ============================================================================
// SUFFIX ARRAY PREFETCH DISTANCES (Enhanced for SA-IS optimization)
// ============================================================================

// SA array prefetch - 16 entries ahead (was 8)
// Increased to hide memory latency during induced sorting scan
constexpr int SA_PREFETCH_DISTANCE = 16;

// Text position prefetch - 24 entries ahead (was 12)
// Requires longer lookahead due to random access pattern in T[SA[i]-1]
constexpr int SA_TEXT_PREFETCH_DISTANCE = 24;

// Bucket cursor write prefetch - 8 entries ahead (NEW)
// Prefetch write location for bucket placement
constexpr int SA_BUCKET_PREFETCH_DISTANCE = 8;

// Legacy constants (for backward compatibility)
constexpr int SA_BUCKET_PREFETCH_LINES = 8;
constexpr int SA_INDUCED_PREFETCH_ENTRIES = SA_PREFETCH_DISTANCE;

// SHA-256: prefetch 4 blocks ahead (4 * 64 = 256 bytes)
constexpr int SHA256_PREFETCH_BLOCKS = 4;

// ============================================================================
// PLATFORM-SPECIFIC PREFETCH MACROS
// ============================================================================

#if defined(__x86_64__) || defined(_M_X64)

// Prefetch for immediate use (L1 + L2 + L3)
#define PREFETCH_T0(addr) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0)

// Prefetch for near-future use (L2 + L3)
#define PREFETCH_T1(addr) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T1)

// Prefetch for later use (L3 only)
#define PREFETCH_T2(addr) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T2)

// Non-temporal prefetch (streaming data, avoid cache pollution)
#define PREFETCH_NTA(addr) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_NTA)

// Write prefetch hint (if supported)
#if defined(__GNUC__) || defined(__clang__)
#define PREFETCH_WRITE(addr) __builtin_prefetch(addr, 1, 3)
#else
#define PREFETCH_WRITE(addr) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0)
#endif

#elif defined(__aarch64__)

// ARM prefetch instructions
#define PREFETCH_T0(addr) __builtin_prefetch(addr, 0, 3)
#define PREFETCH_T1(addr) __builtin_prefetch(addr, 0, 2)
#define PREFETCH_T2(addr) __builtin_prefetch(addr, 0, 1)
#define PREFETCH_NTA(addr) __builtin_prefetch(addr, 0, 0)
#define PREFETCH_WRITE(addr) __builtin_prefetch(addr, 1, 3)

#else

// Fallback: no-op prefetch
#define PREFETCH_T0(addr) ((void)0)
#define PREFETCH_T1(addr) ((void)0)
#define PREFETCH_T2(addr) ((void)0)
#define PREFETCH_NTA(addr) ((void)0)
#define PREFETCH_WRITE(addr) ((void)0)

#endif

// ============================================================================
// PHASE 1: BRANCH COMPUTATION PREFETCHING
// ============================================================================

/**
 * Branch computation memory access pattern:
 *
 * Per iteration:
 *   1. Read prev_chunk[pos1:pos2] - 1-32 bytes (AVX2 load)
 *   2. Read CodeLUT[op] or CodeLUT_16[op] - 4 bytes (random)
 *   3. Write chunk[pos1:pos2] - 1-32 bytes (AVX2 store)
 *   4. Read chunk[pos2] for v2 dependency - 1 byte
 *
 * Strategy:
 *   - Keep both chunks pinned in L1 (512 bytes total)
 *   - Prefetch next iteration's CodeLUT entry
 *   - Use NT stores if chunk won't be read again soon
 */

/**
 * Prefetch branch lookup table entries for upcoming iterations
 *
 * @param code_lut Pointer to CodeLUT or CodeLUT_16 array
 * @param current_op Current opcode being processed
 * @param next_ops Array of next N opcodes to process
 * @param count Number of opcodes to prefetch
 */
template<typename T>
inline void prefetch_branch_lut(
    const T* code_lut,
    const uint8_t* next_ops,
    int count
) {
    // Prefetch next few LUT entries to L1
    for (int i = 0; i < count && i < BRANCH_PREFETCH_DISTANCE; ++i) {
        PREFETCH_T0(&code_lut[next_ops[i]]);
    }
}

/**
 * Prefetch chunk data for branch computation
 *
 * Since chunks are only 256 bytes each, we can prefetch the entire
 * working set at the start of each hash operation.
 *
 * @param prev_chunk Previous iteration's chunk (read)
 * @param chunk Current chunk being written
 */
inline void prefetch_branch_chunks(
    const uint8_t* prev_chunk,
    uint8_t* chunk
) {
    // Prefetch entire 256-byte chunk to L1 (4 cache lines)
    PREFETCH_T0(prev_chunk);
    PREFETCH_T0(prev_chunk + 64);
    PREFETCH_T0(prev_chunk + 128);
    PREFETCH_T0(prev_chunk + 192);

    // Prefetch output chunk for writing
    PREFETCH_WRITE(chunk);
    PREFETCH_WRITE(chunk + 64);
    PREFETCH_WRITE(chunk + 128);
    PREFETCH_WRITE(chunk + 192);
}

/**
 * Inline prefetch for branch loop - call at start of each iteration
 *
 * This is the key insertion point for branch prefetching.
 * Insert this at the top of the wolfPermute/branchComputeCPU loop.
 *
 * @param prev_chunk Source data array
 * @param chunk Destination data array
 * @param pos1 Start position for this iteration
 * @param pos2 End position for this iteration
 * @param next_pos1 Start position for next iteration (computed ahead)
 */
inline void prefetch_branch_iteration(
    const uint8_t* prev_chunk,
    uint8_t* chunk,
    uint8_t pos1,
    uint8_t pos2,
    uint8_t next_pos1
) {
    // Current iteration data should already be in L1 from previous prefetch

    // Prefetch next iteration's source data
    // Only need to prefetch if it's a different cache line
    if ((next_pos1 >> 6) != (pos1 >> 6)) {
        PREFETCH_T0(&prev_chunk[next_pos1 & ~63]);
    }

    // Prefetch write location for next iteration
    if ((next_pos1 >> 6) != (pos1 >> 6)) {
        PREFETCH_WRITE(&chunk[next_pos1 & ~63]);
    }
}

// ============================================================================
// PHASE 2: SUFFIX ARRAY CONSTRUCTION PREFETCHING
// ============================================================================

/**
 * Suffix array memory access patterns:
 *
 * 1. sort_typeBstar:
 *    - Sequential scan of input T[0:n] for character counting
 *    - Random access to bucket_A[256] and bucket_B[65536]
 *    - Random writes to SA during B* suffix collection
 *
 * 2. sssort (string sorting):
 *    - Median-of-3/5 pivot selection (random access within partition)
 *    - Partitioning causes semi-random access patterns
 *    - Comparison causes random suffix access
 *
 * 3. construct_SA (induced sorting):
 *    - Left-to-right scan of SA
 *    - Each entry causes lookup in T[SA[i]-1] (random in T)
 *    - Write to bucket position (random in SA)
 *
 * Strategy:
 *   - Prefetch bucket_B entries based on upcoming characters
 *   - Prefetch SA entries several iterations ahead during induced sorting
 *   - Use T2 hint for bucket_B (large, sparse access)
 *   - Use T0 hint for SA during sequential scan
 */

/**
 * Prefetch bucket_B entry for a character pair
 *
 * bucket_B[c0][c1] = bucket_B[c1 * 256 + c0] for little-endian layout
 *
 * @param bucket_b Pointer to bucket_B array (65536 * 4 = 256KB)
 * @param c0 First character
 * @param c1 Second character
 */
inline void prefetch_bucket_b(
    const int32_t* bucket_b,
    uint8_t c0,
    uint8_t c1
) {
    // Bucket B is 256KB - use L2/L3 prefetch
    PREFETCH_T1(&bucket_b[(c1 << 8) | c0]);
}

/**
 * Prefetch for induced sorting scan (construct_SA)
 *
 * This is inserted into the main left-to-right scan loop.
 *
 * @param sa Suffix array
 * @param t Input text
 * @param i Current position in SA scan
 * @param n Total length
 */
inline void prefetch_induced_sort(
    const int32_t* sa,
    const uint8_t* t,
    size_t i,
    size_t n
) {
    // Prefetch SA entries ahead
    if (i + SA_INDUCED_PREFETCH_ENTRIES < n) {
        PREFETCH_T0(&sa[i + SA_INDUCED_PREFETCH_ENTRIES]);
    }

    // If we know the SA value ahead of time, prefetch the corresponding T position
    // This requires looking at SA[i + distance] to get the text position
    if (i + 8 < n) {
        int32_t future_sa = sa[i + 8];
        if (future_sa > 0) {
            PREFETCH_T1(&t[future_sa - 1]);  // Will access T[SA[i]-1]
        }
    }
}

/**
 * Prefetch for string sorting (sssort) comparison phase
 *
 * During suffix comparison, we access T[PA[suffix] + depth]
 *
 * @param t Input text
 * @param pa Suffix array of B* positions
 * @param sa Current SA partition being sorted
 * @param idx Current index in partition
 * @param depth Current comparison depth
 */
inline void prefetch_sssort_compare(
    const uint8_t* t,
    const int32_t* pa,
    const int32_t* sa,
    size_t idx,
    size_t depth
) {
    // Prefetch text positions for upcoming comparisons
    for (int d = 0; d < 4; ++d) {
        if (idx + d < 256) {  // Assume partition size
            int32_t suffix_idx = pa[sa[idx + d]];
            PREFETCH_T0(&t[suffix_idx + depth]);
        }
    }
}

/**
 * Batch prefetch for bucket initialization
 *
 * Call before starting suffix array construction to warm up bucket arrays.
 *
 * @param bucket_a bucket_A array (256 entries = 1KB)
 * @param bucket_b bucket_B array (65536 entries = 256KB)
 */
inline void prefetch_buckets_init(
    int32_t* bucket_a,
    int32_t* bucket_b
) {
    // bucket_A fits in L1 - prefetch all
    for (size_t i = 0; i < 256; i += CACHE_LINE_SIZE / sizeof(int32_t)) {
        PREFETCH_T0(&bucket_a[i]);
    }

    // bucket_B is large - prefetch first portion to L2
    // Let hardware prefetcher handle the rest during sequential init
    for (size_t i = 0; i < 1024; i += CACHE_LINE_SIZE / sizeof(int32_t)) {
        PREFETCH_T1(&bucket_b[i]);
    }
}

// ============================================================================
// PHASE 3: SHA-256 HASHING PREFETCHING
// ============================================================================

/**
 * SHA-256 memory access pattern:
 *
 * - Strictly sequential: read 64 bytes, process, repeat
 * - Perfect for hardware prefetcher
 * - Software prefetch mainly helps at start and block boundaries
 *
 * Strategy:
 *   - Use NTA (non-temporal) prefetch since data is read once
 *   - Prefetch 2-4 blocks ahead (128-256 bytes)
 *   - Avoid polluting cache with SA data that won't be reused
 */

/**
 * Prefetch SHA-256 input blocks
 *
 * @param sa_data Suffix array data being hashed
 * @param block_idx Current block index (0-based)
 * @param total_blocks Total number of 64-byte blocks
 */
inline void prefetch_sha256_block(
    const int32_t* sa_data,
    size_t block_idx,
    size_t total_blocks
) {
    // Prefetch next few blocks with NTA hint (streaming, don't cache)
    for (int i = 1; i <= SHA256_PREFETCH_BLOCKS; ++i) {
        size_t future_block = block_idx + i;
        if (future_block < total_blocks) {
            // Each block is 64 bytes = 16 int32_t values
            PREFETCH_NTA(&sa_data[future_block * 16]);
        }
    }
}

/**
 * Prefetch for SHA-256 with SPSA decompression
 *
 * When using compressed SA (SPSA), we need to prefetch both
 * the compressed entries and any stamp data.
 *
 * @param compressed_sa Compressed SA entries
 * @param stamps Stamp metadata array
 * @param entry_idx Current entry index
 * @param total_entries Total entries
 */
template<typename SaEntry, typename Stamp>
inline void prefetch_sha256_spsa(
    const SaEntry* compressed_sa,
    const Stamp* stamps,
    size_t entry_idx,
    size_t total_entries
) {
    // Prefetch compressed SA entries
    size_t future_entry = entry_idx + SHA256_PREFETCH_BLOCKS * 16;
    if (future_entry < total_entries) {
        PREFETCH_NTA(&compressed_sa[future_entry]);
    }

    // Stamps are small, keep in cache
    // (typically only ~256-300 stamps for entire hash)
}

// ============================================================================
// DATA LAYOUT OPTIMIZATIONS
// ============================================================================

/**
 * Cache-line aligned data structure recommendations:
 *
 * 1. workerData alignment:
 *    - chunk[256] and prev_chunk[256] should be 64-byte aligned
 *    - Keep them adjacent for spatial locality
 *
 * 2. Suffix array alignment:
 *    - SA should be 64-byte aligned for SIMD operations
 *    - Consider padding to avoid false sharing in multi-threaded code
 *
 * 3. Bucket arrays:
 *    - bucket_A: 64-byte alignment sufficient
 *    - bucket_B: Consider interleaving or tiling for better cache use
 */

// Alignment helpers
#define CACHE_ALIGNED alignas(CACHE_LINE_SIZE)

// Padded structure to avoid false sharing
template<typename T>
struct CacheLinePadded {
    CACHE_ALIGNED T data;
    char padding[CACHE_LINE_SIZE - (sizeof(T) % CACHE_LINE_SIZE)];
};

// ============================================================================
// CACHE POLLUTION AVOIDANCE
// ============================================================================

/**
 * Strategies to avoid cache pollution:
 *
 * 1. Use NTA prefetch for streaming data (SHA-256 input)
 * 2. Use streaming stores for data that won't be read again soon
 * 3. Partition working set to fit in L2 when possible
 *
 * For AstroBWTv3:
 * - Branch computation: Keep in L1 (working set ~2KB)
 * - SA construction: bucket_B will evict other data, accept this
 * - SHA-256: Use NTA to avoid evicting branch/SA data if reused
 */

#if defined(__x86_64__) || defined(_M_X64)

/**
 * Non-temporal 256-bit store (AVX)
 * Use for writing data that won't be read again soon
 */
inline void stream_store_256(void* dest, __m256i data) {
    _mm256_stream_si256(reinterpret_cast<__m256i*>(dest), data);
}

/**
 * Memory fence after streaming stores
 */
inline void stream_fence() {
    _mm_sfence();
}

#endif

// ============================================================================
// INTEGRATED PREFETCH STRATEGY
// ============================================================================

/**
 * Call at the start of AstroBWTv3 to set up prefetching for the entire hash.
 *
 * This warms up the most critical data structures.
 */
inline void prefetch_astrobwt_init(
    uint8_t* chunk,
    uint8_t* prev_chunk,
    int32_t* bucket_a,
    int32_t* bucket_b,
    const uint32_t* code_lut
) {
    // Warm up branch computation data
    prefetch_branch_chunks(prev_chunk, chunk);

    // Prefetch CodeLUT (entire 1KB fits in L1)
    for (size_t i = 0; i < 256; i += CACHE_LINE_SIZE / sizeof(uint32_t)) {
        PREFETCH_T0(&code_lut[i]);
    }

    // Initialize bucket arrays in cache
    prefetch_buckets_init(bucket_a, bucket_b);
}

} // namespace prefetch

#endif // PREFETCH_STRATEGY_HPP
