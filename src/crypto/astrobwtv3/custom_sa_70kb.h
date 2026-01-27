/*
 * custom_sa_70kb.h - Custom SA-IS for AstroBWT's 70KB fixed-size inputs
 *
 * Optimized for:
 * - 70KB inputs that fit entirely in L2/L3 cache
 * - Cache-aware design with reduced overhead
 * - Bitvector type storage (9KB vs 70KB byte array)
 * - SIMD-accelerated bucket computation (AVX2)
 * - Triple-stream prefetch for induced sorting
 * - 8-way loop unrolling for better ILP
 * - Branchless type operations
 *
 * Drop-in replacement for divsufsort with identical API.
 *
 * Copyright (c) 2025 DERO Miner Project
 * License: MIT
 */

#ifndef CUSTOM_SA_70KB_H
#define CUSTOM_SA_70KB_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

/* Include divsufsort types for compatibility */
#include "divsufsort.h"

/* ========================================================================== */
/* Build-Time Configuration Flags                                             */
/* ========================================================================== */

/* AVX2 optimizations (SIMD histogram merge, LMS comparison) */
#ifndef ENABLE_SA_AVX2
    #if defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__))
        #define ENABLE_SA_AVX2 1
    #else
        #define ENABLE_SA_AVX2 0
    #endif
#endif

/* Enhanced prefetch (triple-stream) */
#ifndef ENABLE_SA_PREFETCH
    #define ENABLE_SA_PREFETCH 1
#endif

/* 8-way loop unrolling */
#ifndef ENABLE_SA_8WAY_UNROLL
    #define ENABLE_SA_8WAY_UNROLL 1
#endif

/* Branchless operations */
#ifndef ENABLE_SA_BRANCHLESS
    #define ENABLE_SA_BRANCHLESS 1
#endif

/* Instrumentation for profiling */
#ifndef ENABLE_SA_INSTRUMENTATION
    #define ENABLE_SA_INSTRUMENTATION 0
#endif

/* 2-character radix sort for LMS suffixes (better initial ordering) */
#ifndef ENABLE_SA_RADIX_LMS
    #define ENABLE_SA_RADIX_LMS 1
#endif

/* ========================================================================== */
/* Cache and Memory Constants                                                  */
/* ========================================================================== */

/* Maximum input size for custom SA (72KB, slightly larger than AstroBWT max) */
#define CUSTOM_SA_MAX_SIZE (72 * 1024)

/* Minimum input size */
#define CUSTOM_SA_MIN_SIZE 2

/* Cache line alignment macro (64 bytes on most modern CPUs) */
#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64
#endif

/* Cache alignment macro - portable across C and C++ */
#if defined(__cplusplus) && __cplusplus >= 201103L
    #define CACHE_ALIGNED alignas(CACHE_LINE_SIZE)
#elif defined(__GNUC__) || defined(__clang__)
    #define CACHE_ALIGNED __attribute__((aligned(CACHE_LINE_SIZE)))
#elif defined(_MSC_VER)
    #define CACHE_ALIGNED __declspec(align(64))
#else
    #define CACHE_ALIGNED
#endif

/* ========================================================================== */
/* Enhanced Prefetch Distances (Phase 2)                                      */
/* ========================================================================== */

/* SA array prefetch - 16 entries ahead (was 8) */
#ifndef CUSTOM_SA_PREFETCH_DISTANCE
#define CUSTOM_SA_PREFETCH_DISTANCE 16
#endif

/* Text position prefetch - 24 entries ahead (was 12) */
#ifndef CUSTOM_TEXT_PREFETCH_DISTANCE
#define CUSTOM_TEXT_PREFETCH_DISTANCE 24
#endif

/* Bucket cursor write prefetch - NEW */
#ifndef CUSTOM_BUCKET_PREFETCH_DISTANCE
#define CUSTOM_BUCKET_PREFETCH_DISTANCE 8
#endif

/* ========================================================================== */
/* Runtime-Configurable SA Parameters (for Autotune)                          */
/* ========================================================================== */

/**
 * SAConfig - Runtime-configurable prefetch parameters
 *
 * These values can be tuned at runtime via saTune() to find optimal
 * settings for each CPU architecture.
 */
typedef struct {
    int sa_prefetch;      /* SA array prefetch distance (default: 16) */
    int text_prefetch;    /* Text position prefetch distance (default: 24) */
    int bucket_prefetch;  /* Bucket cursor prefetch distance (default: 8) */
} SAConfig;

/* Global SA configuration (set by autotune or CLI) */
extern SAConfig g_sa_config;

/**
 * Initialize SA configuration with compile-time defaults.
 * Called automatically at startup, but can be called to reset.
 */
void sa_config_init(void);

/**
 * Set SA configuration with custom prefetch distances.
 * @param sa_pf     SA array prefetch distance
 * @param text_pf   Text position prefetch distance
 * @param bucket_pf Bucket cursor prefetch distance
 */
void sa_config_set(int sa_pf, int text_pf, int bucket_pf);

/**
 * Get current SA configuration.
 * @return Pointer to current SAConfig (read-only)
 */
const SAConfig* sa_config_get(void);

/* ========================================================================== */
/* Type Definitions                                                           */
/* ========================================================================== */

/* Bitvector for suffix types: 0 = L-type, 1 = S-type */
/* For 72KB input, we need ~9KB = 72000/8 = 9000 bytes */
#define TYPE_BITVEC_SIZE ((CUSTOM_SA_MAX_SIZE + 63) / 64)

/* Bitvector structure for suffix types */
typedef struct {
    uint64_t bits[TYPE_BITVEC_SIZE];  /* ~9KB for 72KB input */
} TypeBitVector;

/* ========================================================================== */
/* Cache-Aligned Workspace (Phase 1: Memory Layout Optimization)              */
/* ========================================================================== */

/**
 * SAWorkspace - Cache-aligned workspace for SA-IS algorithm
 *
 * This structure groups frequently accessed data together to:
 * 1. Minimize cache misses by keeping related data nearby
 * 2. Prevent false sharing in multi-threaded scenarios
 * 3. Ensure proper alignment for SIMD operations
 *
 * Total size: ~13KB - fits comfortably in L2 cache
 */
typedef struct CACHE_ALIGNED {
    saidx_t bucket_counts[256];      /* 1KB - character frequency counts */
    saidx_t bucket_head[256];        /* 1KB - bucket start positions */
    saidx_t bucket_tail[256];        /* 1KB - bucket end positions */
    TypeBitVector types;             /* ~9KB - suffix type bitvector */
    char padding[CACHE_LINE_SIZE];   /* Prevent false sharing */
} SAWorkspace;

/* ========================================================================== */
/* API Functions                                                               */
/* ========================================================================== */

/**
 * Constructs the suffix array using SA-IS algorithm optimized for 70KB inputs.
 *
 * This is a drop-in replacement for divsufsort() with identical semantics.
 * Optimizations:
 * - Bitvector suffix type storage (8x less memory, fits in L1)
 * - 4-way histogram counting to reduce cache conflicts
 * - Stack-based multikey quicksort (no recursion overhead)
 * - Triple-stream prefetch in induced sorting
 * - 8-way loop unrolling for better ILP
 * - AVX2 SIMD for histogram merge and LMS comparison
 * - Branchless type operations
 *
 * @param T[0..n-1]    The input string (text)
 * @param SA[0..n-1]   The output suffix array
 * @param n            The length of the input string (2..72KB)
 * @param bucket_A     256-entry bucket array (caller-provided)
 * @param bucket_B     65536-entry bucket array (caller-provided)
 * @return 0 on success, -1 on invalid parameters, -2 on internal error
 */
saint_t custom_sa_70kb(
    const sauchar_t *T,
    saidx_t *SA,
    saidx_t n,
    saidx_t *bucket_A,
    saidx_t *bucket_B
);

/**
 * Validates a suffix array against the input text.
 *
 * Checks that:
 * - All positions 0..n-1 appear exactly once
 * - Suffixes are lexicographically sorted
 *
 * @param T[0..n-1]    The input string
 * @param SA[0..n-1]   The suffix array to validate
 * @param n            The length of the input
 * @return true if valid, false otherwise
 */
bool custom_sa_validate(
    const sauchar_t *T,
    const saidx_t *SA,
    saidx_t n
);

/**
 * Compares suffix arrays from custom_sa_70kb and divsufsort.
 *
 * Returns true if both produce identical output.
 * Useful for testing correctness.
 *
 * @param T[0..n-1]    The input string
 * @param n            The length of the input
 * @param bucket_A     256-entry bucket array
 * @param bucket_B     65536-entry bucket array
 * @return true if outputs match, false otherwise
 */
bool custom_sa_matches_divsufsort(
    const sauchar_t *T,
    saidx_t n,
    saidx_t *bucket_A,
    saidx_t *bucket_B
);

/* ========================================================================== */
/* Instrumentation API (only when ENABLE_SA_INSTRUMENTATION=1)                */
/* ========================================================================== */

#if ENABLE_SA_INSTRUMENTATION

/**
 * Performance counters for SA-IS phases
 */
typedef struct {
    uint64_t classify_cycles;        /* Suffix classification */
    uint64_t histogram_cycles;       /* Bucket histogram computation */
    uint64_t induce_l_cycles;        /* L-suffix induction */
    uint64_t induce_s_cycles;        /* S-suffix induction */
    uint64_t lms_sort_cycles;        /* LMS suffix sorting */
    uint64_t total_cycles;           /* Total execution */
    uint32_t lms_count;              /* Number of LMS suffixes */
    uint32_t call_count;             /* Number of SA calls */
} SAInstrumentationStats;

/**
 * Get instrumentation statistics
 */
const SAInstrumentationStats* custom_sa_get_stats(void);

/**
 * Reset instrumentation statistics
 */
void custom_sa_reset_stats(void);

#endif /* ENABLE_SA_INSTRUMENTATION */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CUSTOM_SA_70KB_H */
