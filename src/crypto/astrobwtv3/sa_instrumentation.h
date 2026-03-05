/*
 * sa_instrumentation.h - Compile-time toggleable counters for SA analysis
 *
 * Purpose: Collect empirical access pattern data from divsufsort to inform
 *          the design of a custom SA-IS algorithm for the 70KB AstroBWT workload.
 *
 * Usage:
 *   cmake -B build -DENABLE_SA_INSTRUMENTATION=ON
 *   cmake --build build --config Release
 *
 * When disabled (default), all macros compile to nothing for zero overhead.
 */

#ifndef SA_INSTRUMENTATION_H
#define SA_INSTRUMENTATION_H

#ifdef ENABLE_SA_INSTRUMENTATION

#ifdef __cplusplus
#include <cstdint>
#include <cstdio>
#include <cstring>
#else
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#endif

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

/*
 * SACounters - Thread-local counters for SA access pattern analysis
 *
 * Collected metrics:
 *   - sa_reads:        Number of SA[i] read operations
 *   - sa_writes:       Number of SA[i] write operations
 *   - text_reads:      Number of T[s-1] text accesses (indirection)
 *   - bucket_a_access: Accesses to bucket_A (256 entries)
 *   - bucket_b_access: Accesses to bucket_B (256*256 entries)
 *   - loop_iterations: Total loop iterations (for unrolling calibration)
 *   - phase_cycles[4]: Cycle counts per phase
 *       [0] = sort_typeBstar
 *       [1] = construct_SA (or construct_BWT B-type pass)
 *       [2] = construct_SA/BWT L-type pass
 *       [3] = total divsufsort call
 *   - input_size:      Input size n for this iteration
 */
typedef struct SACounters {
    uint64_t sa_reads;
    uint64_t sa_writes;
    uint64_t text_reads;
    uint64_t bucket_a_access;
    uint64_t bucket_b_access;
    uint64_t loop_iterations;
    uint64_t phase_cycles[4];
    int32_t  input_size;
} SACounters;

/* Thread-local counter storage - declared extern, defined in one TU */
#ifdef __cplusplus
extern "C" {
#endif

#ifdef _MSC_VER
/* MSVC: use __declspec(thread) for thread-local */
extern __declspec(thread) SACounters g_sa_counters;
#else
/* GCC/Clang/MinGW: use __thread for thread-local */
extern __thread SACounters g_sa_counters;
#endif

#ifdef __cplusplus
}
#endif

/* Counter increment macros */
#define SA_COUNT_READ()      (++g_sa_counters.sa_reads)
#define SA_COUNT_WRITE()     (++g_sa_counters.sa_writes)
#define SA_COUNT_TEXT()      (++g_sa_counters.text_reads)
#define SA_COUNT_BUCKET_A()  (++g_sa_counters.bucket_a_access)
#define SA_COUNT_BUCKET_B()  (++g_sa_counters.bucket_b_access)
#define SA_COUNT_LOOP()      (++g_sa_counters.loop_iterations)

/* Reset all counters to zero */
#define SA_RESET_COUNTERS()  (memset(&g_sa_counters, 0, sizeof(g_sa_counters)))

/* Set input size for current iteration */
#define SA_SET_SIZE(n)       (g_sa_counters.input_size = (n))

/*
 * High-precision timing using RDTSC
 *
 * rdtsc_start(): Use LFENCE;RDTSC for consistent start timing
 * rdtsc_end():   Use RDTSCP;LFENCE for consistent end timing
 *
 * Note: For cycle-accurate measurements, ensure constant TSC is enabled
 *       (modern CPUs) or disable frequency scaling during benchmarks.
 */
#ifdef _MSC_VER

/* MSVC intrinsics */
static __forceinline uint64_t rdtsc_start(void) {
    _mm_lfence();
    return __rdtsc();
}

static __forceinline uint64_t rdtsc_end(void) {
    unsigned int aux;
    uint64_t tsc = __rdtscp(&aux);
    _mm_lfence();
    return tsc;
}

#else

/* GCC/Clang inline assembly */
static inline uint64_t rdtsc_start(void) {
    unsigned int lo, hi;
    __asm__ volatile ("lfence; rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

static inline uint64_t rdtsc_end(void) {
    unsigned int lo, hi;
    __asm__ volatile ("rdtscp" : "=a"(lo), "=d"(hi) :: "rcx");
    __asm__ volatile ("lfence");
    return ((uint64_t)hi << 32) | lo;
}

#endif

/* Phase timing macros - store start time in local variable */
#define SA_PHASE_START(idx)  uint64_t _phase_start_##idx = rdtsc_start()
#define SA_PHASE_END(idx)    g_sa_counters.phase_cycles[idx] += (rdtsc_end() - _phase_start_##idx)

/*
 * SA_PRINT_METRICS - Output metrics in CSV-compatible format
 *
 * Format: SA_METRICS:iter,size,reads,writes,text,bucketA,bucketB,loops,p0,p1,p2,total
 *
 * This can be parsed by external scripts to generate analysis reports.
 */
#define SA_PRINT_METRICS(iteration) \
    printf("SA_METRICS:%d,%d,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n", \
           (iteration), \
           g_sa_counters.input_size, \
           (unsigned long long)g_sa_counters.sa_reads, \
           (unsigned long long)g_sa_counters.sa_writes, \
           (unsigned long long)g_sa_counters.text_reads, \
           (unsigned long long)g_sa_counters.bucket_a_access, \
           (unsigned long long)g_sa_counters.bucket_b_access, \
           (unsigned long long)g_sa_counters.loop_iterations, \
           (unsigned long long)g_sa_counters.phase_cycles[0], \
           (unsigned long long)g_sa_counters.phase_cycles[1], \
           (unsigned long long)g_sa_counters.phase_cycles[2], \
           (unsigned long long)g_sa_counters.phase_cycles[3])

/*
 * SA_GET_COUNTERS - Get pointer to current thread's counters
 * Useful for programmatic access to metrics.
 */
#define SA_GET_COUNTERS()    (&g_sa_counters)

#else /* !ENABLE_SA_INSTRUMENTATION */

/* When disabled, all macros compile to nothing */
#define SA_COUNT_READ()
#define SA_COUNT_WRITE()
#define SA_COUNT_TEXT()
#define SA_COUNT_BUCKET_A()
#define SA_COUNT_BUCKET_B()
#define SA_COUNT_LOOP()
#define SA_RESET_COUNTERS()
#define SA_SET_SIZE(n)
#define SA_PHASE_START(idx)
#define SA_PHASE_END(idx)
#define SA_PRINT_METRICS(iteration)
#define SA_GET_COUNTERS()    ((void*)0)

#endif /* ENABLE_SA_INSTRUMENTATION */

#endif /* SA_INSTRUMENTATION_H */
