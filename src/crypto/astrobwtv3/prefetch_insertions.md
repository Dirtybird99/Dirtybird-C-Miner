# DERO Miner Prefetch Insertion Points

This document provides exact code locations for inserting prefetch instructions
in the AstroBWTv3 implementation.

## Table of Contents
1. [Branch Computation Insertions](#1-branch-computation-insertions)
2. [Suffix Array Construction Insertions](#2-suffix-array-construction-insertions)
3. [SHA-256 Hashing Insertions](#3-sha-256-hashing-insertions)
4. [Performance Tuning Guidelines](#4-performance-tuning-guidelines)

---

## 1. Branch Computation Insertions

### 1.1 wolfbranching.cpp - wolfPermute Function

**File:** `wolfbranching.cpp`
**Location:** Top of `wolfPermute` function

```cpp
// BEFORE (existing code):
void wolfPermute(uint8_t *in, uint8_t *out, uint16_t op, uint8_t pos1, uint8_t pos2, workerData &worker)
{
  uint32_t Opcode = CodeLUT[op];

  for(int i = pos1; i < pos2; ++i)
  {
    out[i] = wolfBranch(in[i], in[pos2], Opcode);
  }
}

// AFTER (with prefetching):
#include "prefetch_strategy.hpp"

void wolfPermute(uint8_t *in, uint8_t *out, uint16_t op, uint8_t pos1, uint8_t pos2, workerData &worker)
{
  // Prefetch CodeLUT entry for current operation
  PREFETCH_T0(&CodeLUT[op]);

  // Prefetch next likely opcode (statistical: op+1 or similar)
  if (op < 255) {
    PREFETCH_T1(&CodeLUT[op + 1]);
  }

  uint32_t Opcode = CodeLUT[op];

  // Prefetch input and output cache lines
  PREFETCH_T0(&in[pos1 & ~63]);  // Align to cache line
  PREFETCH_WRITE(&out[pos1 & ~63]);

  for(int i = pos1; i < pos2; ++i)
  {
    out[i] = wolfBranch(in[i], in[pos2], Opcode);
  }
}
```

### 1.2 branched_AVX2.h - SIMD Branch Operations

**File:** `branched_AVX2.h`
**Location:** Each `op*` function that performs AVX2 operations

```cpp
// BEFORE (example - op0):
void op0(workerData &worker, __m256i &data, __m256i &old, int wIndex) {
    p0(data);
    r5(data);
    m0(data);
    r0(data);
    blendStep(worker, data, old, wIndex);
    storeStep(worker, data, old, wIndex);
    // ...
}

// AFTER (with prefetching for next iteration):
void op0(workerData &worker, __m256i &data, __m256i &old, int wIndex) {
    // Prefetch mask table for blend operation
    PREFETCH_T0(&worker.maskTable_bytes[32*(worker.pos2-worker.pos1)]);

    p0(data);
    r5(data);
    m0(data);
    r0(data);
    blendStep(worker, data, old, wIndex);
    storeStep(worker, data, old, wIndex);
    // ...
}
```

### 1.3 Main Branch Loop in astrobwtv3.cpp

**Location:** Inside the main `tries` loop (branchComputeCPU or wolfCompute)

```cpp
// Insert at the START of the branchy loop iteration:

for (int tries = 1; tries < 277; tries++) {
    // ========== PREFETCH INSERTION POINT ==========
    // Prefetch data for NEXT iteration to hide memory latency
    if (tries + 1 < 277) {
        uint8_t next_op = /* calculate next opcode */;
        PREFETCH_T0(&CodeLUT_16[next_op]);

        // Calculate next pos1/pos2 (simplified - actual calculation depends on hash values)
        // Prefetch the cache line that will contain next iteration's data
        uint8_t estimated_next_pos1 = ((tries + 1) & 0xFF);  // Simplified
        PREFETCH_T0(&worker.prev_chunk[estimated_next_pos1 & ~63]);
    }
    // ===============================================

    // ... existing branchy loop code ...
}
```

---

## 2. Suffix Array Construction Insertions

### 2.1 divsufsort.c - sort_typeBstar Function

**File:** `divsufsort.c`
**Location:** Bucket initialization loop

```c
// BEFORE:
/* Initialize bucket arrays. */
for(i = 0; i < BUCKET_A_SIZE; ++i) { bucket_A[i] = 0; }
for(i = 0; i < BUCKET_B_SIZE; ++i) { bucket_B[i] = 0; }

// AFTER (with prefetching):
/* Initialize bucket arrays with prefetching. */
for(i = 0; i < BUCKET_A_SIZE; ++i) {
    if (i + 16 < BUCKET_A_SIZE) {
        __builtin_prefetch(&bucket_A[i + 16], 1, 3);  // Write prefetch
    }
    bucket_A[i] = 0;
}

// bucket_B is large (256KB) - prefetch 8 cache lines ahead
for(i = 0; i < BUCKET_B_SIZE; ++i) {
    if ((i & 15) == 0 && i + 128 < BUCKET_B_SIZE) {
        __builtin_prefetch(&bucket_B[i + 128], 1, 2);  // L2 write prefetch
    }
    bucket_B[i] = 0;
}
```

### 2.2 divsufsort.c - construct_SA Function (ALREADY PARTIALLY DONE)

**File:** `divsufsort.c`
**Location:** Line 244 (existing prefetch) - enhance it

```c
// EXISTING code at line 243-254:
for(i = SA, j = SA + n; i < j; ++i) {
    __builtin_prefetch(i+64, 0, 3);  // EXISTING - prefetch SA entries
    if(0 < (s = *i)) {
      // ...
    }
}

// ENHANCED version with additional prefetches:
for(i = SA, j = SA + n; i < j; ++i) {
    // Existing SA prefetch
    __builtin_prefetch(i + 64, 0, 3);

    // NEW: Prefetch text position for upcoming SA values
    if (i + 8 < j) {
        saidx_t future_s = *(i + 8);
        if (future_s > 0) {
            __builtin_prefetch(&T[future_s - 1], 0, 1);  // L2 prefetch for T access
        }
    }

    // NEW: Prefetch bucket destination
    if (i + 4 < j) {
        saidx_t s4 = *(i + 4);
        if (s4 > 0) {
            saint_t c_ahead = T[s4 - 1];
            __builtin_prefetch(&bucket_A[c_ahead], 0, 2);
        }
    }

    if(0 < (s = *i)) {
        // ... existing code ...
    }
}
```

### 2.3 sssort.c - Suffix Comparison Prefetch

**File:** `sssort.c`
**Location:** `ss_compare` function

```c
// BEFORE:
static INLINE saint_t
ss_compare(const sauchar_t *T,
           const saidx_t *p1, const saidx_t *p2,
           saidx_t depth) {
  const sauchar_t *U1, *U2, *U1n, *U2n;

  for(U1 = T + depth + *p1,
      U2 = T + depth + *p2,
      // ...

// AFTER (with prefetching):
static INLINE saint_t
ss_compare(const sauchar_t *T,
           const saidx_t *p1, const saidx_t *p2,
           saidx_t depth) {
  const sauchar_t *U1, *U2, *U1n, *U2n;

  // Prefetch comparison strings
  __builtin_prefetch(T + depth + *p1, 0, 0);
  __builtin_prefetch(T + depth + *p2, 0, 0);

  // Prefetch end bounds
  __builtin_prefetch(T + *(p1 + 1), 0, 1);
  __builtin_prefetch(T + *(p2 + 1), 0, 1);

  for(U1 = T + depth + *p1,
      U2 = T + depth + *p2,
      // ...
```

### 2.4 sssort.c - Insertion Sort Prefetch

**File:** `sssort.c`
**Location:** `ss_insertionsort` function

```c
// BEFORE:
static void
ss_insertionsort(const sauchar_t *T, const saidx_t *PA,
                 saidx_t *first, saidx_t *last, saidx_t depth) {
  saidx_t *i, *j;
  saidx_t t;
  saint_t r;

  for(i = last - 2; first <= i; --i) {
    for(t = *i, j = i + 1; 0 < (r = ss_compare(T, PA + t, PA + *j, depth));) {
      // ...

// AFTER (with prefetching):
static void
ss_insertionsort(const sauchar_t *T, const saidx_t *PA,
                 saidx_t *first, saidx_t *last, saidx_t depth) {
  saidx_t *i, *j;
  saidx_t t;
  saint_t r;

  for(i = last - 2; first <= i; --i) {
    // Prefetch PA entries for upcoming comparisons
    if (i > first + 2) {
        __builtin_prefetch(&PA[*(i-2)], 0, 1);
    }

    // Prefetch text at suffix positions
    __builtin_prefetch(&T[PA[*i] + depth], 0, 0);

    for(t = *i, j = i + 1; 0 < (r = ss_compare(T, PA + t, PA + *j, depth));) {
      // ...
```

---

## 3. SHA-256 Hashing Insertions

### 3.1 sha256_spsa.cpp - Software Implementation

**File:** `sha256_spsa.cpp`
**Location:** `sha256_compressed_soft` function, main block loop

```cpp
// BEFORE:
// Process full blocks
for (size_t block_idx = 0; block_idx < full_blocks; block_idx++) {
    size_t start_entry = block_idx * entries_per_block;
    fill_block_from_compressed_sa(block, compressed_sa, start_entry, entries_per_block, sa_len);
    sha256_process_block_soft(state, block);
}

// AFTER (with prefetching):
// Process full blocks with prefetching
for (size_t block_idx = 0; block_idx < full_blocks; block_idx++) {
    // Prefetch next block's SA entries (non-temporal since each entry read once)
    if (block_idx + 1 < full_blocks) {
        size_t next_start = (block_idx + 1) * entries_per_block;
        _mm_prefetch(reinterpret_cast<const char*>(&compressed_sa[next_start]), _MM_HINT_NTA);
        _mm_prefetch(reinterpret_cast<const char*>(&compressed_sa[next_start + 8]), _MM_HINT_NTA);
    }

    // Prefetch 2 blocks ahead for deep pipeline
    if (block_idx + 2 < full_blocks) {
        size_t future_start = (block_idx + 2) * entries_per_block;
        _mm_prefetch(reinterpret_cast<const char*>(&compressed_sa[future_start]), _MM_HINT_NTA);
    }

    size_t start_entry = block_idx * entries_per_block;
    fill_block_from_compressed_sa(block, compressed_sa, start_entry, entries_per_block, sa_len);
    sha256_process_block_soft(state, block);
}
```

### 3.2 sha256_spsa.cpp - SHA-NI Implementation

**File:** `sha256_spsa.cpp`
**Location:** `sha256_compressed_ni` function

```cpp
// BEFORE:
// Process full blocks
for (size_t block_idx = 0; block_idx < full_blocks; block_idx++) {
    size_t start_entry = block_idx * entries_per_block;
    fill_block_from_compressed_sa(block, compressed_sa, start_entry, entries_per_block, sa_len);
    sha256_process_block_ni(&state0, &state1, block, shuf_mask);
}

// AFTER (with prefetching):
// Process full blocks with aggressive prefetching for SHA-NI pipeline
for (size_t block_idx = 0; block_idx < full_blocks; block_idx++) {
    // SHA-NI has ~4 block pipeline depth, prefetch 4 blocks ahead
    for (int ahead = 1; ahead <= 4 && block_idx + ahead < full_blocks; ahead++) {
        size_t future_start = (block_idx + ahead) * entries_per_block;
        _mm_prefetch(reinterpret_cast<const char*>(&compressed_sa[future_start]), _MM_HINT_NTA);
    }

    size_t start_entry = block_idx * entries_per_block;
    fill_block_from_compressed_sa(block, compressed_sa, start_entry, entries_per_block, sa_len);
    sha256_process_block_ni(&state0, &state1, block, shuf_mask);
}
```

### 3.3 fill_block_from_compressed_sa Helper

**File:** `sha256_spsa.cpp`
**Location:** `fill_block_from_compressed_sa` function

```cpp
// BEFORE:
static void fill_block_from_compressed_sa(
    uint8_t* block,
    const std::vector<SaEntry>& compressed_sa,
    size_t start_entry,
    size_t num_entries,
    size_t sa_len
) {
    for (size_t i = 0; i < num_entries; i++) {
        size_t entry_idx = start_entry + i;
        // ...
    }
}

// AFTER (with prefetching):
static void fill_block_from_compressed_sa(
    uint8_t* block,
    const std::vector<SaEntry>& compressed_sa,
    size_t start_entry,
    size_t num_entries,
    size_t sa_len
) {
    // Prefetch block output location for writing
    _mm_prefetch(reinterpret_cast<const char*>(block), _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char*>(block + 32), _MM_HINT_T0);

    for (size_t i = 0; i < num_entries; i++) {
        // Prefetch next SA entry
        if (i + 4 < num_entries && start_entry + i + 4 < sa_len) {
            _mm_prefetch(reinterpret_cast<const char*>(&compressed_sa[start_entry + i + 4]), _MM_HINT_T0);
        }

        size_t entry_idx = start_entry + i;
        // ...
    }
}
```

---

## 4. Performance Tuning Guidelines

### 4.1 Prefetch Distance Calibration

The optimal prefetch distance depends on:
- Memory latency (~200 cycles for DRAM)
- Computation per element (~20-100 cycles)
- Pipeline depth

**Formula:** `distance = memory_latency / cycles_per_element`

| Phase | Cycles/Element | Recommended Distance |
|-------|---------------|---------------------|
| Branch Computation | 50 | 4 iterations |
| SA Construction | 20 | 8-10 elements |
| SHA-256 | 100 | 2-4 blocks |

### 4.2 Cache Hint Selection

| Data Type | Hint | Rationale |
|-----------|------|-----------|
| branch chunk | T0 | Frequent reuse, fits in L1 |
| CodeLUT | T0 | Small, hot data |
| bucket_A | T0 | Small, frequent access |
| bucket_B | T1/T2 | Large (256KB), sporadic access |
| SA during build | T0 | Sequential scan |
| SA for SHA-256 | NTA | Read once, streaming |

### 4.3 Avoiding Over-Prefetching

Signs of over-prefetching:
- Performance degrades with prefetch (memory bandwidth saturated)
- High L1/L2 miss rates despite prefetching
- Prefetch instructions dominate execution time

**Mitigation:**
- Use `#pragma unroll` to reduce prefetch overhead
- Gate prefetches with stride checks (only prefetch on cache line crossings)
- Profile with `perf stat` to measure effectiveness

### 4.4 Testing Methodology

```bash
# Compile with different prefetch configs
g++ -DPREFETCH_DISTANCE=4 ...
g++ -DPREFETCH_DISTANCE=8 ...
g++ -DPREFETCH_DISTANCE=16 ...

# Run benchmarks
./dero-miner --benchmark --duration 30

# Check cache performance
perf stat -e cache-misses,cache-references,L1-dcache-load-misses ./dero-miner --benchmark
```

### 4.5 Compiler Hints for Auto-Vectorization

When using prefetches with SIMD code, help the compiler:

```cpp
// Tell compiler about alignment
void process_chunk(uint8_t* __restrict__ chunk) __attribute__((assume_aligned(64)));

// Unroll hint for prefetch loops
#pragma GCC unroll 8
for (int i = 0; i < 256; i += 32) {
    PREFETCH_T0(&chunk[i]);
}
```

---

## Summary of All Insertion Points

| File | Function | Line (approx) | Prefetch Type |
|------|----------|---------------|---------------|
| wolfbranching.cpp | wolfPermute | top | T0 for LUT, WRITE for output |
| branched_AVX2.h | op0-op127 | top of each | T0 for maskTable |
| divsufsort.c | sort_typeBstar | bucket init | WRITE for buckets |
| divsufsort.c | construct_SA | main loop | T0 for SA, T1 for T |
| sssort.c | ss_compare | top | T0 for strings |
| sssort.c | ss_insertionsort | outer loop | T1 for PA entries |
| sha256_spsa.cpp | sha256_compressed_* | block loop | NTA for SA input |
| sha256_spsa.cpp | fill_block_* | inner loop | T0 for next entry |
