# AstroBWTv3 Memory Access Analysis

## Algorithm Flow

```
Input (48 bytes)
    ↓
SHA256 (hardware accelerated via SHA-NI)
    ↓
Salsa20 (256 bytes) - stream cipher
    ↓
RC4 (256 bytes) - encryption round
    ↓
wolfCompute (278 iterations) ← MAIN CPU WORKLOAD (~60% of hash time)
    ↓
divsufsort (suffix array) ← MEMORY BOUND (~35% of hash time)
    ↓
SHA256 (final hash)
    ↓
Output (32 bytes)
```

## Memory Access Hotspots

### 1. wolfCompute Loop (astrobwtv3.cpp:8681)

**278 iterations**, each doing:

```cpp
// INEFFICIENCY #1: Full 256-byte copy every iteration
memcpy(worker.chunk, worker.prev_chunk, 256);  // Even though only pos1:pos2 changes

// Operations on random range [pos1, pos2] where pos2-pos1 ≤ 32
wolfPermute(prev_chunk, chunk, op, pos1, pos2, worker);

// INEFFICIENCY #2: Hash entire chunk even though small region changed
worker.lhash = XXHash64::hash(worker.chunk, worker.pos2, 0);

// INEFFICIENCY #3: 25% probability of full RC4 pass (256 bytes)
if (worker.A <= 0x40) {
    RC4(&worker.key, 256, worker.chunk, worker.chunk);
}
```

**Memory traffic per iteration:**
- 256 bytes read (prev_chunk)
- 256 bytes write (memcpy to chunk)
- ~32 bytes modified (pos1:pos2 range)
- ~256 bytes read (hash)
- 25% chance: +512 bytes (RC4 read+write)

**Total per hash:** 278 × (512 + 256 + 128) ≈ **250 KB memory traffic**

### 2. divsufsort (divsufsort.c)

Suffix array construction on ~70KB of data:
- O(n log n) comparisons
- Random access pattern during induced sorting
- Prefetch distance tuning helps but can't eliminate random access

**Memory traffic:** ~70KB × multiple passes ≈ **200-400 KB**

### 3. workerData Structure

```cpp
struct workerData {
    byte sData[ASTRO_SCRATCH_SIZE];     // 71680 bytes (70KB)
    int sa[MAX_LENGTH];                  // 280KB (70K × 4 bytes)
    int bA[256];                         // 1KB
    int bB[256*256];                     // 256KB
    // ... other fields
};
```

**Total working set:** ~600KB per thread
**For 20 threads:** ~12 MB (fits in L3, but contention occurs)

## Identified Inefficiencies

### Critical (High Impact)

| Issue | Location | Impact | Fix Difficulty |
|-------|----------|--------|----------------|
| Redundant memcpy | wolfCompute:8733 | ~70KB/hash wasted | Medium |
| Full chunk hash | wolfCompute:8757,8805,8815 | Cache misses | Medium |
| Random pos1/pos2 | wolfCompute | Poor prefetch | Hard |

### Moderate (Medium Impact)

| Issue | Location | Impact | Fix Difficulty |
|-------|----------|--------|----------------|
| Thread contention on L3 | All | ~5-10% | Config only |
| No huge pages | Memory allocation | ~10-15% | Config only |
| E-core scheduling | OS scheduler | ~5-10% variance | Config only |

### Low (Minor Impact)

| Issue | Location | Impact | Fix Difficulty |
|-------|----------|--------|----------------|
| Conditional branches | wolfPermute | ~2-3% | Algorithmic |
| Hash function selection | wolfCompute:8802-8832 | ~1-2% | Minor |

## Optimization Opportunities

### 1. Huge Pages (10-15% gain, no code changes)

```
# Windows: Enable "Lock pages in memory" in Local Security Policy
# Then miner will use huge pages automatically via malloc_huge_pages()
```

### 2. Thread Pinning for Hybrid CPU (5-10% stability)

```bash
# Test P-core only (8 physical cores, 16 threads)
--threads 16

# Vs current mixed P+E core
--threads 20
```

### 3. Algorithmic Changes (requires code modification)

**Incremental memcpy:**
```cpp
// Instead of copying full 256 bytes:
// memcpy(worker.chunk, worker.prev_chunk, 256);

// Only copy the modified region:
memcpy(worker.chunk + pos1, worker.prev_chunk + pos1, pos2 - pos1 + 1);
```

**Incremental hashing:**
```cpp
// Instead of hashing full chunk:
// worker.lhash = XXHash64::hash(worker.chunk, worker.pos2, 0);

// Use rolling hash or delta hash
worker.lhash = update_hash(worker.lhash, worker.chunk, pos1, pos2);
```

### 4. AVX2/AVX-512 Vectorization Opportunities

Currently vectorized:
- ✅ wolfPermute (AVX2)
- ✅ RC4 (AVX-512 when available)
- ✅ SHA256 (SHA-NI hardware)

Not vectorized:
- ❌ memcpy in hot loop (could use streaming stores)
- ❌ Hash functions (some already use SIMD)
- ❌ divsufsort (inherently sequential)

## Profiling Commands

### Windows (Visual Studio)

```bash
# CPU sampling
VTune: Hotspots analysis on dirtybird-miner-cpu.exe

# Memory bandwidth
VTune: Memory Access analysis
```

### Linux (perf)

```bash
# Cache misses
perf stat -e cache-misses,cache-references ./dirtybird-miner-cpu

# Memory bandwidth
perf stat -e LLC-load-misses,LLC-store-misses ./dirtybird-miner-cpu

# Call graph
perf record -g ./dirtybird-miner-cpu
perf report
```

## Theoretical Ceiling Analysis

| Component | Current | Optimized | Max Gain |
|-----------|---------|-----------|----------|
| wolfCompute | 60% | 50% | +20% |
| divsufsort | 35% | 30% | +15% |
| SHA256 | 5% | 5% | - |
| **Total** | 100% | 85% | **+17%** |

**Current:** 17.5 KH/s
**With huge pages:** 19-20 KH/s (+10-15%)
**With algorithmic fixes:** 21-22 KH/s (+20%)
**Theoretical max (AVX2):** ~22-23 KH/s

**To reach 25 KH/s requires:**
- AVX-512 CPU (Zen4/5, Sapphire Rapids)
- OR significant algorithmic work reduction
- OR GPU offload of divsufsort
