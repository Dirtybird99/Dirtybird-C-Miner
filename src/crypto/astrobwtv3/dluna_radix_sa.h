#pragma once
/**
 * dluna_radix_sa.h - DeroLuna-style SA backend
 * 
 * Exact algorithmic match for DeroLuna v1.14 radix_sort (RVA 0x45cb0):
 * - 2-pass MSD radix sort
 * - 12-bit radix (4096-entry buckets)
 * - bswap for lexicographical order in 32-bit keys
 * - Fused collision detection + introsort fallback
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

bool isPhaseTelemetryEnabled();
void addSABreakdownTelemetry(uint64_t encode_ns, uint64_t radix_ns, uint64_t collision_ns, uint64_t copy_ns);

namespace dluna_radix_sa {

static constexpr int32_t RADIX_MAX_N = (277 * 256) + 1;
static constexpr uint32_t RADIX_BITS = 12;
static constexpr uint32_t RADIX_SIZE = 1u << RADIX_BITS; 
static constexpr uint32_t RADIX_MASK = RADIX_SIZE - 1u;

struct SortRecord {
    uint32_t key;
    uint32_t pos;
};

static inline bool suffix_less(const uint8_t* T, int32_t n, int32_t a, int32_t b) {
    if (a == b) return false;
    int32_t pa = a;
    int32_t pb = b;
    
#if defined(__AVX2__)
    while ((pa + 32) <= n && (pb + 32) <= n) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(T + pa));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(T + pb));
        __m256i eq = _mm256_cmpeq_epi8(va, vb);
        unsigned int mask = (unsigned int)_mm256_movemask_epi8(eq);
        if (mask != 0xFFFFFFFFu) {
#if defined(_MSC_VER)
            unsigned long pos; _BitScanForward(&pos, ~mask);
#else
            unsigned int pos = __builtin_ctz(~mask);
#endif
            return T[pa + pos] < T[pb + pos];
        }
        pa += 32; pb += 32;
    }
#endif

    while (pa < n && pb < n) {
        if (T[pa] != T[pb]) return T[pa] < T[pb];
        ++pa; ++pb;
    }
    return (n - a) < (n - b);
}

static inline int radix_sort_sa(const uint8_t* T, int32_t* SA, int32_t n,
                                int* /*bucket_A*/ = nullptr, int* /*bucket_B*/ = nullptr) {
    if (T == nullptr || SA == nullptr || n < 0) return -1;
    if (n == 0) return 0;
    if (n == 1) { SA[0] = 0; return 0; }
    if (n > RADIX_MAX_N) return -1;

    const bool phase_telemetry = isPhaseTelemetryEnabled();
    const auto now_ns = []() -> uint64_t {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    };
    uint64_t encode_ns = 0, radix_ns = 0, collision_ns = 0, copy_ns = 0;

    alignas(64) thread_local SortRecord records[RADIX_MAX_N];
    alignas(64) thread_local SortRecord temp_records[RADIX_MAX_N];
    alignas(64) thread_local uint32_t bkt1[RADIX_SIZE];
    alignas(64) thread_local uint32_t bkt2[RADIX_SIZE];

    std::memset(bkt1, 0, sizeof(bkt1));
    std::memset(bkt2, 0, sizeof(bkt2));

    uint64_t t_start = now_ns();

    // Phase 1: Encode (24-bit key) + Histogram
    // DeroLuna loads 4 bytes and masks to 24-bit
    for (int32_t i = 0; i < n; i++) {
        uint32_t key32 = 0;
        // Safe load 4 bytes (AstroBWT padding allows this)
        std::memcpy(&key32, T + i, 4);
#if defined(_MSC_VER)
        key32 = _byteswap_ulong(key32);
#else
        key32 = __builtin_bswap32(key32);
#endif
        uint32_t key24 = key32 >> 8;
        records[i] = {key24, static_cast<uint32_t>(i)};
        
        // DeroLuna's exact bucket extraction:
        // top12 = key >> 12
        // mid12 = key & 0xFFF
        bkt2[key24 >> 12]++;
        bkt1[key24 & 0xFFF]++;
    }

    if (phase_telemetry) encode_ns = now_ns() - t_start;
    uint64_t t_radix = now_ns();

    // Phase 2: Prefix Sums (Exclusive)
    uint32_t off1 = 0, off2 = 0;
    for (uint32_t r = 0; r < RADIX_SIZE; r++) {
        uint32_t c1 = bkt1[r];
        uint32_t c2 = bkt2[r];
        bkt1[r] = off1;
        bkt2[r] = off2;
        off1 += c1;
        off2 += c2;
    }

    // Phase 3: 2 Radix Passes
    // Pass 1: LSB 12 bits (mid12)
    for (int32_t r = 0; r < n; r++) {
        temp_records[bkt1[records[r].key & 0xFFF]++] = records[r];
    }
    // Pass 2: MSB 12 bits (top12)
    for (int32_t r = 0; r < n; r++) {
        records[bkt2[temp_records[r].key >> 12]++] = temp_records[r];
    }

    if (phase_telemetry) radix_ns = now_ns() - t_radix;
    uint64_t t_coll = now_ns();

    // Phase 4: Collision Resolution
    int32_t out = 0;
    for (int32_t i = 0; i < n; ) {
        uint32_t key = records[i].key;
        int32_t j = i + 1;
        while (j < n && records[j].key == key) j++;

        int32_t bsize = j - i;
        if (bsize == 1) {
            SA[out++] = static_cast<int32_t>(records[i].pos);
        } else {
            // DeroLuna uses introsort for collisions
            std::sort(records + i, records + j, [T, n](const SortRecord& a, const SortRecord& b) {
                return suffix_less(T, n, a.pos, b.pos);
            });
            for (int32_t k = i; k < j; k++) SA[out++] = static_cast<int32_t>(records[k].pos);
        }
        i = j;
    }

    if (phase_telemetry) {
        collision_ns = now_ns() - t_coll;
        addSABreakdownTelemetry(encode_ns, radix_ns, collision_ns, copy_ns);
    }

    return 0;
}

}  // namespace dluna_radix_sa
