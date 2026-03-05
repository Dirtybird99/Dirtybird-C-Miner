#pragma once
// dluna_radix_sa.h - DeroLuna-style SA backend:
// 1) 24-bit prefix key extraction (4x unrolled + bswap)
// 2) Stable 2-pass radix sort (12 bits + 12 bits)
// 3) Fused collision resolution + SA output (no separate copy phase)
//
// This is tuned for AstroBWT random-like data where 24-bit collisions are sparse.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>

bool isPhaseTelemetryEnabled();
void addSABreakdownTelemetry(uint64_t encode_ns, uint64_t radix_ns, uint64_t collision_ns, uint64_t copy_ns);

namespace dluna_radix_sa {

static constexpr uint32_t RADIX_BITS = 12;
static constexpr uint32_t RADIX_SIZE = 1u << RADIX_BITS;  // 4096
static constexpr uint32_t RADIX_MASK = RADIX_SIZE - 1u;
static constexpr int32_t RADIX_MAX_N = (277 * 256) + 1;

// Compare suffixes lexicographically for collision buckets.
// Skips first 3 bytes (already matched by 24-bit radix key).
static inline bool suffix_less(const uint8_t* T, int32_t n, int32_t a, int32_t b) {
    if (a == b) return false;

    int32_t pa = a + 3;
    int32_t pb = b + 3;

    while ((pa + 8) <= n && (pb + 8) <= n) {
        uint64_t va = 0;
        uint64_t vb = 0;
        std::memcpy(&va, T + pa, sizeof(uint64_t));
        std::memcpy(&vb, T + pb, sizeof(uint64_t));
        if (va != vb) {
#if defined(_MSC_VER)
            va = _byteswap_uint64(va);
            vb = _byteswap_uint64(vb);
#else
            va = __builtin_bswap64(va);
            vb = __builtin_bswap64(vb);
#endif
            return va < vb;
        }
        pa += 8;
        pb += 8;
    }

    while (pa < n && pb < n) {
        if (T[pa] != T[pb]) return T[pa] < T[pb];
        ++pa;
        ++pb;
    }

    // Shorter suffix is lexicographically smaller if one is a prefix of the other.
    return (n - a) < (n - b);
}

// API-compatible with divsufsort/libsais wrappers.
static inline int radix_sort_sa(const uint8_t* T, int32_t* SA, int32_t n,
                                int* /*bucket_A*/ = nullptr, int* /*bucket_B*/ = nullptr) {
    if (T == nullptr || SA == nullptr || n < 0) return -1;
    if (n == 0) return 0;
    if (n == 1) {
        SA[0] = 0;
        return 0;
    }
    if (n > RADIX_MAX_N) return -1;

    const bool phase_telemetry = isPhaseTelemetryEnabled();
    const auto now_ns = []() -> uint64_t {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    };
    uint64_t encode_ns = 0;
    uint64_t radix_ns = 0;
    uint64_t collision_ns = 0;
    uint64_t copy_ns = 0;  // always 0: copy is fused into collision phase

    // Pack each suffix as:
    // upper 32 bits = 24-bit key, lower 32 bits = position
    // Fixed thread-local arrays avoid per-hash size checks and heap-backed vector growth.
    // Cache-line aligned for optimal memory access patterns.
    alignas(64) thread_local uint64_t records[RADIX_MAX_N];
    alignas(64) thread_local uint64_t temp_records[RADIX_MAX_N];

    alignas(64) thread_local uint32_t count_lo[RADIX_SIZE];
    alignas(64) thread_local uint32_t count_hi[RADIX_SIZE];
    alignas(64) thread_local uint32_t offs_lo[RADIX_SIZE];
    alignas(64) thread_local uint32_t offs_hi[RADIX_SIZE];
    std::memset(count_lo, 0, sizeof(count_lo));
    std::memset(count_hi, 0, sizeof(count_hi));

    // ── Phase 1: Encode ─────────────────────────────────────────────────
    // 4x-unrolled key extraction using bswap for big-endian 3-byte prefix.
    uint64_t t_encode_start = 0;
    if (phase_telemetry) {
        t_encode_start = now_ns();
    }

    // Safe limit: need at least 3 bytes beyond position i for a uint32_t read,
    // and 4x unroll needs 4 iterations headroom.
    const int32_t safe_limit = (n >= 3) ? (n - 3) : 0;
    const int32_t unroll_limit = safe_limit & ~3;  // round down to multiple of 4
    int32_t i = 0;

    for (; i < unroll_limit; i += 4) {
#if defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(T + i + 128, 0, 1);
#endif
        uint32_t k0, k1, k2, k3;
#if defined(_MSC_VER)
        k0 = _byteswap_ulong(*(const uint32_t*)(T + i))     >> 8;
        k1 = _byteswap_ulong(*(const uint32_t*)(T + i + 1)) >> 8;
        k2 = _byteswap_ulong(*(const uint32_t*)(T + i + 2)) >> 8;
        k3 = _byteswap_ulong(*(const uint32_t*)(T + i + 3)) >> 8;
#else
        k0 = __builtin_bswap32(*(const uint32_t*)(T + i))     >> 8;
        k1 = __builtin_bswap32(*(const uint32_t*)(T + i + 1)) >> 8;
        k2 = __builtin_bswap32(*(const uint32_t*)(T + i + 2)) >> 8;
        k3 = __builtin_bswap32(*(const uint32_t*)(T + i + 3)) >> 8;
#endif
        records[i]     = (static_cast<uint64_t>(k0) << 32) | static_cast<uint32_t>(i);
        records[i + 1] = (static_cast<uint64_t>(k1) << 32) | static_cast<uint32_t>(i + 1);
        records[i + 2] = (static_cast<uint64_t>(k2) << 32) | static_cast<uint32_t>(i + 2);
        records[i + 3] = (static_cast<uint64_t>(k3) << 32) | static_cast<uint32_t>(i + 3);

        count_lo[k0 & RADIX_MASK]++;
        count_lo[k1 & RADIX_MASK]++;
        count_lo[k2 & RADIX_MASK]++;
        count_lo[k3 & RADIX_MASK]++;

        count_hi[(k0 >> RADIX_BITS) & RADIX_MASK]++;
        count_hi[(k1 >> RADIX_BITS) & RADIX_MASK]++;
        count_hi[(k2 >> RADIX_BITS) & RADIX_MASK]++;
        count_hi[(k3 >> RADIX_BITS) & RADIX_MASK]++;
    }

    // Scalar tail for positions that can still use fast bswap (i < safe_limit)
    for (; i < safe_limit; ++i) {
#if defined(_MSC_VER)
        const uint32_t key24 = _byteswap_ulong(*(const uint32_t*)(T + i)) >> 8;
#else
        const uint32_t key24 = __builtin_bswap32(*(const uint32_t*)(T + i)) >> 8;
#endif
        records[i] = (static_cast<uint64_t>(key24) << 32) | static_cast<uint32_t>(i);
        count_lo[key24 & RADIX_MASK]++;
        count_hi[(key24 >> RADIX_BITS) & RADIX_MASK]++;
    }

    // Boundary positions (last 3): safe byte-by-byte extraction
    for (; i < n; ++i) {
        const uint32_t b0 = static_cast<uint32_t>(T[i]);
        const uint32_t b1 = (i + 1 < n) ? static_cast<uint32_t>(T[i + 1]) : 0u;
        const uint32_t b2 = (i + 2 < n) ? static_cast<uint32_t>(T[i + 2]) : 0u;
        const uint32_t key24 = (b0 << 16) | (b1 << 8) | b2;
        records[i] = (static_cast<uint64_t>(key24) << 32) | static_cast<uint32_t>(i);
        count_lo[key24 & RADIX_MASK]++;
        count_hi[(key24 >> RADIX_BITS) & RADIX_MASK]++;
    }

    if (phase_telemetry) {
        encode_ns = now_ns() - t_encode_start;
    }

    // ── Prefix sums ─────────────────────────────────────────────────────
    offs_lo[0] = 0;
    offs_hi[0] = 0;
    for (uint32_t b = 1; b < RADIX_SIZE; ++b) {
        offs_lo[b] = offs_lo[b - 1] + count_lo[b - 1];
        offs_hi[b] = offs_hi[b - 1] + count_hi[b - 1];
    }

    // ── Phase 2: Radix sort ─────────────────────────────────────────────
    uint64_t t_radix_start = 0;
    if (phase_telemetry) {
        t_radix_start = now_ns();
    }

    // Pass 1: stable sort by low 12 bits.
    for (int32_t r = 0; r < n; ++r) {
        const uint64_t rec = records[r];
        const uint32_t bucket = static_cast<uint32_t>((rec >> 32) & RADIX_MASK);
        temp_records[offs_lo[bucket]++] = rec;
    }

    // Pass 2: stable sort by high 12 bits.
    for (int32_t r = 0; r < n; ++r) {
        const uint64_t rec = temp_records[r];
        const uint32_t key24 = static_cast<uint32_t>(rec >> 32);
        const uint32_t bucket = (key24 >> RADIX_BITS) & RADIX_MASK;
        records[offs_hi[bucket]++] = rec;
    }
    if (phase_telemetry) {
        radix_ns = now_ns() - t_radix_start;
    }

    // ── Phase 3: Fused collision resolution + SA output ─────────────────
    // Scans sorted records, resolves same-key collisions in-place,
    // and writes directly to SA[] — no separate copy phase needed.
    int32_t out = 0;
    i = 0;
    uint64_t t_collision_start = 0;
    if (phase_telemetry) {
        t_collision_start = now_ns();
    }
    while (i < n) {
        const uint32_t key = static_cast<uint32_t>(records[i] >> 32);
        int32_t j = i + 1;
        while (j < n && static_cast<uint32_t>(records[j] >> 32) == key)
            ++j;

        const int32_t bucket_size = j - i;
        if (bucket_size == 1) {
            // Unique key — write directly to SA.
            SA[out++] = static_cast<int32_t>(records[i] & 0xFFFFFFFFu);
        } else if (bucket_size == 2) {
            // 2-element collision — direct compare and swap.
            const int32_t a = static_cast<int32_t>(records[i] & 0xFFFFFFFFu);
            const int32_t b = static_cast<int32_t>(records[i + 1] & 0xFFFFFFFFu);
            if (suffix_less(T, n, a, b)) {
                SA[out]     = a;
                SA[out + 1] = b;
            } else {
                SA[out]     = b;
                SA[out + 1] = a;
            }
            out += 2;
        } else {
            // 3+ collisions — sort and write.
            std::sort(records + i, records + j,
                      [T, n](uint64_t lhs, uint64_t rhs) {
                          const int32_t a = static_cast<int32_t>(lhs & 0xFFFFFFFFu);
                          const int32_t b = static_cast<int32_t>(rhs & 0xFFFFFFFFu);
                          return suffix_less(T, n, a, b);
                      });
            for (int32_t k = i; k < j; ++k) {
                SA[out++] = static_cast<int32_t>(records[k] & 0xFFFFFFFFu);
            }
        }

        i = j;
    }
    if (phase_telemetry) {
        collision_ns = now_ns() - t_collision_start;
        addSABreakdownTelemetry(encode_ns, radix_ns, collision_ns, copy_ns);
    }

    return 0;
}

}  // namespace dluna_radix_sa
