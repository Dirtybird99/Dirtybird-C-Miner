/* Two-lane interleaved SHA-NI compress. The schedule interleaves 4-round groups
 * so the two dependent sha256rnds2 chains overlap in the out-of-order window;
 * the instruction count is identical to two sequential streams, the win is
 * purely scheduling. */

#include "sha256_x2.h"

#include <cstring>
#include <immintrin.h>

#if defined(_MSC_VER)
#  include <intrin.h>
#else
#  include <cpuid.h>
#endif

#if !defined(__GNUC__) && !defined(__clang__)
#  error "requires GCC-style extended inline assembly"
#endif

namespace {

/* SHA-256 round constants, 16-byte aligned. The compress body adds them with
 * `paddd off(%reg), %xmm`, a legacy-SSE memory operand that #GPs on an
 * unaligned address; the natural alignment of uint32_t[64] is only 4. */
alignas(16) const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

const uint32_t INITIAL[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
};

void cpuid_count(uint32_t leaf, uint32_t sub, uint32_t out[4]) {
#if defined(_MSC_VER)
    int r[4];
    __cpuidex(r, static_cast<int>(leaf), static_cast<int>(sub));
    out[0] = static_cast<uint32_t>(r[0]);
    out[1] = static_cast<uint32_t>(r[1]);
    out[2] = static_cast<uint32_t>(r[2]);
    out[3] = static_cast<uint32_t>(r[3]);
#else
    __cpuid_count(leaf, sub, out[0], out[1], out[2], out[3]);
#endif
}

uint64_t read_xcr0() {
    uint32_t lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return (static_cast<uint64_t>(hi) << 32) | lo;
}

bool detect_shani() {
    uint32_t r[4];
    cpuid_count(0, 0, r);
    if (r[0] < 7) return false;
    cpuid_count(1, 0, r);
    const bool ssse3 = (r[2] >> 9) & 1u;
    const bool sse41 = (r[2] >> 19) & 1u;
    cpuid_count(7, 0, r);
    const bool sha = (r[1] >> 29) & 1u;
    return sha && ssse3 && sse41;
}

bool detect_avx() {
    uint32_t r[4];
    cpuid_count(0, 0, r);
    if (r[0] < 1) return false;
    cpuid_count(1, 0, r);
    const bool osxsave = (r[2] >> 27) & 1u;
    const bool avx = (r[2] >> 28) & 1u;
    if (!osxsave || !avx) return false;
    return (read_xcr0() & 0x6u) == 0x6u;
}

/* Isolated behind its own target attribute so the vzeroupper is emitted only
 * here and only reached through the runtime AVX check. */
__attribute__((target("avx"), noinline)) void zeroupper() {
    _mm256_zeroupper();
}

__attribute__((target("ssse3,sse4.1")))
void load_state(const uint32_t s[8], __m128i& abef, __m128i& cdgh) {
    const __m128i* p = reinterpret_cast<const __m128i*>(s);
    const __m128i dcba = _mm_loadu_si128(p);
    const __m128i efgh = _mm_shuffle_epi32(_mm_loadu_si128(p + 1), 0x1b);
    const __m128i cdab = _mm_shuffle_epi32(dcba, 0xb1);
    abef = _mm_alignr_epi8(cdab, efgh, 8);
    cdgh = _mm_blend_epi16(efgh, cdab, 0xf0);
}

__attribute__((target("ssse3,sse4.1")))
void store_state(uint32_t s[8], __m128i abef, __m128i cdgh) {
    const __m128i feba = _mm_shuffle_epi32(abef, 0x1b);
    const __m128i dchg = _mm_shuffle_epi32(cdgh, 0xb1);
    __m128i* p = reinterpret_cast<__m128i*>(s);
    _mm_storeu_si128(p, _mm_blend_epi16(feba, dchg, 0xf0));
    _mm_storeu_si128(p + 1, _mm_alignr_epi8(dchg, feba, 8));
}

/* Build the 1 or 2 padding blocks for a message and return how many are used. */
size_t build_tail(const uint8_t* in, size_t len, uint8_t out[2][64]) {
    const size_t rem = len % 64;
    std::memset(out, 0, 128);
    if (rem) std::memcpy(out[0], in + (len - rem), rem);
    out[0][rem] = 0x80;
    const size_t blocks = (rem < 56) ? 1 : 2;
    const uint64_t bits = static_cast<uint64_t>(len) * 8;
    for (int i = 0; i < 8; ++i)
        out[blocks - 1][56 + i] = static_cast<uint8_t>(bits >> (56 - 8 * i));
    return blocks;
}

void finish(const uint32_t s[8], uint8_t out[32]) {
    for (int i = 0; i < 8; ++i) {
        out[i * 4 + 0] = static_cast<uint8_t>(s[i] >> 24);
        out[i * 4 + 1] = static_cast<uint8_t>(s[i] >> 16);
        out[i * 4 + 2] = static_cast<uint8_t>(s[i] >> 8);
        out[i * 4 + 3] = static_cast<uint8_t>(s[i]);
    }
}

/* ---- compress body ------------------------------------------------------ */
/*
 * Everything between the prologue and epilogue must stay legacy-SSE: a VEX
 * encoding anywhere in here would put the core in the AVX state and every
 * subsequent legacy sha256rnds2 would pay a transition penalty. The plain
 * mnemonics below assemble to legacy encodings regardless of the -m flags the
 * translation unit is built with; the assembler never promotes them to VEX.
 *
 * Register map, fixed because the operands are written into the asm text (a
 * constraint cannot pin a named xmm):
 *   xmm1/xmm2   lane A state ABEF / CDGH
 *   xmm3/xmm4   lane B state ABEF / CDGH
 *   xmm5..xmm8  lane A message windows W0..W3
 *   xmm9..xmm12 lane B message windows W0..W3
 *   xmm0        shared round-constant scratch, and the implicit third operand
 *               of sha256rnds2 -- which is why the two lanes cannot have one
 *               each, and why each lane's pair of rounds must be emitted
 *               contiguously
 *   xmm14/xmm15 lane A / lane B schedule temporaries
 *
 * At group g the window register holding the words for this group's rounds is
 * W[g mod 4]; the schedule updates W[(g+1) mod 4] (msg2) and W[(g+3) mod 4]
 * (msg1). W[(g+2) mod 4] is never named.
 */

#define SA0 "%%xmm1"
#define SA1 "%%xmm2"
#define SB0 "%%xmm3"
#define SB1 "%%xmm4"
#define TA  "%%xmm14"
#define TB  "%%xmm15"
#define WA0 "%%xmm5"
#define WA1 "%%xmm6"
#define WA2 "%%xmm7"
#define WA3 "%%xmm8"
#define WB0 "%%xmm9"
#define WB1 "%%xmm10"
#define WB2 "%%xmm11"
#define WB3 "%%xmm12"

#define RND_HEAD(KOFF, M0, S0, S1)          \
    "movdqa " M0 ", %%xmm0\n\t"             \
    "paddd " KOFF "(%[k]), %%xmm0\n\t"      \
    "sha256rnds2 " S0 ", " S1 "\n\t"

#define RND_TAIL(S0, S1)                    \
    "pshufd $0x0E, %%xmm0, %%xmm0\n\t"      \
    "sha256rnds2 " S1 ", " S0 "\n\t"

#define MSG1(M0, M3)                        \
    "sha256msg1 " M0 ", " M3 "\n\t"

#define MSG2(M0, M1, M3, T)                 \
    "movdqa " M0 ", " T "\n\t"              \
    "palignr $4, " M3 ", " T "\n\t"         \
    "paddd " T ", " M1 "\n\t"               \
    "sha256msg2 " M0 ", " M1 "\n\t"

/* groups 0 and 15: rounds only */
#define GRP_PLAIN(KOFF, AM0, BM0)                       \
    RND_HEAD(KOFF, AM0, SA0, SA1) RND_TAIL(SA0, SA1)    \
    RND_HEAD(KOFF, BM0, SB0, SB1) RND_TAIL(SB0, SB1)

/* groups 1-2: schedule has started, nothing to finish yet */
#define GRP_M1(KOFF, AM0, AM3, BM0, BM3)                \
    RND_HEAD(KOFF, AM0, SA0, SA1)                       \
    MSG1(AM0, AM3)                                      \
    RND_TAIL(SA0, SA1)                                  \
    RND_HEAD(KOFF, BM0, SB0, SB1)                       \
    MSG1(BM0, BM3)                                      \
    RND_TAIL(SB0, SB1)

/* groups 3-12: full schedule */
#define GRP_FULL(KOFF, AM0, AM1, AM3, BM0, BM1, BM3)    \
    RND_HEAD(KOFF, AM0, SA0, SA1)                       \
    MSG2(AM0, AM1, AM3, TA)                             \
    MSG1(AM0, AM3)                                      \
    RND_TAIL(SA0, SA1)                                  \
    RND_HEAD(KOFF, BM0, SB0, SB1)                       \
    MSG2(BM0, BM1, BM3, TB)                             \
    MSG1(BM0, BM3)                                      \
    RND_TAIL(SB0, SB1)

/* groups 13-14: last two windows only need finishing */
#define GRP_M2(KOFF, AM0, AM1, AM3, BM0, BM1, BM3)      \
    RND_HEAD(KOFF, AM0, SA0, SA1)                       \
    MSG2(AM0, AM1, AM3, TA)                             \
    RND_TAIL(SA0, SA1)                                  \
    RND_HEAD(KOFF, BM0, SB0, SB1)                       \
    MSG2(BM0, BM1, BM3, TB)                             \
    RND_TAIL(SB0, SB1)

/* One 64-byte block per lane. `st` holds ABEF/CDGH for lane A then lane B and
 * receives the post-64-round working state; the caller adds the pre-block
 * state. `msg` holds four big-endian message windows per lane. Both must be
 * 16-byte aligned. */
__attribute__((always_inline)) inline
void compress2(__m128i st[4], const __m128i msg[8]) {
    __asm__ volatile(
        "movdqa 0(%[st]), %%xmm1\n\t"
        "movdqa 16(%[st]), %%xmm2\n\t"
        "movdqa 32(%[st]), %%xmm3\n\t"
        "movdqa 48(%[st]), %%xmm4\n\t"
        "movdqa 0(%[msg]), %%xmm5\n\t"
        "movdqa 16(%[msg]), %%xmm6\n\t"
        "movdqa 32(%[msg]), %%xmm7\n\t"
        "movdqa 48(%[msg]), %%xmm8\n\t"
        "movdqa 64(%[msg]), %%xmm9\n\t"
        "movdqa 80(%[msg]), %%xmm10\n\t"
        "movdqa 96(%[msg]), %%xmm11\n\t"
        "movdqa 112(%[msg]), %%xmm12\n\t"

        GRP_PLAIN("0",   WA0,           WB0)
        GRP_M1   ("16",  WA1, WA0,      WB1, WB0)
        GRP_M1   ("32",  WA2, WA1,      WB2, WB1)
        GRP_FULL ("48",  WA3, WA0, WA2, WB3, WB0, WB2)
        GRP_FULL ("64",  WA0, WA1, WA3, WB0, WB1, WB3)
        GRP_FULL ("80",  WA1, WA2, WA0, WB1, WB2, WB0)
        GRP_FULL ("96",  WA2, WA3, WA1, WB2, WB3, WB1)
        GRP_FULL ("112", WA3, WA0, WA2, WB3, WB0, WB2)
        GRP_FULL ("128", WA0, WA1, WA3, WB0, WB1, WB3)
        GRP_FULL ("144", WA1, WA2, WA0, WB1, WB2, WB0)
        GRP_FULL ("160", WA2, WA3, WA1, WB2, WB3, WB1)
        GRP_FULL ("176", WA3, WA0, WA2, WB3, WB0, WB2)
        GRP_FULL ("192", WA0, WA1, WA3, WB0, WB1, WB3)
        GRP_M2   ("208", WA1, WA2, WA0, WB1, WB2, WB0)
        GRP_M2   ("224", WA2, WA3, WA1, WB2, WB3, WB1)
        GRP_PLAIN("240", WA3,           WB3)

        "movdqa %%xmm1, 0(%[st])\n\t"
        "movdqa %%xmm2, 16(%[st])\n\t"
        "movdqa %%xmm3, 32(%[st])\n\t"
        "movdqa %%xmm4, 48(%[st])\n\t"
        :
        : [k] "r"(K), [st] "r"(st), [msg] "r"(msg)
        : "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7",
          "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15",
          "memory", "cc");
}

}  // namespace

bool dluna_sha256_x2_available() {
    static const bool ok = detect_shani();
    return ok;
}

__attribute__((target("ssse3,sse4.1")))
void dluna_sha256_x2(const uint8_t* in_a, size_t len_a, uint8_t out_a[32],
                     const uint8_t* in_b, size_t len_b, uint8_t out_b[32]) {
    /* Drop any dirty YMM upper left by an AVX2 caller before the legacy-SSE
     * loop below. */
    static const bool have_avx = detect_avx();
    if (have_avx) zeroupper();

    alignas(16) uint8_t tail_a[2][64];
    alignas(16) uint8_t tail_b[2][64];
    alignas(16) static const uint8_t zero_block[64] = {};

    const size_t ntail_a = build_tail(in_a, len_a, tail_a);
    const size_t ntail_b = build_tail(in_b, len_b, tail_b);
    const size_t full_a = len_a / 64;
    const size_t full_b = len_b / 64;
    const size_t total_a = full_a + ntail_a;
    const size_t total_b = full_b + ntail_b;
    const size_t blocks = total_a > total_b ? total_a : total_b;

    __m128i abef_a, cdgh_a, abef_b, cdgh_b;
    load_state(INITIAL, abef_a, cdgh_a);
    load_state(INITIAL, abef_b, cdgh_b);

    const __m128i bswap = _mm_set_epi64x(
        static_cast<long long>(0x0c0d0e0f08090a0bULL),
        static_cast<long long>(0x0405060700010203ULL));

    alignas(16) __m128i st[4];
    alignas(16) __m128i msg[8];

    for (size_t i = 0; i < blocks; ++i) {
        const bool active_a = i < total_a;
        const bool active_b = i < total_b;
        /* A lane that has run out of message compresses a zero block and then
         * throws the result away by reloading its saved state, so both lanes
         * stay in lockstep without branching inside the compress body. */
        const uint8_t* ba = (i < full_a)  ? in_a + i * 64
                            : active_a    ? tail_a[i - full_a]
                                          : zero_block;
        const uint8_t* bb = (i < full_b)  ? in_b + i * 64
                            : active_b    ? tail_b[i - full_b]
                                          : zero_block;

        const __m128i save_abef_a = abef_a;
        const __m128i save_cdgh_a = cdgh_a;
        const __m128i save_abef_b = abef_b;
        const __m128i save_cdgh_b = cdgh_b;

        st[0] = abef_a;
        st[1] = cdgh_a;
        st[2] = abef_b;
        st[3] = cdgh_b;

        /* Big-endian word order for the message windows. VEX-128 here is
         * deliberate: it leaves the YMM upper clean, so it costs nothing at the
         * boundary with the legacy-SSE compress. */
        const __m128i* pa = reinterpret_cast<const __m128i*>(ba);
        const __m128i* pb = reinterpret_cast<const __m128i*>(bb);
        msg[0] = _mm_shuffle_epi8(_mm_loadu_si128(pa + 0), bswap);
        msg[1] = _mm_shuffle_epi8(_mm_loadu_si128(pa + 1), bswap);
        msg[2] = _mm_shuffle_epi8(_mm_loadu_si128(pa + 2), bswap);
        msg[3] = _mm_shuffle_epi8(_mm_loadu_si128(pa + 3), bswap);
        msg[4] = _mm_shuffle_epi8(_mm_loadu_si128(pb + 0), bswap);
        msg[5] = _mm_shuffle_epi8(_mm_loadu_si128(pb + 1), bswap);
        msg[6] = _mm_shuffle_epi8(_mm_loadu_si128(pb + 2), bswap);
        msg[7] = _mm_shuffle_epi8(_mm_loadu_si128(pb + 3), bswap);

        compress2(st, msg);

        if (active_a) {
            abef_a = _mm_add_epi32(st[0], save_abef_a);
            cdgh_a = _mm_add_epi32(st[1], save_cdgh_a);
        } else {
            abef_a = save_abef_a;
            cdgh_a = save_cdgh_a;
        }
        if (active_b) {
            abef_b = _mm_add_epi32(st[2], save_abef_b);
            cdgh_b = _mm_add_epi32(st[3], save_cdgh_b);
        } else {
            abef_b = save_abef_b;
            cdgh_b = save_cdgh_b;
        }
    }

    uint32_t sa[8], sb[8];
    store_state(sa, abef_a, cdgh_a);
    store_state(sb, abef_b, cdgh_b);
    finish(sa, out_a);
    finish(sb, out_b);
}

#undef SA0
#undef SA1
#undef SB0
#undef SB1
#undef TA
#undef TB
#undef WA0
#undef WA1
#undef WA2
#undef WA3
#undef WB0
#undef WB1
#undef WB2
#undef WB3
#undef RND_HEAD
#undef RND_TAIL
#undef MSG1
#undef MSG2
#undef GRP_PLAIN
#undef GRP_M1
#undef GRP_FULL
#undef GRP_M2
