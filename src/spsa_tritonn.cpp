// spsa_tritonn.cpp — Clean-room port of libastroSPSA::SPSA() entry point.
//
// Source disassemblies consulted (read-only):
//   pike-miner/spsa_main.asm                     (1254 lines, full SPSA orchestrator)
//   pike-miner/spsa_setBuckets.asm               (148 lines, prefix-sum CDF — already ported)
//   pike-miner/spsa_processTemplate_avx2.asm     (353 lines, already ported by other agent)
//   pike-miner/spsa_processStamps_avx2.asm       (497 lines, already ported by other agent)
//   pike-miner/spsa_decompress_avx2.asm          (418 lines, already ported by other agent)
//   pike-miner/spsa_update_sa_block32.asm        (298 lines, decoded inline below)
//
// Cross-checked against:
//   vault/02-projects/dirtybird-miner/data-science/agent6-spsa-port-spec.md
//   vault/02-projects/dirtybird-miner/data-science/tritonn-spsa-algorithm-spec-from-dumps.md
//
// All DRM/nanosleep paths intentionally omitted (clean-room).

#include "astroworker.h"
#include "spsa_tritonn_dump.h"

#include <algorithm>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace deroluna_tritonn {

// ============================================================================
// External helper forward decls already implemented by sibling agents.
// ============================================================================
extern void     setBuckets(::workerData& w);
extern void     process_template_optimized(::workerData& w,
                                           const uint8_t* data, int data_size,
                                           uint32_t* out_buf,
                                           uint32_t (*buckets)[256]);
extern void     processStamps(::workerData& w,
                              const uint8_t* data, int data_size,
                              uint32_t* out_buf,
                              int* templateIdxIO);
extern uint64_t decompress(::workerData& w, int data_size);

// ============================================================================
// update_sa_block32 — fully decoded from spsa_update_sa_block32.asm.
// The asm is a 32-iteration unrolled loop; each iteration does:
//   a    = data[i];
//   b    = data[i+1];
//   slot = bHeadIdx[a][b]++;       // post-increment (lea +1 + store)
//   sa[slot] = base_pos + i;
// Caller supplies: data = &input[base_pos], sa = sa_prelim,
//                  bHeadIdx = workerData::bHeadIdx, base_pos = i.
// Cost: 32 unrolled radix scatters per call. Hot path inside Phase 14 when
// a bucket has >= 0x41 entries (see spsa_main.asm line 4120). Below that
// threshold the asm falls through to the non-unrolled scalar tail.
// ============================================================================
static inline void update_sa_block32(const uint8_t* data,
                                     uint32_t*      sa,
                                     uint32_t       (*bHeadIdx)[256],
                                     uint32_t       base_pos) noexcept
{
    #pragma clang loop unroll(full)
    for (uint32_t i = 0; i < 32; ++i) {
        const uint8_t  a    = data[i];
        const uint8_t  b    = data[i + 1];
        const uint32_t slot = bHeadIdx[a][b];
        bHeadIdx[a][b]      = slot + 1;
        sa[slot]            = base_pos + i;
    }
}

// ============================================================================
// SortSliceLess — the per-template byte-bucket sort comparator.
//
// HOW THIS WAS DERIVED (read carefully — this is the keystone):
//
// At spsa_main.asm:3cd1 (forward stamp) and :3e13 (backward stamp) the asm
// stages FOUR 8-byte capture pointers on the stack at offsets 0xb0/0xb8/
// 0xc0/0xc8, then calls (via stripped relocations) two functions in
// succession with rcx = stampKeys+writePos, rdx = stampKeys+writePos+count,
// r9 = &captures. The first call has a depth-limit arg (r8 = 2*log2(count)),
// the second does not — this is the textbook signature of
// std::__introsort_loop followed by std::__final_insertion_sort. The
// captures, in store order:
//
//   forward (lines 3cbe-3d23):
//     [0xb0] -> &local_6c   (int)   = ebp = data_size       (loaded line 3cca: mov %ebp,0x6c(%rsp))
//     [0xb8] -> &local_74   (int)   = ebx = (posData & 0xff80) << 1 = rowBase (line 3cbe)
//     [0xc0] -> &local_70   (int)   = 0   (line 3cc2: movl $0,0x70)        <-- p_sentinel = p1's "0"
//     [0xc8] -> &local_a8   (ptr)   = input pointer (line 3cb1: copy from 0x50)
//
//   backward (lines 3df3-3e65):  identical layout EXCEPT
//     [0xb0] -> &local_6c   = edi = data_size - 1
//     [0xb8] -> &local_74   = ebx = rowBase (same)
//     [0xc0] -> &local_70   = 0xff   (line 3e04: movl $0xff,0x70)         <-- p_sentinel = p2's terminator
//     [0xc8] -> &local_a8   = input
//
// The comparator therefore receives FOUR captures in a fixed order:
//   (int len_or_lenMinus1, int rowBase, int p_sentinel, const uint8_t* input)
//
// NO `sortSlice` symbol exists in any RE dump. The introsort template's
// printed name in the relocation strings (lines 1242-1248) only shows the
// `$_0` lambda — and that one sorts uint32_t*, not uint8_t*. The byte-
// bucket sort target was emitted as a non-templated (or fully inlined)
// helper whose symbol got stripped. agent6's earlier guess of
// "data[rowBase + chunk*256 + p1] < data[rowBase + chunk*256 + p2]" was
// SPECULATIVE — it does not match the captures.
//
// What the predicate actually does (cross-checked against the iota8 seed
// memcpy at line 3ca4 + the stampKeys layout in tritonn-spsa-algorithm-
// spec-from-dumps.md): the bytes being sorted are indices a, b in [0, count).
// They get sorted by the input byte at row-relative position
// (rowBase + p_sentinel + index). The forward stamp has p_sentinel=0, so
// the sort key is `input[rowBase + idx]`. The backward stamp has
// p_sentinel=0xff, so the sort key is `input[rowBase + 0xff + idx]` —
// effectively the byte one row ahead at offset (idx - 1) modulo 256.
//
// In other words: the 0xff sentinel is the "look at the byte just before
// the next chunk" trick — for each candidate index in [0, count) we read
// data[rowBase + 0xff + index] = data[rowBase - 1 + (index + 0x100)] —
// equivalent to scanning the *suffix* prefix one row ahead. This matches
// the "p2 stamp = backward facing" semantics described in
// agent6-spsa-port-spec.md line 226.
//
// CONFIDENCE: HIGH on the capture set and signature. MEDIUM on the
// "input[rowBase + p_sentinel + index]" body — this is consistent with the
// captures and the iota8-seeded sort target, but I did not single-step a
// runtime trace against ground truth.
// ============================================================================
struct SortSliceCaptures {
    int            len_or_lenMinus1;   // [0xb0]
    int            rowBase;            // [0xb8]
    int            p_sentinel;         // [0xc0]   (0 forward, 0xff backward)
    const uint8_t* input;              // [0xc8]
};

struct SortSliceLess {
    const SortSliceCaptures* cap;
    bool operator()(uint8_t a, uint8_t b) const noexcept {
        // KEYSTONE FIX (2026-04-30 .obj-disasm RE):
        // The proprietary sortSlice lambda is a 256-STRIDE SUFFIX memcmp,
        // not a single-byte compare. Decoded from the inlined comparator body
        // in `__insertion_sort<sortSlice...>` at lines 4a-1bf of section
        // .text$_ZSt16__insertion_sort... (reference build).
        //
        // Asm semantic (lines 96-1ba):
        //   r14d = a << 8                   ; a * 256
        //   r9d  = cap1 - (a*256 + cap2 + cap3)   ; la
        //   r8d  = cap1 - (b*256 + cap2 + cap3)   ; lb
        //   r8d  = min(r8d, r9d)            ; cmovge
        //   if r8d <= 0: fallback branch (compare a vs b)
        //   else: memcmp(input+cap2+cap3+a*256, input+cap2+cap3+b*256, r8d)
        //         then tie-break by `a > b` when memcmp == 0 (lines 11d/180/1f1)
        //
        // Tie-break rationale: la < lb iff a > b (since a*256 > b*256 → smaller la).
        // "Shorter suffix sorts first" means the larger index returns true here.
        const int stride_a = static_cast<int>(a) * 256;
        const int stride_b = static_cast<int>(b) * 256;
        const int base     = cap->rowBase + cap->p_sentinel;
        const int la       = cap->len_or_lenMinus1 - stride_a - base;
        const int lb       = cap->len_or_lenMinus1 - stride_b - base;
        const int n        = la < lb ? la : lb;
        if (n <= 0) {
            // Both suffixes empty/negative → tie-break by index value
            // (asm line 130: cmp r10b,r11b; jbe — sorts larger index first).
            return a > b;
        }
        const uint8_t* pa = cap->input + base + stride_a;
        const uint8_t* pb = cap->input + base + stride_b;
        const int c = std::memcmp(pa, pb, static_cast<size_t>(n));
        if (c != 0) return c < 0;
        return a > b;   // memcmp tie → shorter suffix first
    }
};

// ============================================================================
// SuffixLess — derived from the SA introsort callsite at lines 4070-4509.
// At line 41a5..4319 the predicate body is fully visible:
//   r15 = a & 0x1ffff;  if (len - r15 < 0) skip
//   r12 = b & 0x1ffff;  if (len - r12 < 0) skip
//   min_len = min(len-r15, len-r12)
//   memcmp(input + r15, input + r12, min_len)
//   tie-break: shorter suffix is "less"
// The 0x1ffff mask = 17-bit position field (max 131071, fits in 70K input).
// This matches agent6 spec exactly.
// ============================================================================
struct SuffixLess {
    const uint8_t* input;
    int            len;
    bool operator()(uint32_t a, uint32_t b) const noexcept {
        // BUG-FIX (2026-04-30 .obj-disasm RE):
        // Decoded from __adjust_heap.IPjxj (sec 1, offset 0x4af0) at lines
        // 4b6c..4bcf in spsa.cpp.obj. Asm checks `lb < 0` and `la < 0` BOTH
        // jump to the same label (4b50 = LEFT-is-max path) which corresponds
        // to comp returning FALSE. Previous code returned TRUE for the bi-OOB
        // case — biased the SA introsort to mis-order out-of-bounds records.
        const uint32_t ai = a & 0x1ffffu;
        const uint32_t bi = b & 0x1ffffu;
        const int la = len - static_cast<int>(ai);
        const int lb = len - static_cast<int>(bi);
        if (lb < 0) return false;       // asm 4b84 jl 4b50
        if (la < 0) return false;       // asm 4b93 jl 4b50 (same target!)
        const int n  = la < lb ? la : lb;
        if (n > 0) {
            const int c = std::memcmp(input + ai, input + bi, static_cast<size_t>(n));
            if (c != 0) return c < 0;
        }
        // Tie-break: longer suffix sorts later → ai > bi means LHS shorter → LHS < RHS.
        // Equivalent to `la < lb`. Asm: 4bc4 cmp ebx,r15d / 4bc7 seta cl
        // (cl = ib > ia, unsigned) / 4bca and cl,al / 4bcc cmp 1 / 4bcf je 4b50.
        // Going to 4b50 means comp returned FALSE; we hit 4b50 when ib > ia
        // → LEFT_pos < RIGHT_pos → LEFT longer → LEFT > RIGHT → FALSE.
        return ai > bi;
    }
};

// ============================================================================
// Phase 4 helpers
//
// The asm at lines 33a2-35d4 has THREE concentric sweeps over a single
// template marker, all updating buckets_d[256][256]. The agent6 skeleton
// captured the outer-forward and outer-backward sweeps but marked the
// "doubly-nested middle sweep" (lines 3489-39d4 in agent6's count, which
// maps to spsa_main.asm:3489-35d4 here) as TODO. Decoded below.
// ============================================================================
namespace phase4_inner {

// Outer-forward sweep (asm lines 33a2-3469).
// Bumps buckets_d[input[off]][input[off+1]] for off in [rowBase, rowBase+pStart),
// step-throttled by the cnt counter so that step(=cnt>>8 + 1) is clamped to
// keyMask. When step is zero the bucket is skipped (asm line 3442 setg).
static inline void outer_forward(uint16_t (&bd)[256][256],
                                 const uint8_t* in,
                                 int      lenMinus4,
                                 uint32_t rowBase,
                                 uint32_t pStart,
                                 uint32_t keyMask,
                                 int      limit) noexcept
{
    int      cnt = limit - static_cast<int>((rowBase * 2) & 0x1ff00);
    uint32_t off = rowBase;
    for (uint32_t i = 0; i < pStart; ++i) {
        if (static_cast<int64_t>(off) >= lenMinus4) break;
        uint32_t step = static_cast<uint32_t>(cnt >> 8) + 1u;
        if (step > keyMask) step = keyMask;
        if (step > 0) bd[in[off]][in[off + 1]]++;
        --cnt;
        ++off;
    }
}

// Outer-backward sweep (asm lines 3489-34ce).
// The pre-condition (line 346a `cmp $0xfa,r12b`) gates this off when p2 > 0xfa.
static inline void outer_backward(uint16_t (&bd)[256][256],
                                  const uint8_t* in,
                                  int      lenU_minus4,
                                  uint32_t rowBase,
                                  uint32_t p2,
                                  uint32_t keyMask,
                                  int      limit) noexcept
{
    if (p2 > 0xfa) return;
    uint32_t off = rowBase + p2;
    int      cnt = limit - static_cast<int>((rowBase) & 0x1ff00) - static_cast<int>(p2 + 1);
    for (uint32_t i = p2; i < 0xfa; ++i) {
        ++off;
        if (static_cast<int>(off) >= lenU_minus4) break;
        uint32_t step = static_cast<uint32_t>(cnt >> 8) + 1u;
        if (step > keyMask) step = keyMask;
        if (step > 0) bd[in[off]][in[off + 1]]++;
        --cnt;
    }
}

// Doubly-nested MIDDLE sweep (asm lines 34d0-35d4 — the agent6 TODO).
// Decoded structure:
//   for (rowI = pStart_capped; rowI < keyMask; ++rowI):
//     base = rowBase + rowI*256 + (p1-3)
//     if (base + 256 > lenMinus4) break
//     if (keyMask == 1):
//        # short-row specialisation (asm 3534-356a):
//        # only the (rowI, lo) pair where lo = p1's low byte is bumped.
//        # bumps bucket once per matching pair across the row.
//        bd[in[base]][in[base+1]]++          // single increment per row
//     else:
//        # full row: walk j = 0..keyMask-1 (line 3580 add 0x200 stride)
//        # at each j compute idx = base + j*0x100 (one row at a time)
//        # then bd[in[idx]][in[idx+1]]++ (asm lines 35a5-35bf)
//        for (j = 0; j < keyMask; ++j):
//          idx = base + j*0x100
//          if idx < lenMinus4:
//             bd[in[idx]][in[idx+1]]++
//             bd[in[idx+0x100-1]][in[idx+0x100]]++   // +0xff/+0x100 pair
// The 4-byte unroll in the asm (3508 `and $0xfffffffe, %r11d`) means the
// j loop has step 2 internally with a tail. We collapse this back to scalar.
//
// CONFIDENCE: MEDIUM. The outer/inner index arithmetic is plain to read;
// the exact cnt-throttling for the middle sweep is not perfectly mirrored
// here (agent6 noted the asm uses three different cnt update rules). For
// PoW correctness, the bucket sums must match exactly, so this MUST be
// validated against ground-truth dumps before promotion.
static inline void middle_doubly_nested(uint16_t (&bd)[256][256],
                                        const uint8_t* in,
                                        int      lenMinus4,
                                        uint32_t rowBase,
                                        uint32_t p1,
                                        uint32_t keyMask,
                                        uint32_t pStartCapped,
                                        bool     pdOdd) noexcept
{
    // REWRITE (2026-04-30 audit per Ghidra decomp 294-336):
    //   - rowI iterates in BYTE units (not row-count * 256). Asm cursor
    //     `lVar30 = (rowBase + pStart - 3) + 0x100` advances by +1 per outer.
    //   - Inner loop pairs are BACK [cur-0x100][cur-0xff] + FORWARD [cur][cur+1]
    //     with j step 2 and bound `keyMask & 0x7E` (NOT keyMask, NOT step 1).
    //   - Odd-tail emit at `jEnd*0x100 + rowBase + rowI` only when (pd & 1) set.
    //   - keyMask==1 case skips inner loop entirely (uVar19=0 in asm), only
    //     odd-tail fires (always, since 1 & 1 = 1).
    //
    // Caller threads `pdOdd = (m.posData & 1) != 0`.
    const uint32_t pStart = (p1 < 4 ? 3u : p1) - 3u;
    if (pStartCapped < pStart) return;
    if (keyMask == 0) return;

    const uint32_t pairBound = keyMask & 0x7Eu;

    for (uint32_t rowI = pStart; rowI < pStartCapped; ++rowI) {
        // Outer gate: rowI < lenMinus4 - rowBase (decomp line 300)
        if (static_cast<int64_t>(rowI) >=
            static_cast<int64_t>(lenMinus4) - static_cast<int64_t>(rowBase)) break;

        const uint32_t cur0 = rowBase + rowI + 0x100u;

        if (keyMask != 1u) {
            uint32_t j   = 0;
            uint32_t cur = cur0;
            while (j != pairBound) {
                // BACK pair: bd[in[cur-0x100]][in[cur-0xff]]++ when in-bounds
                if (cur >= 0x100u &&
                    static_cast<int>(cur - 0x100u) < lenMinus4) {
                    bd[in[cur - 0x100u]][in[cur - 0xff]]++;
                }
                // FORWARD pair: bd[in[cur]][in[cur+1]]++ when in-bounds
                if (static_cast<int>(cur) < lenMinus4) {
                    bd[in[cur]][in[cur + 1]]++;
                }
                j   += 2;
                cur += 0x200u;
            }
        }

        // Odd-tail (asm 324-331): only when (pd & 1) set.
        if (pdOdd) {
            const uint32_t jEnd = (keyMask == 1u) ? 0u : pairBound;
            const int tailIdx =
                static_cast<int>(jEnd) * 0x100 +
                static_cast<int>(rowBase + rowI);
            if (tailIdx < lenMinus4) {
                bd[in[tailIdx]][in[tailIdx + 1]]++;
            }
        }
    }
}

}  // namespace phase4_inner

// ============================================================================
// Bigram tally helper (Phase 6 — lines 38e0-393e of the asm).
// 4-way unrolled inner loop: bd[in[i]][in[i+1]]++ for i in [0, n).
// Caller guarantees n+1 readable bytes.
// ============================================================================
static inline void tally_bigrams_4unroll(uint16_t (&bd)[256][256],
                                         const uint8_t* p, int n) noexcept
{
    int i = 0;
    const int n4 = n & ~3;
    for (; i < n4; i += 4) {
        bd[p[i  ]][p[i + 1]]++;
        bd[p[i + 1]][p[i + 2]]++;
        bd[p[i + 2]][p[i + 3]]++;
        bd[p[i + 3]][p[i + 4]]++;
    }
    for (; i < n; ++i) bd[p[i]][p[i + 1]]++;
}

// ============================================================================
// SPSA — entry point.
// Mirrors spsa_main.asm:3200..4500 (orchestrator).
// ============================================================================
uint64_t SPSA(const uint8_t* input, int len, ::workerData& w)
{
    // === DUMP CHECKPOINT 1: input blob at SPSA entry ========================
    if (dluna_dump_enabled()) {
        DLUNA_DUMP("input", input, static_cast<size_t>(len));
    }

    // Phase 1: DRM nanosleep — SKIP. The asm at 321f reads a byte global
    // and calls nanosleep; we omit entirely.

    // Phase 2/3: zero scratch buffers (asm 327c-3326).
    std::memset(&w.isBSlice,       0, sizeof(w.isBSlice));        // 70 B  (554 bits)
    std::memset(w.stampKeys,       0, sizeof(w.stampKeys));       // 554 B
    std::memset(w.stampTemplates,  0, sizeof(w.stampTemplates));
    std::memset(w.buckets_d,       0, sizeof(w.buckets_d));       // 128 KB
    // Note: sa_prelim is NOT cleared here. The asm doesn't clear it either —
    // bHeads[a][b] CDF + bHeadIdx[a][b] cursors define valid record range for
    // each bucket; gaps and stale data outside that range are not read.

    const int      lenMinus4 = len - 4;
    const uint32_t lenU      = static_cast<uint32_t>(len);
    const int      rowCount  = len >> 8;
    const int      tplCount  = w.templateIdx;

    // Local stack bitmap (asm 35d6-35e3 + 3848 cap=0x115).
    std::bitset<0x115> rowVisited;

    // ----- Phase 4: bigram seeding from astroTemplate[] ----------------------
    // asm 3331-35d4. For each template marker, accumulate forward + middle +
    // backward bigrams into buckets_d.
    if (tplCount > 0) {
        for (int t = 0; t < tplCount; ++t) {
            const ::templateMarker& m  = w.astroTemplate[t];
            const uint16_t          pd = m.posData;
            const uint32_t          rowBase = static_cast<uint32_t>(pd & 0xff80) * 2u;
            if (static_cast<int>(rowBase) >= lenMinus4) continue;

            uint8_t p1 = m.p1;
            uint8_t p2 = m.p2;
            // Skip if (p1==0 && p2==0xff) — asm 33b9-33c0.
            if (p1 == 0 && p2 == 0xff) continue;

            const uint32_t keyMask = pd & 0x7f;
            uint32_t       limit   = keyMask * 0x100u + rowBase;
            if (static_cast<uint32_t>(lenMinus4) < limit) limit = lenMinus4;

            // p1 clamp (asm 33e2-33ea).
            if (p1 < 4) p1 = 3;
            const uint32_t pStart       = p1 - 3u;
            const uint32_t p2Cap        = p2 < 0xfb ? p2 : 0xfb;
            const uint32_t pStartCapped = p2Cap + 1u;

            // Forward sweep — asm 33a2-3469.
            phase4_inner::outer_forward(w.buckets_d, input, lenMinus4,
                                        rowBase, pStart, keyMask,
                                        static_cast<int>(limit));

            // Backward sweep — asm 3489-34ce.
            phase4_inner::outer_backward(w.buckets_d, input,
                                         static_cast<int>(lenU - 4u),
                                         rowBase, p2, keyMask,
                                         static_cast<int>(limit));

            // Middle doubly-nested sweep — asm 34e6-35d4 / Ghidra 294-336.
            // pdOdd = (pd & 1) gates the odd-tail emit (decomp line 324).
            const bool pdOdd = (pd & 1u) != 0u;
            phase4_inner::middle_doubly_nested(w.buckets_d, input, lenMinus4,
                                               rowBase, p1, keyMask,
                                               pStartCapped, pdOdd);
        }
    }

    // ----- Phase 5: rowVisited bitmap fill (asm 35e3-37a4) -------------------
    // For each template marker that isn't (p1<4 && p2>=0xfc) and has nonzero
    // span, set bits [posData>>7 .. +span). 4-way unrolled in the asm via
    // shlx+or; semantically a contiguous bit range set.
    bool early_phase6 = false;
    if (tplCount > 0) {
        for (int t = 0; t < tplCount; ++t) {
            const ::templateMarker& m = w.astroTemplate[t];
            const uint16_t pd   = m.posData;
            const uint32_t row0 = static_cast<uint32_t>(pd >> 7);
            const bool degen   = (m.p1 < 4) && (m.p2 >= 0xfc);
            const uint32_t span = pd & 0x7f;
            if (degen) continue;
            if (span == 0) continue;
            for (uint32_t k = 0; k < span; ++k) {
                const uint32_t bit = row0 + k;
                if (bit < 0x115) rowVisited.set(bit);
            }
            if (row0 + span >= static_cast<uint32_t>(rowCount)) {
                early_phase6 = true;  // asm 36b0..36b8
                break;
            }
        }
    }
    (void)early_phase6;

    // ----- Phase 6: row sweep over un-visited rows (asm 37c2-3940) -----------
    for (int r = 0; r < rowCount; ++r) {
        if (rowVisited.test(r)) continue;
        const int rowStart = r << 8;
        int       rowEnd   = rowStart + 0xfc;
        if (rowEnd > lenMinus4) rowEnd = lenMinus4;
        if (rowStart >= rowEnd) continue;
        tally_bigrams_4unroll(w.buckets_d, input + rowStart, rowEnd - rowStart);
    }

    // ----- Phase 7: tail bigrams (asm 3945-3a13) ----------------------------
    {
        const int tailStart = (rowCount << 8);
        const int tailEnd   = lenMinus4;
        for (int i = tailStart; i < tailEnd; ++i) {
            w.buckets_d[input[i]][input[i + 1]]++;
        }
        // Final 4 bytes (asm 3a13-3b5b).
        const int last4Start = lenMinus4;
        for (int i = last4Start; i < len - 1; ++i) {
            w.buckets_d[input[i]][input[i + 1]]++;
        }
        // Special last byte: bd[in[len-1]][0]++ via shl 9 + incw at offset 0.
        // asm 3b54-3b5b: movzbl (%rax,%rdx,1),%eax; shl $0x9,%eax; incw (%rbx,%rax,1)
        // — that's bd[in[len-1]][0]++.
        w.buckets_d[input[len - 1]][0]++;
    }

    // === DUMP CHECKPOINT 2: buckets_d post-Phase 4 (asm 0x3b5f) ============
    if (dluna_dump_enabled()) {
        DLUNA_DUMP("bd_post_phase4", &w.buckets_d[0][0], sizeof(w.buckets_d));
    }

    // ----- Phase 8: setBuckets (asm 3b5f-3b65 dispatch) ---------------------
    setBuckets(w);

    if (const char* e = std::getenv("DLUNA_INSTR_STAMPS"); e && e[0] == '1') {
        size_t visBits = 0;
        for (size_t i = 0; i < rowVisited.size(); ++i) if (rowVisited.test(i)) ++visBits;
        uint64_t bd_sum = 0;
        for (int a = 0; a < 256; ++a) for (int b = 0; b < 256; ++b) bd_sum += w.buckets_d[a][b];
        int unvisited = rowCount - static_cast<int>(visBits);
        std::fprintf(stderr,
            "[BUCKETS] tplCount=%d data_len=%d bd_sum=%llu rowCount=%d "
            "rowVis=%zu unvis=%d ph6_est=%d\n",
            tplCount, len, (unsigned long long)bd_sum, rowCount,
            visBits, unvisited, unvisited * 252);
    }

    // === DUMP CHECKPOINT 3+4: bHeads + bHeadIdx post-setBuckets (asm 0x3b6a) ==
    if (dluna_dump_enabled()) {
        DLUNA_DUMP("bheads",   &w.bHeads[0][0],   sizeof(w.bHeads));
        DLUNA_DUMP("bheadidx", &w.bHeadIdx[0][0], sizeof(w.bHeadIdx));
    }

    // ----- Phase 9: early-out gate (asm 3b6a-3b95) --------------------------
    // *(u16*)(p3 + 0x997ea) + *(i32*)(p3 + 0xd97e8); skip if >= 0x7800.
    // These map to two counters in workerData. agent6 marked TODO; we use
    // named fields exposed in astroworker.h: zeroRunCount + counter.
    {
        // Phase 9 gate fix (2026-04-30 audit): proprietary +0xd97e8 maps to
        // bHeadsTotal (sum of buckets_d entries written by setBuckets).
        // Previously hardcoded 0 → gate effectively disabled → orchestrator
        // ran full pipeline even on degenerate inputs.
        const uint32_t gateA = static_cast<uint32_t>(w.zeroRunCount);
        const uint32_t gateB = w.bHeadsTotal;
        if (gateA + gateB >= 0x7800u) return 0;
    }

    // ----- Phase 10: per-template byte-bucket sort + stampKeys fill ---------
    // asm 3b96-3eb9. For each template t with span>0:
    //   1. memcpy(stampKeys+writePos, iota8, span)            // line 3cac
    //   2. introsort_loop + final_insertion_sort over the
    //      stampKeys[writePos .. writePos+span] slice using
    //      SortSliceLess(input, len_or_lenMinus1, rowBase,
    //                    p_sentinel=0_or_0xff)                // lines 3d24-3d7c
    //   3. stampTemplates[stampCount++] = t                   // line 3bf0
    //   4. record writePos in keySpotA (offset +2) or
    //      keySpotB (offset +4) of astroTemplate[t]            // line 3c3b/3da1
    //   5. forward stamp -> isBSlice clear; backward stamp -> isBSlice.set
    //      via shlx + or at line 3c10-3c24.
    int phase10_stampCount = 0;   // hoisted out so Phase 12 can pass it to processStamps
    {
        int& stampCount = phase10_stampCount;
        int writePos   = 0;
        // 2026-04-30 instrumentation: log final writePos / stampCount via env DLUNA_INSTR_STAMPS=1.
        // If writePos > sizeof(stampKeys), Phase 10 corrupted stampTemplates[].

        for (int t = 0; t < tplCount; ++t) {
            ::templateMarker& m  = w.astroTemplate[t];
            const uint16_t    pd = m.posData;
            const uint32_t    rowBase = static_cast<uint32_t>(pd & 0xff80) * 2u;
            if (lenU <= rowBase) break;     // hard exit (asm 3c75)

            const uint8_t span = static_cast<uint8_t>(pd & 0x7f);
            if (span == 0) continue;

            // ---- Forward stamp (m.p1 != 0) — asm 3c8a-3d7c -----------------
            if (m.p1 != 0) {
                assert(stampCount < 0x22a);
                assert(writePos + span <= static_cast<int>(sizeof(w.stampKeys)));

                uint8_t* slice = w.stampKeys + writePos;
                std::memcpy(slice, w.iota8, span);

                SortSliceCaptures cap{ /*len_or_lenMinus1=*/ len,
                                       /*rowBase=*/         static_cast<int>(rowBase),
                                       /*p_sentinel=*/      0,
                                       /*input=*/           input };
                SortSliceLess pred{&cap};
                // std::sort matches introsort_loop + final_insertion_sort
                std::sort(slice, slice + span, pred);

                w.stampTemplates[stampCount] = static_cast<uint8_t>(t);
                // forward stamp: bit STAYS clear in isBSlice
                m.keySpotA = static_cast<uint16_t>(writePos);    // asm 3c3b: ax->offset+4
                ++stampCount;
                writePos += span;
            }

            // ---- Backward stamp (m.p2 != 0xff && in-range) — asm 3dad-3eb9
            if (m.p2 != 0xff) {
                const int probe = static_cast<int>(rowBase) + m.p2 + 1;
                if (probe >= len - 1) continue;   // asm 3dca

                assert(stampCount < 0x22a);
                assert(writePos + span <= static_cast<int>(sizeof(w.stampKeys)));

                uint8_t* slice = w.stampKeys + writePos;
                std::memcpy(slice, w.iota8, span);

                SortSliceCaptures cap{ /*len_or_lenMinus1=*/ len - 1,
                                       /*rowBase=*/         static_cast<int>(rowBase),
                                       /*p_sentinel=*/      0xff,
                                       /*input=*/           input };
                SortSliceLess pred{&cap};
                std::sort(slice, slice + span, pred);

                w.stampTemplates[stampCount] = static_cast<uint8_t>(t);
                // backward stamp: bit SET in isBSlice — asm 3c0b-3c24
                {
                    const uint32_t bit = static_cast<uint32_t>(stampCount);
                    w.isBSlice.set(bit);
                }
                m.keySpotB = static_cast<uint16_t>(writePos);    // asm 3da1: ax->offset+2
                ++stampCount;
                writePos += span;
            }
        }

        // Instrumentation: detect stampKeys overflow.
        if (const char* e = std::getenv("DLUNA_INSTR_STAMPS"); e && e[0] == '1') {
            std::fprintf(stderr,
                "[STAMPS] writePos=%d stampCount=%d cap=%zu overflow=%d\n",
                writePos, stampCount, sizeof(w.stampKeys),
                writePos > static_cast<int>(sizeof(w.stampKeys)));
        }
    }

    // ----- Phase 11: process_template_optimized (asm 3ebe-3ef3) -------------
    process_template_optimized(w, input, len, w.sa_prelim, w.bHeadIdx);

    // ----- Phase 12: processStamps (asm 3ef3-3f1a) --------------------------
    // BUG-FIX (2026-04-30 inference): processStamps's templateIdxIO arg is the
    // STAMP COUNT (from Phase 10), NOT the TEMPLATE COUNT (w.templateIdx).
    // Previous code passed w.templateIdx (~55-79), under-iterating ~30% — direct
    // cause of the consistent-negative LEN-MISMATCH delta in iter2/iter3.
    {
        int templateIdxIO = phase10_stampCount;   // was: w.templateIdx
        processStamps(w, input, len, w.sa_prelim, &templateIdxIO);
        // Note: don't write back to w.templateIdx (which is the template count).
    }

    // === DUMP CHECKPOINT 5: sa_prelim post-Phase 12 (asm 0x3f1a) ===========
    if (dluna_dump_enabled()) {
        DLUNA_DUMP("sa_prelim_post_phase12",
                   &w.sa_prelim[0], sizeof(w.sa_prelim));
        DLUNA_DUMP("bheadidx_post_phase12",
                   &w.bHeadIdx[0][0], sizeof(w.bHeadIdx));
        dluna_dump_manifest(len, static_cast<int>(w.data_len), w.templateIdx);
    }

    // ----- Phase 13: SA tail seeding (asm 3f1a-4013) ------------------------
    // Plant the last 4 SA positions: for k in 0..3 (asm peels each):
    //   pos = lenMinus4 + k
    //   lo  = (pos+1 < len-1) ? input[pos+1] : 0
    //   slot = bHeadIdx[input[pos]][lo]++
    //   sa_prelim[slot] = pos
    {
        const int last = lenMinus4;
        const int upper = len - 1;
        for (int k = 0; k < 4; ++k) {
            const int pos = last + k;
            if (pos >= len) break;
            const uint8_t lo = (pos + 1 < upper) ? input[pos + 1] : 0;
            const uint32_t slot = w.bHeadIdx[input[pos]][lo];
            w.sa_prelim[slot] = static_cast<uint32_t>(pos);
            w.bHeadIdx[input[pos]][lo] = slot + 1;
        }
    }

    // ----- Phase 14: nested bucket walk + suffix introsort ------------------
    // asm 4013-44d0. For each (hi, lo) bucket where (e - s) >= 2:
    //   if (e - s) >= 0x41:  introsort_loop(..., depth=2*log2(n)) using
    //                        SuffixLess; then final_insertion_sort.
    //                        AVX2 update_sa_block32 chains may interleave
    //                        within (asm 4179-43e6).
    //   else:                final_insertion_sort directly (asm 4500-4509).
    //
    // We use std::sort which IS introsort + insertion_sort under the hood.
    // The update_sa_block32 path inside the asm is the radix scatter that
    // FILLS sa_prelim for buckets — it does NOT replace sort, it precedes
    // it. That fill happens in process_template_optimized (Phase 11), so
    // here we only sort.
    {
        SuffixLess pred{input, len};
        for (int hi = 0; hi < 256; ++hi) {
            for (int lo = 0; lo < 256; ++lo) {
                const uint32_t s = w.bHeads[hi][lo];
                const uint32_t e = w.bHeadIdx[hi][lo];
                if (e <= s + 1) continue;
                std::sort(w.sa_prelim + s, w.sa_prelim + e, pred);
            }
        }
    }

    // ----- Gap diagnostic (DLUNA_DIAG_GAPS=1) -------------------------------
    // For each (a,b), compare bHeadIdx (write cursor) vs next bucket's start
    // (bHeads[a][b+1]). Difference = unfilled slots = zero-gap that decompress
    // misreads as literal-position-0 records.
    if (const char* dg = std::getenv("DLUNA_DIAG_GAPS"); dg && dg[0] == '1') {
        uint64_t total_alloc = 0, total_used = 0, gap_buckets = 0;
        uint32_t max_gap = 0;
        for (int a = 0; a < 256; ++a) {
            for (int b = 0; b < 256; ++b) {
                uint32_t next_start;
                if (b < 255)            next_start = w.bHeads[a][b + 1];
                else if (a < 255)       next_start = w.bHeads[a + 1][0];
                else                    next_start = w.bHeadIdx[255][255];
                const uint32_t used = w.bHeadIdx[a][b];
                const uint32_t alloc_end = next_start;
                if (alloc_end > used) {
                    const uint32_t g = alloc_end - used;
                    if (g > max_gap) max_gap = g;
                    ++gap_buckets;
                }
                total_used  += used - w.bHeads[a][b];
                total_alloc += alloc_end - w.bHeads[a][b];
            }
        }
        std::fprintf(stderr,
            "[GAP] alloc=%llu used=%llu diff=%llu gap_buckets=%llu max_gap=%u "
            "headIdx[255][255]=%u\n",
            (unsigned long long)total_alloc,
            (unsigned long long)total_used,
            (unsigned long long)(total_alloc - total_used),
            (unsigned long long)gap_buckets,
            max_gap,
            w.bHeadIdx[255][255]);
    }

    // ----- Compaction (DLUNA_COMPACT_SA=1) ----------------------------------
    // Phase 4-7 bigram seeding over-counts vs PTO+PS actual emits, leaving
    // zero-gap slots in sa_prelim that decompress mis-reads as lit-pos-0.
    // Walk all 65536 buckets, copy used range [bHeads,bHeadIdx) contiguously.
    // This is a hypothesis-test fix; the RE-correct fix is upstream in
    // middle_doubly_nested. If compaction makes pow(a) tritonn match libsais,
    // the gap is the entire bug.
    if (const char* cs = std::getenv("DLUNA_COMPACT_SA"); cs && cs[0] == '1') {
        uint32_t write = 0;
        for (int a = 0; a < 256; ++a) {
            for (int b = 0; b < 256; ++b) {
                const uint32_t s = w.bHeads[a][b];
                const uint32_t e = w.bHeadIdx[a][b];
                if (e > s) {
                    if (write != s) {
                        std::memmove(w.sa_prelim + write, w.sa_prelim + s,
                                     (e - s) * sizeof(uint32_t));
                    }
                    write += (e - s);
                    w.bHeads[a][b]   = write - (e - s);
                    w.bHeadIdx[a][b] = write;
                } else {
                    w.bHeads[a][b]   = write;
                    w.bHeadIdx[a][b] = write;
                }
            }
        }
        if (const char* dg = std::getenv("DLUNA_DIAG_GAPS"); dg && dg[0] == '1') {
            std::fprintf(stderr, "[COMPACT] packed %u records\n", write);
        }
    }

    // ----- Phase 15: terminator + decompress (asm 44d0-44f6) ----------------
    // *(u32*)(p3 + 0x1197e8) is post-incremented; sa_prelim[ctr] = 0xFFFFFFFF.
    // Then decompress(p3, len) is called and its return value is the SPSA
    // result.
    //
    // BUG FIX (2026-04-28): the proprietary lib's +0x1197e8 cursor advances
    // throughout phases 11-13 as positions get scattered (it tracks the
    // high-water mark across all bucket writes). Our clone never advances
    // saWriteIdx during scatters, so on every SPSA() call ctr was 0 and the
    // terminator clobbered sa_prelim[0] — exactly the SA[0]=0xFFFFFFFF
    // sentinel the harness saw. Use bHeadIdx[255][255] (the natural high
    // water mark of the per-bucket scatter) as the terminator slot, then
    // mirror that into saWriteIdx for any downstream consumer.
    {
        // CORRECTNESS FIX (2026-04-30 inference): the proprietary lib's saWriteIdx
        // at +0x1197e8 is NOT a separate field — it's literally bHeadIdx[255][255]
        // (math: bHeadIdx base 0xd97ec + (255*256+255)*4 = 0x1197e8). The asm reads
        // and post-increments that specific slot; we must do the same for asm
        // faithfulness. Iter8's "global max" was a guess that diverged.
        const uint32_t ctr = w.bHeadIdx[255][255];
        w.bHeadIdx[255][255] = ctr + 1;   // asm post-increment at 44d6/44d9
        w.saWriteIdx        = ctr + 1;    // mirror for downstream
        w.sa_prelim[ctr]    = 0xFFFFFFFFu;
    }

    return decompress(w, len);
}

// ============================================================================
// Validation hook for the diff harness in spsa_local.cpp.
// Mirrors the g_dbg_capture API so the existing harness can A/B compare
// SPSA_Integrated() against this clean-room SPSA() byte-for-byte.
// ============================================================================
extern "C" uint64_t SPSA_tritonn_entry(const uint8_t* in, int len, ::workerData& w)
{
    return SPSA(in, len, w);
}

}  // namespace deroluna_tritonn
