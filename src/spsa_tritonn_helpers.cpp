/* spsa_tritonn_helpers.cpp — STUB implementations of the four Tritonn-port
 * helpers expected by spsa_tritonn.cpp. Real implementations to be transcribed
 * from agent reports in `vault/02-projects/dirtybird-miner/data-science/spsa-port-state-2026-04-30.md`
 * and the agent transcripts in `~/.claude/projects/.../subagents/`.
 *
 * Status (2026-04-30): stubs; building towards a compilable validation
 * harness. Each helper returns immediately so the SPSA() orchestrator
 * walks the phases without crashing. Once we wire in the real bodies
 * (~600 LOC across all four), the diff harness in dluna_hash with
 * env DLUNA_VERIFY_TRITONN=1 will compare byte-for-byte against libsais.
 *
 * DO NOT enable USE_TRITONN_SPSA for production until pow("a") matches
 * canonical via this path.
 */

#include "astroworker.h"
#include "spsa_tritonn_dump.h"
#include <cstdint>
#include <cstring>
/* The Tritonn verification path (DLUNA_VERIFY_TRITONN, x86 dev-only) uses AVX2
 * helpers. Guard the whole body so non-x86 (aarch64) builds don't pull
 * <immintrin.h>; nothing references these symbols on non-x86 (see astrobwt.cpp). */
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#include <vector>
#include <openssl/sha.h>

namespace deroluna_tritonn {

namespace { thread_local struct { uint64_t pto_emits=0; uint64_t ps_emits=0; uint64_t dc_literals=0; uint64_t dc_compressed_records=0; uint64_t dc_expanded_positions=0; } g_emit_counts; }
extern "C" void dbg_emit_reset() { g_emit_counts = {}; }
extern "C" void dbg_emit_get(uint64_t* pto, uint64_t* ps, uint64_t* dc_lit, uint64_t* dc_comp, uint64_t* dc_pos) {
    *pto = g_emit_counts.pto_emits; *ps = g_emit_counts.ps_emits;
    *dc_lit = g_emit_counts.dc_literals; *dc_comp = g_emit_counts.dc_compressed_records; *dc_pos = g_emit_counts.dc_expanded_positions;
}

/* Byte-stream capture for the diff harness. When DLUNA_VERIFY_TRITONN is on,
 * dluna_hash sets the capture flag, runs SPSA_tritonn_entry (which calls
 * decompress), then reads the captured byte stream and diffs against libsais's
 * SA bytes. Off-by-default (zero overhead in production). */
namespace { thread_local std::vector<uint8_t> g_dbg_sha_capture; thread_local bool g_dbg_capture_sha = false; }
extern "C" void dbg_sha_capture_set(int on) { g_dbg_capture_sha = (on != 0); if (on) g_dbg_sha_capture.clear(); }
extern "C" const uint8_t* dbg_sha_capture_data(size_t* len_out) { *len_out = g_dbg_sha_capture.size(); return g_dbg_sha_capture.data(); }

/* Wrapper around SHA256_Update that ALSO appends the data to the debug
 * capture buffer (when enabled). Used inside decompress() in place of direct
 * SHA256_Update calls so the diff harness sees the exact byte stream. */
static inline void dbg_sha_update(SHA256_CTX* ctx, const void* data, size_t len) {
    if (g_dbg_capture_sha) {
        size_t old = g_dbg_sha_capture.size();
        g_dbg_sha_capture.resize(old + len);
        std::memcpy(g_dbg_sha_capture.data() + old, data, len);
    }
    // Also stream to ground-truth dump (env DLUNA_DUMP_GROUND_TRUTH=1).
    ::deroluna_tritonn::dluna_dump_sha_append(data, len);
    SHA256_Update(ctx, data, len);
}

void setBuckets(::workerData& w) {
    /* Real impl: prefix-sum buckets_d[256][256] (uint16) into bHeads (uint32),
     * copy bHeads → bHeadIdx as write cursors. From spsa_setBuckets.asm
     * (148 lines, HIGH confidence agent decode).
     *
     * Also writes w.bHeadsTotal (proprietary +0xd97e8) for Phase 9 gate. */
    uint32_t running = 0;
    for (int a = 0; a < 256; ++a) {
        for (int b = 0; b < 256; ++b) {
            uint32_t c = w.buckets_d[a][b];
            w.bHeads[a][b] = running;
            w.bHeadIdx[a][b] = running;
            running += c;
        }
    }
    w.bHeadsTotal = running;   // total bigram count, used by Phase 9 gate
}

// ============================================================================
// process_template_optimized — port of spsa_processTemplate_avx2.asm (353 lines).
//
// Per-template loop:
//   posData       = m.posData                                          (asm b3..d6)
//   chunkStartByte = (posData & 0xff80) << 1                           (asm bf..c2)
//   if (chunkStartByte >= data_size) return;                           (asm cd hard exit)
//   count = posData & 0x7f; if zero, skip                              (asm d3..e1)
//   if (m.p1 < 4) p1 = 3 else p1 = m.p1                                (asm f1..fa)
//   p2 = m.p2                                                          (asm fe)
//   base_const = chunkStartByte + (p1 - 3)                             (asm 124..128, "r14")
//
// Per-stamp loop s in [0, count):
//   off       = chunkStartByte + s*256                                 (asm 190..197, "r10")
//   if (base_const + s*256 >= data_size - 1) skip SIMD+scalar          (asm 1a1..1a8)
//   end_cap   = min(off + p2 + 1, data_size - 4)                       (asm 1aa..1ba)
//
//   SIMD body (asm 210..46e), pos = base_const + s*256, stride 16:
//     For each k in 0..15:
//       a = data[pos + k]; b = data[pos + k + 1];
//       old = buckets[a][b]++;
//       sa_prelim[old] = pos + k;
//
//   Scalar tail (asm 4a5..4e4): same op one position at a time, until pos
//     reaches end_cap. When pos >= ds_m1 the scatter is suppressed but
//     the loop variable still advances.
//
//   Filter loop (asm 519..55a):
//     For r10 in [chunkStartByte + 0xfc + s*256,
//                  min((s+1)*256 + chunkStartByte, data_size - 4)):
//       if ((uint8_t)r10 > p2):                                       (asm 530 jbe → skip when <=)
//         a = data[r10]; b = data[r10+1];
//         old = buckets[a][b]++;
//         sa_prelim[old] = r10;
//
// Note: function name says "_avx2" but the asm uses xmm0/xmm1 (16-byte
// loads) — effectively SSE-style with 16 byte-pairs unrolled per iteration.
// We collapse the unroll back to a scalar inner loop; the compiler can
// re-vectorise if needed.
//
// CONFIDENCE: HIGH on the sa_prelim formula. Cross-checked against asm
//   r14/rdi/rsp+0x78/rbx LEA bases at lines 124..14f and the per-lane
//   stores at 23c, 25d, 27d-282, 2a4, 2c4-2c8, ..., 453-457: each lane
//   k stores `base_const + r12 + k = pos + k` to (r9, r13, 4) =
//   sa_prelim[old]. Agent 1's first guess "pos+k" was correct (where
//   pos = base_const + s*256 + 16*j, NOT off = chunkStartByte + s*256).
// CONFIDENCE: MEDIUM on filter-loop comparand polarity (asm 537 jbe means
//   skip when low byte <= p2; we emit when low byte > p2 — verified against
//   the r10++ at 520 and conditional jump direction).
// ============================================================================
void process_template_optimized(::workerData& w,
                                const uint8_t* data, int data_size,
                                uint32_t* sa_prelim,
                                uint32_t (*buckets)[256])
{
    const int templateIdx = w.templateIdx;
    if (templateIdx <= 0) return;

    const ::templateMarker* tmpl = &w.astroTemplate[0];

    // Bounds derived from data_size (asm 33..52).
    const int ds_m1 = data_size - 1;
    const int ds_m4 = data_size - 4;

    for (int i = 0; i < templateIdx; ++i) {
        const ::templateMarker& m = tmpl[i];

        // Decode posData (asm b3..d6).
        const uint16_t posData       = m.posData;
        const int      chunkStartByte = static_cast<int>((posData & 0xff80u) << 1);
        if (chunkStartByte >= data_size) return;                                    // asm cd hard exit

        const int count = static_cast<int>(posData & 0x7fu);
        if (count == 0) continue;                                                   // asm de..e1

        // p1 clamp + p2 load (asm e8..107).
        int p1 = static_cast<int>(m.p1);
        if (p1 < 4) p1 = 3;
        const int p2 = static_cast<int>(m.p2);

        // base_const = chunkStartByte + (p1 - 3)  (asm 124..128, register r14).
        const int base_const = chunkStartByte + (p1 - 3);

        for (int s = 0; s < count; ++s) {
            // off = (s << 8) + chunkStartByte  (asm 190..197).
            const int off = (s << 8) + chunkStartByte;

            // simd_start = off + (p1 - 3) = base_const + s*256  (asm 1a1..1a8).
            const int simd_start = base_const + (s << 8);

            // asm 0x1a5-0x1a8: `cmp r13(=ds_m1), r11(=simd_start); jge 170` —
            // jumps to PER-STAMP TAIL skipping SIMD+scalar+filter for this stamp.
            // Filter is only reached via control flow that goes through the
            // SIMD/scalar phase (asm 0x4f0 sits AFTER scalar tail).
            if (simd_start >= ds_m1) continue;
            {
                // end_cap = min(off + p2 + 1, ds_m4)  (asm 1af..1ba).
                int end_cap = off + p2 + 1;
                if (end_cap > ds_m4) end_cap = ds_m4;

                // SIMD body: pos advances by 16. DO-WHILE semantics in asm.
                //   asm 0x1d5  : r8 = end_cap - 16
                //   asm 0x1e1  : cmp r8, r11 (=simd_start)
                //   asm 0x1e4  : jl 0x46f  ENTRY when simd_start < end_cap-16
                //                (then unconditionally falls into 0x210 SIMD body)
                //   asm 0x466  : r12 += 16
                //   asm 0x46a  : cmp r8, rcx where rcx = base_const+r12_old+16
                //                                    = pos_old + 16
                //   asm 0x46d  : jge 0x490  BACK-EDGE exit when pos_old+16 >= end_cap-16
                // ⇒ do-while: enter if simd_start < end_cap-16,
                //   continue while pos_old + 16 < end_cap - 16
                //              ⇔ pos_old < end_cap - 32  (STRICT less-than).
                // BUG-FIX (2026-04-28b): previous port used a single while-loop
                // `pos < end_cap-32` which UNDER-emits one SIMD iter (16 lanes)
                // when end_cap-32 <= simd_start < end_cap-16. Matches observed
                // -34KB / -8K positions LEN-MISMATCH (~512 stamps × 16).
                int pos = simd_start;
                const int simd_entry_limit = end_cap - 16;
                // BUG-FIX (2026-04-30 audit): asm 0x462 computes rcx=pos_old+16,
                // then 0x466 increments r12 by 16, then 0x46a-0x46d
                // `cmp r8(=end_cap-16), rcx; jge exit` — back-edge continues
                // while pos_NEW < end_cap-16 (rcx is pos AFTER the increment).
                // C++ tests after `pos+=16` so the limit must be `end_cap-16`,
                // not `end_cap-32`. Previous code under-emitted the final SIMD
                // iteration when end_cap-32 ≤ pos_old < end_cap-16.
                if (pos < simd_entry_limit) {
                    do {
                        for (int k = 0; k < 16; ++k) {
                            const uint8_t  a   = data[pos + k];
                            const uint8_t  b   = data[pos + k + 1];
                            const uint32_t old = buckets[a][b];
                            buckets[a][b]      = old + 1;
                            sa_prelim[old]     = static_cast<uint32_t>(pos + k);
                            ++g_emit_counts.pto_emits;
                        }
                        pos += 16;
                    } while (pos < simd_entry_limit);
                }

                // Scalar tail (asm 4a5..4e4). pos resumes where SIMD left off;
                // run until pos >= end_cap. When pos >= ds_m1 the scatter is
                // suppressed but pos still advances (asm 4b8..4bb cmp+jge).
                for (; pos < end_cap; ++pos) {
                    if (pos >= ds_m1) continue;
                    const uint8_t  a   = data[pos];
                    const uint8_t  b   = data[pos + 1];
                    const uint32_t old = buckets[a][b];
                    buckets[a][b]      = old + 1;
                    sa_prelim[old]     = static_cast<uint32_t>(pos);
                    ++g_emit_counts.pto_emits;
                }
            }

            // Filter loop (asm 519..55a). r10 ∈ [chunkStartByte + 0xfc + s*256,
            // min((s+1)*256 + chunkStartByte, ds_m4)). Emit only when low byte
            // of r10 > p2 (asm 0x530 cmp+jbe-to-skip).
            int r10 = chunkStartByte + 0xfc + (s << 8);
            int filt_end = chunkStartByte + ((s + 1) << 8);
            if (filt_end > ds_m4) filt_end = ds_m4;
            for (; r10 < filt_end; ++r10) {
                if (static_cast<uint8_t>(r10) <= static_cast<uint8_t>(p2)) continue;
                const uint8_t  a   = data[r10];
                const uint8_t  b   = data[r10 + 1];
                const uint32_t old = buckets[a][b];
                buckets[a][b]      = old + 1;
                sa_prelim[old]     = static_cast<uint32_t>(r10);
                ++g_emit_counts.pto_emits;
            }
        }
    }
}

void processStamps(::workerData& w,
                   const uint8_t* data, int data_size,
                   uint32_t* out_buf,
                   int* templateIdxIO)
{
    /* Port of spsa_processStamps_avx2.asm (497 lines, MED-HIGH confidence
     * agent decode). Walks the wd.isBSlice (pairBitmap) bitmap of stamps,
     * resolves each stamp through wd.stampTemplates[] to a templateMarker,
     * derives a (start, count) byte-pair window in `data`, then scatters
     * 24-bit-prefix records ((stamp<<17) | 0x20000 | pos) into per-pair
     * buckets via wd.bHeadIdx[a][b] write cursors.
     *
     * Layout mapping (asm raw offsets -> clone's named fields):
     *   +0x78ab8  -> wd.astroTemplate[277]   (templateMarker, stride 8)
     *               .p1@+0, .p2@+1, .posData@+6 (u16, head_pos)
     *   +0x796d6  -> wd.stampTemplates[277]  (u8 stamp -> template index)
     *   +0x79364  -> wd.isBSlice            (std::bitset<554>; pairBitmap)
     *   +0xd97ec  -> wd.bHeadIdx[256][256]  (u32 write cursors)
     *
     * Note: `templateIdxIO` is a pointer-to-int (count of stamps to process)
     * in the clone's signature, NOT the templateMarker count. Agent's port
     * used `int&`; we dereference each iteration in case caller mutates it.
     */
    if (templateIdxIO == nullptr || *templateIdxIO <= 0) return;     // FROM 1916

    const int32_t  N      = data_size;
    const int32_t  Nm4    = N - 4;                                   // FROM 1925 (r10d)
    const auto&    tmplArr = w.astroTemplate;                        // stride sizeof(templateMarker)=8
    const uint8_t* stampToTmpl = w.stampTemplates;
    auto&          pairBitmap  = w.isBSlice;                         // std::bitset<554>
    uint32_t (*bHead)[256]     = w.bHeadIdx;                         // [256][256]

    const int32_t totalStamps = *templateIdxIO;

    for (int32_t s = 0; s < totalStamps; ++s) {                      // FROM 196b/1958
        // Bounds trap for stamp >= 554 (0x22a) — matches asm's int3 at 20ce
        if (s >= 0x22a) __builtin_trap();                            // FROM 1972

        const uint8_t  tmpl       = stampToTmpl[s];                  // FROM 1978
        const uint16_t head       = tmplArr[tmpl].posData;           // FROM 1980 (+0x78abe)
        const uint32_t headMasked = head & 0xFF80u;                  // FROM 1988
        const uint32_t startBase  = headMasked * 2u;                 // FROM 198b (lea ,rax,2)
        uint32_t       p2plus1    = uint32_t(tmplArr[tmpl].p2) + 1u; // FROM 198f,19af (r8d)
        int32_t        p1m3       = int32_t(tmplArr[tmpl].p1) - 3;   // FROM 1998,19b2 (r11d)

        // pairBitmap fixup — FROM 19a8/19b8/19bb/19c4
        // bt sets CF=bit; cmovae(=NC) zeroes p2plus1 if bit==0;
        // cmovb(=C) overwrites p1m3 with 0xFC (=252) if bit==1.
        const bool flag = pairBitmap.test(static_cast<size_t>(s));
        if (!flag) p2plus1 = 0;                                      // cmovae r8d, edi(=0)
        else       p1m3    = 0xFC;                                   // cmovb  r11d, edx(=0xFC)

        const int32_t inner    = p1m3 - int32_t(p2plus1);            // FROM 19c8
        const uint32_t startOff = startBase + p2plus1;               // FROM 19d0 (r15d)
        // count = N - startOff - 1, capped at `inner` — FROM 19d3..19de
        int32_t count = int32_t(N) - int32_t(startOff) - 1;
        if (count >= inner) count = inner;

        // Common per-stamp tag: ((s << 17) | 0x20000) — FROM 1a23..1a2a / 202f..2036
        const uint32_t stampTag = (uint32_t(s) << 17) + 0x20000u;

        if (count < 0x21) {                                          // FROM 19f1 jl 1b49
            // Skip AVX2 + xmm phases, go straight to scalar tail
            // FROM 2020..2091 — single-byte-pair-per-iter path
            // r14 = (data) + (startBase) + (p2plus1) + 1, indexed by rax in [0,count)
            // pos_record = startBase + p2plus1 + i (note: r15 reload at 2042)
            // pair (a,b) = (data[startBase+p2plus1+i], data[startBase+p2plus1+i+1])
            // TODO: verify scalar-tail pair direction (agent flagged MED-LOW
            //       confidence; agent originally read it as (data[pos-1],data[pos])
            //       but a re-read of asm (lines 2050-2065 with r14 base reload at
            //       203d-2047) gives (data[pos], data[pos+1]) matching the AVX
            //       path. Keeping AVX-aligned form pending byte-for-byte diff
            //       against libsais.).
            for (int32_t i = 0; i < count; ++i) {                    // FROM 2089/208c
                const int32_t pos = int32_t(startBase) + int32_t(p2plus1) + i;
                if (pos >= Nm4) break;                               // FROM 2054
                const uint8_t aa = data[pos];                        // FROM 2065 (re-derived)
                const uint8_t bb = data[pos + 1];                    // FROM 2060 (re-derived)
                uint32_t* slot = &bHead[aa][bb];                     // bHeadIdx[a][b]
                const uint32_t idx = (*slot)++;                      // FROM 2074..207e
                out_buf[idx] = uint32_t(pos) | stampTag;             // FROM 2085
                ++g_emit_counts.ps_emits;
            }
            continue;
        }

        // ---- Block-of-32 main loop — FROM 1a09..1b18 ----
        // r14 = startBase + p2plus1 (cumulative byte offset into data)
        const int32_t winStart   = int32_t(startBase) + int32_t(p2plus1);
        const int32_t block32end = count - 0x20;                     // FROM 1a20
        const int32_t block16end = count - 0x10;                     // FROM 1b49
        bool          earlyOut   = false;

        int32_t i = 0;                                               // FROM 1a34 (rdi)
        // BUG-FIX (2026-04-28): asm at 1a48 `cmp rsi,rdi; jae 1b30`
        // exits when rdi >= count-32 (unsigned), so loop continues
        // STRICTLY while i < count-32. Previous `i <= block32end`
        // emitted one extra 32-byte block per stamp when count-32 was
        // exactly reachable (count divisible by 32).
        for (; i < block32end; i += 32) {                            // FROM 1a40..1a4b
            // sliding 32-byte pair window — FROM asm 1a6e/1a73
            const uint8_t* p0 = data + winStart + i;
            __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p0));
            __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p0 + 1));
            alignas(32) uint8_t a32[32], b32[32];
            _mm256_store_si256(reinterpret_cast<__m256i*>(a32), v0);
            _mm256_store_si256(reinterpret_cast<__m256i*>(b32), v1);
            // 32 byte-pair scatters — FROM 1a90..1b18 (16 unrolls × 2 pairs)
            for (int k = 0; k < 32; ++k) {
                const int32_t pos = winStart + i + k;
                if (pos > Nm4) { earlyOut = true; break; }           // FROM 1a94 jle
                const uint8_t aa = a32[k];
                const uint8_t bb = b32[k];
                uint32_t* slot = &bHead[aa][bb];                     // bHeadIdx[a][b]
                const uint32_t idx = (*slot)++;                      // FROM 1ab5/1ac1
                out_buf[idx] = uint32_t(pos) | stampTag;             // FROM 1ac9 (or %r8d,%ebx)
                ++g_emit_counts.ps_emits;
            }
            if (earlyOut) break;
        }
        if (earlyOut) continue;

        // ---- 16-byte xmm tail — FROM 1b49..2017 ----
        // BUG-FIX (2026-04-28): asm at 1b84 `cmp r11,rdi; jge 2020`
        // exits when rdi >= count-16 (signed), continue strictly while
        // i < count-16. Same off-by-one pattern as block32 loop.
        for (; i < block16end; i += 16) {                            // FROM 1b80..1b84
            const uint8_t* p0 = data + winStart + i;
            __m128i v0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p0));
            __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p0 + 1));
            alignas(16) uint8_t a16[16], b16[16];
            _mm_store_si128(reinterpret_cast<__m128i*>(a16), v0);
            _mm_store_si128(reinterpret_cast<__m128i*>(b16), v1);
            for (int k = 0; k < 16; ++k) {
                const int32_t pos = winStart + i + k;
                if (pos > Nm4) { earlyOut = true; break; }           // FROM 1b9d jle
                const uint8_t aa = a16[k];
                const uint8_t bb = b16[k];
                uint32_t* slot = &bHead[aa][bb];
                const uint32_t idx = (*slot)++;
                out_buf[idx] = uint32_t(pos) | stampTag;
                ++g_emit_counts.ps_emits;
            }
            if (earlyOut) break;
        }
        if (earlyOut) continue;

        // ---- Scalar residue (post-AVX, < 16 bytes left) — FROM 2020..2091 ----
        // TODO: verify scalar-tail pair direction (agent flagged MED-LOW conf).
        for (; i < count; ++i) {
            const int32_t pos = winStart + i;
            if (pos >= Nm4) break;
            const uint8_t aa = data[pos];
            const uint8_t bb = data[pos + 1];
            uint32_t* slot = &bHead[aa][bb];
            const uint32_t idx = (*slot)++;
            out_buf[idx] = uint32_t(pos) | stampTag;
            ++g_emit_counts.ps_emits;
        }
    }
}

uint64_t decompress(::workerData& w, int data_size) {
    /* Port of spsa_decompress_avx2.asm (418 lines, MED-HIGH confidence).
     *
     * Algorithm: walk sa_prelim[] as a compressed-record stream.
     *   - rec == 0xFFFFFFFF                : terminator → flush+finalize
     *   - rec <= 0x1FFFF (17-bit position) : LITERAL, append rec as int32
     *   - else                             : COMPRESSED record. Expand into
     *                                        cnt int32 positions via pos_table
     *                                        deltas (pos = base + delta<<8) and
     *                                        append. Periodically SHA256_Update
     *                                        every 8 KiB (2048 dwords).
     *
     * Final 32-byte SHA256 digest is written to w.padding[] — dluna_hash
     * (in astrobwt.cpp) checks padding for nonzero to detect a successful
     * SPSA path and memcpys w.padding into the caller's `output` buffer.
     *
     * Field mapping clone vs proprietary lib:
     *   sa_prelim[]            <- w.sa_prelim
     *   pos_table base         <- w.stampKeys           (TODO: stampKeys is
     *                                                   only 554 B; agent's
     *                                                   port reads pt_off +
     *                                                   up to 128 bytes —
     *                                                   may overflow for
     *                                                   stamps with high
     *                                                   pt_off. Needs verify.)
     *   stamp_id_to_meta[]     <- w.stampTemplates
     *   meta_table[].hdr       <- w.astroTemplate[i].posData    (offset +6)
     *   meta_table[].pt_off_v0 <- w.astroTemplate[i].keySpotA   (offset +2)
     *   meta_table[].pt_off_v1 <- w.astroTemplate[i].keySpotB   (offset +4)
     *   variant_bitmap         <- w.isBSlice (std::bitset<554>)
     *   SHA256 digest dest     <- w.padding[0..32]
     *   SHA256_CTX             <- local (clone has no inline ctx at the
     *                            proprietary +0x204 offset within workerData)
     *   out_positions scratch  <- heap-allocated std::vector<int32_t>
     *                            (proprietary lib uses workerData+0x336b4;
     *                            no equivalent named field in clone yet).
     */
    if (data_size <= 0) {
        // Empty SHA256 → still write digest to w.padding so caller treats as
        // valid. (Matches asm fall-through at 0x24 → FINAL.)
        SHA256_CTX ctx;
        SHA256_Init(&ctx);
        uint8_t digest[32];
        SHA256_Final(digest, &ctx);
        std::memcpy(w.padding, digest, 32);
        uint64_t ret;
        std::memcpy(&ret, digest, 8);
        if (ret == 0) ret = 1;
        return ret;
    }

    // Output scratch — agent's port uses workerData+0x336b4 (~10K dwords).
    // We allocate on heap to keep the function reentrant; size = 0x800 flush
    // threshold + headroom for one max-sized compressed-record emit (128
    // dwords) + safety margin.
    constexpr size_t kOutCap = 0x1000;  // 4096 dwords = 16 KB scratch
    std::vector<int32_t> outBuf(kOutCap, 0);
    int32_t* out = outBuf.data();

    SHA256_CTX ctx;
    SHA256_Init(&ctx);

    const int32_t   ebp       = data_size - 4;     // tail-filter cap
    const uint32_t* sa_prelim = w.sa_prelim;

    int32_t r14 = 0;   // input record cursor over sa_prelim[]
    int32_t r15 = 0;   // output cursor (dwords) into out[]

    while (r14 < data_size) {
        // ---- FLUSH: when buffer holds >=2048 dwords, hash 8 KiB blocks. ----
        if (r15 >= 0x800) {
            uint32_t kept        = static_cast<uint32_t>(r15) & 0x7FFu;
            uint32_t flushed_dw  = static_cast<uint32_t>(r15) - kept;
            uint32_t flush_bytes = flushed_dw * 4u;
            if (flush_bytes > 0) {
                dbg_sha_update(&ctx, out, flush_bytes);
                std::memmove(out, out + flushed_dw, kept * 4u);
            }
            r15 = static_cast<int32_t>(kept);
        }

        uint32_t rec = sa_prelim[r14];

        // Prefetch +512 records ahead while in literal stream.
        if (rec <= 0x1FFFFu && r14 + 0x200 < data_size) {
            __builtin_prefetch(sa_prelim + r14 + 0x80);
        }

        if (rec == 0xFFFFFFFFu) break;             // terminator sentinel

        // ---- LITERAL run ----
        if (rec <= 0x1FFFFu) {
            while (r14 < data_size) {
                uint32_t v = sa_prelim[r14];
                if (v > 0x1FFFFu) break;
                if (r15 >= static_cast<int32_t>(kOutCap)) break; // safety
                out[r15++] = static_cast<int32_t>(v);
                ++r14;
                ++g_emit_counts.dc_literals;
            }
            continue;
        }

        // ---- COMPRESSED record ----
        ++g_emit_counts.dc_compressed_records;
        // stamp_id = (rec >> 17) - 1, range [0, 0x22a)
        uint32_t stamp_id = (rec >> 17) - 1u;
        if (stamp_id >= 0x22au) {
            // TODO: agent's port aborts (asm 0x18a-0x191 int3); we fail-soft.
            break;
        }

        // meta_idx = stamp_id_to_meta[stamp_id]  (uint8 indirection)
        uint32_t meta_idx = w.stampTemplates[stamp_id];
        if (meta_idx >= 277u) break;  // bounds against astroTemplate[277]

        // Decode header from astroTemplate[meta_idx].posData.
        // (Proprietary lib reads uint16 at offset +6 within an 8-byte
        // templateMarker — equivalent to .posData here.)
        uint16_t hdr  = w.astroTemplate[meta_idx].posData;
        uint32_t cnt  = static_cast<uint32_t>(hdr) & 0x7Fu;        // count: low 7 bits
        uint32_t hi   = static_cast<uint32_t>(hdr) & 0xFF80u;      // mid bits
        uint32_t low8 = rec & 0xFFu;                                // low byte seed

        // Variant bit from isBSlice (one bit per stamp_id).
        uint32_t variant = w.isBSlice.test(stamp_id) ? 1u : 0u;

        // pos_table offset from keySpotA (v0) or keySpotB (v1) within marker.
        // TODO: confirm this mapping. Task spec asserts keySpotA=pt_off_v0
        // (offset +2 in marker), keySpotB=pt_off_v1 (offset +4 in marker).
        uint16_t pt_off = (variant == 0)
                          ? w.astroTemplate[meta_idx].keySpotA
                          : w.astroTemplate[meta_idx].keySpotB;

        // pos_table base: stampKeys[]. Agent's port reads up to 128 bytes
        // starting at base+pt_off; clone's stampKeys is only 554 B, which
        // may be too small for stamps with pt_off near the end.
        // TODO: verify stampKeys is large enough or repoint to a larger
        // table once the meta-table generator is reverse-engineered.
        const uint8_t* pt = w.stampKeys + pt_off;

        // base = low8 + hi*2 (combines stamp position & low-byte seed)
        uint32_t base = low8 + (hi * 2u);

        // ---- Emit cnt positions, in blocks of 32 dwords (AVX2). ----
        // Each output dword = (pt[i] << 8) + base. Up to 4 blocks (128 dwords).
        // Force flush before emit if at risk of overflowing scratch.
        if (r15 + 128 > static_cast<int32_t>(kOutCap)) {
            uint32_t kept        = static_cast<uint32_t>(r15) & 0x7FFu;
            uint32_t flushed_dw  = static_cast<uint32_t>(r15) - kept;
            uint32_t flush_bytes = flushed_dw * 4u;
            if (flush_bytes > 0) {
                dbg_sha_update(&ctx, out, flush_bytes);
                std::memmove(out, out + flushed_dw, kept * 4u);
            }
            r15 = static_cast<int32_t>(kept);
        }

#if defined(__AVX2__)
        const __m256i vbase = _mm256_set1_epi32(static_cast<int>(base));
        auto emit_block = [&](int byte_off, int dw_off) {
            __m256i a = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(pt + byte_off + 0)));
            __m256i b = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(pt + byte_off + 8)));
            __m256i c = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(pt + byte_off + 16)));
            __m256i d = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(pt + byte_off + 24)));
            a = _mm256_add_epi32(_mm256_slli_epi32(a, 8), vbase);
            b = _mm256_add_epi32(_mm256_slli_epi32(b, 8), vbase);
            c = _mm256_add_epi32(_mm256_slli_epi32(c, 8), vbase);
            d = _mm256_add_epi32(_mm256_slli_epi32(d, 8), vbase);
            _mm256_storeu_si256((__m256i*)(out + r15 + dw_off + 0),  a);
            _mm256_storeu_si256((__m256i*)(out + r15 + dw_off + 8),  b);
            _mm256_storeu_si256((__m256i*)(out + r15 + dw_off + 16), c);
            _mm256_storeu_si256((__m256i*)(out + r15 + dw_off + 24), d);
        };
        emit_block(0x00, 0);
        if (cnt >= 0x20u) emit_block(0x20, 32);
        if (cnt >= 0x40u) emit_block(0x40, 64);
        if (cnt >= 0x60u && (~static_cast<uint32_t>(hdr) & 0x60u) == 0u) {
            emit_block(0x60, 96);
        }
#else
        // Scalar fallback (no AVX2).
        const int blocks = (cnt >= 0x60u && (~static_cast<uint32_t>(hdr) & 0x60u) == 0u) ? 4
                         : (cnt >= 0x40u) ? 3
                         : (cnt >= 0x20u) ? 2
                         : 1;
        for (int blk = 0; blk < blocks; ++blk) {
            for (int i = 0; i < 32; ++i) {
                out[r15 + blk*32 + i] =
                    static_cast<int32_t>((static_cast<uint32_t>(pt[blk*32 + i]) << 8) + base);
            }
        }
#endif

        // ---- Cursor advance + tail filter ----
        // EXPERIMENT (2026-04-30): try data_size - 4 as cap (A/B test).
        // A5 mapped +0x79360 to templateCount but with templateIdx ~70 the
        // tail-filter fires for ~8% of records. With cap=data_size-4 (~70k)
        // tail-filter never fires. Test which gives closer match to libsais.
        const int32_t cap = ebp;   // = data_size - 4
        if (cap > static_cast<int32_t>(meta_idx)) {
            r15 += static_cast<int32_t>(cnt);
            g_emit_counts.dc_expanded_positions += cnt;
            r14 += 1;
            continue;
        }

        // Tail-filter path (near end of stream): keep only positions < ebp.
        if (cnt == 0u) { ++r14; continue; }
        if (cnt == 1u) {
            if (hdr & 1u) {
                int32_t v = out[r15];
                if (v < ebp) { ++r15; ++g_emit_counts.dc_expanded_positions; }
            }
            ++r14; continue;
        }
        // cnt > 1, pair-walk variant: keep entries < ebp.
        // TODO: agent flagged MED confidence on the exact pair-stride (asm
        // 0x5b8-0x624). Verify against ground truth.
        uint32_t remaining = cnt - (static_cast<uint32_t>(hdr) & 1u);
        int32_t  src = r15;
        int32_t  dst = r15;
        while (remaining >= 2u) {
            int32_t v0 = out[src];
            if (v0 < ebp) { out[dst++] = v0; ++g_emit_counts.dc_expanded_positions; }
            int32_t v1 = out[src + 1];
            if (v1 < ebp) { out[dst++] = v1; ++g_emit_counts.dc_expanded_positions; }
            src += 2;
            remaining -= 2u;
        }
        r15 = dst;
        ++r14;
    }

    // ---- Final flush + finalize ----
    if (r15 > 0) {
        dbg_sha_update(&ctx, out, static_cast<size_t>(r15) * 4u);
    }
    uint8_t digest[32];
    SHA256_Final(digest, &ctx);

    // Write digest to w.padding so dluna_hash picks it up as the hash output.
    std::memcpy(w.padding, digest, 32);

    // Return low 64 bits of digest as the function result. Caller convention:
    // 0 = no result (fall through to libsais). Force non-zero in the rare
    // case the digest's low 8 bytes are exactly 0.
    uint64_t ret;
    std::memcpy(&ret, digest, 8);
    if (ret == 0) ret = 1;
    return ret;
}

}  // namespace deroluna_tritonn

#endif  /* x86-only Tritonn verify path */
