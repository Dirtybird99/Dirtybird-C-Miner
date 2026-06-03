/**
 * SPSA (Stamp-based Predictive Suffix Array) Implementation
 *
 * Port of Tritonn's SPSA algorithm from Rust to C++.
 */

#include "spsa_state.hpp"
#include <array>
#include <algorithm>
#include <cstring>
#include <openssl/sha.h>
#include "dluna.h"

#include <atomic>
#include <stdexcept>
#include <iostream>

#ifdef _WIN32
  #include <winsock2.h>
#else
  #include <arpa/inet.h>
#endif

// ============================================================================
// Compiler hints
// ============================================================================
#if defined(__GNUC__) || defined(__clang__)
#define HOT_FUNCTION __attribute__((hot))
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#define RESTRICT __restrict
#else
#define HOT_FUNCTION
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#define RESTRICT
#endif

namespace spsa {

// External SHA256-NI update function from sha256_override.cpp
// SHA256-NI handled by sha256_ni.cpp via --wrap linker flags

// ============================================================================
// SpsaState Implementation
// ============================================================================

void SpsaState::finalize_current_stamp() {
}

static ALWAYS_INLINE uint64_t extract_sort_key_8bytes(size_t pos, const uint8_t* RESTRICT data, size_t data_size) {
    const size_t remaining = data_size - pos;
    if (LIKELY(remaining >= 8)) {
        uint64_t val;
        memcpy(&val, data + pos, 8);
        return __builtin_bswap64(val);
    }
    uint64_t key = 0;
    for (size_t i = 0; i < remaining; i++) key = (key << 8) | data[pos + i];
    key <<= (8 - remaining) * 8;
    return key;
}

/* Suffix comparator used for tiebreaks: when two entries share the same
 * 8-byte prefix key, libsais's full-suffix lex order requires comparing
 * bytes 8, 9, 10, ... of the suffix. Without this, the SPSA bucket walk
 * emits positions in a different order than a real SA, producing wrong
 * SHA256 input. Bug observed 2026-04-25 with long zero-runs in sData. */
/* 2026-04-29: thread_local — was file-scope shared, racing across mining
 * threads, causing compare_suffixes to dereference another thread's data
 * pointer mid-update. SIGSEGV at any t>1. */
static thread_local const uint8_t* g_suffix_data = nullptr;
static thread_local size_t g_suffix_size = 0;

static int compare_suffixes(uint32_t a_pos, uint32_t b_pos) {
    if (a_pos == b_pos) return 0;
    size_t la = g_suffix_size - a_pos;
    size_t lb = g_suffix_size - b_pos;
    size_t minlen = la < lb ? la : lb;
    int c = std::memcmp(g_suffix_data + a_pos, g_suffix_data + b_pos, minlen);
    if (c != 0) return c;
    return la < lb ? -1 : (la > lb ? 1 : 0);
}

static HOT_FUNCTION void radix_sort_entries(SpsaState::SortEntry* RESTRICT arr, size_t len, std::vector<SpsaState::SortEntry>& temp) {
    if (UNLIKELY(len < 2)) return;
    const size_t n = len;
    if (n < 32) {
        for (size_t i = 1; i < n; i++) {
            const SpsaState::SortEntry key_entry = arr[i];
            size_t j = i;
            while (j > 0 && (arr[j-1].key > key_entry.key || (arr[j-1].key == key_entry.key && arr[j-1].pos > key_entry.pos))) {
                arr[j] = arr[j-1];
                j--;
            }
            arr[j] = key_entry;
        }
        return;
    }
    temp.resize(n);
    SpsaState::SortEntry* RESTRICT src = arr;
    SpsaState::SortEntry* RESTRICT dst = temp.data();
    // 3-pass radix, ping-pong between src/dst (eliminates 2 memcpy)
    for (int byte_idx = 3; byte_idx >= 1; byte_idx--) {
        uint32_t count[256] = {0};
        const int shift = (7 - byte_idx) * 8;
        for (size_t i = 0; i < n; i++) count[(src[i].key >> shift) & 0xFF]++;
        uint32_t total = 0;
        for (int i = 0; i < 256; i++) { uint32_t old = count[i]; count[i] = total; total += old; }
        for (size_t i = 0; i < n; i++) dst[count[(src[i].key >> shift) & 0xFF]++] = src[i];
        SpsaState::SortEntry* RESTRICT t = src; src = dst; dst = t;
    }
    // After 3 passes (odd), result is in temp. Copy back to arr.
    if (src != arr) memcpy(arr, src, n * sizeof(SpsaState::SortEntry));
    for (size_t i = 0; i < n;) {
        size_t j = i + 1;
        while (j < n && (arr[j].key >> 32) == (arr[i].key >> 32)) j++;
        if (j - i > 1) {
            std::sort(arr + i, arr + j, [](const SpsaState::SortEntry& a, const SpsaState::SortEntry& b) {
                if (a.key != b.key) return a.key < b.key;
                /* Tiebreak by full suffix lex order (matches libsais). */
                return compare_suffixes(a.pos, b.pos) < 0;
            });
        }
        i = j;
    }
}

void SpsaState::merge_stamps(const uint8_t* data, size_t data_size) {
    if (stamps.empty()) return;
    g_suffix_data = data;
    g_suffix_size = data_size;
    std::memset(bucket_counts, 0, sizeof(bucket_counts));

    /* 2026-04-29: per-chunk-per-rel emit. Original (stamp_id, rel) encoding
     * collapsed all chunks of a stamp into one entry, but each chunk has
     * different bytes at the same rel — bucketing/sorting only saw the first
     * chunk. Now emit one entry per (chunk_idx, rel) with raw global pos. */
    for (size_t stamp_id = 0; stamp_id < stamps.size(); stamp_id++) {
        const Stamp& stamp = stamps[stamp_id];
        if (stamp.chunk_count == 0) continue;
        for (size_t c = 0; c < stamp.chunk_count; c++) {
            size_t c_start = (static_cast<size_t>(stamp.start_chunk) + c) * 256;
            for (size_t rel = 0; rel < 255; rel++) {
                if (rel >= stamp.pos1 && rel <= stamp.pos2) continue;
                size_t gp = c_start + rel;
                if (gp >= data_size) break;
                bucket_counts[data[gp]]++;
            }
        }
    }
    for (size_t i = 0; i < modified_bytes_count; i++) {
        if (modified_bytes[i].global_pos < data_size) bucket_counts[data[modified_bytes[i].global_pos]]++;
    }

    size_t total_entries = 0;
    for (int b = 0; b < 256; b++) total_entries += bucket_counts[b];

    if (total_entries > all_entries.size()) {
        all_entries.resize(total_entries + 1000);
    }
    uint32_t* RESTRICT entries_buf = all_entries.data();

    uint32_t offset = 0;
    for (int b = 0; b < 256; b++) {
        bucket_offsets[b] = offset;
        offset += bucket_counts[b];
    }
    alignas(64) thread_local uint32_t scatter_idx[256];
    std::memcpy(scatter_idx, bucket_offsets, sizeof(scatter_idx));

    for (size_t stamp_id = 0; stamp_id < stamps.size(); stamp_id++) {
        const Stamp& stamp = stamps[stamp_id];
        if (stamp.chunk_count == 0) continue;
        for (size_t c = 0; c < stamp.chunk_count; c++) {
            size_t c_start = (static_cast<size_t>(stamp.start_chunk) + c) * 256;
            for (size_t rel = 0; rel < 255; rel++) {
                if (rel >= stamp.pos1 && rel <= stamp.pos2) continue;
                size_t gp = c_start + rel;
                if (gp >= data_size) break;
                uint32_t b = data[gp];
                /* Encode raw global pos (no MERGE_STAMP_MARKER). Decoded by
                 * the else-branch in the SHA256 emit loop via `entry & 0x7FFFFFFF`. */
                entries_buf[scatter_idx[b]++] = static_cast<uint32_t>(gp);
            }
        }
    }
    for (size_t i = 0; i < modified_bytes_count; i++) {
        if (modified_bytes[i].global_pos < data_size) {
            uint8_t b = data[modified_bytes[i].global_pos];
            entries_buf[scatter_idx[b]++] = modified_bytes[i].global_pos;
        }
    }
    const Stamp* RESTRICT s_ptr = stamps.data();
    for (int b = 0; b < 256; b++) {
        uint32_t b_start = bucket_offsets[b], b_size = bucket_counts[b];
        if (b_size > 1) {
            sort_buffer.resize(b_size);
            SortEntry* RESTRICT ptr = sort_buffer.data();
            for (size_t i = 0; i < b_size; i++) {
                uint32_t e = entries_buf[b_start + i];
                size_t pos;
                if (is_position_level_entry(e)) pos = (static_cast<size_t>(s_ptr[decode_merge_stamp_id(e)].start_chunk) << 8) + decode_merge_relative_pos(e);
                else pos = static_cast<size_t>(e & 0x7FFFFFFF);
                ptr[i] = {extract_sort_key_8bytes(pos, data, data_size), static_cast<uint32_t>(pos), e};
            }
            radix_sort_entries(ptr, b_size, radix_temp);
            for (size_t i = 0; i < b_size; i++) entries_buf[b_start + i] = ptr[i].encoded;
        }
    }
}

/* Diagnostic SA dump (2026-04-29 — used for divergence-localization). Keep
 * the API; production hot path checks g_dbg_capture which is normally false. */
namespace { thread_local int32_t g_dbg_spsa_sa[71000]; thread_local int g_dbg_spsa_count = 0; thread_local bool g_dbg_capture = false; }
extern "C" int32_t* dbg_spsa_get_buf(int* out_count) { *out_count = g_dbg_spsa_count; return g_dbg_spsa_sa; }
extern "C" void dbg_spsa_set_capture(bool on) { if (on) g_dbg_spsa_count = 0; g_dbg_capture = on; }

bool SPSA_Integrated(const uint8_t* data, int data_size, ::workerData &ctx, uint8_t* output) {
    if (UNLIKELY(data_size > 71000)) {
        printf("SPSA_Integrated failed: data_size > 71000 (%d)\n", data_size);
        return false;
    }
    static thread_local SpsaState* tl_state = nullptr;
    if (UNLIKELY(tl_state == nullptr)) tl_state = new SpsaState();

        tl_state->reset();
        size_t processed_until = 0;
        for (int i = 0; i < ctx.templateIdx; i++) {
            const auto& marker = ctx.astroTemplate[i];
            uint16_t first = marker.posData >> 7, count = marker.posData & 0x7F;
            tl_state->start_stamp(marker.p1, marker.p2);
            for (uint16_t c = 0; c < count; c++) {
                size_t chunk_idx = first + c, start = chunk_idx * 256;
                if (LIKELY(start + 256 <= static_cast<size_t>(data_size))) {
                    tl_state->add_chunk(data + start);
                    processed_until = std::max(processed_until, start + 256);
                } else break;
            }
        }
        for (size_t p = processed_until; p < static_cast<size_t>(data_size); p++) {
            if (tl_state->modified_bytes_count < MAX_MODIFIED_BYTES) {
                tl_state->modified_bytes[tl_state->modified_bytes_count++] = {static_cast<uint32_t>(p), data[p]};
            }
        }
        tl_state->merge_stamps(data, data_size);

        SHA256_CTX sha_ctx;
        SHA256_Init(&sha_ctx);
        alignas(32) int32_t index_buffer[64];
        int buf_ptr = 0;
        const uint32_t* RESTRICT entries_buf = tl_state->all_entries.data();
        const Stamp* RESTRICT s_ptr = tl_state->stamps.data();
        const size_t s_count = tl_state->stamps.size();

        for (int b = 0; b < 256; b++) {
            const uint32_t b_start = tl_state->bucket_offsets[b], b_len = tl_state->bucket_counts[b];
            for (size_t idx = 0; idx < b_len; idx++) {
                const uint32_t entry = entries_buf[b_start + idx];
                if (is_position_level_entry(entry)) {
                    const uint16_t sid = decode_merge_stamp_id(entry), rel = decode_merge_relative_pos(entry);
                    if (UNLIKELY(sid >= s_count)) continue;
                    const Stamp& stamp = s_ptr[sid];
                    const uint16_t count = stamp.chunk_count, start = stamp.start_chunk;
                    /* NOTE 2026-04-25: emits in chunk-order which does NOT
                     * match libsais's exact lex SA. SPSA is currently
                     * disabled in dluna_hash() until inter-stamp ordering
                     * is correctly reverse-engineered. */
                    for (uint16_t c = 0; c < count; c++) {
                        int32_t pos = ((start + c) << 8) + rel;
                        if (UNLIKELY(g_dbg_capture) && g_dbg_spsa_count < 71000) g_dbg_spsa_sa[g_dbg_spsa_count++] = pos;
                        index_buffer[buf_ptr++] = pos;  /* native LE */
                        if (buf_ptr == 64) {
                            SHA256_Update(&sha_ctx, index_buffer, 256);
                            buf_ptr = 0;
                        }
                    }
                } else {
                    int32_t pos = static_cast<int32_t>(entry & 0x7FFFFFFF);
                    if (g_dbg_capture && g_dbg_spsa_count < 71000) g_dbg_spsa_sa[g_dbg_spsa_count++] = pos;
                    index_buffer[buf_ptr++] = pos;  /* native LE */
                    if (buf_ptr == 64) {
                        SHA256_Update(&sha_ctx, index_buffer, 256);
                        buf_ptr = 0;
                    }
                }
            }
        }
    if (buf_ptr > 0) SHA256_Update(&sha_ctx, index_buffer, buf_ptr * 4);
    SHA256_Final(output, &sha_ctx);
    return true;
}

int32_t merge_entry_to_global_pos(uint32_t entry, const std::vector<Stamp>& stamps) {
    return merge_entry_to_global_pos_fast(entry, stamps.data());
}

} // namespace spsa
