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
#include "astrobwtv3.h"
#include "astroworker.h"
#include "sha256_spsa.hpp"
#include <stdexcept>
#include <iostream>

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
extern "C" void sha256_process_block_ni(uint32_t* state0, uint32_t* state1, const uint8_t* data, const uint8_t* shuf_mask);

// ============================================================================
// SpsaState Implementation
// ============================================================================

void SpsaState::finalize_current_stamp() {
}

static ALWAYS_INLINE uint64_t extract_sort_key_8bytes(size_t pos, const uint8_t* RESTRICT data, size_t data_size) {
    const size_t remaining = data_size - pos;
    if (LIKELY(remaining >= 8)) {
        const uint8_t* RESTRICT p = data + pos;
        return (static_cast<uint64_t>(p[0]) << 56) | (static_cast<uint64_t>(p[1]) << 48) |
               (static_cast<uint64_t>(p[2]) << 40) | (static_cast<uint64_t>(p[3]) << 32) |
               (static_cast<uint64_t>(p[4]) << 24) | (static_cast<uint64_t>(p[5]) << 16) |
               (static_cast<uint64_t>(p[6]) << 8) | static_cast<uint64_t>(p[7]);
    }
    uint64_t key = 0;
    for (size_t i = 0; i < remaining; i++) key = (key << 8) | data[pos + i];
    key <<= (8 - remaining) * 8;
    return key;
}

static HOT_FUNCTION void radix_sort_entries(SpsaState::SortEntry* RESTRICT arr, size_t len, std::vector<SpsaState::SortEntry>& temp) {
    if (UNLIKELY(len < 2)) return;
    const size_t n = len;
    if (n < 32) {
        for (size_t i = 1; i < n; i++) {
            const SpsaState::SortEntry key_entry = arr[i];
            size_t j = i;
            while (j > 0 && (arr[j-1].key > key_entry.key || (arr[j-1].key == key_entry.key && arr[j-1].pos > key_entry.pos))) {
                arr[j] = arr[j-1]; j--;
            }
            arr[j] = key_entry;
        }
        return;
    }
    temp.resize(n);
    SpsaState::SortEntry* RESTRICT temp_ptr = temp.data();
    for (int byte_idx = 3; byte_idx >= 1; byte_idx--) {
        uint32_t count[256] = {0};
        const int shift = (7 - byte_idx) * 8;
        for (size_t i = 0; i < n; i++) count[(arr[i].key >> shift) & 0xFF]++;
        uint32_t total = 0;
        for (int i = 0; i < 256; i++) { uint32_t old = count[i]; count[i] = total; total += old; }
        for (size_t i = 0; i < n; i++) temp_ptr[count[(arr[i].key >> shift) & 0xFF]++] = arr[i];
        memcpy(arr, temp_ptr, n * sizeof(SpsaState::SortEntry));
    }
    for (size_t i = 0; i < n; ) {
        size_t j = i + 1;
        while (j < n && (arr[j].key >> 32) == (arr[i].key >> 32)) j++;
        if (j - i > 1) {
            std::sort(arr + i, arr + j, [](const SpsaState::SortEntry& a, const SpsaState::SortEntry& b) {
                if (a.key != b.key) return a.key < b.key;
                return a.pos < b.pos;
            });
        }
        i = j;
    }
}

void SpsaState::merge_stamps(const uint8_t* data, size_t data_size) {
    if (stamps.empty()) return;
    std::memset(bucket_counts, 0, sizeof(bucket_counts));

    for (size_t stamp_id = 0; stamp_id < stamps.size(); stamp_id++) {
        const Stamp& stamp = stamps[stamp_id];
        if (stamp.chunk_count == 0) continue;
        size_t start = static_cast<size_t>(stamp.start_chunk) * 256;
        for (size_t rel = 0; rel < 255; rel++) {
            if (rel >= stamp.pos1 && rel <= stamp.pos2) continue;
            size_t gp = start + rel; if (gp >= data_size) break;
            bucket_counts[data[gp]]++;
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
    for (int b = 0; b < 256; b++) { bucket_offsets[b] = offset; offset += bucket_counts[b]; }
    alignas(64) thread_local uint32_t scatter_idx[256];
    std::memcpy(scatter_idx, bucket_offsets, sizeof(scatter_idx));
    
    for (size_t stamp_id = 0; stamp_id < stamps.size(); stamp_id++) {
        const Stamp& stamp = stamps[stamp_id];
        if (stamp.chunk_count == 0) continue;
        size_t start = static_cast<size_t>(stamp.start_chunk) * 256;
        for (size_t rel = 0; rel < 255; rel++) {
            if (rel >= stamp.pos1 && rel <= stamp.pos2) continue;
            size_t gp = start + rel; if (gp >= data_size) break;
            uint32_t b = data[gp];
            uint32_t encoded = MERGE_STAMP_MARKER | MERGE_POSITION_FLAG | ((static_cast<uint32_t>(stamp_id) << MERGE_STAMP_ID_SHIFT) & MERGE_STAMP_ID_MASK) | (static_cast<uint32_t>(rel) & MERGE_POS_MASK);
            entries_buf[scatter_idx[b]++] = encoded;
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
            sort_buffer.resize(b_size); SortEntry* RESTRICT ptr = sort_buffer.data();
            for (size_t i = 0; i < b_size; i++) {
                uint32_t e = entries_buf[b_start + i]; size_t pos;
                if (is_position_level_entry(e)) pos = (static_cast<size_t>(s_ptr[decode_merge_stamp_id(e)].start_chunk) << 8) + decode_merge_relative_pos(e);
                else pos = static_cast<size_t>(e & 0x7FFFFFFF);
                ptr[i] = {extract_sort_key_8bytes(pos, data, data_size), static_cast<uint32_t>(pos), e};
            }
            radix_sort_entries(ptr, b_size, radix_temp);
            for (size_t i = 0; i < b_size; i++) entries_buf[b_start + i] = ptr[i].encoded;
        }
    }
}

bool SPSA_Integrated(const uint8_t* data, int data_size, ::workerData &ctx, uint8_t* output) {
    if (data_size > 71000) {
        printf("SPSA_Integrated failed: data_size > 71000 (%d)\n", data_size);
        return false;
    }
    if (ctx.templateIdx == 0) {
        printf("SPSA_Integrated failed: templateIdx is 0\n");
    }
    try {
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
                if (LIKELY(start + 256 <= (size_t)data_size)) {
                    tl_state->add_chunk(data + start);
                    processed_until = std::max(processed_until, start + 256);
                } else break;
            }
        }
        for (size_t p = processed_until; p < (size_t)data_size; p++) {
            if (tl_state->modified_bytes_count < MAX_MODIFIED_BYTES) {
                tl_state->modified_bytes[tl_state->modified_bytes_count++] = {static_cast<uint32_t>(p), data[p]};
            }
        }
        tl_state->merge_stamps(data, data_size);
        
        SHA256_CTX sha_ctx; SHA256_Init(&sha_ctx);
        alignas(32) int32_t index_buffer[64]; int buf_ptr = 0;
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
#if defined(__AVX2__)
                    __m256i v_rel = _mm256_set1_epi32(rel); uint16_t c = 0;
                    for (; c + 7 < count; c += 8) {
                        __m256i v_start = _mm256_set_epi32(c+7, c+6, c+5, c+4, c+3, c+2, c+1, c);
                        v_start = _mm256_add_epi32(v_start, _mm256_set1_epi32(start));
                        __m256i v_pos = _mm256_add_epi32(_mm256_slli_epi32(v_start, 8), v_rel);
                        
                        if (buf_ptr + 8 <= 64) { 
                            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&index_buffer[buf_ptr]), v_pos); 
                            buf_ptr += 8; 
                        } else {
                            SHA256_Update(&sha_ctx, index_buffer, buf_ptr * 4);
                            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&index_buffer[0]), v_pos);
                            buf_ptr = 8;
                        }
                        if (buf_ptr == 64) { SHA256_Update(&sha_ctx, index_buffer, 256); buf_ptr = 0; }
                    }
                    for (; c < count; c++) {
                        index_buffer[buf_ptr++] = ((start + c) << 8) + rel;
                        if (buf_ptr == 64) { SHA256_Update(&sha_ctx, index_buffer, 256); buf_ptr = 0; }
                    }
#else
                    for (uint16_t c = 0; c < count; c++) {
                        index_buffer[buf_ptr++] = ((start + c) << 8) + rel;
                        if (buf_ptr == 64) { SHA256_Update(&sha_ctx, index_buffer, 256); buf_ptr = 0; }
                    }
#endif
                } else {
                    index_buffer[buf_ptr++] = static_cast<int32_t>(entry & 0x7FFFFFFF);
                    if (buf_ptr == 64) { SHA256_Update(&sha_ctx, index_buffer, 256); buf_ptr = 0; }
                }
            }
        }
        if (buf_ptr > 0) SHA256_Update(&sha_ctx, index_buffer, buf_ptr * 4);
        SHA256_Final(output, &sha_ctx);
        return true;
    } catch (const std::exception& ex) {
        printf("SPSA_Integrated EXCEPTION: %s\n", ex.what());
        return false;
    } catch (...) {
        printf("SPSA_Integrated UNKNOWN EXCEPTION\n");
        return false;
    }
}

int32_t merge_entry_to_global_pos(uint32_t entry, const std::vector<Stamp>& stamps) {
    return merge_entry_to_global_pos_fast(entry, stamps.data());
}

} // namespace spsa
