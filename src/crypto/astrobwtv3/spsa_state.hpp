#pragma once
/**
 * SPSA (Stamp-based Predictive Suffix Array)
 *
 * Port of Tritonn's SPSA algorithm from Rust to C++.
 *
 * Key insight: Consecutive 256-byte chunks ("stamps") in the branchy loop
 * typically differ by only 1-2 bytes (~90% of iterations).
 *
 * Instead of building the full 70KB SA every time, we:
 * 1. Group consecutive chunks with same pos1/pos2 values into "stamps"
 * 2. Build mini suffix arrays for each stamp
 * 3. Merge stamp SAs into the final SA
 * 4. Hash directly from compressed representation (on-demand decompression)
 */

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <vector>
#include <algorithm>

namespace spsa {

// Constants
constexpr size_t MAX_STAMPS = 277;
constexpr size_t MAX_CHUNKS_PER_STAMP = 32;
constexpr size_t RADSORT_THRESHOLD = 64;

// Stamp marker for merge output (indicates stamp reference vs direct position)
constexpr uint32_t MERGE_STAMP_MARKER = 0x80000000;
constexpr uint32_t MERGE_STAMP_ID_MASK = 0x01FF0000;
constexpr uint32_t MERGE_STAMP_ID_SHIFT = 16;
constexpr uint32_t MERGE_POS_MASK = 0x0000FFFF;

// Position-level flag (bit 25): indicates entry represents ALL chunks at this position
// When set, the entry expands to chunk_count suffixes during flattening
// This reduces sort complexity from O(chunks * positions) to O(stamps * positions)
constexpr uint32_t MERGE_POSITION_FLAG = 0x02000000;

/**
 * Modified byte position (goes into side array)
 *
 * When a byte is modified during stamp processing, we record its
 * global position and new value for on-demand reconstruction.
 */
struct ModifiedByte {
    uint32_t global_pos;
    uint8_t byte_value;

    ModifiedByte() : global_pos(0), byte_value(0) {}
    ModifiedByte(uint32_t pos, uint8_t val) : global_pos(pos), byte_value(val) {}
};

/**
 * Stamp metadata - tracks a group of consecutive chunks with same pos1/pos2
 *
 * In AstroBWTv3, each iteration writes 256 bytes at stage2_start + tries * 256.
 * Consecutive iterations often have the same pos1 and pos2 values.
 */
struct Stamp {
    uint16_t start_chunk;   // First chunk index in this stamp (0-276)
    uint16_t chunk_count;   // Number of chunks in this stamp
    uint8_t pos1;           // pos1 value for all chunks
    uint8_t pos2;           // pos2 value (typically 0 or 1)

    // Per-chunk data (up to MAX_CHUNKS_PER_STAMP)
    std::vector<uint8_t> pos1_bytes;  // Byte at pos1 for each chunk
    std::vector<uint8_t> byte255;     // Byte at index 255 for each chunk

    Stamp() : start_chunk(0), chunk_count(0), pos1(0), pos2(0) {
        pos1_bytes.reserve(MAX_CHUNKS_PER_STAMP);
        byte255.reserve(MAX_CHUNKS_PER_STAMP);
    }

    // Start a new stamp group
    void begin(uint16_t chunk_idx, uint8_t p1, uint8_t p2) {
        start_chunk = chunk_idx;
        chunk_count = 0;
        pos1 = p1;
        pos2 = p2;
        pos1_bytes.clear();
        byte255.clear();
    }

    // Add a chunk to this stamp group
    void add_chunk(uint8_t pos1_byte, uint8_t byte_255) {
        chunk_count++;
        pos1_bytes.push_back(pos1_byte);
        byte255.push_back(byte_255);
    }

    bool is_empty() const { return chunk_count == 0; }
    uint16_t end_chunk() const { return start_chunk + chunk_count; }

    void clear() {
        start_chunk = 0;
        chunk_count = 0;
        pos1 = 0;
        pos2 = 0;
        pos1_bytes.clear();
        byte255.clear();
    }
};

/**
 * SPSA working state
 *
 * Maintains all state needed for SPSA processing.
 * Designed to be reused across hash computations.
 */
class SpsaState {
public:
    std::vector<Stamp> stamps;
    std::vector<ModifiedByte> modified_bytes;
    std::vector<uint32_t> first_indexes;
    std::vector<std::vector<uint32_t>> stamp_sas;
    std::vector<uint32_t> reduced_sa;

    SpsaState() {
        stamps.reserve(MAX_STAMPS);
        modified_bytes.reserve(MAX_STAMPS * 4);
        first_indexes.reserve(MAX_STAMPS);
        stamp_sas.reserve(MAX_STAMPS);
        reduced_sa.reserve(70000);
    }

    // Reset for new hash computation
    void reset() {
        stamps.clear();
        modified_bytes.clear();
        first_indexes.clear();
        stamp_sas.clear();
        reduced_sa.clear();
        current_stamp_idx = -1;
        current_chunk = 0;
    }

    // Start tracking a new stamp (call when pos1/pos2 changes)
    void start_stamp(uint8_t pos1, uint8_t pos2) {
        // Finalize current stamp if any
        if (current_stamp_idx >= 0) {
            finalize_current_stamp();
        }

        // Start new stamp
        stamps.emplace_back();
        current_stamp_idx = stamps.size() - 1;
        stamps[current_stamp_idx].begin(current_chunk, pos1, pos2);
    }

    // Add a chunk to current stamp
    void add_chunk(uint8_t pos1_byte, uint8_t byte_255) {
        if (current_stamp_idx >= 0) {
            stamps[current_stamp_idx].add_chunk(pos1_byte, byte_255);
            current_chunk++;
        }
    }

    // Add chunk with optional modified byte tracking
    void add_chunk_with_modified(uint8_t pos1_byte, uint8_t byte_255,
                                  uint32_t mod_pos, uint8_t mod_byte, bool has_modified) {
        add_chunk(pos1_byte, byte_255);
        if (has_modified) {
            modified_bytes.emplace_back(mod_pos, mod_byte);
        }
    }

    // End current stamp
    void end_stamp() {
        finalize_current_stamp();
    }

    // Get the reduced SA (call after merge_stamps)
    const std::vector<uint32_t>& get_reduced_sa() const { return reduced_sa; }

    // Get stamps for SHA-256 decompression
    const std::vector<Stamp>& get_stamps() const { return stamps; }

    // Total suffix count (for SHA-256 length calculation)
    size_t get_total_suffix_count() const {
        size_t count = 0;
        for (const auto& s : stamps) {
            count += static_cast<size_t>(s.chunk_count) * 256;
        }
        return count + modified_bytes.size();
    }

    // Build mini suffix arrays for all stamps
    void build_all_mini_sas(const uint8_t* data, size_t data_size);

    // Merge all stamp SAs into reduced_sa
    void merge_stamps(const uint8_t* data, size_t data_size);

private:
    int current_stamp_idx = -1;
    uint16_t current_chunk = 0;

    void finalize_current_stamp();
    void build_mini_sa(Stamp& stamp, std::vector<uint32_t>& mini_sa,
                       const uint8_t* data, size_t data_size);
};

// Utility: Insertion sort for small packed value arrays
// Packed format: (byte_value << 16) | index
inline void insertion_sort_packed(uint32_t* arr, size_t len) {
    for (size_t i = 1; i < len; i++) {
        uint32_t key = arr[i];
        size_t j = i;
        while (j > 0 && arr[j - 1] > key) {
            arr[j] = arr[j - 1];
            j--;
        }
        arr[j] = key;
    }
}

// Radix sort for larger arrays (O(n) complexity)
void radix_sort_u32(uint32_t* arr, size_t len);

// Check if entry is a stamp reference
inline bool is_merge_stamp_ref(uint32_t entry) {
    return (entry & MERGE_STAMP_MARKER) != 0;
}

// Check if entry is a position-level stamp reference (expands to multiple chunks)
inline bool is_position_level_entry(uint32_t entry) {
    return (entry & (MERGE_STAMP_MARKER | MERGE_POSITION_FLAG)) ==
           (MERGE_STAMP_MARKER | MERGE_POSITION_FLAG);
}

// Decode stamp ID from merge entry
inline uint16_t decode_merge_stamp_id(uint32_t entry) {
    return static_cast<uint16_t>((entry & MERGE_STAMP_ID_MASK) >> MERGE_STAMP_ID_SHIFT);
}

// Decode relative position from merge entry
inline uint16_t decode_merge_relative_pos(uint32_t entry) {
    return static_cast<uint16_t>(entry & MERGE_POS_MASK);
}

// Convert merge entry to global position (for SHA-256 hashing)
// Original version with vector reference (kept for compatibility)
int32_t merge_entry_to_global_pos(uint32_t entry, const std::vector<Stamp>& stamps);

/**
 * OPTIMIZED: Branchless merge entry to global position conversion
 *
 * Optimizations applied:
 * 1. Shift instead of multiply: << 8 vs * 256 (1 cycle vs 3 cycles)
 * 2. No bounds checking: SA construction guarantees stamp_id validity
 * 3. Branchless execution: computes both paths, uses conditional select
 * 4. Inline for zero call overhead
 * 5. Raw pointer version avoids vector bounds checking overhead
 *
 * Entry format:
 * - High bit (0x80000000) set = stamp reference
 * - Bits 24-16 (0x01FF0000) = stamp_id
 * - Bits 15-0  (0x0000FFFF) = relative position within stamp
 */
inline int32_t merge_entry_to_global_pos_fast(uint32_t entry, const Stamp* stamps_ptr) {
    // Extract fields (shift ordering optimized for decode)
    uint16_t stamp_id = (entry >> 16) & 0x01FF;
    uint16_t relative_pos = entry & 0xFFFF;

    // Stamp path: shift instead of multiply (1 cycle vs 3 cycles)
    // No bounds check - SA construction guarantees validity
    int32_t stamp_result = static_cast<int32_t>(
        (stamps_ptr[stamp_id].start_chunk << 8) + relative_pos
    );

    // Direct path: mask off high bit to get raw position
    int32_t direct_result = static_cast<int32_t>(entry & 0x7FFFFFFF);

    // Branchless select: high bit determines which result to use
    // Compiler will use CMOV instruction on x86
    bool is_stamp = (entry & 0x80000000) != 0;
    return is_stamp ? stamp_result : direct_result;
}

/**
 * OPTIMIZED: Fully branchless version using bitwise operations
 *
 * This version avoids even the ternary operator by using arithmetic masking.
 * Useful when branch misprediction is a significant concern.
 */
inline int32_t merge_entry_to_global_pos_branchless(uint32_t entry, const Stamp* stamps_ptr) {
    // Extract fields
    uint16_t stamp_id = (entry >> 16) & 0x01FF;
    uint16_t relative_pos = entry & 0xFFFF;

    // Compute both paths
    int32_t stamp_result = static_cast<int32_t>(
        (stamps_ptr[stamp_id].start_chunk << 8) + relative_pos
    );
    int32_t direct_result = static_cast<int32_t>(entry & 0x7FFFFFFF);

    // Create mask from high bit: 0xFFFFFFFF if stamp ref, 0x00000000 otherwise
    // Arithmetic right shift propagates sign bit
    int32_t mask = static_cast<int32_t>(entry) >> 31;

    // Select result: (stamp & mask) | (direct & ~mask)
    return (stamp_result & mask) | (direct_result & ~mask);
}

/**
 * Batch conversion: process 4 entries at once for better instruction pipelining
 *
 * Takes advantage of instruction-level parallelism by computing multiple
 * results simultaneously, hiding memory latency.
 */
inline void merge_entries_to_global_pos_batch4(
    const uint32_t* entries,
    const Stamp* stamps_ptr,
    int32_t* out_positions
) {
    // Load all 4 entries
    uint32_t e0 = entries[0];
    uint32_t e1 = entries[1];
    uint32_t e2 = entries[2];
    uint32_t e3 = entries[3];

    // Extract stamp IDs (parallel extraction)
    uint16_t sid0 = (e0 >> 16) & 0x01FF;
    uint16_t sid1 = (e1 >> 16) & 0x01FF;
    uint16_t sid2 = (e2 >> 16) & 0x01FF;
    uint16_t sid3 = (e3 >> 16) & 0x01FF;

    // Extract relative positions
    uint16_t rp0 = e0 & 0xFFFF;
    uint16_t rp1 = e1 & 0xFFFF;
    uint16_t rp2 = e2 & 0xFFFF;
    uint16_t rp3 = e3 & 0xFFFF;

    // Compute stamp results (shift instead of multiply)
    int32_t sr0 = (stamps_ptr[sid0].start_chunk << 8) + rp0;
    int32_t sr1 = (stamps_ptr[sid1].start_chunk << 8) + rp1;
    int32_t sr2 = (stamps_ptr[sid2].start_chunk << 8) + rp2;
    int32_t sr3 = (stamps_ptr[sid3].start_chunk << 8) + rp3;

    // Compute direct results
    int32_t dr0 = static_cast<int32_t>(e0 & 0x7FFFFFFF);
    int32_t dr1 = static_cast<int32_t>(e1 & 0x7FFFFFFF);
    int32_t dr2 = static_cast<int32_t>(e2 & 0x7FFFFFFF);
    int32_t dr3 = static_cast<int32_t>(e3 & 0x7FFFFFFF);

    // Create masks from high bits
    int32_t m0 = static_cast<int32_t>(e0) >> 31;
    int32_t m1 = static_cast<int32_t>(e1) >> 31;
    int32_t m2 = static_cast<int32_t>(e2) >> 31;
    int32_t m3 = static_cast<int32_t>(e3) >> 31;

    // Select results (branchless)
    out_positions[0] = (sr0 & m0) | (dr0 & ~m0);
    out_positions[1] = (sr1 & m1) | (dr1 & ~m1);
    out_positions[2] = (sr2 & m2) | (dr2 & ~m2);
    out_positions[3] = (sr3 & m3) | (dr3 & ~m3);
}

} // namespace spsa
