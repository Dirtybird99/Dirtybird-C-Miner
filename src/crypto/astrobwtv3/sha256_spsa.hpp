#pragma once
/**
 * SHA-256 with On-Demand SPSA Decompression
 *
 * This is the KEY innovation from Tritonn's SPSA algorithm:
 * > "instead of decompressing back into 70KB, just decompress in the SHA256
 * >  from the compressed buffer as soon as a compressed suffix tag is found
 * >  in the SHA scan"
 *
 * The traditional approach:
 * 1. Build compressed SA with stamp references (~10-20KB)
 * 2. Decompress to full 70KB x 4 = 280KB SA
 * 3. Feed 280KB to SHA-256
 *
 * SPSA approach:
 * 1. Build compressed SA with stamp references (~10-20KB)
 * 2. SHA-256 processes 64-byte blocks, decompressing on-the-fly
 * 3. Never allocate the 280KB expansion buffer
 *
 * Performance benefit:
 * - Eliminates ~280KB memory write (major bottleneck)
 * - Sorting is "ridiculously fast" - memory movement was the issue
 * - SHA-NI processes 64 bytes at a time, perfect for streaming
 */

#include <cstdint>
#include <vector>
#include "spsa_state.hpp"

namespace sha256_spsa {

// SHA-256 initial hash values
static const uint32_t H[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
};

// SHA-256 round constants
alignas(16) static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

/**
 * Stamp storage for efficient lookups during decompression
 *
 * Memory layout: stamps are stored contiguously for cache efficiency.
 * Maximum 277 stamps * 256 bytes = ~71KB total stamp storage.
 */
class StampStorage {
public:
    static constexpr size_t MAX_STAMPS = 277;
    static constexpr size_t STAMP_SIZE = 256;

    StampStorage() : count_(0) {
        data_.resize(MAX_STAMPS * STAMP_SIZE);
    }

    // Add a new stamp and return its ID
    uint16_t push(const uint8_t* stamp_data) {
        uint16_t id = count_;
        if (id < MAX_STAMPS) {
            memcpy(&data_[id * STAMP_SIZE], stamp_data, STAMP_SIZE);
            count_++;
        }
        return id;
    }

    // Get a byte from stamp at (stamp_id, offset within stamp)
    uint8_t get_byte(uint16_t stamp_id, uint8_t offset) const {
        return data_[stamp_id * STAMP_SIZE + offset];
    }

    // Get 4 bytes (one i32 SA entry) from a stamp
    int32_t get_i32(uint16_t stamp_id, uint8_t offset) const {
        const uint8_t* ptr = &data_[stamp_id * STAMP_SIZE + offset];
        int32_t result;
        memcpy(&result, ptr, sizeof(int32_t));
        return result;
    }

    size_t len() const { return count_; }
    bool is_empty() const { return count_ == 0; }
    void clear() { count_ = 0; }

private:
    std::vector<uint8_t> data_;
    uint16_t count_;
};

/**
 * Compressed SA entry
 *
 * Can be either:
 * - Direct position (for suffixes not part of any stamp pattern)
 * - Stamp reference (for suffixes that start within a stamp)
 */
struct SaEntry {
    enum class Type {
        Direct,
        StampRef
    };

    Type type;
    int32_t direct_pos;      // For Direct: the position
    uint16_t stamp_id;       // For StampRef: which stamp
    uint8_t rel_pos;         // For StampRef: position within stamp
    int32_t original_pos;    // For StampRef: original SA value for hashing

    static SaEntry direct(int32_t pos) {
        SaEntry e;
        e.type = Type::Direct;
        e.direct_pos = pos;
        return e;
    }

    static SaEntry stamp_ref(uint16_t stamp_id, uint8_t rel_pos, int32_t original_pos) {
        SaEntry e;
        e.type = Type::StampRef;
        e.stamp_id = stamp_id;
        e.rel_pos = rel_pos;
        e.original_pos = original_pos;
        return e;
    }

    int32_t original_value() const {
        return (type == Type::Direct) ? direct_pos : original_pos;
    }

    bool is_stamp_ref() const {
        return type == Type::StampRef;
    }
};

/**
 * SHA-256 that reads from compressed SA with on-demand decompression
 *
 * This is the core SPSA innovation: instead of expanding the compressed SA
 * to 280KB and then hashing, we decompress bytes on-the-fly as SHA-256
 * needs them.
 *
 * @param compressed_sa The compressed suffix array entries
 * @param stamps The stamp storage for decompressing stamp references
 * @param sa_len Number of entries in the suffix array
 * @param output Output buffer for 32-byte hash
 */
void sha256_compressed(
    const std::vector<SaEntry>& compressed_sa,
    const StampStorage& stamps,
    size_t sa_len,
    uint8_t* output
);

/**
 * SHA-256 that reads from reduced SA (encoded stamp references)
 *
 * Uses the spsa::SpsaState's reduced_sa format where entries are either:
 * - Direct positions (high bit clear)
 * - Stamp references (high bit set)
 *
 * @param reduced_sa The reduced suffix array from SpsaState::merge_stamps()
 * @param stamps The stamp metadata for position calculations
 * @param output Output buffer for 32-byte hash
 */
void sha256_reduced_sa(
    const std::vector<uint32_t>& reduced_sa,
    const std::vector<spsa::Stamp>& stamps,
    uint8_t* output
);

/**
 * Check if SHA-NI is available on this CPU
 */
bool sha_ni_available();

/**
 * Check if AVX2 is available on this CPU
 */
bool avx2_available();

/**
 * Software fallback SHA-256 for compressed SA
 */
void sha256_compressed_soft(
    const std::vector<SaEntry>& compressed_sa,
    size_t sa_len,
    size_t total_bytes,
    uint8_t* output
);

/**
 * Software fallback SHA-256 for reduced SA
 */
void sha256_reduced_sa_soft(
    const std::vector<uint32_t>& reduced_sa,
    const std::vector<spsa::Stamp>& stamps,
    size_t total_bytes,
    uint8_t* output
);

#if defined(__x86_64__) || defined(_M_X64)
/**
 * SHA-NI accelerated SHA-256 with on-demand decompression
 */
void sha256_compressed_ni(
    const std::vector<SaEntry>& compressed_sa,
    size_t sa_len,
    size_t total_bytes,
    uint8_t* output
);

/**
 * SHA-NI accelerated SHA-256 for reduced SA
 */
void sha256_reduced_sa_ni(
    const std::vector<uint32_t>& reduced_sa,
    const std::vector<spsa::Stamp>& stamps,
    size_t total_bytes,
    uint8_t* output
);

/**
 * SHA-NI + AVX2 combined SHA-256 for reduced SA
 * Uses AVX2 SIMD for parallel decompression of stamp references,
 * combined with SHA-NI for hardware-accelerated hashing.
 * This is the fastest path on modern x86_64 CPUs.
 */
void sha256_reduced_sa_ni_avx2(
    const std::vector<uint32_t>& reduced_sa,
    const std::vector<spsa::Stamp>& stamps,
    size_t total_bytes,
    uint8_t* output
);

/**
 * AVX2-only SHA-256 for reduced SA (software SHA, SIMD decompression)
 * For CPUs with AVX2 but without SHA-NI.
 */
void sha256_reduced_sa_avx2(
    const std::vector<uint32_t>& reduced_sa,
    const std::vector<spsa::Stamp>& stamps,
    size_t total_bytes,
    uint8_t* output
);
#endif

} // namespace sha256_spsa
