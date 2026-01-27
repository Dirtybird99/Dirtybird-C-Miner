#pragma once
/**
 * TNN-style wolfPermute Batch Branch Processing for AstroBWTv3
 *
 * Port of Tritonn's branch batch processing from Rust to C++.
 *
 * CRITICAL OPTIMIZATION: This module processes ALL bytes in the range [pos1, pos2)
 * using AVX2 SIMD instead of just the single byte at pos2.
 *
 * The key insight from TNN's wolfPermute is:
 * 1. Load 32 bytes starting at pos1 using _mm256_loadu_si256
 * 2. Apply all 4 branch operations to the ENTIRE 32-byte vector
 * 3. Use a blend mask to only update bytes in range [pos1, pos2)
 * 4. Store the result with _mm256_storeu_si256
 *
 * This achieves massive parallelism: instead of processing 1 byte at a time,
 * we process up to 32 bytes in a single SIMD operation.
 */

#include <cstdint>
#include <cstring>

namespace branch_tables {

// ============================================================================
// CODE_LUT: Encodes 4 operations in 32 bits per byte value
// Format: [op4:8][op3:8][op2:8][op1:8]
// ============================================================================

extern const uint32_t CODE_LUT[257];

// ============================================================================
// CODE_LUT_16: Compressed 16-bit version (4 ops as nibbles)
// Format: [op4:4][op3:4][op2:4][op1:4]
// ============================================================================

extern const uint16_t CODE_LUT_16[257];

// ============================================================================
// Scalar Fallback Implementation
// ============================================================================

/**
 * Scalar implementation for single byte transformation.
 * Matches the TNN wolfBranch operation exactly.
 *
 * @param val The input byte value
 * @param pos2val The value at position pos2 (used for xor/and operations)
 * @param opcode The 32-bit opcode from CODE_LUT
 * @return The transformed byte value
 */
uint8_t wolf_branch_scalar(uint8_t val, uint8_t pos2val, uint32_t opcode);

/**
 * Scalar wolfPermute for systems without AVX2.
 * Processes all bytes in range [pos1, pos2).
 *
 * @param input Input data buffer
 * @param output Output data buffer
 * @param op Operation index (0-255) into CODE_LUT
 * @param pos1 Start position (inclusive)
 * @param pos2 End position (exclusive)
 */
void wolf_permute_scalar(
    const uint8_t* input,
    uint8_t* output,
    uint8_t op,
    uint8_t pos1,
    uint8_t pos2
);

/**
 * Scalar in-place permute.
 */
inline void wolf_permute_scalar_inplace(
    uint8_t* data,
    uint8_t op,
    uint8_t pos1,
    uint8_t pos2
) {
    wolf_permute_scalar(data, data, op, pos1, pos2);
}

// ============================================================================
// High-Level API
// ============================================================================

/**
 * High-level batch branch processing that can replace single-byte lookup.
 *
 * This function processes ALL bytes in the range [pos1, pos2) instead of
 * just the byte at pos2. This matches TNN's optimization.
 *
 * @param data The 256-byte data buffer (modified in place)
 * @param branch_byte The branch operation selector (0-255)
 * @param pos1 Start position (inclusive)
 * @param pos2 End position (exclusive for processing)
 * @return The value at data[pos2-1] after processing
 */
uint8_t apply_branch_batch(uint8_t data[256], uint8_t branch_byte, uint8_t pos1, uint8_t pos2);

/**
 * Process a batch of 8 streams with wolfPermute.
 * This is the main entry point for 8-way batch processing.
 *
 * @param data Array of 8 data buffers (256 bytes each)
 * @param branch_bytes Branch operation for each stream
 * @param pos1_vals Start position for each stream
 * @param pos2_vals End position for each stream
 * @param done_mask Bitmask of streams to skip (bit set = skip)
 */
void apply_branch_batch_8(
    uint8_t data[8][256],
    const uint8_t branch_bytes[8],
    const uint8_t pos1_vals[8],
    const uint8_t pos2_vals[8],
    uint8_t done_mask
);

// ============================================================================
// AVX2 Implementation (x86-64 only)
// ============================================================================

#if defined(__x86_64__) || defined(_M_X64)

/// Check if AVX2 is available on this CPU
bool avx2_available();

/**
 * Generate a blend mask for [0, bytes) range using AVX2.
 * Only 2 AVX2 instructions: set1 + cmpgt
 *
 * Returns a mask where bytes [0, bytes) are 0xFF and [bytes, 32) are 0x00
 *
 * @param bytes Number of bytes to mask (0-32)
 * @param mask Output buffer (32 bytes, 64-byte aligned preferred)
 */
void gen_mask_avx2(uint8_t bytes, uint8_t mask[32]);

/**
 * TNN-style wolfPermute: Process ALL bytes in range [pos1, pos2) with AVX2.
 *
 * This is the CRITICAL optimization that processes the entire affected range
 * instead of just the single byte at pos2.
 *
 * @param input Input data buffer (at least pos1 + 32 bytes accessible)
 * @param output Output data buffer (at least pos1 + 32 bytes accessible)
 * @param op Operation index (0-255) into CODE_LUT
 * @param pos1 Start position (inclusive)
 * @param pos2 End position (exclusive for processing, but input[pos2] is used as pos2_val)
 *
 * Requirements:
 * - Requires AVX2 support
 * - input and output must have at least pos1 + 32 accessible bytes
 * - pos2 > pos1 must be true
 * - pos2 - pos1 <= 32 must be true
 */
void wolf_permute_avx2(
    const uint8_t* input,
    uint8_t* output,
    uint8_t op,
    uint8_t pos1,
    uint8_t pos2
);

/**
 * In-place version of wolfPermute.
 */
inline void wolf_permute_avx2_inplace(
    uint8_t* data,
    uint8_t op,
    uint8_t pos1,
    uint8_t pos2
) {
    wolf_permute_avx2(data, data, op, pos1, pos2);
}

#endif  // x86_64

} // namespace branch_tables
