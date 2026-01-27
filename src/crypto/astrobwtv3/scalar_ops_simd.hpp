/**
 * scalar_ops_simd.hpp - SIMD implementations for scalar fallback branch operations
 *
 * This file contains AVX2 vectorized versions of the 13 scalar operations that
 * previously used byte-by-byte loops with #pragma GCC unroll.
 *
 * Operations successfully vectorized (9/13):
 *   - op25:  XOR popcnt, rol3, rolv, sub(x^97) - with zero handling
 *   - op38:  srlv(x&3), rol3, XOR popcnt, rolv - with zero handling
 *   - op44:  XOR popcnt x2, rol3, rolv - with zero handling
 *   - op51:  XOR p2, XOR rol4 x2, rol5 - with zero handling
 *   - op53:  mul2, XOR popcnt, XOR rol4 x2 - fully vectorized
 *   - op55:  reverse bits, XOR rol4 x2, rol1 - with zero handling
 *   - op188: XOR rol4(prev), XOR popcnt, XOR rol4 x2 - fully vectorized
 *   - op208: mul2, add, srlv(x&3), rol3 - fully vectorized
 *   - op249: reverse bits, XOR rol4 x2, rolv - fully vectorized
 *
 * Operations that CANNOT be vectorized (4/13):
 *   - op96:  Uses runtime lookup tables (simpleLookup, unchangedBytes)
 *   - op128: Uses runtime lookup tables (simpleLookup, unchangedBytes)
 *   - op253: XXHash64 computed per-iteration (data dependency)
 *   - op255: XXHash64 computed per-iteration (data dependency)
 *
 * Performance notes:
 *   - Operations with zero-byte handling use vpcmpeqb + vblendvb for conditionals
 *   - Popcount uses PSHUFB-based parallel byte popcount (popcnt256_epi8)
 *   - Bit reversal uses shift-mask-or technique (_mm256_reverse_epi8)
 *   - Variable rotation uses the custom _mm256_rolv_epi8 implementation
 */

#pragma once

#if defined(__AVX2__)

#include <immintrin.h>
#include "simd_util.hpp"

// Forward declare workerData to avoid circular includes
class workerData;

namespace astro_branched_simd {

// Constants
alignas(32) static const __m256i vec_3 = _mm256_set1_epi8(3);
alignas(32) static const __m256i vec_97 = _mm256_set1_epi8(97);
alignas(32) static const __m256i vec_zero = _mm256_setzero_si256();
alignas(32) static const __m256i vec_0x9F = _mm256_set1_epi8((char)(0x00 - 0x61));

// ============================================================================
// op25 - SIMD version
// Scalar pattern:
//   if (prev[i] == 0) { chunk[i] = 0x00 - 0x61; continue; }
//   chunk[i] = prev[i] ^ popcnt(prev[i])
//   chunk[i] = rol(chunk[i], 3)
//   chunk[i] = rol(chunk[i], chunk[i])
//   chunk[i] -= (chunk[i] ^ 97)
//
// Vectorization strategy:
//   - Use vpcmpeqb to detect zero bytes
//   - Compute normal path for all bytes
//   - Use vblendvb to select zero result (0x9F) where input was zero
// ============================================================================
inline void op25_simd(workerData &worker, __m256i &data, __m256i &old) {
    // Load prev_chunk data
    __m256i prev = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);

    // Detect zero bytes for special handling
    __m256i zero_mask = _mm256_cmpeq_epi8(prev, vec_zero);

    // Normal path: XOR with popcount
    __m256i result = _mm256_xor_si256(prev, popcnt256_epi8(prev));

    // Rotate left by 3
    result = _mm256_rol_epi8(result, 3);

    // Rotate left by variable (self)
    result = _mm256_rolv_epi8(result, result);

    // Subtract (result ^ 97)
    __m256i xor97 = _mm256_xor_si256(result, vec_97);
    result = _mm256_sub_epi8(result, xor97);

    // Blend zero case: where prev was 0, use 0x9F (0x00 - 0x61)
    result = _mm256_blendv_epi8(result, vec_0x9F, zero_mask);

    // Apply position mask and blend with old data
    __m256i mask = _mm256_loadu_si256((__m256i*)&worker.maskTable_bytes[32*(worker.pos2-worker.pos1)]);
    data = _mm256_blendv_epi8(old, result, mask);

    // Store result
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
}

// ============================================================================
// op38 - SIMD version
// Scalar pattern:
//   if (prev[i] == 0) { chunk[i] = 0; continue; }
//   chunk[i] = prev[i] >> (prev[i] & 3)
//   chunk[i] = rol(chunk[i], 3)
//   chunk[i] ^= popcnt(chunk[i])
//   chunk[i] = rol(chunk[i], chunk[i])
//
// Vectorization strategy:
//   - Use _mm256_srlv_epi8 for variable shift
//   - Blend zero result where input was zero
// ============================================================================
inline void op38_simd(workerData &worker, __m256i &data, __m256i &old) {
    __m256i prev = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);

    // Detect zero bytes
    __m256i zero_mask = _mm256_cmpeq_epi8(prev, vec_zero);

    // Shift right by (prev & 3)
    __m256i shift_count = _mm256_and_si256(prev, vec_3);
    __m256i result = _mm256_srlv_epi8(prev, shift_count);

    // Rotate left by 3
    result = _mm256_rol_epi8(result, 3);

    // XOR with popcount
    result = _mm256_xor_si256(result, popcnt256_epi8(result));

    // Rotate left by variable (self)
    result = _mm256_rolv_epi8(result, result);

    // Blend zero case
    result = _mm256_blendv_epi8(result, vec_zero, zero_mask);

    // Apply position mask
    __m256i mask = _mm256_loadu_si256((__m256i*)&worker.maskTable_bytes[32*(worker.pos2-worker.pos1)]);
    data = _mm256_blendv_epi8(old, result, mask);

    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
}

// ============================================================================
// op44 - SIMD version
// Scalar pattern:
//   if (prev[i] == 0) { chunk[i] = 0; continue; }
//   chunk[i] = prev[i] ^ popcnt(prev[i])
//   chunk[i] ^= popcnt(chunk[i])
//   chunk[i] = rol(chunk[i], 3)
//   chunk[i] = rol(chunk[i], chunk[i])
//
// Note: Double popcount XOR is not a no-op because intermediate value changes
// ============================================================================
inline void op44_simd(workerData &worker, __m256i &data, __m256i &old) {
    __m256i prev = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);

    // Detect zero bytes
    __m256i zero_mask = _mm256_cmpeq_epi8(prev, vec_zero);

    // XOR with popcount
    __m256i result = _mm256_xor_si256(prev, popcnt256_epi8(prev));

    // XOR with popcount again
    result = _mm256_xor_si256(result, popcnt256_epi8(result));

    // Rotate left by 3
    result = _mm256_rol_epi8(result, 3);

    // Rotate left by variable (self)
    result = _mm256_rolv_epi8(result, result);

    // Blend zero case
    result = _mm256_blendv_epi8(result, vec_zero, zero_mask);

    // Apply position mask
    __m256i mask = _mm256_loadu_si256((__m256i*)&worker.maskTable_bytes[32*(worker.pos2-worker.pos1)]);
    data = _mm256_blendv_epi8(old, result, mask);

    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
}

// ============================================================================
// op51 - SIMD version
// Scalar pattern:
//   if (prev[i] + chunk[pos2] == 0) { chunk[i] = 0; continue; }
//   chunk[i] = prev[i] ^ chunk[pos2]
//   chunk[i] ^= rol(chunk[i], 4)
//   chunk[i] ^= rol(chunk[i], 4)
//   chunk[i] = rol(chunk[i], 5)
//
// Vectorization strategy:
//   - Compute sum = prev + p2_val, check if zero
//   - This catches the edge case where prev[i] = -chunk[pos2] (mod 256)
// ============================================================================
inline void op51_simd(workerData &worker, __m256i &data, __m256i &old) {
    __m256i prev = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i p2_val = _mm256_set1_epi8(worker.chunk[worker.pos2]);

    // Detect condition: prev[i] + chunk[pos2] == 0
    __m256i sum = _mm256_add_epi8(prev, p2_val);
    __m256i zero_mask = _mm256_cmpeq_epi8(sum, vec_zero);

    // XOR with pos2 value
    __m256i result = _mm256_xor_si256(prev, p2_val);

    // XOR with rotate 4 (twice)
    result = _mm256_xor_si256(result, _mm256_rol_epi8(result, 4));
    result = _mm256_xor_si256(result, _mm256_rol_epi8(result, 4));

    // Rotate left by 5
    result = _mm256_rol_epi8(result, 5);

    // Blend zero case
    result = _mm256_blendv_epi8(result, vec_zero, zero_mask);

    // Apply position mask
    __m256i mask = _mm256_loadu_si256((__m256i*)&worker.maskTable_bytes[32*(worker.pos2-worker.pos1)]);
    data = _mm256_blendv_epi8(old, result, mask);

    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
}

// ============================================================================
// op53 - SIMD version (fully vectorizable - no special cases)
// Scalar pattern:
//   chunk[i] = prev[i] * 2
//   chunk[i] ^= popcnt(chunk[i])
//   chunk[i] ^= rol(chunk[i], 4)
//   chunk[i] ^= rol(chunk[i], 4)
//
// This is the simplest to vectorize - no conditionals
// ============================================================================
inline void op53_simd(workerData &worker, __m256i &data, __m256i &old) {
    __m256i prev = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);

    // Multiply by 2 (add to self)
    __m256i result = _mm256_add_epi8(prev, prev);

    // XOR with popcount
    result = _mm256_xor_si256(result, popcnt256_epi8(result));

    // XOR with rotate 4 (twice)
    result = _mm256_xor_si256(result, _mm256_rol_epi8(result, 4));
    result = _mm256_xor_si256(result, _mm256_rol_epi8(result, 4));

    // Apply position mask
    __m256i mask = _mm256_loadu_si256((__m256i*)&worker.maskTable_bytes[32*(worker.pos2-worker.pos1)]);
    data = _mm256_blendv_epi8(old, result, mask);

    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
}

// ============================================================================
// op55 - SIMD version
// Scalar pattern:
//   if (prev[i] == 0) { chunk[i] = 0; continue; }
//   chunk[i] = reverse8(prev[i])
//   chunk[i] ^= rol(chunk[i], 4)
//   chunk[i] ^= rol(chunk[i], 4)
//   chunk[i] = rol(chunk[i], 1)
//
// Note: reverse8(0) = 0, so the zero check is technically redundant,
// but we keep it for exact behavioral equivalence
// ============================================================================
inline void op55_simd(workerData &worker, __m256i &data, __m256i &old) {
    __m256i prev = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);

    // Detect zero bytes
    __m256i zero_mask = _mm256_cmpeq_epi8(prev, vec_zero);

    // Reverse bits
    __m256i result = _mm256_reverse_epi8(prev);

    // XOR with rotate 4 (twice)
    result = _mm256_xor_si256(result, _mm256_rol_epi8(result, 4));
    result = _mm256_xor_si256(result, _mm256_rol_epi8(result, 4));

    // Rotate left by 1
    result = _mm256_rol_epi8(result, 1);

    // Blend zero case (technically redundant since reverse8(0)=0 anyway)
    result = _mm256_blendv_epi8(result, vec_zero, zero_mask);

    // Apply position mask
    __m256i mask = _mm256_loadu_si256((__m256i*)&worker.maskTable_bytes[32*(worker.pos2-worker.pos1)]);
    data = _mm256_blendv_epi8(old, result, mask);

    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
}

// ============================================================================
// op188 - SIMD version (fully vectorizable - no special cases)
// Scalar pattern:
//   chunk[i] ^= rol(prev[i], 4)   // Note: XOR INTO chunk, using prev as source
//   chunk[i] ^= popcnt(chunk[i])
//   chunk[i] ^= rol(chunk[i], 4)
//   chunk[i] ^= rol(chunk[i], 4)
//
// Important: This op reads from chunk and prev simultaneously
// ============================================================================
inline void op188_simd(workerData &worker, __m256i &data, __m256i &old) {
    __m256i prev = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i chunk = _mm256_loadu_si256((__m256i*)&worker.chunk[worker.pos1]);

    // XOR chunk with rotate 4 of prev
    __m256i result = _mm256_xor_si256(chunk, _mm256_rol_epi8(prev, 4));

    // XOR with popcount
    result = _mm256_xor_si256(result, popcnt256_epi8(result));

    // XOR with rotate 4 (twice)
    result = _mm256_xor_si256(result, _mm256_rol_epi8(result, 4));
    result = _mm256_xor_si256(result, _mm256_rol_epi8(result, 4));

    // Apply position mask
    __m256i mask = _mm256_loadu_si256((__m256i*)&worker.maskTable_bytes[32*(worker.pos2-worker.pos1)]);
    data = _mm256_blendv_epi8(old, result, mask);

    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
}

// ============================================================================
// op208 - SIMD version (fully vectorizable - no special cases)
// Scalar pattern:
//   chunk[i] = prev[i] * 2
//   chunk[i] += chunk[i]        // This doubles again, so total = prev * 4
//   chunk[i] = chunk[i] >> (chunk[i] & 3)
//   chunk[i] = rol(chunk[i], 3)
// ============================================================================
inline void op208_simd(workerData &worker, __m256i &data, __m256i &old) {
    __m256i prev = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);

    // Multiply by 2
    __m256i result = _mm256_add_epi8(prev, prev);

    // Add to self (multiply by 2 again = *4 total)
    result = _mm256_add_epi8(result, result);

    // Shift right by (result & 3)
    __m256i shift_count = _mm256_and_si256(result, vec_3);
    result = _mm256_srlv_epi8(result, shift_count);

    // Rotate left by 3
    result = _mm256_rol_epi8(result, 3);

    // Apply position mask
    __m256i mask = _mm256_loadu_si256((__m256i*)&worker.maskTable_bytes[32*(worker.pos2-worker.pos1)]);
    data = _mm256_blendv_epi8(old, result, mask);

    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
}

// ============================================================================
// op249 - SIMD version (fully vectorizable - no special cases)
// Scalar pattern:
//   chunk[i] = reverse8(prev[i])
//   chunk[i] ^= rol(chunk[i], 4)
//   chunk[i] ^= rol(chunk[i], 4)
//   chunk[i] = rol(chunk[i], chunk[i])
// ============================================================================
inline void op249_simd(workerData &worker, __m256i &data, __m256i &old) {
    __m256i prev = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);

    // Reverse bits
    __m256i result = _mm256_reverse_epi8(prev);

    // XOR with rotate 4 (twice)
    result = _mm256_xor_si256(result, _mm256_rol_epi8(result, 4));
    result = _mm256_xor_si256(result, _mm256_rol_epi8(result, 4));

    // Rotate left by variable (self)
    result = _mm256_rolv_epi8(result, result);

    // Apply position mask
    __m256i mask = _mm256_loadu_si256((__m256i*)&worker.maskTable_bytes[32*(worker.pos2-worker.pos1)]);
    data = _mm256_blendv_epi8(old, result, mask);

    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
}

// ============================================================================
// NOTES ON NON-VECTORIZABLE OPERATIONS
// ============================================================================
//
// op96 & op128: These use runtime lookup tables (simpleLookup, unchangedBytes)
//   that create byte-dependent conditionals. Vectorization would require either:
//   - Precomputed 256-entry lookup tables for the transformation (expensive)
//   - AVX2 gather operations (latency overhead)
//   The current scalar implementations with unrolling are reasonably efficient.
//
// op253: CANNOT be vectorized due to loop-carried XXHash64 dependency.
//   Each iteration:
//     1. Modifies chunk[i] with rotation/XOR transforms
//     2. Computes XXHash64 over the ENTIRE chunk buffer
//     3. The hash result affects prev_lhash for next iteration
//   The hash computation sees the INTERMEDIATE state after each byte change,
//   so vectorizing the transforms would produce INCORRECT results.
//
// op255: Already vectorized in branched_AVX2.h (uses p0, r3, x0_r2, r3)
// ============================================================================

} // namespace astro_branched_simd

#endif // __AVX2__
