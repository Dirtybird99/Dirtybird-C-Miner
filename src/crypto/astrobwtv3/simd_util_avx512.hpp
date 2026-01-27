#pragma once

/**
 * AVX512 SIMD Utilities for AstroBWTv3 Branch Operations
 *
 * These functions provide 64-byte (512-bit) vectorized operations
 * for the branch computation loop, doubling throughput compared to AVX2.
 *
 * Requires: AVX512F, AVX512BW, AVX512VBMI (for vpopcntb where available)
 */

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

// ============================================================================
// Population Count (popcnt per byte)
// ============================================================================

// AVX512BW version using lookup table (works on all AVX512 CPUs)
__attribute__((target("avx512f,avx512bw")))
inline __m512i popcnt512_epi8(__m512i data) {
    const __m512i mask4 = _mm512_set1_epi8(0x0F);
    // _mm512_set_epi8 uses reversed order (element 63 first, element 0 last)
    const __m512i lookup = _mm512_set_epi8(
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0
    );

    __m512i low = _mm512_and_si512(data, mask4);
    __m512i high = _mm512_and_si512(_mm512_srli_epi16(data, 4), mask4);

    __m512i pop_low = _mm512_shuffle_epi8(lookup, low);
    __m512i pop_high = _mm512_shuffle_epi8(lookup, high);

    return _mm512_add_epi8(pop_low, pop_high);
}

// ============================================================================
// Byte-wise Multiplication
// ============================================================================

__attribute__((target("avx512f,avx512bw")))
inline __m512i _mm512_mul_epi8(__m512i x, __m512i y) {
    // Unpack and isolate 2 8-bit numbers from a 16-bit block
    __m512i mask_hi = _mm512_set1_epi16((int16_t)0xFF00);
    __m512i mask_lo = _mm512_set1_epi16(0x00FF);

    // Extract high and low bytes
    __m512i x_hi = _mm512_srli_epi16(_mm512_and_si512(x, mask_hi), 8);
    __m512i y_hi = _mm512_srli_epi16(_mm512_and_si512(y, mask_hi), 8);
    __m512i x_lo = _mm512_and_si512(x, mask_lo);
    __m512i y_lo = _mm512_and_si512(y, mask_lo);

    // Perform 16-bit multiplications
    __m512i prod_hi = _mm512_slli_epi16(_mm512_mullo_epi16(x_hi, y_hi), 8);
    __m512i prod_lo = _mm512_mullo_epi16(x_lo, y_lo);

    // Mask and combine
    prod_hi = _mm512_and_si512(prod_hi, mask_hi);
    prod_lo = _mm512_and_si512(prod_lo, mask_lo);

    return _mm512_or_si512(prod_hi, prod_lo);
}

// ============================================================================
// Variable Left Shift per Byte
// ============================================================================

__attribute__((target("avx512f,avx512bw")))
inline __m512i _mm512_sllv_epi8(__m512i a, __m512i count) {
    __m512i mask_hi = _mm512_set1_epi32(0xFF00FF00);
    __m512i multiplier_lut = _mm512_set_epi8(
        0,0,0,0, 0,0,0,0, (int8_t)0x80,0x40,0x20,0x10, 0x08,0x04,0x02,0x01,
        0,0,0,0, 0,0,0,0, (int8_t)0x80,0x40,0x20,0x10, 0x08,0x04,0x02,0x01,
        0,0,0,0, 0,0,0,0, (int8_t)0x80,0x40,0x20,0x10, 0x08,0x04,0x02,0x01,
        0,0,0,0, 0,0,0,0, (int8_t)0x80,0x40,0x20,0x10, 0x08,0x04,0x02,0x01
    );

    // Clamp shift count to 0-8
    __m512i count_sat = _mm512_min_epu8(count, _mm512_set1_epi8(8));

    // Get multiplier from lookup table
    __m512i multiplier = _mm512_shuffle_epi8(multiplier_lut, count_sat);

    // Process low bytes
    __m512i x_lo = _mm512_mullo_epi16(a, multiplier);

    // Process high bytes
    __m512i multiplier_hi = _mm512_srli_epi16(multiplier, 8);
    __m512i a_hi = _mm512_and_si512(a, mask_hi);
    __m512i x_hi = _mm512_mullo_epi16(a_hi, multiplier_hi);

    // Blend results using mask
    __mmask64 blend_mask = _mm512_cmpgt_epi8_mask(mask_hi, _mm512_setzero_si512());
    return _mm512_mask_blend_epi8(blend_mask, x_lo, x_hi);
}

// ============================================================================
// Variable Right Shift per Byte
// ============================================================================

__attribute__((target("avx512f,avx512bw")))
inline __m512i _mm512_srlv_epi8(__m512i a, __m512i count) {
    __m512i mask_hi = _mm512_set1_epi32(0xFF00FF00);
    __m512i multiplier_lut = _mm512_set_epi8(
        0,0,0,0, 0,0,0,0, 0x01,0x02,0x04,0x08, 0x10,0x20,0x40,(int8_t)0x80,
        0,0,0,0, 0,0,0,0, 0x01,0x02,0x04,0x08, 0x10,0x20,0x40,(int8_t)0x80,
        0,0,0,0, 0,0,0,0, 0x01,0x02,0x04,0x08, 0x10,0x20,0x40,(int8_t)0x80,
        0,0,0,0, 0,0,0,0, 0x01,0x02,0x04,0x08, 0x10,0x20,0x40,(int8_t)0x80
    );

    // Clamp shift count to 0-8
    __m512i count_sat = _mm512_min_epu8(count, _mm512_set1_epi8(8));

    // Get multiplier from lookup table
    __m512i multiplier = _mm512_shuffle_epi8(multiplier_lut, count_sat);

    // Process low bytes
    __m512i a_lo = _mm512_andnot_si512(mask_hi, a);
    __m512i multiplier_lo = _mm512_andnot_si512(mask_hi, multiplier);
    __m512i x_lo = _mm512_mullo_epi16(a_lo, multiplier_lo);
    x_lo = _mm512_srli_epi16(x_lo, 7);

    // Process high bytes
    __m512i multiplier_hi = _mm512_and_si512(mask_hi, multiplier);
    __m512i x_hi = _mm512_mulhi_epu16(a, multiplier_hi);
    x_hi = _mm512_slli_epi16(x_hi, 1);

    // Blend results
    __mmask64 blend_mask = _mm512_cmpgt_epi8_mask(mask_hi, _mm512_setzero_si512());
    return _mm512_mask_blend_epi8(blend_mask, x_lo, x_hi);
}

// ============================================================================
// Variable Rotate Left per Byte
// ============================================================================

__attribute__((target("avx512f,avx512bw")))
inline __m512i _mm512_rolv_epi8(__m512i x, __m512i y) {
    // Ensure shift counts are within 0-7
    __m512i y_mod = _mm512_and_si512(y, _mm512_set1_epi8(7));

    // Left shift x by y_mod
    __m512i left_shift = _mm512_sllv_epi8(x, y_mod);

    // Right shift x by (8 - y_mod)
    __m512i right_shift_counts = _mm512_sub_epi8(_mm512_set1_epi8(8), y_mod);
    __m512i right_shift = _mm512_srlv_epi8(x, right_shift_counts);

    // Combine with OR
    return _mm512_or_si512(left_shift, right_shift);
}

// ============================================================================
// Constant Rotate Left per Byte
// ============================================================================

__attribute__((target("avx512f,avx512bw")))
inline __m512i _mm512_rol_epi8(__m512i x, int r) {
    __m512i mask_lo = _mm512_set1_epi16(0x00FF);
    __m512i mask_hi = _mm512_set1_epi16((int16_t)0xFF00);

    __m512i a = _mm512_and_si512(x, mask_lo);
    __m512i b = _mm512_and_si512(x, mask_hi);

    // Rotate low bytes
    __m512i shiftedA = _mm512_slli_epi16(a, r);
    __m512i wrappedA = _mm512_srli_epi16(a, 8 - r);
    __m512i rotatedA = _mm512_and_si512(_mm512_or_si512(shiftedA, wrappedA), mask_lo);

    // Rotate high bytes
    __m512i shiftedB = _mm512_slli_epi16(b, r);
    __m512i wrappedB = _mm512_srli_epi16(b, 8 - r);
    __m512i rotatedB = _mm512_and_si512(_mm512_or_si512(shiftedB, wrappedB), mask_hi);

    return _mm512_or_si512(rotatedA, rotatedB);
}

// ============================================================================
// Bit Reverse per Byte
// ============================================================================

__attribute__((target("avx512f,avx512bw")))
inline __m512i _mm512_reverse_epi8(__m512i input) {
    const __m512i mask_0f = _mm512_set1_epi8(0x0F);
    const __m512i mask_33 = _mm512_set1_epi8(0x33);
    const __m512i mask_55 = _mm512_set1_epi8(0x55);
    const __m512i mask_ff = _mm512_set1_epi8((int8_t)0xFF);

    // Swap nibbles: (b & 0xF0) >> 4 | (b & 0x0F) << 4
    __m512i temp = _mm512_slli_epi16(_mm512_and_si512(input, mask_0f), 4);
    input = _mm512_srli_epi16(_mm512_andnot_si512(mask_0f, input), 4);
    input = _mm512_or_si512(input, temp);

    // Swap pairs: (b & 0xCC) >> 2 | (b & 0x33) << 2
    temp = _mm512_slli_epi16(_mm512_and_si512(input, mask_33), 2);
    input = _mm512_srli_epi16(_mm512_andnot_si512(mask_33, input), 2);
    input = _mm512_or_si512(input, temp);

    // Swap bits: (b & 0xAA) >> 1 | (b & 0x55) << 1
    temp = _mm512_slli_epi16(_mm512_and_si512(input, mask_55), 1);
    input = _mm512_srli_epi16(_mm512_andnot_si512(mask_55, input), 1);
    input = _mm512_or_si512(input, temp);

    return input;
}

// ============================================================================
// Mask Generation for Partial Vector Operations
// ============================================================================

__attribute__((target("avx512f,avx512bw")))
inline __mmask64 genMask64_avx512(int bytes) {
    // Generate a mask with the first 'bytes' bits set
    if (bytes <= 0) return 0;
    if (bytes >= 64) return ~(__mmask64)0;
    return (((__mmask64)1 << bytes) - 1);
}

#endif // defined(__x86_64__) || defined(_M_X64)
