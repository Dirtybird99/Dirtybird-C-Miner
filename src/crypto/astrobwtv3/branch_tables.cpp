/**
 * TNN-style wolfPermute Batch Branch Processing - Implementation
 *
 * Port of Tritonn's branch batch processing from Rust to C++.
 */

#include "branch_tables.hpp"
#include <algorithm>
#include <array>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

#endif

namespace branch_tables {

// ============================================================================
// CODE_LUT: Encodes 4 operations in 32 bits per byte value
// From TNN - each byte represents one of 16 operations
// Format: [op4:8][op3:8][op2:8][op1:8]
// ============================================================================

const uint32_t CODE_LUT[257] = {
    0x090F020A, 0x060B0500, 0x09080609, 0x0A0D030B, 0x04070A01, 0x09030607, 0x060D0401, 0x000A0904,
    0x040F0F06, 0x030E070C, 0x04020D02, 0x0B0F050A, 0x0C020C04, 0x0B03070F, 0x07060206, 0x0C060501,
    0x0E020B04, 0x03020F04, 0x0E0D0B0F, 0x010F0600, 0x0503080C, 0x0B030005, 0x0608020B, 0x0D0B0905,
    0x00070E0F, 0x090D0A01, 0x02090008, 0x0F050E0F, 0x0600000F, 0x02030700, 0x050E0F06, 0x040C0602,
    0x0C080D0C, 0x0A0E0802, 0x01060601, 0x00040B03, 0x090B0C0B, 0x0A070702, 0x070D090A, 0x0C030705,
    0x0A030903, 0x0F010D0E, 0x0B0D0C0A, 0x05000501, 0x09090D0A, 0x0F0F0509, 0x09000F0E, 0x0F050F06,
    0x0A04040F, 0x0900080E, 0x080D000B, 0x030E0E0F, 0x0A070409, 0x00090E0E, 0x08030404, 0x080E0E0B,
    0x0C02040B, 0x0A0F0D08, 0x080C0500, 0x0B020A04, 0x0304020D, 0x0F060D0F, 0x05040C00, 0x0F090100,
    0x03080E02, 0x0F0D0C02, 0x0C080E0B, 0x0B090C0F, 0x05040E03, 0x00020807, 0x0302070E, 0x0F040206,
    0x08090306, 0x09080F01, 0x020D0805, 0x0209050E, 0x0A0C0F07, 0x0D000609, 0x0A080201, 0x0E0C0002,
    0x0A060005, 0x0E060A09, 0x03040407, 0x06080D08, 0x010B0600, 0x07030A06, 0x0E0A0E04, 0x000D0E00,
    0x0C0B0204, 0x0002040C, 0x080F0B07, 0x09050E08, 0x09040905, 0x0C020500, 0x0B0A0506, 0x0B040F0F,
    0x0C0C090B, 0x0B060907, 0x0E06070E, 0x0E010807, 0x0A060809, 0x07090704, 0x0D01000D, 0x0B08030A,
    0x08090F00, 0x060D0A0C, 0x080E0B02, 0x070C0F0B, 0x0304050C, 0x020A030C, 0x000C0C07, 0x02080207,
    0x0D040F01, 0x0F0B0904, 0x0B080A04, 0x0A0F050D, 0x05030906, 0x060D0605, 0x0700060F, 0x080C0403,
    0x0C020308, 0x07000902, 0x0E0A0F0C, 0x05040D0D, 0x0C0C0304, 0x080C0007, 0x0D0B0F08, 0x06020503,
    0x0A0C0C0F, 0x04090907, 0x070A0B0E, 0x010B0902, 0x05080F0C, 0x030F0C06, 0x040E0B05, 0x070C0008,
    0x0701030F, 0x0F07080A, 0x03030001, 0x0F0D0C0D, 0x0B0C030F, 0x0B010900, 0x050F080C, 0x050D0706,
    0x0A06040A, 0x080E0C0E, 0x05060509, 0x04060E02, 0x050F0601, 0x03080100, 0x06060605, 0x00060206,
    0x0704060C, 0x0B0D0404, 0x0F040309, 0x01030903, 0x07070D0B, 0x07060A0B, 0x090D000B, 0x01030A03,
    0x07080B0D, 0x03030F0A, 0x02080C01, 0x06010E0B, 0x02090104, 0x0E030600, 0x0D000C04, 0x04040207,
    0x0A050A0B, 0x0B060E05, 0x01080102, 0x0D010908, 0x0E01060B, 0x04060200, 0x040A0909, 0x0D01020F,
    0x0302030F, 0x090C0C05, 0x0500040B, 0x0C000708, 0x070E0301, 0x04060C0F, 0x030B0F0E, 0x00010102,
    0x06020F03, 0x040E0F07, 0x0C0E0107, 0x0304000D, 0x0E090E0E, 0x0F0E0301, 0x0F07050C, 0x000D0A07,
    0x00060002, 0x05060A0B, 0x050A0605, 0x090C030E, 0x0D08060B, 0x0E0A0202, 0x0707080B, 0x04000203,
    0x07090808, 0x0D0C0E04, 0x03040A0F, 0x03050B0A, 0x0F0C0A03, 0x090E0600, 0x0E080809, 0x0F0D0909,
    0x0000070D, 0x0F080901, 0x0C0A0F04, 0x0E00010A, 0x0A0C0303, 0x00060D01, 0x03010704, 0x03050602,
    0x0A040105, 0x0F000B0E, 0x08040201, 0x0E0D0508, 0x0B060806, 0x0F030408, 0x07060302, 0x0D030A01,
    0x0C0B0D06, 0x0407080D, 0x08010203, 0x04060105, 0x00070009, 0x0D0A0C09, 0x02050A0A, 0x0D070308,
    0x02020E0F, 0x0B090D09, 0x05020703, 0x0C020D04, 0x03000501, 0x0F060C0D, 0x00000D01, 0x0F0B0205,
    0x04000506, 0x0E09030B, 0x00000103, 0x0F0C090B, 0x040C080F, 0x010F0C07, 0x000B0700, 0x0F0C0F04,
    0x0401090F, 0x080E0E0A, 0x050A090E, 0x0009080C, 0x080E0C06, 0x0D0C030D, 0x090D0C0D, 0x090D0C0D,
    0x00000000,
};

// ============================================================================
// CODE_LUT_16: Compressed 16-bit version (computed at startup)
// ============================================================================

// Static array initialized with computed values
const uint16_t CODE_LUT_16[257] = {
    // Pre-computed from CODE_LUT by extracting nibbles
    0x9F2A, 0x6B50, 0x9869, 0xAD3B, 0x47A1, 0x9367, 0x6D41, 0x0A94,
    0x4FF6, 0x3E7C, 0x42D2, 0xBF5A, 0xC2C4, 0xB37F, 0x7626, 0xC651,
    0xE2B4, 0x32F4, 0xEDBF, 0x1F60, 0x538C, 0xB305, 0x682B, 0xDB95,
    0x07EF, 0x9DA1, 0x2908, 0xF5EF, 0x600F, 0x2370, 0x5EF6, 0x4C62,
    0xC8DC, 0xAE82, 0x1661, 0x04B3, 0x9BCB, 0xA772, 0x7D9A, 0xC375,
    0xA393, 0xF1DE, 0xBDCA, 0x5051, 0x99DA, 0xFF59, 0x90FE, 0xF5F6,
    0xA44F, 0x908E, 0x8D0B, 0x3EEF, 0xA749, 0x09EE, 0x8344, 0x8EEB,
    0xC24B, 0xAFD8, 0x8C50, 0xB2A4, 0x342D, 0xF6DF, 0x54C0, 0xF910,
    0x38E2, 0xFDC2, 0xC8EB, 0xB9CF, 0x54E3, 0x0287, 0x327E, 0xF426,
    0x8936, 0x98F1, 0x2D85, 0x295E, 0xACF7, 0xD069, 0xA821, 0xEC02,
    0xA605, 0xE6A9, 0x3447, 0x68D8, 0x1B60, 0x73A6, 0xEAE4, 0x0DE0,
    0xCB24, 0x024C, 0x8FB7, 0x95E8, 0x9495, 0xC250, 0xBA56, 0xB4FF,
    0xCC9B, 0xB697, 0xE67E, 0xE187, 0xA689, 0x7974, 0xD10D, 0xB83A,
    0x89F0, 0x6DAC, 0x8EB2, 0x7CFB, 0x345C, 0x2A3C, 0x0CC7, 0x2827,
    0xD4F1, 0xFB94, 0xB8A4, 0xAF5D, 0x5396, 0x6D65, 0x706F, 0x8C43,
    0xC238, 0x7092, 0xEAFC, 0x54DD, 0xCC34, 0x8C07, 0xDBF8, 0x6253,
    0xACCF, 0x4997, 0x7ABE, 0x1B92, 0x58FC, 0x3FC6, 0x4EB5, 0x7C08,
    0x713F, 0xF78A, 0x3301, 0xFDCD, 0xBC3F, 0xB190, 0x5F8C, 0x5D76,
    0xA64A, 0x8ECE, 0x5659, 0x46E2, 0x5F61, 0x3810, 0x6665, 0x0626,
    0x746C, 0xBD44, 0xF439, 0x1393, 0x77DB, 0x76AB, 0x9D0B, 0x13A3,
    0x78BD, 0x33FA, 0x28C1, 0x61EB, 0x2914, 0xE360, 0xD0C4, 0x4427,
    0xA5AB, 0xB6E5, 0x1812, 0xD198, 0xE16B, 0x4620, 0x4A99, 0xD12F,
    0x323F, 0x9CC5, 0x504B, 0xC078, 0x7E31, 0x46CF, 0x3BFE, 0x0112,
    0x62F3, 0x4EF7, 0xCE17, 0x340D, 0xE9EE, 0xFE31, 0xF75C, 0x0DA7,
    0x0602, 0x56AB, 0x5A65, 0x9C3E, 0xD86B, 0xEA22, 0x778B, 0x4023,
    0x7988, 0xDCE4, 0x34AF, 0x35BA, 0xFCA3, 0x9E60, 0xE889, 0xFD99,
    0x007D, 0xF891, 0xCAF4, 0xE01A, 0xAC33, 0x06D1, 0x3174, 0x3562,
    0xA415, 0xF0BE, 0x8421, 0xED58, 0xB686, 0xF348, 0x7632, 0xD3A1,
    0xCBD6, 0x478D, 0x8123, 0x4615, 0x0709, 0xDAC9, 0x25AA, 0xD738,
    0x22EF, 0xB9D9, 0x5273, 0xC2D4, 0x3051, 0xF6CD, 0x00D1, 0xFB25,
    0x4056, 0xE93B, 0x0013, 0xFC9B, 0x4C8F, 0x1FC7, 0x0B70, 0xFCF4,
    0x419F, 0x8EEA, 0x5A9E, 0x098C, 0x8EC6, 0xDC3D, 0x9DCD, 0x9DCD,
    0x0000,
};

// ============================================================================
// Scalar Implementation
// ============================================================================

/**
 * Rotate left for 8-bit value.
 */
static inline uint8_t rotl8(uint8_t val, int n) {
    n &= 7;
    return static_cast<uint8_t>((val << n) | (val >> (8 - n)));
}

/**
 * Reverse bits of a byte.
 */
static inline uint8_t reverse_bits(uint8_t val) {
    val = static_cast<uint8_t>(((val & 0xF0) >> 4) | ((val & 0x0F) << 4));
    val = static_cast<uint8_t>(((val & 0xCC) >> 2) | ((val & 0x33) << 2));
    val = static_cast<uint8_t>(((val & 0xAA) >> 1) | ((val & 0x55) << 1));
    return val;
}

/**
 * Population count for a byte.
 */
static inline uint8_t popcount8(uint8_t val) {
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<uint8_t>(__builtin_popcount(val));
#elif defined(_MSC_VER)
    return static_cast<uint8_t>(__popcnt(val));
#else
    // Fallback software implementation
    val = static_cast<uint8_t>((val & 0x55) + ((val >> 1) & 0x55));
    val = static_cast<uint8_t>((val & 0x33) + ((val >> 2) & 0x33));
    val = static_cast<uint8_t>((val & 0x0F) + ((val >> 4) & 0x0F));
    return val;
#endif
}

uint8_t wolf_branch_scalar(uint8_t val, uint8_t pos2val, uint32_t opcode) {
    // Apply 4 operations in reverse order (matching TNN)
    for (int i = 3; i >= 0; i--) {
        uint8_t insn = static_cast<uint8_t>((opcode >> (i << 3)) & 0xFF);

        switch (insn) {
            case 0:  // add self
                val = static_cast<uint8_t>(val + val);
                break;
            case 1:  // sub with XOR 97
                val = static_cast<uint8_t>(val - (val ^ 97));
                break;
            case 2:  // mul self
                val = static_cast<uint8_t>(val * val);
                break;
            case 3:  // xor with pos2 value
                val ^= pos2val;
                break;
            case 4:  // NOT
                val = ~val;
                break;
            case 5:  // and with pos2 value
                val &= pos2val;
                break;
            case 6:  // variable left shift
                val = static_cast<uint8_t>(val << (val & 3));
                break;
            case 7:  // variable right shift
                val = static_cast<uint8_t>(val >> (val & 3));
                break;
            case 8:  // reverse bits
                val = reverse_bits(val);
                break;
            case 9:  // xor with popcount
                val ^= popcount8(val);
                break;
            case 10: // variable rotate left
                val = rotl8(val, val & 7);
                break;
            case 11: // rotate left by 1
                val = rotl8(val, 1);
                break;
            case 12: // xor with rotate left by 2
                val ^= rotl8(val, 2);
                break;
            case 13: // rotate left by 3
                val = rotl8(val, 3);
                break;
            case 14: // xor with rotate left by 4
                val ^= rotl8(val, 4);
                break;
            case 15: // rotate left by 5
                val = rotl8(val, 5);
                break;
            default:
                break;
        }
    }

    return val;
}

void wolf_permute_scalar(
    const uint8_t* input,
    uint8_t* output,
    uint8_t op,
    uint8_t pos1,
    uint8_t pos2
) {
    uint32_t opcode = CODE_LUT[op];
    uint8_t pos2val = input[pos2];

    for (size_t i = pos1; i < pos2; i++) {
        output[i] = wolf_branch_scalar(input[i], pos2val, opcode);
    }
}

// ============================================================================
// AVX2 Implementation (x86-64 only)
// ============================================================================

#if defined(__x86_64__) || defined(_M_X64)

bool avx2_available() {
#ifdef _MSC_VER
    int info[4];
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5)) != 0;  // AVX2 bit in EBX
#else
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return (ebx & (1 << 5)) != 0;  // AVX2 bit in EBX
    }
    return false;
#endif
}

#if defined(__AVX2__)

// Precomputed AVX2 blend masks for lengths 0..32.
// Each row has 0xFF in [0, len) and 0x00 in [len, 32).
constexpr std::array<std::array<uint8_t, 32>, 33> make_blend_mask_table() {
    std::array<std::array<uint8_t, 32>, 33> table{};
    for (size_t len = 0; len < table.size(); ++len) {
        for (size_t i = 0; i < table[len].size(); ++i) {
            table[len][i] = (i < len) ? static_cast<uint8_t>(0xFF) : static_cast<uint8_t>(0x00);
        }
    }
    return table;
}

alignas(32) static constexpr auto BLEND_MASK_TABLE = make_blend_mask_table();

// ============================================================================
// SIMD Helper Functions
// ============================================================================

/**
 * 8-bit parallel population count for 256-bit vector.
 */
static inline __m256i popcnt256_epi8(__m256i data) {
    __m256i lookup = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
    );
    __m256i mask4 = _mm256_set1_epi8(0x0F);

    __m256i low = _mm256_and_si256(mask4, data);
    __m256i high = _mm256_and_si256(mask4, _mm256_srli_epi16(data, 4));

    return _mm256_add_epi8(
        _mm256_shuffle_epi8(lookup, low),
        _mm256_shuffle_epi8(lookup, high)
    );
}

/**
 * Variable 8-bit left shift using multiplication LUT.
 */
static inline __m256i mm256_sllv_epi8(__m256i a, __m256i count) {
    __m256i mask_hi = _mm256_set1_epi32(static_cast<int>(0xFF00FF00));
    __m256i multiplier_lut = _mm256_set_epi8(
        0, 0, 0, 0, 0, 0, 0, 0,
        -128, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01,
        0, 0, 0, 0, 0, 0, 0, 0,
        -128, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01
    );

    __m256i count_sat = _mm256_min_epu8(count, _mm256_set1_epi8(8));
    __m256i multiplier = _mm256_shuffle_epi8(multiplier_lut, count_sat);

    __m256i x_lo = _mm256_mullo_epi16(a, multiplier);

    __m256i multiplier_hi = _mm256_srli_epi16(multiplier, 8);
    __m256i a_hi = _mm256_and_si256(a, mask_hi);
    __m256i x_hi = _mm256_mullo_epi16(a_hi, multiplier_hi);

    return _mm256_blendv_epi8(x_lo, x_hi, mask_hi);
}

/**
 * Variable 8-bit right shift using multiplication LUT.
 */
static inline __m256i mm256_srlv_epi8(__m256i a, __m256i count) {
    __m256i mask_hi = _mm256_set1_epi32(static_cast<int>(0xFF00FF00));
    __m256i multiplier_lut = _mm256_set_epi8(
        0, 0, 0, 0, 0, 0, 0, 0,
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, -128,
        0, 0, 0, 0, 0, 0, 0, 0,
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, -128
    );

    __m256i count_sat = _mm256_min_epu8(count, _mm256_set1_epi8(8));
    __m256i multiplier = _mm256_shuffle_epi8(multiplier_lut, count_sat);
    __m256i a_lo = _mm256_andnot_si256(mask_hi, a);
    __m256i multiplier_lo = _mm256_andnot_si256(mask_hi, multiplier);
    __m256i x_lo = _mm256_mullo_epi16(a_lo, multiplier_lo);
    x_lo = _mm256_srli_epi16(x_lo, 7);

    __m256i multiplier_hi = _mm256_and_si256(mask_hi, multiplier);
    __m256i x_hi = _mm256_mulhi_epu16(a, multiplier_hi);
    x_hi = _mm256_slli_epi16(x_hi, 1);

    return _mm256_blendv_epi8(x_lo, x_hi, mask_hi);
}

/**
 * Variable 8-bit rotate left.
 */
static inline __m256i mm256_rolv_epi8(__m256i x, __m256i y) {
    __m256i y_mod = _mm256_and_si256(y, _mm256_set1_epi8(7));
    __m256i left_shift = mm256_sllv_epi8(x, y_mod);
    __m256i right_shift_counts = _mm256_sub_epi8(_mm256_set1_epi8(8), y_mod);
    __m256i right_shift = mm256_srlv_epi8(x, right_shift_counts);
    return _mm256_or_si256(left_shift, right_shift);
}

/**
 * Fixed 8-bit rotate left by constant.
 */
static inline __m256i mm256_rol_epi8(__m256i x, int r) {
    r &= 7;
    if (r == 0) return x;

    __m256i mask1 = _mm256_set1_epi16(static_cast<int16_t>(0x00FF));
    __m256i mask2 = _mm256_set1_epi16(static_cast<int16_t>(0xFF00u));
    __m256i a = _mm256_and_si256(x, mask1);
    __m256i b = _mm256_and_si256(x, mask2);

    __m256i shifted_a, wrapped_a, rotated_a;
    __m256i shifted_b, wrapped_b, rotated_b;

    switch (r) {
        case 1:
            shifted_a = _mm256_slli_epi16(a, 1);
            wrapped_a = _mm256_srli_epi16(a, 7);
            shifted_b = _mm256_slli_epi16(b, 1);
            wrapped_b = _mm256_srli_epi16(b, 7);
            break;
        case 2:
            shifted_a = _mm256_slli_epi16(a, 2);
            wrapped_a = _mm256_srli_epi16(a, 6);
            shifted_b = _mm256_slli_epi16(b, 2);
            wrapped_b = _mm256_srli_epi16(b, 6);
            break;
        case 3:
            shifted_a = _mm256_slli_epi16(a, 3);
            wrapped_a = _mm256_srli_epi16(a, 5);
            shifted_b = _mm256_slli_epi16(b, 3);
            wrapped_b = _mm256_srli_epi16(b, 5);
            break;
        case 4:
            shifted_a = _mm256_slli_epi16(a, 4);
            wrapped_a = _mm256_srli_epi16(a, 4);
            shifted_b = _mm256_slli_epi16(b, 4);
            wrapped_b = _mm256_srli_epi16(b, 4);
            break;
        case 5:
            shifted_a = _mm256_slli_epi16(a, 5);
            wrapped_a = _mm256_srli_epi16(a, 3);
            shifted_b = _mm256_slli_epi16(b, 5);
            wrapped_b = _mm256_srli_epi16(b, 3);
            break;
        case 6:
            shifted_a = _mm256_slli_epi16(a, 6);
            wrapped_a = _mm256_srli_epi16(a, 2);
            shifted_b = _mm256_slli_epi16(b, 6);
            wrapped_b = _mm256_srli_epi16(b, 2);
            break;
        case 7:
            shifted_a = _mm256_slli_epi16(a, 7);
            wrapped_a = _mm256_srli_epi16(a, 1);
            shifted_b = _mm256_slli_epi16(b, 7);
            wrapped_b = _mm256_srli_epi16(b, 1);
            break;
        default:
            return x;
    }

    rotated_a = _mm256_or_si256(shifted_a, wrapped_a);
    rotated_a = _mm256_and_si256(rotated_a, mask1);
    rotated_b = _mm256_or_si256(shifted_b, wrapped_b);
    rotated_b = _mm256_and_si256(rotated_b, mask2);

    return _mm256_or_si256(rotated_a, rotated_b);
}

// ============================================================================
// AVX2 Primitive Operations
// ============================================================================

/// Op 0: add self (val + val)
static inline __m256i op_add_self(__m256i input) {
    return _mm256_add_epi8(input, input);
}

/// Op 1: sub with XOR 97 -> val - (val ^ 97)
static inline __m256i op_sub_xor97(__m256i input) {
    __m256i xored = _mm256_xor_si256(input, _mm256_set1_epi8(97));
    return _mm256_sub_epi8(input, xored);
}

/// Op 2: mul self (val * val) using 16-bit intermediates
static inline __m256i op_mul_self(__m256i input) {
    __m256i mask1 = _mm256_set1_epi16(static_cast<int16_t>(0xFF00u));
    __m256i mask2 = _mm256_set1_epi16(static_cast<int16_t>(0x00FF));

    // Split odd and even bytes
    __m256i aa = _mm256_srli_epi16(_mm256_and_si256(input, mask1), 8);
    __m256i ba = _mm256_and_si256(input, mask2);

    // Multiply
    __m256i pa = _mm256_slli_epi16(_mm256_mullo_epi16(aa, aa), 8);
    __m256i pb = _mm256_mullo_epi16(ba, ba);

    // Mask and combine
    pa = _mm256_and_si256(pa, mask1);
    pb = _mm256_and_si256(pb, mask2);

    return _mm256_or_si256(pa, pb);
}

/// Op 3: xor with pos2 value (broadcast)
static inline __m256i op_xor_pos2val(__m256i input, __m256i pos2_val) {
    return _mm256_xor_si256(input, pos2_val);
}

/// Op 4: NOT (xor with -1)
static inline __m256i op_not(__m256i input) {
    return _mm256_xor_si256(input, _mm256_set1_epi64x(-1LL));
}

/// Op 5: and with pos2 value
static inline __m256i op_and_pos2val(__m256i input, __m256i pos2_val) {
    return _mm256_and_si256(input, pos2_val);
}

/// Op 6: variable left shift (val << (val & 3))
static inline __m256i op_sllv(__m256i input) {
    __m256i vec_3 = _mm256_set1_epi8(3);
    __m256i count = _mm256_and_si256(input, vec_3);
    return mm256_sllv_epi8(input, count);
}

/// Op 7: variable right shift (val >> (val & 3))
static inline __m256i op_srlv(__m256i input) {
    __m256i vec_3 = _mm256_set1_epi8(3);
    __m256i count = _mm256_and_si256(input, vec_3);
    return mm256_srlv_epi8(input, count);
}

/// Op 8: reverse bits
static inline __m256i op_reverse_bits(__m256i input) {
    __m256i mask_0f = _mm256_set1_epi8(0x0F);
    __m256i mask_33 = _mm256_set1_epi8(0x33);
    __m256i mask_55 = _mm256_set1_epi8(0x55);
    __m256i mask_ff = _mm256_set1_epi8(-1);  // 0xFF

    // b = (b & 0xF0) >> 4 | (b & 0x0F) << 4
    __m256i temp = _mm256_and_si256(input, mask_0f);
    temp = _mm256_slli_epi16(temp, 4);
    __m256i result = _mm256_and_si256(input, _mm256_andnot_si256(mask_0f, mask_ff));
    result = _mm256_srli_epi16(result, 4);
    result = _mm256_or_si256(result, temp);

    // b = (b & 0xCC) >> 2 | (b & 0x33) << 2
    temp = _mm256_and_si256(result, mask_33);
    temp = _mm256_slli_epi16(temp, 2);
    result = _mm256_and_si256(result, _mm256_andnot_si256(mask_33, mask_ff));
    result = _mm256_srli_epi16(result, 2);
    result = _mm256_or_si256(result, temp);

    // b = (b & 0xAA) >> 1 | (b & 0x55) << 1
    temp = _mm256_and_si256(result, mask_55);
    temp = _mm256_slli_epi16(temp, 1);
    result = _mm256_and_si256(result, _mm256_andnot_si256(mask_55, mask_ff));
    result = _mm256_srli_epi16(result, 1);
    result = _mm256_or_si256(result, temp);

    return result;
}

/// Op 9: xor with popcount
static inline __m256i op_xor_popcount(__m256i input) {
    __m256i pop = popcnt256_epi8(input);
    return _mm256_xor_si256(input, pop);
}

/// Op 10: variable rotate left (rol by self)
static inline __m256i op_rolv_self(__m256i input) {
    return mm256_rolv_epi8(input, input);
}

/// Op 11: rotate left by 1
static inline __m256i op_rol1(__m256i input) {
    return mm256_rol_epi8(input, 1);
}

/// Op 12: xor with rotate left by 2
static inline __m256i op_xor_rol2(__m256i input) {
    return _mm256_xor_si256(input, mm256_rol_epi8(input, 2));
}

/// Op 13: rotate left by 3
static inline __m256i op_rol3(__m256i input) {
    return mm256_rol_epi8(input, 3);
}

/// Op 14: xor with rotate left by 4
static inline __m256i op_xor_rol4(__m256i input) {
    return _mm256_xor_si256(input, mm256_rol_epi8(input, 4));
}

/// Op 15: rotate left by 5
static inline __m256i op_rol5(__m256i input) {
    return mm256_rol_epi8(input, 5);
}

/**
 * Apply a single TNN instruction to the vector.
 */
static inline __m256i apply_wolf_insn(__m256i input, __m256i pos2_val, uint8_t insn) {
    switch (insn) {
        case 0:  return op_add_self(input);
        case 1:  return op_sub_xor97(input);
        case 2:  return op_mul_self(input);
        case 3:  return op_xor_pos2val(input, pos2_val);
        case 4:  return op_not(input);
        case 5:  return op_and_pos2val(input, pos2_val);
        case 6:  return op_sllv(input);
        case 7:  return op_srlv(input);
        case 8:  return op_reverse_bits(input);
        case 9:  return op_xor_popcount(input);
        case 10: return op_rolv_self(input);
        case 11: return op_rol1(input);
        case 12: return op_xor_rol2(input);
        case 13: return op_rol3(input);
        case 14: return op_xor_rol4(input);
        case 15: return op_rol5(input);
        default: return input;
    }
}

void gen_mask_avx2(uint8_t bytes, uint8_t mask[32]) {
    uint8_t clamped = std::min<uint8_t>(bytes, 32);
    __m256i result = _mm256_load_si256(
        reinterpret_cast<const __m256i*>(BLEND_MASK_TABLE[clamped].data())
    );
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(mask), result);
}

void wolf_permute_avx2(
    const uint8_t* input,
    uint8_t* output,
    uint8_t op,
    uint8_t pos1,
    uint8_t pos2
) {
    // Get the 16-bit compressed opcode (4 ops as nibbles)
    uint16_t opcode = CODE_LUT_16[op];

    // Load 32 bytes starting at pos1
    __m256i data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input + pos1));
    __m256i old = data;

    // Get pos2 value and broadcast it
    uint8_t pos2_val_byte = input[pos2];
    __m256i pos2_val = _mm256_set1_epi8(static_cast<int8_t>(pos2_val_byte));

    // Apply 4 operations in reverse order (matching TNN)
    __m256i result = data;
    for (int i = 3; i >= 0; i--) {
        uint8_t insn = static_cast<uint8_t>((opcode >> (i << 2)) & 0xF);
        result = apply_wolf_insn(result, pos2_val, insn);
    }

    // Generate blend mask for [0, pos2-pos1) range
    uint8_t bytes_to_update = (pos2 > pos1) ? static_cast<uint8_t>(pos2 - pos1) : 0;
    bytes_to_update = std::min<uint8_t>(bytes_to_update, 32);
    __m256i mask = _mm256_load_si256(
        reinterpret_cast<const __m256i*>(BLEND_MASK_TABLE[bytes_to_update].data())
    );

    // Blend: keep old data where mask is 0, use new data where mask is 0xFF
    result = _mm256_blendv_epi8(old, result, mask);

    // Store result
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + pos1), result);
}

#else  // No AVX2 support at compile time

void gen_mask_avx2(uint8_t bytes, uint8_t mask[32]) {
    for (int i = 0; i < 32; i++) {
        mask[i] = (i < bytes) ? 0xFF : 0x00;
    }
}

void wolf_permute_avx2(
    const uint8_t* input,
    uint8_t* output,
    uint8_t op,
    uint8_t pos1,
    uint8_t pos2
) {
    wolf_permute_scalar(input, output, op, pos1, pos2);
}

#endif  // __AVX2__

#endif  // x86_64

// ============================================================================
// High-Level API Implementation
// ============================================================================

uint8_t apply_branch_batch(uint8_t data[256], uint8_t branch_byte, uint8_t pos1, uint8_t pos2) {
#if defined(__x86_64__) || defined(_M_X64)
    static bool has_avx2 = avx2_available();
    if (has_avx2) {
#if defined(__AVX2__)
        wolf_permute_avx2(data, data, branch_byte, pos1, pos2);
        return data[(pos2 > 0) ? (pos2 - 1) : 0];
#endif
    }
#endif

    // Scalar fallback
    uint32_t opcode = CODE_LUT[branch_byte];
    uint8_t pos2val = data[pos2];
    for (size_t i = pos1; i < pos2; i++) {
        data[i] = wolf_branch_scalar(data[i], pos2val, opcode);
    }
    return data[(pos2 > 0) ? (pos2 - 1) : 0];
}

void apply_branch_batch_8(
    uint8_t data[8][256],
    const uint8_t branch_bytes[8],
    const uint8_t pos1_vals[8],
    const uint8_t pos2_vals[8],
    uint8_t done_mask
) {
#if defined(__x86_64__) || defined(_M_X64)
    static bool has_avx2 = avx2_available();
    if (has_avx2) {
#if defined(__AVX2__)
        for (int i = 0; i < 8; i++) {
            if ((done_mask & (1 << i)) == 0) {
                wolf_permute_avx2(data[i], data[i], branch_bytes[i], pos1_vals[i], pos2_vals[i]);
            }
        }
        return;
#endif
    }
#endif

    // Scalar fallback
    for (int i = 0; i < 8; i++) {
        if ((done_mask & (1 << i)) == 0) {
            wolf_permute_scalar(data[i], data[i], branch_bytes[i], pos1_vals[i], pos2_vals[i]);
        }
    }
}

} // namespace branch_tables
