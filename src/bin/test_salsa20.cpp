/**
 * Debug test: Verify Salsa20 implementations
 */

#include <cstdio>
#include <cstdint>
#include <cstring>

#include "Salsa20.h"
#include "salsa20_simd.h"

static const uint8_t TEST_KEY[32] = {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
    0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f
};

static const uint8_t TEST_IV[8] = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};

void print_hex(const char* label, const uint8_t* data, size_t len) {
    printf("%s: ", label);
    for (size_t i = 0; i < len && i < 32; i++) {
        printf("%02x", data[i]);
    }
    if (len > 32) printf("...");
    printf("\n");
}

int main() {
    printf("=== Salsa20 Implementation Test ===\n\n");

    salsa20_simd_init();
    printf("SIMD impl: %s\n\n", salsa20_simd_impl_name());

    uint8_t input[64] = {0};  // All zeros
    uint8_t output_scalar[64];
    uint8_t output_simd[64];

    // Test 1: Generate keystream (64 bytes)
    printf("Test 1: Generate 64-byte keystream\n");

    // Scalar (ucstk)
    ucstk::Salsa20 scalar(TEST_KEY);
    scalar.setIv(TEST_IV);
    scalar.processBytes(input, output_scalar, 64);

    // SIMD
    salsa20_simd_process(TEST_KEY, TEST_IV, input, output_simd, 64);

    print_hex("Scalar", output_scalar, 64);
    print_hex("SIMD  ", output_simd, 64);

    bool match = memcmp(output_scalar, output_simd, 64) == 0;
    printf("Match: %s\n\n", match ? "YES" : "NO");

    if (!match) {
        printf("First differences:\n");
        for (int i = 0; i < 64; i++) {
            if (output_scalar[i] != output_simd[i]) {
                printf("  [%2d] scalar=0x%02x simd=0x%02x\n",
                       i, output_scalar[i], output_simd[i]);
            }
        }
    }

    // Test 2: Multiple blocks (256 bytes)
    printf("\nTest 2: 256 bytes (4 blocks)\n");

    uint8_t input256[256] = {0};
    uint8_t output_scalar256[256];
    uint8_t output_simd256[256];

    ucstk::Salsa20 scalar2(TEST_KEY);
    scalar2.setIv(TEST_IV);
    scalar2.processBytes(input256, output_scalar256, 256);

    salsa20_simd_process(TEST_KEY, TEST_IV, input256, output_simd256, 256);

    print_hex("Scalar", output_scalar256, 64);
    print_hex("SIMD  ", output_simd256, 64);

    bool match2 = memcmp(output_scalar256, output_simd256, 256) == 0;
    printf("Match: %s\n\n", match2 ? "YES" : "NO");

    return (match && match2) ? 0 : 1;
}
