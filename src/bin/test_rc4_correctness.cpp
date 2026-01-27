/**
 * Test FastRc4 correctness against OpenSSL RC4
 */

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <openssl/rc4.h>
#include "crypto/astrobwtv3/rc4_avx512.hpp"

void print_hex(const char* label, const uint8_t* data, size_t len) {
    printf("%s: ", label);
    for (size_t i = 0; i < len && i < 32; i++) {
        printf("%02x", data[i]);
    }
    if (len > 32) printf("...");
    printf("\n");
}

int main() {
    printf("=== RC4 Correctness Test ===\n\n");

    // Test key (256 bytes like AstroBWT uses)
    uint8_t key[256];
    for (int i = 0; i < 256; i++) {
        key[i] = (uint8_t)(i * 0x47 + 0x13);  // Arbitrary pattern
    }

    // Input data (256 bytes)
    uint8_t input[256];
    for (int i = 0; i < 256; i++) {
        input[i] = (uint8_t)(i ^ 0xAA);
    }

    // Output buffers
    uint8_t out_openssl[256];
    uint8_t out_fast[256];

    // Copy input to outputs
    memcpy(out_openssl, input, 256);
    memcpy(out_fast, input, 256);

    printf("Key (first 32 bytes): ");
    print_hex("", key, 32);
    printf("Input (first 32 bytes): ");
    print_hex("", input, 32);
    printf("\n");

    // Test 1: OpenSSL RC4
    RC4_KEY openssl_key;
    RC4_set_key(&openssl_key, 256, key);
    RC4(&openssl_key, 256, out_openssl, out_openssl);
    print_hex("OpenSSL output", out_openssl, 32);

    // Test 2: FastRc4
    rc4_avx512::FastRc4 fast_key;
    rc4_avx512::fast_rc4_set_key(fast_key, 256, key);
    rc4_avx512::fast_rc4(fast_key, 256, out_fast, out_fast);
    print_hex("FastRc4 output", out_fast, 32);

    // Compare
    bool match = memcmp(out_openssl, out_fast, 256) == 0;
    printf("\nMatch: %s\n", match ? "YES" : "NO");

    if (!match) {
        printf("\nFirst differences:\n");
        int diff_count = 0;
        for (int i = 0; i < 256; i++) {
            if (out_openssl[i] != out_fast[i]) {
                printf("  [%3d] openssl=0x%02x fast=0x%02x\n",
                       i, out_openssl[i], out_fast[i]);
                if (++diff_count >= 10) {
                    printf("  ... (more differences)\n");
                    break;
                }
            }
        }
    }

    // Test 3: Verify apply_keystream_256 specifically
    printf("\n=== Test apply_keystream_256 specifically ===\n");

    // Reset outputs
    memcpy(out_openssl, input, 256);
    memcpy(out_fast, input, 256);

    // OpenSSL
    RC4_set_key(&openssl_key, 256, key);
    RC4(&openssl_key, 256, out_openssl, out_openssl);

    // FastRc4 using apply_keystream_256 directly
    rc4_avx512::FastRc4 fast_key2;
    fast_key2.set_key(key, 256);
    fast_key2.apply_keystream_256(out_fast);

    bool match2 = memcmp(out_openssl, out_fast, 256) == 0;
    printf("Match (apply_keystream_256): %s\n", match2 ? "YES" : "NO");

    // Test 4: Verify the S-box after key setup
    printf("\n=== S-box comparison after key setup ===\n");

    RC4_KEY openssl_key2;
    RC4_set_key(&openssl_key2, 256, key);

    rc4_avx512::FastRc4 fast_key3;
    fast_key3.set_key(key, 256);

    // OpenSSL RC4_KEY internal structure:
    // - On many implementations, first 2 bytes are x and y (i and j)
    // - Followed by 256-byte S-box
    // The exact layout depends on OpenSSL version

    printf("FastRc4 i=%d, j=%d\n", fast_key3.i, fast_key3.j);
    printf("FastRc4 S[0..15]: ");
    for (int i = 0; i < 16; i++) printf("%02x ", fast_key3.s[i]);
    printf("\n");

    // Test 5: Multiple rounds (simulating the miner's pattern)
    printf("\n=== Multiple RC4 applications (simulating miner) ===\n");

    uint8_t chunk1[256], chunk2[256];
    uint8_t chunk1_fast[256], chunk2_fast[256];

    // Initialize with different data
    for (int i = 0; i < 256; i++) {
        chunk1[i] = (uint8_t)(i * 3);
        chunk2[i] = (uint8_t)(i * 7 + 0x55);
        chunk1_fast[i] = chunk1[i];
        chunk2_fast[i] = chunk2[i];
    }

    // OpenSSL: set key from chunk1, apply to chunk2
    RC4_set_key(&openssl_key, 256, chunk1);
    RC4(&openssl_key, 256, chunk2, chunk2);

    // FastRc4: same operations
    rc4_avx512::FastRc4 fast_key4;
    fast_key4.set_key(chunk1_fast, 256);
    fast_key4.apply_keystream_256(chunk2_fast);

    bool match3 = memcmp(chunk2, chunk2_fast, 256) == 0;
    print_hex("OpenSSL result", chunk2, 32);
    print_hex("FastRc4 result", chunk2_fast, 32);
    printf("Match (miner pattern): %s\n", match3 ? "YES" : "NO");

    return (match && match2 && match3) ? 0 : 1;
}
