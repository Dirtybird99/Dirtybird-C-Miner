/**
 * Test that mimics the exact prep phase RC4 flow
 * Compares OpenSSL vs CRYPTOGAMS
 */
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <openssl/rc4.h>
#include "rc4_cryptogams.hpp"

void print_hex(const char* label, const uint8_t* data, size_t len) {
    printf("%s: ", label);
    for (size_t i = 0; i < len && i < 32; i++) {
        printf("%02x", data[i]);
    }
    if (len > 32) printf("...");
    printf("\n");
}

int main() {
    printf("=== Prep Phase RC4 Test ===\n\n");

    // Simulate the scratch buffer after Salsa20
    // Use a deterministic pattern
    uint8_t scratch_openssl[384] = {0};
    uint8_t scratch_cryptogams[384] = {0};

    // Fill with a pattern (simulating Salsa20 output)
    for (int i = 0; i < 256; i++) {
        scratch_openssl[i] = (uint8_t)(i * 0x47 + 0x13);
        scratch_cryptogams[i] = scratch_openssl[i];
    }

    printf("Before RC4:\n");
    print_hex("scratch", scratch_openssl, 32);

    // OpenSSL path (exact miner flow)
    RC4_KEY openssl_key;
    RC4_set_key(&openssl_key, 256, scratch_openssl);
    RC4(&openssl_key, 256, scratch_openssl, scratch_openssl);

    printf("\nOpenSSL path:\n");
    print_hex("after RC4", scratch_openssl, 32);
    printf("OpenSSL state: x=%u, y=%u\n", openssl_key.x, openssl_key.y);

    // CRYPTOGAMS path (exact miner flow with dual state)
    RC4_KEY cryptogams_openssl_key;
    rc4_cryptogams::CryptogamsRc4 cryptogams_rc4;

    // This is the exact flow from the miner:
    cryptogams_rc4.set_key(scratch_cryptogams, 256);
    RC4_set_key(&cryptogams_openssl_key, 256, scratch_cryptogams);
    cryptogams_rc4.apply_keystream_256(scratch_cryptogams);

    printf("\nCRYPTOGAMS dual path:\n");
    print_hex("after RC4", scratch_cryptogams, 32);
    printf("CRYPTOGAMS state: i=%u, j=%u\n", cryptogams_rc4.i, cryptogams_rc4.j);
    printf("OpenSSL shadow state: x=%u, y=%u\n", cryptogams_openssl_key.x, cryptogams_openssl_key.y);

    // Compare encrypted data
    bool data_match = memcmp(scratch_openssl, scratch_cryptogams, 256) == 0;
    printf("\n=== Results ===\n");
    printf("Encrypted data match: %s\n", data_match ? "YES" : "NO");

    if (!data_match) {
        printf("\nFirst differences:\n");
        for (int i = 0; i < 256; i++) {
            if (scratch_openssl[i] != scratch_cryptogams[i]) {
                printf("  [%3d] openssl=0x%02x cryptogams=0x%02x\n",
                       i, scratch_openssl[i], scratch_cryptogams[i]);
            }
        }
    }

    // Now test a wolfCompute-like iteration without key reset
    printf("\n=== wolfCompute simulation (no key reset) ===\n");

    // Initialize chunk data
    uint8_t chunk_openssl[256];
    uint8_t chunk_cryptogams[256];
    for (int i = 0; i < 256; i++) {
        chunk_openssl[i] = (uint8_t)(i ^ 0xAA);
        chunk_cryptogams[i] = chunk_openssl[i];
    }

    // Encrypt with current state (no key reset)
    RC4(&openssl_key, 256, chunk_openssl, chunk_openssl);
    cryptogams_rc4.apply_keystream_256(chunk_cryptogams);

    print_hex("OpenSSL chunk", chunk_openssl, 32);
    print_hex("CRYPTOGAMS chunk", chunk_cryptogams, 32);

    bool chunk_match = memcmp(chunk_openssl, chunk_cryptogams, 256) == 0;
    printf("wolfCompute chunk match: %s\n", chunk_match ? "YES" : "NO");

    if (!chunk_match) {
        printf("\nFirst differences (chunk):\n");
        int diff_count = 0;
        for (int i = 0; i < 256; i++) {
            if (chunk_openssl[i] != chunk_cryptogams[i]) {
                printf("  [%3d] openssl=0x%02x cryptogams=0x%02x\n",
                       i, chunk_openssl[i], chunk_cryptogams[i]);
                if (++diff_count >= 10) {
                    printf("  ... more differences\n");
                    break;
                }
            }
        }
    }

    // Test with key reset (simulating op >= 254)
    printf("\n=== wolfCompute simulation (with key reset) ===\n");

    // Reset keys from encrypted scratch
    RC4_set_key(&openssl_key, 256, scratch_openssl);
    cryptogams_rc4.set_key(scratch_cryptogams, 256);

    // Re-initialize chunk
    for (int i = 0; i < 256; i++) {
        chunk_openssl[i] = (uint8_t)(i ^ 0x55);
        chunk_cryptogams[i] = chunk_openssl[i];
    }

    // Encrypt
    RC4(&openssl_key, 256, chunk_openssl, chunk_openssl);
    cryptogams_rc4.apply_keystream_256(chunk_cryptogams);

    print_hex("OpenSSL chunk (after key reset)", chunk_openssl, 32);
    print_hex("CRYPTOGAMS chunk (after key reset)", chunk_cryptogams, 32);

    bool reset_match = memcmp(chunk_openssl, chunk_cryptogams, 256) == 0;
    printf("wolfCompute chunk match (after key reset): %s\n", reset_match ? "YES" : "NO");

    return (data_match && chunk_match && reset_match) ? 0 : 1;
}
