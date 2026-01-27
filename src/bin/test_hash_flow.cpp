/**
 * Test full AstroBWTv3 hash flow with OpenSSL vs FastRc4
 * This traces every step to find where divergence occurs
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <openssl/sha.h>
#include <openssl/rc4.h>
#include "astroworker.h"
#include "Salsa20.h"
#include "crypto/astrobwtv3/rc4_avx512.hpp"

void print_hex(const char* label, const uint8_t* data, size_t len) {
    printf("%s: ", label);
    for (size_t i = 0; i < len && i < 32; i++) {
        printf("%02x", data[i]);
    }
    if (len > 32) printf("...");
    printf("\n");
}

bool compare_buffers(const char* name, const uint8_t* a, const uint8_t* b, size_t len) {
    bool match = memcmp(a, b, len) == 0;
    if (!match) {
        printf("MISMATCH at %s\n", name);
        print_hex("  OpenSSL", a, len > 64 ? 64 : len);
        print_hex("  FastRc4", b, len > 64 ? 64 : len);
        // Find first difference
        for (size_t i = 0; i < len; i++) {
            if (a[i] != b[i]) {
                printf("  First diff at byte %zu: OpenSSL=%02x, FastRc4=%02x\n", i, a[i], b[i]);
                break;
            }
        }
    }
    return match;
}

int main() {
    printf("=== Full AstroBWTv3 Hash Flow Test ===\n\n");

    // Test input (like a real mining job)
    uint8_t input[48] = {0};
    for (int i = 0; i < 48; i++) {
        input[i] = (uint8_t)(i * 0x17 + 0x42);
    }
    print_hex("Input", input, 48);

    // Allocate two workers - one for each implementation
    printf("\n=== Allocating workers ===\n");

    // Worker for OpenSSL RC4 (simulated by temporarily ignoring USE_FAST_RC4)
    uint8_t scratch_openssl[384] = {0};
    uint8_t scratch_fast[384] = {0};

    // SHA256 of input
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, input, 48);
    SHA256_Final(&scratch_openssl[320], &sha256);

    memcpy(&scratch_fast[320], &scratch_openssl[320], 32);
    print_hex("SHA256 output", &scratch_openssl[320], 32);
    printf("SHA256: %s\n", memcmp(&scratch_openssl[320], &scratch_fast[320], 32) == 0 ? "MATCH" : "MISMATCH");

    // Salsa20
    ucstk::Salsa20 salsa20_ssl, salsa20_fast;
    salsa20_ssl.setKey(&scratch_openssl[320]);
    salsa20_ssl.setIv(&scratch_openssl[256]);

    salsa20_fast.setKey(&scratch_fast[320]);
    salsa20_fast.setIv(&scratch_fast[256]);

    uint8_t salsaInput[256] = {0};
    salsa20_ssl.processBytes(salsaInput, scratch_openssl, 256);
    salsa20_fast.processBytes(salsaInput, scratch_fast, 256);

    print_hex("Salsa20 output (OpenSSL)", scratch_openssl, 32);
    print_hex("Salsa20 output (FastRc4)", scratch_fast, 32);
    printf("Salsa20: %s\n", compare_buffers("Salsa20", scratch_openssl, scratch_fast, 256) ? "MATCH" : "MISMATCH");

    // RC4 setup and encryption
    printf("\n=== RC4 Phase ===\n");

    // OpenSSL RC4
    RC4_KEY rc4_key_ssl;
    RC4_set_key(&rc4_key_ssl, 256, scratch_openssl);
    uint8_t after_rc4_ssl[256];
    memcpy(after_rc4_ssl, scratch_openssl, 256);
    RC4(&rc4_key_ssl, 256, after_rc4_ssl, after_rc4_ssl);

    // FastRc4
    rc4_avx512::FastRc4 rc4_key_fast;
    rc4_key_fast.set_key(scratch_fast, 256);
    uint8_t after_rc4_fast[256];
    memcpy(after_rc4_fast, scratch_fast, 256);
    rc4_key_fast.apply_keystream_256(after_rc4_fast);

    print_hex("After RC4 (OpenSSL)", after_rc4_ssl, 32);
    print_hex("After RC4 (FastRc4)", after_rc4_fast, 32);
    printf("Initial RC4: %s\n", compare_buffers("Initial RC4", after_rc4_ssl, after_rc4_fast, 256) ? "MATCH" : "MISMATCH");

    // Check RC4 state after initial encryption
    printf("\nRC4 state after initial encryption:\n");
    printf("  FastRc4: i=%u, j=%u\n", rc4_key_fast.i, rc4_key_fast.j);

    // Simulate wolfCompute pattern: multiple RC4 calls without key reset
    printf("\n=== Simulating wolfCompute RC4 pattern ===\n");

    uint8_t chunk_ssl[256], chunk_fast[256];
    bool all_match = true;

    for (int iter = 0; iter < 10; iter++) {
        // Prepare chunk data (simulating wolf permutation output)
        for (int i = 0; i < 256; i++) {
            chunk_ssl[i] = after_rc4_ssl[(i + iter * 7) % 256] ^ (uint8_t)iter;
            chunk_fast[i] = after_rc4_fast[(i + iter * 7) % 256] ^ (uint8_t)iter;
        }

        // Apply RC4 without resetting key (when A <= 0x40)
        RC4(&rc4_key_ssl, 256, chunk_ssl, chunk_ssl);
        rc4_key_fast.apply_keystream_256(chunk_fast);

        bool match = memcmp(chunk_ssl, chunk_fast, 256) == 0;
        printf("Iteration %d: %s (FastRc4 i=%u, j=%u)\n",
               iter, match ? "MATCH" : "MISMATCH", rc4_key_fast.i, rc4_key_fast.j);

        if (!match) {
            all_match = false;
            compare_buffers("chunk", chunk_ssl, chunk_fast, 256);
        }
    }

    // Simulate key reset (when op >= 254)
    printf("\n=== Simulating op >= 254 (key reset) ===\n");

    uint8_t new_key[256];
    for (int i = 0; i < 256; i++) {
        new_key[i] = chunk_ssl[i];  // Use final chunk as new key
    }

    RC4_set_key(&rc4_key_ssl, 256, new_key);
    rc4_key_fast.set_key(new_key, 256);

    printf("After key reset: FastRc4 i=%u, j=%u\n", rc4_key_fast.i, rc4_key_fast.j);

    // More iterations after reset
    for (int iter = 0; iter < 5; iter++) {
        for (int i = 0; i < 256; i++) {
            chunk_ssl[i] = new_key[(i + iter * 11) % 256] ^ (uint8_t)(iter + 0x50);
            chunk_fast[i] = new_key[(i + iter * 11) % 256] ^ (uint8_t)(iter + 0x50);
        }

        RC4(&rc4_key_ssl, 256, chunk_ssl, chunk_ssl);
        rc4_key_fast.apply_keystream_256(chunk_fast);

        bool match = memcmp(chunk_ssl, chunk_fast, 256) == 0;
        printf("After-reset iter %d: %s\n", iter, match ? "MATCH" : "MISMATCH");

        if (!match) {
            all_match = false;
            compare_buffers("chunk", chunk_ssl, chunk_fast, 256);
        }
    }

    // Test using wrapper functions (as used in actual miner)
    printf("\n=== Testing wrapper functions ===\n");

    rc4_avx512::FastRc4 wrapper_test;
    RC4_KEY wrapper_ssl;

    uint8_t key_data[256];
    for (int i = 0; i < 256; i++) key_data[i] = (uint8_t)(i * 3 + 7);

    // Use wrapper functions exactly as miner does
    rc4_avx512::fast_rc4_set_key(wrapper_test, 256, key_data);
    RC4_set_key(&wrapper_ssl, 256, key_data);

    uint8_t wrapper_data_ssl[256], wrapper_data_fast[256];
    for (int i = 0; i < 256; i++) {
        wrapper_data_ssl[i] = (uint8_t)i;
        wrapper_data_fast[i] = (uint8_t)i;
    }

    rc4_avx512::fast_rc4(wrapper_test, 256, wrapper_data_fast, wrapper_data_fast);
    RC4(&wrapper_ssl, 256, wrapper_data_ssl, wrapper_data_ssl);

    printf("Wrapper function test: %s\n",
           compare_buffers("wrapper", wrapper_data_ssl, wrapper_data_fast, 256) ? "MATCH" : "MISMATCH");

    printf("\n=== Summary ===\n");
    printf("All tests: %s\n", all_match ? "PASSED" : "FAILED");

    return all_match ? 0 : 1;
}
