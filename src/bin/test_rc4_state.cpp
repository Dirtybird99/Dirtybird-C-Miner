/**
 * Test RC4 state persistence - simulating wolfCompute's pattern
 * where RC4 encryption is called multiple times without resetting the key
 */

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <openssl/rc4.h>
#include "crypto/astrobwtv3/rc4_avx512.hpp"

void print_hex(const char* label, const uint8_t* data, size_t len) {
    printf("%s: ", label);
    for (size_t i = 0; i < len && i < 16; i++) {
        printf("%02x", data[i]);
    }
    if (len > 16) printf("...");
    printf("\n");
}

int main() {
    printf("=== RC4 State Persistence Test ===\n\n");

    // Initial key data (like after Salsa20 in the miner)
    uint8_t initial_key[256];
    for (int i = 0; i < 256; i++) {
        initial_key[i] = (uint8_t)(i * 0x47 + 0x13);
    }

    // Initialize both implementations
    RC4_KEY openssl_key;
    rc4_avx512::FastRc4 fast_key;

    printf("=== Phase 1: Initial key setup + encryption ===\n");

    uint8_t openssl_data[256];
    uint8_t fast_data[256];
    memcpy(openssl_data, initial_key, 256);
    memcpy(fast_data, initial_key, 256);

    // Initial setup (like lines 985-1000 in astrobwtv3.cpp)
    RC4_set_key(&openssl_key, 256, openssl_data);
    RC4(&openssl_key, 256, openssl_data, openssl_data);

    fast_key.set_key(fast_data, 256);
    fast_key.apply_keystream_256(fast_data);

    bool match1 = memcmp(openssl_data, fast_data, 256) == 0;
    print_hex("OpenSSL after init", openssl_data, 16);
    print_hex("FastRc4 after init", fast_data, 16);
    printf("Match after initial setup: %s\n\n", match1 ? "YES" : "NO");

    // Print internal state
    printf("FastRc4 state after init: i=%u, j=%u\n", fast_key.i, fast_key.j);

    printf("=== Phase 2: Multiple RC4 encryptions WITHOUT key reset ===\n");
    printf("(simulating wolfCompute where A <= 0x40 triggers RC4)\n\n");

    // Now call RC4 multiple times WITHOUT resetting the key
    // This is what happens in wolfCompute when A <= 0x40
    bool all_match = true;
    for (int iter = 0; iter < 10; iter++) {
        // Prepare test data for this iteration
        uint8_t test_openssl[256];
        uint8_t test_fast[256];
        for (int i = 0; i < 256; i++) {
            test_openssl[i] = (uint8_t)(i + iter);
            test_fast[i] = (uint8_t)(i + iter);
        }

        // Apply RC4 WITHOUT resetting key
        RC4(&openssl_key, 256, test_openssl, test_openssl);
        fast_key.apply_keystream_256(test_fast);

        bool match = memcmp(test_openssl, test_fast, 256) == 0;
        printf("Iteration %d: %s | FastRc4 state: i=%u, j=%u\n",
               iter, match ? "MATCH" : "MISMATCH", fast_key.i, fast_key.j);

        if (!match) {
            all_match = false;
            print_hex("  OpenSSL", test_openssl, 16);
            print_hex("  FastRc4", test_fast, 16);
        }
    }

    printf("\n=== Phase 3: Key reset mid-stream ===\n");
    printf("(simulating wolfCompute when op >= 254)\n\n");

    // Reset key (like when op >= 254)
    uint8_t new_key[256];
    for (int i = 0; i < 256; i++) {
        new_key[i] = (uint8_t)(i ^ 0xAA);
    }

    RC4_set_key(&openssl_key, 256, new_key);
    fast_key.set_key(new_key, 256);

    printf("After key reset: FastRc4 state: i=%u, j=%u\n", fast_key.i, fast_key.j);

    // Continue encryption after reset
    for (int iter = 0; iter < 5; iter++) {
        uint8_t test_openssl[256];
        uint8_t test_fast[256];
        for (int i = 0; i < 256; i++) {
            test_openssl[i] = (uint8_t)(i + iter + 100);
            test_fast[i] = (uint8_t)(i + iter + 100);
        }

        RC4(&openssl_key, 256, test_openssl, test_openssl);
        fast_key.apply_keystream_256(test_fast);

        bool match = memcmp(test_openssl, test_fast, 256) == 0;
        printf("After reset iter %d: %s | FastRc4 state: i=%u, j=%u\n",
               iter, match ? "MATCH" : "MISMATCH", fast_key.i, fast_key.j);

        if (!match) {
            all_match = false;
            print_hex("  OpenSSL", test_openssl, 16);
            print_hex("  FastRc4", test_fast, 16);
        }
    }

    printf("\n=== Summary ===\n");
    printf("All tests: %s\n", all_match ? "PASSED" : "FAILED");

    return all_match ? 0 : 1;
}
