/**
 * Test workerData malloc allocation + initWorker initialization
 * Verifies that FastRc4 is properly initialized when using malloc
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <openssl/rc4.h>
#include "astroworker.h"
#include "astrobwtv3/astrobwtv3.h"

void print_hex(const char* label, const uint8_t* data, size_t len) {
    printf("%s: ", label);
    for (size_t i = 0; i < len && i < 16; i++) {
        printf("%02x", data[i]);
    }
    if (len > 16) printf("...");
    printf("\n");
}

int main() {
    printf("=== Worker Initialization Test ===\n");
    printf("USE_FAST_RC4 = %d\n", USE_FAST_RC4);
    printf("DERO_BATCH = %d\n\n", DERO_BATCH);

    // Allocate with malloc like mine_dero.cpp does
    workerData* worker = static_cast<workerData*>(std::malloc(sizeof(workerData)));
    if (!worker) {
        printf("Failed to allocate workerData\n");
        return 1;
    }

    printf("sizeof(workerData) = %zu bytes\n", sizeof(workerData));

#if USE_FAST_RC4
    // Check FastRc4 state BEFORE initWorker (should be garbage)
    printf("\nBEFORE initWorker:\n");
    printf("  fast_rc4_key[0].i = %u\n", worker->fast_rc4_key[0].i);
    printf("  fast_rc4_key[0].j = %u\n", worker->fast_rc4_key[0].j);
    printf("  fast_rc4_key[0].s[0..7] = ");
    for (int i = 0; i < 8; i++) printf("%02x ", worker->fast_rc4_key[0].s[i]);
    printf("\n");
#endif

    // Call initWorker
    printf("\nCalling initWorker...\n");
    initWorker(*worker);

#if USE_FAST_RC4
    // Check FastRc4 state AFTER initWorker (should be identity)
    printf("\nAFTER initWorker:\n");
    printf("  fast_rc4_key[0].i = %u (expected 0)\n", worker->fast_rc4_key[0].i);
    printf("  fast_rc4_key[0].j = %u (expected 0)\n", worker->fast_rc4_key[0].j);
    printf("  fast_rc4_key[0].s[0..7] = ");
    for (int i = 0; i < 8; i++) printf("%02x ", worker->fast_rc4_key[0].s[i]);
    printf(" (expected 00 01 02 03 04 05 06 07)\n");

    // Verify S-box is identity permutation
    bool identity_ok = true;
    for (int i = 0; i < 256; i++) {
        if (worker->fast_rc4_key[0].s[i] != (uint8_t)i) {
            printf("  S-box mismatch at [%d]: got %d, expected %d\n",
                   i, worker->fast_rc4_key[0].s[i], i);
            identity_ok = false;
            break;
        }
    }
    printf("  S-box identity: %s\n", identity_ok ? "OK" : "FAIL");
#endif

    // Now test that RC4 produces correct output
    printf("\n=== RC4 Correctness After Init ===\n");

    uint8_t test_key[256];
    uint8_t test_data_openssl[256];
    uint8_t test_data_fast[256];

    // Create test key
    for (int i = 0; i < 256; i++) {
        test_key[i] = (uint8_t)(i * 0x47 + 0x13);
        test_data_openssl[i] = (uint8_t)(i ^ 0xAA);
        test_data_fast[i] = (uint8_t)(i ^ 0xAA);
    }

    // Test OpenSSL RC4
    RC4_set_key(&worker->key[0], 256, test_key);
    RC4(&worker->key[0], 256, test_data_openssl, test_data_openssl);

#if USE_FAST_RC4
    // Test FastRc4
    worker->fast_rc4_key[0].set_key(test_key, 256);
    worker->fast_rc4_key[0].apply_keystream_256(test_data_fast);

    // Compare
    bool match = memcmp(test_data_openssl, test_data_fast, 256) == 0;
    print_hex("OpenSSL output", test_data_openssl, 16);
    print_hex("FastRc4 output", test_data_fast, 16);
    printf("Match: %s\n", match ? "YES" : "NO");
#else
    print_hex("OpenSSL output", test_data_openssl, 16);
    printf("FastRc4 not enabled\n");
#endif

    std::free(worker);
    return 0;
}
