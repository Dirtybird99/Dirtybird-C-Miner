/**
 * Test RC4 behavior using actual workerData struct
 * This tests if the struct layout or malloc_huge_pages affects RC4 behavior
 */
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <openssl/rc4.h>
#include "crypto/astrobwtv3/astrobwtv3.h"

void print_hex(const char* label, const uint8_t* data, size_t len) {
    printf("%s: ", label);
    for (size_t i = 0; i < len && i < 32; i++) {
        printf("%02x", data[i]);
    }
    if (len > 32) printf("...");
    printf("\n");
}

int main() {
    printf("=== Worker RC4 Test ===\n");
    printf("USE_CRYPTOGAMS_RC4_DUAL = %d\n", USE_CRYPTOGAMS_RC4_DUAL);
    printf("\n");

    // Allocate worker like the miner does
    workerData* worker = (workerData*)malloc_huge_pages(sizeof(workerData));
    if (!worker) {
        printf("Failed to allocate worker\n");
        return 1;
    }
    printf("Worker allocated at %p\n", (void*)worker);

    // Initialize like the miner does
    initWorker(*worker);
    printf("Worker initialized\n");

#if USE_CRYPTOGAMS_RC4_DUAL
    // Check the state of cryptogams_rc4 after initWorker
    printf("\nCRYPTOGAMS RC4 state after initWorker:\n");
    printf("  cryptogams_rc4[0] at %p\n", (void*)&worker->cryptogams_rc4[0]);
    printf("  i=%u, j=%u\n", worker->cryptogams_rc4[0].i, worker->cryptogams_rc4[0].j);
    printf("  S[0..7] = ");
    for (int i = 0; i < 8; i++) printf("%02x ", worker->cryptogams_rc4[0].S[i]);
    printf("\n");

    // Verify S-box is identity
    bool sbox_ok = true;
    for (int i = 0; i < 256; i++) {
        if (worker->cryptogams_rc4[0].S[i] != (uint8_t)i) {
            printf("  ERROR: S[%d]=%u (expected %d)\n", i, worker->cryptogams_rc4[0].S[i], i);
            sbox_ok = false;
            break;
        }
    }
    printf("  S-box identity check: %s\n", sbox_ok ? "PASS" : "FAIL");
#endif

    // Now test the prep phase flow
    printf("\n=== Testing Prep Phase ===\n");

    // Initialize scratch like the miner (simulating Salsa20 output)
    uint8_t scratch[384] = {0};
    for (int i = 0; i < 256; i++) {
        scratch[i] = (uint8_t)(i * 0x47 + 0x13);
    }
    print_hex("Before RC4", scratch, 32);

    // Reference: OpenSSL only
    uint8_t ref_scratch[384];
    memcpy(ref_scratch, scratch, 384);
    RC4_KEY ref_key;
    RC4_set_key(&ref_key, 256, ref_scratch);
    RC4(&ref_key, 256, ref_scratch, ref_scratch);
    print_hex("OpenSSL reference", ref_scratch, 32);

    // Test: Using worker's RC4 states
#if USE_CRYPTOGAMS_RC4_DUAL
    worker->cryptogams_rc4[0].set_key(scratch, 256);
    RC4_set_key(&worker->key[0], 256, scratch);
    worker->cryptogams_rc4[0].apply_keystream_256(scratch);
    print_hex("Worker CRYPTOGAMS", scratch, 32);
    printf("  After set_key: i=%u, j=%u\n", worker->cryptogams_rc4[0].i, worker->cryptogams_rc4[0].j);
#else
    RC4_set_key(&worker->key[0], 256, scratch);
    RC4(&worker->key[0], 256, scratch, scratch);
    print_hex("Worker OpenSSL", scratch, 32);
#endif

    bool prep_match = memcmp(ref_scratch, scratch, 256) == 0;
    printf("Prep phase match: %s\n", prep_match ? "PASS" : "FAIL");

    if (!prep_match) {
        printf("\nFirst differences:\n");
        int diff_count = 0;
        for (int i = 0; i < 256; i++) {
            if (ref_scratch[i] != scratch[i]) {
                printf("  [%3d] ref=0x%02x worker=0x%02x\n", i, ref_scratch[i], scratch[i]);
                if (++diff_count >= 10) {
                    printf("  ... more differences\n");
                    break;
                }
            }
        }
    }

    // Test wolfCompute-like encryption
    printf("\n=== Testing wolfCompute Encryption ===\n");

    // Initialize test chunks
    uint8_t ref_chunk[256], worker_chunk[256];
    for (int i = 0; i < 256; i++) {
        ref_chunk[i] = (uint8_t)(i ^ 0xAA);
        worker_chunk[i] = ref_chunk[i];
    }

    // Encrypt with continuation (no key reset)
    RC4(&ref_key, 256, ref_chunk, ref_chunk);
#if USE_CRYPTOGAMS_RC4_DUAL
    worker->cryptogams_rc4[0].apply_keystream_256(worker_chunk);
#else
    RC4(&worker->key[0], 256, worker_chunk, worker_chunk);
#endif

    print_hex("Reference chunk", ref_chunk, 32);
    print_hex("Worker chunk", worker_chunk, 32);

    bool chunk_match = memcmp(ref_chunk, worker_chunk, 256) == 0;
    printf("Continuation encryption match: %s\n", chunk_match ? "PASS" : "FAIL");

    // Test with key reset (simulating op >= 254)
    printf("\n=== Testing Key Reset (op >= 254) ===\n");

    // Use encrypted scratch as new key
    RC4_set_key(&ref_key, 256, ref_scratch);
#if USE_CRYPTOGAMS_RC4_DUAL
    worker->cryptogams_rc4[0].set_key(scratch, 256);
    RC4_set_key(&worker->key[0], 256, scratch);  // For SPSA
#else
    RC4_set_key(&worker->key[0], 256, scratch);
#endif

    // Reset and encrypt new chunks
    for (int i = 0; i < 256; i++) {
        ref_chunk[i] = (uint8_t)(i ^ 0x55);
        worker_chunk[i] = ref_chunk[i];
    }

    RC4(&ref_key, 256, ref_chunk, ref_chunk);
#if USE_CRYPTOGAMS_RC4_DUAL
    worker->cryptogams_rc4[0].apply_keystream_256(worker_chunk);
#else
    RC4(&worker->key[0], 256, worker_chunk, worker_chunk);
#endif

    print_hex("Reference (after reset)", ref_chunk, 32);
    print_hex("Worker (after reset)", worker_chunk, 32);

    bool reset_match = memcmp(ref_chunk, worker_chunk, 256) == 0;
    printf("Key reset encryption match: %s\n", reset_match ? "PASS" : "FAIL");

    printf("\n=== Summary ===\n");
    printf("Prep phase: %s\n", prep_match ? "PASS" : "FAIL");
    printf("Continuation: %s\n", chunk_match ? "PASS" : "FAIL");
    printf("After reset: %s\n", reset_match ? "PASS" : "FAIL");

    return (prep_match && chunk_match && reset_match) ? 0 : 1;
}
