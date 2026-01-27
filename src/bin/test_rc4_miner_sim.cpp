/**
 * Test RC4 in the actual miner workflow pattern
 * Simulates the exact sequence of RC4 operations that happen in astrobwtv3
 */

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <openssl/rc4.h>
#include "crypto/astrobwtv3/rc4_avx512.hpp"

// Simulate the miner's RC4 usage pattern
void run_miner_simulation(bool use_fast_rc4, int iterations) {
    printf("Testing %s RC4 (%d iterations)...\n",
           use_fast_rc4 ? "FastRc4" : "OpenSSL", iterations);

    RC4_KEY openssl_key;
    rc4_avx512::FastRc4 fast_key;

    // Scratch buffer like the miner uses
    uint8_t sData[71680] = {0};  // ASTRO_SCRATCH_SIZE = 71040, plus some padding

    // Initialize scratch with test data
    for (int i = 0; i < 256; i++) {
        sData[i] = (uint8_t)(i * 0x47 + 0x13);
    }

    // Phase 1: Initial key setup (like branchHash lines 777-784)
    if (use_fast_rc4) {
        rc4_avx512::fast_rc4_set_key(fast_key, 256, sData);
        rc4_avx512::fast_rc4(fast_key, 256, sData, sData);
    } else {
        RC4_set_key(&openssl_key, 256, sData);
        RC4(&openssl_key, 256, sData, sData);
    }

    printf("After initial setup, sData[0..15]: ");
    for (int i = 0; i < 16; i++) printf("%02x ", sData[i]);
    printf("\n");

    // Track timing
    auto start = std::chrono::high_resolution_clock::now();

    // Simulate the main loop (278 iterations like wolfCompute)
    uint64_t prev_lhash = 0x1234567890abcdef;
    uint64_t lhash = 0xfedcba0987654321;
    uint16_t tries = 0;

    for (int iter = 0; iter < iterations; iter++) {
        // Reset for each "hash" attempt
        tries = 0;

        for (int it = 0; it < 278; it++) {
            tries++;
            uint64_t random_switcher = prev_lhash ^ lhash ^ tries;
            uint8_t op = (uint8_t)random_switcher;

            int chunk_offset = (tries - 1) * 256;
            uint8_t* chunk = &sData[chunk_offset];
            uint8_t* prev_chunk = (tries == 1) ? chunk : &sData[chunk_offset - 256];

            // Copy prev_chunk to chunk (like the miner does)
            if (tries > 1) {
                memcpy(chunk, prev_chunk, 256);
            }

            // RC4 key setup when op >= 254 (rare: ~0.8% of iterations)
            if (op >= 254) {
                if (use_fast_rc4) {
                    rc4_avx512::fast_rc4_set_key(fast_key, 256, prev_chunk);
                } else {
                    RC4_set_key(&openssl_key, 256, prev_chunk);
                }
            }

            // Simulate some wolfPermute work
            for (int i = 0; i < 32; i++) {
                chunk[i] ^= prev_chunk[i];
            }

            // RC4 encryption when A <= 0x40 (25% of iterations)
            uint8_t A = (chunk[0] - chunk[255]) & 0xFF;
            if (A <= 0x40) {
                if (use_fast_rc4) {
                    rc4_avx512::fast_rc4(fast_key, 256, chunk, chunk);
                } else {
                    RC4(&openssl_key, 256, chunk, chunk);
                }
            }

            // Update hash values
            prev_lhash = lhash + prev_lhash;
            lhash = lhash ^ chunk[0] ^ ((uint64_t)chunk[1] << 8);  // Simplified hash
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    printf("Final sData[0..15]: ");
    for (int i = 0; i < 16; i++) printf("%02x ", sData[i]);
    printf("\n");
    printf("Time: %ld us (%d iterations)\n\n", (long)duration.count(), iterations);
}

int main() {
    printf("=== RC4 Miner Simulation Test ===\n\n");

    const int ITERATIONS = 100;

    // Run with OpenSSL RC4
    run_miner_simulation(false, ITERATIONS);

    // Run with FastRc4
    run_miner_simulation(true, ITERATIONS);

    // Now compare outputs more precisely
    printf("=== Deterministic Comparison ===\n\n");

    // Use same seed for both
    uint8_t initial_data[256];
    for (int i = 0; i < 256; i++) {
        initial_data[i] = (uint8_t)(i ^ 0x55);
    }

    uint8_t sData_openssl[71680];
    uint8_t sData_fast[71680];
    memcpy(sData_openssl, initial_data, 256);
    memcpy(sData_fast, initial_data, 256);

    RC4_KEY openssl_key;
    rc4_avx512::FastRc4 fast_key;

    // Identical initial setup
    RC4_set_key(&openssl_key, 256, sData_openssl);
    RC4(&openssl_key, 256, sData_openssl, sData_openssl);

    fast_key.set_key(sData_fast, 256);
    fast_key.apply_keystream_256(sData_fast);

    // Check if initial outputs match
    bool initial_match = memcmp(sData_openssl, sData_fast, 256) == 0;
    printf("After initial setup - Match: %s\n", initial_match ? "YES" : "NO");

    if (!initial_match) {
        printf("First differences:\n");
        for (int i = 0; i < 256; i++) {
            if (sData_openssl[i] != sData_fast[i]) {
                printf("  [%3d] openssl=0x%02x fast=0x%02x\n",
                       i, sData_openssl[i], sData_fast[i]);
                if (i > 5) {
                    printf("  ...\n");
                    break;
                }
            }
        }
        return 1;
    }

    // Simulate wolfCompute loop
    uint64_t prev_lhash = 0x1234567890abcdef;
    uint64_t lhash = 0xfedcba0987654321;
    uint16_t tries = 0;

    bool all_match = true;
    for (int it = 0; it < 278; it++) {
        tries++;
        uint64_t random_switcher = prev_lhash ^ lhash ^ tries;
        uint8_t op = (uint8_t)random_switcher;

        int chunk_offset = (tries - 1) * 256;

        // Copy for both
        if (tries > 1) {
            memcpy(&sData_openssl[chunk_offset], &sData_openssl[chunk_offset - 256], 256);
            memcpy(&sData_fast[chunk_offset], &sData_fast[chunk_offset - 256], 256);
        }

        uint8_t* chunk_openssl = &sData_openssl[chunk_offset];
        uint8_t* chunk_fast = &sData_fast[chunk_offset];
        uint8_t* prev_openssl = (tries == 1) ? chunk_openssl : &sData_openssl[chunk_offset - 256];
        uint8_t* prev_fast = (tries == 1) ? chunk_fast : &sData_fast[chunk_offset - 256];

        // RC4 key setup
        if (op >= 254) {
            RC4_set_key(&openssl_key, 256, prev_openssl);
            fast_key.set_key(prev_fast, 256);
        }

        // Permutation
        for (int i = 0; i < 32; i++) {
            chunk_openssl[i] ^= prev_openssl[i];
            chunk_fast[i] ^= prev_fast[i];
        }

        // RC4 encryption
        uint8_t A_openssl = (chunk_openssl[0] - chunk_openssl[255]) & 0xFF;
        uint8_t A_fast = (chunk_fast[0] - chunk_fast[255]) & 0xFF;

        if (A_openssl <= 0x40) {
            RC4(&openssl_key, 256, chunk_openssl, chunk_openssl);
        }
        if (A_fast <= 0x40) {
            fast_key.apply_keystream_256(chunk_fast);
        }

        // Check match
        if (memcmp(chunk_openssl, chunk_fast, 256) != 0) {
            printf("Mismatch at iteration %d:\n", it);
            printf("  op=%u, A_openssl=%u, A_fast=%u\n", op, A_openssl, A_fast);
            for (int i = 0; i < 16; i++) {
                if (chunk_openssl[i] != chunk_fast[i]) {
                    printf("  [%d] openssl=0x%02x fast=0x%02x\n",
                           i, chunk_openssl[i], chunk_fast[i]);
                }
            }
            all_match = false;
            break;
        }

        // Update hashes
        prev_lhash = lhash + prev_lhash;
        lhash = lhash ^ chunk_openssl[0] ^ ((uint64_t)chunk_openssl[1] << 8);
    }

    if (all_match) {
        printf("All 278 iterations MATCH!\n");
    }

    return all_match ? 0 : 1;
}
