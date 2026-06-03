/*
 * difficulty.cpp -- zero-allocation difficulty comparison
 *
 * Target = 2^256 / difficulty, stored as 32-byte big-endian.
 * Uses __uint128_t for long division. No heap allocations.
 * No big integer library. No string conversions.
 *
 * This is what the Go miner does with big.Int and GC pressure.
 * We do it with 35 lines of integer arithmetic.
 */

#include "dluna.h"

void compute_target(int64_t difficulty, uint8_t target[32])
{
    memset(target, 0, 32);
    if (difficulty <= 0) {
        memset(target, 0xFF, 32);
        return;
    }

    /* 2^256 as a 33-byte big-endian number: [1, 0, 0, ..., 0] */
    uint8_t dividend[33] = {0};
    dividend[0] = 1;

    __uint128_t rem = 0;
    uint64_t d = (uint64_t)difficulty;

    for (int i = 0; i < 33; i++) {
        rem = (rem << 8) | dividend[i];
        if (i > 0)
            target[i - 1] = (uint8_t)(uint64_t)(rem / d);
        rem = rem % d;
    }
}

bool check_hash(const uint8_t hash[32], const uint8_t target[32])
{
    /* hash is little-endian (SHA256), target is big-endian (MSB first). */
    for (int i = 0; i < 32; i++) {
        uint8_t h = hash[31 - i];
        uint8_t t = target[i];
        if (h < t) return true;
        if (h > t) return false;
    }
    return true; /* exact match */
}
