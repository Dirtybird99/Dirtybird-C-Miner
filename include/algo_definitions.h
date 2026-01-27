#pragma once

#define PROTO_DERO_SOLO 0
#define COIN_UNKNOWN -1
#define COIN_DERO 0
#define COIN_COUNT 1
#define ALGO_UNSUPPORTED 0
#define ALGO_ASTROBWTV3 10

inline const char* algoName(int algo) {
    return algo == ALGO_ASTROBWTV3 ? "AstroBWTv3" : "Unknown";
}

typedef enum { ENDIAN_LITTLE, ENDIAN_BIG, ENDIAN_SWAP_32, ENDIAN_SWAP_32_BE, ENDIAN_MIXED } endian_mode_t;

typedef struct {
    endian_mode_t header_endian;
    bool swap_merkle_root;
    bool swap_prev_hash;
    int nbits_index;
} algo_config_t;

extern algo_config_t current_algo_config;
