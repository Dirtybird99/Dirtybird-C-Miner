#pragma once

// DERO Miner - Algorithm Definitions
// DERO-only: AstroBWTv3

// Protocol IDs
#define PROTO_DERO_SOLO 0

// Coin IDs
#define COIN_UNKNOWN -1
#define COIN_DERO 0
#define COIN_COUNT 1

// Algorithm IDs
#define ALGO_UNSUPPORTED 0
#define ALGO_ASTROBWTV3 10

inline const char* algoName(int algo) {
  switch(algo) {
    case ALGO_ASTROBWTV3:
      return "AstroBWTv3";
    default:
      return "Unknown";
  }
}

typedef enum {
  ENDIAN_LITTLE,
  ENDIAN_BIG,
  ENDIAN_SWAP_32,
  ENDIAN_SWAP_32_BE,
  ENDIAN_MIXED
} endian_mode_t;

typedef struct {
  endian_mode_t header_endian;
  bool swap_merkle_root;
  bool swap_prev_hash;
  int nbits_index;
} algo_config_t;

extern algo_config_t current_algo_config;

#define CONFIG_ENDIAN_SHA256 0
#define CONFIG_ENDIAN_SCRYPT 1
#define CONFIG_ENDIAN_X11 2
#define CONFIG_ENDIAN_YESPOWER 3

// AstroBWT algorithms (DERO only)
static const int astroAlgos[] = {
  ALGO_ASTROBWTV3
};

static const algo_config_t algo_configs[] = {
  // Bitcoin/SHA256
  { .header_endian = ENDIAN_SWAP_32, .swap_merkle_root = true, .swap_prev_hash = true, .nbits_index = 18 },
  // Scrypt
  { .header_endian = ENDIAN_SWAP_32_BE, .swap_merkle_root = false, .swap_prev_hash = true, .nbits_index = 18 },
  // X11
  { .header_endian = ENDIAN_LITTLE, .swap_merkle_root = false, .swap_prev_hash = false, .nbits_index = 18 },
  // YESPOWER
  { .header_endian = ENDIAN_SWAP_32, .swap_merkle_root = true, .swap_prev_hash = false, .nbits_index = 18 },
};
