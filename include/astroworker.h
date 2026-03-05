#ifndef astroworker
#define astroworker

#include <bitset>
#include <stdint.h>
#include <vector>

// Salsa20 implementation selector
// USE_SIMD_SALSA20: Use AVX2 SIMD Salsa20 for ~2-5% speedup (tested working)
// Note: ucstk::Salsa20 is always included for struct layout compatibility with SPSA
#define USE_SIMD_SALSA20 0  // Scalar Salsa20: less AVX2 heat for sustained mining

#include "Salsa20.h"           // Always needed for struct layout (SPSA binary compat)
#if USE_SIMD_SALSA20
#include "salsa20_simd.h"      // SIMD functions for actual processing
#endif

#include <openssl/sha.h>
#include <openssl/rc4.h>
#include "crypto/astrobwtv3/rc4_avx512.hpp"
#include "rc4_cryptogams.hpp"

// USE_CRYPTOGAMS_RC4_DUAL: Enable dual RC4 state optimization
// CRYPTOGAMS RC4 is used for encryption (25% of iterations)
// OpenSSL RC4_KEY is maintained for SPSA S-box access
// ENABLED: Phase 1 optimization
#define USE_CRYPTOGAMS_RC4_DUAL 0  // Single RC4 KSA: saves ~70 KSAs/hash for less heat

// USE_SELECTIVE_MEMCPY: Enable selective copy optimization in wolfCompute
// PERMANENTLY DISABLED: wolfPermute reads from chunk[p1:p2) before/during writes,
// so we MUST copy the entire 256-byte chunk. The optimization premise was wrong:
// we assumed wolfPermute only writes to [p1,p2) but it reads from chunk too.
// Investigation complete - this optimization is not possible for AstroBWTv3.
#define USE_SELECTIVE_MEMCPY 0

// USE_DEROBWT: Enable DeroBWT suffix array implementation
// Custom tiered fingerprint sorting optimized for AstroBWTv3 workload
// DISABLED: Testing if this causes regression from 28 KH/s to 10 KH/s
#define USE_DEROBWT 0

// USE_LOOKUP_TABLES: Enable lookup table optimization for branch operations
// Replaces compute-intensive AVX2/AVX512 SIMD with memory lookups (~17MB tables)
// Expected: Reduced thermal throttling due to memory-bound vs compute-bound workload
// Can be overridden via cmake: -DCMAKE_CXX_FLAGS="-DUSE_LOOKUP_TABLES=0"
#ifndef USE_LOOKUP_TABLES
#define USE_LOOKUP_TABLES 1
#endif

// USE_WORKER_SA_BUCKETS:
// 1 = keep bA/bB arrays inside workerData (legacy behavior)
// 0 = use thread-local SA bucket scratch (reduces workerData by ~257KB)
#ifndef USE_WORKER_SA_BUCKETS
#define USE_WORKER_SA_BUCKETS 0
#endif

#define DERO_BATCH 1
// USE_FAST_RC4: Disabled - OpenSSL RC4 is ~12% faster than FastRc4.
// Testing showed: FastRc4 = 12.89 KH/s, OpenSSL = 14.41 KH/s (16 threads)
// The struct layout was fixed (fast_rc4_key moved to end), but it's not beneficial.
#define USE_FAST_RC4 0
#define MAX_LENGTH ((256 * 277) - 1) // this is the maximum
#define ASTRO_SCRATCH_SIZE ((MAX_LENGTH + 64))
#define MAX_RUN_LEN 48
#define MAX_RUNS_PER_BUCKET 128


const int branchedOps_size = 104; // Manually counted size of branchedOps_global
const int regOps_size = 256-branchedOps_size; // 256 - branchedOps_global.size()

//--------------------------------------------------------//

typedef unsigned char byte;

typedef struct templateMarker {
  uint8_t p1;
  uint8_t p2;
  uint16_t keySpotA;
  uint16_t keySpotB;
  uint16_t posData;
} templateMarker;

typedef struct zeroRun {
  int32_t startPos;
  uint8_t len;
} zeroRun;

const uint8_t iota8_g[256] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255};

class workerData
{
public:
  // For aarch64
  byte aarchFixup[256];
  byte opt[256];
  // byte simpleLookup[regOps_size*(256*256)];
  // byte lookup3D[branchedOps_size*256*256];
  // uint16_t lookup2D[regOps_size*(256*256)];
  // std::bitset<256> clippedBytes[regOps_size];
  // std::bitset<256> unchangedBytes[regOps_size];
  // std::bitset<256> isBranched;

  // byte branchedOps[branchedOps_size*2];
  // byte regularOps[regOps_size*2];

  // byte branched_idx[256];
  // byte reg_idx[256];

  int lucky = 0;

  SHA256_CTX sha256;
  uint8_t sha_padding[64];
  ucstk::Salsa20 salsa20;
  RC4_KEY key[DERO_BATCH];
  // NOTE: fast_rc4_key and cryptogams_rc4 moved to end of struct to preserve SPSA-compatible layout

  // std::vector<std::tuple<int,int,int>> repeats;

  byte salsaInput[256] = {0};
  byte op;

  byte A;
  uint32_t data_len;

  byte *chunk;
  byte *prev_chunk;

  byte maskTable_bytes[32*33];
  byte padding[32];

  bool isSame = false;

  #if !defined(USE_ASTRO_SPSA) && USE_WORKER_SA_BUCKETS
  int bA[256];
  int bB[256*256];
  #endif

  byte step_3[256];
  byte sData[ASTRO_SCRATCH_SIZE*DERO_BATCH];

  byte pos1;
  byte pos2;
  byte t1;
  byte t2;

  uint64_t random_switcher;
  uint64_t lhash;
  uint64_t prev_lhash;
  uint16_t tries[DERO_BATCH];

  #if defined(USE_ASTRO_SPSA)
  std::bitset<MAX_LENGTH> isInZeroRun;
  zeroRun zeroRuns[277*2];
  uint16_t zeroRunCount = 0;
  uint32_t sa_prelim[120*256+1] = {(uint32_t)-1};
  #endif
  int32_t sa[277*256+1];

  templateMarker astroTemplate[277];
  int templateIdx = 0;

  std::bitset<554> isBSlice;
  // NOTE: iota8 required for SPSA library struct layout compatibility (cannot remove)
  uint8_t iota8[256] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255};
  uint8_t stampKeys[554];
  // int stampStarts[554] = {0};
  // int modifiedBytes[MODBUFFER] = {0};
  uint8_t stampTemplates[277];

  #if defined(USE_ASTRO_SPSA)
  // SPSA-only bucket state. Keep these arrays only when SPSA is compiled in.
  // This avoids carrying ~896KB/worker in non-SPSA builds.
  uint16_t buckets_d[256][256];
  uint32_t bHeads[256][256];
  uint32_t bHeadIdx[256][256];
  #endif

#if USE_CRYPTOGAMS_RC4_DUAL
  // CryptogamsRc4 placed at end to preserve SPSA-compatible struct layout
  // Used for fast encryption (25% of iterations), OpenSSL RC4_KEY kept for SPSA S-box access
  rc4_cryptogams::CryptogamsRc4 cryptogams_rc4[DERO_BATCH];
#endif

#if USE_FAST_RC4
  // FastRc4 placed at end to preserve SPSA-compatible struct layout
  rc4_avx512::FastRc4 fast_rc4_key[DERO_BATCH];
#endif

  uint8_t end_padding[64];

  // std::bitset<277*256> isIn;
  // std::bitset<277*256> isKey;

  // std::vector<byte> opsA;
  // std::vector<byte> opsB;

  friend std::ostream& operator<<(std::ostream& os, const workerData& wd);
};
#endif
