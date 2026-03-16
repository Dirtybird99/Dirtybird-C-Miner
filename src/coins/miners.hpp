#pragma once

#include "dirtybird-common.hpp"

#include <net.hpp>

#include <num.h>
#include <hex.h>
#include <endian.hpp>
#include <terminal.hpp>
#include <cstring>

using byte = unsigned char;

inline Num ConvertDifficultyToBig(Num d, int algo)
{
  // DERO Miner: Only AstroBWTv3 supported
  switch(algo) {
    case ALGO_ASTROBWTV3:
      return oneLsh256 / d;
    default:
      return 0;
  }
}

inline bool CheckHash(unsigned char *hash, int64_t diff, int algo)
{
  if (littleEndian()) std::reverse(hash, hash+32);
  bool cmp = Num(hexStr(hash, 32).c_str(), 16) <= ConvertDifficultyToBig(diff, algo);
  if (littleEndian()) std::reverse(hash, hash+32);
  return (cmp);
}

inline bool CheckHash(unsigned char *hash, Num diff, int algo)
{
  if (littleEndian()) std::reverse(hash, hash+32);
  bool cmp = Num(hexStr(hash, 32).c_str(), 16) <= diff;
  if (littleEndian()) std::reverse(hash, hash+32);
  return (cmp);
}

// Convert a Num (up to 256-bit) into a 32-byte big-endian target for direct comparison.
// The Num stores words in little-endian order (words[0] = least significant).
inline void NumToTarget32(const Num &n, unsigned char target[32])
{
  // If value exceeds 256 bits, all hashes pass (set max target)
  if (n.size() > 4) {
    std::memset(target, 0xFF, 32);
    return;
  }
  std::memset(target, 0, 32);
  // Num.words[] are uint64_t, words[0] = least significant
  for (size_t i = 0; i < n.size(); ++i) {
    uint64_t w = n[i];
    // Place word i into bytes [24-i*8 .. 31-i*8] in big-endian
    int base = 31 - static_cast<int>(i) * 8;
    for (int b = 0; b < 8 && base - b >= 0; ++b) {
      target[base - b] = static_cast<unsigned char>(w & 0xFF);
      w >>= 8;
    }
  }
}

// Zero-allocation hash check: compare SHA256 output directly against a 32-byte
// big-endian target, without hexStr() heap allocation or Num bignum parsing.
// Preserves the original CheckHash semantics: reverse to little-endian, compare, reverse back.
// Returns true if hash (interpreted as little-endian 256-bit int) <= target.
inline bool CheckHashBytes(const unsigned char *hash, const unsigned char target[32])
{
  // Compare hash (little-endian) against target (big-endian) without mutating hash.
  // Reads hash bytes in reverse order to simulate std::reverse + memcmp.
  for (int i = 0; i < 32; i++) {
    unsigned char h = hash[31 - i], t = target[i];
    if (h != t) return h < t;
  }
  return true;
}

inline std::string uint32ToHex(uint32_t value) {
  std::stringstream ss;
  ss << std::hex << std::setw(8) << std::setfill('0') << value;
  return ss.str();
}

static inline void unsupportedCPU(int tid) {
  printf("This coin is not supported on CPUs\n");
}

static inline void unsupportedGpu(int tid) {
  printf("This coin is not supported on GPUs\n");
}

// DERO mining function (AstroBWTv3)
void mineDero(int tid);

// Cache-Focused Batched DERO mining (k1's optimization)
// Uses staged batch processing to keep SA code hot in L1I cache
void mineDero_Batched(int tid);

// Hybrid mining - automatically selects best approach via benchmark
void mineDero_Hybrid(int tid);

// Interleaved mining - DeroLuna-style "two miners per thread"
// Uses ILP to hide L3 cache latency by interleaving two hash computations
void mineDero_Interleaved(int tid);

// Lock-free mining - Go-style thread coordination
// Uses atomic operations instead of mutex for job polling
void mineDero_LockFree(int tid);

// Mining mode selection
enum DeroMiningMode {
  DERO_MINE_STANDARD = 0,    // Standard sequential mining
  DERO_MINE_BATCHED = 1,     // Cache-focused batched mining
  DERO_MINE_HYBRID = 2,      // Auto-select via benchmark
  DERO_MINE_INTERLEAVED = 3, // Two miners per thread (DeroLuna-style)
  DERO_MINE_LOCKFREE = 4     // Lock-free job polling (Go-style)
};

// Global mining mode (default: standard for compatibility)
inline DeroMiningMode g_deroMiningMode = DERO_MINE_STANDARD;

typedef void (*mineFunc)(int);

inline mineFunc getMiningFunc(int algoNum, bool gpu) {
  // DERO Miner: Only AstroBWTv3 supported
  #ifdef DERO_HIP
  if(gpu) {
    // Future: Add AstroBWTv3 GPU implementation
    return unsupportedGpu;
  }
  #endif

  switch(algoNum) {
    case ALGO_ASTROBWTV3:
      // Select mining mode based on configuration
      switch(g_deroMiningMode) {
        case DERO_MINE_BATCHED:
          return mineDero_Batched;
        case DERO_MINE_HYBRID:
          return mineDero_Hybrid;
        case DERO_MINE_INTERLEAVED:
          return mineDero_Interleaved;
        case DERO_MINE_LOCKFREE:
          return mineDero_LockFree;
        case DERO_MINE_STANDARD:
        default:
          return mineDero;
      }
    default:
      return unsupportedCPU;
  }
}
