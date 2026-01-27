#pragma once

#include "dirtybird-common.hpp"

#include <net.hpp>

#include <num.h>
#include <hex.h>
#include <endian.hpp>
#include <terminal.hpp>

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
