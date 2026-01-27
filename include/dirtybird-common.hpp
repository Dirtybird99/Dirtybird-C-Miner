#pragma once

#ifndef dirtybirdcommon_hpp
#define dirtybirdcommon_hpp

#include <stdint.h>
#include <vector>
#include <string>
#include <random>
#include <map>

#include <boost/program_options.hpp>

#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/asio/spawn.hpp>
#include <boost/asio/ssl/error.hpp>
#include <boost/asio/ip/host_name.hpp>

#include <boost/json.hpp>

#include <boost/thread.hpp>
#include <boost/atomic.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/atomic.hpp>
#include <boost/thread.hpp>
#include <boost/tokenizer.hpp>

#include <boost/multiprecision/cpp_int.hpp>

#include <num.h>

#include "algo_definitions.h"

#define XSTR(x) STR(x)
#define STR(x) #x

extern const char *dirtybirdTargetArch;

#define CMP_LT_U256(X, Y) (X[3] != Y[3] ? X[3] < Y[3] : X[2] != Y[2] ? X[2] < Y[2] \
                                                                            : X[1] != Y[1]   ? X[1] < Y[1] \
                                                                                                           : X[0] < Y[0])

extern bool ABORT_MINER;

#define CHECK_CLOSE if (ABORT_MINER) return;
#define CHECK_CLOSE_RET(s) if (ABORT_MINER) return s;

extern double latest_hashrate;

extern bool lockThreads;
extern bool g_pcores_only;  // Limit mining to P-cores only (for hybrid CPUs like i7-13700HX)
extern int g_pcore_count;   // Number of P-core logical processors detected
extern int threads;
extern int g_omp_threads;  // OpenMP threads per mining thread (0 = auto)

extern std::string workerName;
extern std::string workerNameFromWallet;
extern std::string stratumPassword;
extern bool useLookupMine;

extern bool gpuMine;

extern std::map<int, int> threadToPhysicalCore;
extern std::mutex threadMapMutex;

typedef struct {
    int coinId;
    int miningAlgo;
    std::string coinSymbol;
    std::string coinPrettyName;
} Coin;

const Coin unknownCoin = {COIN_UNKNOWN, ALGO_UNSUPPORTED, "unknown", "unknown"};
const Coin coins[COIN_COUNT] = {
  {COIN_DERO,     ALGO_ASTROBWTV3,  "DERO",     "Dero"}
};

// ============================================================================
// Algorithm Versioning Support
// ============================================================================

struct AlgoVersion {
  int algo;
  std::string displayName;  // For pretty printing, e.g., "v3"
};

struct VersionedAlgo {
  std::string coinSymbol;
  int defaultAlgo;
  std::map<std::string, AlgoVersion> versions;  // Key: "V1", "1", etc.
};

// Define versioned algorithms - DERO Miner has no versioned algorithms
inline const std::vector<VersionedAlgo>& getVersionedAlgos() {
  static const std::vector<VersionedAlgo> versionedAlgos = {};
  return versionedAlgos;
}

struct CoinParseResult {
  std::string baseSymbol;
  int algoOverride;           // -1 if no version specified
  std::string versionDisplay; // e.g., "v3" for display purposes
  bool isVersioned;           // True if this was a versioned coin match
};

// Parse coin symbol with optional version suffix
// Supports: XEL, XEL-V3, XEL=V3, XELV3, XEL-3, XEL=3, XEL3
inline CoinParseResult parseCoinWithVersion(const std::string& input) {
  CoinParseResult result = {"", -1, "", false};
  
  std::string upperInput = input;
  std::transform(upperInput.begin(), upperInput.end(), upperInput.begin(), ::toupper);
  
  const auto& versionedAlgos = getVersionedAlgos();
  
  for (const auto& va : versionedAlgos) {
    // Exact match (base symbol only) - use default
    if (upperInput == va.coinSymbol) {
      result.baseSymbol = va.coinSymbol;
      result.algoOverride = va.defaultAlgo;
      result.isVersioned = true;
      
      // Find display name for default
      for (const auto& [key, ver] : va.versions) {
        if (ver.algo == va.defaultAlgo && key.length() == 2) {  // Prefer "V3" over "3"
          result.versionDisplay = ver.displayName;
          break;
        }
      }
      return result;
    }
    
    // Try different separator patterns: -, =, or none
    const std::vector<std::string> separators = {"-", "=", ""};
    
    for (const auto& sep : separators) {
      std::string prefix = va.coinSymbol + sep;
      
      if (upperInput.length() > prefix.length() &&
          upperInput.substr(0, prefix.length()) == prefix) {
        
        std::string versionPart = upperInput.substr(prefix.length());
        
        auto it = va.versions.find(versionPart);
        if (it != va.versions.end()) {
          result.baseSymbol = va.coinSymbol;
          result.algoOverride = it->second.algo;
          result.versionDisplay = it->second.displayName;
          result.isVersioned = true;
          return result;
        }
      }
    }
  }
  
  // Not a versioned coin - return base symbol for regular lookup
  result.baseSymbol = upperInput;
  result.isVersioned = false;
  return result;
}

// Find coin by symbol (case-insensitive)
inline const Coin* findCoinBySymbol(const std::string& symbol) {
  std::string upperSymbol = symbol;
  std::transform(upperSymbol.begin(), upperSymbol.end(), upperSymbol.begin(), ::toupper);
  
  for (int i = 0; i < COIN_COUNT; i++) {
    std::string coinSymbol = coins[i].coinSymbol;
    std::transform(coinSymbol.begin(), coinSymbol.end(), coinSymbol.begin(), ::toupper);
    if (coinSymbol == upperSymbol) {
      return &coins[i];
    }
  }
  return nullptr;
}

// Get display string for algorithm versions available for a coin
inline std::string getVersionsHelpString(const std::string& symbol) {
  std::string upperSymbol = symbol;
  std::transform(upperSymbol.begin(), upperSymbol.end(), upperSymbol.begin(), ::toupper);
  
  const auto& versionedAlgos = getVersionedAlgos();
  for (const auto& va : versionedAlgos) {
    if (va.coinSymbol == upperSymbol) {
      std::string help = "Available versions: ";
      std::set<std::string> displayed;
      for (const auto& [key, ver] : va.versions) {
        if (key.length() == 2 && displayed.find(ver.displayName) == displayed.end()) {
          if (displayed.size() > 0) help += ", ";
          help += ver.displayName;
          if (ver.algo == va.defaultAlgo) help += " (default)";
          displayed.insert(ver.displayName);
        }
      }
      return help;
    }
  }
  return "";
}

// ============================================================================
// MiningProfile class
// ============================================================================

class MiningProfile {
  public:
    MiningProfile() {
      coin = unknownCoin;
    };
    ~MiningProfile() {}

    Coin coin;
    int protocol;
    std::string host;
    std::string port;
    std::string wallet;
    std::string workerName;
    std::string transportLayer;
    bool useStratum = false;
    bool doShutdown;

    void setPoolAddress(std::string hst) {
      this->host = hst;
      boost::char_separator<char> sep(":");
      boost::tokenizer<boost::char_separator<char>> tok(hst, sep);
      std::vector<std::string> tokens;
      std::copy(tok.begin(), tok.end(), std::back_inserter<std::vector<std::string> >(tokens));
      if(tokens.size() == 2) {
        this->host = tokens[0];
        try {
          const int i{std::stoi(tokens[1])};
          this->port = tokens[1];
        }
        catch (...) {
          this->transportLayer = tokens[0];
          this->host = tokens[1];
        }
      } else if(tokens.size() == 3) {
        this->transportLayer = tokens[0];
        this->host = tokens[1];
        this->port = tokens[2];
      }
      boost::replace_all(this->host, "/", "");
      if (this->transportLayer.size() > 0) {
        if (this->transportLayer.find("stratum") != std::string::npos) this->useStratum = true;
        // DERO-only: no xatum protocol support
      }
      this->setProtocol();
    }

    void setProtocol() {
      // DERO-only: AstroBWTv3 protocol
      this->protocol = PROTO_DERO_SOLO;
    }
    
    // Set coin with optional version override
    void setCoin(const Coin& c, int algoOverride = -1) {
      this->coin = c;
      if (algoOverride != -1) {
        this->coin.miningAlgo = algoOverride;
      }
    }
};

extern MiningProfile miningProfile;

extern Num oneLsh256;      
extern Num maxU256;

extern boost::multiprecision::uint256_t bigDiff;

extern int batchSize;

extern int jobCounter;
extern int reportCounter;
extern int reportInterval;

extern int blockCounter;
extern int miniBlockCounter;
extern int rejected;
extern int accepted;

extern int64_t ourHeight;

extern int nonceLen;

extern int64_t difficulty;

extern uint64_t nonce0;

extern double doubleDiff;

extern int HIP_deviceCount;
extern std::string HIP_names[32];
extern std::string HIP_pcieID[32];
extern std::vector<std::atomic<uint64_t>> HIP_kIndex;
extern std::vector<std::atomic<uint64_t>> HIP_counters;
extern std::vector<std::atomic<uint64_t>> HIP_counters;
extern std::vector<std::vector<int64_t>> HIP_rates5min;
extern std::vector<std::vector<int64_t>> HIP_rates1min;
extern std::vector<std::vector<int64_t>> HIP_rates30sec;

extern std::vector<int64_t> rate5min;
extern std::vector<int64_t> rate1min;
extern std::vector<int64_t> rate30sec;

extern std::atomic<int64_t> counter;
extern std::atomic<int64_t> benchCounter;

extern bool isConnected;

extern bool beQuiet;

extern boost::asio::io_context my_context;
extern boost::asio::steady_timer update_timer;
extern boost::asio::steady_timer mine_duration_timer;
extern std::chrono::time_point<std::chrono::steady_clock> g_start_time;
extern int mine_time;

inline std::string cpp_int_toHex(boost::multiprecision::cpp_int in) {
  std::ostringstream oss;
  oss << std::hex << in;
  return oss.str();
}

inline void cpp_int_to_byte_array(const boost::multiprecision::uint256_t &num, uint8_t *out) {  
  for (size_t i = 0; i < 32; ++i) {
    out[i] = static_cast<uint8_t>(num >> (i * 8) & 0xFF);
  }
}

inline void cpp_int_to_be_byte_array(const boost::multiprecision::uint256_t &num, uint8_t *out) {  
  for (size_t i = 0; i < 32; ++i) {
    out[i] = static_cast<uint8_t>(num >> ((32 - i - 1) * 8) & 0xFF);
  }
}

inline int randomSleepTimeMs(int low=9000, int high=11000) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distr(low, high);
  return distr(gen);
}

#endif