#include "miner_main.h"
#include "dirtybird-common.hpp"
#include "dirtybird-hugepages.hpp"
#include "numa_optimizer.hpp"
#include "thermal_governor.hpp"
#include "msr.hpp"
#include "gpulibs.h"
#include "hipkill.h"

#include "rootcert.h"
#include <DNSResolver.hpp>
#include "net.hpp"

#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <numeric>
#include <sstream>

#include "miner.h"

#include <random>

#include <hex.h>
#include "algos.h"
#include <thread>

#ifdef DIRTYBIRD_OPENMP
#include <omp.h>
#endif

#include <chrono>
#include <cmath>
#include <cstring>
#include <future>
#include <limits>
#include <libcubwt.cuh>

#include <openssl/err.h>
#include <openssl/ssl.h>
#include <base64.hpp>

#include <bit>
#include <broadcastServer.hpp>
#include <stratum/stratum.h>

#include <exception>

#if defined(_WIN32)
#include <timeapi.h>
#include <cstdio>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#  if defined(_MSC_VER)
#    include <intrin.h>
#  else
#    include <cpuid.h>
#  endif
#endif

#include "reporter.hpp"

#include <coins/miners.hpp>
#include <coins/dero_worker_pool.hpp>
#include <astrobwtv3/cache_batching.hpp>
#include <astrobwtv3/interleaved_miner.hpp>
#include <astrobwtv3/lookup_full.hpp>
#include <lookup_mode.hpp>
#include <dirtybird_hip/core/devInfo.hip.h>
#include <boost/algorithm/string.hpp>
#include <boost/json.hpp>

#ifdef DIRTYBIRD_HIP
#include <boost/json/src.hpp>
#include <dirtybird_hip/common/gpu_rtc.hpp>
#include <dirtybird_hip/common/gpu_algo.hpp>
#endif

// DERO Miner: Non-DERO algorithm includes removed

#if defined(USE_ASTRO_SPSA)
  #include "spsa.hpp"
#endif

#if USE_LOOKUP_TABLES
  #include "lookup_tables.hpp"
#endif

// DERO Miner: uint256_t for difficulty calculations
using boost::multiprecision::uint256_t;

// INITIALIZE COMMON STUFF
algo_config_t current_algo_config;

int reportCounter = 0;
int reportInterval = 1;
int threads = 0;

bool ABORT_MINER = false;
double latest_hashrate = 0.0;

bool gpuMine = false;
bool printHashrateOnExit = false;
std::string wallet = "NULL";

int HIP_deviceCount = 0;

uint256_t bigDiff(0);

uint64_t nonce0 = 0;

std::string HIP_names[32];
std::string HIP_pcieID[32];
std::vector<std::atomic<uint64_t>> HIP_kIndex(32);
std::vector<std::atomic<uint64_t>> HIP_counters(32);
std::vector<std::vector<int64_t>> HIP_rates5min(32);
std::vector<std::vector<int64_t>> HIP_rates1min(32);
std::vector<std::vector<int64_t>> HIP_rates30sec(32);

alignas(64) std::atomic<int64_t> counter = 0;
alignas(64) boost::asio::io_context my_context;
boost::asio::steady_timer update_timer = boost::asio::steady_timer(my_context);
std::chrono::time_point<std::chrono::steady_clock> g_start_time = std::chrono::steady_clock::now();
int mine_time = 0;
boost::asio::steady_timer mine_duration_timer = boost::asio::steady_timer(my_context);
bool printHugepagesError = true;

Num oneLsh256 = Num(1) << 256;
Num maxU256 = Num(2).pow(256) - 1;

const auto processor_count = std::thread::hardware_concurrency();
#ifdef DIRTYBIRD_HIP
GPUTuningOverrides g_tuning_overrides;
std::atomic<bool> g_mining_started{false};
#endif

/* Start definitions from dirtybird-common.hpp */

MiningProfile miningProfile = MiningProfile();

int batchSize = 5000;

int jobCounter;

int blockCounter;
int miniBlockCounter;
int rejected;
int accepted;

#if defined(_WIN32)
bool lockThreads = true;
#else
bool lockThreads = false;
#endif
bool g_no_per_thread_affinity = false;
bool g_pcores_only = false;
  // Limit mining to P-cores only (for hybrid CPUs)
int g_pcore_count = 0;       // Number of P-core logical processors detected
int g_differential_affinity = 0;  // Differential affinity mode (0=default, 1=P-first, 2=physical-only, 3=balanced)
bool g_show_affinity = false;     // Show detailed core assignments
#if defined(_WIN32)
std::vector<uint32_t> g_pcore_logical_ids;  // Logical core IDs for P-cores only
std::vector<uint32_t> g_ecore_logical_ids;  // Logical core IDs for E-cores
int detectPCores();  // Forward declaration
#endif
int g_omp_threads = 0;  // OpenMP threads per mining thread (0 = auto)
bool g_no_power_override = false;  // --no-power-override: disable all power management
bool g_is_laptop = false;          // Auto-detected: system has a battery
bool g_on_battery_power = false;   // Auto-detected: battery-capable system is off AC power

// Adaptive thread management (DeroLuna-style over-provisioning)
std::atomic<int64_t> thread_counters[MAX_MINING_THREADS] = {};
std::atomic<bool> thread_stop[MAX_MINING_THREADS] = {};
bool g_adaptive_threads = false;
int g_adaptive_warmup_secs = 15;
int g_overprovision_count = 0;

#if defined(_WIN32)
// Detect whether the system is battery-capable and whether it is currently off AC power.
static void detectPowerState() {
    g_is_laptop = false;
    g_on_battery_power = false;

    SYSTEM_POWER_STATUS sps;
    if (GetSystemPowerStatus(&sps)) {
        // BatteryFlag 128 = no battery, 255 = unknown.
        g_is_laptop = (sps.BatteryFlag != 128 && sps.BatteryFlag != 255);
        // ACLineStatus: 0 = offline, 1 = online, 255 = unknown.
        // Treat unknown as battery mode on mobile systems so we fail safe.
        g_on_battery_power = g_is_laptop && sps.ACLineStatus != 1;
    }
}

// Power plan save/restore
static char g_saved_power_scheme[64] = {0};

static void savePowerScheme() {
    FILE* pipe = _popen("powercfg /getactivescheme 2>nul", "r");
    if (pipe) {
        char buf[256];
        if (fgets(buf, sizeof(buf), pipe)) {
            char* start = strchr(buf, ':');
            if (start) {
                start += 2;
                char* end = strchr(start, ' ');
                if (end) {
                    size_t len = end - start;
                    if (len < sizeof(g_saved_power_scheme)) {
                        strncpy(g_saved_power_scheme, start, len);
                        g_saved_power_scheme[len] = '\0';
                    }
                }
            }
        }
        _pclose(pipe);
    }
}

static void applyHighPerformancePower() {
    savePowerScheme();

    if (g_on_battery_power) {
        // Battery power: use Balanced power plan with moderate tuning
        // Avoids draining the battery and fighting the platform's battery policy.
        system("powercfg /setactive 381b4222-f694-41f0-9685-ff5bb260df2e >nul 2>&1");
        // Allow full range but don't force minimum to 100%
        system("powercfg -setacvalueindex scheme_current sub_processor PROCTHROTTLEMIN 5 >nul 2>&1");
        system("powercfg -setacvalueindex scheme_current sub_processor PROCTHROTTLEMAX 100 >nul 2>&1");
        // Moderate boost
        system("powercfg -setacvalueindex scheme_current sub_processor PERFBOOSTMODE 2 >nul 2>&1");
        system("powercfg -setacvalueindex scheme_current sub_processor PERFBOOSTPOL 60 >nul 2>&1");
        // EPP = 128 (balanced) instead of 0 (max perf) — lets CPU manage thermals
        system("powercfg -setacvalueindex scheme_current sub_processor PERFEPP 128 >nul 2>&1");
        system("powercfg -setacvalueindex scheme_current sub_processor PERFINCPOL 2 >nul 2>&1");
        system("powercfg -setacvalueindex scheme_current sub_processor PERFDECPOL 1 >nul 2>&1");
        system("powercfg -setactive scheme_current >nul 2>&1");

        setcolor(CYAN);
        printf("Power: Battery power detected - using balanced profile (EPP=128)\n");
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
    } else {
        // AC power: use the aggressive performance path on both desktops and plugged-in laptops.
        system("powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c >nul 2>&1");
        system("powercfg -setacvalueindex scheme_current sub_processor PROCTHROTTLEMIN 100 >nul 2>&1");
        system("powercfg -setacvalueindex scheme_current sub_processor PROCTHROTTLEMAX 100 >nul 2>&1");
        system("powercfg -setacvalueindex scheme_current sub_processor PERFBOOSTMODE 2 >nul 2>&1");
        system("powercfg -setacvalueindex scheme_current sub_processor PERFBOOSTPOL 100 >nul 2>&1");
        system("powercfg -setacvalueindex scheme_current sub_processor PERFEPP 0 >nul 2>&1");
        system("powercfg -setacvalueindex scheme_current sub_processor PERFINCPOL 2 >nul 2>&1");
        system("powercfg -setacvalueindex scheme_current sub_processor PERFDECPOL 1 >nul 2>&1");
        system("powercfg -setactive scheme_current >nul 2>&1");

        if (g_is_laptop) {
            setcolor(CYAN);
            printf("Power: AC power detected on battery-capable system - using performance profile (EPP=0)\n");
            fflush(stdout);
            setcolor(BRIGHT_WHITE);
        }
    }
}

static void restorePowerScheme() {
    if (g_saved_power_scheme[0]) {
        char cmd[128];
        snprintf(cmd, sizeof(cmd), "powercfg /setactive %s >nul 2>&1", g_saved_power_scheme);
        system(cmd);
    }
}

// MSR-based HWP optimization (per-core)
static void applyMSRPerformanceOptimizations() {
    // Try to initialize MSR access via WinRing0
    if (!initMSRAccess()) return;

    int numCores = std::thread::hardware_concurrency();

    // Read and report RAPL power limits
    {
        uint64_t unitRaw = 0;
        if (readMSR(0x606, unitRaw, 0)) {
            double powerUnit = 1.0 / (1 << (unitRaw & 0xF));
            double timeUnit = 1.0 / (1 << ((unitRaw >> 16) & 0xF));

            uint64_t pkgLimit = 0;
            if (readMSR(0x610, pkgLimit, 0)) {
                bool locked = (pkgLimit >> 63) & 1;
                double pl1 = (pkgLimit & 0x7FFF) * powerUnit;
                double pl2 = ((pkgLimit >> 32) & 0x7FFF) * powerUnit;

                uint32_t pl2_time_raw = (pkgLimit >> 49) & 0x7F;
                uint32_t pl2_y = pl2_time_raw & 0x1F;
                uint32_t pl2_z = (pl2_time_raw >> 5) & 0x3;
                double pl2_tau = pow(2.0, pl2_y) * (1.0 + pl2_z / 4.0) * timeUnit;

                setcolor(CYAN);
                printf("RAPL: PL1=%.0fW PL2=%.0fW tau=%.1fs %s\n",
                       pl1, pl2, pl2_tau, locked ? "[LOCKED]" : "[UNLOCKED]");
                fflush(stdout);
                setcolor(BRIGHT_WHITE);

                // On battery power: do NOT override RAPL PL1.
                // On AC power: raise PL1 to match PL2 for stable peak performance.
                if (!locked && !g_on_battery_power) {
                    uint64_t pl2_raw_val = (pkgLimit >> 32) & 0x7FFF;
                    uint64_t newValue = pl2_raw_val          // PL1 power = PL2 power
                                      | (1ULL << 15)         // PL1 enable
                                      | (1ULL << 16)         // PL1 clamp
                                      | (0x7FULL << 17)      // PL1 max time window
                                      | (pl2_raw_val << 32)  // PL2 power unchanged
                                      | (1ULL << 47)         // PL2 enable
                                      | (1ULL << 48)         // PL2 clamp
                                      | (0x7FULL << 49);     // PL2 max time window
                    if (writeMSR(0x610, newValue, 0)) {
                        setcolor(BRIGHT_GREEN);
                        printf("RAPL: Raised PL1 to %.0fW (was %.0fW), max time windows\n", pl2, pl1);
                        fflush(stdout);
                        setcolor(BRIGHT_WHITE);
                    }
                } else if (!locked && g_on_battery_power) {
                    setcolor(CYAN);
                    printf("RAPL: Battery mode - preserving PL1=%.0fW (not raising to PL2)\n", pl1);
                    fflush(stdout);
                    setcolor(BRIGHT_WHITE);
                }
            }
        }
    }

    // Set HWP performance on all cores
    // Battery mode: EPP=128 (balanced), AC mode: EPP=0 (max performance)
    uint8_t epp = g_on_battery_power ? 0x80 : 0x00;
    for (int core = 0; core < numCores; core++) {
        uint64_t caps = 0;
        if (readMSR(0x771, caps, core)) {
            uint8_t highest = caps & 0xFF;
            uint64_t hwp = (uint64_t)highest
                         | ((uint64_t)highest << 8)
                         | ((uint64_t)highest << 16)
                         | ((uint64_t)epp << 24);
            writeMSR(0x774, hwp, core);
        }
        // Energy perf bias: battery=8 (balanced), AC=0 (max performance)
        writeMSR(0x1B0, g_on_battery_power ? 0x8 : 0x0, core);
    }

    setcolor(CYAN);
    if (g_on_battery_power) {
        printf("MSR: HWP balanced (EPP=%d) applied to %d cores\n", epp, numCores);
    } else {
        printf("MSR: HWP max performance + EPP=0 applied to %d cores\n", numCores);
    }
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
}
#endif

//static int firstRejected;

//uint64_t hashrate;
int64_t ourHeight;

int nonceLen;

int64_t difficulty;

double doubleDiff = 0;

bool useLookupMine = false;

// Stub definitions for SPSA library compatibility (dev fee removed)
std::string devWallet = "";
double devFee = 0.0;

// SPSA control - default OFF until replay/test parity is restored.
// Use --spsa on to opt back into the library fast path for live tuning.
bool g_use_spsa = false;
bool g_use_local_spsa = true; // Default to the integrated C++ SPSA path; current live tests show it is both correct and faster
bool g_spsa_stamp_fast = true;
bool g_spsa_decode_bases = true;
int g_spsa_bucket_prefetch = 0;
int g_spsa_max_data_len = 0;
bool g_spsa_hit_counters = true;
bool g_spsa_sha_profile = false;
bool g_spsa_sha_pair = false;
bool g_spsa_sha_zeroize = true;  // Enabled by default: current live probes keep the SPSA core slightly faster with zeroize on
bool g_verbose_tune = false;
bool g_array_telemetry = false;
bool g_phase_telemetry = false;
int g_lookup_smart_threshold = 12;
bool g_lookup_smart_telemetry = false;

std::vector<int64_t> rate5min;
std::vector<int64_t> rate1min;
std::vector<int64_t> rate30sec;

std::string workerName = "default";
std::string workerNameFromWallet = "";
std::string stratumPassword = "x";

bool isConnected = false;

bool beQuiet = false;
/* End definitions from dirtybird-common.hpp */

/* Start definitions from astrobwtv3.h */
#if defined(DIRTYBIRD_ASTROBWTV3)
AstroFunc allAstroFuncs[] = {
#if USE_LOOKUP_TABLES
  {"lookup", lookupCompute},  // Memory-bound lookup tables (reduced thermal throttling)
#endif
  {"wolf", wolfCompute},
  {"wolf_memopt", wolfCompute_memopt},  // Memory-optimized version with incremental copy
#if defined(__AVX2__)
  // NOTE: branchComputeCPU_avx2_zOptimized is empty stub - disabled
  // {"avx2z", branchComputeCPU_avx2_zOptimized},
#elif defined(__aarch64__)
  {"aarch64", branchComputeCPU_aarch64},
#endif
};
size_t numAstroFuncs;
#endif
/* End definitions from astrobwtv3.h */

namespace net = boost::asio;            // from <boost/asio.hpp>
namespace ssl = boost::asio::ssl;       // from <boost/asio/ssl.hpp>
namespace po = boost::program_options;  // from <boost/program_options.hpp>

boost::mutex mutex;
boost::mutex reportMutex;
boost::mutex jobMutex;  // Protects job-related globals (ourHeight, isConnected, etc.)

uint8_t *lookup1D_global;  // Regular ops: 152 x 256 = 38 KB (L1 cache)
byte *lookup3D_global;     // Optional branched table: 104 x 256 x 256 = 6.5 MB
bool g_print_runtime_config = false;
std::string g_runtime_parity = "off";
bool g_cpu_auto_profile = true;
std::string g_auto_default_astro_kernel = "wolf_memopt";
static const char* lookupModeToString(LookupMode mode);

using byte = unsigned char;

namespace {

struct CpuRuntimeProfileSummary {
  std::string vendor;
  std::string vendor_name;
  std::string brand;
  unsigned logical_threads = 0;
  int p_logical = 0;
  int e_logical = 0;
  bool hybrid = false;
  bool avx2 = false;
  bool avx512f = false;
  bool aes = false;
  bool sha = false;
};

bool optionExplicit(const po::variables_map& vm, const char* name) {
  auto it = vm.find(name);
  return it != vm.end() && !it->second.defaulted();
}

std::string collapseWhitespace(std::string value) {
  std::istringstream iss(value);
  std::ostringstream oss;
  std::string token;
  bool first = true;
  while (iss >> token) {
    if (!first) {
      oss << ' ';
    }
    oss << token;
    first = false;
  }
  return oss.str();
}

void cpuidQuery(uint32_t level, uint32_t count, uint32_t* abcd) {
#if defined(__x86_64__) || defined(_M_X64)
#  if defined(_MSC_VER)
  int regs[4] = {};
  __cpuidex(regs, static_cast<int>(level), static_cast<int>(count));
  for (int i = 0; i < 4; ++i) {
    abcd[i] = static_cast<uint32_t>(regs[i]);
  }
#  else
  uint32_t a = 0;
  uint32_t b = 0;
  uint32_t c = 0;
  uint32_t d = 0;
  __cpuid_count(level, count, a, b, c, d);
  abcd[0] = a;
  abcd[1] = b;
  abcd[2] = c;
  abcd[3] = d;
#  endif
#else
  abcd[0] = abcd[1] = abcd[2] = abcd[3] = 0;
  (void)level;
  (void)count;
#endif
}

std::string detectCpuVendorString() {
#if defined(__x86_64__) || defined(_M_X64)
  uint32_t abcd[4] = {};
  cpuidQuery(0, 0, abcd);
  char vendor[13] = {};
  std::memcpy(vendor, &abcd[1], 4);
  std::memcpy(vendor + 4, &abcd[3], 4);
  std::memcpy(vendor + 8, &abcd[2], 4);
  vendor[12] = '\0';
  return collapseWhitespace(vendor);
#elif defined(__aarch64__) || defined(_M_ARM64)
  return "ARM64";
#else
  return dirtybirdTargetArch;
#endif
}

std::string detectCpuBrandString() {
#if defined(__x86_64__) || defined(_M_X64)
  uint32_t abcd[4] = {};
  cpuidQuery(0x80000000U, 0, abcd);
  if (abcd[0] < 0x80000004U) {
    return detectCpuVendorString();
  }

  char brand[49] = {};
  for (uint32_t i = 0; i < 3; ++i) {
    cpuidQuery(0x80000002U + i, 0, abcd);
    std::memcpy(brand + (i * 16), abcd, sizeof(abcd));
  }
  brand[48] = '\0';
  return collapseWhitespace(brand);
#else
  return detectCpuVendorString();
#endif
}

std::string normalizeCpuVendorName(const std::string& vendor) {
  if (vendor == "GenuineIntel") {
    return "Intel";
  }
  if (vendor == "AuthenticAMD") {
    return "AMD";
  }
  if (vendor == "ARM64") {
    return "ARM64";
  }
  if (!vendor.empty()) {
    return vendor;
  }
  return dirtybirdTargetArch;
}

CpuRuntimeProfileSummary detectCpuRuntimeProfileSummary() {
  CpuRuntimeProfileSummary cpu{};
  cpu.vendor = detectCpuVendorString();
  cpu.vendor_name = normalizeCpuVendorName(cpu.vendor);
  cpu.brand = detectCpuBrandString();
  cpu.logical_threads = processor_count > 0 ? processor_count : 1;

#if defined(__x86_64__) || defined(_M_X64)
  cpu.avx2 = __builtin_cpu_supports("avx2");
  cpu.avx512f = __builtin_cpu_supports("avx512f");
  cpu.aes = __builtin_cpu_supports("aes");
#endif
  cpu.sha = has_sha_ni_support_cached();

#if defined(_WIN32)
  if (g_pcore_count == 0) {
    detectPCores();
  }
  cpu.p_logical = g_pcore_count;
  cpu.e_logical = static_cast<int>(g_ecore_logical_ids.size());
  cpu.hybrid = cpu.p_logical > 0 && cpu.e_logical > 0;
#endif

  return cpu;
}

void applyCpuRuntimeProfile(const po::variables_map& vm, const CpuRuntimeProfileSummary& cpu) {
  if (!g_cpu_auto_profile || g_runtime_parity != "off") {
    return;
  }

  const bool explicit_kernel =
      optionExplicit(vm, "lookup") ||
      optionExplicit(vm, "no-tune") ||
      optionExplicit(vm, "auto-tune") ||
      optionExplicit(vm, "interleaved") ||
      optionExplicit(vm, "lockfree") ||
      optionExplicit(vm, "cache-batch") ||
      optionExplicit(vm, "cache-batch-hybrid");

  if (!explicit_kernel && cpu.avx2) {
    useLookupMine = true;
    g_auto_default_astro_kernel = "lookup";
    if (!optionExplicit(vm, "lookup-mode")) {
      g_lookup_mode = LOOKUP_MODE_3D;
    }
  }

  if (!optionExplicit(vm, "spsa") && !optionExplicit(vm, "no-spsa")) {
    g_use_spsa = true;
  }

  if (!optionExplicit(vm, "local-spsa") && !optionExplicit(vm, "library-spsa")) {
    g_use_local_spsa = true;
  }
}

std::string buildCpuSummaryLine(const CpuRuntimeProfileSummary& cpu) {
  std::ostringstream oss;
  oss << (cpu.vendor_name.empty() ? cpu.vendor : cpu.vendor_name)
      << " | " << cpu.logical_threads << " threads";

  if (cpu.hybrid) {
    oss << " | hybrid P" << cpu.p_logical << "/E" << cpu.e_logical;
  }

  if (!cpu.brand.empty() &&
      cpu.brand != cpu.vendor &&
      cpu.brand != cpu.vendor_name) {
    oss << " | " << cpu.brand;
  }

  return oss.str();
}

std::string buildRuntimeSummaryLine() {
  std::ostringstream oss;
  oss << "kernel=" << getCurrentAstroAlgoName();
  if (useLookupMine) {
    oss << "/" << lookupModeToString(g_lookup_mode);
  }
  oss << " | spsa=";
  if (g_use_spsa) {
    oss << (g_use_local_spsa ? "local" : "library");
  } else {
    oss << "off";
  }
  oss << " | threads=" << threads;
#ifdef DIRTYBIRD_OPENMP
  oss << " | omp=" << (g_omp_threads > 0 ? g_omp_threads : get_auto_omp_threads_for_mining(threads > 0 ? threads : 1));
#endif
#if defined(_WIN32)
  if (!lockThreads) {
    oss << " | affinity=off";
  } else if (g_no_per_thread_affinity) {
    oss << " | affinity=process";
  } else {
    oss << " | affinity=per-thread";
  }
#else
  oss << " | affinity=" << (lockThreads ? "on" : "off");
#endif
  return oss.str();
}

} // namespace

//------------------------------------------------------------------------------

void openssl_log_callback(const SSL *ssl, int where, int ret)
{
  if (ret <= 0)
  {
    int error = SSL_get_error(ssl, ret);
    char errbuf[256];
    ERR_error_string_n(error, errbuf, sizeof(errbuf));
    std::cerr << "OpenSSL Error: " << errbuf << std::endl;
  }
}

//------------------------------------------------------------------------------

#if defined(DIRTYBIRD_ASTROBWTV3)
void initializeExterns() {
  numAstroFuncs = std::size(allAstroFuncs); //sizeof(allAstroFuncs)/sizeof(allAstroFuncs[0]);
}
#endif

inline void preserveAlgoOverride(MiningProfile& profile, int coinId) {
  int savedAlgo = profile.coin.miningAlgo;
  bool hadOverride = (savedAlgo != coins[coinId].miningAlgo && savedAlgo != ALGO_UNSUPPORTED);
  
  profile.coin = coins[coinId];
  
  if (hadOverride) {
    profile.coin.miningAlgo = savedAlgo;
  }
}

int enhanceWallet(MiningProfile *currentProfile, bool checkWallet) {
  // DERO Miner: Only DERO wallet validation
  if(checkWallet) {
    if (currentProfile->coin.coinSymbol == "DERO" && !(currentProfile->wallet.find("der") == std::string::npos || currentProfile->wallet.find("det") == std::string::npos))
    {
      std::cout << "Provided wallet address is not valid for Dero" << std::endl;
      return EXIT_FAILURE;
    }
  }

  if(currentProfile->wallet.find("dero", 0) != std::string::npos) {
    preserveAlgoOverride(*currentProfile, COIN_DERO);
  }
  return EXIT_SUCCESS;
}


void hipKill() {
  #ifdef DIRTYBIRD_HIP
  hipDeviceReset_wrapper();
  #endif
}

void onExit() {
  #if defined(_WIN32)
  timeEndPeriod(1);
  restorePowerScheme();
  #endif
  hipKill();
  ABORT_MINER = true;
  setcolor(BRIGHT_WHITE);
  if(printHashrateOnExit) {
    printf("\n\n%s: %d threads @ %2.2f with %d miniblocks and %d blocks (built with ",
           miningProfile.coin.coinPrettyName.c_str(), threads, latest_hashrate, accepted, blockCounter);
#ifdef __clang__
    std::cout << "Clang "
              << __clang_major__ << "."
              << __clang_minor__ << "."
              << __clang_patchlevel__ << ")" << std::endl;
#elif defined(__GNUC__)
    std::cout << "GCC "
              << __GNUC__ << "."
              << __GNUC_MINOR__ << "."
              << __GNUC_PATCHLEVEL__ << ")" << std::endl;
#else
    std::cout << "Unknown compiler" << std::endl;
#endif
  }

  printf("\nExiting Miner...\n");
  fflush(stdout);

  //boost::this_thread::sleep_for(boost::chrono::seconds(1));
  //fflush(stdout);

#if defined(_WIN32)
  SetConsoleMode(hInput, ENABLE_EXTENDED_FLAGS | (prev_mode));
#endif
}

void sigterm(int signum) {
  std::cout << "\n\nTerminate signal (" << signum << ") received." << std::flush;
  exit(signum);
}

void sigint(int signum) {
  std::cout << "\n\nInterrupt signal (" << signum << ") received." << std::flush;
  exit(signum);
}

#ifdef DIRTYBIRD_HIP
void parse_gpu_batch_sizes(const std::string& arg) {
    std::stringstream ss(arg);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            g_tuning_overrides.gpu_batch_sizes.push_back(0);
        } else {
            try {
                g_tuning_overrides.gpu_batch_sizes.push_back(std::stoul(item));
            } catch (const std::exception& e) {
                fprintf(stderr, "Invalid GPU batch size value '%s': %s\n", item.c_str(), e.what());
                g_tuning_overrides.gpu_batch_sizes.push_back(0);
            }
        }
    }
}

// Parse --gpu-block-sizes=64,128,64
void parse_gpu_block_sizes(const std::string& arg) {
    std::stringstream ss(arg);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            g_tuning_overrides.gpu_block_sizes.push_back(0);
        } else {
            try {
                g_tuning_overrides.gpu_block_sizes.push_back(std::stoi(item));
            } catch (const std::exception& e) {
                fprintf(stderr, "Invalid GPU block size value '%s': %s\n", item.c_str(), e.what());
                g_tuning_overrides.gpu_block_sizes.push_back(0);
            }
        }
    }
}

// Parse --no-autotune
void disable_autotune() {
    g_tuning_overrides.disable_autotune = true;
}
#endif

// Load configuration from config.json file
// Returns true if config was loaded, false otherwise
// CLI arguments take priority over config file values
bool loadConfig(const std::string& configPath, po::variables_map& vm) {
    std::ifstream file(configPath);
    if (!file.is_open()) {
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();

    boost::system::error_code ec;
    auto configValue = boost::json::parse(buffer.str(), ec);
    if (ec) {
        setcolor(YELLOW);
        std::cerr << "Warning: Failed to parse " << configPath << ": " << ec.message() << std::endl;
        setcolor(BRIGHT_WHITE);
        return false;
    }

    if (!configValue.is_object()) {
        setcolor(YELLOW);
        std::cerr << "Warning: " << configPath << " is not a valid JSON object" << std::endl;
        setcolor(BRIGHT_WHITE);
        return false;
    }

    auto& obj = configValue.as_object();

    // Helper lambda to insert string value into vm if not already set by CLI
    auto insertString = [&](const char* key) {
        if (obj.contains(key) && !vm.count(key)) {
            if (obj.at(key).is_string()) {
                std::string val(obj.at(key).as_string());
                vm.insert(std::make_pair(key, po::variable_value(val, false)));
            }
        }
    };

    // Helper lambda to insert int value into vm if not already set by CLI
    auto insertInt = [&](const char* key) {
        if (obj.contains(key) && !vm.count(key)) {
            if (obj.at(key).is_int64()) {
                int val = static_cast<int>(obj.at(key).as_int64());
                vm.insert(std::make_pair(key, po::variable_value(val, false)));
            }
        }
    };

    // Helper lambda to insert bool value into vm if not already set by CLI
    auto insertBool = [&](const char* key) {
        if (obj.contains(key) && !vm.count(key)) {
            if (obj.at(key).is_bool()) {
                bool val = obj.at(key).as_bool();
                vm.insert(std::make_pair(key, po::variable_value(val, false)));
            }
        }
    };

    // Helper lambda to insert double value into vm if not already set by CLI
    auto insertDouble = [&](const char* key) {
        if (obj.contains(key) && !vm.count(key)) {
            if (obj.at(key).is_double()) {
                double val = obj.at(key).as_double();
                vm.insert(std::make_pair(key, po::variable_value(val, false)));
            } else if (obj.at(key).is_int64()) {
                double val = static_cast<double>(obj.at(key).as_int64());
                vm.insert(std::make_pair(key, po::variable_value(val, false)));
            }
        }
    };

    // Load string options
    insertString("daemon-address");
    insertString("wallet");
    insertString("worker-name");
    insertString("password");
    insertString("no-tune");
    insertString("sa-prefetch");
    insertString("lookup-mode");
    insertString("spsa");
    insertString("spsa-stamp-fast");
    insertString("spsa-decode-bases");
    insertString("spsa-bucket-prefetch");
    insertString("thermal-yield");
    insertString("runtime-parity");

    // Load integer options
    insertInt("port");
    insertInt("threads");
    insertInt("report-interval");
    insertInt("batch-size");
    insertInt("op");
    insertInt("len");
    insertInt("tune-warmup");
    insertInt("tune-duration");
    insertInt("mine-time");
    insertInt("dero-benchmark");
    insertInt("omp-threads");
    insertInt("pace-interval");
    insertInt("pace-sleep-us");
    insertInt("adaptive-warmup");
    insertInt("lookup-smart-threshold");

    // Load boolean options (these are flags, so we only set if true in config)
    if (obj.contains("no-lock") && !vm.count("no-lock")) {
        if (obj.at("no-lock").is_bool() && obj.at("no-lock").as_bool()) {
            vm.insert(std::make_pair("no-lock", po::variable_value()));
        }
    }
    if (obj.contains("lock-threads") && !vm.count("no-lock")) {
        // lock-threads: false means --no-lock
        if (obj.at("lock-threads").is_bool() && !obj.at("lock-threads").as_bool()) {
            vm.insert(std::make_pair("no-lock", po::variable_value()));
        }
    }
    if (obj.contains("gpu") && !vm.count("gpu")) {
        if (obj.at("gpu").is_bool() && obj.at("gpu").as_bool()) {
            vm.insert(std::make_pair("gpu", po::variable_value()));
        }
    }
    if (obj.contains("broadcast") && !vm.count("broadcast")) {
        if (obj.at("broadcast").is_bool() && obj.at("broadcast").as_bool()) {
            vm.insert(std::make_pair("broadcast", po::variable_value()));
        }
    }
    if (obj.contains("lookup") && !vm.count("lookup")) {
        if (obj.at("lookup").is_bool() && obj.at("lookup").as_bool()) {
            vm.insert(std::make_pair("lookup", po::variable_value()));
        }
    }
    if (obj.contains("auto-tune") && !vm.count("auto-tune")) {
        if (obj.at("auto-tune").is_bool() && obj.at("auto-tune").as_bool()) {
            vm.insert(std::make_pair("auto-tune", po::variable_value()));
        }
    }
    if (obj.contains("stratum") && !vm.count("stratum")) {
        if (obj.at("stratum").is_bool() && obj.at("stratum").as_bool()) {
            vm.insert(std::make_pair("stratum", po::variable_value()));
        }
    }
    if (obj.contains("quiet") && !vm.count("quiet")) {
        if (obj.at("quiet").is_bool() && obj.at("quiet").as_bool()) {
            vm.insert(std::make_pair("quiet", po::variable_value()));
        }
    }
    if (obj.contains("ignore-wallet") && !vm.count("ignore-wallet")) {
        if (obj.at("ignore-wallet").is_bool() && obj.at("ignore-wallet").as_bool()) {
            vm.insert(std::make_pair("ignore-wallet", po::variable_value()));
        }
    }
    if (obj.contains("sa-tune") && !vm.count("sa-tune")) {
        if (obj.at("sa-tune").is_bool() && obj.at("sa-tune").as_bool()) {
            vm.insert(std::make_pair("sa-tune", po::variable_value()));
        }
    }
    if (obj.contains("no-sa-tune") && !vm.count("no-sa-tune")) {
        if (obj.at("no-sa-tune").is_bool() && obj.at("no-sa-tune").as_bool()) {
            vm.insert(std::make_pair("no-sa-tune", po::variable_value()));
        }
    }
    if (obj.contains("cache-batch") && !vm.count("cache-batch")) {
        if (obj.at("cache-batch").is_bool() && obj.at("cache-batch").as_bool()) {
            vm.insert(std::make_pair("cache-batch", po::variable_value()));
        }
    }
    if (obj.contains("cache-batch-hybrid") && !vm.count("cache-batch-hybrid")) {
        if (obj.at("cache-batch-hybrid").is_bool() && obj.at("cache-batch-hybrid").as_bool()) {
            vm.insert(std::make_pair("cache-batch-hybrid", po::variable_value()));
        }
    }
    if (obj.contains("bench-cache-batch") && !vm.count("bench-cache-batch")) {
        if (obj.at("bench-cache-batch").is_bool() && obj.at("bench-cache-batch").as_bool()) {
            vm.insert(std::make_pair("bench-cache-batch", po::variable_value()));
        }
    }
    if (obj.contains("interleaved") && !vm.count("interleaved")) {
        if (obj.at("interleaved").is_bool() && obj.at("interleaved").as_bool()) {
            vm.insert(std::make_pair("interleaved", po::variable_value()));
        }
    }
    if (obj.contains("bench-interleaved") && !vm.count("bench-interleaved")) {
        if (obj.at("bench-interleaved").is_bool() && obj.at("bench-interleaved").as_bool()) {
            vm.insert(std::make_pair("bench-interleaved", po::variable_value()));
        }
    }
    if (obj.contains("lockfree") && !vm.count("lockfree")) {
        if (obj.at("lockfree").is_bool() && obj.at("lockfree").as_bool()) {
            vm.insert(std::make_pair("lockfree", po::variable_value()));
        }
    }
    if (obj.contains("print-runtime-config") && !vm.count("print-runtime-config")) {
        if (obj.at("print-runtime-config").is_bool() && obj.at("print-runtime-config").as_bool()) {
            vm.insert(std::make_pair("print-runtime-config", po::variable_value()));
        }
    }
    if (obj.contains("array-telemetry") && !vm.count("array-telemetry")) {
        if (obj.at("array-telemetry").is_bool() && obj.at("array-telemetry").as_bool()) {
            vm.insert(std::make_pair("array-telemetry", po::variable_value()));
        }
    }
    if (obj.contains("phase-telemetry") && !vm.count("phase-telemetry")) {
        if (obj.at("phase-telemetry").is_bool() && obj.at("phase-telemetry").as_bool()) {
            vm.insert(std::make_pair("phase-telemetry", po::variable_value()));
        }
    }
    if (obj.contains("lookup-smart-telemetry") && !vm.count("lookup-smart-telemetry")) {
        if (obj.at("lookup-smart-telemetry").is_bool() && obj.at("lookup-smart-telemetry").as_bool()) {
            vm.insert(std::make_pair("lookup-smart-telemetry", po::variable_value()));
        }
    }
    if (obj.contains("local-spsa") && !vm.count("local-spsa")) {
        if (obj.at("local-spsa").is_bool() && obj.at("local-spsa").as_bool()) {
            vm.insert(std::make_pair("local-spsa", po::variable_value()));
        }
    }
    if (obj.contains("library-spsa") && !vm.count("library-spsa")) {
        if (obj.at("library-spsa").is_bool() && obj.at("library-spsa").as_bool()) {
            vm.insert(std::make_pair("library-spsa", po::variable_value()));
        }
    }
    if (obj.contains("spsa") && !vm.count("spsa")) {
        if (obj.at("spsa").is_bool()) {
            vm.insert(std::make_pair(
                "spsa",
                po::variable_value(std::string(obj.at("spsa").as_bool() ? "on" : "off"), false)));
        }
    }
    if (obj.contains("spsa-max-data-len") && !vm.count("spsa-max-data-len")) {
        if (obj.at("spsa-max-data-len").is_int64()) {
            vm.insert(std::make_pair("spsa-max-data-len", po::variable_value(obj.at("spsa-max-data-len").as_int64(), false)));
        } else if (obj.at("spsa-max-data-len").is_uint64()) {
            vm.insert(std::make_pair("spsa-max-data-len", po::variable_value(static_cast<int64_t>(obj.at("spsa-max-data-len").as_uint64()), false)));
        }
    }
    if (obj.contains("no-runtime-config") && !vm.count("no-runtime-config")) {
        if (obj.at("no-runtime-config").is_bool() && obj.at("no-runtime-config").as_bool()) {
            vm.insert(std::make_pair("no-runtime-config", po::variable_value()));
        }
    }
    if (obj.contains("no-cpu-auto") && !vm.count("no-cpu-auto")) {
        if (obj.at("no-cpu-auto").is_bool() && obj.at("no-cpu-auto").as_bool()) {
            vm.insert(std::make_pair("no-cpu-auto", po::variable_value()));
        }
    }

    return true;
}

// Find config.json in executable directory or current working directory
std::string findConfigFile() {
    // Try current working directory first
    if (std::ifstream("config.json").good()) {
        return "config.json";
    }

#if defined(_WIN32)
    // Try executable directory on Windows
    char exePath[MAX_PATH];
    if (GetModuleFileNameA(NULL, exePath, MAX_PATH) > 0) {
        std::string exeDir(exePath);
        size_t lastSlash = exeDir.find_last_of("\\/");
        if (lastSlash != std::string::npos) {
            std::string configPath = exeDir.substr(0, lastSlash + 1) + "config.json";
            if (std::ifstream(configPath).good()) {
                return configPath;
            }
        }
    }
#else
    // Try executable directory on Linux/Mac
    char exePath[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);
    if (len != -1) {
        exePath[len] = '\0';
        std::string exeDir(exePath);
        size_t lastSlash = exeDir.find_last_of('/');
        if (lastSlash != std::string::npos) {
            std::string configPath = exeDir.substr(0, lastSlash + 1) + "config.json";
            if (std::ifstream(configPath).good()) {
                return configPath;
            }
        }
    }
#endif

    return "";  // No config file found
}

static const char* lookupModeToString(LookupMode mode) {
  switch (mode) {
    case LOOKUP_MODE_1D: return "1d";
    case LOOKUP_MODE_3D: return "3d";
    case LOOKUP_MODE_FULL: return "full";
    case LOOKUP_MODE_HYBRID: return "hybrid";
    case LOOKUP_MODE_SMART: return "smart";
    default: return "unknown";
  }
}

static const char* deroModeToString(DeroMiningMode mode) {
  switch (mode) {
    case DERO_MINE_STANDARD: return "standard";
    case DERO_MINE_BATCHED: return "cache-batch";
    case DERO_MINE_HYBRID: return "cache-batch-hybrid";
    case DERO_MINE_INTERLEAVED: return "interleaved";
    case DERO_MINE_LOCKFREE: return "lockfree";
    default: return "unknown";
  }
}

static const char* spsaBucketPrefetchModeToString(int mode) {
  switch (mode) {
    case 0: return "off";
    case 1: return "light";
    case 2: return "full";
    default: return "off";
  }
}

static LookupMode parseLookupMode(const std::string& rawValue) {
  std::string value = rawValue;
  boost::algorithm::to_lower(value);

  if (value == "auto") {
    return LOOKUP_MODE_HYBRID;
  }
  if (value == "1d") {
    return LOOKUP_MODE_1D;
  }
  if (value == "3d") {
    return LOOKUP_MODE_3D;
  }
  if (value == "full") {
    return LOOKUP_MODE_FULL;
  }
  if (value == "hybrid") {
    return LOOKUP_MODE_HYBRID;
  }
  if (value == "smart") {
    return LOOKUP_MODE_SMART;
  }

  throw po::validation_error(po::validation_error::invalid_option_value, "lookup-mode");
}

static int parseSpsaBucketPrefetchMode(const std::string& rawValue) {
  std::string value = rawValue;
  boost::algorithm::to_lower(value);

  if (value == "off" || value == "0") {
    return 0;
  }
  if (value == "light" || value == "1") {
    return 1;
  }
  if (value == "full" || value == "2") {
    return 2;
  }

  throw po::validation_error(po::validation_error::invalid_option_value, "spsa-bucket-prefetch");
}

static bool parseOnOffValue(const std::string& rawValue, const char* optionName) {
  std::string value = rawValue;
  boost::algorithm::to_lower(value);
  if (value == "on" || value == "true" || value == "1") {
    return true;
  }
  if (value == "off" || value == "false" || value == "0") {
    return false;
  }
  throw po::validation_error(po::validation_error::invalid_option_value, optionName);
}

static bool applyRuntimeParityPreset(const po::variables_map& vm, bool quiet) {
  if (g_runtime_parity.empty() || g_runtime_parity == "off") {
    return true;
  }

  if (g_runtime_parity != "deroluna") {
    setcolor(YELLOW);
    std::cerr << "Warning: unknown runtime-parity preset '" << g_runtime_parity
              << "', using 'off'\n";
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
    g_runtime_parity = "off";
    return true;
  }

  // Preserve explicit user flags; only fill in defaults.
  if (!vm.count("lookup")) {
    useLookupMine = true;
  }
  if (!vm.count("lookup-mode")) {
    g_lookup_mode = LOOKUP_MODE_HYBRID;
  }
  if (!vm.count("omp-threads")) {
    g_omp_threads = 1;
  }
  if (!vm.count("adaptive")) {
    g_adaptive_threads = false;
    g_overprovision_count = 0;
  }
  if (vm.count("no-lock")) {
    lockThreads = false;
  }
  if (vm.count("no-per-thread-affinity")) {
    g_no_per_thread_affinity = true;
    #ifdef _WIN32
    // Set process affinity to all available cores (DeroLuna style)
    // This allows the OS to migrate threads for thermal balance.
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    SetProcessAffinityMask(GetCurrentProcess(), sysInfo.dwActiveProcessorMask);
    #endif
  }

#if defined(_WIN32)
  if (!vm.count("differential-affinity")) {
    g_differential_affinity = 3;  // Balanced mode
    if (g_pcore_count == 0) {
      detectPCores();
    }
  }
#endif

  g_print_runtime_config = true;

  if (!quiet) {
    setcolor(CYAN);
    std::cout << "Runtime parity preset applied: deroluna" << std::endl;
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
  }
  return true;
}

static bool initializeFullLookupTable() {
#if USE_LOOKUP_TABLES
  if (lookup_full::g_lookup3D != nullptr) {
    return true;
  }

  lookup_full::g_lookup3D = static_cast<uint8_t*>(
      malloc_huge_pages(lookup_full::LOOKUP3D_SIZE));
  if (lookup_full::g_lookup3D == nullptr) {
    lookup_full::g_lookup3D = static_cast<uint8_t*>(std::malloc(lookup_full::LOOKUP3D_SIZE));
  }
  if (lookup_full::g_lookup3D == nullptr) {
    setcolor(RED);
    std::cerr << "Failed to allocate full lookup table (16 MB)\n";
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
    return false;
  }

  setcolor(CYAN);
  std::cout << "Initializing full lookup table (16 MB)..." << std::endl;
  fflush(stdout);
  setcolor(BRIGHT_WHITE);

  lookup_full::generate_lookup3D(lookup_full::g_lookup3D);
  return true;
#else
  return false;
#endif
}

static void printResolvedRuntimeConfig(int miningThreads) {
  if (!g_print_runtime_config) {
    return;
  }

  std::ostringstream line;
  line << "mode=" << deroModeToString(g_deroMiningMode)
       << " kernel=" << getCurrentAstroAlgoName()
       << " lookup_mode=" << lookupModeToString(g_lookup_mode)
       << " lookup_smart_threshold=" << g_lookup_smart_threshold
       << " lookup_smart_telemetry=" << (g_lookup_smart_telemetry ? "1" : "0")
       << " thermal_yield=" << (thermal::thermal_yield_enabled() ? "on" : "off")
       << " parity=" << g_runtime_parity
       << " lookup_enabled=" << (useLookupMine ? "1" : "0")
       << " lookup3d_table=" << (lookup3D_global != nullptr ? "1" : "0")
       << " full_lookup_table=" << (lookup_full::g_lookup3D != nullptr ? "1" : "0")
       << " sa_backend=" << getSABackendName()
       << " spsa=" << (g_use_spsa ? "on" : "off")
       << " spsa_impl=" << (g_use_local_spsa ? "local" : "library")
       << " spsa_stamp_fast=" << (g_spsa_stamp_fast ? "on" : "off")
       << " spsa_decode_bases=" << (g_spsa_decode_bases ? "on" : "off")
       << " spsa_bucket_prefetch=" << spsaBucketPrefetchModeToString(g_spsa_bucket_prefetch)
       << " spsa_max_data_len=" << g_spsa_max_data_len
       << " spsa_hit_counters=" << (g_spsa_hit_counters ? "on" : "off")
       << " spsa_sha_profile=" << (g_spsa_sha_profile ? "on" : "off")
       << " spsa_sha_pair=" << (g_spsa_sha_pair ? "on" : "off")
       << " spsa_sha_zeroize=" << (g_spsa_sha_zeroize ? "on" : "off")
       << " lock_threads=" << (lockThreads ? "1" : "0")
       << " array_telemetry=" << (g_array_telemetry ? "1" : "0")
       << " phase_telemetry=" << (g_phase_telemetry ? "1" : "0")
       << " omp_threads=" << g_omp_threads
       << " threads=" << miningThreads;
  logInfo(line.str());
}

int dirtybird_main(int argc, char** argv)
{
  // test_cshake256();
  CpuRuntimeProfileSummary cpu_profile;

  #ifdef DIRTYBIRD_HIP
  GPUTest();
  if (reportInterval == 3) reportInterval = 5;
  HIP_deviceCount = getGPUCount();
  for (int i = 0; i < HIP_deviceCount; i++) {
    HIP_names[i] = getDeviceName(i);
    HIP_pcieID[i] = getPCIBusId(i);
  }
  #endif

  std::atexit(onExit);
  signal(SIGTERM, sigterm);
  signal(SIGINT, sigint);
  alignas(64) char buf[65536];
  setvbuf(stdout, buf, _IOFBF, 65536);
  srand(time(NULL)); // Placing higher here to ensure the effect cascades through the entire program

  #if defined(DIRTYBIRD_ASTROBWTV3)
  initWolfLUT();
  detectAVX512();  // Detect AVX512 for optimized branch kernels
  initializeExterns();
  #endif

  // Check command line arguments.
#if USE_LOOKUP_TABLES
  lookup1D_global = (uint8_t *)malloc_huge_pages(regOps_size * 256 * sizeof(uint8_t));
  if (lookup1D_global == nullptr) {
    lookup1D_global = (uint8_t *)std::malloc(regOps_size * 256 * sizeof(uint8_t));
  }
  if (lookup1D_global == nullptr) {
    setcolor(RED);
    std::cerr << "Failed to allocate 1D lookup table\n";
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
    return -1;
  }
  // HYBRID is the default: 38KB 1D table + AVX2 ALU for branched ops.
  // Skip 6.5MB 3D table allocation to preserve L3 cache at 20+ threads.
  // Users can still request 3D via --lookup-mode 3d (lazy allocation).
  static bool lookup3D_from_hugepages = false;
  lookup3D_global = nullptr;
  printf("Initializing lookup tables for memory-bound branch operations...\n");
  lookup_tables::generateTables(lookup1D_global, nullptr);
  printf("Lookup tables initialized (1D: %zu KB, hybrid mode — 3D table skipped)\n",
         (size_t)regOps_size * 256 / 1024);
  g_lookup_mode = LOOKUP_MODE_HYBRID;
#endif

  if (!NUMAOptimizer::initialize()) {
    std::cerr << "NUMA optimization unavailable, falling back to default" << std::endl;
  }

  po::variables_map vm;
  po::options_description opts = get_prog_opts();
  try
  {
    int style = get_prog_style();
    po::parsed_options parsed = po::command_line_parser(argc, argv)
                                  .options(opts)
                                  .style(style)
                                  .allow_unregistered()  // Allow unknown args
                                  .run();
    
    // Get unrecognized options
    std::vector<std::string> unrecognized = po::collect_unrecognized(parsed.options, po::include_positional);
    
    // Process recognized options
    po::store(parsed, vm);
    po::notify(vm);

    // Load config file (CLI args take priority)
    std::string configPath = findConfigFile();
    if (!configPath.empty()) {
        if (loadConfig(configPath, vm)) {
            if (!vm.count("quiet")) {
                setcolor(CYAN);
                printf("Loaded configuration from %s\n", configPath.c_str());
                fflush(stdout);
                setcolor(BRIGHT_WHITE);
            }
        }
    }

    #if defined(_WIN32)
      SetConsoleOutputCP(CP_UTF8);
      hInput = GetStdHandle(STD_INPUT_HANDLE);
      GetConsoleMode(hInput, &prev_mode);
      SetConsoleMode(hInput, ENABLE_EXTENDED_FLAGS | (prev_mode & ~ENABLE_QUICK_EDIT_MODE));

      // Detect power source before applying power overrides.
      detectPowerState();

      // Check for --no-power-override flag
      g_no_power_override = vm.count("no-power-override") > 0;

      if (!g_no_power_override) {
          // Elevate process priority (same as DeroLuna's Start-Optimized.bat)
          SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);

          // Reduce timer resolution for tighter scheduling
          timeBeginPeriod(1);

          // Apply High Performance power plan + processor tuning
          applyHighPerformancePower();

          // Apply MSR-level optimizations (HWP, EPP, RAPL if unlocked)
          applyMSRPerformanceOptimizations();
      } else {
          // Still set timer resolution (needed for pacer accuracy)
          timeBeginPeriod(1);

          setcolor(CYAN);
          printf("Power: All power overrides disabled (--no-power-override)\n");
          fflush(stdout);
          setcolor(BRIGHT_WHITE);
      }
    #endif
      setcolor(BRIGHT_WHITE);
      if(vm.count("quiet")) {
        beQuiet = true;
      }
      // Clean startup - version info printed later after all options parsed

    // Check if any unrecognized option is a coin symbol
    std::vector<std::string> stillUnrecognized;

    for (const auto& arg : unrecognized) { 
      std::string cleanArg = arg;
      
      // Strip leading dashes
      while (!cleanArg.empty() && cleanArg[0] == '-') {
        cleanArg = cleanArg.substr(1);
      }
      
      // Try to parse as a (possibly versioned) coin
      CoinParseResult parseResult = parseCoinWithVersion(cleanArg);
      
      // Look up the base coin
      const Coin* foundCoin = findCoinBySymbol(parseResult.baseSymbol);
      
      if (foundCoin) {
        miningProfile.setCoin(*foundCoin, parseResult.algoOverride);
        
        // Build display string
        std::string versionStr = "";
        if (parseResult.isVersioned && !parseResult.versionDisplay.empty()) {
          versionStr = " (" + parseResult.versionDisplay + ")";
        }
                
        setcolor(BRIGHT_YELLOW);
        printf("Set to mine %s%s from command line argument '%s'\n\n", 
              miningProfile.coin.coinPrettyName.c_str(),
              versionStr.c_str(),
              arg.c_str());
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
      }
      else {
        stillUnrecognized.push_back(arg);
      }
    }

    if (!stillUnrecognized.empty()) {
      std::string errorMsg = "unrecognized option";
      if (stillUnrecognized.size() > 1) {
        errorMsg += "s";
      }
      for (size_t i = 0; i < stillUnrecognized.size(); ++i) {
        if (i == 0) errorMsg += " '";
        else if (i == stillUnrecognized.size() - 1) errorMsg += "' and '";
        else errorMsg += "', '";
        errorMsg += stillUnrecognized[i];
      }
      errorMsg += "'";
      
      throw po::error(errorMsg);
    }
  }
  catch (std::exception &e)
  {
    printf("%s v%s %s\n", consoleLine, versionString, targetArch);
    setcolor(RED);
    std::cerr << "Error: " << e.what() << "\n";
    std::cerr << "Remember: Long options now use a double-dash -- instead of a single-dash -\n";
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
    return -1;
  }
  catch (...)
  {
    setcolor(RED);
    std::cerr << "Unknown error!" << "\n";
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
    return -1;
  }

  if (vm.count("help"))
  {
    std::cout << opts << std::endl;
    boost::this_thread::sleep_for(boost::chrono::seconds(1));
    return 0;
  }
  
#if defined(_WIN32)
  SetConsoleOutputCP(CP_UTF8);
  HANDLE hSelfToken = NULL;

  ::OpenProcessToken(::GetCurrentProcess(), TOKEN_ALL_ACCESS, &hSelfToken);
  if (SetPrivilege(hSelfToken, SE_LOCK_MEMORY_NAME, true)) {
    if (!beQuiet) {
      setcolor(BRIGHT_GREEN);
      printf("Permission Granted for Huge Pages!\n");
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }
  } else {
    setcolor(YELLOW);
    printf("Huge Pages: Permission Failed...\n");
    printf("  To enable: Run secpol.msc -> Local Policies -> User Rights Assignment\n");
    printf("  Add your user to 'Lock pages in memory' and reboot\n");
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
  }
#endif

  // #if defined(DIRTYBIRD_YESPOWER)
  // yespower_bench_result_t results[10];
  // size_t num_results = 10;
  
  // if (benchmark_yespower_comparison_mt(results, &num_results) == 0) {
  //   print_yespower_benchmark_results(results, num_results);
  // }
  // #endif

  #define UNSUPPORTED_ALGO_ERROR(ERROR_MSG) \
    setcolor(RED); \
    printf("%s", ERROR_MSG); \
    fflush(stdout); \
    setcolor(BRIGHT_WHITE); \
    return 1;

  // Complete coin validation section:
  if (miningProfile.coin.coinId == COIN_DERO) {
    #if defined(DIRTYBIRD_ASTROBWTV3)
    preserveAlgoOverride(miningProfile, COIN_DERO);
    #else
    UNSUPPORTED_ALGO_ERROR(unsupported_astro);
    #endif
  }

  // DERO Miner - Only DERO (AstroBWTv3) supported

  // No xatum protocol support in DERO-only miner

  miningProfile.useStratum |= vm.count("stratum");

  // DERO Miner: Non-DERO test commands removed (including test-spectre)

  if (vm.count("sabench"))
  {
    #if defined(DIRTYBIRD_ASTROBWTV3)
    runDivsufsortBenchmark();
    return 0;
    #else
    setcolor(RED);
    printf("%s", unsupported_astro);
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
    #endif
  }

  if (vm.count("daemon-address"))
  {
    miningProfile.setPoolAddress(vm["daemon-address"].as<std::string>());
  }

  if (vm.count("port"))
  {
    miningProfile.port = std::to_string(vm["port"].as<int>());
    try {
      const int i{std::stoi(miningProfile.port)};
    } catch (...) {
      printf("ERROR: provided port is invalid: %s\n", miningProfile.port.c_str());
      return 1;
    }
  }
  if (vm.count("wallet"))
  {
    miningProfile.wallet = vm["wallet"].as<std::string>();
    // DERO Miner: Only DERO wallet detection
    if(miningProfile.wallet.find("dero", 0) != std::string::npos) {
      preserveAlgoOverride(miningProfile, COIN_DERO);
    }

    boost::char_separator<char> sep(".");
    boost::tokenizer<boost::char_separator<char>> tok(miningProfile.wallet, sep);
    std::vector<std::string> tokens;
    std::copy(tok.begin(), tok.end(), std::back_inserter<std::vector<std::string> >(tokens));
    if(tokens.size() == 2) {
      miningProfile.wallet = tokens[0];
      workerNameFromWallet = tokens[1];
    }
  }
  if (vm.count("ignore-wallet"))
  {
    checkWallet = false;
  }
  if (vm.count("worker-name"))
  {
    workerName = vm["worker-name"].as<std::string>();
  }
  else
  {
    if(workerNameFromWallet != "") {
      workerName = workerNameFromWallet;
    } else {
      workerName = boost::asio::ip::host_name();
    }
  }
  miningProfile.workerName = workerName;
  if (vm.count("threads"))
  {
    threads = vm["threads"].as<int>();
  }
  if (vm.count("report-interval"))
  {
    reportInterval = vm["report-interval"].as<int>();
  }
  if (vm.count("no-cpu-auto"))
  {
    g_cpu_auto_profile = false;
  }
  // Dev fee completely disabled - all hashrate goes to user
  if (vm.count("password"))
  {
    stratumPassword = vm["password"].as<std::string>();
  }
  if (vm.count("no-lock"))
  {
    if (!beQuiet) {
      setcolor(CYAN);
      printf("CPU affinity has been disabled\n");
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }
    lockThreads = false;
  }
  if (vm.count("runtime-parity"))
  {
    g_runtime_parity = vm["runtime-parity"].as<std::string>();
    boost::algorithm::to_lower(g_runtime_parity);
  }
  if (vm.count("spsa"))
  {
    g_use_spsa = parseOnOffValue(vm["spsa"].as<std::string>(), "spsa");
    if (!beQuiet) {
      setcolor(CYAN);
      printf("SPSA optimization %s\n", g_use_spsa ? "enabled" : "disabled");
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }
  }
  if (vm.count("local-spsa") && vm.count("library-spsa"))
  {
    throw po::validation_error(po::validation_error::invalid_option_value, "library-spsa");
  }
  if (vm.count("local-spsa"))
  {
    g_use_local_spsa = true;
    if (!beQuiet) {
      setcolor(CYAN);
      printf("Integrated C++ SPSA path enabled\n");
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }
  }
  if (vm.count("library-spsa"))
  {
    g_use_local_spsa = false;
    if (!beQuiet) {
      setcolor(CYAN);
      printf("Legacy SPSA library path enabled\n");
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }
  }
  if (vm.count("spsa-stamp-fast"))
  {
    g_spsa_stamp_fast = parseOnOffValue(vm["spsa-stamp-fast"].as<std::string>(), "spsa-stamp-fast");
  }
  if (vm.count("spsa-decode-bases"))
  {
    g_spsa_decode_bases = parseOnOffValue(vm["spsa-decode-bases"].as<std::string>(), "spsa-decode-bases");
  }
  if (vm.count("spsa-bucket-prefetch"))
  {
    g_spsa_bucket_prefetch = parseSpsaBucketPrefetchMode(vm["spsa-bucket-prefetch"].as<std::string>());
  }
  if (vm.count("spsa-max-data-len"))
  {
    g_spsa_max_data_len = vm["spsa-max-data-len"].as<int>();
    if (g_spsa_max_data_len < 0) {
      throw po::validation_error(po::validation_error::invalid_option_value, "spsa-max-data-len");
    }
  }
  if (vm.count("spsa-hit-counters"))
  {
    g_spsa_hit_counters = parseOnOffValue(vm["spsa-hit-counters"].as<std::string>(), "spsa-hit-counters");
  }
  if (vm.count("spsa-sha-profile"))
  {
    g_spsa_sha_profile = parseOnOffValue(vm["spsa-sha-profile"].as<std::string>(), "spsa-sha-profile");
  }
  if (vm.count("spsa-sha-pair"))
  {
    g_spsa_sha_pair = parseOnOffValue(vm["spsa-sha-pair"].as<std::string>(), "spsa-sha-pair");
  }
  if (vm.count("spsa-sha-zeroize"))
  {
    g_spsa_sha_zeroize = parseOnOffValue(vm["spsa-sha-zeroize"].as<std::string>(), "spsa-sha-zeroize");
  }
  if (vm.count("lookup-smart-threshold"))
  {
    g_lookup_smart_threshold = vm["lookup-smart-threshold"].as<int>();
    if (g_lookup_smart_threshold < 0 || g_lookup_smart_threshold > 32) {
      throw po::validation_error(po::validation_error::invalid_option_value, "lookup-smart-threshold");
    }
  }
  if (vm.count("lookup-smart-telemetry"))
  {
    g_lookup_smart_telemetry = true;
    g_array_telemetry = true;
  }
#if defined(_WIN32)
  // P-core detection and affinity (for hybrid Intel CPUs like i7-13700HX)
  if (vm.count("p-cores-only"))
  {
    g_pcores_only = true;
    int pcoreCount = detectPCores();
    if (pcoreCount > 0) {
      setcolor(CYAN);
      printf("P-cores only mode: detected %d P-core logical processors\n", pcoreCount);
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    } else {
      setcolor(YELLOW);
      printf("Warning: Could not detect P-cores, falling back to all cores\n");
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
      g_pcores_only = false;
    }
  }

  // Differential affinity mode for hybrid CPUs
  if (vm.count("differential-affinity"))
  {
    g_differential_affinity = vm["differential-affinity"].as<int>();
    if (g_differential_affinity < 0 || g_differential_affinity > 3) {
      setcolor(YELLOW);
      printf("Warning: Invalid differential-affinity mode %d, using default (0)\n", g_differential_affinity);
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
      g_differential_affinity = 0;
    }

    // Ensure P-core detection runs for differential affinity modes
    if (g_differential_affinity > 0 && g_pcore_count == 0) {
      detectPCores();
    }

    if (g_differential_affinity > 0 && !beQuiet) {
      const char* modeNames[] = {"Default", "P-cores first", "Physical only", "Balanced"};
      setcolor(CYAN);
      printf("Differential affinity mode %d: %s\n", g_differential_affinity, modeNames[g_differential_affinity]);
      if (g_pcore_count > 0) {
        printf("  P-cores: %d logical, E-cores: %d logical\n",
               g_pcore_count, (int)g_ecore_logical_ids.size());
      }
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }
  }

  if (vm.count("show-affinity"))
  {
    g_show_affinity = true;
  }
#endif
  if (vm.count("no-spsa"))
  {
    g_use_spsa = false;
    if (!beQuiet) {
      setcolor(CYAN);
      printf("SPSA optimization disabled\n");
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }
  }
  if (vm.count("omp-threads"))
  {
    g_omp_threads = vm["omp-threads"].as<int>();
    if (g_omp_threads < 0) g_omp_threads = 0;  // Treat negative as auto
  }
  if (vm.count("adaptive"))
  {
    g_adaptive_threads = true;
    g_overprovision_count = vm["adaptive"].as<int>();
    // 0 = auto (2x threads), >0 = explicit start count
    g_adaptive_warmup_secs = vm["adaptive-warmup"].as<int>();
    if (g_adaptive_warmup_secs < 5) g_adaptive_warmup_secs = 5;
    lockThreads = false;  // Over-provisioning requires free OS scheduling
    if (!beQuiet) {
      setcolor(CYAN);
      printf("Adaptive mode: over-provision + cull (warmup: %ds)\n", g_adaptive_warmup_secs);
      printf("CPU affinity has been disabled\n");
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }
  }

  if (vm.count("force-lock"))
  {
    if (g_adaptive_threads) {
      setcolor(YELLOW);
      printf("Warning: --force-lock ignored because adaptive mode requires free scheduling\n");
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    } else {
      lockThreads = true;
      g_no_per_thread_affinity = false;
      if (!beQuiet) {
        setcolor(CYAN);
        printf("CPU affinity forced on\n");
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
      }
    }
  }

  applyRuntimeParityPreset(vm, beQuiet);
  cpu_profile = detectCpuRuntimeProfileSummary();
  applyCpuRuntimeProfile(vm, cpu_profile);

  {
    bool thermalYield = false;
    if (vm.count("thermal-yield")) {
      thermalYield = parseOnOffValue(vm["thermal-yield"].as<std::string>(), "thermal-yield");
    }
    thermal::set_thermal_yield_enabled(thermalYield);
  }

#ifdef DIRTYBIRD_OPENMP
  // Initialize OpenMP thread count globally before mining starts
  // This must happen after both --threads and --omp-threads are parsed
  // IMPORTANT: We set OMP_NUM_THREADS env var because omp_set_num_threads() only
  // affects the calling thread, not worker threads created by std::thread
  {
    int omp_threads;
    int mining_threads = threads > 0 ? threads : static_cast<int>(std::thread::hardware_concurrency());
    if (g_omp_threads > 0) {
      omp_threads = g_omp_threads;
    } else {
      omp_threads = get_auto_omp_threads_for_mining(mining_threads);
      g_omp_threads = omp_threads;  // Update global so mining threads use a single resolved value
    }

    // Set environment variable - this affects ALL threads including worker threads
    std::string omp_env = "OMP_NUM_THREADS=" + std::to_string(omp_threads);
    #if defined(_WIN32)
      _putenv(omp_env.c_str());
      _putenv("OMP_DYNAMIC=false");
    #else
      setenv("OMP_NUM_THREADS", std::to_string(omp_threads).c_str(), 1);
      setenv("OMP_DYNAMIC", "false", 1);
    #endif

    // Also call omp_set_* for the main thread (belt and suspenders)
    omp_set_num_threads(omp_threads);
    omp_set_dynamic(0);

    if (!beQuiet) {
      setcolor(CYAN);
      printf("OpenMP: %d threads per miner (max ~%d total with %d miners)\n",
             omp_threads, mining_threads * omp_threads, mining_threads);
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }
  }
#endif

  // Duty-cycle pacer: insert micro-sleeps to prevent thermal throttling
  {
    int pace_interval = 0;
    int pace_sleep_ms = 0;
    if (vm.count("pace-interval")) pace_interval = vm["pace-interval"].as<int>();
    if (vm.count("pace-sleep-us")) pace_sleep_ms = vm["pace-sleep-us"].as<int>();  // Reusing CLI name for now
    if (pace_interval > 0 && pace_sleep_ms > 0) {
      thermal::init_pacer(static_cast<uint32_t>(pace_interval), static_cast<uint32_t>(pace_sleep_ms));
      if (!beQuiet) {
        setcolor(CYAN);
        printf("Duty-cycle pacer: Sleep(%d ms) every %d hashes (timeBeginPeriod=1)\n", pace_sleep_ms, pace_interval);
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
      }
    }
  }

  if (vm.count("verbose-tune"))
  {
    g_verbose_tune = true;
  }
  if (vm.count("gpu"))
  {
    gpuMine = true;
  }

  if (vm.count("broadcast"))
  {
    broadcastStats = true;
  }
  // GPU-specific
  if (vm.count("batch-size"))
  {
    batchSize = vm["batch-size"].as<int>();
  }

  // Test-specific
  if (vm.count("op"))
  {
    testOp = vm["op"].as<int>();
  }
  if (vm.count("len"))
  {
    testLen = vm["len"].as<int>();
  }
  if (vm.count("lookup"))
  {
    if (!beQuiet) {
      printf("Use Lookup\n");
    }
    useLookupMine = true;
  }
  if (vm.count("lookup-mode"))
  {
    if (optionExplicit(vm, "lookup-mode") ||
        g_auto_default_astro_kernel == "wolf_memopt" ||
        !g_cpu_auto_profile ||
        g_runtime_parity != "off") {
      g_lookup_mode = parseLookupMode(vm["lookup-mode"].as<std::string>());
    }
    if (g_lookup_mode == LOOKUP_MODE_3D && lookup3D_global == nullptr) {
      // Keep large-page budget for workerData. The 3D LUT is read-only and tolerates
      // regular pages better than the per-thread scratch/SA working set does.
      lookup3D_global = (byte *)std::malloc(branchedOps_size * (256 * 256) * sizeof(byte));
      lookup3D_from_hugepages = false;
      if (lookup3D_global != nullptr) {
        lookup_tables::generateTables(lookup1D_global, lookup3D_global);
        setcolor(CYAN);
        std::cout << "Allocated 3D lookup table on demand (6.5 MB)" << std::endl;
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
      } else {
        setcolor(YELLOW);
        std::cerr << "lookup-mode=3d requested but 3D table allocation failed. Falling back to hybrid.\n";
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        g_lookup_mode = LOOKUP_MODE_HYBRID;
      }
    }
    if (g_lookup_mode == LOOKUP_MODE_FULL) {
      useLookupMine = true;
      if (!initializeFullLookupTable()) {
        setcolor(YELLOW);
        std::cerr << "lookup-mode=full unavailable (allocation/init failed). Falling back to hybrid.\n";
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        g_lookup_mode = LOOKUP_MODE_HYBRID;
      }
    }
    if (g_lookup_mode == LOOKUP_MODE_HYBRID) {
      useLookupMine = true;
      // Free the 6.5MB 3D branched table — hybrid uses 1D LUT + AVX2 ALU
      if (lookup3D_global != nullptr) {
        if (lookup3D_from_hugepages) free_huge_pages(lookup3D_global);
        else std::free(lookup3D_global);
        lookup3D_global = nullptr;
        setcolor(CYAN);
        std::cout << "Hybrid mode: freed 3D lookup table (6.5 MB saved)" << std::endl;
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
      }
    }
  }

  if (vm.count("no-runtime-config")) {
    g_print_runtime_config = false;
  }
  if (vm.count("print-runtime-config")) {
    g_print_runtime_config = true;
  }
  if (vm.count("array-telemetry")) {
    g_array_telemetry = true;
    if (!beQuiet) {
      setcolor(CYAN);
      std::cout << "Array telemetry enabled (15s interval)" << std::endl;
      setcolor(BRIGHT_WHITE);
    }
  }
  if (vm.count("phase-telemetry")) {
    g_phase_telemetry = true;
    if (!beQuiet) {
      setcolor(CYAN);
      std::cout << "Phase telemetry enabled (15s interval)" << std::endl;
      setcolor(BRIGHT_WHITE);
    }
  }
  if (vm.count("lookup-smart-telemetry")) {
    if (!beQuiet) {
      setcolor(CYAN);
      std::cout << "Smart lookup telemetry enabled (15s interval)" << std::endl;
      setcolor(BRIGHT_WHITE);
    }
  }

  // We can do this because we've set default in terminal.hpp
  tuneWarmupSec = vm["tune-warmup"].as<int>();
  tuneDurationSec = vm["tune-duration"].as<int>();

  mine_time = vm["mine-time"].as<int>();

  // Reject ambiguous mode selections so benchmark runs are reproducible.
  int deroModeFlags =
      (vm.count("interleaved") ? 1 : 0) +
      (vm.count("cache-batch") ? 1 : 0) +
      (vm.count("cache-batch-hybrid") ? 1 : 0) +
      (vm.count("lockfree") ? 1 : 0);
  if (deroModeFlags > 1) {
    setcolor(RED);
    std::cerr << "Error: choose only one mining mode flag from "
                 "--interleaved, --cache-batch, --cache-batch-hybrid, --lockfree\n";
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
    return -1;
  }

  // Ensure we capture *all* of the other options before we start using goto
  if (vm.count("test-dero"))
  {
    #if defined(DIRTYBIRD_ASTROBWTV3)
    // temporary for optimization fishing:
    mapZeroes();
    // end of temporary section

    #if defined(USE_ASTRO_SPSA)
      if (g_use_spsa) {
        initSPSA();
      }
    #endif
    int rc = DeroTesting(testOp, testLen, useLookupMine);
    if(rc > 255) {
      rc = 1;
    }
    return rc;
    #else 
    setcolor(RED);
    printf("%s", unsupported_astro);
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
    return 1;
    #endif
  }
  miningProfile.coin = coins[COIN_DERO];

  if (threads == 0) {
    threads = processor_count;
  }

#if defined(_WIN32)
  // Limit threads to P-core count when --p-cores-only is enabled
  if (g_pcores_only && g_pcore_count > 0 && threads > g_pcore_count) {
    setcolor(YELLOW);
    printf("P-cores only: reducing threads from %d to %d (P-core logical count)\n", threads, g_pcore_count);
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
    threads = g_pcore_count;
  }
#endif

  setcolor(BRIGHT_YELLOW);
  #ifdef DIRTYBIRD_ASTROBWTV3
  if (miningProfile.coin.miningAlgo == ALGO_ASTROBWTV3) {
    if (vm.count("no-tune")) {
      std::string noTune = vm["no-tune"].as<std::string>();
      if (noTune == "hybrid") {
        // --no-tune hybrid: use lookupCompute with HYBRID mode (38KB 1D + AVX2 ALU)
        if (!setAstroAlgo("lookup")) {
          throw po::validation_error(po::validation_error::invalid_option_value, "no-tune");
        }
        if (!vm.count("lookup-mode")) {
          g_lookup_mode = LOOKUP_MODE_HYBRID;
          useLookupMine = true;
          if (lookup3D_global != nullptr) {
            if (lookup3D_from_hugepages) free_huge_pages(lookup3D_global);
            else std::free(lookup3D_global);
            lookup3D_global = nullptr;
          }
          setcolor(CYAN);
          std::cout << "Hybrid mode: 1D LUT (38 KB) + AVX2 ALU for branched ops" << std::endl;
          fflush(stdout);
          setcolor(BRIGHT_WHITE);
        }
      } else if(!setAstroAlgo(noTune)) {
        throw po::validation_error(po::validation_error::invalid_option_value, "no-tune");
      }
      if (noTune == "lookup") {
        // Respect the explicit lookup kernel request even when the caller also
        // pins a lookup-mode like 3d/full. The old logic only enabled
        // useLookupMine when lookup-mode was omitted, which left the runtime in
        // a contradictory "kernel=lookup, lookup_enabled=0" state.
        useLookupMine = true;
        // When --no-tune lookup is specified without explicit --lookup-mode,
        // default to the safer hybrid lookup layout.
        if (!vm.count("lookup-mode")) {
          g_lookup_mode = LOOKUP_MODE_HYBRID;
          if (lookup3D_global != nullptr) {
            if (lookup3D_from_hugepages) free_huge_pages(lookup3D_global);
            else std::free(lookup3D_global);
            lookup3D_global = nullptr;
          }
          setcolor(CYAN);
          std::cout << "Hybrid mode (default for lookup): 1D LUT (38 KB) + AVX2 ALU for branched ops" << std::endl;
          fflush(stdout);
          setcolor(BRIGHT_WHITE);
        }
      }
    } else if (vm.count("auto-tune")) {
      astroTune(threads, tuneWarmupSec, tuneDurationSec);
    } else {
      // Deterministic default: keep lookup tables enabled, but only switch the
      // kernel itself when the caller explicitly asked for --lookup.
      useLookupMine = true;
      const char* default_kernel = vm.count("lookup") ? "lookup" : g_auto_default_astro_kernel.c_str();
      if (!setAstroAlgo(default_kernel)) {
        // Safety fallback if kernel naming changes in a future build.
        astroTune(threads, tuneWarmupSec, tuneDurationSec);
      }
    }

    // SA prefetch configuration (manual override or autotuning)
    #ifdef USE_CUSTOM_SA
    if (vm.count("sa-prefetch")) {
      // Manual override takes priority
      std::string saPrefetch = vm["sa-prefetch"].as<std::string>();
      if (!parseSAPrefetch(saPrefetch)) {
        throw po::validation_error(po::validation_error::invalid_option_value, "sa-prefetch");
      }
    } else if (vm.count("sa-tune") && !vm.count("no-sa-tune")) {
      // Run SA autotuning
      saTune(threads, tuneWarmupSec, tuneDurationSec);
    }
    // If neither is specified, default SA config is used
    #else
    if (vm.count("sa-prefetch") || (vm.count("sa-tune") && !vm.count("no-sa-tune"))) {
      setcolor(YELLOW);
      std::cerr << "SA prefetch tuning flags are ignored: backend is not USE_CUSTOM_SA (current: "
                << getSABackendName() << ")\n";
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }
    #endif
  }
  fflush(stdout);
  setcolor(BRIGHT_WHITE);
  #endif

  // Cache-batch mining mode selection (k1's L1 cache optimization)
  #ifdef DIRTYBIRD_ASTROBWTV3
  if (miningProfile.coin.miningAlgo == ALGO_ASTROBWTV3) {
    if (vm.count("bench-cache-batch")) {
      // Run cache-batching benchmark and exit
      setcolor(CYAN);
      std::cout << "Running cache-batching benchmark..." << std::endl;
      setcolor(BRIGHT_WHITE);
      int optimal = benchmarkCacheBatching(100);
      setcolor(BRIGHT_YELLOW);
      std::cout << "Optimal batch size for this CPU: " << optimal << std::endl;
      if (optimal <= 1) {
        std::cout << "Recommendation: Use standard mining (--no-cache-batch or omit cache options)" << std::endl;
      } else {
        std::cout << "Recommendation: Use --cache-batch for improved performance" << std::endl;
      }
      setcolor(BRIGHT_WHITE);
      return 0;
    }

    if (vm.count("bench-interleaved")) {
      // Run interleaved vs standard benchmark and exit
      setcolor(CYAN);
      std::cout << "Running interleaved vs standard benchmark..." << std::endl;
      setcolor(BRIGHT_WHITE);
      double improvement = benchmarkInterleaved(500);
      setcolor(BRIGHT_YELLOW);
      if (improvement > 0) {
        std::cout << "\nConclusion: Interleaved mode provides " << improvement << "% speedup" << std::endl;
        std::cout << "Run with --interleaved flag for improved performance" << std::endl;
      } else {
        std::cout << "\nConclusion: Standard mode is " << -improvement << "% faster" << std::endl;
        std::cout << "Run without --interleaved flag" << std::endl;
      }
      setcolor(BRIGHT_WHITE);
      return 0;
    }

    if (vm.count("interleaved")) {
      g_deroMiningMode = DERO_MINE_INTERLEAVED;
      setcolor(CYAN);
      std::cout << "Interleaved mining enabled (DeroLuna-style two-miners-per-thread ILP)" << std::endl;
      setcolor(BRIGHT_WHITE);
    } else if (vm.count("cache-batch")) {
      g_deroMiningMode = DERO_MINE_BATCHED;
      setcolor(CYAN);
      std::cout << "Cache-focused batched mining enabled (k1's L1I optimization)" << std::endl;
      setcolor(BRIGHT_WHITE);
    } else if (vm.count("cache-batch-hybrid")) {
      g_deroMiningMode = DERO_MINE_HYBRID;
      setcolor(CYAN);
      std::cout << "Hybrid mining mode: will auto-detect optimal approach" << std::endl;
      setcolor(BRIGHT_WHITE);
    } else if (vm.count("lockfree")) {
      g_deroMiningMode = DERO_MINE_LOCKFREE;
      setcolor(CYAN);
      std::cout << "Lock-free mining enabled (Go-style thread coordination)" << std::endl;
      setcolor(BRIGHT_WHITE);
    }
  }
  #endif

  // DERO Miner: DIRTYBIRD_SHAIHIVE block removed

  printf("\n");

  printHashrateOnExit = true;
  setPhaseTelemetryEnabled(g_phase_telemetry);
  if (g_phase_telemetry) {
    resetPhaseTelemetry();
  }
  if (g_lookup_smart_telemetry) {
    resetLookupTelemetry();
  }

  printResolvedRuntimeConfig(threads);

  // Clean startup display (DeroLuna style)
  if (!beQuiet) {
    std::string versionStr = std::string("DIRTYBIRD Miner ") + versionString;
    std::string serverStr = miningProfile.host + ":" + miningProfile.port;
    std::string threadStr = std::to_string(threads);

    printf("\n");
    logInfoPair("Version", versionStr);
    logInfoPair("Server", serverStr);
    logInfoPair("Wallet", miningProfile.wallet);
    logInfoPair("Threads", threadStr);
    logInfoPair("CPU", buildCpuSummaryLine(cpu_profile));
    if (g_print_runtime_config) {
      logInfoPair("Runtime", buildRuntimeSummaryLine());
    }
    printf("\n");
  }

  #ifndef DIRTYBIRD_HIP
    // printSupported();  // Suppressed for cleaner output
  #else
    gpuMine = true;
    precompile_all_kernels();
  #endif
  int rc = enhanceWallet(&miningProfile, checkWallet);
  if(rc != 0) {
    return rc;
  }
  #if defined(USE_ASTRO_SPSA)
    if (g_use_spsa && miningProfile.coin.miningAlgo == ALGO_ASTROBWTV3) {
      initSPSA();
    }
  #endif

  unsigned int n = std::thread::hardware_concurrency();
  (void)n; // DERO Miner: Suppress unused variable warning (was used by RandomX)

  boost::thread GETWORK(getWork_v2, &miningProfile);

  // Calculate actual thread count (with adaptive over-provisioning)
  int start_threads = threads;
  if (g_adaptive_threads && !gpuMine) {
    if (g_overprovision_count > 0) {
      start_threads = g_overprovision_count;
    } else {
      start_threads = threads * 2;  // Default: 2x over-provision
    }
    if (start_threads > MAX_MINING_THREADS - 1) start_threads = MAX_MINING_THREADS - 1;
  }

  // Initialize per-thread counters and stop flags
  for (int i = 0; i < start_threads; i++) {
    thread_counters[i + 1].store(0, std::memory_order_relaxed);
    thread_stop[i + 1].store(false, std::memory_order_relaxed);
  }

  if (!gpuMine &&
      miningProfile.coin.miningAlgo == ALGO_ASTROBWTV3 &&
      (g_deroMiningMode == DERO_MINE_STANDARD || g_deroMiningMode == DERO_MINE_LOCKFREE)) {
    if (!initializeDeroWorkerPool(start_threads)) {
      setcolor(RED);
      std::cerr << "Failed to initialize DERO worker pool" << std::endl << std::flush;
      setcolor(BRIGHT_WHITE);
      return EXIT_FAILURE;
    }

    if (!beQuiet) {
      const auto& worker_pool_stats = getDeroWorkerPoolStats();
      setcolor(worker_pool_stats.regular_workers == 0 ? CYAN : YELLOW);
      printf("Worker pool: %d workers (%d large-2MB, %d large-1GB, %d regular, %d locked, %.1f MB requested)\n",
             worker_pool_stats.requested_workers,
             worker_pool_stats.large_2mb_workers,
             worker_pool_stats.large_1gb_workers,
             worker_pool_stats.regular_workers,
             worker_pool_stats.virtual_locked_workers,
             static_cast<double>(worker_pool_stats.requested_bytes) / (1024.0 * 1024.0));
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }
  }

  // Create worker threads
  std::vector<boost::thread> minerThreads(start_threads);
  std::vector<bool> thread_alive(start_threads, true);

  if (gpuMine)
  {
    threads = 0;
    #ifdef DIRTYBIRD_HIP
    if (!beQuiet) {
      std::cout << "Starting GPU worker.." << std::endl;
    }
    boost::thread t(getMiningFunc(miningProfile.coin.miningAlgo, true), 0);
    #else
    printf("Please use a GPU DIRTYBIRD Miner binary...\n");
    return -1;
    #endif
  } else {
    if (g_adaptive_threads && start_threads > threads) {
      setcolor(CYAN);
      printf("Starting %d threads (will cull to %d after %ds warmup)\n",
             start_threads, threads, g_adaptive_warmup_secs);
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }

    for (int i = 0; i < start_threads; i++)
    {
      minerThreads[i] = boost::thread(getMiningFunc(miningProfile.coin.miningAlgo, false), i + 1);

      if (lockThreads && !g_no_per_thread_affinity)
      {
        setAffinity(minerThreads[i].native_handle(), i);
      }
    }
  }

  g_start_time = std::chrono::steady_clock::now();
  if (broadcastStats)
  {
    boost::thread BROADCAST(BroadcastServer::serverThread, &rate30sec, &accepted, &rejected, algoName(miningProfile.coin.miningAlgo), versionString, reportInterval);
  }

  while (!isConnected)
  {
    boost::this_thread::yield();
  }

  // Adaptive thread culling: spawn a watcher that culls slowest threads after warmup
  boost::thread* cull_thread_ptr = nullptr;
  if (g_adaptive_threads && start_threads > threads && !gpuMine) {
    int target_threads = threads;  // Capture before lambda
    cull_thread_ptr = new boost::thread(
      [&minerThreads, &thread_alive, start_threads, target_threads]() {
        // Wait for warmup period
        for (int s = 0; s < g_adaptive_warmup_secs; s++) {
          boost::this_thread::sleep_for(boost::chrono::seconds(1));
          if (ABORT_MINER) return;
        }
        if (ABORT_MINER) return;

        // Collect per-thread hash counts (accumulated over warmup)
        std::vector<std::pair<int64_t, int>> rates;
        for (int i = 0; i < start_threads; i++) {
          int tid = i + 1;
          int64_t hashes = thread_counters[tid].load(std::memory_order_relaxed);
          rates.push_back({hashes, tid});
        }

        // Sort ascending (slowest first)
        std::sort(rates.begin(), rates.end());

        int to_kill = start_threads - target_threads;

        setcolor(CYAN);
        printf("\n[ADAPTIVE] Warmup done. Culling %d slowest threads:\n", to_kill);
        for (int i = 0; i < to_kill; i++) {
          int tid = rates[i].second;
          double rate_kh = (double)rates[i].first / g_adaptive_warmup_secs / 1000.0;
          printf("  Stop T%02d (%.1f KH/s)\n", tid, rate_kh);
          thread_stop[tid].store(true, std::memory_order_release);
        }

        // Join culled threads (they exit quickly after seeing stop flag)
        for (int i = 0; i < to_kill; i++) {
          int tid = rates[i].second;
          int idx = tid - 1;
          if (minerThreads[idx].joinable()) {
            minerThreads[idx].join();
          }
          thread_alive[idx] = false;
        }

        printf("[ADAPTIVE] Kept best %d threads:", target_threads);
        for (int i = to_kill; i < (int)rates.size(); i++) {
          double rate_kh = (double)rates[i].first / g_adaptive_warmup_secs / 1000.0;
          printf(" T%02d(%.1f)", rates[i].second, rate_kh);
        }
        printf("\n");
        fflush(stdout);
        setcolor(BRIGHT_WHITE);

        // Reset global counter to get clean post-cull measurement
        counter.store(0, std::memory_order_relaxed);
        rate30sec.clear();
      }
    );
  }

  if(mine_time > 5) {
    mine_duration_timer.expires_after(std::chrono::seconds(mine_time));
    std::cout << "Will mine for " << mine_time << " seconds" << std::endl;
    mine_duration_timer.async_wait([&](const boost::system::error_code &ec)
      {
        ABORT_MINER = true;
        std::cout << std::endl << "Mined for " << mine_time << " seconds" << std::endl;
        update_timer.cancel();
        mine_duration_timer.cancel();
        // Stop all the io_context. So we can actually leave!
        my_context.stop();
        CHECK_CLOSE;
      });
  }

  // boost::thread reportThread([&]() {
    // Set an expiry time relative to now.
    update_timer.expires_after(std::chrono::seconds(1));

    // Start an asynchronous wait.
    update_timer.async_wait(update_handler);
    my_context.run();
  // });
  // setPriority(reportThread.native_handle(), THREAD_PRIORITY_TIME_CRITICAL);

  while(!ABORT_MINER) {
    std::this_thread::yield();
  }
  //ioc.reset();
  GETWORK.interrupt();
  // DERO Miner: Dev fee disabled - DEVWORK thread not created
  // DEVWORK.interrupt();
  if (!beQuiet) {
    std::cout << "Interrupting all threads...\n";
  }
  // Shutdown: interrupt and join only threads that are still alive
  for (int i = 0; i < start_threads; ++i) {
    if (thread_alive[i] && minerThreads[i].joinable()) {
      minerThreads[i].interrupt();
      minerThreads[i].join();
    }
  }
  // Clean up cull thread
  if (cull_thread_ptr) {
    if (cull_thread_ptr->joinable()) {
      cull_thread_ptr->interrupt();
      cull_thread_ptr->join();
    }
    delete cull_thread_ptr;
  }

  shutdownDeroWorkerPool();

  return EXIT_SUCCESS;
}

std::map<int, int> threadToPhysicalCore;
std::mutex threadMapMutex;

#if defined(_WIN32)
DWORD_PTR SetThreadAffinityWithGroups(HANDLE threadHandle, DWORD_PTR coreIndex)
{
  DWORD numGroups = GetActiveProcessorGroupCount();

  // Calculate group and processor within the group
  DWORD group = static_cast<DWORD>(coreIndex / 64);
  DWORD numProcessorsInGroup = GetMaximumProcessorCount(group);
  DWORD processorInGroup = static_cast<DWORD>(coreIndex % numProcessorsInGroup);

  if (group < numGroups)
  {
    GROUP_AFFINITY groupAffinity = {};
    groupAffinity.Group = static_cast<WORD>(group);
    groupAffinity.Mask = static_cast<KAFFINITY>(1ULL << processorInGroup);

    GROUP_AFFINITY previousGroupAffinity;
    if (!SetThreadGroupAffinity(threadHandle, &groupAffinity, &previousGroupAffinity))
    {
      return 0; // Fail case, return 0 like SetThreadAffinityMask
    }

    // Return the previous affinity mask for compatibility with your code
    return previousGroupAffinity.Mask;
  }

  return 0; // If out of bounds
}
#endif

#if defined(_WIN32)

struct CoreInfo {
  DWORD physicalCore;
  DWORD logicalCore;
  bool isPCore;
  bool isHyperthread;
};

std::vector<CoreInfo> getCoreTopology() {
  std::vector<CoreInfo> cores;
  DWORD bufferSize = 0;
  
  GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bufferSize);
  
  std::vector<BYTE> buffer(bufferSize);
  PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info = 
    reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());
  
  if (GetLogicalProcessorInformationEx(RelationProcessorCore, info, &bufferSize)) {
    DWORD offset = 0;
    DWORD physicalCoreId = 0;
    
    while (offset < bufferSize) {
      auto current = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
        reinterpret_cast<BYTE*>(info) + offset);
      
      if (current->Relationship == RelationProcessorCore) {
        for (WORD groupIdx = 0; groupIdx < current->Processor.GroupCount; groupIdx++) {
          KAFFINITY mask = current->Processor.GroupMask[groupIdx].Mask;
          WORD group = current->Processor.GroupMask[groupIdx].Group;
          DWORD logicalCount = __popcnt64(mask);
          DWORD baseLogicalCore = 0;
          
          // Calculate base logical core for this group
          for (WORD g = 0; g < group; g++) {
            baseLogicalCore += GetMaximumProcessorCount(g);
          }
          
          // Detect if this is a P-core (supports hyperthreading) or E-core
          bool isPCore = (logicalCount > 1);
          
          for (DWORD bit = 0; bit < 64; bit++) {
            if (mask & (1ULL << bit)) {
              CoreInfo core;
              core.physicalCore = physicalCoreId;
              core.logicalCore = baseLogicalCore + bit;
              core.isHyperthread = (logicalCount > 1 && __popcnt64(mask & ((1ULL << bit) - 1)) > 0);
              core.isPCore = isPCore;  // New field
              cores.push_back(core);
            }
          }
        }
        physicalCoreId++;
      }
      offset += current->Size;
    }
  }
  
  return cores;
}

// Detect P-cores and E-cores, populate g_pcore_logical_ids and g_ecore_logical_ids
// Returns the number of P-core logical processors (with hyperthreading)
int detectPCores() {
  std::vector<CoreInfo> topology = getCoreTopology();
  g_pcore_logical_ids.clear();
  g_ecore_logical_ids.clear();

  // Count unique P-core and E-core physical cores and collect their logical IDs
  std::map<uint32_t, std::vector<uint32_t>> pcoreMap;
  std::map<uint32_t, std::vector<uint32_t>> ecoreMap;

  for (const auto& core : topology) {
    if (core.isPCore) {
      pcoreMap[core.physicalCore].push_back(core.logicalCore);
    } else {
      ecoreMap[core.physicalCore].push_back(core.logicalCore);
    }
  }

  // Build P-core ordering: primary threads first, then hyperthreads
  for (auto& pair : pcoreMap) {
    if (!pair.second.empty()) {
      g_pcore_logical_ids.push_back(pair.second[0]);  // Primary thread
    }
  }
  for (auto& pair : pcoreMap) {
    for (size_t i = 1; i < pair.second.size(); i++) {
      g_pcore_logical_ids.push_back(pair.second[i]);  // Hyperthreads
    }
  }

  // Build E-core ordering
  for (auto& pair : ecoreMap) {
    for (auto& logicalId : pair.second) {
      g_ecore_logical_ids.push_back(logicalId);
    }
  }

  g_pcore_count = static_cast<int>(g_pcore_logical_ids.size());
  return g_pcore_count;
}

#endif

#if defined(_WIN32)
static std::vector<CoreInfo>& getCachedCoreTopology() {
  static std::vector<CoreInfo> topology = getCoreTopology();
  return topology;
}

static std::vector<DWORD>& getCachedAffinityOrder() {
  static std::vector<DWORD> affinityOrder;
  return affinityOrder;
}

static int& getCachedAffinityMode() {
  static int cachedMode = -1;
  return cachedMode;
}

static void ensureAffinityOrder() {
  auto& topology = getCachedCoreTopology();
  auto& affinityOrder = getCachedAffinityOrder();
  int& cachedMode = getCachedAffinityMode();

  if (cachedMode == g_differential_affinity && !affinityOrder.empty()) {
    return;
  }

  cachedMode = g_differential_affinity;
  affinityOrder.clear();

  std::map<DWORD, std::vector<DWORD>> pCoreMap, eCoreMap;
  for (size_t i = 0; i < topology.size(); i++) {
    if (topology[i].isPCore) {
      pCoreMap[topology[i].physicalCore].push_back(static_cast<DWORD>(i));
    } else {
      eCoreMap[topology[i].physicalCore].push_back(static_cast<DWORD>(i));
    }
  }

  switch (g_differential_affinity) {
    case 1:
      for (auto& pair : pCoreMap) {
        affinityOrder.push_back(pair.second[0]);
      }
      for (auto& pair : pCoreMap) {
        for (size_t i = 1; i < pair.second.size(); i++) {
          affinityOrder.push_back(pair.second[i]);
        }
      }
      if (!g_pcores_only) {
        for (auto& pair : eCoreMap) {
          affinityOrder.push_back(pair.second[0]);
        }
      }
      break;

    case 2:
      for (auto& pair : pCoreMap) {
        affinityOrder.push_back(pair.second[0]);
      }
      if (!g_pcores_only) {
        for (auto& pair : eCoreMap) {
          affinityOrder.push_back(pair.second[0]);
        }
      }
      for (auto& pair : pCoreMap) {
        for (size_t i = 1; i < pair.second.size(); i++) {
          affinityOrder.push_back(pair.second[i]);
        }
      }
      break;

    case 3: {
      auto pIt = pCoreMap.begin();
      auto eIt = eCoreMap.begin();
      while (pIt != pCoreMap.end() || (eIt != eCoreMap.end() && !g_pcores_only)) {
        if (pIt != pCoreMap.end()) {
          affinityOrder.push_back(pIt->second[0]);
          ++pIt;
        }
        if (eIt != eCoreMap.end() && !g_pcores_only) {
          affinityOrder.push_back(eIt->second[0]);
          ++eIt;
        }
      }
      for (auto& pair : pCoreMap) {
        for (size_t i = 1; i < pair.second.size(); i++) {
          affinityOrder.push_back(pair.second[i]);
        }
      }
      break;
    }

    case 0:
    default:
      for (auto& pair : pCoreMap) {
        affinityOrder.push_back(pair.second[0]);
      }
      if (!g_pcores_only) {
        for (auto& pair : eCoreMap) {
          affinityOrder.push_back(pair.second[0]);
        }
      }
      for (auto& pair : pCoreMap) {
        for (size_t i = 1; i < pair.second.size(); i++) {
          affinityOrder.push_back(pair.second[i]);
        }
      }
      break;
  }

  if (g_show_affinity && !affinityOrder.empty()) {
    setcolor(CYAN);
    printf("\nDifferential Affinity Mode %d - Core Assignment Order:\n", g_differential_affinity);
    printf("Thread -> Logical Core (Type)\n");
    for (size_t i = 0; i < affinityOrder.size(); i++) {
      DWORD idx = affinityOrder[i];
      printf("  T%02zu -> Core %2u (%s%s)\n",
             i + 1,
             topology[idx].logicalCore,
             topology[idx].isPCore ? "P-core" : "E-core",
             topology[idx].isHyperthread ? " HT" : "");
    }
    printf("\n");
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
  }
}

static bool resolveThreadPlacementTarget(uint64_t core, DWORD& targetCore, DWORD& logicalCoreIndex) {
  auto& topology = getCachedCoreTopology();
  auto& affinityOrder = getCachedAffinityOrder();
  ensureAffinityOrder();

  if (g_pcores_only && core < g_pcore_logical_ids.size()) {
    targetCore = g_pcore_logical_ids[core];
    logicalCoreIndex = 0;
    for (size_t i = 0; i < topology.size(); i++) {
      if (topology[i].logicalCore == targetCore) {
        logicalCoreIndex = static_cast<DWORD>(i);
        return true;
      }
    }
  } else if (core < affinityOrder.size()) {
    logicalCoreIndex = affinityOrder[core];
    targetCore = topology[logicalCoreIndex].logicalCore;
    return true;
  }

  setcolor(RED);
  std::cerr << "Core ID " << core << " exceeds available cores ("
            << (g_pcores_only ? g_pcore_logical_ids.size() : affinityOrder.size()) << ")" << std::endl;
  fflush(stdout);
  setcolor(BRIGHT_WHITE);
  return false;
}

static bool logicalCoreToProcessorGroup(DWORD targetCore, WORD& group, DWORD& processorInGroup) {
  DWORD numGroups = GetActiveProcessorGroupCount();
  DWORD remaining = targetCore;

  for (DWORD i = 0; i < numGroups; i++) {
    DWORD groupSize = GetMaximumProcessorCount(i);
    if (remaining < groupSize) {
      group = static_cast<WORD>(i);
      processorInGroup = remaining;
      return true;
    }
    remaining -= groupSize;
  }

  setcolor(RED);
  std::cerr << "Logical core ID " << targetCore
            << " exceeds available logical cores" << std::endl;
  fflush(stdout);
  setcolor(BRIGHT_WHITE);
  return false;
}
#endif

void setAffinity(boost::thread::native_handle_type t, uint64_t core)
{
#if defined(_WIN32)
  auto& topology = getCachedCoreTopology();
  DWORD targetCore;
  DWORD logicalCoreIndex;
  if (!resolveThreadPlacementTarget(core, targetCore, logicalCoreIndex)) {
    return;
  }

  HANDLE threadHandle = t;

  {
    std::lock_guard<std::mutex> lock(threadMapMutex);
    threadToPhysicalCore[core] = topology[logicalCoreIndex].physicalCore;
  }

  WORD group = 0;
  DWORD processorInGroup = 0;
  if (!logicalCoreToProcessorGroup(targetCore, group, processorInGroup)) {
    return;
  }

  GROUP_AFFINITY groupAffinity = {0};
  groupAffinity.Group = group;
  groupAffinity.Mask = 1ULL << processorInGroup;

  GROUP_AFFINITY previousAffinity;
  if (!SetThreadGroupAffinity(threadHandle, &groupAffinity, &previousAffinity)) {
    DWORD error = GetLastError();
    setcolor(RED);
    std::cerr << "Failed to set CPU affinity for thread " << core
              << " to physical core " << topology[logicalCoreIndex].physicalCore
              << ", logical core " << targetCore
              << " (group " << group << ", mask " << std::hex << groupAffinity.Mask
              << std::dec << "). Error: " << error << std::endl;
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
  }

#elif !defined(__APPLE__)
  pthread_t threadHandle = t;
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core, &cpuset);

  {
    std::lock_guard<std::mutex> lock(threadMapMutex);
    
    // Read physical core ID from /sys
    std::ifstream core_id_file(
      "/sys/devices/system/cpu/cpu" + std::to_string(core) + "/topology/core_id"
    );
    int physical_core_id = core;  // fallback
    if (core_id_file.is_open()) {
      core_id_file >> physical_core_id;
    }
    
    threadToPhysicalCore[core] = physical_core_id;
  }

  #if defined(DIRTYBIRD_OS_ANDROID)
  (void)threadHandle;
  #else
  if (pthread_setaffinity_np(threadHandle, sizeof(cpu_set_t), &cpuset) != 0)
  {
    std::cerr << "Failed to set CPU affinity for thread" << std::endl;
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
  }
  #endif
#endif
}

void setPriority(boost::thread::native_handle_type t, int priority)
{
#if defined(_WIN32)

  HANDLE threadHandle = t;

  // Set the thread priority
  int threadPriority = priority;
  BOOL success = SetThreadPriority(threadHandle, threadPriority);
  if (!success)
  {
    DWORD error = GetLastError();
    std::cerr << "Failed to set thread priority. Error code: " << error << std::endl;
  }

#else
  // Get the native handle of the thread
  pthread_t threadHandle = t;

  // Set the thread priority
  int threadPriority = priority;
  // do nothing

#endif
}

void getWork_v2(MiningProfile *miningProf)
{
  net::io_context ioc;
  ssl::context ctx = ssl::context{ssl::context::tlsv12_client};
  load_root_certificates(ctx);

  bool caughtDisconnect = false;

connectionAttempt:
  CHECK_CLOSE;
  isConnected = false;
  {
    std::string connMsg = "Connecting (" + miningProf->host + ":" + miningProf->port + ")";
    logInfo(connMsg);
  }
  try
  {
    // Launch the asynchronous operation
    bool err = false;
    boost::asio::spawn(ioc, std::bind(&do_session_v2, miningProf, std::ref(ioc), std::ref(ctx), std::placeholders::_1),
                       // on completion, spawn will call this function
                       [&](std::exception_ptr ex)
                       {
                         if (ex)
                         {
                           std::rethrow_exception(ex);
                           err = true;
                         }
                       });
    ioc.run();

    if (err)
    {
      setcolor(RED);
      std::cerr << "\nError establishing connections" << std::endl
                << "Will try again in about 10 seconds...\n\n" << std::flush;
      setcolor(BRIGHT_WHITE);
      boost::this_thread::sleep_for(boost::chrono::milliseconds(randomSleepTimeMs()));
      ioc.restart();
      goto connectionAttempt;
    }
    else
    {
      caughtDisconnect = false;
    }
  }
  catch (boost::thread_interrupted&) {
    //std::cout << "Thread was interrupted!" << std::endl;
    ioc.restart();
    return;
  }
  catch (const std::exception& e)
  {
    CHECK_CLOSE;
    setcolor(RED);
    std::cerr << "\nError establishing connections: " << e.what() << std::endl
              << "Will try again in about 10 seconds...\n\n" << std::flush;
    setcolor(BRIGHT_WHITE);
    boost::this_thread::sleep_for(boost::chrono::milliseconds(randomSleepTimeMs()));
    ioc.restart();
    goto connectionAttempt;
  }
  while (isConnected)
  {
    caughtDisconnect = false;
    boost::this_thread::sleep_for(boost::chrono::milliseconds(200));
  }
  CHECK_CLOSE;
  {
    setcolor(RED);
    if (!caughtDisconnect)
      std::cerr << "\nERROR: lost connection" << std::endl
                << "Will try to reconnect in about 10 seconds...\n\n";
    else
      std::cerr << "\nError establishing connection" << std::endl
                << "Will try again in about 10 seconds...\n\n";

    fflush(stdout);
    setcolor(BRIGHT_WHITE);

    rate30sec.clear();
  }
  caughtDisconnect = true;
  CHECK_CLOSE;
  boost::this_thread::sleep_for(boost::chrono::milliseconds(randomSleepTimeMs()));
  ioc.restart();
  goto connectionAttempt;
}

#ifndef DIRTYBIRD_HIP
int main(int argc, char** argv) {
    return dirtybird_main(argc, argv);
}
#endif
