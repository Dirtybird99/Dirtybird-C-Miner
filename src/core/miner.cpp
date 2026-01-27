#include "miner_main.h"
#include "dirtybird-common.hpp"
#include "dirtybird-hugepages.hpp"
#include "numa_optimizer.hpp"
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

#include "miner.h"

#include <random>

#include <hex.h>
#include "algos.h"
#include <thread>

#ifdef DIRTYBIRD_OPENMP
#include <omp.h>
#endif

#include <chrono>

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

#include "reporter.hpp"

#include <coins/miners.hpp>
#include <astrobwtv3/cache_batching.hpp>
#include <astrobwtv3/interleaved_miner.hpp>
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

// DERO Miner: Import uint256_t from KasStratum namespace for difficulty calculations
using KasStratum::uint256_t;

// INITIALIZE COMMON STUFF
algo_config_t current_algo_config;

int reportCounter = 0;
int reportInterval = 3;
int threads = 0;

bool ABORT_MINER = false;
const char *dirtybirdTargetArch = XSTR(CPU_ARCHTARGET);
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

std::atomic<int64_t> counter = 0;
boost::asio::io_context my_context;
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

bool lockThreads = true;
bool g_pcores_only = false;  // Limit mining to P-cores only (for hybrid CPUs)
int g_pcore_count = 0;       // Number of P-core logical processors detected
int g_differential_affinity = 0;  // Differential affinity mode (0=default, 1=P-first, 2=physical-only, 3=balanced)
bool g_show_affinity = false;     // Show detailed core assignments
#if defined(_WIN32)
std::vector<uint32_t> g_pcore_logical_ids;  // Logical core IDs for P-cores only
std::vector<uint32_t> g_ecore_logical_ids;  // Logical core IDs for E-cores
int detectPCores();  // Forward declaration
#endif
int g_omp_threads = 0;  // OpenMP threads per mining thread (0 = auto)

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

// SPSA control - can be disabled with --no-spsa for benchmarking
bool g_use_spsa = true;
bool g_verbose_tune = false;

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
  // {"branch", branchComputeCPU},
  // {"lookup", lookupCompute},
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

uint16_t *lookup2D_global; // Storage for computed values of 2-byte chunks
byte *lookup3D_global;     // Storage for deterministically computed values of 1-byte chunks

using byte = unsigned char;

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
  hipKill();
  ABORT_MINER = true;
  setcolor(BRIGHT_WHITE);
  if(printHashrateOnExit) {
    printf("\n\n%s: %d threads @ %2.2f with %d shares accepted (built with ", miningProfile.coin.coinPrettyName.c_str(), threads, latest_hashrate, accepted);
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

int dirtybird_main(int argc, char** argv)
{
  // test_cshake256();

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
  lookup2D_global = (uint16_t *)malloc_huge_pages(regOps_size * (256 * 256) * sizeof(uint16_t));
  lookup3D_global = (byte *)malloc_huge_pages(branchedOps_size * (256 * 256) * sizeof(byte));

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

  if (vm.count("help"))
  {
    std::cout << opts << std::endl;
    boost::this_thread::sleep_for(boost::chrono::seconds(1));
    return 0;
  }

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

#ifdef DIRTYBIRD_OPENMP
  // Initialize OpenMP thread count globally before mining starts
  // This must happen after both --threads and --omp-threads are parsed
  // IMPORTANT: We set OMP_NUM_THREADS env var because omp_set_num_threads() only
  // affects the calling thread, not worker threads created by std::thread
  {
    int omp_threads;
    if (g_omp_threads > 0) {
      omp_threads = g_omp_threads;
    } else {
      // Auto mode: use 1 OMP thread for less oversubscription
      // Testing shows OMP=1 performs better than OMP=2 for this workload
      omp_threads = 1;
      g_omp_threads = 1;  // Update global so mine_dero.cpp uses consistent value
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
      int mining_threads = threads > 0 ? threads : static_cast<int>(std::thread::hardware_concurrency());
      printf("OpenMP: %d threads per miner (max ~%d total with %d miners)\n",
             omp_threads, mining_threads * omp_threads, mining_threads);
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }
  }
#endif

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

  // We can do this because we've set default in terminal.hpp
  tuneWarmupSec = vm["tune-warmup"].as<int>();
  tuneDurationSec = vm["tune-duration"].as<int>();

  mine_time = vm["mine-time"].as<int>();

  // Ensure we capture *all* of the other options before we start using goto
  if (vm.count("test-dero"))
  {
    #if defined(DIRTYBIRD_ASTROBWTV3)
    // temporary for optimization fishing:
    mapZeroes();
    // end of temporary section

    #if defined(USE_ASTRO_SPSA)
      initSPSA();
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
fillBlanks:
{
  std::string localSymbol;
  if (miningProfile.coin.coinId == unknownCoin.coinId)
  {
    setcolor(CYAN);
    printf("%s\n", coinPrompt);
    fflush(stdout);
    setcolor(BRIGHT_WHITE);

    std::string cmdLine;
    std::getline(std::cin, cmdLine);
    if (cmdLine != "" && cmdLine.find_first_not_of(' ') != std::string::npos)
    {
      localSymbol = cmdLine;
      std::transform(localSymbol.begin(), localSymbol.end(), localSymbol.begin(), ::toupper);
    }
    else
    {
      localSymbol = "DERO";
      setcolor(BRIGHT_YELLOW);
      printf("Default value will be used: %s\n\n", "DERO");
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }
  }

  for(int x = 0; x < COIN_COUNT; x++) {
    if(boost::iequals(coins[x].coinSymbol, localSymbol)) {
      miningProfile.coin = coins[x];

      setcolor(BRIGHT_YELLOW);
      printf(" Set to mine %s\n\n", miningProfile.coin.coinPrettyName.c_str());
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }
  }
  if(miningProfile.coin.coinId == unknownCoin.coinId)
  {
    setcolor(RED);
    std::cout << "ERROR: Invalid coin symbol: " << localSymbol << std::endl << std::flush;
    setcolor(BRIGHT_YELLOW);
    printf("Supported symbols are:\n");
    for(int x = 0; x < COIN_COUNT; x++) {
      printf("%s\n", coins[x].coinSymbol.c_str());
    }
    printf("\n");
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
    miningProfile.coin = unknownCoin;
    goto fillBlanks;
  }

  // DERO Miner: Stratum optional (not required for DERO)

  int i = 0;
  std::vector<std::string *> stringParams = {&miningProfile.host, &miningProfile.port, &miningProfile.wallet};
  std::vector<const char *> stringDefaults = {"", "", ""};  // No defaults - user must provide
  std::vector<const char *> stringPrompts = {daemonPrompt, portPrompt, walletPrompt};
  for (std::string *param : stringParams)
  {
    if (param->empty())
    {
      setcolor(CYAN);
      printf("%s\n", stringPrompts[i]);
      fflush(stdout);
      setcolor(BRIGHT_WHITE);

      std::string cmdLine;
      std::getline(std::cin, cmdLine);
      if (cmdLine != "" && cmdLine.find_first_not_of(' ') != std::string::npos)
      {
        *param = cmdLine;
      }
      else
      {
        *param = stringDefaults[i];
        setcolor(BRIGHT_YELLOW);
        printf("Default value will be used: %s\n\n", (*param).c_str());
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
      }

      if (param == &miningProfile.host) {
        miningProfile.setPoolAddress(miningProfile.host);
      }
    }
    i++;
  }


  switch (miningProfile.coin.miningAlgo)
  {
    #if defined(DIRTYBIRD_YESPOWER)
    case ALGO_YESPOWER:
      miningProfile.protocol = PROTO_BTC_STRATUM;

      if (miningProfile.coin.coinId == coins[COIN_ADVC].coinId) {
        current_algo_config = algo_configs[CONFIG_ENDIAN_YESPOWER];

        initADVCParams(&currentYespowerParams);
        initADVCParams(&devYespowerParams);
      }
      break;
    #endif
    default:
      break;
  }

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
      if(!setAstroAlgo(noTune)) {
        throw po::validation_error(po::validation_error::invalid_option_value, "no-tune");
      }
    } else {
      astroTune(threads, tuneWarmupSec, tuneDurationSec);
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
}

Mining:
{
  printHashrateOnExit = true;

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
    if (miningProfile.coin.miningAlgo == ALGO_ASTROBWTV3) {
      initSPSA();
    }
  #endif

  unsigned int n = std::thread::hardware_concurrency();
  (void)n; // DERO Miner: Suppress unused variable warning (was used by RandomX)

  boost::thread GETWORK(getWork_v2, &miningProfile);

  // Create worker threads and set CPU affinity
  boost::thread minerThreads[threads];
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
    // Start worker threads silently (thread count already shown in startup info)
    for (int i = 0; i < threads; i++)
    {
      minerThreads[i] = boost::thread(getMiningFunc(miningProfile.coin.miningAlgo, false), i + 1);

      if (lockThreads)
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
  for (unsigned i = 0; i < threads; ++i) {
    minerThreads[i].interrupt();
    minerThreads[i].join();
  }

  return EXIT_SUCCESS;
}
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

void setAffinity(boost::thread::native_handle_type t, uint64_t core)
{
#if defined(_WIN32)
  static std::vector<CoreInfo> topology = getCoreTopology();
  static std::vector<DWORD> affinityOrder;
  static int cachedMode = -1;

  // Rebuild affinity order if mode changed or first time
  if (cachedMode != g_differential_affinity || affinityOrder.empty()) {
    cachedMode = g_differential_affinity;
    affinityOrder.clear();

    std::map<DWORD, std::vector<DWORD>> pCoreMap, eCoreMap;

    // Group logical cores by physical core, separating P and E cores
    for (size_t i = 0; i < topology.size(); i++) {
      if (topology[i].isPCore) {
        pCoreMap[topology[i].physicalCore].push_back(static_cast<DWORD>(i));
      } else {
        eCoreMap[topology[i].physicalCore].push_back(static_cast<DWORD>(i));
      }
    }

    // Build ordering based on differential affinity mode
    switch (g_differential_affinity) {
      case 1:  // P-cores first: P-primaries -> P-HT -> E-cores
        // P-core primary threads
        for (auto& pair : pCoreMap) {
          affinityOrder.push_back(pair.second[0]);
        }
        // P-core hyperthreads
        for (auto& pair : pCoreMap) {
          for (size_t i = 1; i < pair.second.size(); i++) {
            affinityOrder.push_back(pair.second[i]);
          }
        }
        // E-cores last (unless p-cores-only)
        if (!g_pcores_only) {
          for (auto& pair : eCoreMap) {
            affinityOrder.push_back(pair.second[0]);
          }
        }
        break;

      case 2:  // Physical only: P-primaries -> E-cores (no HT)
        // P-core primary threads only
        for (auto& pair : pCoreMap) {
          affinityOrder.push_back(pair.second[0]);
        }
        // E-cores (unless p-cores-only)
        if (!g_pcores_only) {
          for (auto& pair : eCoreMap) {
            affinityOrder.push_back(pair.second[0]);
          }
        }
        // Skip hyperthreads entirely
        break;

      case 3:  // Balanced: interleave P and E cores
        {
          // Interleave P-primary and E-cores first
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
          // Then P-core hyperthreads
          for (auto& pair : pCoreMap) {
            for (size_t i = 1; i < pair.second.size(); i++) {
              affinityOrder.push_back(pair.second[i]);
            }
          }
        }
        break;

      case 0:  // Default: P-primaries -> E-cores -> P-HT
      default:
        // P-core primary threads
        for (auto& pair : pCoreMap) {
          affinityOrder.push_back(pair.second[0]);
        }
        // E-cores (unless p-cores-only)
        if (!g_pcores_only) {
          for (auto& pair : eCoreMap) {
            affinityOrder.push_back(pair.second[0]);
          }
        }
        // P-core hyperthreads last
        for (auto& pair : pCoreMap) {
          for (size_t i = 1; i < pair.second.size(); i++) {
            affinityOrder.push_back(pair.second[i]);
          }
        }
        break;
    }

    // Print affinity order if show-affinity is enabled
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

  // In P-cores-only mode, use the pre-computed P-core logical IDs directly
  DWORD targetCore;
  DWORD logicalCoreIndex;

  if (g_pcores_only && core < g_pcore_logical_ids.size()) {
    // Direct mapping to P-core logical IDs
    targetCore = g_pcore_logical_ids[core];
    logicalCoreIndex = 0;  // Not used in this path

    // Find the topology entry for this core
    for (size_t i = 0; i < topology.size(); i++) {
      if (topology[i].logicalCore == targetCore) {
        logicalCoreIndex = static_cast<DWORD>(i);
        break;
      }
    }
  } else if (core < affinityOrder.size()) {
    // Use the differential affinity ordering
    logicalCoreIndex = affinityOrder[core];
    targetCore = topology[logicalCoreIndex].logicalCore;
  } else {
    setcolor(RED);
    std::cerr << "Core ID " << core << " exceeds available cores ("
              << (g_pcores_only ? g_pcore_logical_ids.size() : affinityOrder.size()) << ")" << std::endl;
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
    return;
  }

  HANDLE threadHandle = t;

  {
    std::lock_guard<std::mutex> lock(threadMapMutex);
    threadToPhysicalCore[core] = topology[logicalCoreIndex].physicalCore;
  }

  // Get the total logical processor count across all groups
  DWORD numGroups = GetActiveProcessorGroupCount();
  DWORD totalProcessors = 0;

  for (DWORD i = 0; i < numGroups; i++) {
    totalProcessors += GetMaximumProcessorCount(i);
  }

  if (targetCore < totalProcessors) {
    // Calculate group and processor within the group
    DWORD group = 0;
    DWORD processorInGroup = targetCore;

    // Find the correct group
    for (DWORD i = 0; i < numGroups; i++) {
      DWORD groupSize = GetMaximumProcessorCount(i);
      if (processorInGroup < groupSize) {
        group = i;
        break;
      }
      processorInGroup -= groupSize;
    }

    if (group >= numGroups) {
      setcolor(RED);
      std::cerr << "Invalid processor group calculated" << std::endl;
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
      return;
    }

    // Use group affinity for all cases for consistency
    GROUP_AFFINITY groupAffinity = {0};
    groupAffinity.Group = static_cast<WORD>(group);
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
  } else {
    setcolor(RED);
    std::cerr << "Logical core ID " << targetCore
              << " exceeds available logical cores (" << totalProcessors << ")" << std::endl;
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

  if (pthread_setaffinity_np(threadHandle, sizeof(cpu_set_t), &cpuset) != 0)
  {
    std::cerr << "Failed to set CPU affinity for thread" << std::endl;
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
  }
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