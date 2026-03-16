#include <endian.hpp>
#include <inttypes.h>

#ifdef _MSC_VER
#include <io.h>
#else
#include <unistd.h>
#endif

#include <boost/thread.hpp>
#include <algorithm>
#include <bitset>
#include <iostream>
#include <fstream>
#include <thread>

#include <fnv1a.h>
#include <xxhash64.h>
#include "astrobwtv3.h"
#include "dirtybird-hugepages.hpp"
#include "astrotest.hpp"
#include "memory_optimized.hpp"
#include "derobwt.hpp"
#include "branched_AVX2.h"

#include <unordered_map>
#include <array>
#include <algorithm>

#if defined(__x86_64__)
  #include <xmmintrin.h>
#endif
#if defined(__aarch64__)
  #include "astro_aarch64.hpp"
#endif

#include <random>
#include <chrono>

// #include <sodium.h>
// Salsa20 included via astrobwtv3.h -> astroworker.h (salsa20_simd.h)

// #include <alcp/digest.h>

#include <highwayhash/sip_hash.h>
#include <filesystem>
#include <functional>
#include "lookupcompute.h"
#include "lookup_full.hpp"
#include "lookup_mode.hpp"
#include "branch_tables.hpp"
#if USE_LOOKUP_TABLES
  #include "lookup_tables.hpp"
#endif

LookupMode g_lookup_mode = LOOKUP_MODE_3D;
extern bool useLookupMine;

#if defined(USE_ASTRO_SPSA)
  #include "spsa.hpp"
  static std::atomic<uint64_t> spsa_hits_total{0};
  static std::atomic<uint64_t> spsa_misses_total{0};

  // SPSA_STATS: Set to 1 to enable atomic statistics tracking (small perf overhead)
  #ifndef SPSA_STATS
    #define SPSA_STATS 0
  #endif

  // SPSA_PROFILE: Set to 1 to enable timing instrumentation (significant overhead!)
  #ifndef SPSA_PROFILE
    #define SPSA_PROFILE 0  // Disabled - found SPSA takes 93% of time
  #endif

  #if SPSA_PROFILE
    #include <atomic>
    static std::atomic<uint64_t> profile_setup_ns{0};
    static std::atomic<uint64_t> profile_wolf_ns{0};
    static std::atomic<uint64_t> profile_spsa_ns{0};
    static std::atomic<uint64_t> profile_count{0};

    static inline void log_profile_stats() {
      uint64_t count = profile_count.fetch_add(1) + 1;
      if (count % 10000 == 0) {
        double setup_us = profile_setup_ns.load() / 1000.0 / count;
        double wolf_us = profile_wolf_ns.load() / 1000.0 / count;
        double spsa_us = profile_spsa_ns.load() / 1000.0 / count;
        double total_us = setup_us + wolf_us + spsa_us;
        printf("Profile: setup=%.1fus (%.0f%%) wolf=%.1fus (%.0f%%) spsa=%.1fus (%.0f%%) total=%.1fus\n",
               setup_us, 100.0 * setup_us / total_us,
               wolf_us, 100.0 * wolf_us / total_us,
               spsa_us, 100.0 * spsa_us / total_us,
               total_us);
        fflush(stdout);
      }
    }
  #endif

  #if SPSA_STATS
    #include <atomic>
    static std::atomic<bool> spsa_logged{false};
    static std::atomic<uint64_t> spsa_hits{0};
    static std::atomic<uint64_t> spsa_misses{0};
    static std::atomic<uint64_t> spsa_total{0};

    static inline void log_spsa_enabled() {
      bool expected = false;
      if (spsa_logged.compare_exchange_strong(expected, true)) {
        printf("\n*** SPSA ENABLED - Using Stamped Permutation Suffix Array ***\n");
        fflush(stdout);
      }
    }

    static inline void log_spsa_stats() {
      uint64_t total = spsa_total.fetch_add(1) + 1;
      if (total % 10000 == 0) {
        uint64_t hits = spsa_hits.load();
        uint64_t misses = spsa_misses.load();
        double hit_rate = (hits + misses > 0) ? (100.0 * hits / (hits + misses)) : 0.0;
        printf("SPSA Stats: %llu hits, %llu misses (%.2f%% hit rate)\n",
               (unsigned long long)hits, (unsigned long long)misses, hit_rate);
        fflush(stdout);
      }
    }
    #define SPSA_HIT() do { \
      if (g_spsa_hit_counters) { \
        spsa_hits.fetch_add(1, std::memory_order_relaxed); \
        spsa_hits_total.fetch_add(1, std::memory_order_relaxed); \
      } \
    } while (0)
    #define SPSA_MISS() do { \
      if (g_spsa_hit_counters) { \
        spsa_misses.fetch_add(1, std::memory_order_relaxed); \
        spsa_misses_total.fetch_add(1, std::memory_order_relaxed); \
      } \
    } while (0)
    #define SPSA_LOG_ENABLED() log_spsa_enabled()
    #define SPSA_LOG_STATS() log_spsa_stats()
  #else
    // No-op macros when stats are disabled for maximum performance
    static std::atomic<bool> spsa_banner_printed{false};
    static inline void print_spsa_banner_once() {
      bool expected = false;
      if (spsa_banner_printed.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
        printf("\n*** SPSA ENABLED - Using Stamped Permutation Suffix Array ***\n");
        fflush(stdout);
      }
    }
    #define SPSA_HIT() do { \
      if (g_spsa_hit_counters) { \
        spsa_hits_total.fetch_add(1, std::memory_order_relaxed); \
      } \
    } while (0)
    #define SPSA_MISS() do { \
      if (g_spsa_hit_counters) { \
        spsa_misses_total.fetch_add(1, std::memory_order_relaxed); \
      } \
    } while (0)
    #define SPSA_LOG_ENABLED() print_spsa_banner_once()
    #define SPSA_LOG_STATS() ((void)0)
  #endif
#else
  #define MINPREFLEN 4
#endif

static inline int classifySpsaCoreLenBin(int data_len) {
  if (data_len < 65536) {
    return 0;
  }
  if (data_len < 67584) {
    return 1;
  }
  if (data_len < 69632) {
    return 2;
  }
  return 3;
}

#if defined(USE_ASTRO_SPSA)
static inline void prefetchSpsaBucketArrays(workerData &worker) {
  if (g_spsa_bucket_prefetch == SPSA_BUCKET_PREFETCH_OFF) {
    return;
  }

  // "light" touches fewer pages, "full" touches every 16 rows.
  const int stride = (g_spsa_bucket_prefetch == SPSA_BUCKET_PREFETCH_FULL) ? 16 : 64;
  for (int pf = 0; pf < 256; pf += stride) {
    __builtin_prefetch(&worker.buckets_d[pf][0], 1, 2);
    __builtin_prefetch(&worker.bHeads[pf][0], 1, 2);
    __builtin_prefetch(&worker.bHeadIdx[pf][0], 1, 2);
  }
}
#endif

namespace {
constexpr size_t kSpsaOpFamilyNonBranched = 0;
constexpr size_t kSpsaOpFamilyBranched = 1;
constexpr size_t kSpsaOpFamilyOp253 = 2;
constexpr size_t kSpsaOpFamilyRc4 = 3;
constexpr size_t kSpsaOpFamilyCount = 4;

static std::atomic<bool> g_phase_telemetry_enabled{false};
static std::atomic<uint64_t> g_phase_hashes_total{0};
static std::atomic<uint64_t> g_phase_data_len_sum_total{0};
static std::atomic<uint64_t> g_phase_prep_ns_total{0};
static std::atomic<uint64_t> g_phase_wolf_ns_total{0};
static std::atomic<uint64_t> g_phase_spsa_ns_total{0};
static std::atomic<uint64_t> g_phase_spsa_prefetch_ns_total{0};
static std::atomic<uint64_t> g_phase_spsa_hit_copy_ns_total{0};
static std::atomic<uint64_t> g_phase_spsa_miss_hash_ns_total{0};
static std::atomic<uint64_t> g_phase_spsa_core_ns_total{0};
static std::atomic<uint64_t> g_phase_spsa_core_calls_total{0};
static std::array<std::atomic<uint64_t>, 4> g_phase_spsa_core_bin_calls_total{};
static std::array<std::atomic<uint64_t>, 4> g_phase_spsa_core_bin_ns_total{};
static std::array<std::atomic<uint64_t>, kSpsaOpFamilyCount> g_phase_spsa_op_family_calls_total{};
static std::array<std::atomic<uint64_t>, kSpsaOpFamilyCount> g_phase_spsa_op_family_bytes_total{};
static std::atomic<uint64_t> g_phase_sa_ns_total{0};
static std::atomic<uint64_t> g_phase_hash_ns_total{0};
static std::atomic<uint64_t> g_phase_total_ns_total{0};
static std::atomic<uint64_t> g_phase_sa_encode_ns_total{0};
static std::atomic<uint64_t> g_phase_sa_radix_ns_total{0};
static std::atomic<uint64_t> g_phase_sa_collision_ns_total{0};
static std::atomic<uint64_t> g_phase_sa_copy_ns_total{0};
static std::atomic<uint64_t> g_lookup_smart_branched_total{0};
static std::atomic<uint64_t> g_lookup_smart_path_lut{0};
static std::atomic<uint64_t> g_lookup_smart_path_avx2{0};
static std::array<std::atomic<uint64_t>, 33> g_lookup_smart_span_hist{};

static inline size_t classifySpsaOpFamily(uint8_t op) {
  if (op == 253) {
    return kSpsaOpFamilyOp253;
  }
  if (op >= 254) {
    return kSpsaOpFamilyRc4;
  }
  return g_is_branched[op] ? kSpsaOpFamilyBranched : kSpsaOpFamilyNonBranched;
}

static bool lookupSmartAvx2Available() {
#if defined(__x86_64__) || defined(_M_X64)
  static const bool has_avx2 = branch_tables::avx2_available();
  return has_avx2;
#else
  return false;
#endif
}
}  // namespace

size_t classifySpsaOpFamilyForTelemetry(uint8_t op) {
  return classifySpsaOpFamily(op);
}

void addSpsaOpFamilyTelemetryBatch(const std::array<uint64_t, 4>& calls,
                                   const std::array<uint64_t, 4>& bytes) {
  for (size_t i = 0; i < kSpsaOpFamilyCount; ++i) {
    g_phase_spsa_op_family_calls_total[i].fetch_add(calls[i], std::memory_order_relaxed);
    g_phase_spsa_op_family_bytes_total[i].fetch_add(bytes[i], std::memory_order_relaxed);
  }
}

extern "C"
{
  #include "divsufsort_private.h"
  #include "divsufsort.h"
  #include "sa_instrumentation.h"
  #ifdef USE_CUSTOM_SA
    #include "custom_sa_70kb.h"
  #endif
  #ifdef USE_LIBSAIS
    #include "libsais.h"
  #endif
}

/* SA_FUNCTION: Dispatch to best available suffix array implementation
 * Priority:
 * USE_DLUNA_RADIX_SA > USE_BUCKET_SA > USE_RADIX_SA > USE_CUSTOM_SA > USE_LIBSAIS > divsufsort
 *
 * libsais 2.10.4 is state-of-the-art SA construction (2021-2025)
 * - No external bucket arrays needed (handles internally)
 * - Supports context reuse for memory efficiency
 * - ~10-20% faster than divsufsort for our workload size
 */
#ifdef USE_DLUNA_RADIX_SA
  #include "dluna_radix_sa.h"
  #define SA_FUNCTION dluna_radix_sa::radix_sort_sa
#elif defined(USE_BUCKET_SA)
  #include "bucket_sa.h"
  #define SA_FUNCTION bucket_sa::bucket_sort_sa
#elif defined(USE_RADIX_SA)
  #include "radix_sa.h"
  #define SA_FUNCTION radix_sa::radix_sort_sa
#elif defined(USE_CUSTOM_SA)
  #define SA_FUNCTION custom_sa_70kb
#elif defined(USE_LIBSAIS)
  // libsais wrapper: adapts libsais API to match divsufsort signature
  // The bucket arrays (bA, bB) are unused - libsais handles them internally
  static inline int32_t libsais_wrapper(const uint8_t* T, int32_t* SA, int32_t n,
                                        int32_t* /*bucket_A*/, int32_t* /*bucket_B*/) {
    // Thread-local context for memory reuse across calls
    static thread_local void* tl_sais_ctx = nullptr;
    if (tl_sais_ctx == nullptr) {
      tl_sais_ctx = libsais_create_ctx();
    }
    // fs=0: no extra space needed, freq=NULL: don't compute frequencies
    return libsais_ctx(tl_sais_ctx, T, SA, n, 0, nullptr);
  }
  #define SA_FUNCTION libsais_wrapper
#else
  #define SA_FUNCTION divsufsort
#endif

#include <utility>

#include <hex.h>
#include <openssl/rc4.h>
#include "rc4_avx512.hpp"
#include "spsa_state.hpp"

#include <fstream>

#include <bit>
// #include <libcubwt.cuh>
// #include <device_sa.cuh>
#include <lookup.h>
// #include <sacak-lcp.h>
#if defined(__x86_64__) || defined(_M_X64)
  #include <immintrin.h>
  #include <emmintrin.h>
  #ifdef _MSC_VER
    #include <intrin.h>
  #else
    #include <cpuid.h>
  #endif
#endif


using byte = unsigned char;

int ops[256];

// DERO Miner: Removed unused static lookup tables that caused startup failure
// These were ~27MB of BSS that Windows couldn't allocate at load time
// The dynamic allocations in miner.cpp (lookup1D_global, lookup3D_global) are used instead

std::vector<byte> opsA;
std::vector<byte> opsB;

bool debugOpOrder = false;

// AVX512 runtime detection for wolfPermute
bool g_has_avx512 = false;

void detectAVX512() {
#if defined(__x86_64__) || defined(_M_X64)
    // Check for AVX512F and AVX512BW support using CPUID
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;

    // Use __get_cpuid_count from <cpuid.h> on GCC/Clang
    // or __cpuidex on MSVC
    #if defined(_MSC_VER)
    int cpuInfo[4] = {0};
    __cpuid(cpuInfo, 0);
    if (cpuInfo[0] < 7) {
        g_has_avx512 = false;
        return;
    }
    __cpuidex(cpuInfo, 7, 0);
    ebx = cpuInfo[1];
    #elif defined(__GNUC__) || defined(__clang__)
    // Check max supported leaf first
    unsigned int maxLeaf = __get_cpuid_max(0, NULL);
    if (maxLeaf < 7) {
        g_has_avx512 = false;
        return;
    }
    // Query leaf 7, subleaf 0 for extended features
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        g_has_avx512 = false;
        return;
    }
    #else
    g_has_avx512 = false;
    return;
    #endif

    // Check EBX for AVX512F (bit 16) and AVX512BW (bit 30)
    bool has_avx512f = (ebx & (1 << 16)) != 0;
    bool has_avx512bw = (ebx & (1 << 30)) != 0;

    g_has_avx512 = has_avx512f && has_avx512bw;

    if (g_has_avx512) {
        printf("AVX512 detected: Using optimized 512-bit branch kernels\n");
    }
#else
    g_has_avx512 = false;
#endif
}

// Default to wolfCompute - AVX2 branch processing is faster than lookup tables.
// Lookup tables available via --no-tune lookup but benchmark shows AVX2 wins
// because wolf_permute_avx2 processes 32 bytes/instruction vs 1 byte/lookup.
void (*astroCompFunc)(workerData &worker, bool isTest, int wIndex) = wolfCompute;

const char* getCurrentAstroAlgoName() {
  for (size_t i = 0; i < numAstroFuncs; i++) {
    if (allAstroFuncs[i].funcPtr == astroCompFunc) {
      return allAstroFuncs[i].funcName.c_str();
    }
  }
  return "unknown";
}

const char* getSABackendName() {
#if defined(USE_DLUNA_RADIX_SA)
  return "dluna-radix";
#elif defined(USE_BUCKET_SA)
  return "bucket";
#elif defined(USE_RADIX_SA)
  return "radix";
#elif defined(USE_CUSTOM_SA)
  return "custom-sa";
#elif defined(USE_LIBSAIS)
  return "libsais";
#else
  return "divsufsort";
#endif
}

uint64_t getSPSAHits() {
#if defined(USE_ASTRO_SPSA)
  return spsa_hits_total.load(std::memory_order_relaxed);
#else
  return 0;
#endif
}

uint64_t getSPSAMisses() {
#if defined(USE_ASTRO_SPSA)
  return spsa_misses_total.load(std::memory_order_relaxed);
#else
  return 0;
#endif
}

void setPhaseTelemetryEnabled(bool enabled) {
  g_phase_telemetry_enabled.store(enabled, std::memory_order_relaxed);
}

bool isPhaseTelemetryEnabled() {
  return g_phase_telemetry_enabled.load(std::memory_order_relaxed);
}

void resetPhaseTelemetry() {
  g_phase_hashes_total.store(0, std::memory_order_relaxed);
  g_phase_data_len_sum_total.store(0, std::memory_order_relaxed);
  g_phase_prep_ns_total.store(0, std::memory_order_relaxed);
  g_phase_wolf_ns_total.store(0, std::memory_order_relaxed);
  g_phase_spsa_ns_total.store(0, std::memory_order_relaxed);
  g_phase_spsa_prefetch_ns_total.store(0, std::memory_order_relaxed);
  g_phase_spsa_hit_copy_ns_total.store(0, std::memory_order_relaxed);
  g_phase_spsa_miss_hash_ns_total.store(0, std::memory_order_relaxed);
  g_phase_spsa_core_ns_total.store(0, std::memory_order_relaxed);
  g_phase_spsa_core_calls_total.store(0, std::memory_order_relaxed);
  for (size_t i = 0; i < g_phase_spsa_core_bin_calls_total.size(); ++i) {
    g_phase_spsa_core_bin_calls_total[i].store(0, std::memory_order_relaxed);
    g_phase_spsa_core_bin_ns_total[i].store(0, std::memory_order_relaxed);
  }
  for (size_t i = 0; i < g_phase_spsa_op_family_calls_total.size(); ++i) {
    g_phase_spsa_op_family_calls_total[i].store(0, std::memory_order_relaxed);
    g_phase_spsa_op_family_bytes_total[i].store(0, std::memory_order_relaxed);
  }
  g_phase_sa_ns_total.store(0, std::memory_order_relaxed);
  g_phase_hash_ns_total.store(0, std::memory_order_relaxed);
  g_phase_total_ns_total.store(0, std::memory_order_relaxed);
  g_phase_sa_encode_ns_total.store(0, std::memory_order_relaxed);
  g_phase_sa_radix_ns_total.store(0, std::memory_order_relaxed);
  g_phase_sa_collision_ns_total.store(0, std::memory_order_relaxed);
  g_phase_sa_copy_ns_total.store(0, std::memory_order_relaxed);
}

void resetLookupTelemetry() {
  g_lookup_smart_branched_total.store(0, std::memory_order_relaxed);
  g_lookup_smart_path_lut.store(0, std::memory_order_relaxed);
  g_lookup_smart_path_avx2.store(0, std::memory_order_relaxed);
  for (size_t i = 0; i < g_lookup_smart_span_hist.size(); ++i) {
    g_lookup_smart_span_hist[i].store(0, std::memory_order_relaxed);
  }
}

void addSABreakdownTelemetry(uint64_t encode_ns, uint64_t radix_ns, uint64_t collision_ns, uint64_t copy_ns) {
  if (!isPhaseTelemetryEnabled()) {
    return;
  }

  g_phase_sa_encode_ns_total.fetch_add(encode_ns, std::memory_order_relaxed);
  g_phase_sa_radix_ns_total.fetch_add(radix_ns, std::memory_order_relaxed);
  g_phase_sa_collision_ns_total.fetch_add(collision_ns, std::memory_order_relaxed);
  g_phase_sa_copy_ns_total.fetch_add(copy_ns, std::memory_order_relaxed);
}

AstroPhaseTelemetrySnapshot getPhaseTelemetrySnapshot() {
  AstroPhaseTelemetrySnapshot snapshot{};
  snapshot.hashes = g_phase_hashes_total.load(std::memory_order_relaxed);
  snapshot.data_len_sum = g_phase_data_len_sum_total.load(std::memory_order_relaxed);
  snapshot.spsa_hits = getSPSAHits();
  snapshot.spsa_misses = getSPSAMisses();
  snapshot.prep_ns = g_phase_prep_ns_total.load(std::memory_order_relaxed);
  snapshot.wolf_ns = g_phase_wolf_ns_total.load(std::memory_order_relaxed);
  snapshot.spsa_call_ns = g_phase_spsa_ns_total.load(std::memory_order_relaxed);
  snapshot.spsa_prefetch_ns = g_phase_spsa_prefetch_ns_total.load(std::memory_order_relaxed);
  snapshot.spsa_hit_copy_ns = g_phase_spsa_hit_copy_ns_total.load(std::memory_order_relaxed);
  snapshot.spsa_miss_hash_ns = g_phase_spsa_miss_hash_ns_total.load(std::memory_order_relaxed);
  snapshot.spsa_core_ns = g_phase_spsa_core_ns_total.load(std::memory_order_relaxed);
  snapshot.spsa_core_calls = g_phase_spsa_core_calls_total.load(std::memory_order_relaxed);
  snapshot.spsa_core_bin0_calls = g_phase_spsa_core_bin_calls_total[0].load(std::memory_order_relaxed);
  snapshot.spsa_core_bin0_ns = g_phase_spsa_core_bin_ns_total[0].load(std::memory_order_relaxed);
  snapshot.spsa_core_bin1_calls = g_phase_spsa_core_bin_calls_total[1].load(std::memory_order_relaxed);
  snapshot.spsa_core_bin1_ns = g_phase_spsa_core_bin_ns_total[1].load(std::memory_order_relaxed);
  snapshot.spsa_core_bin2_calls = g_phase_spsa_core_bin_calls_total[2].load(std::memory_order_relaxed);
  snapshot.spsa_core_bin2_ns = g_phase_spsa_core_bin_ns_total[2].load(std::memory_order_relaxed);
  snapshot.spsa_core_bin3_calls = g_phase_spsa_core_bin_calls_total[3].load(std::memory_order_relaxed);
  snapshot.spsa_core_bin3_ns = g_phase_spsa_core_bin_ns_total[3].load(std::memory_order_relaxed);
  snapshot.spsa_op_family_nonbranched_calls = g_phase_spsa_op_family_calls_total[kSpsaOpFamilyNonBranched].load(std::memory_order_relaxed);
  snapshot.spsa_op_family_nonbranched_bytes = g_phase_spsa_op_family_bytes_total[kSpsaOpFamilyNonBranched].load(std::memory_order_relaxed);
  snapshot.spsa_op_family_branched_calls = g_phase_spsa_op_family_calls_total[kSpsaOpFamilyBranched].load(std::memory_order_relaxed);
  snapshot.spsa_op_family_branched_bytes = g_phase_spsa_op_family_bytes_total[kSpsaOpFamilyBranched].load(std::memory_order_relaxed);
  snapshot.spsa_op_family_op253_calls = g_phase_spsa_op_family_calls_total[kSpsaOpFamilyOp253].load(std::memory_order_relaxed);
  snapshot.spsa_op_family_op253_bytes = g_phase_spsa_op_family_bytes_total[kSpsaOpFamilyOp253].load(std::memory_order_relaxed);
  snapshot.spsa_op_family_rc4_calls = g_phase_spsa_op_family_calls_total[kSpsaOpFamilyRc4].load(std::memory_order_relaxed);
  snapshot.spsa_op_family_rc4_bytes = g_phase_spsa_op_family_bytes_total[kSpsaOpFamilyRc4].load(std::memory_order_relaxed);
  snapshot.sa_fallback_ns = g_phase_sa_ns_total.load(std::memory_order_relaxed);
  snapshot.final_hash_ns = g_phase_hash_ns_total.load(std::memory_order_relaxed);
  snapshot.total_ns = g_phase_total_ns_total.load(std::memory_order_relaxed);
  snapshot.sa_encode_ns = g_phase_sa_encode_ns_total.load(std::memory_order_relaxed);
  snapshot.sa_radix_ns = g_phase_sa_radix_ns_total.load(std::memory_order_relaxed);
  snapshot.sa_collision_ns = g_phase_sa_collision_ns_total.load(std::memory_order_relaxed);
  snapshot.sa_copy_ns = g_phase_sa_copy_ns_total.load(std::memory_order_relaxed);
  return snapshot;
}

AstroLookupTelemetrySnapshot getLookupTelemetrySnapshot() {
  AstroLookupTelemetrySnapshot snapshot{};
  snapshot.smart_branched_total = g_lookup_smart_branched_total.load(std::memory_order_relaxed);
  snapshot.smart_path_lut = g_lookup_smart_path_lut.load(std::memory_order_relaxed);
  snapshot.smart_path_avx2 = g_lookup_smart_path_avx2.load(std::memory_order_relaxed);
  for (size_t i = 0; i < g_lookup_smart_span_hist.size(); ++i) {
    snapshot.smart_span_hist[i] = g_lookup_smart_span_hist[i].load(std::memory_order_relaxed);
  }
  return snapshot;
}

void saveBufferToFile(const std::string& filename, const byte* buffer, size_t size) {
    // Generate unique filename using timestamp
    std::string timestamp = std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::chrono::steady_clock::now().time_since_epoch()).count());
    std::string unique_filename = "tests/worker_sData_snapshot_" + timestamp;

    std::ofstream file(unique_filename, std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(buffer), size);
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}

// TODO: Implement dynamic SIMD checks for branchCompute
/*
void checkSIMDSupport() {
    // Setup a function pointer to detect AVX2 
    void (*func_ptr)() = nullptr;
#ifdef __AVX2__
    func_ptr = __builtin_cpu_supports("avx2");
#endif
    if (func_ptr && func_ptr()) {
        // AVX2 is supported - use AVX2 intrinsics
    } else {
        // Setup a function pointer to detect SSE2
        func_ptr = nullptr; 
#ifdef __SSE2__ 
        func_ptr = __builtin_cpu_supports("sse2"); 
#endif
        if (func_ptr && func_ptr()) {
            // SSE2 is supported - use SSE2 intrinsics
        } else {
            // Use scalar code
        }
    }
}
*/

template <std::size_t N>
inline void generateRandomBytesForTune(std::uint8_t (&iv_buff)[N])
{
  auto const hes = std::random_device{}();

  using random_bytes_engine = std::independent_bits_engine<std::default_random_engine,
                                                           CHAR_BIT, unsigned short>;

  random_bytes_engine rbe;
  rbe.seed(hes);

  std::generate(std::begin(iv_buff), std::end(iv_buff), std::ref(rbe));
}

bool setAstroAlgo(std::string desiredAlgo) {
  bool toRet = false;
  for(int x = 0; x < numAstroFuncs; x++) {
    if(desiredAlgo.compare(allAstroFuncs[x].funcName) == 0) {
      printf("Setting AstroBWTv3 override: %s\n", allAstroFuncs[x].funcName.c_str());
      astroCompFunc = allAstroFuncs[x].funcPtr;
      toRet = true;
      break;
    }
  }
  if(!toRet) {
    printf("Unrecognized AstroBWTv3 algo: %s\nAllowed options are: ", desiredAlgo.c_str());
    for(int x = 0; x < numAstroFuncs; x++) {
      printf("%s ", allAstroFuncs[x].funcName.c_str());
    }
    printf("\n");
  }
  return toRet;
}

void astroTune(int num_threads, int tuneWarmupSec, int tuneDurationSec) {
  int64_t tuneWarmupMs = tuneWarmupSec * 1000;
  int64_t tuneDurationMs = tuneDurationSec * 1000;

  int totalTuneTime = numAstroFuncs * (tuneWarmupSec + tuneDurationSec);
  (void)totalTuneTime;  // Suppress unused warning
  // Silent tuning - no verbose output

  boost::mutex durLock;
  std::vector<int64_t> durations[numAstroFuncs];
  
  boost::mutex hashLock;
  int64_t numHashes[numAstroFuncs];
  for(int x = 0; x < numAstroFuncs; x++) {
    numHashes[x] = 0;
  }

  int fastestCompIdx = 0;
  void (*fastestComp)(workerData &worker, bool isTest, int wIndex) = branchComputeCPU;

  try {
    byte random_buffer[48];
    generateRandomBytesForTune<48>(random_buffer);
    byte res[32];

    boost::thread tune_threads[num_threads];
    for (int x = 0; x < numAstroFuncs; x++)
    {
      astroCompFunc = allAstroFuncs[x].funcPtr;

      // Start each thread with an inline lambda function
      for (int i = 0; i < num_threads; ++i) {
        tune_threads[i] = boost::thread([&]() {
          int tid = i;
          workerData *worker = (workerData *)malloc_huge_pages(sizeof(workerData));
          if (!worker) {
            fprintf(stderr, "Failed to allocate workerData for tuning thread %d\n", tid);
            return;
          }
          initWorker(*worker);
          lookupGen(*worker, nullptr, nullptr);

          auto warmupStart = std::chrono::steady_clock::now();
          for(;;) {
            AstroBWTv3(random_buffer, 48, res, *worker, false);
            if(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - warmupStart).count() > tuneWarmupMs) {
              break;
            }
          }

          int hashes = 0;
          auto start = std::chrono::steady_clock::now();
          for(;;) {
            AstroBWTv3(random_buffer, 48, res, *worker, false);
            hashes++;
            if(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count() > tuneDurationMs) {
              break;
            }
          }
          hashLock.lock();
          numHashes[x] += hashes;
          hashLock.unlock();
        });
      }
      // Wait for all threads to finish
      for (int i = 0; i < num_threads; ++i) {
          tune_threads[i].join();
      }
      // Silent tuning - no per-algo output
      (void)tuneDurationMs;
    }
  } catch (const std::exception& e) {
      std::cerr << "Exception: " << e.what() << "\n";
  }

  astroCompFunc = allAstroFuncs[0].funcPtr;
  int64_t mostHashes = numHashes[0];

  // Verbose output if requested
  if (g_verbose_tune) {
    printf("\n=== Autotune Results ===\n");
    for (int x = 0; x < numAstroFuncs; x++) {
      double khps = (double)numHashes[x] / (tuneDurationMs / 1000.0) / 1000.0;
      printf("  %s: %.2f KH/s (%lld hashes in %llds)\n",
             allAstroFuncs[x].funcName.c_str(), khps,
             (long long)numHashes[x], (long long)(tuneDurationMs/1000));
    }
  }

  for (int x = 0; x < numAstroFuncs; x++)
  {
    if (numHashes[x] > mostHashes)
    {
      astroCompFunc = allAstroFuncs[x].funcPtr;
      mostHashes = numHashes[x];
      fastestCompIdx = x;
    }
  }

  if (g_verbose_tune) {
    printf("Using: %s\n", allAstroFuncs[fastestCompIdx].funcName.c_str());
    printf("========================\n\n");
  }
}

/* ========================================================================== */
/* SA Prefetch Autotune                                                       */
/* ========================================================================== */

#ifdef USE_CUSTOM_SA
/* Prefetch configurations to test during SA autotune */
struct SAConfigEntry {
    int sa_pf;
    int text_pf;
    int bucket_pf;
    const char* name;
};

static const SAConfigEntry sa_configs_to_test[] = {
    {8,  12, 4,  "Conservative"},    /* Small L1 */
    {12, 16, 6,  "Medium"},          /* Medium L1/L2 */
    {16, 24, 8,  "Default"},         /* Large L2 (current default) */
    {20, 28, 10, "Aggressive"},      /* Large L3 */
    {24, 32, 12, "Very Aggressive"}, /* Very large L3 */
};

static const size_t num_sa_configs = sizeof(sa_configs_to_test) / sizeof(sa_configs_to_test[0]);

bool parseSAPrefetch(const std::string& prefetch_str) {
    int sa_pf, text_pf, bucket_pf;
    if (sscanf(prefetch_str.c_str(), "%d,%d,%d", &sa_pf, &text_pf, &bucket_pf) == 3) {
        if (sa_pf > 0 && text_pf > 0 && bucket_pf > 0 &&
            sa_pf <= 64 && text_pf <= 64 && bucket_pf <= 64) {
            sa_config_set(sa_pf, text_pf, bucket_pf);
            printf("SA prefetch set to: %d,%d,%d\n", sa_pf, text_pf, bucket_pf);
            fflush(stdout);
            return true;
        }
    }
    printf("Invalid SA prefetch format: %s (expected: sa,text,bucket e.g., '16,24,8')\n",
           prefetch_str.c_str());
    fflush(stdout);
    return false;
}

void saTune(int num_threads, int tuneWarmupSec, int tuneDurationSec) {
    int64_t tuneWarmupMs = tuneWarmupSec * 1000;
    int64_t tuneDurationMs = tuneDurationSec * 1000;

    if (g_verbose_tune) {
        printf("\n=== SA Prefetch Autotune ===\n");
        printf("Testing %zu configurations with %d threads...\n\n",
               num_sa_configs, num_threads);
        fflush(stdout);
    }

    boost::mutex hashLock;
    std::vector<int64_t> numHashes(num_sa_configs, 0);

    int fastestConfigIdx = 2; /* Default config index */

    try {
        byte random_buffer[48];
        generateRandomBytesForTune<48>(random_buffer);
        byte res[32];

        boost::thread* tune_threads = new boost::thread[num_threads];

        for (size_t cfg_idx = 0; cfg_idx < num_sa_configs; cfg_idx++) {
            /* Set the SA config for this test */
            const SAConfigEntry& cfg = sa_configs_to_test[cfg_idx];
            sa_config_set(cfg.sa_pf, cfg.text_pf, cfg.bucket_pf);

            /* Start each thread with an inline lambda function */
            for (int i = 0; i < num_threads; ++i) {
                tune_threads[i] = boost::thread([&, cfg_idx, i]() {
                    workerData *worker = (workerData *)malloc_huge_pages(sizeof(workerData));
                    if (!worker) {
                        fprintf(stderr, "Failed to allocate workerData for SA tuning thread %d\n", i);
                        return;
                    }
                    initWorker(*worker);
                    lookupGen(*worker, nullptr, nullptr);

                    /* Warmup phase */
                    auto warmupStart = std::chrono::steady_clock::now();
                    for (;;) {
                        AstroBWTv3(random_buffer, 48, res, *worker, false);
                        if (std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - warmupStart).count() > tuneWarmupMs) {
                            break;
                        }
                    }

                    /* Benchmark phase */
                    int hashes = 0;
                    auto start = std::chrono::steady_clock::now();
                    for (;;) {
                        AstroBWTv3(random_buffer, 48, res, *worker, false);
                        hashes++;
                        if (std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - start).count() > tuneDurationMs) {
                            break;
                        }
                    }

                    hashLock.lock();
                    numHashes[cfg_idx] += hashes;
                    hashLock.unlock();

                    free_huge_pages(worker);
                });
            }

            /* Wait for all threads to finish this configuration */
            for (int i = 0; i < num_threads; ++i) {
                tune_threads[i].join();
            }
        }

        delete[] tune_threads;

    } catch (const std::exception& e) {
        std::cerr << "SA Tune Exception: " << e.what() << "\n";
    }

    /* Find the fastest configuration */
    int64_t mostHashes = numHashes[0];
    fastestConfigIdx = 0;

    if (g_verbose_tune) {
        printf("SA Tune Results:\n");
    }

    for (size_t cfg_idx = 0; cfg_idx < num_sa_configs; cfg_idx++) {
        const SAConfigEntry& cfg = sa_configs_to_test[cfg_idx];
        double khps = (double)numHashes[cfg_idx] / (tuneDurationMs / 1000.0) / 1000.0;

        if (g_verbose_tune) {
            const char* marker = (numHashes[cfg_idx] > mostHashes ||
                                 (cfg_idx == 0 && numHashes[cfg_idx] == mostHashes)) ?
                                 " <-- Best" : "";
            printf("  %s (%d,%d,%d): %.2f KH/s%s\n",
                   cfg.name, cfg.sa_pf, cfg.text_pf, cfg.bucket_pf, khps, marker);
        }

        if (numHashes[cfg_idx] > mostHashes) {
            mostHashes = numHashes[cfg_idx];
            fastestConfigIdx = (int)cfg_idx;
        }
    }

    /* Apply the fastest configuration */
    const SAConfigEntry& best = sa_configs_to_test[fastestConfigIdx];
    sa_config_set(best.sa_pf, best.text_pf, best.bucket_pf);

    if (g_verbose_tune) {
        printf("\nUsing SA config: %s (%d,%d,%d)\n",
               best.name, best.sa_pf, best.text_pf, best.bucket_pf);
        printf("============================\n\n");
        fflush(stdout);
    }
}
#endif /* USE_CUSTOM_SA */

void hashSHA256(SHA256_CTX &sha256, const byte *input, byte *digest, unsigned long inputSize)
{
  SHA256_Init(&sha256);
  SHA256_Update(&sha256, input, inputSize);
  SHA256_Final(digest, &sha256);
}

std::vector<uint8_t> padSHA256Input(const uint8_t* input, size_t length) {
    // Calculate the length of the padded message
    size_t paddedLength = length + 1; // Original length plus the 0x80 byte
    size_t mod = paddedLength % 64;
    if (mod > 56) {
        paddedLength += 64 + 56 - mod; // Pad so there's room for the length
    } else {
        paddedLength += 56 - mod; // Pad so there's room for the length
    }
    paddedLength += 8; // Add 8 bytes for the original length

    // Create the padded message
    std::vector<uint8_t> padded(paddedLength, 0);
    memcpy(padded.data(), input, length);

    // Append the '1' bit
    padded[length] = 0x80;

    // Append the original length in bits as a 64-bit big-endian integer
    uint64_t bitLength = static_cast<uint64_t>(length) * 8; // Convert length to bits
    for (size_t i = 0; i < 8; ++i) {
        padded[paddedLength - 1 - i] = static_cast<uint8_t>((bitLength >> (8 * i)) & 0xff);
    }

    return padded;
}

void checkFalseRep(workerData &worker) {
  byte f = worker.prev_chunk[worker.pos1];
  bool falsePos = false;
  for (int i = worker.pos1; i < worker.pos2; i++) {
    if (worker.prev_chunk[i] != f) {
      falsePos = true;
      break;
    }
  }

  if (falsePos) {
    printf("false positive\n");
    for (int i = worker.pos1; i < worker.pos2; i++) {
      printf("%02x ", worker.prev_chunk[i]);
    }
    printf("\n");
  }
}

// Function to compare two suffixes based on lexicographical order
bool cmp(const uint8_t* data, const std::pair<uint32_t, uint32_t>& a, const std::pair<uint32_t, uint32_t>& b) {
    const uint8_t* p1 = data + a.second;
    const uint8_t* p2 = data + b.second;

    while (*p1 != 0 && *p1 == *p2) {
        ++p1;
        ++p2;
    }

    return *p1 < *p2;
}

void buildSuffixArray(const uint8_t* data, int n, int* suffixArray, int* buckets, int* sorted) {
    if (n <= 0 || !data || !suffixArray || !buckets || !sorted) {
        // Handle invalid input
        return;
    }

    // Step 1: Counting sort on first character
    for (int i = 0; i < 256; ++i) {
        buckets[i] = 0;  // Initialize buckets
    }

    for (int i = 0; i < n; ++i) {
        buckets[data[i]]++;
    }

    for (int i = 1; i < 256; ++i) {
        buckets[i] += buckets[i - 1];
    }

    for (int i = n - 1; i >= 0; --i) {
        sorted[--buckets[data[i]]] = i;
    }

    // Step 2: Sort suffixes recursively
    int* indices = new int[n];
    indices[sorted[0]] = 0;
    int rank = 0;
    for (int i = 1; i < n; ++i) {
        if (data[sorted[i]] != data[sorted[i - 1]]) {
            rank++;
        }
        indices[sorted[i]] = rank;
    }

    int* temp = new int[n];
    for (int k = 1; (1 << k) < n; k <<= 1) {
        for (int i = 0; i < n; ++i) {
            temp[i] = (indices[i] << k) | (i + (1 << k) < n ? indices[i + (1 << k)] : 0);
        }

        for (int i = 0; i < n; ++i) {
            buckets[temp[i]] = 0;  // Reset buckets
        }

        for (int i = 0; i < n; ++i) {
            buckets[temp[i]]++;
        }

        for (int i = 1; i < n; ++i) {
            buckets[i] += buckets[i - 1];
        }

        for (int i = n - 1; i >= 0; --i) {
            suffixArray[--buckets[temp[i]]] = sorted[i];
        }

        for (int i = 0; i < n; ++i) {
            indices[suffixArray[i]] = temp[suffixArray[i]] == temp[suffixArray[i - 1]] && i > 0 ? indices[suffixArray[i - 1]] : i;
        }
    }

    delete[] indices;
    delete[] temp;
}

#if defined(__AVX2__)

void computeByteFrequencyAVX2(const unsigned char* data, size_t dataSize, int frequencyTable[256]) {
    __m256i chunk;
    const size_t simdWidth = 32; // AVX2 SIMD register width in bytes

    // Zero-initialize a local frequency table to avoid read-modify-write AVX2 operations
    int localFrequencyTable[256] = {0};

    // Process chunks of 32 bytes
    for (size_t i = 0; i < dataSize; i += simdWidth) {
        if (i + simdWidth <= dataSize) { // Ensure we don't read past the end
            chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));

            // Update frequency table in a non-vectorized manner due to AVX2 limitations
            unsigned char temp[simdWidth];
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(temp), chunk);

            for (size_t j = 0; j < simdWidth; ++j) {
                ++localFrequencyTable[temp[j]];
            }
        } else {
            // Handle remainder bytes that don't fit into a full AVX2 register
            for (size_t j = i; j < dataSize; ++j) {
                ++localFrequencyTable[data[j]];
            }
            break; // Exit the loop after processing the remainder
        }
    }

    // Accumulate the results into the provided frequency table
    for (int i = 0; i < 256; ++i) {
        frequencyTable[i] += localFrequencyTable[i];
    }
}

#endif

// alc_digest_info_t dinfo = {
//   .dt_type = ALC_DIGEST_TYPE_SHA2,
//   .dt_len = ALC_DIGEST_LEN_256,
//   .dt_mode = {.dm_sha2 = ALC_SHA2_256,},
// };

// uint64_t aoclSHA256(const Uint8* src,
//           Uint64       src_size,
//           Uint8*       output)
// {
//     alc_error_t err;
//     static alc_digest_handle_t s_dg_handle;

//     Uint64 size         = alcp_digest_context_size(&dinfo);
//     s_dg_handle.context = malloc(size);

//     err = alcp_digest_request(&dinfo, &s_dg_handle);

//     // divide the input size into multiple chunks
//     const Uint64 last_buf_size = src_size % 64;
//     const Uint8* p             = src;

//     int c = src_size/64;
//     while (c-- > 0) {
//         err = alcp_digest_update(&s_dg_handle, p, 64);
//         p += 64;
//         c--;
//     }

//     if (last_buf_size == 0) {
//         p = NULL;
//     }

//     alcp_digest_finalize(&s_dg_handle, p, last_buf_size);

//     err = alcp_digest_copy(&s_dg_handle, output, 32);

//     alcp_digest_finish(&s_dg_handle);
//     free(s_dg_handle.context);
//     return err;
// }

DIRTYBIRD_TARGETS
void AstroBWTv3_batch(byte *input, int inputLen, byte *outputhash, workerData &worker, bool unused)
{
  // auto recoverFunc = [&outputhash](void *r)
  // {
  //   std::random_device rd;
  //   std::mt19937 gen(rd());
  //   std::uniform_int_distribution<uint8_t> dist(0, 255);
  //   std::array<uint8_t, 16> buf;
  //   std::generate(buf.begin(), buf.end(), [&dist, &gen]()
  //                 { return dist(gen); });
  //   std::memcpy(outputhash, buf.data(), buf.size());
  //   std::cout << "exception occured, returning random hash" << std::endl;
  // };
  // std::function<void(void *)> recover = recoverFunc;

  try
  {
    for (int i = 0; i < DERO_BATCH; i++) {
      // auto start = std::chrono::steady_clock::now();
      // Zero 64 bytes after first 256 (only need one operation, not both)
      memset(worker.sData + ASTRO_SCRATCH_SIZE*i + 256, 0, 64);

      __builtin_prefetch(&worker.sData[ASTRO_SCRATCH_SIZE*i + 256], 1, 3);
      __builtin_prefetch(&worker.sData[ASTRO_SCRATCH_SIZE*i + 256+64], 1, 3);
      __builtin_prefetch(&worker.sData[ASTRO_SCRATCH_SIZE*i + 256+128], 1, 3);
      __builtin_prefetch(&worker.sData[ASTRO_SCRATCH_SIZE*i + 256+192], 1, 3);
      
      hashSHA256(worker.sha256, &input[i*inputLen], &worker.sData[ASTRO_SCRATCH_SIZE*i + 320], inputLen);
      __builtin_prefetch(worker.sData + ASTRO_SCRATCH_SIZE*i, 1, 3);
      __builtin_prefetch(&worker.sData[ASTRO_SCRATCH_SIZE*i + 64], 1, 3);
      __builtin_prefetch(&worker.sData[ASTRO_SCRATCH_SIZE*i + 128], 1, 3);
      __builtin_prefetch(&worker.sData[ASTRO_SCRATCH_SIZE*i + 192], 1, 3);

#if USE_SIMD_SALSA20
      // SIMD Salsa20 - AVX2 optimized (~2-5% faster)
      // Key at offset 320 (SHA256 output), IV at offset 256 (zeros)
      salsa20_simd_process(&worker.sData[ASTRO_SCRATCH_SIZE*i + 320],
                           &worker.sData[ASTRO_SCRATCH_SIZE*i + 256],
                           worker.salsaInput,
                           &worker.sData[ASTRO_SCRATCH_SIZE*i], 256);
#else
      worker.salsa20.setKey(&worker.sData[ASTRO_SCRATCH_SIZE*i + 320]);
      worker.salsa20.setIv(&worker.sData[ASTRO_SCRATCH_SIZE*i + 256]);
      worker.salsa20.processBytes(worker.salsaInput, &worker.sData[ASTRO_SCRATCH_SIZE*i], 256);
#endif

      __builtin_prefetch(&worker.key[i] + 8, 1, 3);
      __builtin_prefetch(&worker.key[i] + 8+64, 1, 3);
      __builtin_prefetch(&worker.key[i] + 8+128, 1, 3);
      __builtin_prefetch(&worker.key[i] + 8+192, 1, 3);

#if USE_CRYPTOGAMS_RC4_DUAL
      // Dual-state: Initialize both CRYPTOGAMS (fast) and OpenSSL (SPSA compat)
      worker.cryptogams_rc4[i].set_key(&worker.sData[ASTRO_SCRATCH_SIZE*i], 256);
      RC4_set_key(&worker.key[i], 256, &worker.sData[ASTRO_SCRATCH_SIZE*i]);
      worker.cryptogams_rc4[i].apply_keystream_256(&worker.sData[ASTRO_SCRATCH_SIZE*i]);
#elif USE_FAST_RC4
      // Use optimized RC4 implementation with SPSA sync
      rc4_avx512::fast_rc4_set_key_dual(worker.fast_rc4_key[i], &worker.key[i], 256, &worker.sData[ASTRO_SCRATCH_SIZE*i]);
      rc4_avx512::fast_rc4_dual(worker.fast_rc4_key[i], &worker.key[i], 256, &worker.sData[ASTRO_SCRATCH_SIZE*i], &worker.sData[ASTRO_SCRATCH_SIZE*i]);
#else
      RC4_set_key(&worker.key[i], 256,  &worker.sData[ASTRO_SCRATCH_SIZE*i]);
      RC4(&worker.key[i], 256, &worker.sData[ASTRO_SCRATCH_SIZE*i], &worker.sData[ASTRO_SCRATCH_SIZE*i]);
#endif

      // auto end = std::chrono::steady_clock::now();
      // auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start);
      // if (i == 0) printf("Prep section #%d section took %dns\n", i+1, time.count());
      // if (i == DERO_BATCH / 2) printf("Prep section #%d section took %dns\n", i+1, time.count());
      // if (i == DERO_BATCH - 1) printf("Prep section #%d section took %dns\n", i+1, time.count());
    }
    // syncTag.fetch_add(1);
    // syncTag.notify_all();

    // for(;;) {
    //   if (syncTag.load() == syncTarget) break;
    //   std::this_thread::yield();
    // }

    // syncTag.store(0); syncTag.notify_all();
    // syncTag.wait(syncTarget);

    // printf(hexStr(worker.chunk, 256).c_str());
    // printf("\n\n");

    
    // auto start = std::chrono::steady_clock::now();
    // auto end = std::chrono::steady_clock::now();

    /*
    if (useLookupMine) {
      // start = std::chrono::steady_clock::now();
      lookupCompute(worker, false);
      // end = std::chrono::steady_clock::now();
    }
    else {
      // // start = std::chrono::steady_clock::now();
      #if defined(__AVX2__)
        wolfCompute(worker, false);
      #elif defined(__aarch64__)
        branchComputeCPU_aarch64(worker, false);
      #else
        branchComputeCPU(worker, false);
      #endif
      // // end = std::chrono::steady_clock::now();
      //wolfCompute(worker, false);
    }
    */

    for (int i = 0; i < DERO_BATCH; i++) {
      worker.lhash = hash_64_fnv1a_256_optimized(&worker.sData[ASTRO_SCRATCH_SIZE*i]);
      worker.prev_lhash = worker.lhash;

      worker.tries[i] = 0;
      worker.isSame = false;

      // auto start = std::chrono::steady_clock::now();
      astroCompFunc(worker, false, i);
      // auto end = std::chrono::steady_clock::now();
      // auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start);
      // if (i == 0) printf("Branched section #%d section took %dns\n", i+1, time.count());
      // if (i == DERO_BATCH / 2) printf("Branched section #%d section took %dns\n", i+1, time.count());
      // if (i == DERO_BATCH - 1) printf("Branched section #%d section took %dns\n", i+1, time.count());
    }
    // syncTag.fetch_add(1);
    // syncTag.notify_all();

    // for(;;) {
    //   if (syncTag == syncTarget) break;
    //   std::this_thread::yield();
    // }

    // syncTag.store(0); syncTag.notify_all();
    // syncTag.wait(syncTarget);

    // auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start);
    // if (!useLookupMine) printf("AVX2: ");
    // else printf("Lookup: ");
    // printf("branched section took %dns\n", time.count());
    // if (debugOpOrder) {
    //   if (worker.opsB.size() > 0) {
    //     printf("Lookup Table:\n-----------\n");
    //     for (int i = 0; i < worker.opsB.size(); i++) {
    //       printf("%d, ", worker.opsB[i]);
    //     }
    //   } else {
    //     printf("Scalar:\n-----------\n");
    //     for (int i = 0; i < worker.opsA.size(); i++) {
    //       printf("%d, ", worker.opsA[i]);
    //     }
    //   }

    //   printf("\n");
    //   worker.opsA.clear();
    //   worker.opsB.clear();
    // }
    // // worker.data_len = 70000;
    // saveBufferToFile("worker_sData_snapshot.bin", worker.sData, worker.data_len);
    // printf("data length: %d\n", worker.data_len);
    // auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < DERO_BATCH; i++) {
      worker.data_len = static_cast<uint32_t>(
        (worker.tries[i] - 4) * 256 + 
        (((static_cast<uint64_t>(worker.sData[i*ASTRO_SCRATCH_SIZE + (worker.tries[i]-1)*256 + 253]) << 8) | 
        static_cast<uint64_t>(worker.sData[i*ASTRO_SCRATCH_SIZE + (worker.tries[i]-1)*256 + 254])) & 0x3ff)
      );
      // auto start = std::chrono::steady_clock::now();
      // divsufsort(&worker.sData[i * ASTRO_SCRATCH_SIZE], worker.sa, worker.data_len, worker.bA, worker.bB);
      #if defined(USE_ASTRO_SPSA)
        if (g_use_spsa) SPSA_LOG_ENABLED();
        prefetchSpsaBucketArrays(worker);
        if(shouldUseSpsaForDataLen(worker.data_len) && SPSA(&worker.sData[i * ASTRO_SCRATCH_SIZE], worker.data_len, worker)) {
          SPSA_HIT();
          memcpy(outputhash, worker.padding, 32);
        } else {
          SPSA_MISS();
          #if USE_DEROBWT && !defined(USE_DLUNA_RADIX_SA) && !defined(USE_RADIX_SA) && !defined(USE_BUCKET_SA)
          derobwt::compute_sa_threadsafe(&worker.sData[i * ASTRO_SCRATCH_SIZE], worker.data_len, worker.sa);
      #else
          int32_t *bucketA = nullptr;
          int32_t *bucketB = nullptr;
          getSABucketScratch(worker, bucketA, bucketB);
          SA_FUNCTION(&worker.sData[i * ASTRO_SCRATCH_SIZE], worker.sa, worker.data_len, bucketA, bucketB);
          #endif
          byte *B = reinterpret_cast<byte *>(worker.sa);
          hashSHA256(worker.sha256, B, (outputhash + 32*i), worker.data_len*4);
        }
        SPSA_LOG_STATS();
      #else
        #if USE_DEROBWT && !defined(USE_DLUNA_RADIX_SA) && !defined(USE_RADIX_SA) && !defined(USE_BUCKET_SA)
          derobwt::compute_sa_threadsafe(&worker.sData[i * ASTRO_SCRATCH_SIZE], worker.data_len, worker.sa);
        #else
          int32_t *bucketA = nullptr;
          int32_t *bucketB = nullptr;
          getSABucketScratch(worker, bucketA, bucketB);
          SA_FUNCTION(&worker.sData[i * ASTRO_SCRATCH_SIZE], worker.sa, worker.data_len, bucketA, bucketB);
        #endif
        #ifdef ENABLE_SA_INSTRUMENTATION
        {
          static thread_local int sa_iteration = 0;
          SA_PRINT_METRICS(++sa_iteration);
        }
        #endif
        byte *B = reinterpret_cast<byte *>(worker.sa);
        hashSHA256(worker.sha256, B, (outputhash + 32*i), worker.data_len*4);
      #endif
      // auto end = std::chrono::steady_clock::now();
      // auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start);
      // if (i == 0) printf("divsufsort section #%d section took %dns\n", i+1, time.count());
      // if (i == DERO_BATCH / 2) printf("divsufsort section #%d section took %dns\n", i+1, time.count());
      // if (i == DERO_BATCH - 1) printf("divsufsort section #%d section took %dns\n", i+1, time.count());
    //  if (littleEndian())
    //  {
        //byte *B = reinterpret_cast<byte *>(worker.sa);
        //hashSHA256(worker.sha256, B, (outputhash + 32*i), worker.data_len*4);
    //  } else {
    //    byte s[MAX_LENGTH * 4];
    //    for (int i = 0; i < worker.data_len; i++)
    //    {
    //      s[i << 1] = htonl(worker.sa[i]);
    //    }
    //    hashSHA256(worker.sha256, s, worker.sHash, worker.data_len*4);
    //  }
    //  memcpy(outputhash + 32*i, worker.sHash, 32);
    //  // memset(outputhash + 32*i, 0xFF, 32);
    }
    // syncTag.fetch_add(1);
    // syncTag.notify_all();

    // for(;;) {
    //   if (syncTag == syncTarget) break;
    //   std::this_thread::yield();
    // }

    // if (tid == 1) {syncTag.store(0); syncTag.notify_all();}
    // syncTag.wait(syncTarget);
    // printf("\n\n\n");
    // auto end = std::chrono::steady_clock::now();
    // auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start);
    // printf("SA section took %dns\n", time.count());

    // auto start = std::chrono::steady_clock::now();
    // for (;;) {
    //   auto end = std::chrono::steady_clock::now();
    //   if (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() >= 650000) break;
    //   std::this_thread::yield();
    // }
    // std::this_thread::sleep_for(std::chrono::nanoseconds(1107000));

    // computeByteFrequencyAVX2(worker.sData, worker.data_len, worker.freq);
    // libsais_ctx(worker.ctx, worker.sData, worker.sa, worker.data_len, MAX_LENGTH-worker.data_len, NULL);
  }
  catch (const std::exception &ex)
  {
    // recover(outputhash);
    std::cerr << ex.what() << std::endl;
  }
}

DIRTYBIRD_TARGETS
void AstroBWTv3(byte *input, int inputLen, byte *outputhash, workerData &worker, bool unused, int tid)
{
  // auto recoverFunc = [&outputhash](void *r)
  // {
  //   std::random_device rd;
  //   std::mt19937 gen(rd());
  //   std::uniform_int_distribution<uint8_t> dist(0, 255);
  //   std::array<uint8_t, 16> buf;
  //   std::generate(buf.begin(), buf.end(), [&dist, &gen]()
  //                 { return dist(gen); });
  //   std::memcpy(outputhash, buf.data(), buf.size());
  //   std::cout << "exception occured, returning random hash" << std::endl;
  // };
  // std::function<void(void *)> recover = recoverFunc;

  try
  {
    const bool phaseTelemetry = isPhaseTelemetryEnabled();
    const auto phase_now_ns = []() -> uint64_t {
      return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch()).count());
    };
    uint64_t phase_t0 = 0;
    uint64_t phase_t1 = 0;
    uint64_t phase_t2 = 0;
    uint64_t phase_spsa_ns = 0;
    uint64_t phase_spsa_prefetch_ns = 0;
    uint64_t phase_spsa_hit_copy_ns = 0;
    uint64_t phase_spsa_miss_hash_ns = 0;
    uint64_t phase_spsa_core_ns = 0;
    uint64_t phase_spsa_core_calls = 0;
    std::array<uint64_t, 4> phase_spsa_core_bin_calls{0, 0, 0, 0};
    std::array<uint64_t, 4> phase_spsa_core_bin_ns{0, 0, 0, 0};
    uint64_t phase_sa_ns = 0;
    uint64_t phase_hash_ns = 0;
    if (phaseTelemetry) {
      phase_t0 = phase_now_ns();
    }

#if SPSA_PROFILE && defined(USE_ASTRO_SPSA)
    auto t0 = std::chrono::steady_clock::now();
#endif

    alignas(32) uint8_t scratch[384] = {0};
    memset(worker.sData, 0, ASTRO_SCRATCH_SIZE);

    hashSHA256(worker.sha256, input, &scratch[320], inputLen);
#if USE_SIMD_SALSA20
    // SIMD Salsa20 - AVX2 optimized (~2-5% faster)
    salsa20_simd_process(&scratch[320], &scratch[256], worker.salsaInput, scratch, 256);
#else
    worker.salsa20.setKey(&scratch[320]);
    worker.salsa20.setIv(&scratch[256]);
    worker.salsa20.processBytes(worker.salsaInput, scratch, 256);
#endif

#if USE_CRYPTOGAMS_RC4_DUAL
    // Dual-state: CRYPTOGAMS for fast encryption, OpenSSL for SPSA S-box access
    worker.cryptogams_rc4[0].set_key(scratch, 256);
    RC4_set_key(&worker.key[0], 256, scratch);
    worker.cryptogams_rc4[0].apply_keystream_256(scratch);
#elif USE_FAST_RC4
    // Use optimized RC4 implementation with SPSA sync
    rc4_avx512::fast_rc4_set_key_dual(worker.fast_rc4_key[0], &worker.key[0], 256, scratch);
    rc4_avx512::fast_rc4_dual(worker.fast_rc4_key[0], &worker.key[0], 256, scratch, scratch);
#else
    RC4_set_key(&worker.key[0], 256,  scratch);
    RC4(&worker.key[0], 256, scratch,  scratch);
#endif


    worker.lhash = hash_64_fnv1a_256(scratch);
    worker.prev_lhash = worker.lhash;

    worker.tries[0] = 0;
    worker.isSame = false;

    memcpy(worker.sData, scratch, 256);

    if (phaseTelemetry) {
      phase_t1 = phase_now_ns();
    }

#if SPSA_PROFILE && defined(USE_ASTRO_SPSA)
    auto t1 = std::chrono::steady_clock::now();
#endif

    // printf(hexStr(worker.chunk, 256).c_str());
    // printf("\n\n");

    
    // auto start = std::chrono::steady_clock::now();
    // auto end = std::chrono::steady_clock::now();

    /*
    if (useLookupMine) {
      // start = std::chrono::steady_clock::now();
      lookupCompute(worker, false);
      // end = std::chrono::steady_clock::now();
    }
    else {
      // // start = std::chrono::steady_clock::now();
      #if defined(__AVX2__)
        wolfCompute(worker, false);
      #elif defined(__aarch64__)
        branchComputeCPU_aarch64(worker, false);
      #else
        branchComputeCPU(worker, false);
      #endif
      // // end = std::chrono::steady_clock::now();
      //wolfCompute(worker, false);
    }
    */

    astroCompFunc(worker, false, 0);

    if (phaseTelemetry) {
      phase_t2 = phase_now_ns();
    }

#if SPSA_PROFILE && defined(USE_ASTRO_SPSA)
    auto t2 = std::chrono::steady_clock::now();
#endif

    // auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start);
    // if (!useLookupMine) printf("AVX2: ");
    // else printf("Lookup: ");
    // printf("branched section took %dns\n", time.count());
    // if (debugOpOrder) {
    //   if (worker.opsB.size() > 0) {
    //     printf("Lookup Table:\n-----------\n");
    //     for (int i = 0; i < worker.opsB.size(); i++) {
    //       printf("%d, ", worker.opsB[i]);
    //     }
    //   } else {
    //     printf("Scalar:\n-----------\n");
    //     for (int i = 0; i < worker.opsA.size(); i++) {
    //       printf("%d, ", worker.opsA[i]);
    //     }
    //   }

    //   printf("\n");
    //   worker.opsA.clear();
    //   worker.opsB.clear();
    // }
    // // worker.data_len = 70000;
    // saveBufferToFile("worker_sData_snapshot.bin", worker.sData, worker.data_len);
    // printf("data length: %d\n", worker.data_len);
    // auto start = std::chrono::steady_clock::now();
    // divsufsort(worker.sData, worker.sa, worker.data_len, worker.bA, worker.bB);
    #if defined(USE_ASTRO_SPSA)
      if (shouldUseSpsaForDataLen(worker.data_len)) {
        SPSA_LOG_ENABLED();
        if (phaseTelemetry) {
          const uint64_t prefetch_start_ns = phase_now_ns();
          prefetchSpsaBucketArrays(worker);
          phase_spsa_prefetch_ns += (phase_now_ns() - prefetch_start_ns);
        } else {
          prefetchSpsaBucketArrays(worker);
        }
        uint64_t phase_spsa_start_ns = 0;
        if (phaseTelemetry) {
          phase_spsa_start_ns = phase_now_ns();
        }
        // TNN-style direct SPSA path: SPSA prepares SA and may also compute final SHA.
        bool alreadySha = false;
        if (g_use_local_spsa) {
          alreadySha = spsa::SPSA_Integrated(worker.sData, worker.data_len, worker, outputhash);
        } else {
          alreadySha = SPSA(worker.sData, worker.data_len, worker);
          if (alreadySha) {
            SPSA_HIT();
            uint64_t phase_hit_copy_start_ns = 0;
            if (phaseTelemetry) {
              phase_hit_copy_start_ns = phase_now_ns();
            }
            memcpy(outputhash, worker.padding, 32);
            if (phaseTelemetry) {
              phase_spsa_hit_copy_ns += (phase_now_ns() - phase_hit_copy_start_ns);
            }
          }
        }

        if (phaseTelemetry) {
          phase_spsa_ns += (phase_now_ns() - phase_spsa_start_ns);
        }
        
        if (alreadySha) {
          if (!g_use_local_spsa) {
            // Already handled above for library path
          } else {
            SPSA_HIT();
          }
        } else {
          SPSA_MISS();
          if (g_verbose_tune) {
            printf("AstroBWTv3: SPSA fallback!\n");
          }
          if (g_use_local_spsa) {
            int32_t *bucketA = nullptr, *bucketB = nullptr;
            getSABucketScratch(worker, bucketA, bucketB);
            SA_FUNCTION(worker.sData, worker.sa, worker.data_len, bucketA, bucketB);
          }
          byte *B = reinterpret_cast<byte *>(worker.sa);
          uint64_t phase_hash_start_ns = 0;
          if (phaseTelemetry) {
            phase_hash_start_ns = phase_now_ns();
          }
          hashSHA256(worker.sha256, B, outputhash, worker.data_len * 4);
          if (phaseTelemetry) {
            const uint64_t miss_hash_ns = phase_now_ns() - phase_hash_start_ns;
            phase_hash_ns += miss_hash_ns;
            phase_spsa_miss_hash_ns += miss_hash_ns;
          }
        }
        #if SPSA_PROFILE
        {
          auto t3 = std::chrono::steady_clock::now();
          profile_setup_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
          profile_wolf_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
          profile_spsa_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
          log_profile_stats();
        }
        #endif
      } else {
    #endif
      #ifdef USE_RADIX_SA
        // Zero padding after data for deterministic radix sort reads near end
        memset(worker.sData + worker.data_len, 0, 16);
      #endif
      #if USE_DEROBWT && !defined(USE_DLUNA_RADIX_SA) && !defined(USE_RADIX_SA) && !defined(USE_BUCKET_SA)
        // DeroBWT: Custom tiered fingerprint sorting for AstroBWTv3
        derobwt::compute_sa_threadsafe(worker.sData, worker.data_len, worker.sa);
      #else
        int32_t *bucketA = nullptr;
        int32_t *bucketB = nullptr;
        getSABucketScratch(worker, bucketA, bucketB);
        uint64_t phase_sa_start_ns = 0;
        if (phaseTelemetry) {
          phase_sa_start_ns = phase_now_ns();
        }
        if (g_verbose_tune) {
          printf("  [Std] Prep phase done, data_len=%u\n", worker.data_len);
          fflush(stdout);
        }
        SA_FUNCTION(worker.sData, worker.sa, worker.data_len, bucketA, bucketB);
        if (g_verbose_tune) {
          printf("  [Std] SA phase done\n");
          fflush(stdout);
        }
        if (phaseTelemetry) {
          phase_sa_ns += (phase_now_ns() - phase_sa_start_ns);
        }
      #endif
      #ifdef ENABLE_SA_INSTRUMENTATION
      {
        static thread_local int sa_iteration = 0;
        SA_PRINT_METRICS(++sa_iteration);
      }
      #endif
      byte *B = reinterpret_cast<byte *>(worker.sa);
      uint64_t phase_hash_start_ns = 0;
      if (phaseTelemetry) {
        phase_hash_start_ns = phase_now_ns();
      }
      hashSHA256(worker.sha256, B, outputhash, worker.data_len*4);
      if (g_verbose_tune) {
        printf("  [Std] Final hash done\n");
        fflush(stdout);
      }
      if (phaseTelemetry) {
        phase_hash_ns += (phase_now_ns() - phase_hash_start_ns);
      }
    #if defined(USE_ASTRO_SPSA)
      }
    #endif

    if (phaseTelemetry && phase_spsa_ns > 0) {
      phase_spsa_core_calls = 1;
      uint64_t core_ns = phase_spsa_ns;
      core_ns = (core_ns > phase_spsa_prefetch_ns) ? (core_ns - phase_spsa_prefetch_ns) : 0;
      core_ns = (core_ns > phase_spsa_hit_copy_ns) ? (core_ns - phase_spsa_hit_copy_ns) : 0;
      core_ns = (core_ns > phase_spsa_miss_hash_ns) ? (core_ns - phase_spsa_miss_hash_ns) : 0;
      phase_spsa_core_ns = core_ns;

      const int core_bin = classifySpsaCoreLenBin(worker.data_len);
      phase_spsa_core_bin_calls[core_bin] = 1;
      phase_spsa_core_bin_ns[core_bin] = core_ns;
    }

    if (phaseTelemetry) {
      const uint64_t phase_end = phase_now_ns();
      g_phase_hashes_total.fetch_add(1, std::memory_order_relaxed);
      g_phase_data_len_sum_total.fetch_add(static_cast<uint64_t>(worker.data_len), std::memory_order_relaxed);
      g_phase_prep_ns_total.fetch_add(phase_t1 - phase_t0, std::memory_order_relaxed);
      g_phase_wolf_ns_total.fetch_add(phase_t2 - phase_t1, std::memory_order_relaxed);
      g_phase_spsa_ns_total.fetch_add(phase_spsa_ns, std::memory_order_relaxed);
      g_phase_spsa_prefetch_ns_total.fetch_add(phase_spsa_prefetch_ns, std::memory_order_relaxed);
      g_phase_spsa_hit_copy_ns_total.fetch_add(phase_spsa_hit_copy_ns, std::memory_order_relaxed);
      g_phase_spsa_miss_hash_ns_total.fetch_add(phase_spsa_miss_hash_ns, std::memory_order_relaxed);
      g_phase_spsa_core_ns_total.fetch_add(phase_spsa_core_ns, std::memory_order_relaxed);
      g_phase_spsa_core_calls_total.fetch_add(phase_spsa_core_calls, std::memory_order_relaxed);
      for (size_t i = 0; i < phase_spsa_core_bin_calls.size(); ++i) {
        g_phase_spsa_core_bin_calls_total[i].fetch_add(phase_spsa_core_bin_calls[i], std::memory_order_relaxed);
        g_phase_spsa_core_bin_ns_total[i].fetch_add(phase_spsa_core_bin_ns[i], std::memory_order_relaxed);
      }
      g_phase_sa_ns_total.fetch_add(phase_sa_ns, std::memory_order_relaxed);
      g_phase_hash_ns_total.fetch_add(phase_hash_ns, std::memory_order_relaxed);
      g_phase_total_ns_total.fetch_add(phase_end - phase_t0, std::memory_order_relaxed);
    }

    // auto end = std::chrono::steady_clock::now();
    // auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start);
    // printf("SA section took %dns\n", time.count());

    // auto start = std::chrono::steady_clock::now();
    // for (;;) {
    //   auto end = std::chrono::steady_clock::now();
    //   if (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() >= 650000) break;
    //   std::this_thread::yield();
    // }
    // std::this_thread::sleep_for(std::chrono::nanoseconds(1107000));

    // computeByteFrequencyAVX2(worker.sData, worker.data_len, worker.freq);
    // libsais_ctx(worker.ctx, worker.sData, worker.sa, worker.data_len, MAX_LENGTH-worker.data_len, NULL);

    // if (littleEndian())
    // {
/*
byte *B = reinterpret_cast<byte *>(worker.sa);
	int total_length = worker.data_len*4;
        printf("hashLen: %d\n", total_length);
        printf("added byte(astro): %d(%d) ", worker.data_len, total_length);
        for(int x = total_length-20; x < total_length; x++) {
          printf("%02x ", B[x]);
          //printf("%02x ", (byte*)worker.sa[(x/4)]);
        }
        printf("\n");
        */
      //byte *B = reinterpret_cast<byte *>(worker.sa);
      //hashSHA256(worker.sha256, B, outputhash, worker.data_len*4);

	     //memcpy(outputhash, worker.padding, 32);
      // aoclSHA256(B, worker.data_len*4, worker.sHash);
      // worker.sHash = nHash;
    // }
    // else
    // {
    //   byte *s = new byte[MAX_LENGTH * 4];
    //   for (int i = 0; i < worker.data_len; i++)
    //   {
    //     s[i << 1] = htonl(worker.sa[i]);
    //   }
    //   hashSHA256(worker.sha256, s, worker.sHash, worker.data_len*4);
    //   // worker.sHash = nHash;
    //   delete[] s;
    // }
    //memcpy(outputhash, worker.sHash, 32);
    // memset(outputhash, 0xFF, 32);
  }
  catch (const std::exception &ex)
  {
    // recover(outputhash);
    std::cerr << ex.what() << std::endl;
  }
}

DIRTYBIRD_TARGETS
void branchComputeCPU(workerData &worker, bool isTest, int wIndex)

{
  worker.templateIdx = 0;
  uint8_t chunkCount = 1;
  int firstChunk = 0;

  uint8_t lp1 = 0;
  uint8_t lp2 = 255;

  while (true)
  {
    if(isTest) {

    } else {
      worker.tries[wIndex]++;
      #ifdef DEBUG_OP_ORDER
      if (debugOpOrder) printf("t: 0x%hx p: 0x%llx l: 0x%llx\n", worker.tries[wIndex], worker.prev_lhash, worker.lhash);
      #endif
      worker.random_switcher = worker.prev_lhash ^ worker.lhash ^ worker.tries[wIndex];
      // __builtin_prefetch(&worker.random_switcher,0,3);
      // printf("%d worker.random_switcher %d %08jx\n", worker.tries[wIndex], worker.random_switcher, worker.random_switcher);

      worker.op = static_cast<byte>(worker.random_switcher);
      //if (debugOpOrder) worker.opsA.push_back(worker.op);

      // printf("op: %d\n", worker.op);

      worker.pos1 = static_cast<byte>(worker.random_switcher >> 8);
      worker.pos2 = static_cast<byte>(worker.random_switcher >> 16);

      if (worker.pos1 > worker.pos2)
      {
        std::swap(worker.pos1, worker.pos2);
      }

      if (worker.pos2 - worker.pos1 > 32)
      {
        worker.pos2 = worker.pos1 + ((worker.pos2 - worker.pos1) & 0x1f);
      }

      if (worker.tries[wIndex] > 0) {
        lp1 = std::min(lp1, worker.pos1);
        lp2 = std::max(lp2, worker.pos2);
      }

      worker.chunk = &worker.sData[wIndex * ASTRO_SCRATCH_SIZE + (worker.tries[wIndex] - 1) * 256];
      #ifdef DEBUG_OP_ORDER
      if (debugOpOrder) printf("worker.op: %03d p1: %03d p2: %03d\n", worker.op, worker.pos1, worker.pos2);
      #endif

      if (worker.tries[wIndex] == 1) {
        worker.prev_chunk = worker.chunk;
      } else {
        worker.prev_chunk = &worker.sData[wIndex * ASTRO_SCRATCH_SIZE + (worker.tries[wIndex] - 2) * 256];
      }

      #ifdef DEBUG_OP_ORDER
      if (debugOpOrder) {
        printf("tries: %03hu chunk_before[  0->%03d]: ", worker.tries[wIndex], worker.pos2);
        for (int x = 0; x <= worker.pos2+16 && worker.pos2+16 < 256; x++) {
          printf("%02x", worker.chunk[x]);
        }
        printf("\n");
      }
      #endif

      memcpy(worker.chunk, worker.prev_chunk, 256);
      #ifdef DEBUG_OP_ORDER
      if (debugOpOrder) {
        printf("tries: %03hu  chunk_fixed[  0->%03d]: ", worker.tries[wIndex], worker.pos2);
        for (int x = 0; x <= worker.pos2+16 && worker.pos2+16 < 256; x++) {
          //printf("%d \n", x);
          printf("%02x", worker.chunk[x]);
        }
        printf("\n");
      }
      #endif
    }

    switch (worker.op)
    {
    case 0:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
        worker.chunk[i] = rl8(worker.chunk[i], 5);                // rotate  bits by 5
        worker.chunk[i] *= worker.chunk[i];                             // *
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random

        // INSERT_RANDOM_CODE_END
        worker.t1 = worker.chunk[worker.pos1];
        worker.t2 = worker.chunk[worker.pos2];
        worker.chunk[worker.pos1] = reverse8(worker.t2);
        worker.chunk[worker.pos2] = reverse8(worker.t1);
      }
      break;
    case 1:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = rl8(worker.chunk[i], 1);                // rotate  bits by 1
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] += worker.chunk[i];                             // +
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 2:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];          // ones count bits
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];          // ones count bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 3:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = rl8(worker.chunk[i], 3);                // rotate  bits by 3
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] = rl8(worker.chunk[i], 1);                // rotate  bits by 1
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 4:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);    // shift right
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                      // XOR and -
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 5:
    {
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {

        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];          // ones count bits
        worker.chunk[i] ^= worker.chunk[worker.pos2];                // XOR
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right

        // INSERT_RANDOM_CODE_END
      }
    }
    break;
    case 6:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] = rl8(worker.chunk[i], 3);             // rotate  bits by 3
        worker.chunk[i] = ~worker.chunk[i];                          // binary NOT operator
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                   // XOR and -

        // INSERT_RANDOM_CODE_END
      }
      break;
    case 7:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];                             // +
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 8:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];               // binary NOT operator
        worker.chunk[i] = rl8(worker.chunk[i], 10); // rotate  bits by 5
        // worker.chunk[i] = rl8(worker.chunk[i], 5);// rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 9:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];                // XOR
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);            // rotate  bits by 4
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);            // rotate  bits by 2
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 10:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];              // binary NOT operator
        worker.chunk[i] *= worker.chunk[i];              // *
        worker.chunk[i] = rl8(worker.chunk[i], 3); // rotate  bits by 3
        worker.chunk[i] *= worker.chunk[i];              // *
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 11:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 6); // rotate  bits by 1
        // worker.chunk[i] = rl8(worker.chunk[i], 5);            // rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 12:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] *= worker.chunk[i];               // *
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] = ~worker.chunk[i];               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 13:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 1);             // rotate  bits by 1
        worker.chunk[i] ^= worker.chunk[worker.pos2];                // XOR
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] = rl8(worker.chunk[i], 5);             // rotate  bits by 5
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 14:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] *= worker.chunk[i];                          // *
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 15:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                      // XOR and -
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 16:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
        worker.chunk[i] *= worker.chunk[i];               // *
        worker.chunk[i] = rl8(worker.chunk[i], 1);  // rotate  bits by 1
        worker.chunk[i] = ~worker.chunk[i];               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 17:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];    // XOR
        worker.chunk[i] *= worker.chunk[i];              // *
        worker.chunk[i] = rl8(worker.chunk[i], 5); // rotate  bits by 5
        worker.chunk[i] = ~worker.chunk[i];              // binary NOT operator
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 18:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
        worker.chunk[i] = rl8(worker.chunk[i], 9);  // rotate  bits by 3
        // worker.chunk[i] = rl8(worker.chunk[i], 1);             // rotate  bits by 1
        // worker.chunk[i] = rl8(worker.chunk[i], 5);         // rotate  bits by 5
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 19:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                   // XOR and -
        worker.chunk[i] = rl8(worker.chunk[i], 5);             // rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] += worker.chunk[i];                          // +
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 20:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] = reverse8(worker.chunk[i]);                    // reverse bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 21:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 1);                // rotate  bits by 1
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] += worker.chunk[i];                             // +
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 22:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] *= worker.chunk[i];                          // *
        worker.chunk[i] = rl8(worker.chunk[i], 1);             // rotate  bits by 1
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 23:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 4); // rotate  bits by 3
        // worker.chunk[i] = rl8(worker.chunk[i], 1);                           // rotate  bits by 1
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 24:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);            // rotate  bits by 4
        worker.chunk[i] = rl8(worker.chunk[i], 5);             // rotate  bits by 5
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 25:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
        worker.chunk[i] = rl8(worker.chunk[i], 3);                // rotate  bits by 3
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                      // XOR and -
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 26:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] *= worker.chunk[i];                 // *
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] += worker.chunk[i];                 // +
        worker.chunk[i] = reverse8(worker.chunk[i]);        // reverse bits
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 27:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 5);                // rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
        worker.chunk[i] = rl8(worker.chunk[i], 5);                // rotate  bits by 5
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 28:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] = rl8(worker.chunk[i], 5);             // rotate  bits by 5
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 29:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] *= worker.chunk[i];                          // *
        worker.chunk[i] ^= worker.chunk[worker.pos2];                // XOR
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] += worker.chunk[i];                          // +
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 30:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
        worker.chunk[i] = rl8(worker.chunk[i], 5);                // rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 31:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];                          // binary NOT operator
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);            // rotate  bits by 2
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] *= worker.chunk[i];                          // *
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 32:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] = reverse8(worker.chunk[i]);      // reverse bits
        worker.chunk[i] = rl8(worker.chunk[i], 3);  // rotate  bits by 3
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 33:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
        worker.chunk[i] = reverse8(worker.chunk[i]);                    // reverse bits
        worker.chunk[i] *= worker.chunk[i];                             // *
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 34:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                   // XOR and -
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                   // XOR and -
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 35:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];              // +
        worker.chunk[i] = ~worker.chunk[i];              // binary NOT operator
        worker.chunk[i] = rl8(worker.chunk[i], 1); // rotate  bits by 1
        worker.chunk[i] ^= worker.chunk[worker.pos2];    // XOR
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 36:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] = rl8(worker.chunk[i], 1);    // rotate  bits by 1
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);   // rotate  bits by 2
        worker.chunk[i] = rl8(worker.chunk[i], 1);    // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 37:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);    // shift right
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);    // shift right
        worker.chunk[i] *= worker.chunk[i];                             // *
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 38:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);    // shift right
        worker.chunk[i] = rl8(worker.chunk[i], 3);                // rotate  bits by 3
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 39:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);    // shift right
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 40:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 41:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 5);  // rotate  bits by 5
        worker.chunk[i] -= (worker.chunk[i] ^ 97);        // XOR and -
        worker.chunk[i] = rl8(worker.chunk[i], 3);  // rotate  bits by 3
        worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 42:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 4); // rotate  bits by 1
        // worker.chunk[i] = rl8(worker.chunk[i], 3);                // rotate  bits by 3
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 43:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] += worker.chunk[i];                             // +
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                      // XOR and -
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 44:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
        worker.chunk[i] = rl8(worker.chunk[i], 3);                // rotate  bits by 3
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 45:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 10); // rotate  bits by 5
        // worker.chunk[i] = rl8(worker.chunk[i], 5);                       // rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 46:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] += worker.chunk[i];                 // +
        worker.chunk[i] = rl8(worker.chunk[i], 5);    // rotate  bits by 5
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);   // rotate  bits by 4
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 47:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 5);                // rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = rl8(worker.chunk[i], 5);                // rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 48:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        // worker.chunk[i] = ~worker.chunk[i];                    // binary NOT operator
        // worker.chunk[i] = ~worker.chunk[i];                    // binary NOT operator
        worker.chunk[i] = rl8(worker.chunk[i], 5); // rotate  bits by 5
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 49:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] += worker.chunk[i];                 // +
        worker.chunk[i] = reverse8(worker.chunk[i]);        // reverse bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);   // rotate  bits by 4
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 50:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]);     // reverse bits
        worker.chunk[i] = rl8(worker.chunk[i], 3); // rotate  bits by 3
        worker.chunk[i] += worker.chunk[i];              // +
        worker.chunk[i] = rl8(worker.chunk[i], 1); // rotate  bits by 1
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 51:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];     // XOR
        worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
        worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
        worker.chunk[i] = rl8(worker.chunk[i], 5);  // rotate  bits by 5
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 52:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);    // shift right
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 53:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];                 // +
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);   // rotate  bits by 4
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);   // rotate  bits by 4
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 54:

#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]);  // reverse bits
        worker.chunk[i] ^= worker.chunk[worker.pos2]; // XOR
        // worker.chunk[i] = ~worker.chunk[i];    // binary NOT operator
        // worker.chunk[i] = ~worker.chunk[i];    // binary NOT operator
        // INSERT_RANDOM_CODE_END
      }

      break;
    case 55:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]);      // reverse bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
        worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
        worker.chunk[i] = rl8(worker.chunk[i], 1);  // rotate  bits by 1
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 56:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] *= worker.chunk[i];               // *
        worker.chunk[i] = ~worker.chunk[i];               // binary NOT operator
        worker.chunk[i] = rl8(worker.chunk[i], 1);  // rotate  bits by 1
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 57:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = rl8(worker.chunk[i], 8);                // rotate  bits by 5
        // worker.chunk[i] = rl8(worker.chunk[i], 3);                // rotate  bits by 3
        worker.chunk[i] = reverse8(worker.chunk[i]); // reverse bits
                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 58:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]);                    // reverse bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] += worker.chunk[i];                             // +
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 59:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 1);                // rotate  bits by 1
        worker.chunk[i] *= worker.chunk[i];                             // *
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 60:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];    // XOR
        worker.chunk[i] = ~worker.chunk[i];              // binary NOT operator
        worker.chunk[i] *= worker.chunk[i];              // *
        worker.chunk[i] = rl8(worker.chunk[i], 3); // rotate  bits by 3
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 61:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 5);             // rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] = rl8(worker.chunk[i], 8);             // rotate  bits by 3
        // worker.chunk[i] = rl8(worker.chunk[i], 5);// rotate  bits by 5
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 62:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
        worker.chunk[i] += worker.chunk[i];                             // +
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 63:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 5);    // rotate  bits by 5
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] -= (worker.chunk[i] ^ 97);          // XOR and -
        worker.chunk[i] += worker.chunk[i];                 // +
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 64:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];     // XOR
        worker.chunk[i] = reverse8(worker.chunk[i]);      // reverse bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
        worker.chunk[i] *= worker.chunk[i];               // *
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 65:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 8); // rotate  bits by 5
        // worker.chunk[i] = rl8(worker.chunk[i], 3);             // rotate  bits by 3
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] *= worker.chunk[i];               // *
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 66:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] = reverse8(worker.chunk[i]);      // reverse bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
        worker.chunk[i] = rl8(worker.chunk[i], 1);  // rotate  bits by 1
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 67:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 1);    // rotate  bits by 1
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);   // rotate  bits by 2
        worker.chunk[i] = rl8(worker.chunk[i], 5);    // rotate  bits by 5
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 68:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 69:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] *= worker.chunk[i];                          // *
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 70:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];                // XOR
        worker.chunk[i] *= worker.chunk[i];                          // *
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);            // rotate  bits by 4
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 71:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 5);             // rotate  bits by 5
        worker.chunk[i] = ~worker.chunk[i];                          // binary NOT operator
        worker.chunk[i] *= worker.chunk[i];                          // *
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 72:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];          // ones count bits
        worker.chunk[i] ^= worker.chunk[worker.pos2];                // XOR
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 73:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] = reverse8(worker.chunk[i]);        // reverse bits
        worker.chunk[i] = rl8(worker.chunk[i], 5);    // rotate  bits by 5
        worker.chunk[i] -= (worker.chunk[i] ^ 97);          // XOR and -
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 74:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] *= worker.chunk[i];                             // *
        worker.chunk[i] = rl8(worker.chunk[i], 3);                // rotate  bits by 3
        worker.chunk[i] = reverse8(worker.chunk[i]);                    // reverse bits
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 75:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] *= worker.chunk[i];                             // *
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 76:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
        worker.chunk[i] = rl8(worker.chunk[i], 5);                // rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);    // shift right
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 77:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 3);             // rotate  bits by 3
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];          // ones count bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 78:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = reverse8(worker.chunk[i]);                    // reverse bits
        worker.chunk[i] *= worker.chunk[i];                             // *
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                      // XOR and -
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 79:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] += worker.chunk[i];               // +
        worker.chunk[i] *= worker.chunk[i];               // *
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 80:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] += worker.chunk[i];                             // +
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 81:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 82:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2]; // XOR
        // worker.chunk[i] = ~worker.chunk[i];        // binary NOT operator
        // worker.chunk[i] = ~worker.chunk[i];        // binary NOT operator
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 83:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] = rl8(worker.chunk[i], 3);             // rotate  bits by 3
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 84:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                   // XOR and -
        worker.chunk[i] = rl8(worker.chunk[i], 1);             // rotate  bits by 1
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] += worker.chunk[i];                          // +
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 85:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);    // shift right
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 86:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 87:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];               // +
        worker.chunk[i] = rl8(worker.chunk[i], 3);  // rotate  bits by 3
        worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
        worker.chunk[i] += worker.chunk[i];               // +
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 88:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] = rl8(worker.chunk[i], 1);  // rotate  bits by 1
        worker.chunk[i] *= worker.chunk[i];               // *
        worker.chunk[i] = ~worker.chunk[i];               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 89:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];               // +
        worker.chunk[i] *= worker.chunk[i];               // *
        worker.chunk[i] = ~worker.chunk[i];               // binary NOT operator
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 90:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]);     // reverse bits
        worker.chunk[i] = rl8(worker.chunk[i], 6); // rotate  bits by 5
        // worker.chunk[i] = rl8(worker.chunk[i], 1);    // rotate  bits by 1
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 91:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
        worker.chunk[i] = reverse8(worker.chunk[i]);                    // reverse bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 92:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 93:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
        worker.chunk[i] *= worker.chunk[i];                             // *
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] += worker.chunk[i];                             // +
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 94:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 1);                // rotate  bits by 1
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 95:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 1);  // rotate  bits by 1
        worker.chunk[i] = ~worker.chunk[i];               // binary NOT operator
        worker.chunk[i] = rl8(worker.chunk[i], 10); // rotate  bits by 5
        // worker.chunk[i] = rl8(worker.chunk[i], 5); // rotate  bits by 5
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 96:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);   // rotate  bits by 2
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);   // rotate  bits by 2
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] = rl8(worker.chunk[i], 1);    // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 97:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 1);             // rotate  bits by 1
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];          // ones count bits
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 98:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);            // rotate  bits by 4
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);            // rotate  bits by 4
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 99:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);            // rotate  bits by 4
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                   // XOR and -
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 100:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = reverse8(worker.chunk[i]);                    // reverse bits
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 101:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];          // ones count bits
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] = ~worker.chunk[i];                          // binary NOT operator
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 102:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 3); // rotate  bits by 3
        worker.chunk[i] -= (worker.chunk[i] ^ 97);       // XOR and -
        worker.chunk[i] += worker.chunk[i];              // +
        worker.chunk[i] = rl8(worker.chunk[i], 3); // rotate  bits by 3
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 103:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 1);                // rotate  bits by 1
        worker.chunk[i] = reverse8(worker.chunk[i]);                    // reverse bits
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 104:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]);        // reverse bits
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] = rl8(worker.chunk[i], 5);    // rotate  bits by 5
        worker.chunk[i] += worker.chunk[i];                 // +
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 105:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = rl8(worker.chunk[i], 3);                // rotate  bits by 3
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 106:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]);      // reverse bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
        worker.chunk[i] = rl8(worker.chunk[i], 1);  // rotate  bits by 1
        worker.chunk[i] *= worker.chunk[i];               // *
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 107:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);            // rotate  bits by 2
        worker.chunk[i] = rl8(worker.chunk[i], 6);             // rotate  bits by 5
        // worker.chunk[i] = rl8(worker.chunk[i], 1);             // rotate  bits by 1
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 108:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 109:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] *= worker.chunk[i];                             // *
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 110:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);            // rotate  bits by 2
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);            // rotate  bits by 2
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 111:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] *= worker.chunk[i];                          // *
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] *= worker.chunk[i];                          // *
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 112:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 3); // rotate  bits by 3
        worker.chunk[i] = ~worker.chunk[i];              // binary NOT operator
        worker.chunk[i] = rl8(worker.chunk[i], 5); // rotate  bits by 5
        worker.chunk[i] -= (worker.chunk[i] ^ 97);       // XOR and -
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 113:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 6); // rotate  bits by 5
        // worker.chunk[i] = rl8(worker.chunk[i], 1);                           // rotate  bits by 1
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] = ~worker.chunk[i];                 // binary NOT operator
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 114:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 1);                // rotate  bits by 1
        worker.chunk[i] = reverse8(worker.chunk[i]);                    // reverse bits
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 115:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = rl8(worker.chunk[i], 5);                // rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = rl8(worker.chunk[i], 3);                // rotate  bits by 3
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 116:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 117:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = rl8(worker.chunk[i], 3);                // rotate  bits by 3
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 118:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] = rl8(worker.chunk[i], 5);             // rotate  bits by 5
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 119:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]);      // reverse bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] = ~worker.chunk[i];               // binary NOT operator
        worker.chunk[i] ^= worker.chunk[worker.pos2];     // XOR
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 120:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] *= worker.chunk[i];               // *
        worker.chunk[i] ^= worker.chunk[worker.pos2];     // XOR
        worker.chunk[i] = reverse8(worker.chunk[i]);      // reverse bits
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 121:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];          // ones count bits
        worker.chunk[i] *= worker.chunk[i];                          // *
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 122:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = rl8(worker.chunk[i], 5);                // rotate  bits by 5
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 123:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] = rl8(worker.chunk[i], 6);                // rotate  bits by 3
        // worker.chunk[i] = rl8(worker.chunk[i], 3); // rotate  bits by 3
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 124:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] ^= worker.chunk[worker.pos2];     // XOR
        worker.chunk[i] = ~worker.chunk[i];               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 125:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);            // rotate  bits by 2
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 126:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 9); // rotate  bits by 3
        // worker.chunk[i] = rl8(worker.chunk[i], 1); // rotate  bits by 1
        // worker.chunk[i] = rl8(worker.chunk[i], 5); // rotate  bits by 5
        worker.chunk[i] = reverse8(worker.chunk[i]); // reverse bits
                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 127:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] *= worker.chunk[i];                             // *
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 128:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
        worker.chunk[i] = rl8(worker.chunk[i], 5);                // rotate  bits by 5
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 129:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];                          // binary NOT operator
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];          // ones count bits
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];          // ones count bits
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 130:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);    // shift right
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = rl8(worker.chunk[i], 1);                // rotate  bits by 1
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 131:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] -= (worker.chunk[i] ^ 97);          // XOR and -
        worker.chunk[i] = rl8(worker.chunk[i], 1);    // rotate  bits by 1
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] *= worker.chunk[i];                 // *
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 132:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = reverse8(worker.chunk[i]);                    // reverse bits
        worker.chunk[i] = rl8(worker.chunk[i], 5);                // rotate  bits by 5
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 133:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];                // XOR
        worker.chunk[i] = rl8(worker.chunk[i], 5);             // rotate  bits by 5
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);            // rotate  bits by 2
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 134:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
        worker.chunk[i] = rl8(worker.chunk[i], 1);                // rotate  bits by 1
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 135:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);            // rotate  bits by 2
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 136:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                   // XOR and -
        worker.chunk[i] ^= worker.chunk[worker.pos2];                // XOR
        worker.chunk[i] = rl8(worker.chunk[i], 5);             // rotate  bits by 5
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 137:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 5);                // rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);    // shift right
        worker.chunk[i] = reverse8(worker.chunk[i]);                    // reverse bits
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 138:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2]; // XOR
        worker.chunk[i] ^= worker.chunk[worker.pos2]; // XOR
        worker.chunk[i] += worker.chunk[i];           // +
        worker.chunk[i] -= (worker.chunk[i] ^ 97);    // XOR and -
                                                        // INSERT_RANDOM_CODE_END
      }
      break;
    case 139:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 8); // rotate  bits by 5
        // worker.chunk[i] = rl8(worker.chunk[i], 3);             // rotate  bits by 3
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] = rl8(worker.chunk[i], 3);  // rotate  bits by 3
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 140:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 1);  // rotate  bits by 1
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] ^= worker.chunk[worker.pos2];     // XOR
        worker.chunk[i] = rl8(worker.chunk[i], 5);  // rotate  bits by 5
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 141:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 1);    // rotate  bits by 1
        worker.chunk[i] -= (worker.chunk[i] ^ 97);          // XOR and -
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] += worker.chunk[i];                 // +
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 142:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = rl8(worker.chunk[i], 5);                // rotate  bits by 5
        worker.chunk[i] = reverse8(worker.chunk[i]);                    // reverse bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 143:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = rl8(worker.chunk[i], 3);                // rotate  bits by 3
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);    // shift right
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 144:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 145:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]);      // reverse bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 146:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 147:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];                          // binary NOT operator
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);            // rotate  bits by 4
        worker.chunk[i] *= worker.chunk[i];                          // *
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 148:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = rl8(worker.chunk[i], 5);                // rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                      // XOR and -
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 149:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2]; // XOR
        worker.chunk[i] = reverse8(worker.chunk[i]);  // reverse bits
        worker.chunk[i] -= (worker.chunk[i] ^ 97);    // XOR and -
        worker.chunk[i] += worker.chunk[i];           // +
                                                        // INSERT_RANDOM_CODE_END
      }
      break;
    case 150:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 151:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] *= worker.chunk[i];                          // *
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 152:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] = ~worker.chunk[i];                          // binary NOT operator
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);            // rotate  bits by 2
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 153:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 4); // rotate  bits by 1
        // worker.chunk[i] = rl8(worker.chunk[i], 3); // rotate  bits by 3
        // worker.chunk[i] = ~worker.chunk[i];     // binary NOT operator
        // worker.chunk[i] = ~worker.chunk[i];     // binary NOT operator
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 154:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 5);    // rotate  bits by 5
        worker.chunk[i] = ~worker.chunk[i];                 // binary NOT operator
        worker.chunk[i] ^= worker.chunk[worker.pos2];       // XOR
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 155:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] -= (worker.chunk[i] ^ 97);          // XOR and -
        worker.chunk[i] ^= worker.chunk[worker.pos2];       // XOR
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] ^= worker.chunk[worker.pos2];       // XOR
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 156:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] = rl8(worker.chunk[i], 4);             // rotate  bits by 3
        // worker.chunk[i] = rl8(worker.chunk[i], 1);    // rotate  bits by 1
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 157:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);    // shift right
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = rl8(worker.chunk[i], 1);                // rotate  bits by 1
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 158:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] = rl8(worker.chunk[i], 3);    // rotate  bits by 3
        worker.chunk[i] += worker.chunk[i];                 // +
        worker.chunk[i] = rl8(worker.chunk[i], 1);    // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 159:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                      // XOR and -
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 160:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] = rl8(worker.chunk[i], 4);             // rotate  bits by 1
        // worker.chunk[i] = rl8(worker.chunk[i], 3);    // rotate  bits by 3
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 161:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] = rl8(worker.chunk[i], 5);                // rotate  bits by 5
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 162:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] *= worker.chunk[i];               // *
        worker.chunk[i] = reverse8(worker.chunk[i]);      // reverse bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] -= (worker.chunk[i] ^ 97);        // XOR and -
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 163:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                   // XOR and -
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);            // rotate  bits by 4
        worker.chunk[i] = rl8(worker.chunk[i], 1);             // rotate  bits by 1
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 164:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] *= worker.chunk[i];                 // *
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] -= (worker.chunk[i] ^ 97);          // XOR and -
        worker.chunk[i] = ~worker.chunk[i];                 // binary NOT operator
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 165:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);            // rotate  bits by 4
        worker.chunk[i] ^= worker.chunk[worker.pos2];                // XOR
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] += worker.chunk[i];                          // +
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 166:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 3);  // rotate  bits by 3
        worker.chunk[i] += worker.chunk[i];               // +
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] = ~worker.chunk[i];               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 167:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        // worker.chunk[i] = ~worker.chunk[i];        // binary NOT operator
        // worker.chunk[i] = ~worker.chunk[i];        // binary NOT operator
        worker.chunk[i] *= worker.chunk[i];                          // *
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 168:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = rl8(worker.chunk[i], 1);                // rotate  bits by 1
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 169:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 1);                // rotate  bits by 1
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 170:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] -= (worker.chunk[i] ^ 97);   // XOR and -
        worker.chunk[i] = reverse8(worker.chunk[i]); // reverse bits
        worker.chunk[i] -= (worker.chunk[i] ^ 97);   // XOR and -
        worker.chunk[i] *= worker.chunk[i];          // *
                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 171:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 3);    // rotate  bits by 3
        worker.chunk[i] -= (worker.chunk[i] ^ 97);          // XOR and -
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] = reverse8(worker.chunk[i]);        // reverse bits
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 172:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);            // rotate  bits by 4
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                   // XOR and -
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] = rl8(worker.chunk[i], 1);             // rotate  bits by 1
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 173:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];                          // binary NOT operator
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] *= worker.chunk[i];                          // *
        worker.chunk[i] += worker.chunk[i];                          // +
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 174:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 175:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 3); // rotate  bits by 3
        worker.chunk[i] -= (worker.chunk[i] ^ 97);       // XOR and -
        worker.chunk[i] *= worker.chunk[i];              // *
        worker.chunk[i] = rl8(worker.chunk[i], 5); // rotate  bits by 5
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 176:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];    // XOR
        worker.chunk[i] *= worker.chunk[i];              // *
        worker.chunk[i] ^= worker.chunk[worker.pos2];    // XOR
        worker.chunk[i] = rl8(worker.chunk[i], 5); // rotate  bits by 5
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 177:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 178:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] += worker.chunk[i];                             // +
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] = rl8(worker.chunk[i], 1);                // rotate  bits by 1
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 179:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);            // rotate  bits by 2
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 180:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);            // rotate  bits by 4
        worker.chunk[i] ^= worker.chunk[worker.pos2];                // XOR
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                   // XOR and -
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 181:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];                          // binary NOT operator
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);            // rotate  bits by 2
        worker.chunk[i] = rl8(worker.chunk[i], 5);             // rotate  bits by 5
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 182:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];    // XOR
        worker.chunk[i] = rl8(worker.chunk[i], 6); // rotate  bits by 1
        // worker.chunk[i] = rl8(worker.chunk[i], 5);         // rotate  bits by 5
        worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 183:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];        // +
        worker.chunk[i] -= (worker.chunk[i] ^ 97); // XOR and -
        worker.chunk[i] -= (worker.chunk[i] ^ 97); // XOR and -
        worker.chunk[i] *= worker.chunk[i];        // *
                                                     // INSERT_RANDOM_CODE_END
      }
      break;
    case 184:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] *= worker.chunk[i];                          // *
        worker.chunk[i] = rl8(worker.chunk[i], 5);             // rotate  bits by 5
        worker.chunk[i] ^= worker.chunk[worker.pos2];                // XOR
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 185:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];                          // binary NOT operator
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);            // rotate  bits by 4
        worker.chunk[i] = rl8(worker.chunk[i], 5);             // rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 186:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);            // rotate  bits by 2
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);            // rotate  bits by 4
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                   // XOR and -
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 187:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];    // XOR
        worker.chunk[i] = ~worker.chunk[i];              // binary NOT operator
        worker.chunk[i] += worker.chunk[i];              // +
        worker.chunk[i] = rl8(worker.chunk[i], 3); // rotate  bits by 3
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 188:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);   // rotate  bits by 4
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);   // rotate  bits by 4
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);   // rotate  bits by 4
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 189:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 5);  // rotate  bits by 5
        worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
        worker.chunk[i] ^= worker.chunk[worker.pos2];     // XOR
        worker.chunk[i] -= (worker.chunk[i] ^ 97);        // XOR and -
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 190:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 5);                // rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);    // shift right
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 191:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];                             // +
        worker.chunk[i] = rl8(worker.chunk[i], 3);                // rotate  bits by 3
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);    // shift right
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 192:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] *= worker.chunk[i];                          // *
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 193:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = rl8(worker.chunk[i], 1);                // rotate  bits by 1
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 194:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 195:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);   // rotate  bits by 2
        worker.chunk[i] ^= worker.chunk[worker.pos2];       // XOR
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);   // rotate  bits by 4
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 196:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 3);             // rotate  bits by 3
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] = rl8(worker.chunk[i], 1);             // rotate  bits by 1
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 197:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] *= worker.chunk[i];                             // *
        worker.chunk[i] *= worker.chunk[i];                             // *
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 198:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] = rl8(worker.chunk[i], 1);             // rotate  bits by 1
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 199:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];           // binary NOT operator
        worker.chunk[i] += worker.chunk[i];           // +
        worker.chunk[i] *= worker.chunk[i];           // *
        worker.chunk[i] ^= worker.chunk[worker.pos2]; // XOR
                                                        // INSERT_RANDOM_CODE_END
      }
      break;
    case 200:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];          // ones count bits
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 201:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 3);  // rotate  bits by 3
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
        worker.chunk[i] = ~worker.chunk[i];               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 202:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = rl8(worker.chunk[i], 5);                // rotate  bits by 5
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 203:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = rl8(worker.chunk[i], 1);                // rotate  bits by 1
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 204:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 5);                // rotate  bits by 5
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 205:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];          // ones count bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);            // rotate  bits by 4
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] += worker.chunk[i];                          // +
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 206:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);   // rotate  bits by 4
        worker.chunk[i] = reverse8(worker.chunk[i]);        // reverse bits
        worker.chunk[i] = reverse8(worker.chunk[i]);        // reverse bits
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 207:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 8); // rotate  bits by 5
        // worker.chunk[i] = rl8(worker.chunk[i], 3);                           // rotate  bits by 3
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 208:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] = rl8(worker.chunk[i], 3);             // rotate  bits by 3
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 209:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 5);    // rotate  bits by 5
        worker.chunk[i] = reverse8(worker.chunk[i]);        // reverse bits
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] -= (worker.chunk[i] ^ 97);          // XOR and -
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 210:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = rl8(worker.chunk[i], 5);                // rotate  bits by 5
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 211:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
        worker.chunk[i] += worker.chunk[i];                             // +
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                      // XOR and -
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 212:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 213:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] = rl8(worker.chunk[i], 3);             // rotate  bits by 3
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                   // XOR and -
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 214:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];                // XOR
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                   // XOR and -
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] = ~worker.chunk[i];                          // binary NOT operator
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 215:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] *= worker.chunk[i];                             // *
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 216:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                      // XOR and -
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 217:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 5);  // rotate  bits by 5
        worker.chunk[i] += worker.chunk[i];               // +
        worker.chunk[i] = rl8(worker.chunk[i], 1);  // rotate  bits by 1
        worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 218:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]); // reverse bits
        worker.chunk[i] = ~worker.chunk[i];          // binary NOT operator
        worker.chunk[i] *= worker.chunk[i];          // *
        worker.chunk[i] -= (worker.chunk[i] ^ 97);   // XOR and -
                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 219:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
        worker.chunk[i] = rl8(worker.chunk[i], 3);                // rotate  bits by 3
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = reverse8(worker.chunk[i]);                    // reverse bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 220:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 1);             // rotate  bits by 1
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 221:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 5); // rotate  bits by 5
        worker.chunk[i] ^= worker.chunk[worker.pos2];    // XOR
        worker.chunk[i] = ~worker.chunk[i];              // binary NOT operator
        worker.chunk[i] = reverse8(worker.chunk[i]);     // reverse bits
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 222:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] ^= worker.chunk[worker.pos2];                // XOR
        worker.chunk[i] *= worker.chunk[i];                          // *
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 223:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 3);                // rotate  bits by 3
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                      // XOR and -
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 224:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] = rl8(worker.chunk[i], 4);  // rotate  bits by 1
        // worker.chunk[i] = rl8(worker.chunk[i], 3);             // rotate  bits by 3
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
                                                                       //
      }
      break;
    case 225:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];                          // binary NOT operator
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] = rl8(worker.chunk[i], 3);             // rotate  bits by 3
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 226:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]);  // reverse bits
        worker.chunk[i] -= (worker.chunk[i] ^ 97);    // XOR and -
        worker.chunk[i] *= worker.chunk[i];           // *
        worker.chunk[i] ^= worker.chunk[worker.pos2]; // XOR
                                                        // INSERT_RANDOM_CODE_END
      }
      break;
    case 227:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                      // XOR and -
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 228:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];          // ones count bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 229:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 3);                // rotate  bits by 3
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 230:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] *= worker.chunk[i];                             // *
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 231:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 3);             // rotate  bits by 3
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] ^= worker.chunk[worker.pos2];                // XOR
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 232:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] *= worker.chunk[i];               // *
        worker.chunk[i] *= worker.chunk[i];               // *
        worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
        worker.chunk[i] = rl8(worker.chunk[i], 5);  // rotate  bits by 5
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 233:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 1);    // rotate  bits by 1
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] = rl8(worker.chunk[i], 3);    // rotate  bits by 3
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 234:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] *= worker.chunk[i];                             // *
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);    // shift right
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 235:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] *= worker.chunk[i];               // *
        worker.chunk[i] = rl8(worker.chunk[i], 3);  // rotate  bits by 3
        worker.chunk[i] = ~worker.chunk[i];               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 236:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] += worker.chunk[i];                             // +
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                      // XOR and -
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 237:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 5);             // rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);            // rotate  bits by 2
        worker.chunk[i] = rl8(worker.chunk[i], 3);             // rotate  bits by 3
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 238:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];              // +
        worker.chunk[i] += worker.chunk[i];              // +
        worker.chunk[i] = rl8(worker.chunk[i], 3); // rotate  bits by 3
        worker.chunk[i] -= (worker.chunk[i] ^ 97);       // XOR and -
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 239:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 6); // rotate  bits by 5
        // worker.chunk[i] = rl8(worker.chunk[i], 1); // rotate  bits by 1
        worker.chunk[i] *= worker.chunk[i];                             // *
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 240:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] += worker.chunk[i];                             // +
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 241:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);   // rotate  bits by 4
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] ^= worker.chunk[worker.pos2];       // XOR
        worker.chunk[i] = rl8(worker.chunk[i], 1);    // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 242:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];           // +
        worker.chunk[i] += worker.chunk[i];           // +
        worker.chunk[i] -= (worker.chunk[i] ^ 97);    // XOR and -
        worker.chunk[i] ^= worker.chunk[worker.pos2]; // XOR
                                                        // INSERT_RANDOM_CODE_END
      }
      break;
    case 243:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 5);    // rotate  bits by 5
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);   // rotate  bits by 2
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] = rl8(worker.chunk[i], 1);    // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 244:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];               // binary NOT operator
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] = reverse8(worker.chunk[i]);      // reverse bits
        worker.chunk[i] = rl8(worker.chunk[i], 5);  // rotate  bits by 5
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 245:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                   // XOR and -
        worker.chunk[i] = rl8(worker.chunk[i], 5);             // rotate  bits by 5
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);            // rotate  bits by 2
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 246:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] = rl8(worker.chunk[i], 1);             // rotate  bits by 1
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] += worker.chunk[i];                          // +
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 247:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 5);  // rotate  bits by 5
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] = rl8(worker.chunk[i], 5);  // rotate  bits by 5
        worker.chunk[i] = ~worker.chunk[i];               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 248:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];                 // binary NOT operator
        worker.chunk[i] -= (worker.chunk[i] ^ 97);          // XOR and -
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] = rl8(worker.chunk[i], 5);    // rotate  bits by 5
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 249:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]);                    // reverse bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 250:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 251:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];                 // +
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] = reverse8(worker.chunk[i]);        // reverse bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);   // rotate  bits by 2
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 252:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);            // rotate  bits by 4
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);            // rotate  bits by 2
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 253:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 3);  // rotate  bits by 3
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] ^= worker.chunk[worker.pos2];     // XOR
        worker.chunk[i] = rl8(worker.chunk[i], 3);  // rotate  bits by 3
        // INSERT_RANDOM_CODE_END

        worker.prev_lhash = worker.lhash + worker.prev_lhash;
        worker.lhash = XXHash64::hash(worker.chunk, worker.pos2,0);
      }
      break;
    case 254:
    case 255:
      RC4_set_key(&worker.key[wIndex], 256,  worker.chunk);
// worker.chunk = highwayhash.Sum(worker.chunk[:], worker.chunk[:])
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= static_cast<uint8_t>(std::bitset<8>(worker.chunk[i]).count()); // ones count bits
        worker.chunk[i] = rl8(worker.chunk[i], 3);                                  // rotate  bits by 3
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);                                 // rotate  bits by 2
        worker.chunk[i] = rl8(worker.chunk[i], 3);                                  // rotate  bits by 3
                                                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    default:
      break;
    }

    if(isTest) {
      break;
    }

    // if (op == 53) {
    //   std::cout << hexStr(worker.chunk, 256) << std::endl << std::endl;
    //   std::cout << hexStr(&worker.chunk[worker.pos1], 1) << std::endl;
    //   std::cout << hexStr(&worker.chunk[worker.pos2], 1) << std::endl;
    // }

    uint8_t pushPos1 = lp1;
    uint8_t pushPos2 = lp2;

    if (worker.pos1 == worker.pos2) {
      pushPos1 = -1;
      pushPos2 = -1;
    }

    worker.A = (worker.chunk[worker.pos1] - worker.chunk[worker.pos2]);
    worker.A = (256 + (worker.A % 256)) % 256;

    #ifdef DEBUG_OP_ORDER
    if (debugOpOrder){printf("worker.A: %02X\n", worker.A);}
    #endif

    if (worker.A < 0x10)
    { // 6.25 % probability
      #ifdef DEBUG_OP_ORDER
      if (debugOpOrder){printf("A\n");}
      #endif
      __builtin_prefetch(worker.chunk, 0, 0);
      worker.prev_lhash = worker.lhash + worker.prev_lhash;
      worker.lhash = XXHash64::hash(worker.chunk, worker.pos2, 0);
      // if (debugOpOrder) printf("A: new worker.lhash: %08jx\n", worker.lhash);
    }

    if (worker.A < 0x20)
    { // 12.5 % probability
      #ifdef DEBUG_OP_ORDER
      if (debugOpOrder){printf("B\n");}
      #endif
      __builtin_prefetch(worker.chunk, 0, 0);
      worker.prev_lhash = worker.lhash + worker.prev_lhash;
      worker.lhash = hash_64_fnv1a(worker.chunk, worker.pos2);
      // if (debugOpOrder) printf("B: new worker.lhash: %08jx\n", worker.lhash);
    }

    if (worker.A < 0x30)
    { // 18.75 % probability
      // std::copy(worker.chunk, worker.chunk + worker.pos2, s3);
      #ifdef DEBUG_OP_ORDER
      if (debugOpOrder){printf("C\n");}
      #endif
      __builtin_prefetch(worker.chunk, 0, 0);
      worker.prev_lhash = worker.lhash + worker.prev_lhash;
      HH_ALIGNAS(16)
      const highwayhash::HH_U64 key2[2] = {worker.tries[wIndex], worker.prev_lhash};
      worker.lhash = highwayhash::SipHash(key2, (char*)worker.chunk, worker.pos2); // more deviations
      // if (debugOpOrder) printf("C: new worker.lhash: %08jx\n", worker.lhash);
    }

    if (worker.A <= 0x40)
    { // 25% probablility
      // if (debugOpOrder) {
      //   printf("D: RC4 key:\n");
      //   for (int i = 0; i < 256; i++) {
      //     printf("%d, ", worker.key.data[i]);
      //   }
      // }
      #ifdef DEBUG_OP_ORDER
      if (debugOpOrder){printf("D\n");}
      #endif
      __builtin_prefetch(&worker.key[wIndex], 0, 0);
      RC4(&worker.key[wIndex], 256, worker.chunk,  worker.chunk);
      if (255 - pushPos2 < MINPREFLEN)
        pushPos2 = 255;
      if (pushPos1 < MINPREFLEN)
        pushPos1 = 0;


      if (pushPos1 == 255) pushPos1 = 0;
      
      worker.astroTemplate[worker.templateIdx] = templateMarker{
        (uint8_t)(chunkCount > 1 ? pushPos1 : 0),
        (uint8_t)(chunkCount > 1 ? pushPos2 : 255),
        (uint16_t)0,
        (uint16_t)0,
        (uint16_t)((firstChunk << 7) | chunkCount)
      };

      pushPos1 = 0;
      pushPos2 = 255;
      worker.templateIdx += (worker.tries[wIndex] > 1);
      firstChunk = worker.tries[wIndex]-1;
      lp1 = 255;
      lp2 = 0;
      chunkCount = 1;
    } else {
      chunkCount++;
    }

    if (255 - pushPos2 < MINPREFLEN)
      pushPos2 = 255;
    if (pushPos1 < MINPREFLEN)
      pushPos1 = 0;

    worker.chunk[255] = worker.chunk[255] ^ worker.chunk[worker.pos1] ^ worker.chunk[worker.pos2];

    prefetch(worker.chunk, 256, 1);
    memcpy(&worker.sData[(worker.tries[wIndex] - 1) * 256], worker.chunk, 256);

    #ifdef DEBUG_OP_ORDER
    if (debugOpOrder && worker.op == sus_op) {
      printf("op %d result:\n", worker.op);
      for (int i = 0; i < 256; i++) {
          printf("%02X ", worker.chunk[i]);
      } 
      printf("\n");
    }
    #endif
    // std::copy(worker.chunk, worker.chunk + 256, &worker.sData[(worker.tries[wIndex] - 1) * 256]);

    // memcpy(&worker->data.data()[(worker.tries[wIndex] - 1) * 256], worker.chunk, 256);

    // std::cout << hexStr(worker.chunk, 256) << std::endl;

    if (worker.tries[wIndex] > 260 + 16 || (worker.chunk[255] >= 0xf0 && worker.tries[wIndex] > 260))
    {
      break;
    }
  }

  if (chunkCount > 0) {
    if (255 - lp2 < MINPREFLEN)
      lp2 = 255;
    if (lp1 < MINPREFLEN)
      lp1 = 0;
    worker.astroTemplate[worker.templateIdx] = templateMarker{
      (uint8_t)(chunkCount > 1 ? lp1 : 0),
      (uint8_t)(chunkCount > 1 ? lp2 : 255),
      (uint16_t)0,
      (uint16_t)0,
      (uint16_t)((firstChunk << 7) | chunkCount)
    };
    worker.templateIdx++;
  }

  worker.data_len = static_cast<uint32_t>((worker.tries[wIndex] - 4) * 256 + (((static_cast<uint64_t>(worker.chunk[253]) << 8) | static_cast<uint64_t>(worker.chunk[254])) & 0x3ff));
}

// #if defined(__AVX2__)

// void branchComputeCPU_avx2(workerData &worker, bool isTest, int wIndex)
// {
//   worker.templateIdx = 0;
//   uint8_t chunkCount = 1;
//   int firstChunk = 0;

//   uint8_t lp1 = 0;
//   uint8_t lp2 = 255;
//   while (true)
//   {
//     if(isTest) {

//     } else {
//       worker.tries[wIndex]++;
//       worker.random_switcher = worker.prev_lhash ^ worker.lhash ^ worker.tries[wIndex];
//       // __builtin_prefetch(&worker.random_switcher,0,3);
//       // printf("%d worker.random_switcher %d %08jx\n", worker.tries[wIndex], worker.random_switcher, worker.random_switcher);

//       worker.op = static_cast<byte>(worker.random_switcher);
//       // if (debugOpOrder) worker.opsA.push_back(worker.op);

//       // printf("op: %d\n", worker.op);

//       worker.pos1 = static_cast<byte>(worker.random_switcher >> 8);
//       worker.pos2 = static_cast<byte>(worker.random_switcher >> 16);

//       if (worker.pos1 > worker.pos2)
//       {
//         std::swap(worker.pos1, worker.pos2);
//       }

//       if (worker.pos2 - worker.pos1 > 32)
//       {
//         worker.pos2 = worker.pos1 + ((worker.pos2 - worker.pos1) & 0x1f);
//       }

//       if (worker.tries[wIndex] > 0) {
//         lp1 = std::min(lp1, worker.pos1);
//         lp2 = std::max(lp2, worker.pos2);
//       }

//       worker.chunk = &worker.sData[wIndex * ASTRO_SCRATCH_SIZE + (worker.tries[wIndex] - 1) * 256];

//       if (worker.tries[wIndex] == 1) {
//         worker.prev_chunk = worker.chunk;
//       } else {
//         worker.prev_chunk = &worker.sData[wIndex * ASTRO_SCRATCH_SIZE + (worker.tries[wIndex] - 2) * 256];

//         __builtin_prefetch(worker.prev_chunk,0,3);
//         __builtin_prefetch(worker.prev_chunk+64,0,3);
//         __builtin_prefetch(worker.prev_chunk+128,0,3);
//         __builtin_prefetch(worker.prev_chunk+192,0,3);

//         // Calculate the start and end blocks
//         int start_block = 0;
//         int end_block = worker.pos1 / 16;

//         // Copy the blocks before worker.pos1
//         for (int i = start_block; i < end_block; i++) {
//             __m128i prev_data = _mm_loadu_si128((__m128i*)&worker.prev_chunk[i * 16]);
//             _mm_storeu_si128((__m128i*)&worker.chunk[i * 16], prev_data);
//         }

//         // Copy the remaining bytes before worker.pos1
//         for (int i = end_block * 16; i < worker.pos1; i++) {
//             worker.chunk[i] = worker.prev_chunk[i];
//         }

//         // Calculate the start and end blocks
//         start_block = (worker.pos2 + 15) / 16;
//         end_block = 16;

//         // Copy the blocks after worker.pos2
//         for (int i = start_block; i < end_block; i++) {
//             __m128i prev_data = _mm_loadu_si128((__m128i*)&worker.prev_chunk[i * 16]);
//             _mm_storeu_si128((__m128i*)&worker.chunk[i * 16], prev_data);
//         }

//         // Copy the remaining bytes after worker.pos2
//         for (int i = worker.pos2; i < start_block * 16; i++) {
//           worker.chunk[i] = worker.prev_chunk[i];
//         }
//       }
//     }

//     __builtin_prefetch(&worker.chunk[worker.pos1],1,3);

//     // if (debugOpOrder && worker.op == sus_op) {
//     //   printf("SIMD pre op %d:\n", worker.op);
//     //   for (int i = 0; i < 256; i++) {
//     //       printf("%02X ", worker.prev_chunk[i]);
//     //   } 
//     //   printf("\n");
//     // }
//     // fmt::printf("op: %d, ", worker.op);
//     // fmt::printf("worker.pos1: %d, worker.pos2: %d\n", worker.pos1, worker.pos2);

//     switch(worker.op) {
//       case 0:
//         // #pragma GCC unroll 16
//         {
//           // Load 32 bytes of worker.prev_chunk starting from i into an AVX2 256-bit register
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           __m256i pop = popcnt256_epi8(data);
          
//           data = _mm256_xor_si256(data,pop);

//           // Rotate left by 5
//           data = _mm256_rol_epi8(data, 5);

//           // Full 16-bit multiplication
//           data = _mm256_mul_epi8(data, data);
//           data = _mm256_rolv_epi8(data, data);

//           // Write results to workerData

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         if ((worker.pos2-worker.pos1)%2 == 1) {
//           worker.t1 = worker.chunk[worker.pos1];
//           worker.t2 = worker.chunk[worker.pos2];
//           worker.chunk[worker.pos1] = reverse8(worker.t2);
//           worker.chunk[worker.pos2] = reverse8(worker.t1);
//         }
//         break;
//       case 1:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           __m256i shift = _mm256_and_si256(data, vec_3);
//           data = _mm256_sllv_epi8(data, shift);
//           data = _mm256_rol_epi8(data,1);
//           data = _mm256_and_si256(data, _mm256_set1_epi8(worker.prev_chunk[worker.pos2]));
//           data = _mm256_add_epi8(data, data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 2:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           __m256i pop = popcnt256_epi8(data);
//           data = _mm256_xor_si256(data,pop);
//           data = _mm256_reverse_epi8(data);

//           __m256i shift = _mm256_and_si256(data, vec_3);
//           data = _mm256_sllv_epi8(data, shift);

//           pop = popcnt256_epi8(data);
//           data = _mm256_xor_si256(data,pop);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 3:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_rolv_epi8(data,_mm256_add_epi8(data,vec_3));
//           data = _mm256_xor_si256(data,_mm256_set1_epi8(worker.chunk[worker.pos2]));
//           data = _mm256_rol_epi8(data,1);
          
//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 4:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
//           data = _mm256_srlv_epi8(data,_mm256_and_si256(data,vec_3));
//           data = _mm256_rolv_epi8(data,data);
//           data = _mm256_sub_epi8(data,_mm256_xor_si256(data,_mm256_set1_epi8(97)));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 5:
//         {
//           // Load 32 bytes of worker.prev_chunk starting from i into an AVX2 256-bit register
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           __m256i pop = popcnt256_epi8(data);
//           data = _mm256_xor_si256(data,pop);
//           data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//           data = _mm256_sllv_epi8(data,_mm256_and_si256(data,vec_3));
//           data = _mm256_srlv_epi8(data,_mm256_and_si256(data,vec_3));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
        
//         break;
//       case 6:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_sllv_epi8(data,_mm256_and_si256(data,vec_3));
//           data = _mm256_rol_epi8(data, 3);
//           data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

//           __m256i x = _mm256_xor_si256(data,_mm256_set1_epi8(97));
//           data = _mm256_sub_epi8(data,x);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 7:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_add_epi8(data, data);;
//           data = _mm256_rolv_epi8(data, data);

//           __m256i pop = popcnt256_epi8(data);
//           data = _mm256_xor_si256(data,pop);
//           data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 8:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
//           data = _mm256_rol_epi8(data,2);
//           data = _mm256_sllv_epi8(data,_mm256_and_si256(data,vec_3));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 9:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//           data = _mm256_xor_si256(data, _mm256_rol_epi8(data,4));
//           data = _mm256_srlv_epi8(data, _mm256_and_si256(data,vec_3));
//           data = _mm256_xor_si256(data, _mm256_rol_epi8(data,2));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 10:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
//           data = _mm256_mul_epi8(data, data);
//           data = _mm256_rol_epi8(data, 3);
//           data = _mm256_mul_epi8(data, data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 11:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_rol_epi8(data, 6);
//           data = _mm256_and_si256(data,_mm256_set1_epi8(worker.chunk[worker.pos2]));
//           data = _mm256_rolv_epi8(data, data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 12:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_xor_si256(data, _mm256_rol_epi8(data,2));
//           data = _mm256_mul_epi8(data, data);
//           data = _mm256_xor_si256(data, _mm256_rol_epi8(data,2));
//           data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 13:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_rol_epi8(data, 1);
//           data = _mm256_xor_si256(data,_mm256_set1_epi8(worker.chunk[worker.pos2]));
//           data = _mm256_srlv_epi8(data,_mm256_and_si256(data,vec_3));
//           data = _mm256_rol_epi8(data, 5);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 14:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_srlv_epi8(data,_mm256_and_si256(data,vec_3));
//           data = _mm256_sllv_epi8(data,_mm256_and_si256(data,vec_3));
//           data = _mm256_mul_epi8(data, data);
//           data = _mm256_sllv_epi8(data,_mm256_and_si256(data,vec_3));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 15:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_xor_si256(data, _mm256_rol_epi8(data,2));
//           data = _mm256_sllv_epi8(data,_mm256_and_si256(data,vec_3));
//           data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//           data = _mm256_sub_epi8(data,_mm256_xor_si256(data,_mm256_set1_epi8(97)));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 16:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_xor_si256(data, _mm256_rol_epi8(data,4));
//           data = _mm256_mul_epi8(data, data);
//           data = _mm256_rol_epi8(data,1);
//           data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 17:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//           data = _mm256_mul_epi8(data, data);
//           data = _mm256_rol_epi8(data,5);
//           data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 18:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
//           data = _mm256_rol_epi8(data, 1);
//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 19:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_sub_epi8(data,_mm256_xor_si256(data,_mm256_set1_epi8(97)));
//           data = _mm256_rol_epi8(data, 5);
//           data = _mm256_sllv_epi8(data,_mm256_and_si256(data,vec_3));
//           data = _mm256_add_epi8(data, data);;;

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 20:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//           data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//           data = _mm256_reverse_epi8(data);
//           data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 21:

//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_rol_epi8(data, 1);
//           data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//           data = _mm256_add_epi8(data, data);
//           data = _mm256_and_si256(data,_mm256_set1_epi8(worker.chunk[worker.pos2]));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//     break;
//       case 22:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_sllv_epi8(data, _mm256_and_si256(data,vec_3));
//           data = _mm256_reverse_epi8(data);
//           data = _mm256_mul_epi8(data,data);
//           data = _mm256_rol_epi8(data,1);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 23:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_rol_epi8(data, 4);
//           data = _mm256_xor_si256(data,popcnt256_epi8(data));
//           data = _mm256_and_si256(data,_mm256_set1_epi8(worker.chunk[worker.pos2]));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//       break;
//       case 24:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_add_epi8(data, data);
//           data = _mm256_srlv_epi8(data, _mm256_and_si256(data,vec_3));
//           data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
//           data = _mm256_rol_epi8(data, 5);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 25:
//   #pragma GCC unroll 32
//         for (int i = worker.pos1; i < worker.pos2; i++)
//         {
//           // INSERT_RANDOM_CODE_START
//           worker.chunk[i] = worker.prev_chunk[i] ^ (byte)bitTable[worker.prev_chunk[i]];             // ones count bits
//           worker.chunk[i] = rl8(worker.chunk[i], 3);                // rotate  bits by 3
//           worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
//           worker.chunk[i] -= (worker.chunk[i] ^ 97);                      // XOR and -
//                                                                             // INSERT_RANDOM_CODE_END
//         }
//         break;
//       case 26:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_mul_epi8(data, data);
//           data = _mm256_xor_si256(data,popcnt256_epi8(data));
//           data = _mm256_add_epi8(data, data);
//           data = _mm256_reverse_epi8(data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 27:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_rol_epi8(data, 5);
//           data = _mm256_and_si256(data,_mm256_set1_epi8(worker.chunk[worker.pos2]));
//           data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
//           data = _mm256_rol_epi8(data, 5);
          
//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 28:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_sllv_epi8(data, _mm256_and_si256(data,vec_3));
//           data = _mm256_add_epi8(data, data);
//           data = _mm256_add_epi8(data, data);
//           data = _mm256_rol_epi8(data, 5);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 29:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_mul_epi8(data, data);
//           data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//           data = _mm256_srlv_epi8(data, _mm256_and_si256(data,vec_3));
//           data = _mm256_add_epi8(data, data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 30:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//           data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
//           data = _mm256_rol_epi8(data, 5);
//           data = _mm256_sllv_epi8(data, _mm256_and_si256(data,vec_3));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 31:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;
        
//           data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
//           data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//           data = _mm256_sllv_epi8(data, _mm256_and_si256(data,vec_3));
//           data = _mm256_mul_epi8(data, data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 32:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//           data = _mm256_reverse_epi8(data);
//           data = _mm256_rol_epi8(data, 3);
//           data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 33:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_rolv_epi8(data, data);
//           data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
//           data = _mm256_reverse_epi8(data);
//           data = _mm256_mul_epi8(data, data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 34:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
//           data = _mm256_sllv_epi8(data, _mm256_and_si256(data,vec_3));
//           data = _mm256_sllv_epi8(data, _mm256_and_si256(data,vec_3));
//           data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 35:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_add_epi8(data, data);
//           data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
//           data = _mm256_rol_epi8(data, 1);
//           data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 36:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_xor_si256(data, popcnt256_epi8(data));
//           data = _mm256_rol_epi8(data, 1);
//           data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//           data = _mm256_rol_epi8(data, 1);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 37:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_rolv_epi8(data, data);
//           data = _mm256_srlv_epi8(data, _mm256_and_si256(data,vec_3));
//           data = _mm256_srlv_epi8(data, _mm256_and_si256(data,vec_3));
//           data = _mm256_mul_epi8(data, data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 38:
//   #pragma GCC unroll 32
//         for (int i = worker.pos1; i < worker.pos2; i++)
//         {
//           // INSERT_RANDOM_CODE_START
//           worker.chunk[i] = worker.prev_chunk[i] >> (worker.prev_chunk[i] & 3);    // shift right
//           worker.chunk[i] = rl8(worker.chunk[i], 3);                // rotate  bits by 3
//           worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
//           worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
//                                                                             // INSERT_RANDOM_CODE_END
//         }
//         break;
//       case 39:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//           data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//           data = _mm256_srlv_epi8(data, _mm256_and_si256(data,vec_3));
//           data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 40:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_rolv_epi8(data, data);
//           data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//           data = _mm256_xor_si256(data, popcnt256_epi8(data));
//           data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 41:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_rol_epi8(data, 5);
//           data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
//           data = _mm256_rol_epi8(data, 3);
//           data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 42:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_rol_epi8(data, 4);
//           data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//           data = _mm256_rolv_epi8(data, data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 43:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//           data = _mm256_add_epi8(data, data);
//           data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//           data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 44:
//   #pragma GCC unroll 32
//         for (int i = worker.pos1; i < worker.pos2; i++)
//         {
//           // INSERT_RANDOM_CODE_START
//           worker.chunk[i] = worker.prev_chunk[i] ^ (byte)bitTable[worker.prev_chunk[i]];             // ones count bits
//           worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
//           worker.chunk[i] = rl8(worker.chunk[i], 3);                // rotate  bits by 3
//           worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
//                                                                             // INSERT_RANDOM_CODE_END
//         }
//         break;
//       case 45:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;
        
//           data = _mm256_rol_epi8(data, 2);
//           data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//           data = _mm256_xor_si256(data, popcnt256_epi8(data));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 46:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;
        
//           data = _mm256_xor_si256(data, popcnt256_epi8(data));
//           data = _mm256_add_epi8(data, data);
//           data = _mm256_rol_epi8(data, 5);
//           data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 47:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_rol_epi8(data, 5);
//           data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//           data = _mm256_rol_epi8(data, 5);
//           data = _mm256_sllv_epi8(data, _mm256_and_si256(data,vec_3));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 48:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_rolv_epi8(data, data);
//           data = _mm256_rol_epi8(data, 5);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 49:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;
        
//           data = _mm256_xor_si256(data, popcnt256_epi8(data));
//           data = _mm256_add_epi8(data, data);
//           data = _mm256_reverse_epi8(data);
//           data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 50:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;
        
//           data = _mm256_reverse_epi8(data);
//           data = _mm256_rol_epi8(data, 3);
//           data = _mm256_add_epi8(data, data);
//           data = _mm256_rol_epi8(data, 1);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 51:
//   #pragma GCC unroll 32
//         for (int i = worker.pos1; i < worker.pos2; i++)
//         {
//           // INSERT_RANDOM_CODE_START
//           worker.chunk[i] = worker.prev_chunk[i] ^ worker.chunk[worker.pos2];     // XOR
//           worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
//           worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
//           worker.chunk[i] = rl8(worker.chunk[i], 5);  // rotate  bits by 5
//                                                               // INSERT_RANDOM_CODE_END
//         }
//         break;
//       case 52:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_rolv_epi8(data, data);
//           data = _mm256_srlv_epi8(data, _mm256_and_si256(data,vec_3));
//           data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
//           data = _mm256_xor_si256(data, popcnt256_epi8(data));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 53:
//   #pragma GCC unroll 32
//         for (int i = worker.pos1; i < worker.pos2; i++)
//         {
//           // INSERT_RANDOM_CODE_START
//           worker.chunk[i] = worker.prev_chunk[i]*2;                 // +
//           worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
//           worker.chunk[i] ^= rl8(worker.chunk[i], 4);   // rotate  bits by 4
//           worker.chunk[i] ^= rl8(worker.chunk[i], 4);   // rotate  bits by 4
//                                                                 // INSERT_RANDOM_CODE_END
//         }
//         break;
//       case 54:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;
        
//           data = _mm256_reverse_epi8(data);
//           data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }

//         break;
//       case 55:
//   #pragma GCC unroll 32
//         for (int i = worker.pos1; i < worker.pos2; i++)
//         {
//           // INSERT_RANDOM_CODE_START
//           worker.chunk[i] = reverse8(worker.prev_chunk[i]);      // reverse bits
//           worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
//           worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
//           worker.chunk[i] = rl8(worker.chunk[i], 1);  // rotate  bits by 1
//                                                               // INSERT_RANDOM_CODE_END
//         }
//         break;
//       case 56:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//           data = _mm256_mul_epi8(data, data);
//           data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
//           data = _mm256_rol_epi8(data, 1);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 57:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_rolv_epi8(data, data);
//           data = _mm256_reverse_epi8(data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 58:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;
        
//           data = _mm256_reverse_epi8(data);
//           data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//           data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//           data = _mm256_add_epi8(data, data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }  
//         break;
//       case 59:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_rol_epi8(data, 1);
//           data = _mm256_mul_epi8(data, data);
//           data = _mm256_rolv_epi8(data, data);
//           data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 60:
//         {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
//             data = _mm256_mul_epi8(data, data);
//             data = _mm256_rol_epi8(data, 3);

//             #ifdef _WIN32
//               data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             #else
//               data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             #endif
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }

//         break;
//       case 61:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_rol_epi8(data, 5);
//           data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 62:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//           data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
//           data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//           data = _mm256_add_epi8(data, data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 63:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_rol_epi8(data, 5);
//           data = _mm256_xor_si256(data, popcnt256_epi8(data));
//           data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
//           data = _mm256_add_epi8(data, data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }

//         break;
//       case 64:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//           data = _mm256_reverse_epi8(data);
//           data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
//           data = _mm256_mul_epi8(data, data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 65:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;


//           data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//           data = _mm256_mul_epi8(data, data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 66:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//           data = _mm256_reverse_epi8(data);
//           data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
//           data = _mm256_rol_epi8(data, 1);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 67:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_rol_epi8(data, 1);
//           data = _mm256_xor_si256(data, popcnt256_epi8(data));
//           data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//           data = _mm256_rol_epi8(data, 5);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 68:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//           data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
//           data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
//           data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 69:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_add_epi8(data, data);
//           data = _mm256_mul_epi8(data, data);
//           data = _mm256_reverse_epi8(data);
//           data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 70:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//           data = _mm256_mul_epi8(data, data);
//           data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
//           data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 71:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_rol_epi8(data, 5);
//           data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
//           data = _mm256_mul_epi8(data, data);
//           data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 72:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_reverse_epi8(data);
//           data = _mm256_xor_si256(data, popcnt256_epi8(data));
//           data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//           data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 73:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_xor_si256(data, popcnt256_epi8(data));
//           data = _mm256_reverse_epi8(data);
//           data = _mm256_rol_epi8(data, 5);
//           data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 74:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_mul_epi8(data, data);
//           data = _mm256_rol_epi8(data, 3);
//           data = _mm256_reverse_epi8(data);
//           data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//       case 75:
//         {
//           __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//           __m256i old = data;

//           data = _mm256_mul_epi8(data, data);
//           data = _mm256_xor_si256(data, popcnt256_epi8(data));
//           data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//           data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//           _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//         }
//         break;
//         case 76:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rolv_epi8(data, data);
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//             data = _mm256_rol_epi8(data, 5);
//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 77:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rol_epi8(data, 3);
//             data = _mm256_add_epi8(data, data);
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_xor_si256(data, popcnt256_epi8(data));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 78:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rolv_epi8(data, data);
//             data = _mm256_reverse_epi8(data);
//             data = _mm256_mul_epi8(data, data);
//             data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 79:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//             data = _mm256_add_epi8(data, data);
//             data = _mm256_mul_epi8(data, data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 80:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rolv_epi8(data, data);
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_add_epi8(data, data);
//             data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 81:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_rolv_epi8(data, data);
//             data = _mm256_xor_si256(data, popcnt256_epi8(data));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 82:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 83:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_reverse_epi8(data);
//             data = _mm256_rol_epi8(data, 3);
//             data = _mm256_reverse_epi8(data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 84:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
//             data = _mm256_rol_epi8(data, 1);
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_add_epi8(data, data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 85:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_rolv_epi8(data, data);
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 86:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
//             data = _mm256_rolv_epi8(data, data);
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 87:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_add_epi8(data, data);
//             data = _mm256_rol_epi8(data, 3);
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
//             data = _mm256_add_epi8(data, data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 88:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//             data = _mm256_rol_epi8(data, 1);
//             data = _mm256_mul_epi8(data, data);
//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 89:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_add_epi8(data, data);
//             data = _mm256_mul_epi8(data, data);
//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 90:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_reverse_epi8(data);
//             data = _mm256_rol_epi8(data, 6);
//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 91:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, popcnt256_epi8(data));
//             data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
//             data = _mm256_reverse_epi8(data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 92:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, popcnt256_epi8(data));
//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
//             data = _mm256_xor_si256(data, popcnt256_epi8(data));
//             data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 93:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//             data = _mm256_mul_epi8(data, data);
//             data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_add_epi8(data, data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 94:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rol_epi8(data, 1);
//             data = _mm256_rolv_epi8(data, data);
//             data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 95:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rol_epi8(data, 1);
//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
//             data = _mm256_rol_epi8(data, 2);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 96:
//     #pragma GCC unroll 32
//           for (int i = worker.pos1; i < worker.pos2; i++)
//           {
//             // INSERT_RANDOM_CODE_START
//             worker.chunk[i] = worker.prev_chunk[i] ^ rl8(worker.prev_chunk[i], 2);   // rotate  bits by 2
//             worker.chunk[i] ^= rl8(worker.chunk[i], 2);   // rotate  bits by 2
//             worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
//             worker.chunk[i] = rl8(worker.chunk[i], 1);    // rotate  bits by 1
//                                                                   // INSERT_RANDOM_CODE_END
//           }
//           break;
//         case 97:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rol_epi8(data, 1);
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_xor_si256(data, popcnt256_epi8(data));
//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 98:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 99:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
//             data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
//             data = _mm256_reverse_epi8(data);
//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 100:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rolv_epi8(data, data);
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_reverse_epi8(data);
//             data = _mm256_xor_si256(data, popcnt256_epi8(data));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 101:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_xor_si256(data, popcnt256_epi8(data));
//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 102:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rol_epi8(data, 3);
//             data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
//             data = _mm256_add_epi8(data, data);
//             data = _mm256_rol_epi8(data, 3);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 103:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rol_epi8(data, 1);
//             data = _mm256_reverse_epi8(data);
//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_rolv_epi8(data, data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 104:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_reverse_epi8(data);
//             data = _mm256_xor_si256(data, popcnt256_epi8(data));
//             data = _mm256_rol_epi8(data, 5);
//             data = _mm256_add_epi8(data, data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 105:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_rol_epi8(data, 3);
//             data = _mm256_rolv_epi8(data, data);
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 106:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_reverse_epi8(data);
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
//             data = _mm256_rol_epi8(data, 1);
//             data = _mm256_mul_epi8(data, data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 107:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//             data = _mm256_rol_epi8(data, 6);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 108:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
//             data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 109:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_mul_epi8(data, data);
//             data = _mm256_rolv_epi8(data, data);
//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 110:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_add_epi8(data, data);
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 111:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_mul_epi8(data, data);
//             data = _mm256_reverse_epi8(data);
//             data = _mm256_mul_epi8(data, data);
//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 112:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rol_epi8(data, 3);
//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
//             data = _mm256_rol_epi8(data, 5);
//             data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 113:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rol_epi8(data, 6);
//             data = _mm256_xor_si256(data, popcnt256_epi8(data));
//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 114:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rol_epi8(data, 1);
//             data = _mm256_reverse_epi8(data);
//             data = _mm256_rolv_epi8(data, data);
//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 115:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rolv_epi8(data, data);
//             data = _mm256_rol_epi8(data, 5);
//             data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_rol_epi8(data, 3);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 116:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_xor_si256(data, popcnt256_epi8(data));
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 117:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_rol_epi8(data, 3);
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 118:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_add_epi8(data, data);
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_rol_epi8(data, 5);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 119:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_reverse_epi8(data);
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 120:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//             data = _mm256_mul_epi8(data, data);
//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_reverse_epi8(data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 121:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_add_epi8(data, data);
//             __m256i pop = popcnt256_epi8(data);
//             data = _mm256_xor_si256(data, pop);
//             data = _mm256_mul_epi8(data, data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 122:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
//             data = _mm256_rolv_epi8(data, data);
//             data = _mm256_rol_epi8(data, 5);
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 123:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
//             data = _mm256_rol_epi8(data, 6);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 124:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 125:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_reverse_epi8(data);
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//             data = _mm256_add_epi8(data, data);
//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 126:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rol_epi8(data, 1);
//             data = _mm256_reverse_epi8(data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 127:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_mul_epi8(data, data);
//             data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 128:
//     #pragma GCC unroll 32
//           for (int i = worker.pos1; i < worker.pos2; i++)
//           {
//             // INSERT_RANDOM_CODE_START
//             worker.chunk[i] = rl8(worker.prev_chunk[i], worker.prev_chunk[i]); // rotate  bits by random
//             worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
//             worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
//             worker.chunk[i] = rl8(worker.chunk[i], 5);                // rotate  bits by 5
//                                                                               // INSERT_RANDOM_CODE_END
//           }
//           break;
//         case 129:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
//             __m256i pop = popcnt256_epi8(data);
//             data = _mm256_xor_si256(data, pop);
//             pop = popcnt256_epi8(data);
//             data = _mm256_xor_si256(data, pop);
//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 130:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_rolv_epi8(data, data);
//             data = _mm256_rol_epi8(data, 1);
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 131:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
//             data = _mm256_rol_epi8(data, 1);
//             __m256i pop = popcnt256_epi8(data);
//             data = _mm256_xor_si256(data, pop);
//             data = _mm256_mul_epi8(data, data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 132:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_reverse_epi8(data);
//             data = _mm256_rol_epi8(data, 5);
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 133:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_rol_epi8(data, 5);
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 134:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
//             data = _mm256_rol_epi8(data, 1);
//             data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 135:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//             data = _mm256_add_epi8(data, data);
//             data = _mm256_reverse_epi8(data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 136:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_rol_epi8(data, 5);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 137:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rol_epi8(data, 5);
//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_reverse_epi8(data);
//             data = _mm256_rolv_epi8(data, data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 138:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_add_epi8(data, data);
//             data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 139:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//             data = _mm256_rol_epi8(data, 3);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 140:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rol_epi8(data, 1);
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_rol_epi8(data, 5);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 141:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rol_epi8(data, 1);
//             data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
//             __m256i pop = popcnt256_epi8(data);
//             data = _mm256_xor_si256(data, pop);
//             data = _mm256_add_epi8(data, data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 142:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_rol_epi8(data, 5);
//             data = _mm256_reverse_epi8(data);
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 143:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_rol_epi8(data, 3);
//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 144:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rolv_epi8(data, data);
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
//             data = _mm256_rolv_epi8(data, data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 145:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_reverse_epi8(data);
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 146:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             __m256i pop = popcnt256_epi8(data);
//             data = _mm256_xor_si256(data, pop);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 147:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
//             data = _mm256_mul_epi8(data, data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 148:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_rol_epi8(data, 5);
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 149:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_reverse_epi8(data);
//             data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
//             data = _mm256_add_epi8(data, data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 150:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 151:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_add_epi8(data, data);
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_mul_epi8(data, data);
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 152:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 153:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rol_epi8(data, 4);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 154:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rol_epi8(data, 5);
//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             __m256i pop = popcnt256_epi8(data);
//             data = _mm256_xor_si256(data, pop);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 155:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             __m256i pop = popcnt256_epi8(data);
//             data = _mm256_xor_si256(data, pop);
//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 156:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_rol_epi8(data, 4);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 157:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_rolv_epi8(data, data);
//             data = _mm256_rol_epi8(data, 1);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 158:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             __m256i pop = popcnt256_epi8(data);
//             data = _mm256_xor_si256(data, pop);
//             data = _mm256_rol_epi8(data, 3);
//             data = _mm256_add_epi8(data, data);
//             data = _mm256_rol_epi8(data, 1);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 159:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_rolv_epi8(data, data);
//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 160:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_reverse_epi8(data);
//             data = _mm256_rol_epi8(data, 4);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 161:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_rol_epi8(data, 5);
//             data = _mm256_rolv_epi8(data, data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 162:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_mul_epi8(data, data);
//             data = _mm256_reverse_epi8(data);
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//             data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 163:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
//             data = _mm256_rol_epi8(data, 1);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 164:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_mul_epi8(data, data);
//             __m256i pop = popcnt256_epi8(data);
//             data = _mm256_xor_si256(data, pop);
//             data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 165:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_add_epi8(data, data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 166:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rol_epi8(data, 3);
//             data = _mm256_add_epi8(data, data);
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 167:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_mul_epi8(data, data);
//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 168:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rolv_epi8(data, data);
//             data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_rolv_epi8(data, data);
//             data = _mm256_rol_epi8(data, 1);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 169:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rol_epi8(data, 1);
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
//             data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 170:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
//             data = _mm256_reverse_epi8(data);
//             data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
//             data = _mm256_mul_epi8(data, data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 171:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rol_epi8(data, 3);
//             data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
//             __m256i pop = popcnt256_epi8(data);
//             data = _mm256_xor_si256(data, pop);
//             data = _mm256_reverse_epi8(data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 172:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
//             data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_rol_epi8(data, 1);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 173:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_mul_epi8(data, data);
//             data = _mm256_add_epi8(data, data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 174:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
//             data = _mm256_rolv_epi8(data, data);
//             __m256i pop = popcnt256_epi8(data);
//             data = _mm256_xor_si256(data, pop);
//             pop = popcnt256_epi8(data);
//             data = _mm256_xor_si256(data, pop);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 175:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rol_epi8(data, 3);
//             data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
//             data = _mm256_mul_epi8(data, data);
//             data = _mm256_rol_epi8(data, 5);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 176:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_mul_epi8(data, data);
//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_rol_epi8(data, 5);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 177:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             __m256i pop = popcnt256_epi8(data);
//             data = _mm256_xor_si256(data, pop);
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//             data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 178:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_add_epi8(data, data);
//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
//             data = _mm256_rol_epi8(data, 1);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 179:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//             data = _mm256_add_epi8(data, data);
//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_reverse_epi8(data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 180:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 181:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//             data = _mm256_rol_epi8(data, 5);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 182:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_rol_epi8(data, 6);
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 183:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_add_epi8(data, data);
//             data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
//             data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
//             data = _mm256_mul_epi8(data, data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 184:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_mul_epi8(data, data);
//             data = _mm256_rol_epi8(data, 5);
//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 185:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
//             data = _mm256_rol_epi8(data, 5);
//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 186:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
//             data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 187:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
//             data = _mm256_add_epi8(data, data);
//             data = _mm256_rol_epi8(data, 3);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 188:
//     #pragma GCC unroll 32
//           for (int i = worker.pos1; i < worker.pos2; i++)
//           {
//             // INSERT_RANDOM_CODE_START
//             worker.chunk[i] ^= rl8(worker.prev_chunk[i], 4);   // rotate  bits by 4
//             worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
//             worker.chunk[i] ^= rl8(worker.chunk[i], 4);   // rotate  bits by 4
//             worker.chunk[i] ^= rl8(worker.chunk[i], 4);   // rotate  bits by 4
//                                                                   // INSERT_RANDOM_CODE_END
//           }
//           break;
//         case 189:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rol_epi8(data, 5);
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 190:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rol_epi8(data, 5);
//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 191:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_add_epi8(data, data);
//             data = _mm256_rol_epi8(data, 3);
//             data = _mm256_rolv_epi8(data, data);
//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 192:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_add_epi8(data, data);
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_add_epi8(data, data);
//             data = _mm256_mul_epi8(data, data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 193:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_rolv_epi8(data, data);
//             data = _mm256_rol_epi8(data, 1);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 194:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_rolv_epi8(data, data);
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 195:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             __m256i pop = popcnt256_epi8(data);
//             data = _mm256_xor_si256(data, pop);
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 196:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rol_epi8(data, 3);
//             data = _mm256_reverse_epi8(data);
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_rol_epi8(data, 1);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 197:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
//             data = _mm256_rolv_epi8(data, data);
//             data = _mm256_mul_epi8(data, data);
//             data = _mm256_mul_epi8(data, data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 198:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_reverse_epi8(data);
//             data = _mm256_rol_epi8(data, 1);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 199:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
//             data = _mm256_add_epi8(data, data);
//             data = _mm256_mul_epi8(data, data);
//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 200:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
//             __m256i pop = popcnt256_epi8(data);
//             data = _mm256_xor_si256(data, pop);
//             data = _mm256_reverse_epi8(data);
//             data = _mm256_reverse_epi8(data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 201:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rol_epi8(data, 3);
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 202:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
//             data = _mm256_rolv_epi8(data, data);
//             data = _mm256_rol_epi8(data, 5);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 203:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_rol_epi8(data, 1);
//             data = _mm256_rolv_epi8(data, data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 204:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rol_epi8(data, 5);
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//             data = _mm256_rolv_epi8(data, data);
//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 205:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             __m256i pop = popcnt256_epi8(data);
//             data = _mm256_xor_si256(data, pop);
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_add_epi8(data, data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 206:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
//             data = _mm256_reverse_epi8(data);
//             data = _mm256_reverse_epi8(data);
//             __m256i pop = popcnt256_epi8(data);
//             data = _mm256_xor_si256(data, pop);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 207:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             __m256i pop = popcnt256_epi8(data);
//             data = _mm256_xor_si256(data, pop);
//             pop = popcnt256_epi8(data);
//             data = _mm256_xor_si256(data, pop);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 208:
//     #pragma GCC unroll 32
//           for (int i = worker.pos1; i < worker.pos2; i++)
//           {
//             // INSERT_RANDOM_CODE_START
//             worker.chunk[i] = worker.prev_chunk[i]*2;                          // +
//             worker.chunk[i] += worker.chunk[i];                          // +
//             worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
//             worker.chunk[i] = rl8(worker.chunk[i], 3);             // rotate  bits by 3
//                                                                           // INSERT_RANDOM_CODE_END
//           }
//           break;
//         case 209:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rol_epi8(data, 5);
//             data = _mm256_reverse_epi8(data);
//             __m256i pop = popcnt256_epi8(data);
//             data = _mm256_xor_si256(data, pop);
//             data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 210:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//             data = _mm256_rolv_epi8(data, data);
//             data = _mm256_rol_epi8(data, 5);
//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 211:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
//             data = _mm256_add_epi8(data, data);
//             data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
//             data = _mm256_rolv_epi8(data, data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 212:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rolv_epi8(data, data);
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//             // data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             // data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 213:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_add_epi8(data, data);
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_rol_epi8(data, 3);
//             data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 214:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 215:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_mul_epi8(data, data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 216:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rolv_epi8(data, data);
//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
//             data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
//             data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 217:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rol_epi8(data, 5);
//             data = _mm256_add_epi8(data, data);
//             data = _mm256_rol_epi8(data, 1);
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 218:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_reverse_epi8(data);
//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
//             data = _mm256_mul_epi8(data, data);
//             data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 219:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
//             data = _mm256_rol_epi8(data, 3);
//             data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_reverse_epi8(data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 220:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rol_epi8(data, 1);
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_reverse_epi8(data);
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 221:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rol_epi8(data, 5);
//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
//             data = _mm256_reverse_epi8(data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 222:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_mul_epi8(data, data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 223:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rol_epi8(data, 3);
//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_rolv_epi8(data, data);
//             data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 224:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//             data = _mm256_rol_epi8(data, 4);
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 225:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_reverse_epi8(data);
//             data = _mm256_rol_epi8(data, 3);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 226:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_reverse_epi8(data);
//             data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
//             data = _mm256_mul_epi8(data, data);
//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 227:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
//             data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 228:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_add_epi8(data, data);
//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_add_epi8(data, data);
//             __m256i pop = popcnt256_epi8(data);
//             data = _mm256_xor_si256(data, pop);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 229:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rol_epi8(data, 3);
//             data = _mm256_rolv_epi8(data, data);
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//             __m256i pop = popcnt256_epi8(data);
//             data = _mm256_xor_si256(data, pop);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 230:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_mul_epi8(data, data);
//             data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_rolv_epi8(data, data);
//             data = _mm256_rolv_epi8(data, data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 231:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rol_epi8(data, 3);
//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_reverse_epi8(data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 232:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_mul_epi8(data, data);
//             data = _mm256_mul_epi8(data, data);
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
//             data = _mm256_rol_epi8(data, 5);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 233:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rol_epi8(data, 1);
//             __m256i pop = popcnt256_epi8(data);
//             data = _mm256_xor_si256(data, pop);
//             data = _mm256_rol_epi8(data, 3);
//             pop = popcnt256_epi8(data);
//             data = _mm256_xor_si256(data, pop);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 234:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_mul_epi8(data, data);
//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 235:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//             data = _mm256_mul_epi8(data, data);
//             data = _mm256_rol_epi8(data, 3);
//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 236:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_add_epi8(data, data);
//             data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 237:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rol_epi8(data, 5);
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//             data = _mm256_rol_epi8(data, 3);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 238:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_add_epi8(data, data);
//             data = _mm256_add_epi8(data, data);
//             data = _mm256_rol_epi8(data, 3);
//             data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 239:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rol_epi8(data, 6);
//             data = _mm256_mul_epi8(data, data);
//             data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 240:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
//             data = _mm256_add_epi8(data, data);
//             data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 241:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
//             __m256i pop = popcnt256_epi8(data);
//             data = _mm256_xor_si256(data, pop);
//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_rol_epi8(data, 1);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 242:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_add_epi8(data, data);
//             data = _mm256_add_epi8(data, data);
//             data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
//             data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 243:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rol_epi8(data, 5);
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//             __m256i pop = popcnt256_epi8(data);
//             data = _mm256_xor_si256(data, pop);
//             data = _mm256_rol_epi8(data, 1);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 244:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//             data = _mm256_reverse_epi8(data);
//             data = _mm256_rol_epi8(data, 5);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 245:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
//             data = _mm256_rol_epi8(data, 5);
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 246:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_add_epi8(data, data);
//             data = _mm256_rol_epi8(data, 1);
//             data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
//             data = _mm256_add_epi8(data, data);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 247:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_rol_epi8(data, 5);
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//             data = _mm256_rol_epi8(data, 5);
//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 248:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
//             data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
//             __m256i pop = popcnt256_epi8(data);
//             data = _mm256_xor_si256(data, pop);
//             data = _mm256_rol_epi8(data, 5);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 249:
//     #pragma GCC unroll 32
//           for (int i = worker.pos1; i < worker.pos2; i++)
//           {
//             // INSERT_RANDOM_CODE_START
//             worker.chunk[i] = reverse8(worker.prev_chunk[i]);                    // reverse bits
//             worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
//             worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
//             worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
//                                                                               // INSERT_RANDOM_CODE_END
//           }
//           break;
//         case 250:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
//             data = _mm256_rolv_epi8(data, data);
//             __m256i pop = popcnt256_epi8(data);
//             data = _mm256_xor_si256(data, pop);
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 251:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_add_epi8(data, data);
//             __m256i pop = popcnt256_epi8(data);
//             data = _mm256_xor_si256(data, pop);
//             data = _mm256_reverse_epi8(data);
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 252:
//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             data = _mm256_reverse_epi8(data);
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//             data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         case 253:
//           {
//             std::copy(&worker.prev_chunk[worker.pos1], &worker.prev_chunk[worker.pos2], &worker.chunk[worker.pos1]);
//     #pragma GCC unroll 32
//             for (int i = worker.pos1; i < worker.pos2; i++)
//             {
//               // INSERT_RANDOM_CODE_START
//               worker.chunk[i] = rl8(worker.chunk[i], 3);  // rotate  bits by 3
//               worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
//               worker.chunk[i] ^= worker.chunk[worker.pos2];     // XOR
//               worker.chunk[i] = rl8(worker.chunk[i], 3);  // rotate  bits by 3
//               // INSERT_RANDOM_CODE_END

//               worker.prev_lhash = worker.lhash + worker.prev_lhash;
//               worker.lhash = XXHash64::hash(worker.chunk, worker.pos2,0);
//             }
//             break;
//           }
//         case 254:
//         case 255:
//           RC4_set_key(&worker.key[wIndex], 256, worker.prev_chunk);

//           {
//             __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//             __m256i old = data;

//             __m256i pop = popcnt256_epi8(data);
//             data = _mm256_xor_si256(data, pop);
//             data = _mm256_rol_epi8(data, 3);
//             data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
//             data = _mm256_rol_epi8(data, 3);

//           data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
//             _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
//           }
//           break;
//         default:
//           break;
//       }

//     if(isTest) {
//       break;
//     }
  
//     // if (op == 53) {
//     //   std::cout << hexStr(worker.chunk, 256) << std::endl << std::endl;
//     //   std::cout << hexStr(&worker.chunk[worker.pos1], 1) << std::endl;
//     //   std::cout << hexStr(&worker.chunk[worker.pos2], 1) << std::endl;
//     // }

//     __builtin_prefetch(worker.chunk,0,3);
//     // __builtin_prefetch(worker.chunk+64,0,3);
//     // __builtin_prefetch(worker.chunk+128,0,3);
//     __builtin_prefetch(worker.chunk+192,0,3);


//     uint8_t pushPos1 = lp1;
//     uint8_t pushPos2 = lp2;

//     if (worker.pos1 == worker.pos2) {
//       pushPos1 = -1;
//       pushPos2 = -1;
//     }

//     worker.A = (worker.chunk[worker.pos1] - worker.chunk[worker.pos2]);
//     worker.A = (256 + (worker.A % 256)) % 256;

//     if (worker.A < 0x10)
//     { // 6.25 % probability
//       worker.prev_lhash = worker.lhash + worker.prev_lhash;
//       worker.lhash = XXHash64::hash(worker.chunk, worker.pos2, 0);

//       // uint64_t test = XXHash64::hash(worker.chunk, worker.pos2, 0);
//       #ifdef DEBUG_OP_ORDER
//       if (worker.op == sus_op && debugOpOrder) printf("SIMD: A: new worker.lhash: %08jx\n", worker.lhash);
//       #endif
//     }

//     if (worker.A < 0x20)
//     { // 12.5 % probability
//       worker.prev_lhash = worker.lhash + worker.prev_lhash;
//       worker.lhash = hash_64_fnv1a(worker.chunk, worker.pos2);

//       // uint64_t test = hash_64_fnv1a(worker.chunk, worker.pos2);
//       #ifdef DEBUG_OP_ORDER
//       if (worker.op == sus_op && debugOpOrder) printf("SIMD: B: new worker.lhash: %08jx\n", worker.lhash);
//       #endif
//     }

//     if (worker.A < 0x30)
//     { // 18.75 % probability
//       worker.prev_lhash = worker.lhash + worker.prev_lhash;
//       HH_ALIGNAS(16)
//       const highwayhash::HH_U64 key2[2] = {worker.tries[wIndex], worker.prev_lhash};
//       worker.lhash = highwayhash::SipHash(key2, (char*)worker.chunk, worker.pos2); // more deviations

//       // uint64_t test = highwayhash::SipHash(key2, (char*)worker.chunk, worker.pos2); // more deviations
//       if (worker.op == sus_op && debugOpOrder) printf("SIMD: C: new worker.lhash: %08jx\n", worker.lhash);
//     }

//     if (worker.A <= 0x40)
//     { // 25% probablility
//       // if (debugOpOrder && worker.op == sus_op) {
//       //   printf("SIMD: D: RC4 key:\n");
//       //   for (int i = 0; i < 256; i++) {
//       //     printf("%d, ", worker.key.data[i]);
//       //   }
//       // }
//       RC4(&worker.key[wIndex], 256, worker.chunk, worker.chunk);
//       if (255 - pushPos2 < MINPREFLEN)
//         pushPos2 = 255;
//       if (pushPos1 < MINPREFLEN)
//         pushPos1 = 0;


//       if (pushPos1 == 255) pushPos1 = 0;
      
//       worker.astroTemplate[worker.templateIdx] = templateMarker{
//         (uint8_t)(chunkCount > 1 ? pushPos1 : 0),
//         (uint8_t)(chunkCount > 1 ? pushPos2 : 255),
//         (uint16_t)0,
//         (uint16_t)0,
//         (uint16_t)((firstChunk << 7) | chunkCount)
//       };

//       pushPos1 = 0;
//       pushPos2 = 255;
//       worker.templateIdx += (worker.tries[wIndex] > 1);
//       firstChunk = worker.tries[wIndex]-1;
//       lp1 = 255;
//       lp2 = 0;
//       chunkCount = 1;
//     } else {
//       chunkCount++;
//     }

//     worker.chunk[255] = worker.chunk[255] ^ worker.chunk[worker.pos1] ^ worker.chunk[worker.pos2];

//     if (255 - pushPos2 < MINPREFLEN)
//       pushPos2 = 255;
//     if (pushPos1 < MINPREFLEN)
//       pushPos1 = 0;

//     // if (debugOpOrder && worker.op == sus_op) {
//     //   printf("SIMD op %d result:\n", worker.op);
//     //   for (int i = 0; i < 256; i++) {
//     //       printf("%02X ", worker.chunk[i]);
//     //   } 
//     //   printf("\n");
//     // }

//     // memcpy(&worker.sData[(worker.tries[wIndex] - 1) * 256], worker.chunk, 256);
    
//     // std::copy(worker.chunk, worker.chunk + 256, &worker.sData[(worker.tries[wIndex] - 1) * 256]);

//     // memcpy(&worker->data.data()[(worker.tries[wIndex] - 1) * 256], worker.chunk, 256);

//     // std::cout << hexStr(worker.chunk, 256) << std::endl;

//     if (worker.tries[wIndex] > 260 + 16 || (worker.sData[(worker.tries[wIndex]-1)*256+255] >= 0xf0 && worker.tries[wIndex] > 260))
//     {
//       break;
//     }
//   }

//   if (chunkCount > 0) {
//     if (255 - lp2 < MINPREFLEN)
//       lp2 = 255;
//     if (lp1 < MINPREFLEN)
//       lp1 = 0;
//     worker.astroTemplate[worker.templateIdx] = templateMarker{
//       (uint8_t)(chunkCount > 1 ? lp1 : 0),
//       (uint8_t)(chunkCount > 1 ? lp2 : 255),
//       (uint16_t)0,
//       (uint16_t)0,
//       (uint16_t)((firstChunk << 7) | chunkCount)
//     };
//     worker.templateIdx++;
//   }

//   worker.data_len = static_cast<uint32_t>((worker.tries[wIndex] - 4) * 256 + (((static_cast<uint64_t>(worker.chunk[253]) << 8) | static_cast<uint64_t>(worker.chunk[254])) & 0x3ff));
// }

void branchComputeCPU_avx2_zOptimized(workerData &worker, bool isTest, int wIndex)
{}
// {
//   byte prevOp;
//   // int sameCount = 0;
//   worker.templateIdx = 0;
//   uint8_t chunkCount = 1;
//   int firstChunk = 0;

//   uint8_t lp1 = 0;
//   uint8_t lp2 = 255;

//   int repeatCounter = 0;
//   byte lastChar = 0;

//   int maxRepeat = 0;
//   int reducedChars = 0;

//   #pragma unroll 64
//   for (int it = 0; it < 278; it++) 
//   {
//   // while(true) {
//     if(isTest) {
//       // __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//       // __m256i cmp = _mm256_cmpeq_epi8(data, _mm256_set1_epi8(worker.prev_chunk[worker.pos1]));
//       // uint32_t mask = (1 << (worker.pos2 - worker.pos1)) - 1;
//       // int result = _mm256_movemask_epi8(cmp);

//       // if ((result & mask) == mask) {
//       //   worker.isSame = true;
//       // }
//     } else {
//       worker.tries[wIndex]++;
//       worker.random_switcher = worker.prev_lhash ^ worker.lhash ^ worker.tries[wIndex];
//       // __builtin_prefetch(&worker.random_switcher,0,3);
//       // printf("%d worker.random_switcher %d %08jx\n", worker.tries[wIndex], worker.random_switcher, worker.random_switcher);

//       prevOp = worker.op;
//       worker.op = static_cast<byte>(worker.random_switcher);
//       // if (debugOpOrder) worker.opsA.push_back(worker.op);

//       // printf("op: %d\n", worker.op);

//       byte p1 = static_cast<byte>(worker.random_switcher >> 8);
//       byte p2 = static_cast<byte>(worker.random_switcher >> 16);

//       if (p1 > p2)
//       {
//         std::swap(p1, p2);
//       }

//       if (p2 - p1 > 32)
//       {
//         p2 = p1 + ((p2 - p1) & 0x1f);
//       }

//       if (worker.tries[wIndex] > 0) {
//         lp1 = std::min(lp1, p1);
//         lp2 = std::max(lp2, p2);
//       }

//       if (p1 < worker.pos1 || p2 > worker.pos2) worker.isSame = false;

//       worker.pos1 = p1;
//       worker.pos2 = p2;

//       worker.chunk = &worker.sData[wIndex * ASTRO_SCRATCH_SIZE + (worker.tries[wIndex] - 1) * 256];

//       if (worker.tries[wIndex] == 1) {
//         worker.prev_chunk = worker.chunk;
//       } else {
//         worker.prev_chunk = &worker.sData[wIndex * ASTRO_SCRATCH_SIZE + (worker.tries[wIndex] - 2) * 256];

//         __builtin_prefetch(worker.prev_chunk,0,3);
//         __builtin_prefetch(worker.prev_chunk+64,0,3);
//         __builtin_prefetch(worker.prev_chunk+128,0,3);
//         __builtin_prefetch(worker.prev_chunk+192,0,3);

//         // Calculate the start and end blocks
//         int start_block = 0;
//         int end_block = worker.pos1 / 16;

//         // Copy the blocks before worker.pos1
//         for (int i = start_block; i < end_block; i++) {
//             __m128i prev_data = _mm_loadu_si128((__m128i*)&worker.prev_chunk[i * 16]);
//             _mm_storeu_si128((__m128i*)&worker.chunk[i * 16], prev_data);
//         }

//         // Copy the remaining bytes before worker.pos1
//         for (int i = end_block * 16; i < worker.pos1; i++) {
//             worker.chunk[i] = worker.prev_chunk[i];
//         }

//         // Calculate the start and end blocks
//         start_block = (worker.pos2 + 15) / 16;
//         end_block = 16;

//         // Copy the blocks after worker.pos2
//         for (int i = start_block; i < end_block; i++) {
//             __m128i prev_data = _mm_loadu_si128((__m128i*)&worker.prev_chunk[i * 16]);
//             _mm_storeu_si128((__m128i*)&worker.chunk[i * 16], prev_data);
//         }

//         // Copy the remaining bytes after worker.pos2
//         for (int i = worker.pos2; i < start_block * 16; i++) {
//           worker.chunk[i] = worker.prev_chunk[i];
//         }
//       }
//     }

//     __builtin_prefetch(&worker.chunk[worker.pos1],1,3);

//     // if (debugOpOrder && worker.op == sus_op) {
//     //   printf("SIMD pre op %d:\n", worker.op);
//     //   for (int i = 0; i < 256; i++) {
//     //       printf("%02X ", worker.prev_chunk[i]);
//     //   } 
//     //   printf("\n");
//     // }
//     // fmt::printf("op: %d, ", worker.op);
//     // fmt::printf("worker.pos1: %d, worker.pos2: %d\n", worker.pos1, worker.pos2);

//     // if (worker.op > astro_branched_zOp::branchComputeSize) return;
//     // if (!isTest) {
//     //   printf("op: %d, Input State: ", worker.op);
//     //   for (int i = worker.pos1; i < worker.pos2; i++) {
//     //     printf("%02X ", worker.prev_chunk[i]);
//     //   }
//     //   printf(", same: %d", worker.isSame);
//     //   printf("\n");
//     // }

//     __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
//     __m256i old = data;
//     if (!isTest && !worker.isSame) {
//       __m256i cmp = _mm256_cmpeq_epi8(data, _mm256_set1_epi8(worker.prev_chunk[worker.pos1]));
//       uint32_t mask = (1 << (worker.pos2 - worker.pos1)) - 1;
//       int result = _mm256_movemask_epi8(cmp);

//       if ((result & mask) == mask) {
//         worker.isSame = true;
//         // if (worker.pos2 - worker.pos1 > 0) printf("same after op: %d, for op: %d\n", prevOp, worker.op);
//       }
//     } else {
//       // sameCount++;
//     }

//     astro_branched_zOp::branchCompute[worker.op + (256*(worker.isSame))](worker, data, old, wIndex);

//     if(isTest) {
//       break;
//     }
  
//     // if (op == 53) {
//     //   std::cout << hexStr(worker.chunk, 256) << std::endl << std::endl;
//     //   std::cout << hexStr(&worker.chunk[worker.pos1], 1) << std::endl;
//     //   std::cout << hexStr(&worker.chunk[worker.pos2], 1) << std::endl;
//     // }

//     __builtin_prefetch(worker.chunk,0,3);
//     // __builtin_prefetch(worker.chunk+64,0,3);
//     // __builtin_prefetch(worker.chunk+128,0,3);
//     __builtin_prefetch(worker.chunk+192,0,3);

//     uint8_t pushPos1 = lp1;
//     uint8_t pushPos2 = lp2;

//     if (worker.pos1 == worker.pos2) {
//       pushPos1 = -1;
//       pushPos2 = -1;
//     }

//     worker.A = (worker.chunk[worker.pos1] - worker.chunk[worker.pos2]);
//     worker.A = (256 + (worker.A % 256)) % 256;

//     if (worker.A < 0x10)
//     { // 6.25 % probability
//       worker.prev_lhash = worker.lhash + worker.prev_lhash;
//       worker.lhash = XXHash64::hash(worker.chunk, worker.pos2, 0);

//       // uint64_t test = XXHash64::hash(worker.chunk, worker.pos2, 0);
//       #ifdef DEBUG_OP_ORDER
//       if (worker.op == sus_op && debugOpOrder) printf("SIMD: A: new worker.lhash: %08jx\n", worker.lhash);
//       #endif
//     }

//     if (worker.A < 0x20)
//     { // 12.5 % probability
//       worker.prev_lhash = worker.lhash + worker.prev_lhash;
//       worker.lhash = hash_64_fnv1a(worker.chunk, worker.pos2);

//       // uint64_t test = hash_64_fnv1a(worker.chunk, worker.pos2);
//       #ifdef DEBUG_OP_ORDER
//       if (worker.op == sus_op && debugOpOrder) printf("SIMD: B: new worker.lhash: %08jx\n", worker.lhash);
//       #endif
//     }

//     if (worker.A < 0x30)
//     { // 18.75 % probability
//       worker.prev_lhash = worker.lhash + worker.prev_lhash;
//       HH_ALIGNAS(16)
//       const highwayhash::HH_U64 key2[2] = {worker.tries[wIndex], worker.prev_lhash};
//       worker.lhash = highwayhash::SipHash(key2, (char*)worker.chunk, worker.pos2); // more deviations

//       // uint64_t test = highwayhash::SipHash(key2, (char*)worker.chunk, worker.pos2); // more deviations
//       #ifdef DEBUG_OP_ORDER
//       if (worker.op == sus_op && debugOpOrder) printf("SIMD: C: new worker.lhash: %08jx\n", worker.lhash);
//       #endif
//     }

//     if (worker.A <= 0x40)
//     { // 25% probablility
//       // if (debugOpOrder && worker.op == sus_op) {
//       //   printf("SIMD: D: RC4 key:\n");
//       //   for (int i = 0; i < 256; i++) {
//       //     printf("%d, ", worker.key.data[i]);
//       //   }
//       // }
//       RC4(&worker.key[wIndex], 256, worker.chunk, worker.chunk);
//       worker.isSame = false;
//       if (255 - pushPos2 < MINPREFLEN)
//         pushPos2 = 255;
//       if (pushPos1 < MINPREFLEN)
//         pushPos1 = 0;


//       if (pushPos1 == 255) pushPos1 = 0;
      
//       worker.astroTemplate[worker.templateIdx] = templateMarker{
//         (uint8_t)(chunkCount > 1 ? pushPos1 : 0),
//         (uint8_t)(chunkCount > 1 ? pushPos2 : 255),
//         (uint16_t)0,
//         (uint16_t)0,
//         (uint16_t)((firstChunk << 7) | chunkCount)
//       };

//       pushPos1 = 0;
//       pushPos2 = 255;
//       worker.templateIdx += (worker.tries[wIndex] > 1);
//       firstChunk = worker.tries[wIndex]-1;
//       lp1 = 255;
//       lp2 = 0;
//       chunkCount = 1;
//     } else {
//       chunkCount++;
//     }

//     worker.chunk[255] = worker.chunk[255] ^ worker.chunk[worker.pos1] ^ worker.chunk[worker.pos2];

//     if (255 - pushPos2 < MINPREFLEN)
//       pushPos2 = 255;
//     if (pushPos1 < MINPREFLEN)
//       pushPos1 = 0;


//     // if (debugOpOrder && worker.op == sus_op) {
//     //   printf("SIMD op %d result:\n", worker.op);
//     //   for (int i = 0; i < 256; i++) {
//     //       printf("%02X ", worker.chunk[i]);
//     //   } 
//     //   printf("\n");
//     // }

//     // memcpy(&worker.sData[(worker.tries[wIndex] - 1) * 256], worker.chunk, 256);
    
//     // std::copy(worker.chunk, worker.chunk + 256, &worker.sData[(worker.tries[wIndex] - 1) * 256]);

//     // memcpy(&worker->data.data()[(worker.tries[wIndex] - 1) * 256], worker.chunk, 256);

//     // std::cout << hexStr(worker.chunk, 256) << std::endl;

//     // if (worker.tries[wIndex] == 1) lastChar = worker.sData[0];
//     // for (int i = 0; i < 256; i++) {
//     //   if ((worker.tries[wIndex] > 1 || i > 0) && worker.chunk[i] != lastChar) {
//     //     lastChar = worker.chunk[i];
//     //     maxRepeat = std::max(maxRepeat, repeatCounter);
//     //     repeatCounter = 1;
//     //     reducedChars++;
//     //   } else {
//     //     repeatCounter++;
//     //   }
//     // }

//     if (worker.tries[wIndex] > 260 + 16 || (worker.sData[(worker.tries[wIndex]-1)*256+255] >= 0xf0 && worker.tries[wIndex] > 260))
//     {
//       break;
//     }
//   }

//   if (chunkCount > 0) {
//     if (255 - lp2 < MINPREFLEN)
//       lp2 = 255;
//     if (lp1 < MINPREFLEN)
//       lp1 = 0;
//     worker.astroTemplate[worker.templateIdx] = templateMarker{
//       (uint8_t)(chunkCount > 1 ? lp1 : 0),
//       (uint8_t)(chunkCount > 1 ? lp2 : 255),
//       (uint16_t)0,
//       (uint16_t)0,
//       (uint16_t)((firstChunk << 7) | chunkCount)
//     };
//     worker.templateIdx++;
//   }

//   // if (!isTest) printf("longest repeated length: %d\n", maxRepeat);
//   // if (!isTest) printf("%d out of %d ops had repeating char inputs\n", sameCount, worker.tries[wIndex]);
//   worker.data_len = static_cast<uint32_t>((worker.tries[wIndex] - 4) * 256 + (((static_cast<uint64_t>(worker.chunk[253]) << 8) | static_cast<uint64_t>(worker.chunk[254])) & 0x3ff));
//   // if (!isTest) printf("dataLen vs reducedChars: %d / %d\n", worker.data_len, reducedChars);
// }

// #endif

// SIMD chunk copy

#if defined(__x86_64__)

    //     __builtin_prefetch(worker.prev_chunk,0,3);
    //     __builtin_prefetch(worker.prev_chunk+64,0,3);
    //     __builtin_prefetch(worker.prev_chunk+128,0,3);
    //     __builtin_prefetch(worker.prev_chunk+192,0,3);

    //     // Calculate the start and end blocks
    //     int start_block = 0;
    //     int end_block = worker.pos1 / 16;

    //     // Copy the blocks before worker.pos1
    //     for (int i = start_block; i < end_block; i++) {
    //         __m128i prev_data = _mm_loadu_si128((__m128i*)&worker.prev_chunk[i * 16]);
    //         _mm_storeu_si128((__m128i*)&worker.chunk[i * 16], prev_data);
    //     }

    //     // Copy the remaining bytes before worker.pos1
    //     for (int i = end_block * 16; i < worker.pos1; i++) {
    //         worker.chunk[i] = worker.prev_chunk[i];
    //     }

    //     // Calculate the start and end blocks
    //     start_block = (worker.pos2 + 15) / 16;
    //     end_block = 16;

    //     // Copy the blocks after worker.pos2
    //     for (int i = start_block; i < end_block; i++) {
    //         __m128i prev_data = _mm_loadu_si128((__m128i*)&worker.prev_chunk[i * 16]);
    //         _mm_storeu_si128((__m128i*)&worker.chunk[i * 16], prev_data);
    //     }

    //     // Copy the remaining bytes after worker.pos2
    //     for (int i = worker.pos2; i < start_block * 16; i++) {
    //       worker.chunk[i] = worker.prev_chunk[i];
    //     }
    //   }
    // }

__attribute__ ((target("avx512f")))
// // Copy prev_chunk between start -> end to chunk (inclusive)
inline void copyChunkData(workerData &worker, uint8_t start, uint8_t end) {
  for (int i = start; i + 63 < end; i += 64) {
    __m512i prev_data = _mm512_loadu_si512((__m512i*)&worker.prev_chunk[i]);
    _mm512_storeu_si512((__m512i*)&worker.chunk[i], prev_data);
  }
}

__attribute__ ((target("avx2")))
// Copy prev_chunk between start -> end to chunk (inclusive)
void copyChunkData(workerData &worker, int start, int end) {
  for (int i = start; i < end; i += 32) {
    __m256i prev_data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[i]);
    _mm256_storeu_si256((__m256i*)&worker.chunk[i], prev_data);
  }
}
__attribute__ ((target("sse2")))
// Copy prev_chunk between start -> end to chunk (inclusive)
void copyChunkData(workerData &worker, int start, int end) {
  for (int i = start; i < end; i += 16) {
    __m128i prev_data = _mm_loadu_si128((__m128i*)&worker.prev_chunk[i]);
    _mm_storeu_si128((__m128i*)&worker.chunk[i], prev_data);
  }
}
__attribute__ ((target("default")))
#endif

// Copy prev_chunk between start -> end to chunk (inclusive)
void copyChunkData(workerData &worker, int start, int end) {
  std::copy_n(&worker.prev_chunk[start], end - start, &worker.chunk[start]);
}

// WOLF CODE

void wolfCompute(workerData &worker, bool isTest, int wIndex)
{
  byte prevOp;
  int changeCount = 0;

  worker.templateIdx = 0;
  uint8_t chunkCount = 1;
  int firstChunk = 0;

  uint8_t lp1 = 0;
  uint8_t lp2 = 255;
  const bool phaseTelemetry = isPhaseTelemetryEnabled();
  std::array<uint64_t, kSpsaOpFamilyCount> phase_spsa_op_family_calls{0, 0, 0, 0};
  std::array<uint64_t, kSpsaOpFamilyCount> phase_spsa_op_family_bytes{0, 0, 0, 0};

  worker.tries[wIndex] = 0;
  for (int it = 0; it < 278; ++it)
  {
      // TODO prefetch next chunk into L2
      worker.tries[wIndex]++;
      worker.random_switcher = worker.prev_lhash ^ worker.lhash ^ worker.tries[wIndex];

      prevOp = worker.op;
      worker.op = static_cast<byte>(worker.random_switcher);

      byte p1 = static_cast<byte>(worker.random_switcher >> 8);
      byte p2 = static_cast<byte>(worker.random_switcher >> 16);

      if (p1 > p2)
      {
        std::swap(p1, p2);
      }

      if (p2 - p1 > 32)
      {
        p2 = p1 + ((p2 - p1) & 0x1f);
      }

      if (worker.tries[wIndex] > 0) {
        lp1 = std::min(lp1, p1);
        lp2 = std::max(lp2, p2);
      }

      if (p1 < worker.pos1 || p2 > worker.pos2) {worker.isSame = false; changeCount++;}

      worker.pos1 = p1;
      worker.pos2 = p2;
      const uint8_t span = (worker.pos2 > worker.pos1)
          ? static_cast<uint8_t>(worker.pos2 - worker.pos1)
          : 0;
      if (phaseTelemetry) {
        const size_t family = classifySpsaOpFamily(worker.op);
        phase_spsa_op_family_calls[family] += 1;
        phase_spsa_op_family_bytes[family] += span;
      }

      worker.chunk = &worker.sData[wIndex * ASTRO_SCRATCH_SIZE + (worker.tries[wIndex] - 1) * 256];

      if (worker.tries[wIndex] == 1) {
        worker.prev_chunk = worker.chunk;
      } else {
        worker.prev_chunk = &worker.sData[wIndex * ASTRO_SCRATCH_SIZE + (worker.tries[wIndex] - 2) * 256];

        // Full 256-byte copy required: wolfPermute reads from chunk[p1:p2)
        // See USE_SELECTIVE_MEMCPY comment in astroworker.h for details
        memcpy(worker.chunk, worker.prev_chunk, 256);
      }
    // }

    // TODO: Make below in all SIMD variants in a function, using FMV for architecture-agnostic calling
    // if FMV causes slowdown from overhead, use a live cached dispatch similar to wolfPermute
    #if defined(__AVX2__)
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    (void)data;  // suppress unused warning - load is for prefetch
    #endif

    if (worker.op == 253)
    {
      copyChunkData(worker, worker.pos1, worker.pos2);
      for (int i = worker.pos1; i < worker.pos2; i++)
      {

        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 3);  // rotate  bits by 3
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] ^= worker.prev_chunk[worker.pos2];     // XOR
        worker.chunk[i] = rl8(worker.chunk[i], 3);  // rotate  bits by 3
        // INSERT_RANDOM_CODE_END

        worker.prev_lhash = worker.lhash + worker.prev_lhash;
        worker.lhash = XXHash64::hash(worker.chunk, worker.pos2,0);
      }

      goto after;
    }
    if (worker.op >= 254) {
#if USE_FAST_RC4
      // Sync to OpenSSL key for SPSA compatibility
      rc4_avx512::fast_rc4_set_key_dual(worker.fast_rc4_key[wIndex], &worker.key[wIndex], 256, worker.prev_chunk);
#elif USE_CRYPTOGAMS_RC4_DUAL
      // Dual-state: Initialize both CRYPTOGAMS (fast) and OpenSSL (SPSA compat)
      // This happens ~0.78% of iterations, so sync cost is minimal
      worker.cryptogams_rc4[wIndex].set_key(worker.prev_chunk, 256);
      RC4_set_key(&worker.key[wIndex], 256, worker.prev_chunk);
#else
      RC4_set_key(&worker.key[wIndex], 256,  worker.prev_chunk);
#endif
    }
    // Use FMV-resolved wolfPermute - best version selected at program load time
    // No runtime dispatch overhead (was causing ~15-20% slowdown)
    wolfPermute(worker.prev_chunk, worker.chunk, worker.op, worker.pos1, worker.pos2, worker);

    if (!worker.op) {
      if ((worker.pos2-worker.pos1)%2 == 1) {
        worker.t1 = worker.chunk[worker.pos1];
        worker.t2 = worker.chunk[worker.pos2];
        worker.chunk[worker.pos1] = reverse8(worker.t2);
        worker.chunk[worker.pos2] = reverse8(worker.t1);
        worker.isSame = false;
      }
    }

after:
    uint8_t pushPos1 = lp1;
    uint8_t pushPos2 = lp2;

    if (worker.pos1 == worker.pos2) {
      pushPos1 = -1;
      pushPos2 = -1;
    }

    worker.A = (worker.chunk[worker.pos1] - worker.chunk[worker.pos2]);
    worker.A = (256 + (worker.A % 256)) % 256;

    // Branchless hash dispatch: replace 3 unpredictable cascading if/else with
    // a single computed selector + switch/fallthrough. ~81% of the time sel==0
    // (no hash needed), so the fast path is a single comparison.
    {
      const int hash_sel = (worker.A < 0x30) + (worker.A < 0x20) + (worker.A < 0x10);
      switch (hash_sel) {
        case 3:  // A < 0x10: XXHash + FNV + SipHash
          worker.prev_lhash = worker.lhash + worker.prev_lhash;
          worker.lhash = XXHash64::hash(worker.chunk, worker.pos2, 0);
          #ifdef DEBUG_OP_ORDER
          if (worker.op == sus_op && debugOpOrder)  printf("Wolf: A: new worker.lhash: %08jx\n", worker.lhash);
          #endif
          [[fallthrough]];
        case 2:  // A < 0x20: FNV + SipHash
          worker.prev_lhash = worker.lhash + worker.prev_lhash;
          worker.lhash = hash_64_fnv1a(worker.chunk, worker.pos2);
          #ifdef DEBUG_OP_ORDER
          if (worker.op == sus_op && debugOpOrder)  printf("Wolf: B: new worker.lhash: %08jx\n", worker.lhash);
          #endif
          [[fallthrough]];
        case 1:  // A < 0x30: SipHash only
          {
            worker.prev_lhash = worker.lhash + worker.prev_lhash;
            HH_ALIGNAS(16)
            const highwayhash::HH_U64 key2[2] = {worker.tries[wIndex], worker.prev_lhash};
            worker.lhash = highwayhash::SipHash(key2, (char*)worker.chunk, worker.pos2);
            #ifdef DEBUG_OP_ORDER
            if (worker.op == sus_op && debugOpOrder)  printf("Wolf: C: new worker.lhash: %08jx\n", worker.lhash);
            #endif
          }
          break;
        default:  // hash_sel == 0, A >= 0x30: no hash (~81%)
          break;
      }
    }

    if (worker.A <= 0x40)
    { // 25% probablility
#if USE_FAST_RC4
      // Sync to OpenSSL key for SPSA compatibility
      rc4_avx512::fast_rc4_dual(worker.fast_rc4_key[wIndex], &worker.key[wIndex], 256, worker.chunk, worker.chunk);
#elif USE_CRYPTOGAMS_RC4_DUAL
      // Use fast CRYPTOGAMS RC4 for encryption
      // OpenSSL state not updated - SPSA only needs S-box from RC4_set_key
      worker.cryptogams_rc4[wIndex].apply_keystream_256(worker.chunk);
#else
      RC4(&worker.key[wIndex], 256, worker.chunk,  worker.chunk);
#endif
      worker.isSame = false;
      if (255 - pushPos2 < MINPREFLEN)
        pushPos2 = 255;
      if (pushPos1 < MINPREFLEN)
        pushPos1 = 0;


      if (pushPos1 == 255) pushPos1 = 0;
      
      worker.astroTemplate[worker.templateIdx] = templateMarker{
        (uint8_t)(chunkCount > 1 ? pushPos1 : 0),
        (uint8_t)(chunkCount > 1 ? pushPos2 : 255),
        (uint16_t)0,
        (uint16_t)0,
        (uint16_t)((firstChunk << 7) | chunkCount)
      };

      pushPos1 = 0;
      pushPos2 = 255;
      worker.templateIdx += (worker.tries[wIndex] > 1);
      firstChunk = worker.tries[wIndex]-1;
      lp1 = 255;
      lp2 = 0;
      chunkCount = 1;
    } else {
      chunkCount++;
    }

    worker.chunk[255] = worker.chunk[255] ^ worker.chunk[worker.pos1] ^ worker.chunk[worker.pos2];

    if (255 - pushPos2 < MINPREFLEN)
      pushPos2 = 255;
    if (pushPos1 < MINPREFLEN)
      pushPos1 = 0;

    #ifdef DEBUG_OP_ORDER
    if (debugOpOrder && worker.op == sus_op) {
      printf("Wolf op %d result:\n", worker.op);
      for (int i = 0; i < 256; i++) {
        printf("%02X ", worker.chunk[i]);
      } 
      printf("\n");
    }
    #endif

    if (worker.tries[wIndex] > 260 + 16 || (worker.sData[(worker.tries[wIndex]-1)*256+255] >= 0xf0 && worker.tries[wIndex] > 260))
    {
      break;
    }
  }

  if (chunkCount > 0) {
    if (255 - lp2 < MINPREFLEN)
      lp2 = 255;
    if (lp1 < MINPREFLEN)
      lp1 = 0;
    worker.astroTemplate[worker.templateIdx] = templateMarker{
      (uint8_t)(chunkCount > 1 ? lp1 : 0),
      (uint8_t)(chunkCount > 1 ? lp2 : 255),
      (uint16_t)0,
      (uint16_t)0,
      (uint16_t)((firstChunk << 7) | chunkCount)
    };

    worker.templateIdx++;
  }

  // printf("%dc\n", changeCount);
  worker.data_len = static_cast<uint32_t>((worker.tries[wIndex] - 4) * 256 + (((static_cast<uint64_t>(worker.chunk[253]) << 8) | static_cast<uint64_t>(worker.chunk[254])) & 0x3ff));
  if (phaseTelemetry) {
    addSpsaOpFamilyTelemetryBatch(phase_spsa_op_family_calls, phase_spsa_op_family_bytes);
  }
}


// Compute the new values for worker.chunk using layered lookup tables instead of
// branched computational operations (memory-bound, reduces thermal throttling)
void lookupCompute(workerData &worker, bool isTest, int wIndex)
{
#if USE_LOOKUP_TABLES
  byte prevOp;
  int changeCount = 0;

  worker.templateIdx = 0;
  uint8_t chunkCount = 1;
  int firstChunk = 0;

  uint8_t lp1 = 0;
  uint8_t lp2 = 255;
  const bool phaseTelemetry = isPhaseTelemetryEnabled();
  std::array<uint64_t, kSpsaOpFamilyCount> phase_spsa_op_family_calls{0, 0, 0, 0};
  std::array<uint64_t, kSpsaOpFamilyCount> phase_spsa_op_family_bytes{0, 0, 0, 0};

  worker.tries[wIndex] = 0;
  for (int it = 0; it < 278; ++it)
  {
      worker.tries[wIndex]++;
      worker.random_switcher = worker.prev_lhash ^ worker.lhash ^ worker.tries[wIndex];

      prevOp = worker.op;
      worker.op = static_cast<byte>(worker.random_switcher);

      byte p1 = static_cast<byte>(worker.random_switcher >> 8);
      byte p2 = static_cast<byte>(worker.random_switcher >> 16);

      if (p1 > p2)
      {
        std::swap(p1, p2);
      }

      if (p2 - p1 > 32)
      {
        p2 = p1 + ((p2 - p1) & 0x1f);
      }

      if (worker.tries[wIndex] > 0) {
        lp1 = std::min(lp1, p1);
        lp2 = std::max(lp2, p2);
      }

      if (p1 < worker.pos1 || p2 > worker.pos2) {worker.isSame = false; changeCount++;}

      worker.pos1 = p1;
      worker.pos2 = p2;
      const uint8_t span = (worker.pos2 > worker.pos1)
          ? static_cast<uint8_t>(worker.pos2 - worker.pos1)
          : 0;
      if (phaseTelemetry) {
        const size_t family = classifySpsaOpFamily(worker.op);
        phase_spsa_op_family_calls[family] += 1;
        phase_spsa_op_family_bytes[family] += span;
      }

      worker.chunk = &worker.sData[wIndex * ASTRO_SCRATCH_SIZE + (worker.tries[wIndex] - 1) * 256];

      if (worker.tries[wIndex] == 1) {
        worker.prev_chunk = worker.chunk;
      } else {
        worker.prev_chunk = &worker.sData[wIndex * ASTRO_SCRATCH_SIZE + (worker.tries[wIndex] - 2) * 256];
        memcpy(worker.chunk, worker.prev_chunk, 256);
      }

    if (worker.op == 253)
    {
      copyChunkData(worker, worker.pos1, worker.pos2);
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        worker.chunk[i] = rl8(worker.chunk[i], 3);
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);
        worker.chunk[i] ^= worker.prev_chunk[worker.pos2];
        worker.chunk[i] = rl8(worker.chunk[i], 3);

        worker.prev_lhash = worker.lhash + worker.prev_lhash;
        worker.lhash = XXHash64::hash(worker.chunk, worker.pos2,0);
      }

      goto after_lookup;
    }
    if (worker.op >= 254) {
#if USE_FAST_RC4
      rc4_avx512::fast_rc4_set_key_dual(worker.fast_rc4_key[wIndex], &worker.key[wIndex], 256, worker.prev_chunk);
#elif USE_CRYPTOGAMS_RC4_DUAL
      worker.cryptogams_rc4[wIndex].set_key(worker.prev_chunk, 256);
      RC4_set_key(&worker.key[wIndex], 256, worker.prev_chunk);
#else
      RC4_set_key(&worker.key[wIndex], 256,  worker.prev_chunk);
#endif
    }

    // === LOOKUP TABLE BRANCH (replaces wolfPermute) ===
    {
      const bool isBranched = g_is_branched[worker.op];
      const bool smart_mode = (g_lookup_mode == LOOKUP_MODE_SMART);
      const bool branched_hybrid = isBranched && (g_lookup_mode == LOOKUP_MODE_HYBRID);
      const uint8_t span = (worker.pos2 > worker.pos1)
          ? static_cast<uint8_t>(worker.pos2 - worker.pos1)
          : 0;
      const bool smart_can_use_avx2 = smart_mode && isBranched && lookupSmartAvx2Available();
      const bool branched_smart_avx2 =
          smart_can_use_avx2 && (span > static_cast<uint8_t>(g_lookup_smart_threshold));
      const uint8_t* lut = nullptr;
      bool use_lut = !(branched_hybrid || branched_smart_avx2);
      alignas(64) uint8_t branch_row[256];

      if (smart_mode && isBranched && g_lookup_smart_telemetry) {
        const size_t hist_index = static_cast<size_t>(std::min<int>(span, 32));
        g_lookup_smart_branched_total.fetch_add(1, std::memory_order_relaxed);
        g_lookup_smart_span_hist[hist_index].fetch_add(1, std::memory_order_relaxed);
        if (branched_smart_avx2) {
          g_lookup_smart_path_avx2.fetch_add(1, std::memory_order_relaxed);
        } else {
          g_lookup_smart_path_lut.fetch_add(1, std::memory_order_relaxed);
        }
      }

      if (branched_hybrid || branched_smart_avx2) {
        // Hybrid/smart AVX2 path: regular ops still use 1D LUT, branched ops run wolfPermute.
        wolfPermute(worker.prev_chunk, worker.chunk, worker.op, worker.pos1, worker.pos2, worker);
      } else if (g_lookup_mode == LOOKUP_MODE_FULL && lookup_full::g_lookup3D != nullptr) {
        const size_t full_base = (static_cast<size_t>(worker.op) << 16) +
                                 (static_cast<size_t>(worker.prev_chunk[worker.pos2]) << 8);
        lut = &lookup_full::g_lookup3D[full_base];
      } else if (!isBranched) {
        // 1D lookup: regular ops (no pos2val dependency)
        // Table: 152 x 256 = 38 KB (fits in L1 cache!)
        lut = &lookup1D_global[static_cast<size_t>(g_reg_idx[worker.op]) * 256];
      } else if (g_lookup_mode == LOOKUP_MODE_3D && lookup3D_global != nullptr) {
        const size_t firstIndex =
            static_cast<size_t>(g_branched_idx[worker.op]) * 256 * 256 +
            static_cast<size_t>(worker.prev_chunk[worker.pos2]) * 256;
        lut = &lookup3D_global[firstIndex];
      } else {
        // 1D mode for branched ops: build a 256-byte row per iteration.
        const uint8_t pos2val = worker.prev_chunk[worker.pos2];
        const uint32_t opcode = CodeLUT[worker.op];
        for (int v = 0; v < 256; v++) {
          branch_row[v] = lookup_tables::computeBranchCorrect(
              static_cast<uint8_t>(v), pos2val, opcode);
        }
        lut = branch_row;
      }

      if (use_lut) {
#if defined(__x86_64__) || defined(_M_X64)
        _mm_prefetch(reinterpret_cast<const char*>(lut), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(lut + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(lut + 128), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(lut + 192), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(&worker.prev_chunk[worker.pos1]), _MM_HINT_T0);
#endif

        // Process individual bytes with unrolled loops
        int i = worker.pos1;
        for (; i + 15 < worker.pos2; i += 16) {
          worker.chunk[i]    = lut[worker.prev_chunk[i]];
          worker.chunk[i+1]  = lut[worker.prev_chunk[i+1]];
          worker.chunk[i+2]  = lut[worker.prev_chunk[i+2]];
          worker.chunk[i+3]  = lut[worker.prev_chunk[i+3]];
          worker.chunk[i+4]  = lut[worker.prev_chunk[i+4]];
          worker.chunk[i+5]  = lut[worker.prev_chunk[i+5]];
          worker.chunk[i+6]  = lut[worker.prev_chunk[i+6]];
          worker.chunk[i+7]  = lut[worker.prev_chunk[i+7]];
          worker.chunk[i+8]  = lut[worker.prev_chunk[i+8]];
          worker.chunk[i+9]  = lut[worker.prev_chunk[i+9]];
          worker.chunk[i+10] = lut[worker.prev_chunk[i+10]];
          worker.chunk[i+11] = lut[worker.prev_chunk[i+11]];
          worker.chunk[i+12] = lut[worker.prev_chunk[i+12]];
          worker.chunk[i+13] = lut[worker.prev_chunk[i+13]];
          worker.chunk[i+14] = lut[worker.prev_chunk[i+14]];
          worker.chunk[i+15] = lut[worker.prev_chunk[i+15]];
        }
        for (; i + 7 < worker.pos2; i += 8) {
          worker.chunk[i]   = lut[worker.prev_chunk[i]];
          worker.chunk[i+1] = lut[worker.prev_chunk[i+1]];
          worker.chunk[i+2] = lut[worker.prev_chunk[i+2]];
          worker.chunk[i+3] = lut[worker.prev_chunk[i+3]];
          worker.chunk[i+4] = lut[worker.prev_chunk[i+4]];
          worker.chunk[i+5] = lut[worker.prev_chunk[i+5]];
          worker.chunk[i+6] = lut[worker.prev_chunk[i+6]];
          worker.chunk[i+7] = lut[worker.prev_chunk[i+7]];
        }
        for (; i < worker.pos2; i++) {
          worker.chunk[i] = lut[worker.prev_chunk[i]];
        }
      }
    }

    if (!worker.op) {
      if ((worker.pos2-worker.pos1)%2 == 1) {
        worker.t1 = worker.chunk[worker.pos1];
        worker.t2 = worker.chunk[worker.pos2];
        worker.chunk[worker.pos1] = reverse8(worker.t2);
        worker.chunk[worker.pos2] = reverse8(worker.t1);
        worker.isSame = false;
      }
    }

after_lookup:
    uint8_t pushPos1 = lp1;
    uint8_t pushPos2 = lp2;

    if (worker.pos1 == worker.pos2) {
      pushPos1 = -1;
      pushPos2 = -1;
    }

    worker.A = (worker.chunk[worker.pos1] - worker.chunk[worker.pos2]);
    worker.A = (256 + (worker.A % 256)) % 256;

    // Branchless hash dispatch: single selector + switch/fallthrough
    // replaces 3 unpredictable cascading if/else branches
    {
      const int hash_sel = (worker.A < 0x30) + (worker.A < 0x20) + (worker.A < 0x10);
      switch (hash_sel) {
        case 3:  // A < 0x10: XXHash + FNV + SipHash
          worker.prev_lhash = worker.lhash + worker.prev_lhash;
          worker.lhash = XXHash64::hash(worker.chunk, worker.pos2, 0);
          [[fallthrough]];
        case 2:  // A < 0x20: FNV + SipHash
          worker.prev_lhash = worker.lhash + worker.prev_lhash;
          worker.lhash = hash_64_fnv1a(worker.chunk, worker.pos2);
          [[fallthrough]];
        case 1:  // A < 0x30: SipHash only
          {
            worker.prev_lhash = worker.lhash + worker.prev_lhash;
            HH_ALIGNAS(16)
            const highwayhash::HH_U64 key2[2] = {worker.tries[wIndex], worker.prev_lhash};
            worker.lhash = highwayhash::SipHash(key2, (char*)worker.chunk, worker.pos2);
          }
          break;
        default:  // hash_sel == 0, A >= 0x30: no hash (~81%)
          break;
      }
    }

    if (worker.A <= 0x40)
    {
#if USE_FAST_RC4
      rc4_avx512::fast_rc4_dual(worker.fast_rc4_key[wIndex], &worker.key[wIndex], 256, worker.chunk, worker.chunk);
#elif USE_CRYPTOGAMS_RC4_DUAL
      worker.cryptogams_rc4[wIndex].apply_keystream_256(worker.chunk);
#else
      RC4(&worker.key[wIndex], 256, worker.chunk,  worker.chunk);
#endif
      worker.isSame = false;
      if (255 - pushPos2 < MINPREFLEN)
        pushPos2 = 255;
      if (pushPos1 < MINPREFLEN)
        pushPos1 = 0;

      if (pushPos1 == 255) pushPos1 = 0;

      worker.astroTemplate[worker.templateIdx] = templateMarker{
        (uint8_t)(chunkCount > 1 ? pushPos1 : 0),
        (uint8_t)(chunkCount > 1 ? pushPos2 : 255),
        (uint16_t)0,
        (uint16_t)0,
        (uint16_t)((firstChunk << 7) | chunkCount)
      };

      pushPos1 = 0;
      pushPos2 = 255;
      worker.templateIdx += (worker.tries[wIndex] > 1);
      firstChunk = worker.tries[wIndex]-1;
      lp1 = 255;
      lp2 = 0;
      chunkCount = 1;
    } else {
      chunkCount++;
    }

    worker.chunk[255] = worker.chunk[255] ^ worker.chunk[worker.pos1] ^ worker.chunk[worker.pos2];

    if (255 - pushPos2 < MINPREFLEN)
      pushPos2 = 255;
    if (pushPos1 < MINPREFLEN)
      pushPos1 = 0;

    if (worker.tries[wIndex] > 260 + 16 || (worker.sData[(worker.tries[wIndex]-1)*256+255] >= 0xf0 && worker.tries[wIndex] > 260))
    {
      break;
    }
  }

  if (chunkCount > 0) {
    if (255 - lp2 < MINPREFLEN)
      lp2 = 255;
    if (lp1 < MINPREFLEN)
      lp1 = 0;
    worker.astroTemplate[worker.templateIdx] = templateMarker{
      (uint8_t)(chunkCount > 1 ? lp1 : 0),
      (uint8_t)(chunkCount > 1 ? lp2 : 255),
      (uint16_t)0,
      (uint16_t)0,
      (uint16_t)((firstChunk << 7) | chunkCount)
    };

    worker.templateIdx++;
  }

  worker.data_len = static_cast<uint32_t>((worker.tries[wIndex] - 4) * 256 + (((static_cast<uint64_t>(worker.chunk[253]) << 8) | static_cast<uint64_t>(worker.chunk[254])) & 0x3ff));
  if (phaseTelemetry) {
    addSpsaOpFamilyTelemetryBatch(phase_spsa_op_family_calls, phase_spsa_op_family_bytes);
  }
#else
  // Fallback to wolfCompute when lookup tables are disabled
  wolfCompute(worker, isTest, wIndex);
#endif
}
// {
//   worker.templateIdx = 0;
//   uint8_t chunkCount = 1;
//   int firstChunk = 0;

//   uint8_t lp1 = 0;
//   uint8_t lp2 = 255;
//   while (true)
//   {
//     if(isTest) {

//     } else {
//       worker.tries[wIndex]++;
//       worker.random_switcher = worker.prev_lhash ^ worker.lhash ^ worker.tries[wIndex];
//       // printf("%d worker.random_switcher %d %08jx\n", worker.tries[wIndex], worker.random_switcher, worker.random_switcher);

//       worker.op = static_cast<byte>(worker.random_switcher);
//       #ifdef DEBUG_OP_ORDER
//       if (debugOpOrder) worker.opsB.push_back(worker.op);
//       #endif

//       // printf("op: %d\n", worker.op);

//       worker.pos1 = static_cast<byte>(worker.random_switcher >> 8);
//       worker.pos2 = static_cast<byte>(worker.random_switcher >> 16);

//       // __builtin_prefetch(worker.chunk + worker.pos1, 0, 1);
//       // __builtin_prefetch(worker.maskTable, 0, 0);

//       if (worker.pos1 > worker.pos2)
//       {
//         std::swap(worker.pos1, worker.pos2);
//       }

//       if (worker.pos2 - worker.pos1 > 32)
//       {
//         worker.pos2 = worker.pos1 + ((worker.pos2 - worker.pos1) & 0x1f);
//       }

//       if (worker.tries[wIndex] > 0) {
//         lp1 = std::min(lp1, worker.pos1);
//         lp2 = std::max(lp2, worker.pos2);
//       }

//       // int otherpos = std::find(branchedOps.begin(), branchedOps.end(), worker.op) == branchedOps.end() ? 0 : worker.chunk[worker.pos2];
//       // __builtin_prefetch(&worker.chunk[worker.pos1], 0, 0);
//       // __builtin_prefetch(&worker.lookup[lookupIndex(worker.op,0,otherpos)]);
//       worker.chunk = &worker.sData[wIndex * ASTRO_SCRATCH_SIZE + (worker.tries[wIndex] - 1) * 256];
//       if (worker.tries[wIndex] == 1) {
//         worker.prev_chunk = worker.chunk;
//       } else {
//         worker.prev_chunk = &worker.sData[wIndex * ASTRO_SCRATCH_SIZE + (worker.tries[wIndex] - 2) * 256];

//         #if defined(__AVX2__)
//           // Calculate the start and end blocks
//           int start_block = 0;
//           int end_block = worker.pos1 / 16;

//           // Copy the blocks before worker.pos1
//           for (int i = start_block; i < end_block; i++) {
//               __m128i prev_data = _mm_loadu_si128((__m128i*)&worker.prev_chunk[i * 16]);
//               _mm_storeu_si128((__m128i*)&worker.chunk[i * 16], prev_data);
//           }

//           // Copy the remaining bytes before worker.pos1
//           for (int i = end_block * 16; i < worker.pos1; i++) {
//               worker.chunk[i] = worker.prev_chunk[i];
//           }

//           // Calculate the start and end blocks
//           start_block = (worker.pos2 + 15) / 16;
//           end_block = 16;

//           // Copy the blocks after worker.pos2
//           for (int i = start_block; i < end_block; i++) {
//               __m128i prev_data = _mm_loadu_si128((__m128i*)&worker.prev_chunk[i * 16]);
//               _mm_storeu_si128((__m128i*)&worker.chunk[i * 16], prev_data);
//           }

//           // Copy the remaining bytes after worker.pos2
//           for (int i = worker.pos2; i < start_block * 16; i++) {
//             worker.chunk[i] = worker.prev_chunk[i];
//           }
//         #else
//           memcpy(worker.chunk, worker.prev_chunk, 256);
//         #endif
//       }

//       #ifdef DEBUG_OP_ORDER
//       if (debugOpOrder && worker.op == sus_op) {
//         printf("Lookup pre op %d, pos1: %d, pos2: %d::\n", worker.op, worker.pos1, worker.pos2);
//         for (int i = 0; i < 256; i++) {
//             printf("%02X ", worker.prev_chunk[i]);
//         } 
//         printf("\n");
//       }
//       #endif
//     }
//     // fmt::printf("op: %d, ", worker.op);
//     // fmt::printf("worker.pos1: %d, worker.pos2: %d\n", worker.pos1, worker.pos2);

//     // printf("index: %d\n", lookupIndex(op, worker.chunk[worker.pos1], worker.chunk[worker.pos2]));

//     if (worker.op == 253) {
// #pragma GCC unroll 32
//       for (int i = worker.pos1; i < worker.pos2; i++)
//       {
//         worker.chunk[i] = worker.prev_chunk[i];
//       }
//       for (int i = worker.pos1; i < worker.pos2; i++)
//       {

//         // INSERT_RANDOM_CODE_START
//         worker.chunk[i] = rl8(worker.chunk[i], 3);  // rotate  bits by 3
//         worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
//         worker.chunk[i] ^= worker.chunk[worker.pos2];     // XOR
//         worker.chunk[i] = rl8(worker.chunk[i], 3);  // rotate  bits by 3
//         // INSERT_RANDOM_CODE_END

//         worker.prev_lhash = worker.lhash + worker.prev_lhash;
//         worker.lhash = XXHash64::hash(worker.chunk, worker.pos2,0);
//       }
//       goto after;
//     }
//     if (worker.op >= 254) {
//       RC4_set_key(&worker.key[wIndex], 256,  worker.prev_chunk);
//     }
//     {
//       bool use2D = std::find(worker.branchedOps, worker.branchedOps + branchedOps_size, worker.op) == worker.branchedOps + branchedOps_size;
//       uint16_t *lookup2D = use2D ? &worker.lookup2D[0] : nullptr;
//       byte *lookup3D = use2D ? nullptr : &worker.lookup3D[0];

//       int firstIndex;
//       __builtin_prefetch(&worker.prev_chunk[worker.pos1],0,3);
//       __builtin_prefetch(&worker.prev_chunk[worker.pos1]+192,0,3);

//       if (use2D) {
//         firstIndex = worker.reg_idx[worker.op]*(256*256);
//         int n = 0;

//         // Manually unrolled loops for repetetive efficiency. Worst possible loop count for 2D
//         // lookups is now 4, with less than 4 being pretty common.

//         //TODO: ask AI if assignment would be faster below

//         // Groups of 8
//         for (int i = worker.pos1; i < worker.pos2-7; i += 8) {
//           __builtin_prefetch(&lookup2D[firstIndex + 256*n++],0,3);
//           uint32_t val1 = (lookup2D[(firstIndex + (worker.prev_chunk[i+1] << 8)) | worker.prev_chunk[i]]) |
//             (lookup2D[(firstIndex + (worker.prev_chunk[i+3] << 8)) | worker.prev_chunk[i+2]] << 16);
//           uint32_t val2 =(lookup2D[(firstIndex + (worker.prev_chunk[i+5] << 8)) | worker.prev_chunk[i+4]]) |
//             (lookup2D[(firstIndex + (worker.prev_chunk[i+7] << 8)) | worker.prev_chunk[i+6]] << 16);

//           *(uint64_t*)&worker.chunk[i] = val1 | ((uint64_t)val2 << 32);
//         }

//         // Groups of 4
//         for (int i = worker.pos2-((worker.pos2-worker.pos1)%8); i < worker.pos2-3; i += 4) {
//           __builtin_prefetch(&lookup2D[firstIndex + 256*n++],0,3);
//           uint32_t val = lookup2D[(firstIndex + (worker.prev_chunk[i+1] << 8)) | worker.prev_chunk[i]] |
//             (lookup2D[(firstIndex + (worker.prev_chunk[i+3] << 8)) | worker.prev_chunk[i+2]] << 16);
//           *(uint32_t*)&worker.chunk[i] = val;
//         }

//         // Groups of 2
//         for (int i = worker.pos2-((worker.pos2-worker.pos1)%4); i < worker.pos2-1; i += 2) {
//           __builtin_prefetch(&lookup2D[firstIndex + 256*n++],0,3);
//           uint16_t val = lookup2D[(firstIndex + (worker.prev_chunk[i+1] << 8)) | worker.prev_chunk[i]];
//           *(uint16_t*)&worker.chunk[i] = val;
//         }

//         // Last if odd
//         if ((worker.pos2-worker.pos1)%2 != 0) {
//           uint16_t val = lookup2D[firstIndex + (worker.prev_chunk[worker.pos2-1] << 8)];
//           worker.chunk[worker.pos2-1] = (val & 0xFF00) >> 8;
//         }
//       } else {
//         firstIndex = worker.branched_idx[worker.op]*256*256 + worker.chunk[worker.pos2]*256;
//         int n = 0;

//         // Manually unrolled loops for repetetive efficiency. Worst possible loop count for 3D
//         // lookups is now 4, with less than 4 being pretty common.

//         // Groups of 16
//         for(int i = worker.pos1; i < worker.pos2-15; i += 16) {
//           __builtin_prefetch(&lookup3D[firstIndex + 64*n++],0,3);
//           worker.chunk[i] = lookup3D[firstIndex + worker.prev_chunk[i]];
//           worker.chunk[i+1] = lookup3D[firstIndex + worker.prev_chunk[i+1]];
//           worker.chunk[i+2] = lookup3D[firstIndex + worker.prev_chunk[i+2]];
//           worker.chunk[i+3] = lookup3D[firstIndex + worker.prev_chunk[i+3]];
//           worker.chunk[i+4] = lookup3D[firstIndex + worker.prev_chunk[i+4]];
//           worker.chunk[i+5] = lookup3D[firstIndex + worker.prev_chunk[i+5]];
//           worker.chunk[i+6] = lookup3D[firstIndex + worker.prev_chunk[i+6]];
//           worker.chunk[i+7] = lookup3D[firstIndex + worker.prev_chunk[i+7]];

//           worker.chunk[i+8] = lookup3D[firstIndex + worker.prev_chunk[i+8]];
//           worker.chunk[i+9] = lookup3D[firstIndex + worker.prev_chunk[i+9]];
//           worker.chunk[i+10] = lookup3D[firstIndex + worker.prev_chunk[i+10]];
//           worker.chunk[i+11] = lookup3D[firstIndex + worker.prev_chunk[i+11]];
//           worker.chunk[i+12] = lookup3D[firstIndex + worker.prev_chunk[i+12]];
//           worker.chunk[i+13] = lookup3D[firstIndex + worker.prev_chunk[i+13]];
//           worker.chunk[i+14] = lookup3D[firstIndex + worker.prev_chunk[i+14]];
//           worker.chunk[i+15] = lookup3D[firstIndex + worker.prev_chunk[i+15]];
//         }

//         // Groups of 8
//         for(int i = worker.pos2-((worker.pos2-worker.pos1)%16); i < worker.pos2-7; i += 8) {
//           __builtin_prefetch(&lookup3D[firstIndex + 64*n++],0,3);
//           worker.chunk[i] = lookup3D[firstIndex + worker.prev_chunk[i]];
//           worker.chunk[i+1] = lookup3D[firstIndex + worker.prev_chunk[i+1]];
//           worker.chunk[i+2] = lookup3D[firstIndex + worker.prev_chunk[i+2]];
//           worker.chunk[i+3] = lookup3D[firstIndex + worker.prev_chunk[i+3]];
//           worker.chunk[i+4] = lookup3D[firstIndex + worker.prev_chunk[i+4]];
//           worker.chunk[i+5] = lookup3D[firstIndex + worker.prev_chunk[i+5]];
//           worker.chunk[i+6] = lookup3D[firstIndex + worker.prev_chunk[i+6]];
//           worker.chunk[i+7] = lookup3D[firstIndex + worker.prev_chunk[i+7]];
//         }

//         // Groups of 4
//         for(int i = worker.pos2-((worker.pos2-worker.pos1)%8); i < worker.pos2-3; i+= 4) {
//           __builtin_prefetch(&lookup3D[firstIndex + 64*n++],0,3);
//           worker.chunk[i] = lookup3D[firstIndex + worker.prev_chunk[i]];
//           worker.chunk[i+1] = lookup3D[firstIndex + worker.prev_chunk[i+1]];
//           worker.chunk[i+2] = lookup3D[firstIndex + worker.prev_chunk[i+2]];
//           worker.chunk[i+3] = lookup3D[firstIndex + worker.prev_chunk[i+3]];
//         }

//         // Groups of 2
//         for(int i = worker.pos2-((worker.pos2-worker.pos1)%4); i < worker.pos2-1; i+= 2) {
//           __builtin_prefetch(&lookup3D[firstIndex + 64*n++],0,3);
//           worker.chunk[i] = lookup3D[firstIndex + worker.prev_chunk[i]];
//           worker.chunk[i+1] = lookup3D[firstIndex + worker.prev_chunk[i+1]];
//         }

//         // Last if odd
//         if ((worker.pos2-worker.pos1)%2 != 0) {
//           worker.chunk[worker.pos2-1] = lookup3D[firstIndex + worker.prev_chunk[worker.pos2-1]];
//         }
//       }
//       if (worker.op == 0) {
//         if ((worker.pos2-worker.pos1)%2 == 1) {
//           worker.t1 = worker.chunk[worker.pos1];
//           worker.t2 = worker.chunk[worker.pos2];
//           worker.chunk[worker.pos1] = reverse8(worker.t2);
//           worker.chunk[worker.pos2] = reverse8(worker.t1);
//         }
//       }
//     }

// after:

//     if(isTest) {
//       break;
//     }
//     // if (op == 53) {
//     //   std::cout << hexStr(worker.chunk, 256) << std::endl << std::endl;
//     //   std::cout << hexStr(&worker.chunk[worker.pos1], 1) << std::endl;
//     //   std::cout << hexStr(&worker.chunk[worker.pos2], 1) << std::endl;
//     // }

//     uint8_t pushPos1 = lp1;
//     uint8_t pushPos2 = lp2;

//     if (worker.pos1 == worker.pos2) {
//       pushPos1 = -1;
//       pushPos2 = -1;
//     }

//     worker.A = (worker.chunk[worker.pos1] - worker.chunk[worker.pos2]);
//     worker.A = (256 + (worker.A % 256)) % 256;

//     if (worker.A < 0x10)
//     { // 6.25 % probability
//       // __builtin_prefetch(worker.chunk);
//       worker.prev_lhash = worker.lhash + worker.prev_lhash;
//       worker.lhash = XXHash64::hash(worker.chunk, worker.pos2, 0);

//       // uint64_t test = XXHash64::hash(worker.chunk, worker.pos2, 0);
//       #ifdef DEBUG_OP_ORDER
//       if (worker.op == sus_op && debugOpOrder)  printf("Lookup: A: new worker.lhash: %08jx\n", worker.lhash);
//       #endif
//     }

//     if (worker.A < 0x20)
//     { // 12.5 % probability
//       // __builtin_prefetch(worker.chunk);
//       worker.prev_lhash = worker.lhash + worker.prev_lhash;
//       worker.lhash = hash_64_fnv1a(worker.chunk, worker.pos2);

//       // uint64_t test = hash_64_fnv1a(worker.chunk, worker.pos2);
//       #ifdef DEBUG_OP_ORDER
//       if (worker.op == sus_op && debugOpOrder)  printf("Lookup: B: new worker.lhash: %08jx\n", worker.lhash);
//       #endif
//     }

//     if (worker.A < 0x30)
//     { // 18.75 % probability
//       // std::copy(worker.chunk, worker.chunk + worker.pos2, s3);
//       // __builtin_prefetch(worker.chunk);
//       worker.prev_lhash = worker.lhash + worker.prev_lhash;
//       HH_ALIGNAS(16)
//       const highwayhash::HH_U64 key2[2] = {worker.tries[wIndex], worker.prev_lhash};
//       worker.lhash = highwayhash::SipHash(key2, (char*)worker.chunk, worker.pos2); // more deviations

//       // uint64_t test = highwayhash::SipHash(key2, (char*)worker.chunk, worker.pos2); // more deviations
//       #ifdef DEBUG_OP_ORDER
//       if (worker.op == sus_op && debugOpOrder)  printf("Lookup: C: new worker.lhash: %08jx\n", worker.lhash);
//       #endif
//     }

//     if (worker.A <= 0x40)
//     { // 25% probablility
//       // if (worker.op == sus_op && debugOpOrder) {
//       //   printf("Lookup: D: RC4 key:\n");
//       //   for (int i = 0; i < 256; i++) {
//       //     printf("%d, ", worker.key.data[i]);
//       //   }
//       // }
//       // prefetch(worker.chunk, 0, 1);
//       RC4(&worker.key[wIndex], 256, worker.chunk,  worker.chunk);
//       if (255 - pushPos2 < MINPREFLEN)
//         pushPos2 = 255;
//       if (pushPos1 < MINPREFLEN)
//         pushPos1 = 0;


//       if (pushPos1 == 255) pushPos1 = 0;
      
//       worker.astroTemplate[worker.templateIdx] = templateMarker{
//         (uint8_t)(chunkCount > 1 ? pushPos1 : 0),
//         (uint8_t)(chunkCount > 1 ? pushPos2 : 255),
//         (uint16_t)0,
//         (uint16_t)0,
//         (uint16_t)((firstChunk << 7) | chunkCount)
//       };

//       pushPos1 = 0;
//       pushPos2 = 255;
//       worker.templateIdx += (worker.tries[wIndex] > 1);
//       firstChunk = worker.tries[wIndex]-1;
//       lp1 = 255;
//       lp2 = 0;
//       chunkCount = 1;
//     } else {
//       chunkCount++;
//     }

//     worker.chunk[255] = worker.chunk[255] ^ worker.chunk[worker.pos1] ^ worker.chunk[worker.pos2];

//     if (255 - pushPos2 < MINPREFLEN)
//       pushPos2 = 255;
//     if (pushPos1 < MINPREFLEN)
//       pushPos1 = 0;

//     #ifdef DEBUG_OP_ORDER
//     if (debugOpOrder && worker.op == sus_op) {
//       printf("Lookup op %d result:\n", worker.op);
//       for (int i = 0; i < 256; i++) {
//           printf("%02X ", worker.chunk[i]);
//       } 
//       printf("\n");
//     }
//     #endif

//     // memcpy(&worker.sData[(worker.tries[wIndex] - 1) * 256], worker.chunk, 256);
    
//     // std::copy(worker.chunk, worker.chunk + 256, &worker.sData[(worker.tries[wIndex] - 1) * 256]);

//     // memcpy(&worker->data.data()[(worker.tries[wIndex] - 1) * 256], worker.chunk, 256);

//     // std::cout << hexStr(worker.chunk, 256) << std::endl;

//     if (worker.tries[wIndex] > 260 + 16 || (worker.sData[(worker.tries[wIndex]-1)*256+255] >= 0xf0 && worker.tries[wIndex] > 260))
//     {
//       break;
//     }
//   }

//   if (chunkCount > 0) {
//     if (255 - lp2 < MINPREFLEN)
//       lp2 = 255;
//     if (lp1 < MINPREFLEN)
//       lp1 = 0;
//     worker.astroTemplate[worker.templateIdx] = templateMarker{
//       (uint8_t)(chunkCount > 1 ? lp1 : 0),
//       (uint8_t)(chunkCount > 1 ? lp2 : 255),
//       (uint16_t)0,
//       (uint16_t)0,
//       (uint16_t)((firstChunk << 7) | chunkCount)
//     };
//     worker.templateIdx++;
//   }

//   worker.data_len = static_cast<uint32_t>((worker.tries[wIndex] - 4) * 256 + (((static_cast<uint64_t>(worker.chunk[253]) << 8) | static_cast<uint64_t>(worker.chunk[254])) & 0x3ff));
// }
