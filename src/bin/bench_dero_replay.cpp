#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "astrobwtv3/astrobwtv3.h"
#include "astrobwtv3/lookup_full.hpp"
#include "astrobwtv3/lookupcompute.h"
#include "dirtybird-common.hpp"
#include "dirtybird-hugepages.hpp"
#include "hex.h"
#include "lookup_mode.hpp"
#include "lookup_tables.hpp"
#include "numa_optimizer.hpp"
#include "spsa.hpp"

bool printHugepagesError = false;
bool ABORT_MINER = false;
bool g_use_spsa = true;
bool g_use_local_spsa = false; // Test library SPSA
bool g_spsa_stamp_fast = true;
bool g_spsa_decode_bases = false;
int g_spsa_bucket_prefetch = 0;
int g_spsa_max_data_len = 0;
bool g_spsa_hit_counters = true;
bool g_spsa_sha_profile = false;
bool g_spsa_sha_pair = false;
bool g_spsa_sha_zeroize = false;
bool g_verbose_tune = false;
bool g_array_telemetry = false;
bool g_phase_telemetry = true;
int g_lookup_smart_threshold = 12;
bool g_lookup_smart_telemetry = false;
bool g_print_runtime_config = false;
bool lockThreads = false;
bool g_pcores_only = false;
int g_pcore_count = 0;
int threads = 1;
int g_omp_threads = 2;
bool g_adaptive_threads = false;
int g_adaptive_warmup_secs = 15;
int g_overprovision_count = 0;
bool useLookupMine = true;
bool lookupMine = true;
bool beQuiet = true;
std::chrono::time_point<std::chrono::steady_clock> g_start_time;
std::string devWallet = "";
double devFee = 0.0;
uint8_t* lookup1D_global = nullptr;
unsigned char* lookup3D_global = nullptr;
AstroFunc allAstroFuncs[] = {
    {"branch", branchComputeCPU},
#if USE_LOOKUP_TABLES
    {"lookup", lookupCompute},
#endif
    {"wolf", wolfCompute},
    {"wolf_memopt", wolfCompute_memopt},
#if defined(__aarch64__)
    {"aarch64", branchComputeCPU_aarch64},
#endif
};
size_t numAstroFuncs = std::size(allAstroFuncs);

namespace {

using Clock = std::chrono::steady_clock;
using ReplayInput = std::array<uint8_t, MINIBLOCK_SIZE>;

struct Config {
    std::string input_file;
    std::string input_hex;
    std::string json_out;
    std::string algo = "wolf_memopt";
    std::string worker_alloc = "huge";
    int threads = 1;
    int omp_threads = 2;
    int warmup_seconds = 2;
    int duration_seconds = 10;
    int job_span_hashes = 4096;
    LookupMode lookup_mode = LOOKUP_MODE_HYBRID;
    bool exact_replay = false;
    bool use_spsa = true;
    bool spsa_sha_zeroize = false;
};

struct AllocRecord {
    enum class Kind {
        none,
        huge,
        malloc_fallback,
        numa_local
    };

    void* ptr = nullptr;
    size_t size = 0;
    Kind kind = Kind::none;
};

struct ThreadResult {
    uint64_t warmup_hashes = 0;
    uint64_t measure_hashes = 0;
};

struct WorkerAlloc {
    workerData* worker = nullptr;
    AllocRecord record;
};

bool lookup3D_from_hugepages = false;
bool g_worker_alloc_local = false;
AllocRecord lookup1d_alloc;
AllocRecord lookup3d_alloc;
AllocRecord lookup_full_alloc;

inline void write_nonce_be(uint8_t* work, uint32_t nonce) {
    work[MINIBLOCK_SIZE - 5] = static_cast<uint8_t>((nonce >> 24) & 0xFF);
    work[MINIBLOCK_SIZE - 4] = static_cast<uint8_t>((nonce >> 16) & 0xFF);
    work[MINIBLOCK_SIZE - 3] = static_cast<uint8_t>((nonce >> 8) & 0xFF);
    work[MINIBLOCK_SIZE - 2] = static_cast<uint8_t>(nonce & 0xFF);
}

std::string trim(const std::string& input) {
    const auto first = input.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const auto last = input.find_last_not_of(" \t\r\n");
    return input.substr(first, last - first + 1);
}

bool parse_on_off(const std::string& value) {
    if (value == "on" || value == "1" || value == "true") {
        return true;
    }
    if (value == "off" || value == "0" || value == "false") {
        return false;
    }
    throw std::runtime_error("Expected on/off value, got: " + value);
}

LookupMode parse_lookup_mode(const std::string& value) {
    if (value == "1d") return LOOKUP_MODE_1D;
    if (value == "3d") return LOOKUP_MODE_3D;
    if (value == "full") return LOOKUP_MODE_FULL;
    if (value == "hybrid") return LOOKUP_MODE_HYBRID;
    if (value == "smart") return LOOKUP_MODE_SMART;
    throw std::runtime_error("Unknown lookup mode: " + value);
}

const char* lookup_mode_name_local(LookupMode mode) {
    switch (mode) {
        case LOOKUP_MODE_1D: return "1d";
        case LOOKUP_MODE_3D: return "3d";
        case LOOKUP_MODE_FULL: return "full";
        case LOOKUP_MODE_HYBRID: return "hybrid";
        case LOOKUP_MODE_SMART: return "smart";
    }
    return "unknown";
}

void set_lookup_enabled(bool enabled) {
    useLookupMine = enabled;
    lookupMine = enabled;
}

void print_usage() {
    std::cout
        << "bench-dero-replay options:\n"
        << "  --input-file <path>         File containing 48-byte hex work blobs\n"
        << "  --input-hex <hex>           Single 48-byte hex work blob\n"
        << "  --threads <n>               Worker threads (default: 1)\n"
        << "  --omp-threads <n>           OpenMP threads per worker (default: 2)\n"
        << "  --warmup <sec>              Warmup seconds (default: 2)\n"
        << "  --duration <sec>            Measure seconds (default: 10)\n"
        << "  --job-span <hashes>         Hashes before rotating replay base blob (default: 4096)\n"
        << "  --lookup-mode <mode>        1d|3d|full|hybrid|smart (default: hybrid)\n"
        << "  --worker-alloc <mode>       huge|local (default: huge)\n"
        << "  --spsa <on|off>             Enable SPSA path (default: on)\n"
        << "  --spsa-sha-zeroize <on|off> Toggle SHA zeroization (default: off)\n"
        << "  --algo <name>               Astro kernel name (default: wolf_memopt)\n"
        << "  --exact-replay              Use the corpus exactly, no nonce mutation\n"
        << "  --json-out <path>           Write summary JSON\n"
        << "  --help                      Show this message\n";
}

Config parse_args(int argc, char** argv) {
    Config cfg;
    const unsigned hw_threads = std::thread::hardware_concurrency();
    if (hw_threads > 0) {
        cfg.threads = std::max(1u, hw_threads);
    }

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + flag);
            }
            return argv[++i];
        };

        if (arg == "--input-file") {
            cfg.input_file = require_value("--input-file");
        } else if (arg == "--input-hex") {
            cfg.input_hex = require_value("--input-hex");
        } else if (arg == "--threads") {
            cfg.threads = std::max(1, std::stoi(require_value("--threads")));
        } else if (arg == "--omp-threads") {
            cfg.omp_threads = std::max(0, std::stoi(require_value("--omp-threads")));
        } else if (arg == "--warmup") {
            cfg.warmup_seconds = std::max(0, std::stoi(require_value("--warmup")));
        } else if (arg == "--duration") {
            cfg.duration_seconds = std::max(1, std::stoi(require_value("--duration")));
        } else if (arg == "--job-span") {
            cfg.job_span_hashes = std::max(0, std::stoi(require_value("--job-span")));
        } else if (arg == "--lookup-mode") {
            cfg.lookup_mode = parse_lookup_mode(require_value("--lookup-mode"));
        } else if (arg == "--worker-alloc") {
            cfg.worker_alloc = require_value("--worker-alloc");
            if (cfg.worker_alloc != "huge" && cfg.worker_alloc != "local") {
                throw std::runtime_error("Unknown --worker-alloc value: " + cfg.worker_alloc);
            }
        } else if (arg == "--spsa") {
            cfg.use_spsa = parse_on_off(require_value("--spsa"));
        } else if (arg == "--spsa-sha-zeroize") {
            cfg.spsa_sha_zeroize = parse_on_off(require_value("--spsa-sha-zeroize"));
        } else if (arg == "--algo") {
            cfg.algo = require_value("--algo");
        } else if (arg == "--exact-replay") {
            cfg.exact_replay = true;
        } else if (arg == "--json-out") {
            cfg.json_out = require_value("--json-out");
        } else if (arg == "--help" || arg == "-h") {
            print_usage();
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    return cfg;
}

ReplayInput parse_input_hex_line(const std::string& raw_line) {
    std::string line = trim(raw_line);
    const auto comment = line.find('#');
    if (comment != std::string::npos) {
        line = trim(line.substr(0, comment));
    }
    if (line.empty()) {
        return {};
    }
    if (line.rfind("0x", 0) == 0 || line.rfind("0X", 0) == 0) {
        line = line.substr(2);
    }
    if (line.size() != MINIBLOCK_SIZE * 2) {
        throw std::runtime_error("Replay input must be exactly 96 hex chars, got length " +
                                 std::to_string(line.size()));
    }

    ReplayInput input{};
    hexstrToBytes(line, input.data());
    return input;
}

std::vector<ReplayInput> load_inputs(const Config& cfg) {
    std::vector<ReplayInput> inputs;

    if (!cfg.input_file.empty()) {
        std::ifstream in(cfg.input_file);
        if (!in) {
            throw std::runtime_error("Failed to open input file: " + cfg.input_file);
        }

        std::string line;
        while (std::getline(in, line)) {
            const std::string cleaned = trim(line);
            if (cleaned.empty() || cleaned[0] == '#') {
                continue;
            }
            inputs.push_back(parse_input_hex_line(cleaned));
        }
    }

    if (!cfg.input_hex.empty()) {
        inputs.push_back(parse_input_hex_line(cfg.input_hex));
    }

    if (!inputs.empty()) {
        return inputs;
    }

    std::vector<ReplayInput> synthetic(64);
    uint64_t seed = 0x9E3779B97F4A7C15ull;
    for (auto& input : synthetic) {
        for (size_t i = 0; i < input.size(); ++i) {
            seed ^= seed >> 12;
            seed ^= seed << 25;
            seed ^= seed >> 27;
            seed *= 0x2545F4914F6CDD1Dull;
            input[i] = static_cast<uint8_t>(seed >> 56);
        }
        input[0] = static_cast<uint8_t>((input[0] & 0xF0u) | 0x01u);
    }
    return synthetic;
}

AllocRecord alloc_pages(size_t size) {
    AllocRecord record{};
    record.size = size;
    record.ptr = malloc_huge_pages(size);
    if (record.ptr == nullptr) {
        record.ptr = std::malloc(size);
        record.kind = AllocRecord::Kind::malloc_fallback;
    } else {
        record.kind = AllocRecord::Kind::huge;
    }
    return record;
}

AllocRecord alloc_local_pages(size_t size) {
    AllocRecord record{};
    record.size = size;
    record.ptr = NUMAOptimizer::allocateLocal(size);
    if (record.ptr == nullptr) {
        record.ptr = std::malloc(size);
        record.kind = AllocRecord::Kind::malloc_fallback;
    } else {
        record.kind = AllocRecord::Kind::numa_local;
    }
    return record;
}

void free_alloc(const AllocRecord& record) {
    if (record.ptr == nullptr) {
        return;
    }
    if (record.kind == AllocRecord::Kind::huge) {
        free_huge_pages(record.ptr);
    } else if (record.kind == AllocRecord::Kind::numa_local) {
        NUMAOptimizer::deallocate(record.ptr, record.size);
    } else {
        std::free(record.ptr);
    }
}

void initialize_lookup_tables_for_mode(LookupMode mode) {
    lookup1d_alloc = alloc_pages(regOps_size * 256 * sizeof(uint8_t));
    if (lookup1d_alloc.ptr == nullptr) {
        throw std::runtime_error("Failed to allocate lookup1D table");
    }
    lookup1D_global = static_cast<uint8_t*>(lookup1d_alloc.ptr);

    if (mode == LOOKUP_MODE_3D) {
        lookup3d_alloc = alloc_pages(static_cast<size_t>(branchedOps_size) * 256 * 256 * sizeof(uint8_t));
        if (lookup3d_alloc.ptr == nullptr) {
            throw std::runtime_error("Failed to allocate lookup3D table");
        }
        lookup3D_global = static_cast<unsigned char*>(lookup3d_alloc.ptr);
        lookup3D_from_hugepages = (lookup3d_alloc.kind == AllocRecord::Kind::huge);
        lookup_tables::generateTables(lookup1D_global, lookup3D_global);
        return;
    }

    lookup_tables::generateTables(lookup1D_global, nullptr);

    if (mode == LOOKUP_MODE_FULL) {
        lookup_full_alloc = alloc_pages(lookup_full::LOOKUP3D_SIZE);
        if (lookup_full_alloc.ptr == nullptr) {
            throw std::runtime_error("Failed to allocate full lookup table");
        }
        lookup_full::g_lookup3D = static_cast<uint8_t*>(lookup_full_alloc.ptr);
        lookup_full::generate_lookup3D(lookup_full::g_lookup3D);
    }
}

WorkerAlloc create_worker() {
    WorkerAlloc alloc{};
    if (g_worker_alloc_local) {
        alloc.record = alloc_local_pages(sizeof(workerData));
    } else {
        alloc.record = alloc_pages(sizeof(workerData));
    }
    if (alloc.record.ptr == nullptr) {
        throw std::runtime_error("Failed to allocate workerData");
    }

    alloc.worker = static_cast<workerData*>(alloc.record.ptr);
    NUMAOptimizer::optimizeMemoryForMining(alloc.worker, sizeof(workerData));
    initWorker(*alloc.worker);
    lookupGen(*alloc.worker, nullptr, nullptr);
    return alloc;
}

void destroy_worker(const WorkerAlloc& alloc) {
    free_alloc(alloc.record);
}

double ns_per_hash(uint64_t ns, uint64_t hashes) {
    if (hashes == 0) {
        return 0.0;
    }
    return static_cast<double>(ns) / static_cast<double>(hashes);
}

double us_per_hash(uint64_t ns, uint64_t hashes) {
    return ns_per_hash(ns, hashes) / 1000.0;
}

double us_per_call(uint64_t ns, uint64_t calls) {
    if (calls == 0) {
        return 0.0;
    }
    return static_cast<double>(ns) / static_cast<double>(calls) / 1000.0;
}

void print_phase_summary(const AstroPhaseTelemetrySnapshot& phase) {
    const uint64_t hashes = std::max<uint64_t>(phase.hashes, 1);
    const uint64_t spsa_total = phase.spsa_hits + phase.spsa_misses;
    const double spsa_hit_pct = (spsa_total == 0)
        ? 0.0
        : (100.0 * static_cast<double>(phase.spsa_hits) / static_cast<double>(spsa_total));

    std::printf("phase.hashes=%llu avg_data_len=%.2f spsa_hit_pct=%.2f%%\n",
                static_cast<unsigned long long>(phase.hashes),
                static_cast<double>(phase.data_len_sum) / static_cast<double>(hashes),
                spsa_hit_pct);

    std::printf("phase.prep_us=%.3f phase.wolf_us=%.3f phase.spsa_call_us=%.3f "
                "phase.spsa_prefetch_us=%.3f phase.spsa_hit_copy_us=%.3f "
                "phase.spsa_miss_hash_us=%.3f phase.sa_fallback_us=%.3f phase.final_hash_us=%.3f "
                "phase.total_us=%.3f\n",
                us_per_hash(phase.prep_ns, hashes),
                us_per_hash(phase.wolf_ns, hashes),
                us_per_hash(phase.spsa_call_ns, hashes),
                us_per_hash(phase.spsa_prefetch_ns, hashes),
                us_per_hash(phase.spsa_hit_copy_ns, hashes),
                us_per_hash(phase.spsa_miss_hash_ns, hashes),
                us_per_hash(phase.sa_fallback_ns, hashes),
                us_per_hash(phase.final_hash_ns, hashes),
                us_per_hash(phase.total_ns, hashes));

    std::printf("phase.spsa_core_us=%.3f spsa_core_calls=%llu "
                "sa_encode_us=%.3f sa_radix_us=%.3f sa_collision_us=%.3f sa_copy_us=%.3f\n",
                us_per_call(phase.spsa_core_ns, phase.spsa_core_calls),
                static_cast<unsigned long long>(phase.spsa_core_calls),
                us_per_hash(phase.sa_encode_ns, hashes),
                us_per_hash(phase.sa_radix_ns, hashes),
                us_per_hash(phase.sa_collision_ns, hashes),
                us_per_hash(phase.sa_copy_ns, hashes));

    std::printf("phase.spsa_bins_us=[%.3f, %.3f, %.3f, %.3f] calls=[%llu, %llu, %llu, %llu]\n",
                us_per_call(phase.spsa_core_bin0_ns, phase.spsa_core_bin0_calls),
                us_per_call(phase.spsa_core_bin1_ns, phase.spsa_core_bin1_calls),
                us_per_call(phase.spsa_core_bin2_ns, phase.spsa_core_bin2_calls),
                us_per_call(phase.spsa_core_bin3_ns, phase.spsa_core_bin3_calls),
                static_cast<unsigned long long>(phase.spsa_core_bin0_calls),
                static_cast<unsigned long long>(phase.spsa_core_bin1_calls),
                static_cast<unsigned long long>(phase.spsa_core_bin2_calls),
                static_cast<unsigned long long>(phase.spsa_core_bin3_calls));

    std::printf("phase.spsa_families_calls=[nonbranched:%llu branched:%llu op253:%llu rc4:%llu] "
                "bytes=[%llu %llu %llu %llu]\n",
                static_cast<unsigned long long>(phase.spsa_op_family_nonbranched_calls),
                static_cast<unsigned long long>(phase.spsa_op_family_branched_calls),
                static_cast<unsigned long long>(phase.spsa_op_family_op253_calls),
                static_cast<unsigned long long>(phase.spsa_op_family_rc4_calls),
                static_cast<unsigned long long>(phase.spsa_op_family_nonbranched_bytes),
                static_cast<unsigned long long>(phase.spsa_op_family_branched_bytes),
                static_cast<unsigned long long>(phase.spsa_op_family_op253_bytes),
                static_cast<unsigned long long>(phase.spsa_op_family_rc4_bytes));
}

void write_json_summary(const std::string& path,
                        const Config& cfg,
                        bool self_check_passed,
                        uint64_t measure_hashes,
                        double elapsed_seconds,
                        const AstroPhaseTelemetrySnapshot& phase,
                        const AstroLookupTelemetrySnapshot& lookup) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open json output path: " + path);
    }

    out << "{\n";
    out << "  \"threads\": " << cfg.threads << ",\n";
    out << "  \"omp_threads\": " << cfg.omp_threads << ",\n";
    out << "  \"lookup_mode\": \"" << lookup_mode_name_local(cfg.lookup_mode) << "\",\n";
    out << "  \"algo\": \"" << cfg.algo << "\",\n";
    out << "  \"worker_alloc\": \"" << cfg.worker_alloc << "\",\n";
    out << "  \"exact_replay\": " << (cfg.exact_replay ? "true" : "false") << ",\n";
    out << "  \"self_check_passed\": " << (self_check_passed ? "true" : "false") << ",\n";
    out << "  \"job_span_hashes\": " << cfg.job_span_hashes << ",\n";
    out << "  \"spsa\": " << (cfg.use_spsa ? "true" : "false") << ",\n";
    out << "  \"spsa_sha_zeroize\": " << (cfg.spsa_sha_zeroize ? "true" : "false") << ",\n";
    out << "  \"hashes\": " << measure_hashes << ",\n";
    out << "  \"elapsed_seconds\": " << elapsed_seconds << ",\n";
    out << "  \"hash_rate_hs\": " << (elapsed_seconds > 0.0 ? static_cast<double>(measure_hashes) / elapsed_seconds : 0.0) << ",\n";
    out << "  \"spsa_core_us\": " << us_per_call(phase.spsa_core_ns, phase.spsa_core_calls) << ",\n";
    out << "  \"phase_total_us\": " << us_per_hash(phase.total_ns, std::max<uint64_t>(phase.hashes, 1)) << ",\n";
    out << "  \"phase_hashes\": " << phase.hashes << ",\n";
    out << "  \"phase_spsa_hits\": " << phase.spsa_hits << ",\n";
    out << "  \"phase_spsa_misses\": " << phase.spsa_misses << ",\n";
    out << "  \"lookup_smart_branched_total\": " << lookup.smart_branched_total << ",\n";
    out << "  \"lookup_smart_path_lut\": " << lookup.smart_path_lut << ",\n";
    out << "  \"lookup_smart_path_avx2\": " << lookup.smart_path_avx2 << "\n";
    out << "}\n";
}

bool run_correctness_self_check(const Config& cfg, const std::vector<ReplayInput>& inputs) {
    WorkerAlloc worker_spsa = create_worker();
    WorkerAlloc worker_ref = create_worker();
    std::array<uint8_t, MINIBLOCK_SIZE> work = inputs.front();
    std::array<uint8_t, 32> out_spsa{};
    std::array<uint8_t, 32> out_ref{};

    if (!cfg.exact_replay) {
        write_nonce_be(work.data(), 1);
        work[MINIBLOCK_SIZE - 1] = 0;
    }

    const bool saved_use_spsa = g_use_spsa;
    AstroBWTv3(work.data(), MINIBLOCK_SIZE, out_spsa.data(), *worker_spsa.worker, useLookupMine);

    g_use_spsa = false;
    AstroBWTv3(work.data(), MINIBLOCK_SIZE, out_ref.data(), *worker_ref.worker, useLookupMine);
    g_use_spsa = saved_use_spsa;

    destroy_worker(worker_spsa);
    destroy_worker(worker_ref);

    return std::memcmp(out_spsa.data(), out_ref.data(), out_spsa.size()) == 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Config cfg = parse_args(argc, argv);
        const std::vector<ReplayInput> inputs = load_inputs(cfg);

        threads = cfg.threads;
        g_omp_threads = cfg.omp_threads;
        g_use_spsa = cfg.use_spsa;
        g_spsa_sha_zeroize = cfg.spsa_sha_zeroize;
        g_lookup_mode = cfg.lookup_mode;
        g_worker_alloc_local = (cfg.worker_alloc == "local");
        set_lookup_enabled(true);
        g_start_time = Clock::now();

#ifdef _OPENMP
        if (g_omp_threads > 0) {
            omp_set_num_threads(g_omp_threads);
        }
#endif

        NUMAOptimizer::initialize();
        initialize_lookup_tables_for_mode(cfg.lookup_mode);

        detectAVX512();
        if (!setAstroAlgo(cfg.algo)) {
            throw std::runtime_error("Failed to select Astro kernel: " + cfg.algo);
        }

#if defined(USE_ASTRO_SPSA)
        if (g_use_spsa) {
            initSPSA();
        }
#endif

        const bool self_check_passed = run_correctness_self_check(cfg, inputs);
        if (!self_check_passed) {
            std::fprintf(stderr,
                         "bench-dero-replay warning: SPSA/non-SPSA self-check mismatch on the current corpus\n");
        }

        setPhaseTelemetryEnabled(true);
        resetPhaseTelemetry();
        resetLookupTelemetry();

        std::printf("=== bench-dero-replay ===\n");
        std::printf("inputs=%zu threads=%d omp_threads=%d warmup=%d duration=%d "
                    "lookup_mode=%s worker_alloc=%s algo=%s exact_replay=%s job_span=%d "
                    "spsa=%s spsa_sha_zeroize=%s\n",
                    inputs.size(),
                    cfg.threads,
                    cfg.omp_threads,
                    cfg.warmup_seconds,
                    cfg.duration_seconds,
                    lookup_mode_name_local(cfg.lookup_mode),
                    cfg.worker_alloc.c_str(),
                    cfg.algo.c_str(),
                    cfg.exact_replay ? "true" : "false",
                    cfg.job_span_hashes,
                    cfg.use_spsa ? "on" : "off",
                    cfg.spsa_sha_zeroize ? "on" : "off");

        std::atomic<int> phase{0};  // 0=warmup, 1=measure, 2=stop
        std::atomic<int> ready_count{0};
        std::vector<std::thread> workers;
        std::vector<ThreadResult> results(static_cast<size_t>(cfg.threads));

        for (int tid = 0; tid < cfg.threads; ++tid) {
            workers.emplace_back([&, tid]() {
                WorkerAlloc worker_alloc = create_worker();
                std::array<uint8_t, MINIBLOCK_SIZE> work = inputs[static_cast<size_t>(tid) % inputs.size()];
                std::array<uint8_t, 32> out{};
                size_t input_index = static_cast<size_t>(tid) % inputs.size();
                uint32_t nonce = 0;
                ThreadResult local{};

                if (!cfg.exact_replay) {
                    work[MINIBLOCK_SIZE - 1] = static_cast<uint8_t>(tid & 0xFF);
                }

                ready_count.fetch_add(1, std::memory_order_release);

                while (phase.load(std::memory_order_acquire) != 2) {
                    const int current_phase = phase.load(std::memory_order_relaxed);

                    if (cfg.exact_replay) {
                        const ReplayInput& src = inputs[input_index];
                        std::memcpy(work.data(), src.data(), MINIBLOCK_SIZE);
                        input_index = (input_index + 1) % inputs.size();
                    } else {
                        if (cfg.job_span_hashes > 0 &&
                            (local.warmup_hashes + local.measure_hashes) > 0 &&
                            ((local.warmup_hashes + local.measure_hashes) % static_cast<uint64_t>(cfg.job_span_hashes)) == 0) {
                            input_index = (input_index + 1) % inputs.size();
                            std::memcpy(work.data(), inputs[input_index].data(), MINIBLOCK_SIZE);
                            work[MINIBLOCK_SIZE - 1] = static_cast<uint8_t>(tid & 0xFF);
                            nonce = 0;
                        }
                        ++nonce;
                        write_nonce_be(work.data(), nonce);
                    }

                    AstroBWTv3(work.data(), MINIBLOCK_SIZE, out.data(), *worker_alloc.worker, useLookupMine);

                    if (current_phase == 0) {
                        ++local.warmup_hashes;
                    } else if (current_phase == 1) {
                        ++local.measure_hashes;
                    }
                }

                results[static_cast<size_t>(tid)] = local;
                destroy_worker(worker_alloc);
            });
        }

        while (ready_count.load(std::memory_order_acquire) != cfg.threads) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        if (cfg.warmup_seconds > 0) {
            std::this_thread::sleep_for(std::chrono::seconds(cfg.warmup_seconds));
        }

        resetPhaseTelemetry();
        resetLookupTelemetry();
        
        const auto measure_start = Clock::now();
        phase.store(1, std::memory_order_release);

        std::this_thread::sleep_for(std::chrono::seconds(cfg.duration_seconds));
        const auto measure_end = Clock::now();
        phase.store(2, std::memory_order_release);

        for (auto& worker : workers) {
            worker.join();
        }
        std::printf("bench-dero-replay: all workers joined\n");

        const double elapsed_seconds =
            std::chrono::duration<double>(measure_end - measure_start).count();

        uint64_t measure_hashes = 0;
        uint64_t warmup_hashes = 0;
        for (const auto& result : results) {
            measure_hashes += result.measure_hashes;
            warmup_hashes += result.warmup_hashes;
        }

        const AstroPhaseTelemetrySnapshot phase_snapshot = getPhaseTelemetrySnapshot();
        const AstroLookupTelemetrySnapshot lookup_snapshot = getLookupTelemetrySnapshot();

        std::printf("warmup_hashes=%llu measure_hashes=%llu elapsed=%.3f hash_rate=%.3f H/s\n",
                    static_cast<unsigned long long>(warmup_hashes),
                    static_cast<unsigned long long>(measure_hashes),
                    elapsed_seconds,
                    elapsed_seconds > 0.0 ? static_cast<double>(measure_hashes) / elapsed_seconds : 0.0);
        std::printf("sa_backend=%s astro_algo=%s\n",
                    getSABackendName(),
                    getCurrentAstroAlgoName());
        print_phase_summary(phase_snapshot);

        if (!cfg.json_out.empty()) {
            write_json_summary(cfg.json_out,
                               cfg,
                               self_check_passed,
                               measure_hashes,
                               elapsed_seconds,
                               phase_snapshot,
                               lookup_snapshot);
            std::printf("json_out=%s\n", cfg.json_out.c_str());
        }

        free_alloc(lookup_full_alloc);
        free_alloc(lookup3d_alloc);
        free_alloc(lookup1d_alloc);
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "bench-dero-replay error: %s\n", ex.what());
        return 1;
    }
}
