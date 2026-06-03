// bench_stage5.cpp - deterministic Stage 5 replay microbench.

#include "cap_format.h"
#include "../../include/dluna_v114.h"
#include "../../include/libsais.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <openssl/sha.h>

namespace fs = std::filesystem;
using deroluna::replay::CaptureReader;
using deroluna::replay::StageId;
using deroluna::replay::StageRecord;

enum class BenchMode {
    Libsais,
    Descriptor,
    CompactTail,
    FusedTail,
    FusedSaHashTail,
    FusedHashTail,
    Both,
};

struct Args {
    std::string captures;
    size_t max_records = 0;
    int repeat = 3;
    bool verify = true;
    BenchMode mode = BenchMode::Libsais;
};

struct Sample {
    uint64_t hash_seq = 0;
    std::vector<uint8_t> stage4_input;
    std::vector<uint8_t> input;
    std::vector<uint8_t> expected;
};

#pragma pack(push, 1)
struct Stage4InputHeader {
    char magic[8];
    uint32_t version;
    uint32_t logical_len;
    uint32_t data_len;
    uint32_t flag_len;
    uint32_t reserved;
};
#pragma pack(pop)

struct Stage4InputView {
    uint32_t logical_len = 0;
    uint32_t data_len = 0;
    uint32_t flag_len = 0;
    const uint8_t* flags = nullptr;
    const uint8_t* data = nullptr;
};

static void usage() {
    std::fprintf(stderr,
        "Usage: bench_stage5 --captures DIR [--max N] [--repeat N] "
        "[--mode libsais|descriptor|compact-tail|fused-tail|fused-sa-hash-tail|fused-hash-tail|both] "
        "[--no-verify]\n");
}

static const char* mode_name(BenchMode mode) {
    switch (mode) {
    case BenchMode::Libsais: return "libsais";
    case BenchMode::Descriptor: return "descriptor";
    case BenchMode::CompactTail: return "compact-tail";
    case BenchMode::FusedTail: return "fused-tail";
    case BenchMode::FusedSaHashTail: return "fused-sa-hash-tail";
    case BenchMode::FusedHashTail: return "fused-hash-tail";
    case BenchMode::Both: return "both";
    }
    return "unknown";
}

static deroluna::stages::v114::Stage5SaBuildMode stage_mode(BenchMode mode) {
    return mode == BenchMode::Descriptor
        ? deroluna::stages::v114::Stage5SaBuildMode::DescriptorArena
        : deroluna::stages::v114::Stage5SaBuildMode::Libsais;
}

static bool parse_mode(const std::string& value, BenchMode* mode) {
    if (value == "libsais") {
        *mode = BenchMode::Libsais;
        return true;
    }
    if (value == "descriptor") {
        *mode = BenchMode::Descriptor;
        return true;
    }
    if (value == "compact-tail") {
        *mode = BenchMode::CompactTail;
        return true;
    }
    if (value == "fused-tail") {
        *mode = BenchMode::FusedTail;
        return true;
    }
    if (value == "fused-sa-hash-tail") {
        *mode = BenchMode::FusedSaHashTail;
        return true;
    }
    if (value == "fused-hash-tail") {
        *mode = BenchMode::FusedHashTail;
        return true;
    }
    if (value == "both") {
        *mode = BenchMode::Both;
        return true;
    }
    std::fprintf(stderr, "unknown --mode: %s\n", value.c_str());
    return false;
}

static uint32_t read_u32_le(const uint8_t* in) {
    return static_cast<uint32_t>(in[0]) |
           (static_cast<uint32_t>(in[1]) << 8) |
           (static_cast<uint32_t>(in[2]) << 16) |
           (static_cast<uint32_t>(in[3]) << 24);
}

static void write_u32_le(uint8_t* out, uint32_t v) {
    out[0] = static_cast<uint8_t>(v);
    out[1] = static_cast<uint8_t>(v >> 8);
    out[2] = static_cast<uint8_t>(v >> 16);
    out[3] = static_cast<uint8_t>(v >> 24);
}

static void sha256_bytes(const uint8_t* data, size_t len, uint8_t out[32]) {
    SHA256(data, len, out);
}

static bool parse_stage4_input(const std::vector<uint8_t>& input,
                               Stage4InputView* view) {
    static constexpr char kStage4Magic[8] = {'D', 'L', 'S', '4', 'I', 'N', 0, 0};
    if (!view || input.size() < sizeof(Stage4InputHeader)) return false;

    Stage4InputHeader h{};
    std::memcpy(&h, input.data(), sizeof(h));
    if (std::memcmp(h.magic, kStage4Magic, sizeof(h.magic)) != 0) return false;
    if (h.version != 1) return false;
    if (h.logical_len == 0 || h.logical_len > 0x20000u ||
        h.data_len < h.logical_len) {
        return false;
    }

    const uint64_t need = static_cast<uint64_t>(sizeof(Stage4InputHeader)) +
                          h.flag_len + h.data_len;
    if (need != input.size()) return false;
    if (h.flag_len <= (h.logical_len >> 8)) return false;

    view->logical_len = h.logical_len;
    view->data_len = h.data_len;
    view->flag_len = h.flag_len;
    view->flags = input.data() + sizeof(Stage4InputHeader);
    view->data = view->flags + h.flag_len;
    return true;
}

static bool build_stage4_libsais_expected(const Sample& sample,
                                          std::vector<uint8_t>* expected) {
    Stage4InputView view;
    if (!parse_stage4_input(sample.stage4_input, &view)) {
        std::fprintf(stderr, "missing/unreadable Stage 4 input hash_seq=%llu\n",
                     static_cast<unsigned long long>(sample.hash_seq));
        return false;
    }
    std::vector<int32_t> sa(view.logical_len);
    if (libsais(view.data, sa.data(), static_cast<int32_t>(view.logical_len),
                0, nullptr) != 0) {
        return false;
    }
    expected->resize(static_cast<size_t>(view.logical_len) * 4u);
    for (uint32_t i = 0; i < view.logical_len; ++i) {
        write_u32_le(expected->data() + static_cast<size_t>(i) * 4u,
                     static_cast<uint32_t>(sa[i]));
    }
    return true;
}

static bool parse_args(int argc, char** argv, Args* args) {
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto need_value = [&](const char* name, std::string* out) -> bool {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s requires a value\n", name);
                return false;
            }
            *out = argv[++i];
            return true;
        };

        if (s == "--captures") {
            if (!need_value("--captures", &args->captures)) return false;
        } else if (s == "--max") {
            std::string v;
            if (!need_value("--max", &v)) return false;
            args->max_records = static_cast<size_t>(std::stoull(v));
        } else if (s == "--repeat") {
            std::string v;
            if (!need_value("--repeat", &v)) return false;
            args->repeat = std::stoi(v);
            if (args->repeat <= 0) {
                std::fprintf(stderr, "--repeat must be positive\n");
                return false;
            }
        } else if (s == "--no-verify") {
            args->verify = false;
        } else if (s == "--mode") {
            std::string v;
            if (!need_value("--mode", &v)) return false;
            if (!parse_mode(v, &args->mode)) return false;
        } else if (s == "--help" || s == "-h") {
            usage();
            return false;
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", s.c_str());
            return false;
        }
    }

    if (args->captures.empty()) {
        std::fprintf(stderr, "--captures is required\n");
        return false;
    }
    return true;
}

static bool load_samples(const Args& args, std::vector<Sample>* samples) {
    std::vector<fs::path> caps;
    for (const auto& e : fs::directory_iterator(args.captures)) {
        if (e.is_regular_file() && e.path().extension() == ".cap") {
            caps.push_back(e.path());
        }
    }
    std::sort(caps.begin(), caps.end());

    for (const auto& cap_path : caps) {
        if (args.max_records && samples->size() >= args.max_records) break;

        CaptureReader reader;
        if (!reader.open(cap_path.string().c_str())) {
            std::fprintf(stderr, "skip unreadable capture: %s\n",
                         cap_path.string().c_str());
            continue;
        }

        Sample sample;
        sample.hash_seq = reader.hash_seq();
        bool have_stage5 = false;
        for (uint32_t idx = 0; idx < reader.stage_count(); ++idx) {
            StageRecord rec;
            if (!reader.read_stage(idx, &rec)) continue;
            if (rec.stage_id == StageId::V114Encode) {
                sample.stage4_input.assign(rec.in_bytes, rec.in_bytes + rec.in_len);
            } else if (rec.stage_id == StageId::V114SaBuild) {
                sample.input.assign(rec.in_bytes, rec.in_bytes + rec.in_len);
                sample.expected.assign(rec.out_bytes, rec.out_bytes + rec.out_len);
                have_stage5 = true;
            }
        }
        if (have_stage5) {
            samples->push_back(std::move(sample));
        }
    }

    if (samples->empty()) {
        std::fprintf(stderr, "no Stage 5 records found in %s\n",
                     args.captures.c_str());
        return false;
    }
    return true;
}

struct SplitTiming {
    double encode_us = 0.0;
    double sa_us = 0.0;
    double fused_us = 0.0;
    double hash_us = 0.0;
};

static bool run_one_compact_tail(const Sample& sample, std::vector<uint8_t>* out,
                                 size_t* out_len, SplitTiming* timing) {
    Stage4InputView view;
    if (!parse_stage4_input(sample.stage4_input, &view)) {
        std::fprintf(stderr, "missing/unreadable Stage 4 input hash_seq=%llu\n",
                     static_cast<unsigned long long>(sample.hash_seq));
        return false;
    }

    static thread_local std::vector<uint8_t> desc;
    static thread_local std::vector<uint8_t> arena;
    const size_t desc_cap = static_cast<size_t>(view.logical_len) * 8u;
    const size_t arena_cap = static_cast<size_t>(view.logical_len) * 4u;
    if (desc.size() < desc_cap) desc.resize(desc_cap);
    if (arena.size() < arena_cap) arena.resize(arena_cap);
    size_t desc_len = 0;
    size_t arena_len = 0;

    const auto enc0 = std::chrono::steady_clock::now();
    const bool enc_ok = deroluna::stages::v114::stage_v114_encode_compact_raw(
        view.data, view.logical_len, view.data_len,
        view.flags, view.flag_len,
        desc.data(), desc.size(), &desc_len,
        arena.data(), arena.size(), &arena_len);
    const auto enc1 = std::chrono::steady_clock::now();
    if (!enc_ok) return false;

    if (out->size() < sample.expected.size()) {
        out->resize(sample.expected.size());
    }
    *out_len = 0;
    const auto sa0 = std::chrono::steady_clock::now();
    const bool sa_ok = deroluna::stages::v114::stage_v114_sa_build_descriptor_trusted_raw(
        view.data, view.logical_len, view.data_len,
        arena.data(), static_cast<uint32_t>(arena_len),
        desc.data(), static_cast<uint32_t>(desc_len),
        out->data(), out->size(), out_len);
    const auto sa1 = std::chrono::steady_clock::now();

    if (timing) {
        timing->encode_us += std::chrono::duration<double, std::micro>(enc1 - enc0).count();
        timing->sa_us += std::chrono::duration<double, std::micro>(sa1 - sa0).count();
    }
    return sa_ok;
}

static bool run_one_fused_tail(const Sample& sample, std::vector<uint8_t>* out,
                               size_t* out_len, SplitTiming* timing) {
    Stage4InputView view;
    if (!parse_stage4_input(sample.stage4_input, &view)) {
        std::fprintf(stderr, "missing/unreadable Stage 4 input hash_seq=%llu\n",
                     static_cast<unsigned long long>(sample.hash_seq));
        return false;
    }

    if (out->size() < sample.expected.size()) {
        out->resize(sample.expected.size());
    }
    *out_len = 0;

    const auto fused0 = std::chrono::steady_clock::now();
    const bool fused_ok = deroluna::stages::v114::stage_v114_sa_build_compact_fused_raw(
        view.data, view.logical_len, view.data_len,
        view.flags, view.flag_len,
        out->data(), out->size(), out_len);
    const auto fused1 = std::chrono::steady_clock::now();

    if (timing) {
        timing->fused_us +=
            std::chrono::duration<double, std::micro>(fused1 - fused0).count();
    }
    return fused_ok;
}

static bool run_one_fused_hash_tail(const Sample& sample,
                                    std::vector<uint8_t>* out,
                                    size_t* out_len,
                                    SplitTiming* timing) {
    Stage4InputView view;
    if (!parse_stage4_input(sample.stage4_input, &view)) {
        std::fprintf(stderr, "missing/unreadable Stage 4 input hash_seq=%llu\n",
                     static_cast<unsigned long long>(sample.hash_seq));
        return false;
    }

    if (out->size() < 32u) {
        out->resize(32u);
    }
    *out_len = 0;

    const auto hash0 = std::chrono::steady_clock::now();
    const bool hash_ok = deroluna::stages::v114::stage_v114_hash_compact_fused_raw(
        view.data, view.logical_len, view.data_len,
        view.flags, view.flag_len,
        out->data());
    const auto hash1 = std::chrono::steady_clock::now();

    if (!hash_ok) return false;
    *out_len = 32u;
    if (timing) {
        timing->hash_us +=
            std::chrono::duration<double, std::micro>(hash1 - hash0).count();
    }
    return true;
}

static bool run_one_fused_sa_hash_tail(const Sample& sample,
                                       std::vector<uint8_t>* out,
                                       size_t* out_len,
                                       SplitTiming* timing) {
    static thread_local std::vector<uint8_t> sa;
    size_t sa_len = 0;
    SplitTiming inner;
    if (!run_one_fused_tail(sample, &sa, &sa_len, &inner)) {
        return false;
    }

    if (out->size() < 32u) {
        out->resize(32u);
    }
    const auto hash0 = std::chrono::steady_clock::now();
    sha256_bytes(sa.data(), sa_len, out->data());
    const auto hash1 = std::chrono::steady_clock::now();
    *out_len = 32u;

    if (timing) {
        timing->fused_us += inner.fused_us;
        timing->hash_us +=
            std::chrono::duration<double, std::micro>(hash1 - hash0).count();
    }
    return true;
}

static bool run_one(const Sample& sample, std::vector<uint8_t>* out,
                    size_t* out_len, BenchMode mode, SplitTiming* timing = nullptr) {
    if (mode == BenchMode::CompactTail) {
        return run_one_compact_tail(sample, out, out_len, timing);
    }
    if (mode == BenchMode::FusedTail) {
        return run_one_fused_tail(sample, out, out_len, timing);
    }
    if (mode == BenchMode::FusedSaHashTail) {
        return run_one_fused_sa_hash_tail(sample, out, out_len, timing);
    }
    if (mode == BenchMode::FusedHashTail) {
        return run_one_fused_hash_tail(sample, out, out_len, timing);
    }
    if (out->size() < sample.expected.size()) {
        out->resize(sample.expected.size());
    }
    *out_len = 0;
    return deroluna::stages::v114::stage_v114_sa_build_with_mode(
        sample.input.data(), sample.input.size(), out->data(), out->size(), out_len,
        stage_mode(mode));
}

static void report_mismatch(const Sample& sample, const std::vector<uint8_t>& out,
                            size_t out_len, BenchMode mode) {
    std::fprintf(stderr,
                 "Stage 5 mismatch mode=%s hash_seq=%llu out_len=%zu expected=%zu\n",
                 mode_name(mode), static_cast<unsigned long long>(sample.hash_seq),
                 out_len, sample.expected.size());

    const size_t common = std::min(out_len, sample.expected.size());
    for (size_t i = 0; i < common; ++i) {
        if (out[i] == sample.expected[i]) continue;
        const size_t sa_index = i / 4u;
        uint32_t actual_pos = 0xffffffffu;
        uint32_t expected_pos = 0xffffffffu;
        if ((sa_index + 1u) * 4u <= out_len) {
            actual_pos = read_u32_le(out.data() + sa_index * 4u);
        }
        if ((sa_index + 1u) * 4u <= sample.expected.size()) {
            expected_pos = read_u32_le(sample.expected.data() + sa_index * 4u);
        }
        std::fprintf(stderr,
                     "first_diff_byte=%zu sa_index=%zu actual_pos=%u expected_pos=%u\n",
                     i, sa_index, actual_pos, expected_pos);
        return;
    }
    std::fprintf(stderr, "common prefix matched for %zu bytes; lengths differ\n",
                 common);
}

static bool verify_samples(const std::vector<Sample>& samples, BenchMode mode) {
    std::vector<uint8_t> out;
    std::vector<uint8_t> stage4_expected;
    std::vector<uint8_t> fused_sa;
    uint8_t fused_digest[32] = {};
    for (const auto& sample : samples) {
        size_t out_len = 0;
        bool ok = run_one(sample, &out, &out_len, mode);
        if (!ok) {
            std::fprintf(stderr, "Stage 5 failed mode=%s hash_seq=%llu\n",
                         mode_name(mode),
                         static_cast<unsigned long long>(sample.hash_seq));
            return false;
        }
        const std::vector<uint8_t>* expected = &sample.expected;
        if (mode == BenchMode::CompactTail || mode == BenchMode::FusedTail) {
            if (!build_stage4_libsais_expected(sample, &stage4_expected)) {
                return false;
            }
            expected = &stage4_expected;
        } else if (mode == BenchMode::FusedSaHashTail ||
                   mode == BenchMode::FusedHashTail) {
            size_t fused_len = 0;
            if (!run_one_fused_tail(sample, &fused_sa, &fused_len, nullptr)) {
                return false;
            }
            sha256_bytes(fused_sa.data(), fused_len, fused_digest);
            stage4_expected.assign(fused_digest, fused_digest + 32u);
            expected = &stage4_expected;
        }
        if (out_len != expected->size() ||
            std::memcmp(out.data(), expected->data(), out_len) != 0) {
            if (mode == BenchMode::CompactTail || mode == BenchMode::FusedTail) {
                Sample tmp = sample;
                tmp.expected = *expected;
                report_mismatch(tmp, out, out_len, mode);
            } else {
                report_mismatch(sample, out, out_len, mode);
            }
            return false;
        }
    }
    return true;
}

static bool verify_both_modes_equal(const std::vector<Sample>& samples) {
    std::vector<uint8_t> libsais_out;
    std::vector<uint8_t> descriptor_out;
    for (const auto& sample : samples) {
        size_t libsais_len = 0;
        size_t descriptor_len = 0;
        bool libsais_ok = run_one(sample, &libsais_out, &libsais_len, BenchMode::Libsais);
        bool descriptor_ok = run_one(sample, &descriptor_out, &descriptor_len,
                                     BenchMode::Descriptor);
        if (!libsais_ok || !descriptor_ok) {
            std::fprintf(stderr,
                         "Stage 5 both-mode failed hash_seq=%llu libsais_ok=%d descriptor_ok=%d\n",
                         static_cast<unsigned long long>(sample.hash_seq),
                         libsais_ok ? 1 : 0, descriptor_ok ? 1 : 0);
            return false;
        }
        const size_t common = std::min(libsais_len, descriptor_len);
        if (libsais_len != descriptor_len ||
            std::memcmp(libsais_out.data(), descriptor_out.data(), common) != 0) {
            std::fprintf(stderr,
                         "Stage 5 both-mode mismatch hash_seq=%llu libsais_len=%zu descriptor_len=%zu\n",
                         static_cast<unsigned long long>(sample.hash_seq),
                         libsais_len, descriptor_len);
            for (size_t i = 0; i < common; ++i) {
                if (libsais_out[i] == descriptor_out[i]) continue;
                const size_t sa_index = i / 4u;
                std::fprintf(stderr,
                             "first_diff_byte=%zu sa_index=%zu libsais_pos=%u descriptor_pos=%u\n",
                             i, sa_index,
                             read_u32_le(libsais_out.data() + sa_index * 4u),
                             read_u32_le(descriptor_out.data() + sa_index * 4u));
                return false;
            }
            std::fprintf(stderr,
                         "common prefix matched for %zu bytes; lengths differ\n",
                         common);
            return false;
        }
    }
    return true;
}

struct BenchResult {
    double total_us = 0.0;
    double us_per_call = 0.0;
    double encode_us_per_call = 0.0;
    double sa_us_per_call = 0.0;
    double fused_us_per_call = 0.0;
    double hash_us_per_call = 0.0;
    double mib_per_s = 0.0;
    uint64_t checksum = 0;
    size_t calls = 0;
};

static bool bench_mode(const std::vector<Sample>& samples, const Args& args,
                       BenchMode mode, BenchResult* result) {
    std::vector<uint8_t> out;
    SplitTiming split;
    result->checksum = 1469598103934665603ull;
    size_t total_bytes = 0;
    const auto start = std::chrono::steady_clock::now();
    for (int r = 0; r < args.repeat; ++r) {
        for (const auto& sample : samples) {
            size_t out_len = 0;
            bool ok = run_one(sample, &out, &out_len, mode, &split);
            if (!ok) {
                std::fprintf(stderr, "Stage 5 failed mode=%s hash_seq=%llu\n",
                             mode_name(mode),
                             static_cast<unsigned long long>(sample.hash_seq));
                return false;
            }
            const size_t expected_len = (mode == BenchMode::CompactTail ||
                                         mode == BenchMode::FusedTail ||
                                         mode == BenchMode::FusedSaHashTail ||
                                         mode == BenchMode::FusedHashTail)
                ? out_len
                : sample.expected.size();
            if (out_len != expected_len) {
                report_mismatch(sample, out, out_len, mode);
                return false;
            }
            if (mode == BenchMode::FusedSaHashTail ||
                mode == BenchMode::FusedHashTail) {
                Stage4InputView view;
                if (!parse_stage4_input(sample.stage4_input, &view)) {
                    return false;
                }
                total_bytes += static_cast<size_t>(view.logical_len) * 4u;
            } else {
                total_bytes += out_len;
            }
            for (size_t i = 0; i < out_len; ++i) {
                const uint8_t b = out[i];
                result->checksum ^= b;
                result->checksum *= 1099511628211ull;
            }
        }
    }
    const auto end = std::chrono::steady_clock::now();

    result->total_us =
        std::chrono::duration<double, std::micro>(end - start).count();
    result->calls = samples.size() * static_cast<size_t>(args.repeat);
    result->us_per_call = result->total_us / static_cast<double>(result->calls);
    result->encode_us_per_call = split.encode_us / static_cast<double>(result->calls);
    result->sa_us_per_call = split.sa_us / static_cast<double>(result->calls);
    result->fused_us_per_call = split.fused_us / static_cast<double>(result->calls);
    result->hash_us_per_call = split.hash_us / static_cast<double>(result->calls);
    result->mib_per_s =
        (static_cast<double>(total_bytes) / (1024.0 * 1024.0)) /
        (result->total_us / 1000000.0);
    return true;
}

static void print_result(const std::vector<Sample>& samples, const Args& args,
                         BenchMode mode, const BenchResult& result) {
    std::printf(
        "bench_stage5 mode=%s records=%zu repeat=%d calls=%zu verify=%s "
        "total_us=%.0f us_per_call=%.2f",
        mode_name(mode), samples.size(), args.repeat, result.calls,
        args.verify ? "yes" : "no",
        result.total_us, result.us_per_call);
    if (mode == BenchMode::CompactTail) {
        std::printf(" encode_us_per_call=%.2f sa_us_per_call=%.2f",
                    result.encode_us_per_call, result.sa_us_per_call);
    } else if (mode == BenchMode::FusedTail) {
        std::printf(" fused_us_per_call=%.2f", result.fused_us_per_call);
    } else if (mode == BenchMode::FusedSaHashTail) {
        std::printf(" fused_us_per_call=%.2f hash_us_per_call=%.2f",
                    result.fused_us_per_call, result.hash_us_per_call);
    } else if (mode == BenchMode::FusedHashTail) {
        std::printf(" hash_us_per_call=%.2f", result.hash_us_per_call);
    }
    std::printf(" mib_per_s=%.2f checksum=%016llx\n",
                result.mib_per_s,
                static_cast<unsigned long long>(result.checksum));
}

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, &args)) return 2;

    std::vector<Sample> samples;
    if (!load_samples(args, &samples)) return 1;

    const bool run_libsais = args.mode == BenchMode::Libsais || args.mode == BenchMode::Both;
    const bool run_descriptor = args.mode == BenchMode::Descriptor || args.mode == BenchMode::Both;
    const bool run_compact_tail = args.mode == BenchMode::CompactTail;
    const bool run_fused_tail = args.mode == BenchMode::FusedTail;
    const bool run_fused_sa_hash_tail = args.mode == BenchMode::FusedSaHashTail;
    const bool run_fused_hash_tail = args.mode == BenchMode::FusedHashTail;

    if (args.verify) {
        if (run_libsais && !verify_samples(samples, BenchMode::Libsais)) return 1;
        if (run_descriptor && !verify_samples(samples, BenchMode::Descriptor)) return 1;
        if (run_compact_tail && !verify_samples(samples, BenchMode::CompactTail)) return 1;
        if (run_fused_tail && !verify_samples(samples, BenchMode::FusedTail)) return 1;
        if (run_fused_sa_hash_tail && !verify_samples(samples, BenchMode::FusedSaHashTail)) return 1;
        if (run_fused_hash_tail && !verify_samples(samples, BenchMode::FusedHashTail)) return 1;
        if (run_libsais && run_descriptor && !verify_both_modes_equal(samples)) return 1;
    }

    BenchResult libsais_result;
    BenchResult descriptor_result;
    BenchResult compact_tail_result;
    BenchResult fused_tail_result;
    BenchResult fused_sa_hash_tail_result;
    BenchResult fused_hash_tail_result;
    if (run_libsais) {
        if (!bench_mode(samples, args, BenchMode::Libsais, &libsais_result)) return 1;
        print_result(samples, args, BenchMode::Libsais, libsais_result);
    }
    if (run_descriptor) {
        if (!bench_mode(samples, args, BenchMode::Descriptor, &descriptor_result)) return 1;
        print_result(samples, args, BenchMode::Descriptor, descriptor_result);
    }
    if (run_compact_tail) {
        if (!bench_mode(samples, args, BenchMode::CompactTail, &compact_tail_result)) return 1;
        print_result(samples, args, BenchMode::CompactTail, compact_tail_result);
    }
    if (run_fused_tail) {
        if (!bench_mode(samples, args, BenchMode::FusedTail, &fused_tail_result)) return 1;
        print_result(samples, args, BenchMode::FusedTail, fused_tail_result);
    }
    if (run_fused_sa_hash_tail) {
        if (!bench_mode(samples, args, BenchMode::FusedSaHashTail, &fused_sa_hash_tail_result)) return 1;
        print_result(samples, args, BenchMode::FusedSaHashTail, fused_sa_hash_tail_result);
    }
    if (run_fused_hash_tail) {
        if (!bench_mode(samples, args, BenchMode::FusedHashTail, &fused_hash_tail_result)) return 1;
        print_result(samples, args, BenchMode::FusedHashTail, fused_hash_tail_result);
    }
    if (run_libsais && run_descriptor) {
        if (libsais_result.checksum != descriptor_result.checksum) {
            std::fprintf(stderr,
                         "bench_stage5 checksum mismatch libsais=%016llx descriptor=%016llx\n",
                         static_cast<unsigned long long>(libsais_result.checksum),
                         static_cast<unsigned long long>(descriptor_result.checksum));
            return 1;
        }
        if (descriptor_result.us_per_call > 0.0) {
            std::printf("bench_stage5 descriptor_vs_libsais_speedup=%.3fx\n",
                        libsais_result.us_per_call / descriptor_result.us_per_call);
        }
    }
    return 0;
}
