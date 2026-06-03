// report.cpp - Markdown, JSONL, and binary diff-triplet output.

#include "report.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace deroluna::replay {
namespace fs = std::filesystem;

static int first_diverge(const uint8_t* a, const uint8_t* b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) return static_cast<int>(i);
    }
    return -1;
}

static void write_diff_triplet(const fs::path& dir, StageId id, uint64_t hash_seq,
                               const uint8_t* in, size_t in_len,
                               const uint8_t* clone_out, size_t clone_out_len,
                               const uint8_t* cap_out, size_t cap_out_len) {
    fs::create_directories(dir);
    std::ostringstream prefix;
    prefix << "stage" << static_cast<int>(id)
           << "_h" << std::setw(6) << std::setfill('0') << hash_seq;

    auto write_blob = [&](const char* tag, const uint8_t* data, size_t n) {
        fs::path path = dir / (prefix.str() + "_" + tag + ".bin");
        std::FILE* file = std::fopen(path.string().c_str(), "wb");
        if (!file) return;
        if (n != 0 && data) std::fwrite(data, 1, n, file);
        std::fclose(file);
    };

    write_blob("in", in, in_len);
    write_blob("clone_out", clone_out, clone_out_len);
    write_blob("v114_out", cap_out, cap_out_len);
}

void StageStats::record_one(uint64_t hash_seq, uint32_t in_len,
                            uint32_t cap_out_len,
                            bool fn_ok, size_t clone_out_len,
                            const uint8_t* clone_out, const uint8_t* cap_out,
                            const uint8_t* cap_in) {
    coverage++;

    bool matched = false;
    bool out_len_mismatch = false;
    int fd = -1;
    if (!fn_ok) {
        fn_returned_false++;
        mismatches++;
    } else if (clone_out_len != cap_out_len) {
        out_len_mismatches++;
        mismatches++;
        out_len_mismatch = true;
        size_t common = std::min(clone_out_len, static_cast<size_t>(cap_out_len));
        fd = first_diverge(clone_out, cap_out, common);
        if (fd < 0) fd = static_cast<int>(common);
    } else {
        fd = first_diverge(clone_out, cap_out, clone_out_len);
        if (fd < 0) {
            exact_matches++;
            matched = true;
        } else {
            mismatches++;
            first_diverge_sum += static_cast<uint64_t>(fd);
            first_diverge_n++;
        }
    }

    if (!matched && diffs_written < max_diffs) {
        write_diff_triplet(diffs_dir, id, hash_seq,
                           cap_in, in_len,
                           clone_out, clone_out_len,
                           cap_out, cap_out_len);
        diffs_written++;
    }

    observations.push_back(StageObservation{
        hash_seq,
        in_len,
        cap_out_len,
        clone_out_len,
        fn_ok,
        matched,
        out_len_mismatch,
        fd,
    });
}

static int recommended_next(const StageStats* stats, int stage_count) {
    for (int i = 0; i < stage_count; ++i) {
        if (stats[i].is_stub || stats[i].coverage == 0) continue;
        if (stats[i].exact_matches != stats[i].coverage) {
            return static_cast<int>(stats[i].id);
        }
    }
    return 0;
}

void write_report(const std::string& out_dir,
                  const std::string& caps_dir,
                  const StageStats* stats,
                  int stage_count) {
    fs::create_directories(out_dir);
    fs::create_directories(fs::path(out_dir) / "diffs");
    fs::path md_path = fs::path(out_dir) / "report.md";
    fs::path jsonl_path = fs::path(out_dir) / "report.jsonl";

    auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &now);
#else
    gmtime_r(&now, &tm);
#endif
    char ts_buf[32];
    std::strftime(ts_buf, sizeof(ts_buf), "%Y-%m-%dT%H:%MZ", &tm);

    std::ofstream md(md_path);
    md << "# v1.14 Parity Sweep Report - " << ts_buf << "\n\n"
       << "Captures: " << caps_dir << "\n\n"
       << "## Stage divergence (worst first)\n\n"
       << "| Stage | Name | Coverage | Match rate | Avg first diverge | Notes |\n"
       << "|-------|------|---------:|-----------:|------------------:|-------|\n";

    std::vector<int> order(stage_count);
    for (int i = 0; i < stage_count; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (stats[a].is_stub != stats[b].is_stub) return stats[a].is_stub;
        double ma = stats[a].coverage
            ? static_cast<double>(stats[a].exact_matches) / stats[a].coverage
            : 1.0;
        double mb = stats[b].coverage
            ? static_cast<double>(stats[b].exact_matches) / stats[b].coverage
            : 1.0;
        return ma < mb;
    });

    for (int idx : order) {
        const auto& s = stats[idx];
        double rate = s.coverage
            ? 100.0 * static_cast<double>(s.exact_matches) / s.coverage
            : 0.0;
        double avg_fd = s.first_diverge_n
            ? static_cast<double>(s.first_diverge_sum) / s.first_diverge_n
            : 0.0;
        const char* note = "";
        if (s.is_stub) note = "clone fn stub - expected";
        else if (s.coverage == 0) note = "no v1.14 ground truth available";

        md << "| " << static_cast<int>(s.id)
           << " | " << stage_name(s.id)
           << " | " << s.coverage
           << " | " << std::fixed << std::setprecision(2) << rate << " %"
           << " | " << std::fixed << std::setprecision(0) << avg_fd
           << " | " << note << " |\n";
    }

    int rec = recommended_next(stats, stage_count);
    md << "\n## Recommended next stage to attack: ";
    if (rec == 0) {
        md << "none - all non-stub stages match\n";
    } else {
        md << rec << " (" << stage_name(stats[rec - 1].id) << ")\n";
    }
    md.close();

    std::ofstream jl(jsonl_path);
    for (int i = 0; i < stage_count; ++i) {
        const auto& s = stats[i];
        for (const auto& obs : s.observations) {
            jl << "{\"stage\":" << static_cast<int>(s.id)
               << ",\"name\":\"" << stage_name(s.id) << "\""
               << ",\"hash_seq\":" << obs.hash_seq
               << ",\"match\":" << (obs.match ? "true" : "false")
               << ",\"in_len\":" << obs.in_len
               << ",\"out_len\":" << obs.cap_out_len
               << ",\"clone_out_len\":" << obs.clone_out_len
               << ",\"fn_ok\":" << (obs.fn_ok ? "true" : "false")
               << ",\"out_len_mismatch\":"
               << (obs.out_len_mismatch ? "true" : "false")
               << ",\"is_stub\":" << (s.is_stub ? "true" : "false")
               << ",\"first_diverge\":";
            if (obs.first_diverge < 0) {
                jl << "null";
            } else {
                jl << obs.first_diverge;
            }
            jl << "}\n";
        }
    }
}

}  // namespace deroluna::replay
