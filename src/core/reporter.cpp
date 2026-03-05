#include "reporter.hpp"
#include "thermal_governor.hpp"
#include <astrobwtv3/astrobwtv3.h>
#include <lookup_mode.hpp>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cmath>

static const char* units[] = {" ", " K", " M", " G", " T", " P"};

static const char* lookup_mode_name(LookupMode mode) {
    switch (mode) {
        case LOOKUP_MODE_1D: return "1d";
        case LOOKUP_MODE_3D: return "3d";
        case LOOKUP_MODE_FULL: return "full";
        case LOOKUP_MODE_HYBRID: return "hybrid";
        case LOOKUP_MODE_SMART: return "smart";
        default: return "unknown";
    }
}

#ifdef DIRTYBIRD_HIP
extern std::atomic<bool> g_mining_started;
extern bool beQuiet;
#endif

int update_handler(const boost::system::error_code& error) {
    CHECK_CLOSE_RET(0);
    if (error == boost::asio::error::operation_aborted) return 1;

    auto schedule_next_tick = []() {
        // Schedule from handler exit time so stalled handlers do not queue immediate
        // catch-up callbacks that spam logs and consume extra CPU.
        update_timer.expires_after(std::chrono::seconds(1));
        update_timer.async_wait(update_handler);
    };

    if (!isConnected) {
        schedule_next_tick();
        return 1;
    }

    reportCounter++;
    auto now = std::chrono::steady_clock::now();

#ifdef DIRTYBIRD_HIP
    bool any_hashing = false;
    std::vector<double> gpu_hr(HIP_deviceCount);
    std::vector<int> gpu_unit(HIP_deviceCount);

    if (gpuMine) {
        bool prev_quiet = beQuiet;
        beQuiet = true;
        for (int i = 0; i < HIP_deviceCount; i++) {
            uint64_t h = HIP_counters[i].exchange(0);
            if (h > 0) { any_hashing = true; beQuiet = prev_quiet; }
            if (HIP_rates1min[i].size() > 60) HIP_rates1min[i].erase(HIP_rates1min[i].begin());
            HIP_rates1min[i].push_back(h);
            double hr = std::accumulate(HIP_rates1min[i].begin(), HIP_rates1min[i].end(), 0LL) / (double)HIP_rates1min[i].size();
            int u = 0;
            while (hr >= 1000 && u < 5) { u++; hr /= 1000.0; }
            gpu_hr[i] = hr;
            gpu_unit[i] = u;
        }
    }
#endif

    static auto last_counter_sample_time = now;
    const auto current_sample_time = now;
    double elapsed_sample_sec = std::chrono::duration<double>(current_sample_time - last_counter_sample_time).count();
    if (elapsed_sample_sec <= 0.0) {
        elapsed_sample_sec = 1.0;
    }
    last_counter_sample_time = current_sample_time;

    uint64_t h = counter.exchange(0);
    const uint64_t h_per_sec = static_cast<uint64_t>(std::llround(static_cast<double>(h) / elapsed_sample_sec));
    if (rate30sec.size() > 30) rate30sec.erase(rate30sec.begin());
    rate30sec.push_back(h_per_sec);

    double hr = std::accumulate(rate30sec.begin(), rate30sec.end(), 0LL) / (double)rate30sec.size();
    int u = 0;
    while (hr >= 1000 && u < 5) { u++; hr /= 1000.0; }
    latest_hashrate = hr;

    // Thermal governor: adjust pacing based on hashrate trend
    int duty = thermal::update(rate30sec);

#ifdef DIRTYBIRD_HIP
    if (!g_mining_started.load()) {
        if (any_hashing || hr > 0.001) {
            g_mining_started.store(true);
            printf("\n[DIRTYBIRD] Mining started\n");
            fflush(stdout);
        } else return 0;
    }
#endif

    if (reportCounter >= reportInterval) {
#ifdef DIRTYBIRD_HIP
        if (gpuMine) {
            setcolor(BRIGHT_YELLOW);
            for (int i = 0; i < HIP_deviceCount; i++)
                printf("\n[GPU %d] %s | %.2f%sH/s", i, HIP_names[i].c_str(), gpu_hr[i], units[gpu_unit[i]]);
            fflush(stdout);
            setcolor(BRIGHT_WHITE);
        }
#endif
        std::cout << (gpuMine ? "\n" : "\r");

        auto t = std::chrono::duration_cast<std::chrono::seconds>(now - g_start_time).count();
        char diff[16];
        if (difficulty >= 1000000000) snprintf(diff, 16, "%lluG", difficulty / 1000000000);
        else if (difficulty >= 1000000) snprintf(diff, 16, "%lluM", difficulty / 1000000);
        else if (difficulty >= 1000) snprintf(diff, 16, "%lluK", difficulty / 1000);
        else snprintf(diff, 16, "%llu", difficulty);

        setcolor(BRIGHT_YELLOW); std::cout << "[DIRTYBIRD] ";
        setcolor(BRIGHT_GREEN); std::cout << std::fixed << std::setprecision(2) << hr << units[u] << "H/s";
        setcolor(BRIGHT_WHITE); std::cout << " | ";
        setcolor(CYAN); std::cout << "MB:" << accepted;
        setcolor(BRIGHT_WHITE); std::cout << " ";
        setcolor(GREEN); std::cout << "IB:" << blockCounter;
        setcolor(BRIGHT_WHITE); std::cout << " ";
        if (rejected > 0) setcolor(BRIGHT_RED); else setcolor(WHITE);
        std::cout << "REJ:" << rejected;
        setcolor(BRIGHT_WHITE); std::cout << " | ";
        setcolor(MAGENTA); std::cout << "Diff:" << diff;
        setcolor(BRIGHT_WHITE); std::cout << " | ";
        setcolor(BLUE); std::cout << "#" << ourHeight;
        setcolor(BRIGHT_WHITE); std::cout << " | ";
        setcolor(WHITE); printf("%02d:%02d:%02d", (int)(t/3600), (int)(t%3600/60), (int)(t%60));
        if (duty < 100) {
            setcolor(BRIGHT_WHITE); std::cout << " | ";
            setcolor(YELLOW); printf("TG:%d%%", duty);
        }
        setcolor(BRIGHT_WHITE);
        std::cout << std::flush;

        static int64_t last_array_telemetry_sec = -15;
        if (g_array_telemetry && (t - last_array_telemetry_sec >= 15)) {
            const uint64_t hits = getSPSAHits();
            const uint64_t misses = getSPSAMisses();
            const uint64_t total = hits + misses;
            const double hit_rate = (total > 0) ? (100.0 * static_cast<double>(hits) / static_cast<double>(total)) : 0.0;

            std::cout << "\n";
            setcolor(BRIGHT_YELLOW); std::cout << "[ARRAY] ";
            setcolor(BRIGHT_WHITE);
            std::cout << "kernel=" << getCurrentAstroAlgoName()
                      << " lookup=" << lookup_mode_name(g_lookup_mode)
                      << " sa=" << getSABackendName()
                      << " spsa=" << (g_use_spsa ? "on" : "off")
                      << " hit=" << std::fixed << std::setprecision(1) << hit_rate << "%";
            if (total > 0) {
                std::cout << " (" << hits << "/" << total << ")";
            }
            std::cout << std::flush;
            last_array_telemetry_sec = t;
        }

        static int64_t last_lookup_telemetry_sec = -15;
        static AstroLookupTelemetrySnapshot last_lookup_snapshot{};
        if (g_lookup_smart_telemetry &&
            g_lookup_mode == LOOKUP_MODE_SMART &&
            (t - last_lookup_telemetry_sec >= 15)) {
            const AstroLookupTelemetrySnapshot current = getLookupTelemetrySnapshot();
            auto delta = [](uint64_t cur, uint64_t prev) -> uint64_t {
                return (cur >= prev) ? (cur - prev) : cur;
            };

            AstroLookupTelemetrySnapshot window{};
            window.smart_branched_total = delta(current.smart_branched_total, last_lookup_snapshot.smart_branched_total);
            window.smart_path_lut = delta(current.smart_path_lut, last_lookup_snapshot.smart_path_lut);
            window.smart_path_avx2 = delta(current.smart_path_avx2, last_lookup_snapshot.smart_path_avx2);
            for (size_t i = 0; i < 33; ++i) {
                window.smart_span_hist[i] = delta(current.smart_span_hist[i], last_lookup_snapshot.smart_span_hist[i]);
            }

            const uint64_t total_branched = window.smart_branched_total;
            const double lut_pct = (total_branched > 0)
                ? (100.0 * static_cast<double>(window.smart_path_lut) / static_cast<double>(total_branched))
                : 0.0;
            const double avx2_pct = (total_branched > 0)
                ? (100.0 * static_cast<double>(window.smart_path_avx2) / static_cast<double>(total_branched))
                : 0.0;

            auto sum_hist = [&window](size_t from, size_t to) -> uint64_t {
                uint64_t sum = 0;
                for (size_t i = from; i <= to; ++i) {
                    sum += window.smart_span_hist[i];
                }
                return sum;
            };

            const uint64_t span0 = window.smart_span_hist[0];
            const uint64_t span_1_4 = sum_hist(1, 4);
            const uint64_t span_5_8 = sum_hist(5, 8);
            const uint64_t span_9_16 = sum_hist(9, 16);
            const uint64_t span_17_24 = sum_hist(17, 24);
            const uint64_t span_25_32 = sum_hist(25, 32);

            std::cout << "\n";
            setcolor(BRIGHT_YELLOW); std::cout << "[LOOKUP] ";
            setcolor(BRIGHT_WHITE);
            std::cout << "smart_thr=" << g_lookup_smart_threshold
                      << " branched=" << total_branched
                      << " lut=" << window.smart_path_lut << " (" << std::fixed << std::setprecision(1) << lut_pct << "%)"
                      << " avx2=" << window.smart_path_avx2 << " (" << std::fixed << std::setprecision(1) << avx2_pct << "%)"
                      << " span0=" << span0
                      << " s1_4=" << span_1_4
                      << " s5_8=" << span_5_8
                      << " s9_16=" << span_9_16
                      << " s17_24=" << span_17_24
                      << " s25_32=" << span_25_32
                      << std::flush;

            last_lookup_snapshot = current;
            last_lookup_telemetry_sec = t;
        }

        static int64_t last_phase_telemetry_sec = -15;
        static AstroPhaseTelemetrySnapshot last_phase_snapshot{};
        if (isPhaseTelemetryEnabled() && (t - last_phase_telemetry_sec >= 15)) {
            const AstroPhaseTelemetrySnapshot current = getPhaseTelemetrySnapshot();
            auto delta = [](uint64_t cur, uint64_t prev) -> uint64_t {
                return (cur >= prev) ? (cur - prev) : cur;
            };

            AstroPhaseTelemetrySnapshot window{};
            window.hashes = delta(current.hashes, last_phase_snapshot.hashes);
            window.data_len_sum = delta(current.data_len_sum, last_phase_snapshot.data_len_sum);
            window.spsa_hits = delta(current.spsa_hits, last_phase_snapshot.spsa_hits);
            window.spsa_misses = delta(current.spsa_misses, last_phase_snapshot.spsa_misses);
            window.prep_ns = delta(current.prep_ns, last_phase_snapshot.prep_ns);
            window.wolf_ns = delta(current.wolf_ns, last_phase_snapshot.wolf_ns);
            window.spsa_call_ns = delta(current.spsa_call_ns, last_phase_snapshot.spsa_call_ns);
            window.spsa_prefetch_ns = delta(current.spsa_prefetch_ns, last_phase_snapshot.spsa_prefetch_ns);
            window.spsa_hit_copy_ns = delta(current.spsa_hit_copy_ns, last_phase_snapshot.spsa_hit_copy_ns);
            window.spsa_miss_hash_ns = delta(current.spsa_miss_hash_ns, last_phase_snapshot.spsa_miss_hash_ns);
            window.spsa_core_ns = delta(current.spsa_core_ns, last_phase_snapshot.spsa_core_ns);
            window.spsa_core_calls = delta(current.spsa_core_calls, last_phase_snapshot.spsa_core_calls);
            window.spsa_core_bin0_calls = delta(current.spsa_core_bin0_calls, last_phase_snapshot.spsa_core_bin0_calls);
            window.spsa_core_bin0_ns = delta(current.spsa_core_bin0_ns, last_phase_snapshot.spsa_core_bin0_ns);
            window.spsa_core_bin1_calls = delta(current.spsa_core_bin1_calls, last_phase_snapshot.spsa_core_bin1_calls);
            window.spsa_core_bin1_ns = delta(current.spsa_core_bin1_ns, last_phase_snapshot.spsa_core_bin1_ns);
            window.spsa_core_bin2_calls = delta(current.spsa_core_bin2_calls, last_phase_snapshot.spsa_core_bin2_calls);
            window.spsa_core_bin2_ns = delta(current.spsa_core_bin2_ns, last_phase_snapshot.spsa_core_bin2_ns);
            window.spsa_core_bin3_calls = delta(current.spsa_core_bin3_calls, last_phase_snapshot.spsa_core_bin3_calls);
            window.spsa_core_bin3_ns = delta(current.spsa_core_bin3_ns, last_phase_snapshot.spsa_core_bin3_ns);
            window.spsa_op_family_nonbranched_calls = delta(current.spsa_op_family_nonbranched_calls, last_phase_snapshot.spsa_op_family_nonbranched_calls);
            window.spsa_op_family_nonbranched_bytes = delta(current.spsa_op_family_nonbranched_bytes, last_phase_snapshot.spsa_op_family_nonbranched_bytes);
            window.spsa_op_family_branched_calls = delta(current.spsa_op_family_branched_calls, last_phase_snapshot.spsa_op_family_branched_calls);
            window.spsa_op_family_branched_bytes = delta(current.spsa_op_family_branched_bytes, last_phase_snapshot.spsa_op_family_branched_bytes);
            window.spsa_op_family_op253_calls = delta(current.spsa_op_family_op253_calls, last_phase_snapshot.spsa_op_family_op253_calls);
            window.spsa_op_family_op253_bytes = delta(current.spsa_op_family_op253_bytes, last_phase_snapshot.spsa_op_family_op253_bytes);
            window.spsa_op_family_rc4_calls = delta(current.spsa_op_family_rc4_calls, last_phase_snapshot.spsa_op_family_rc4_calls);
            window.spsa_op_family_rc4_bytes = delta(current.spsa_op_family_rc4_bytes, last_phase_snapshot.spsa_op_family_rc4_bytes);
            window.sa_fallback_ns = delta(current.sa_fallback_ns, last_phase_snapshot.sa_fallback_ns);
            window.final_hash_ns = delta(current.final_hash_ns, last_phase_snapshot.final_hash_ns);
            window.total_ns = delta(current.total_ns, last_phase_snapshot.total_ns);
            window.sa_encode_ns = delta(current.sa_encode_ns, last_phase_snapshot.sa_encode_ns);
            window.sa_radix_ns = delta(current.sa_radix_ns, last_phase_snapshot.sa_radix_ns);
            window.sa_collision_ns = delta(current.sa_collision_ns, last_phase_snapshot.sa_collision_ns);
            window.sa_copy_ns = delta(current.sa_copy_ns, last_phase_snapshot.sa_copy_ns);

            if (window.hashes > 0) {
                const double denom = static_cast<double>(window.hashes) * 1000.0;
                const double prep_us = static_cast<double>(window.prep_ns) / denom;
                const double wolf_us = static_cast<double>(window.wolf_ns) / denom;
                const double spsa_us = static_cast<double>(window.spsa_call_ns) / denom;
                const double spsa_prefetch_us = static_cast<double>(window.spsa_prefetch_ns) / denom;
                const double spsa_hit_copy_us = static_cast<double>(window.spsa_hit_copy_ns) / denom;
                const double spsa_miss_hash_us = static_cast<double>(window.spsa_miss_hash_ns) / denom;
                const double sa_us = static_cast<double>(window.sa_fallback_ns) / denom;
                const double hash_us = static_cast<double>(window.final_hash_ns) / denom;
                const double total_us = static_cast<double>(window.total_ns) / denom;
                const uint64_t total_spsa = window.spsa_hits + window.spsa_misses;
                const double hit_rate = (total_spsa > 0)
                    ? (100.0 * static_cast<double>(window.spsa_hits) / static_cast<double>(total_spsa))
                    : 0.0;
                const double avg_data_len = static_cast<double>(window.data_len_sum) / static_cast<double>(window.hashes);
                const double sa_encode_us = static_cast<double>(window.sa_encode_ns) / denom;
                const double sa_radix_us = static_cast<double>(window.sa_radix_ns) / denom;
                const double sa_collision_us = static_cast<double>(window.sa_collision_ns) / denom;
                const double sa_copy_us = static_cast<double>(window.sa_copy_ns) / denom;

                std::cout << "\n";
                setcolor(BRIGHT_YELLOW); std::cout << "[PHASE] ";
                setcolor(BRIGHT_WHITE);
                std::cout << "hashes=" << window.hashes
                          << " len_avg=" << std::fixed << std::setprecision(1) << avg_data_len
                          << " prep_us=" << std::fixed << std::setprecision(2) << prep_us
                          << " wolf_us=" << std::fixed << std::setprecision(2) << wolf_us
                          << " spsa_us=" << std::fixed << std::setprecision(2) << spsa_us
                          << " spsa_pref_us=" << std::fixed << std::setprecision(2) << spsa_prefetch_us
                          << " spsa_hitcpy_us=" << std::fixed << std::setprecision(2) << spsa_hit_copy_us
                          << " spsa_misshash_us=" << std::fixed << std::setprecision(2) << spsa_miss_hash_us
                          << " sa_us=" << std::fixed << std::setprecision(2) << sa_us
                          << " hash_us=" << std::fixed << std::setprecision(2) << hash_us
                          << " total_us=" << std::fixed << std::setprecision(2) << total_us
                          << " hit=" << std::fixed << std::setprecision(1) << hit_rate << "%"
                          << " sa_enc_us=" << std::fixed << std::setprecision(2) << sa_encode_us
                          << " sa_radix_us=" << std::fixed << std::setprecision(2) << sa_radix_us
                          << " sa_coll_us=" << std::fixed << std::setprecision(2) << sa_collision_us
                          << " sa_copy_us=" << std::fixed << std::setprecision(2) << sa_copy_us
                          << std::flush;

                if (window.spsa_core_calls > 0) {
                    const double core_us = static_cast<double>(window.spsa_core_ns) / (static_cast<double>(window.spsa_core_calls) * 1000.0);
                    const double b0_us = (window.spsa_core_bin0_calls > 0)
                        ? (static_cast<double>(window.spsa_core_bin0_ns) / (static_cast<double>(window.spsa_core_bin0_calls) * 1000.0))
                        : 0.0;
                    const double b1_us = (window.spsa_core_bin1_calls > 0)
                        ? (static_cast<double>(window.spsa_core_bin1_ns) / (static_cast<double>(window.spsa_core_bin1_calls) * 1000.0))
                        : 0.0;
                    const double b2_us = (window.spsa_core_bin2_calls > 0)
                        ? (static_cast<double>(window.spsa_core_bin2_ns) / (static_cast<double>(window.spsa_core_bin2_calls) * 1000.0))
                        : 0.0;
                    const double b3_us = (window.spsa_core_bin3_calls > 0)
                        ? (static_cast<double>(window.spsa_core_bin3_ns) / (static_cast<double>(window.spsa_core_bin3_calls) * 1000.0))
                        : 0.0;

                    std::cout << "\n";
                    setcolor(BRIGHT_YELLOW); std::cout << "[SPSA-CORE] ";
                    setcolor(BRIGHT_WHITE);
                    std::cout << "calls=" << window.spsa_core_calls
                              << " core_us=" << std::fixed << std::setprecision(2) << core_us
                              << " b0_calls=" << window.spsa_core_bin0_calls
                              << " b0_us=" << std::fixed << std::setprecision(2) << b0_us
                              << " b1_calls=" << window.spsa_core_bin1_calls
                              << " b1_us=" << std::fixed << std::setprecision(2) << b1_us
                              << " b2_calls=" << window.spsa_core_bin2_calls
                              << " b2_us=" << std::fixed << std::setprecision(2) << b2_us
                              << " b3_calls=" << window.spsa_core_bin3_calls
                              << " b3_us=" << std::fixed << std::setprecision(2) << b3_us
                              << std::flush;
                }

                const uint64_t op_calls_total =
                    window.spsa_op_family_nonbranched_calls +
                    window.spsa_op_family_branched_calls +
                    window.spsa_op_family_op253_calls +
                    window.spsa_op_family_rc4_calls;
                const uint64_t op_bytes_total =
                    window.spsa_op_family_nonbranched_bytes +
                    window.spsa_op_family_branched_bytes +
                    window.spsa_op_family_op253_bytes +
                    window.spsa_op_family_rc4_bytes;
                if (op_calls_total > 0) {
                    auto pct = [op_calls_total](uint64_t calls) -> double {
                        return 100.0 * static_cast<double>(calls) / static_cast<double>(op_calls_total);
                    };
                    auto avg_span = [](uint64_t bytes, uint64_t calls) -> double {
                        return (calls > 0) ? (static_cast<double>(bytes) / static_cast<double>(calls)) : 0.0;
                    };

                    std::cout << "\n";
                    setcolor(BRIGHT_YELLOW); std::cout << "[SPSA-OPS] ";
                    setcolor(BRIGHT_WHITE);
                    std::cout << "calls=" << op_calls_total
                              << " bytes=" << op_bytes_total
                              << " nb=" << window.spsa_op_family_nonbranched_calls
                              << " (" << std::fixed << std::setprecision(1) << pct(window.spsa_op_family_nonbranched_calls) << "%)"
                              << " nb_span=" << std::fixed << std::setprecision(2) << avg_span(window.spsa_op_family_nonbranched_bytes, window.spsa_op_family_nonbranched_calls)
                              << " br=" << window.spsa_op_family_branched_calls
                              << " (" << std::fixed << std::setprecision(1) << pct(window.spsa_op_family_branched_calls) << "%)"
                              << " br_span=" << std::fixed << std::setprecision(2) << avg_span(window.spsa_op_family_branched_bytes, window.spsa_op_family_branched_calls)
                              << " op253=" << window.spsa_op_family_op253_calls
                              << " (" << std::fixed << std::setprecision(1) << pct(window.spsa_op_family_op253_calls) << "%)"
                              << " op253_span=" << std::fixed << std::setprecision(2) << avg_span(window.spsa_op_family_op253_bytes, window.spsa_op_family_op253_calls)
                              << " rc4=" << window.spsa_op_family_rc4_calls
                              << " (" << std::fixed << std::setprecision(1) << pct(window.spsa_op_family_rc4_calls) << "%)"
                              << " rc4_span=" << std::fixed << std::setprecision(2) << avg_span(window.spsa_op_family_rc4_bytes, window.spsa_op_family_rc4_calls)
                              << std::flush;
                }
            }

            last_phase_snapshot = current;
            last_phase_telemetry_sec = t;
        }

        reportCounter = 0;
    }
    schedule_next_tick();
    return 0;
}
