#include "reporter.hpp"
#include <numeric>
#include <iostream>

static const char* units[] = {" ", " K", " M", " G", " T", " P"};

#ifdef DIRTYBIRD_HIP
extern std::atomic<bool> g_mining_started;
extern bool beQuiet;
#endif

int update_handler(const boost::system::error_code& error) {
    CHECK_CLOSE_RET(0);
    if (error == boost::asio::error::operation_aborted) return 1;

    using clock = std::chrono::steady_clock;
    static clock::time_point next_tick = clock::now() + std::chrono::seconds(1);
    next_tick += std::chrono::seconds(1);
    update_timer.expires_at(next_tick);
    update_timer.async_wait(update_handler);

    if (!isConnected) return 1;

    reportCounter++;
    auto now = clock::now();

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

    uint64_t h = counter.exchange(0);
    if (rate30sec.size() > 30) rate30sec.erase(rate30sec.begin());
    rate30sec.push_back(h);

    double hr = std::accumulate(rate30sec.begin(), rate30sec.end(), 0LL) / (double)rate30sec.size();
    int u = 0;
    while (hr >= 1000 && u < 5) { u++; hr /= 1000.0; }
    latest_hashrate = hr;

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
        setcolor(CYAN); std::cout << "Acc:" << accepted;
        if (rejected > 0) { setcolor(BRIGHT_RED); std::cout << " Rej:" << rejected; }
        setcolor(BRIGHT_WHITE); std::cout << " | ";
        setcolor(MAGENTA); std::cout << "Diff:" << diff;
        setcolor(BRIGHT_WHITE); std::cout << " | ";
        setcolor(BLUE); std::cout << "#" << ourHeight;
        setcolor(BRIGHT_WHITE); std::cout << " | ";
        setcolor(WHITE); printf("%02d:%02d:%02d", (int)(t/3600), (int)(t%3600/60), (int)(t%60));
        setcolor(BRIGHT_WHITE);
        std::cout << std::flush;

        reportCounter = 0;
    }
    return 0;
}
