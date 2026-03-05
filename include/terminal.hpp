#pragma once
#include <string>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <boost/program_options.hpp>
#if defined(__linux__)
#include <sys/ioctl.h>
#endif
#if defined(_WIN32)
#include <Windows.h>
#endif

namespace po = boost::program_options;

#define XSTR(x) STR(x)
#define STR(x) #x
#ifdef _WIN32
#define RUN_EXTENSION ".exe"
#define SCRIPT_EXTENSION ".bat"
#else
#define RUN_EXTENSION ""
#define SCRIPT_EXTENSION ".sh"
#endif

static const char *versionString = XSTR(DIRTYBIRD_VERSION);
static const char *consoleLine = " DIRTYBIRD-MINER ";
static const char *targetArch = XSTR(CPU_ARCHTARGET);

static int colorPreTable[] = {0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1};
static int colorTable[] = {30,31,32,33,34,35,36,37,90,91,92,93,94,95,96,97};

#define BLACK 0
#define RED 1
#define GREEN 2
#define YELLOW 3
#define BLUE 4
#define MAGENTA 5
#define CYAN 6
#define WHITE 7
#define BRIGHT_BLACK 8
#define BRIGHT_RED 9
#define BRIGHT_GREEN 10
#define BRIGHT_YELLOW 11
#define BRIGHT_BLUE 12
#define BRIGHT_MAGENTA 13
#define BRIGHT_CYAN 14
#define BRIGHT_WHITE 15

#if defined(_WIN32)
static WORD winColorTable[] = {
    0, FOREGROUND_RED, FOREGROUND_GREEN, FOREGROUND_RED|FOREGROUND_GREEN,
    FOREGROUND_BLUE, FOREGROUND_RED|FOREGROUND_BLUE, FOREGROUND_GREEN|FOREGROUND_BLUE,
    FOREGROUND_RED|FOREGROUND_GREEN|FOREGROUND_BLUE, FOREGROUND_INTENSITY,
    FOREGROUND_RED|FOREGROUND_INTENSITY, FOREGROUND_GREEN|FOREGROUND_INTENSITY,
    FOREGROUND_RED|FOREGROUND_GREEN|FOREGROUND_INTENSITY, FOREGROUND_BLUE|FOREGROUND_INTENSITY,
    FOREGROUND_RED|FOREGROUND_BLUE|FOREGROUND_INTENSITY, FOREGROUND_GREEN|FOREGROUND_BLUE|FOREGROUND_INTENSITY,
    FOREGROUND_RED|FOREGROUND_GREEN|FOREGROUND_BLUE|FOREGROUND_INTENSITY
};
inline void setcolor(int c) { SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), winColorTable[c]); }
#else
inline void setcolor(int c) { printf("\e[%d;%dm", colorPreTable[c], colorTable[c]); }
#endif

inline std::string getTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    auto timer = std::chrono::system_clock::to_time_t(now);
    std::tm bt;
#if defined(_WIN32)
    localtime_s(&bt, &timer);
#else
    localtime_r(&timer, &bt);
#endif
    std::ostringstream oss;
    oss << std::setfill('0') << std::setw(2) << bt.tm_mday << "/"
        << std::setfill('0') << std::setw(2) << (bt.tm_mon + 1) << " "
        << std::setfill('0') << std::setw(2) << bt.tm_hour << ":"
        << std::setfill('0') << std::setw(2) << bt.tm_min << ":"
        << std::setfill('0') << std::setw(2) << bt.tm_sec << "."
        << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

inline void logInfo(const std::string& msg) {
    printf("%s  INFO  %s\n", getTimestamp().c_str(), msg.c_str());
    fflush(stdout);
}

inline void logInfoPair(const std::string& label, const std::string& value) {
    printf("%s  INFO  %-8s %s\n", getTimestamp().c_str(), (label + ":").c_str(), value.c_str());
    fflush(stdout);
}

inline po::options_description get_prog_opts() {
    int w = 80;
#if defined(__linux__)
    try { struct winsize ws; ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws); w = ws.ws_col; } catch (...) {}
#endif
    po::options_description g("General", w);
    g.add_options()
        ("help", "Show help")
        ("broadcast", "Enable HTTP stats broadcast server")
        ("daemon-address", po::value<std::string>(), "Node/pool address")
        ("port", po::value<int>(), "Connection port")
        ("wallet", po::value<std::string>(), "Wallet address")
        ("threads", po::value<int>(), "Mining threads (default: 1)")
        ("report-interval", po::value<int>(), "Status update interval in seconds")
        ("no-lock", "Disable CPU affinity")
        ("p-cores-only", "P-cores only (hybrid CPUs)")
        ("differential-affinity", po::value<int>()->default_value(0), "Affinity: 0=default, 1=P-first, 2=physical, 3=balanced")
        ("show-affinity", "Display detailed CPU affinity assignments")
        ("ignore-wallet", "Skip wallet validation checks")
        ("gpu", "Mine with GPU instead of CPU")
        ("batch-size", po::value<int>(), "(GPU) Batch size")
        ("quiet", "Quiet mode");
    po::options_description s("Stratum", w);
    s.add_options()
        ("stratum", "Enable stratum mode")
        ("password", po::value<std::string>(), "Stratum password")
        ("worker-name", po::value<std::string>(), "Worker name");
    po::options_description a("Advanced", w);
    a.add_options()
        ("tune-warmup", po::value<int>()->default_value(1), "Warmup seconds")
        ("tune-duration", po::value<int>()->default_value(2), "Tune duration per algo")
        ("no-tune", po::value<std::string>(), "Skip tuning: branch|lookup|avx2|wolf|wolf_memopt|aarch64")
        ("auto-tune", "Run startup kernel autotune (legacy behavior)")
        ("mine-time", po::value<int>()->default_value(0), "Mine duration (0=infinite)")
        ("lookup", "Force lookup compute path")
        ("lookup-mode", po::value<std::string>()->default_value("auto"), "Lookup table mode: auto|1d|3d|full|hybrid|smart")
        ("lookup-smart-threshold", po::value<int>()->default_value(12), "Smart lookup span threshold (0..32, default 12)")
        ("lookup-smart-telemetry", "Enable smart lookup path telemetry")
        ("spsa-stamp-fast", po::value<std::string>()->default_value("on"), "SPSA decode fast path: on|off")
        ("spsa-decode-bases", po::value<std::string>()->default_value("off"), "SPSA decode layout: on=use stamp base offsets, off=use start chunks")
        ("spsa-bucket-prefetch", po::value<std::string>()->default_value("off"), "SPSA bucket prefetch policy: off|light|full")
        ("spsa-hit-counters", po::value<std::string>()->default_value("on"), "SPSA hit/miss atomic counters: on|off")
        ("spsa-sha-profile", po::value<std::string>()->default_value("off"), "SPSA SHA call-family telemetry: on|off")
        ("spsa-sha-zeroize", po::value<std::string>()->default_value("on"), "Zero SHA256 context in Final: on|off")
        ("runtime-parity", po::value<std::string>()->default_value("off"), "Runtime parity preset: off|deroluna")
        ("no-spsa", "Disable SPSA")
        ("verbose-tune", "Print detailed autotune results")
        ("cache-batch", "Enable cache-focused batched mining")
        ("cache-batch-hybrid", "Auto-select between standard and cache-batched mining")
        ("bench-cache-batch", "Run cache-batching benchmark and exit")
        ("omp-threads", po::value<int>()->default_value(0), "OpenMP threads (0=auto)")
        ("sa-tune", "Enable SA prefetch autotuning")
        ("sa-prefetch", po::value<std::string>(), "Set SA prefetch tuple manually (sa,text,bucket)")
        ("no-sa-tune", "Disable SA prefetch autotuning")
        ("adaptive", po::value<int>()->implicit_value(0), "Over-provision threads, cull slowest after warmup (0=auto 2x)")
        ("adaptive-warmup", po::value<int>()->default_value(15), "Adaptive warmup seconds before culling")
        ("pace-interval", po::value<int>()->default_value(0), "Duty-cycle pacer: hashes between sleeps (0=off)")
        ("pace-sleep-us", po::value<int>()->default_value(0), "Duty-cycle pacer: microseconds to sleep")
        ("interleaved", "2 miners per thread (DeroLuna ILP)")
        ("lockfree", "Lock-free mining (Go-style thread coordination)")
        ("bench-interleaved", "Benchmark interleaved vs standard")
        ("array-telemetry", "Print array-path telemetry (kernel/lookup/SA/SPSA stats)")
        ("phase-telemetry", "Print per-phase AstroBWT telemetry (prep/wolf/SPSA/SA/hash)")
        ("print-runtime-config", "Always print resolved runtime config at startup")
        ("no-runtime-config", "Disable runtime config startup print")
        ("no-power-override", "Disable all power management overrides (RAPL, HWP, EPP, power plan)");
    po::options_description t("Testing", w);
    t.add_options()
        ("test-dero", "Run AstroBWTv3 tests")
        ("sabench", "SA benchmark")
        ("op", po::value<int>(), "Debug: force specific branch op in test path")
        ("len", po::value<int>(), "Debug: force specific branch length in test path");
    g.add(s); g.add(a); g.add(t);
    return g;
}

inline int get_prog_style() {
    return po::command_line_style::unix_style | po::command_line_style::allow_long_disguise;
}
