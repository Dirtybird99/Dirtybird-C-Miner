#pragma once
#include <string>
#include <algorithm>
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

static const char *DIRTYBIRD = R"(
                                                            YB&&@5
                                              :GBB7         &@@@@P
                          .:^~7J5J.           !@@@@@Y.      #@@@@P
            ..:~!?Y5GB#&@@@@@@@@@@@5.         ~@@@@@@@B^    #@@@@G
  ~7J5GB#&&@@@@@@@@@@@@@@@@@@@@@@@@@@P.       ~@@@@@@@@@&7  #@@@@G
  @@@@@@@@@@@@@@@@@@@&#BG5J7!^5@@@@@@@@G:     ~@@@@@&@@@@@@Y&@@@@B
  @@@@@&#BGPY?~G@@@@&         7@@@@@@@@@@B^   ^@@@@@:.5@@@@@@@@@@B
  ::.          7@@@@&         7@@@@@&@@@@@@#~ :@@@@@:   ?&@@@@@@@B
               7@@@@@         !@@@@& ^B@@@@@@#5@@@@@^     ~#@@@@@#
               7@@@@@         !@@@@@   :B@@@@@@@@@@@^       :G@@@#
               !@@@@@         ~@@@@@.    .P@@@@@@@@@~         .Y@&
               !@@@@@         ~@@@@@.      .5@@@@@@@~            ^
               ~@@@@@.        ^@@@@@.        .J@@@@@~
               ~@@@@@.        ^@@@@@.           ?&@@!
               ~@@@@@.        ^@@@@@.             !&!
               ^@@@@@.        :@@@@@.
               ^@@@@@.        :@@@@@:
               ^@@@@@.         ^G@@@:      ██ ██ █ █   █ ████ ████
               :@@@@@:           .J&:      █████ █ ██  █ █    █  █
               :@@@@@:                     █ █ █ █ ███ █ ███  ████
               :@@@&Y                      █ █ █ █ █ ███ █    █ █
               .&Y:                        █   █ █ █  ██ ████ █ ██
)";

static const char *DERO = R"(
                              @
                         @@       @@
                     @@               @@
                 @                         @
             @                                 @@
        @@                    @                    @@
    @                    @@       @@                    @
    @                @@       .       @@                @
    @            @@       ..     ..       @@            @
    @          @      .               .      @          @
    @          @   .       @@@@@@@       .   @          @
    @          @   .    @@@@@@@@@@@@@    .   @          @
    @          @   .    @@@@@@@@@@@@@    .   @          @
    @          @   .    @@@@@@@@@@@@@    .   @          @
    @          @   .     @@@@@@@@@@@     .   @          @
    @          @    ..     @@@@@@@     ..    @          @
    @          @@        .@@@@@@@@@.        @@          @
    @              @@    @@@@@@@@@@@    @@              @
    @                  @@@@@@@@@@@@@@@                  @
       @                    @@@@@                    @@
           @@                                    @
               @@                           @@
                   @@                   @@
                        @@         @@
                              @
)";

static const char* coinPrompt = "Enter coin (DERO): ";
static const char* daemonPrompt = "Node address: ";
static const char* portPrompt = "Port: ";
static const char* walletPrompt = "Wallet: ";

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
        ("broadcast", "HTTP server for stats")
        ("testnet", "Use testnet params")
        ("daemon-address", po::value<std::string>(), "Node/pool address")
        ("port", po::value<int>(), "Connection port")
        ("wallet", po::value<std::string>(), "Wallet address")
        ("threads", po::value<int>(), "Mining threads (default: 1)")
        ("report-interval", po::value<int>(), "Status interval (seconds)")
        ("no-lock", "Disable CPU affinity")
        ("p-cores-only", "P-cores only (hybrid CPUs)")
        ("differential-affinity", po::value<int>()->default_value(0), "Affinity mode: 0=default, 1=P-first, 2=physical, 3=balanced")
        ("show-affinity", "Show CPU assignments")
        ("ignore-wallet", "Skip wallet validation");
    po::options_description s("Stratum", w);
    s.add_options()
        ("stratum", "Enable stratum mode")
        ("password", po::value<std::string>(), "Stratum password")
        ("worker-name", po::value<std::string>(), "Worker name");
    po::options_description d("DERO", w);
    d.add_options()("dero", "Mine DERO (default)");
    po::options_description t("Testing", w);
    t.add_options()("test-dero", "Run AstroBWTv3 tests");
    po::options_description a("Advanced", w);
    a.add_options()
        ("tune-warmup", po::value<int>()->default_value(1), "Warmup seconds")
        ("tune-duration", po::value<int>()->default_value(2), "Tune duration per algo")
        ("no-tune", po::value<std::string>(), "Skip tuning: branch|lookup|avx2|wolf|aarch64")
        ("mine-time", po::value<int>()->default_value(0), "Mine duration (0=infinite)")
        ("no-spsa", "Disable SPSA")
        ("verbose-tune", "Verbose tuning output")
        ("cache-batch", "Cache-batched mining")
        ("cache-batch-hybrid", "Auto-detect batch mode")
        ("bench-cache-batch", "Benchmark batch sizes")
        ("interleaved", "Two-miners-per-thread")
        ("bench-interleaved", "Benchmark interleaved")
        ("lockfree", "Lock-free job polling")
        ("sa-tune", "SA prefetch autotuning")
        ("sa-prefetch", po::value<std::string>(), "SA prefetch: sa,text,bucket")
        ("no-sa-tune", "Disable SA tuning")
        ("omp-threads", po::value<int>()->default_value(0), "OpenMP threads (0=auto)");
    po::options_description db("Debug", w);
    db.add_options()
        ("op", po::value<int>(), "Branch op to benchmark")
        ("len", po::value<int>(), "Benchmark chunk length")
        ("sabench", "SA benchmark")
        ("quiet", "Quiet mode");
    g.add(s); g.add(d); g.add(t); g.add(a); g.add(db);
    return g;
}

inline int get_prog_style() {
    return po::command_line_style::unix_style | po::command_line_style::allow_long_disguise;
}
