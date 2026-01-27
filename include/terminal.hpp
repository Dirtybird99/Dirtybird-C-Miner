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

namespace po = boost::program_options;  // from <boost/program_options.hpp>

// Get current timestamp string in DD/MM HH:MM:SS.mmm format
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

// Print an INFO log line with timestamp
inline void logInfo(const std::string& message) {
    printf("%s  INFO  %s\n", getTimestamp().c_str(), message.c_str());
    fflush(stdout);
}

// Print an INFO log line with label and value
inline void logInfoPair(const std::string& label, const std::string& value) {
    printf("%s  INFO  %-8s %s\n", getTimestamp().c_str(), (label + ":").c_str(), value.c_str());
    fflush(stdout);
}

// macro tricks so we can use a string to set DIRTYBIRD_VERSION
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

static const char* coinPrompt = "Please enter the symbol for the coin you'd like to mine (DERO)";
static const char* daemonPrompt = "Please enter your mining deamon/host address: ";
static const char* portPrompt = "Please enter your mining port: ";
static const char* walletPrompt = "Please enter your wallet address for mining rewards: ";
static const char* threadPrompt = "Please provide the desired amount of mining threads: ";

static const char* inputIntro = "Please provide your mining settings (leave fields blank to use defaults)";

static int colorPreTable[] = {
    0,   0,   0,   0,   0,   0,   0,   0,     1,   1,   1,   1,   1,   1,   1,   1
};
static int colorTable[] = {
    30,  31,  32,  33,  34,  35,  36,  37,    90,  91,  92,  93,  94,  95,  96,  97
};

#define BLACK           0
#define RED             1
#define GREEN           2
#define YELLOW          3
#define BLUE            4
#define MAGENTA         5
#define CYAN            6
#define WHITE           7
#define BRIGHT_BLACK    8
#define BRIGHT_RED      9
#define BRIGHT_GREEN    10
#define BRIGHT_YELLOW   11
#define BRIGHT_BLUE     12
#define BRIGHT_MAGENTA  13
#define BRIGHT_CYAN     14
#define BRIGHT_WHITE    15

#if defined(_WIN32)
static WORD winColorTable[] = {
    0,                                                                      // BLACK
    FOREGROUND_RED,                                                         // RED
    FOREGROUND_GREEN,                                                       // GREEN
    FOREGROUND_RED | FOREGROUND_GREEN,                                      // YELLOW
    FOREGROUND_BLUE,                                                        // BLUE
    FOREGROUND_RED | FOREGROUND_BLUE,                                       // MAGENTA
    FOREGROUND_GREEN | FOREGROUND_BLUE,                                     // CYAN
    FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE,                    // WHITE
    FOREGROUND_INTENSITY,                                                   // BRIGHT_BLACK
    FOREGROUND_RED | FOREGROUND_INTENSITY,                                  // BRIGHT_RED
    FOREGROUND_GREEN | FOREGROUND_INTENSITY,                                // BRIGHT_GREEN
    FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_INTENSITY,               // BRIGHT_YELLOW
    FOREGROUND_BLUE | FOREGROUND_INTENSITY,                                 // BRIGHT_BLUE
    FOREGROUND_RED | FOREGROUND_BLUE | FOREGROUND_INTENSITY,                // BRIGHT_MAGENTA
    FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY,              // BRIGHT_CYAN
    FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY  // BRIGHT_WHITE
};

inline void setcolor(int color)
{
    SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), winColorTable[color]);
}
#else
inline void setcolor(int color)
{
    printf("\e[%d;%dm", colorPreTable[color], colorTable[color]);
}
#endif

inline po::options_description get_prog_opts()
{
  int col_width = 80;
  #if defined(__linux__)
    try {
      struct winsize w;
      ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
      col_width=w.ws_col;
    }
    catch (...)
    {
    }
  #endif

  po::options_description general("General", col_width);
  general.add_options()
    ("help", "Produce help message")
    ("broadcast", "Creates an http server to query miner stats")
    ("testnet", "Adjusts in-house parameters to mine on testnets")
    ("daemon-address", po::value<std::string>(), "Node/pool URL or IP address to mine to") // todo: parse out port and/or wss:// or ws://
    ("port", po::value<int>(), "The port used to connect to the node")
    ("wallet", po::value<std::string>(), "Wallet address for receiving mining rewards")
    ("threads", po::value<int>(), "The amount of mining threads to create, default is 1")
    ("report-interval", po::value<int>(), "Your desired status update interval in seconds")
    ("no-lock", "Disables CPU affinity / CPU core binding")
    ("p-cores-only", "Limit mining threads to P-cores only (for hybrid CPUs like i7-13700HX)")
    ("differential-affinity", po::value<int>()->default_value(0),
        "P-core vs E-core affinity strategy for hybrid CPUs:\n"
        "  0 = Default (P-primaries -> E-cores -> P-HT)\n"
        "  1 = P-cores first (P-primaries -> P-HT -> E-cores)\n"
        "  2 = Physical first (P-primaries -> E-cores only, no HT)\n"
        "  3 = Balanced (interleave P and E cores)")
    ("show-affinity", "Display detailed CPU affinity assignments")
    ("ignore-wallet", "Disables wallet validation, for specific uses with pool mining")
    // ("gpu", "Mine with GPU instead of CPU")
    // ("batch-size", po::value<int>(), "(GPU Setting) Sets batch size used for GPU mining")
  ;

  po::options_description stratum("Stratum", col_width);
  stratum.add_options()
    ("stratum", "Required for Stratum pools if not using 'stratum+tcp://' or 'stratum+ssl://' in the daemon url")
    ("password", po::value<std::string>(), "Sets the Stratum password")
    ("worker-name", po::value<std::string>(), "Sets the worker name for this instance when mining on Pools or Bridges")
  ;

  po::options_description dero("DERO (AstroBWTv3)", col_width);
  dero.add_options()
    ("dero", "Mine DERO using AstroBWTv3 algorithm (default)")
  ;

  po::options_description testing("Testing", col_width);
  testing.add_options()
    ("test-dero", "Runs a set of tests to verify AstrobwtV3 is working (1 test expected to fail)")
  ;

  po::options_description advanced("Advanced", col_width);
  advanced.add_options()
    ("tune-warmup", po::value<int>()->default_value(1), "Number of seconds to warmup the CPU before starting the AstroBWTv3 tuning")
    ("tune-duration", po::value<int>()->default_value(2), "Number of seconds to tune *each* AstroBWTv3 algorithm. There will 3 or 4 algorithms depending on supported CPU features")
    ("no-tune", po::value<std::string>(), "<branch|lookup|avx2|wolf|aarch64> Use the specified AstroBWTv3 algorithm and skip tuning")
    ("mine-time", po::value<int>()->default_value(0), "Mine for a given number of seconds and then exit")
    ("no-spsa", "Disable SPSA optimization (for benchmarking against TNN)")
    ("verbose-tune", "Show detailed autotune results for each algorithm")
    ("cache-batch", "Enable cache-focused batched mining (k1's optimization) - keeps SA code hot in L1I cache")
    ("cache-batch-hybrid", "Auto-detect best mining mode via benchmark (cache-batched vs sequential)")
    ("bench-cache-batch", "Run cache-batching benchmark to determine optimal batch size for this CPU")
    ("interleaved", "Enable two-miners-per-thread (DeroLuna-style ILP) - hides L3 cache latency")
    ("bench-interleaved", "Run interleaved vs standard benchmark to measure ILP benefit")
    ("lockfree", "Enable lock-free job polling (Go-style thread coordination) - reduces mutex contention")
    ("sa-tune", "Enable SA prefetch autotuning to find optimal prefetch distances for this CPU")
    ("sa-prefetch", po::value<std::string>(), "Set SA prefetch distances manually: sa,text,bucket (e.g., '16,24,8')")
    ("no-sa-tune", "Disable SA autotuning (uses default prefetch distances)")
    ("omp-threads", po::value<int>()->default_value(0), "OpenMP threads per mining thread for parallel SA (0=auto, like TNN/Luna)")
  ;

  po::options_description debug("DEBUG", col_width);
  debug.add_options()
    ("op", po::value<int>(), "Sets which branch op to benchmark (0-255), benchmark will be skipped if unspecified")
    ("len", po::value<int>(), "Sets length of the processed chunk in said benchmark (default 15)")
    ("sabench", "Runs a benchmark for divsufsort on snapshot files in the 'tests' directory")
    ("quiet", "Do not print DIRTYBIRD banner or stratum job messages")
  ;

  general.add(stratum);
  general.add(dero);
  general.add(testing);
  general.add(advanced);
  general.add(debug);
  return general;
}

inline int get_prog_style()
{
    int style = (po::command_line_style::unix_style | po::command_line_style::allow_long_disguise);
    return style;
}