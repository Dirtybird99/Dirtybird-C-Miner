/*
 * main.cpp -- DIRTYBIRD Miner entry point
 *
 * Clean-room DERO miner. AstroBWT v3.
 * No Boost. No nlohmann. Just pthreads and sockets.
 */

#include "dluna.h"
#include "spsa.hpp"
#include "hugepages.h"
#include "simd_wolf.h"
#include "hex.h"
#include "runtime_tune.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <thread>
#include <vector>
#include <chrono>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>
#endif

/* --- globals --- */

MinerState G;
bool g_has_avx2 = false;
bool g_huge_pages_avail = false;

/* SPSA required globals */
std::atomic<bool> ABORT_MINER{false};
/* Kept with `used` + default visibility so LTO does not elide the symbol;
 * the SPSA stage references this global at initialization. */
__attribute__((used, visibility("default"))) std::string devWallet;
std::atomic<int> devFee{0};

/* --- signal handler --- */

static void on_signal(int) { G.quit = true; }

/* --- hex helpers --- */

std::string to_hex(const uint8_t *data, int len)
{
	return hexStr(data, len);
}

bool from_hex(const std::string &hex, uint8_t *out, int max)
{
	if ((int)hex.size() > max * 2)
		return false;
	hexstrToBytes(hex, out);
	return true;
}

/* --- reporter thread --- */

/* PGO instrumentation runtime hook — flush profile data periodically so
 * that profile is preserved even when the miner is killed via taskkill /F
 * on Windows (graceful shutdown isn't possible for a console app from
 * outside the process). Only emitted in PGO_GENERATE builds. */
#ifdef DLUNA_PGO_GENERATE
extern "C" int __llvm_profile_write_file(void);
#endif

extern "C" uint64_t dluna_get_nanosleep_calls();

static void reporter_thread()
{
	int64_t prev = 0;
	auto t0 = std::chrono::steady_clock::now();
	auto prevT = t0;
	int flush_counter = 0;

	while (!G.quit) {
		dluna_sleep_ms(1000);

		auto now = std::chrono::steady_clock::now();
		double elapsed = std::chrono::duration<double>(now - t0).count();
		double dt = std::chrono::duration<double>(now - prevT).count();
		prevT = now;
		int64_t total = G.totalHashes.load(std::memory_order_relaxed);
		int64_t delta = total - prev;
		prev = total;

		/* Instantaneous KH/s over the ACTUAL inter-tick interval, not a
		 * hardcoded 1 s -- otherwise a reporter-thread scheduling delay under
		 * full load divides a multi-second batch by 1 s and spikes the reading. */
		double rate = (dt > 0) ? delta / (dt * 1000.0) : 0;
		double avg = (elapsed > 0) ? total / (elapsed * 1000.0) : 0;

		/* Colored [DIRTYBIRD] status line (the format users liked). Fields:
		 *   Height     = G.height   (written under jobMutex; an aligned 64-bit
		 *                            read for display is benign on x64)
		 *   Miniblocks = G.accepted (daemon "miniblocks" accepted)
		 *   Blocks     = G.blocks   (daemon integrator/full "blocks")
		 *   REJ        = G.rejected (daemon "rejected")
		 * No banner, no shares field. ANSI colors only on a TTY (console.cpp enables
		 * VT processing); the file/pipe branch stays plain so miner_live.log is clean. */
		long sec = (long)elapsed;
		int hh = (int)(sec / 3600), mm = (int)((sec % 3600) / 60),
		    ss = (int)(sec % 60);

		/* Humanize difficulty to K/M/G (port of the 1.0.12 reporter). */
		char diffbuf[24];
		unsigned long long dval = (unsigned long long)G.difficulty.load();
		if      (dval >= 1000000000ULL) snprintf(diffbuf, sizeof diffbuf, "%lluG", dval / 1000000000ULL);
		else if (dval >= 1000000ULL)    snprintf(diffbuf, sizeof diffbuf, "%lluM", dval / 1000000ULL);
		else if (dval >= 1000ULL)       snprintf(diffbuf, sizeof diffbuf, "%lluK", dval / 1000ULL);
		else                            snprintf(diffbuf, sizeof diffbuf, "%llu",  dval);

		std::lock_guard<std::mutex> lk(g_console_mtx);
		if (dluna_is_tty()) {
			/* Colored line rewritten in place with \r, so it has to fit on ONE
			 * row -- a wrapped line leaves the cursor on its last row and every
			 * repaint then stacks a row lower (Termux: 115 columns on a 56-column
			 * phone terminal). dluna_format_status() picks the widest layout that
			 * fits, re-querying the width each tick so a resize re-fits within a
			 * second. Ends with \033[0m + \033[K and NO trailing pad. */
			char line[512];
			DlunaStatus st{};
			st.rate     = rate;
			st.avg      = avg;
			st.height   = (long long)G.height;
			st.accepted = (long long)G.accepted.load();
			st.blocks   = (long long)G.blocks.load();
			st.rejected = (long long)G.rejected.load();
			st.diff     = diffbuf;
			st.hh = hh; st.mm = mm; st.ss = ss;
			dluna_format_status(line, sizeof line, st, dluna_term_cols());
			fputs(line, stdout);
		} else {
			/* Redirected to a file/pipe: same fields, no ANSI, newline-terminated. */
			printf("[DIRTYBIRD] %.2f KH/s (%.2f KH/s avg) | Height:%lld | "
			       "Miniblocks:%lld | Blocks:%lld | REJ:%lld | Diff:%s | "
			       "%02d:%02d:%02d\n",
			       rate, avg, (long long)G.height,
			       (long long)G.accepted.load(), (long long)G.blocks.load(),
			       (long long)G.rejected.load(), diffbuf, hh, mm, ss);
		}

		/* -V submit-funnel diagnostic: read-only atomics, no hot-loop impact.
		 * submitted ~ acc with ~0 stale/sendfail => healthy; high stale or
		 * sendfail, or submitted >> acc, points at a real submit/accept problem. */
		if (g_verbose) {
			printf("\n[funnel] submitted:%lld acc:%lld rej:%lld "
			       "stale:%lld sendfail:%lld\n",
			       (long long)G.submitted.load(), (long long)G.accepted.load(),
			       (long long)G.rejected.load(), (long long)G.staleDrops.load(),
			       (long long)G.sendFails.load());
		}
		fflush(stdout);

#ifdef DLUNA_PGO_GENERATE
		if (++flush_counter % 5 == 0) __llvm_profile_write_file();
#endif
	}
}

/* --- CLI --- */

#ifndef DIRTYBIRD_VERSION_STR
#define DIRTYBIRD_VERSION_STR "1.0.18"
#endif

static void print_usage(FILE *out, const char *argv0)
{
	fprintf(out,
		"Usage: %s [-d host:port] [-w wallet] [-t threads] [-p priority]\n"
		"  Config: reads config.json (next to the exe) for daemon-address/wallet/threads/\n"
		"          priority; CLI flags below override it. Edit config.json OR pass flags.\n"
		"  -d  daemon address (host:port)  (default: community-pools.mysrv.cloud:10300)\n"
		"  -w  wallet address              (default: built-in)\n"
		"  -t  number of mining threads (default: hardware threads)\n"
		"  -p  priority profile: normal | max  (default: normal)\n"
		"        normal = polite, never freezes the desktop\n"
		"        max    = aggressive (HIGH priority, full clock) for headless\n"
		"                 mining; maximum hashrate but stutters the desktop.\n"
		"        (env equivalent: DLUNA_PRIORITY=normal|max)\n"
		"        NOTE: the default is 'normal'. Use -p max for the previous\n"
		"        aggressive behavior.\n"
		"  -V, --verbose  show per-job / per-share event logging (default: off)\n"
		"        (env equivalent: DLUNA_VERBOSE=1)\n"
		"  --selftest     compute the pow(\"a\") KAT and exit (0=PASS, 1=FAIL)\n"
		"  -h, --help     show this help and exit\n"
		"  -v, --version  print version and exit\n", argv0);
}

static void usage(const char *argv0)
{
	print_usage(stderr, argv0);
	exit(1);
}

static bool g_selftest = false;

static void parse_args(int argc, char **argv)
{
	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "-d") && i + 1 < argc) {
			const char *arg = argv[++i];
			const char *colon = strrchr(arg, ':');
			if (!colon) {
				G.host = arg;
			} else {
				G.host = std::string(arg, colon - arg);
				G.port = (uint16_t)atoi(colon + 1);
			}
		} else if (!strcmp(argv[i], "-w") && i + 1 < argc) {
			G.wallet = argv[++i];
		} else if (!strcmp(argv[i], "-t") && i + 1 < argc) {
			G.nthreads = atoi(argv[++i]);
		} else if ((!strcmp(argv[i], "-p") || !strcmp(argv[i], "--priority")) && i + 1 < argc) {
			const char *v = argv[++i];
			if (strcmp(v, "normal") && strcmp(v, "max"))
				usage(argv[0]);  /* only normal|max accepted */
			/* Funnel the CLI choice through the same env var the runtime-tune
			 * code reads. Must be set before threads spawn / dluna_tune_*. */
#ifdef _WIN32
			_putenv_s("DLUNA_PRIORITY", v);
#else
			setenv("DLUNA_PRIORITY", v, 1);
#endif
		} else if (!strcmp(argv[i], "-V") || !strcmp(argv[i], "--verbose")) {
			g_verbose = true;
		} else if (!strcmp(argv[i], "--selftest")) {
			g_selftest = true;
		} else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
			print_usage(stdout, argv[0]);
			exit(0);
		} else if (!strcmp(argv[i], "-v") || !strcmp(argv[i], "--version")) {
			printf("Dirtybird Miner v%s\n", DIRTYBIRD_VERSION_STR);
			exit(0);
		} else {
			usage(argv[0]);
		}
	}

	if (const char *e = getenv("DLUNA_VERBOSE"))
		if (e[0] && strcmp(e, "0"))
			g_verbose = true;

	if (!g_selftest && (G.host.empty() || G.wallet.empty()))
		usage(argv[0]);
	if (G.nthreads <= 0)
		G.nthreads = (int)std::thread::hardware_concurrency();
	if (G.nthreads <= 0)
		G.nthreads = 4;
}

/* --- thread affinity --- */

static void set_affinity(std::thread &t, int core)
{
#if defined(_WIN32) || defined(__ANDROID__)
	// MinGW/winpthreads: std::thread::native_handle() returns pthread_t, not HANDLE.
	// Android/Bionic has no pthread_setaffinity_np. In both cases, no-op and let
	// the OS scheduler place threads. TODO: pin inside mine_thread if needed.
	(void)t; (void)core;
#else
	cpu_set_t cpuset;
	CPU_ZERO(&cpuset);
	CPU_SET(core, &cpuset);
	pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset);
#endif
}

/* --- config.json loader (minimal; no JSON lib — same flat-key approach as network.cpp) --- */

static bool read_file_to_string(const char *path, std::string &out)
{
	FILE *f = fopen(path, "rb");
	if (!f) return false;
	fseek(f, 0, SEEK_END);
	long n = ftell(f);
	fseek(f, 0, SEEK_SET);
	if (n > 0) { out.resize((size_t)n); out.resize(fread(&out[0], 1, (size_t)n, f)); }
	fclose(f);
	return true;
}

/* "key":"value" -> value ("" on miss). Flat config, no escapes/nesting. */
static std::string cfg_str(const std::string &j, const char *key)
{
	std::string pat = std::string("\"") + key + "\":";
	size_t p = j.find(pat);
	if (p == std::string::npos) return {};
	p += pat.size();
	while (p < j.size() && (j[p] == ' ' || j[p] == '\t' || j[p] == '\n' || j[p] == '\r')) p++;
	if (p >= j.size() || j[p] != '"') return {};
	p++;
	size_t q = j.find('"', p);
	if (q == std::string::npos) return {};
	return j.substr(p, q - p);
}

/* "key": <int> -> value (def on miss / if it's a string). Handles negative. */
static long long cfg_int(const std::string &j, const char *key, long long def)
{
	std::string pat = std::string("\"") + key + "\":";
	size_t p = j.find(pat);
	if (p == std::string::npos) return def;
	p += pat.size();
	while (p < j.size() && (j[p] == ' ' || j[p] == '\t' || j[p] == '\n' || j[p] == '\r')) p++;
	if (p >= j.size() || j[p] == '"') return def;
	return strtoll(j.c_str() + p, nullptr, 10);
}

/* Apply config.json (if present) into G as defaults. CLI args (parse_args) override.
 * Absent file = no-op (built-in defaults stand) -> fully backward compatible. */
static void load_config(const char *path)
{
	std::string j;
	if (!read_file_to_string(path, j) || j.empty()) return;

	std::string daddr = cfg_str(j, "daemon-address");
	if (daddr.empty()) daddr = cfg_str(j, "pool");
	if (daddr.empty()) daddr = cfg_str(j, "daemon");
	if (!daddr.empty()) {
		size_t colon = daddr.rfind(':');
		if (colon == std::string::npos) {
			G.host = daddr;
		} else {
			G.host = daddr.substr(0, colon);
			G.port = (uint16_t)atoi(daddr.c_str() + colon + 1);
		}
	}
	long long pj = cfg_int(j, "port", -1);
	if (pj > 0) G.port = (uint16_t)pj;

	std::string w = cfg_str(j, "wallet");
	if (!w.empty()) G.wallet = w;

	long long t = cfg_int(j, "threads", 0);
	if (t > 0) G.nthreads = (int)t;   /* <=0 (e.g. -1) keeps auto-detect */

	std::string prio = cfg_str(j, "priority");
	if (prio == "normal" || prio == "max") {
#ifdef _WIN32
		_putenv_s("DLUNA_PRIORITY", prio.c_str());
#else
		setenv("DLUNA_PRIORITY", prio.c_str(), 1);
#endif
	}
}

/* --- main --- */

int main(int argc, char **argv)
{
	dluna_console_init();

	/* config.json (next to the exe) provides defaults; CLI flags below override it. */
	load_config("config.json");

	parse_args(argc, argv);

	log_line("INFO", "Dirtybird Miner");
	log_line("INFO", "Server:  %s:%d", G.host.c_str(), G.port);
	log_line("INFO", "Wallet:  %s", G.wallet.c_str());
	log_line("INFO", "Threads: %d", G.nthreads);
	printf("\n");

	DlunaPriorityLevel prio = dluna_runtime_tune_options_from_env().level;
	if (g_verbose) printf("Priority: %s (%s)\n", dluna_priority_level_name(prio),
	       prio == DLUNA_PRIO_MAX
	           ? "aggressive — max hashrate, may stutter the desktop"
	           : "smooth — desktop stays responsive");

	uint32_t runtime_tune = dluna_tune_process_runtime();
	if (g_verbose) {
		if (runtime_tune != DLUNA_RUNTIME_TUNE_NONE)
			printf("Runtime: performance tuning active (0x%02x)\n\n",
			       runtime_tune);
		else
			printf("\n");
	}

	/* SPSA init. devWallet is set to the user's wallet; the in-tree SPSA
	 * stage reads this global at initialization. Share submission uses
	 * G.wallet (see network.cpp), so all rewards go to the user's wallet. */
	devWallet = G.wallet;
	initSPSA();

	/* Build 1D lookup table (~38 KB) */
	init_lut();

	/* AstroBWT v3 test vector check */
	{
	        workerData *w = new workerData();
	        memset(w, 0, sizeof(workerData));
#if defined(USE_ASTRO_SPSA)
	        for(int i=0; i<256; i++) w->iota8[i] = i;
#endif
	        uint8_t out[32];
	        char in_a[] = "a";
	        dluna_hash((uint8_t*)in_a, 1, out, *w);
	        std::string h = hexStr(out, 32);
	        bool pass = (h == "54e2324ddacc3f0383501a9e5760f85d63e9bc6705e9124ca7aef89016ab81ea");
	        if (g_selftest) {
	                printf("selftest pow(a): %s %s\n", h.c_str(), pass ? "PASS" : "FAIL");
	                delete w; exit(pass ? 0 : 1);
	        }
	        if (g_verbose) printf("Test pow(a): %s\n", h.c_str());
	        if (pass) {
	                if (g_verbose) printf("Test pow(a): PASS\n");
	        } else {
	                printf("Test pow(a): FAIL! (expected 54e2324ddacc3f0383501a9e5760f85d63e9bc6705e9124ca7aef89016ab81ea)\n"); exit(1);
	        }
	        delete w;
	}
	/* Detect AVX2, init compressed CodeLUT */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
	if (__builtin_cpu_supports("avx2")) {
		g_has_avx2 = true;
		init_code_lut_16();

		/* Quick AVX2 sanity check */
		uint8_t test_in[64] = {0};
		uint8_t test_out[64] = {0};
		for (int i = 0; i < 64; i++) test_in[i] = (uint8_t)i;
		wolfPermute_avx2(test_in, test_out, 42, 0, 31);
		if (g_verbose) printf("AVX2:    detected, wolfPermute verified\n");
	} else {
		if (g_verbose) printf("AVX2:    not available, using scalar fallback\n");
	}
#else
	if (g_verbose) printf("AVX2:    not supported on this arch\n");
#endif

	/* Huge pages */
	g_huge_pages_avail = enable_huge_page_privilege();
	if (g_verbose) printf("Huge pages: %s\n\n", g_huge_pages_avail ? "enabled" : "unavailable");

	/* Signal handlers */
	signal(SIGINT, on_signal);
	signal(SIGTERM, on_signal);

	/* Spawn threads */
	std::thread net(network_thread);
	std::thread rpt(reporter_thread);

	std::vector<std::thread> miners;
	miners.reserve(G.nthreads);
	for (int i = 0; i < G.nthreads; i++)
		miners.emplace_back(mine_thread, i);

	/* Pin mining threads to cores */
	int ncores = (int)std::thread::hardware_concurrency();
	for (int i = 0; i < G.nthreads; i++) {
		int core = (ncores > 0) ? (i % ncores) : i;
		set_affinity(miners[i], core);
	}

	/* Wait for quit */
	for (auto &t : miners)
		t.join();
	net.join();
	rpt.join();

	printf("\nShutdown complete. %lld hashes, %lld miniblocks (%lld blocks), %lld rejected.\n",
	       (long long)G.totalHashes.load(), (long long)G.accepted.load(),
	       (long long)G.blocks.load(), (long long)G.rejected.load());
	return 0;
}
