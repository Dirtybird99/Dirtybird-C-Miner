/*
 * console.cpp -- coordinated console output for DIRTYBIRD Miner
 *
 * One mutex serializes every write to the console so timestamped event lines
 * (log_line) never corrupt the in-place status line drawn by the reporter
 * thread. Lives in dluna_runtime so both the miner and the PGO trainer (which
 * compile miner.cpp without main.cpp) resolve these symbols.
 */

#include "dluna.h"

#include <cstdio>
#include <cstdarg>
#include <ctime>
#include <chrono>

#ifdef _WIN32
#include <windows.h>
#include <io.h>      /* _isatty, _fileno */
#else
#include <unistd.h>  /* isatty, fileno */
#endif

std::mutex g_console_mtx;       /* serializes all console writes */
bool       g_verbose = false;   /* -V / DLUNA_VERBOSE: restore event spam */

/* When stdout is a real console we rewrite one line in place with \r + \033[K.
 * When it is redirected to a file/pipe, \r/\033[K are inert garbage, so we
 * fall back to plain newline-terminated periodic lines. */
static bool g_tty   = true;     /* stdout is an interactive console */
static bool g_vt_ok = true;     /* VT (ANSI \033[K) sequences usable */

bool dluna_is_tty(void) { return g_tty; }

/* Clears from the cursor to end of line. On a VT-capable console it is the ANSI
 * erase-to-EOL; otherwise empty (the status line space-pads instead). */
const char *dluna_clr_eol(void) { return (g_tty && g_vt_ok) ? "\033[K" : ""; }

void dluna_console_init(void)
{
#ifdef _WIN32
	g_tty = _isatty(_fileno(stdout)) != 0;
	if (g_tty) {
		SetConsoleOutputCP(CP_UTF8);
		HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
		DWORD mode = 0;
		if (h != INVALID_HANDLE_VALUE && GetConsoleMode(h, &mode)) {
			g_vt_ok = SetConsoleMode(
				h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0;
		} else {
			g_vt_ok = false;
		}
	} else {
		g_vt_ok = false;
	}
#else
	g_tty   = isatty(fileno(stdout)) != 0;
	g_vt_ok = g_tty; /* POSIX terminals understand \033[K */
#endif
}

/* Timestamped, mutex-guarded log line: "DD/MM HH:MM:SS.mmm  LEVEL  msg".
 * The leading \r + clear-to-EOL wipes any partial in-place status line so the
 * event starts clean at column 0; its trailing \n drops the cursor to a fresh
 * line, and the next reporter tick redraws the status line below it. */
void log_line(const char *level, const char *fmt, ...)
{
	auto now = std::chrono::system_clock::now();
	std::time_t t = std::chrono::system_clock::to_time_t(now);
	int ms = (int)(std::chrono::duration_cast<std::chrono::milliseconds>(
		now.time_since_epoch()).count() % 1000);

	char msg[512];
	va_list ap;
	va_start(ap, fmt);
	vsnprintf(msg, sizeof msg, fmt, ap);
	va_end(ap);

	std::lock_guard<std::mutex> lk(g_console_mtx);
	/* localtime() is safe here: every caller holds g_console_mtx, and it
	 * sidesteps the localtime_s signature differences across toolchains. */
	std::tm tm = *std::localtime(&t);
	char ts[32];
	std::strftime(ts, sizeof ts, "%d/%m %H:%M:%S", &tm);
	if (g_tty)
		printf("\r%s%s.%03d  %-5s %s\n", dluna_clr_eol(), ts, ms, level, msg);
	else
		printf("%s.%03d  %-5s %s\n", ts, ms, level, msg);
	fflush(stdout);
}
