/*
 * test_statusline.cpp -- dluna_format_status() must always produce ONE row.
 *
 * The Termux bug this guards against: the status line is repainted with a bare
 * \r, which rewinds to the start of the row the cursor is on. A line wider than
 * the terminal auto-wraps, so the cursor ends on its LAST row and every repaint
 * starts a row lower -- the line stacks down the screen instead of updating in
 * place. 115 visible columns on a 56-column phone terminal cost two rows per
 * second.
 *
 * So the invariant under test is: visible width (ANSI escapes excluded) never
 * exceeds cols - 1, at every terminal width, for every field magnitude -- and
 * the escapes survive intact while that is enforced.
 */

#include "dluna.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int g_failures = 0;

#define CHECK(cond, ...)                                                     \
	do {                                                                 \
		if (!(cond)) {                                               \
			++g_failures;                                        \
			std::printf("FAIL %s:%d: ", __FILE__, __LINE__);     \
			std::printf(__VA_ARGS__);                            \
			std::printf("\n");                                   \
		}                                                            \
	} while (0)

/* Columns the cursor advances: everything except \r and complete CSI escapes.
 * Also reports whether any escape was left truncated -- a byte-offset cut would
 * wedge the terminal's colour state and print the escape's tail as text. */
static int visible_width(const std::string &s, bool *escape_intact)
{
	int vis = 0;
	*escape_intact = true;
	for (size_t i = 0; i < s.size();) {
		unsigned char c = (unsigned char)s[i];
		if (c == 0x1b) {
			if (i + 1 >= s.size() || s[i + 1] != '[') {
				*escape_intact = false;
				return vis;
			}
			size_t j = i + 2;
			while (j < s.size() && (unsigned char)s[j] >= 0x30 &&
			                       (unsigned char)s[j] <= 0x3f) ++j;
			while (j < s.size() && (unsigned char)s[j] >= 0x20 &&
			                       (unsigned char)s[j] <= 0x2f) ++j;
			if (j >= s.size()) {   /* ran out before the final byte */
				*escape_intact = false;
				return vis;
			}
			i = j + 1;
			continue;
		}
		if (c < 0x20) { ++i; continue; }   /* \r -- zero width */
		++vis;
		++i;
	}
	return vis;
}

struct Case {
	const char *name;
	DlunaStatus st;
};

int main()
{
	/* dluna_console_init() is deliberately NOT called: g_tty/g_vt_ok default
	 * to true, so dluna_clr_eol() is "\033[K" here regardless of how ctest
	 * attaches stdout. The tail assertion uses the same accessor either way. */
	const std::string tail = std::string("\033[0m") + dluna_clr_eol();

	std::vector<Case> cases = {
		/* rate  avg    height     mb      blk     rej   diff     hh mm ss */
		{"zeros",     {0.0,   0.0,  0,         0,      0,      0,    "0",    0,  0,  0}},
		{"screenshot",{0.0,   4.96, 7368356,   962,    93,     25,   "795K", 0,  1, 11}},
		{"typical",   {20.91, 20.71, 7212998,  1998,   262,    0,    "20K",  3, 14,  7}},
		{"maxed",     {9999.99, 9999.99, 999999999LL, 9999999LL, 9999999LL,
		               9999999LL, "999G", 99, 59, 59}},
	};

	const int widths[] = {0, 10, 20, 28, 34, 40, 56, 60, 72, 80, 86, 100,
	                      110, 116, 120, 200};

	int full_width_ref = -1;

	for (const Case &c : cases) {
		for (int cols : widths) {
			char buf[512];
			size_t n = dluna_format_status(buf, sizeof buf, c.st, cols);
			std::string out(buf, n);

			CHECK(n == std::strlen(buf),
			      "[%s cols=%d] returned length %zu != strlen %zu",
			      c.name, cols, n, std::strlen(buf));

			bool intact = false;
			int vis = visible_width(out, &intact);

			CHECK(intact, "[%s cols=%d] truncated ANSI escape in output",
			      c.name, cols);

			if (cols > 0) {
				CHECK(vis <= cols - 1,
				      "[%s cols=%d] visible width %d exceeds budget %d "
				      "-- this line WILL wrap and stack",
				      c.name, cols, vis, cols - 1);
			}

			/* Exactly one leading \r and no newline: a newline would
			 * scroll the row away, defeating in-place repainting. */
			CHECK(out.size() > 0 && out[0] == '\r',
			      "[%s cols=%d] does not start with \\r", c.name, cols);
			CHECK(out.find('\r', 1) == std::string::npos,
			      "[%s cols=%d] contains a second \\r", c.name, cols);
			CHECK(out.find('\n') == std::string::npos,
			      "[%s cols=%d] contains a newline", c.name, cols);

			/* Colour must always be reset and the row cleared, or a
			 * shorter line leaves the previous tick's tail on screen. */
			CHECK(out.size() >= tail.size() &&
			      out.compare(out.size() - tail.size(), tail.size(), tail) == 0,
			      "[%s cols=%d] does not end with reset + erase-to-EOL",
			      c.name, cols);
		}
	}

	/* cols <= 0 means "width unknown" and must reproduce the pre-1.0.26 line
	 * byte for byte, so a failed width query is never a visible regression. */
	{
		DlunaStatus st = cases[1].st;
		char buf[512];
		dluna_format_status(buf, sizeof buf, st, 0);
		char legacy[512];
		std::snprintf(legacy, sizeof legacy,
		              "\r\033[93m[DIRTYBIRD] \033[92m%.2f KH/s\033[97m "
		              "(\033[32m%.2f KH/s avg\033[97m) | \033[34mHeight:%lld\033[97m | "
		              "\033[36mMiniblocks:%lld\033[97m | \033[32mBlocks:%lld\033[97m | "
		              "%sREJ:%lld\033[97m | \033[35mDiff:%s\033[97m | "
		              "\033[37m%02d:%02d:%02d\033[0m%s",
		              st.rate, st.avg, st.height, st.accepted, st.blocks,
		              st.rejected > 0 ? "\033[91m" : "\033[37m", st.rejected,
		              st.diff, st.hh, st.mm, st.ss, dluna_clr_eol());
		CHECK(std::strcmp(buf, legacy) == 0,
		      "cols=0 must be byte-identical to the legacy full line\n"
		      "  got:      %s\n  expected: %s", buf, legacy);

		bool intact = false;
		full_width_ref = visible_width(std::string(buf), &intact);
		CHECK(full_width_ref == 115,
		      "screenshot case should be 115 visible columns, got %d",
		      full_width_ref);
	}

	/* A desktop console wide enough for the full layout must still get it. */
	{
		char wide[512], unknown[512];
		dluna_format_status(wide, sizeof wide, cases[1].st, 200);
		dluna_format_status(unknown, sizeof unknown, cases[1].st, 0);
		CHECK(std::strcmp(wide, unknown) == 0,
		      "cols=200 must select the full layout unchanged");
	}

	/* The reported 56-column Termux width: must fit, and must still carry the
	 * counters -- fitting by throwing every field away is not a fix. */
	{
		char buf[512];
		dluna_format_status(buf, sizeof buf, cases[1].st, 56);
		std::string out(buf);
		bool intact = false;
		int vis = visible_width(out, &intact);
		CHECK(vis <= 55, "56-column render is %d columns wide", vis);
		CHECK(out.find("MB:962") != std::string::npos,
		      "56-column render dropped the miniblock counter: %s", buf);
		CHECK(out.find("B:93") != std::string::npos,
		      "56-column render dropped the block counter: %s", buf);
		CHECK(out.find("R:25") != std::string::npos,
		      "56-column render dropped the reject counter: %s", buf);
		std::printf("56-col render (%d cols visible): %s\n",
		            vis, out.c_str() + 1);
	}

	/* dluna_term_cols() is mostly untestable here -- TIOCGWINSZ and
	 * GetConsoleScreenBufferInfo both need a real terminal, and ctest gives us
	 * a pipe. The DIRTYBIRD_COLS override is the one branch we can pin, and it
	 * is also the escape hatch for setups where the ioctl reports 0. Driven by
	 * the `statusline_env_cols` test registration. */
	if (const char *env = std::getenv("DIRTYBIRD_COLS")) {
		int want = std::atoi(env);
		int got  = dluna_term_cols();
		CHECK(got == want, "DIRTYBIRD_COLS=%s but dluna_term_cols() == %d",
		      env, got);

		char buf[512];
		dluna_format_status(buf, sizeof buf, cases[1].st, got);
		bool intact = false;
		int vis = visible_width(std::string(buf), &intact);
		CHECK(intact && vis <= got - 1,
		      "override width %d rendered %d columns", got, vis);
		std::printf("statusline: DIRTYBIRD_COLS=%d honored (%d columns)\n",
		            got, vis);
	}

	if (g_failures == 0) {
		std::printf("statusline: PASS\n");
		return 0;
	}
	std::printf("statusline: %d FAILURE(S)\n", g_failures);
	return 1;
}
