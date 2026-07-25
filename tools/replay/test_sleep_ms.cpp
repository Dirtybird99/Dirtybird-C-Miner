/*
 * test_sleep_ms.cpp -- dluna_sleep_ms(N) must actually elapse ~N ms.
 *
 * The bug this guards against (PR #9): the binary once linked
 * -Wl,--wrap=nanosleep with a __wrap_nanosleep that returned 0 immediately,
 * which also neutralised std::this_thread::sleep_for (its nanosleep call is
 * resolved at exe link). Every POSIX dluna_sleep_ms became a no-op: the
 * reporter busy-spun a core and displayed 0.00 KH/s, backoff_sleep() ran a
 * 30 s reconnect backoff in microseconds, and workers spun waiting for the
 * first job. Windows (Sleep) was unaffected, so CI stayed green.
 *
 * This test links with the production CMAKE_EXE_LINKER_FLAGS, so any future
 * link-level symbol game that silently voids the sleep fails here. A real
 * 50 ms sleep can only oversleep; a neutralised one returns in microseconds,
 * so the 40 ms lower bound has a wide margin against scheduler jitter.
 */

#include "dluna.h"

#include <chrono>
#include <cstdio>

int main()
{
	using Clock = std::chrono::steady_clock;

	for (int i = 0; i < 3; ++i) {
		auto t0 = Clock::now();
		dluna_sleep_ms(50);
		double ms = std::chrono::duration<double, std::milli>(
		                Clock::now() - t0).count();
		if (ms < 40.0) {
			std::printf("FAIL: dluna_sleep_ms(50) returned after "
			            "%.3f ms (iteration %d) -- the sleep is "
			            "being neutralised at link time\n", ms, i);
			return 1;
		}
	}

	/* ms<=0 must return promptly, not block (poll(-1) would wait forever). */
	auto t0 = Clock::now();
	dluna_sleep_ms(0);
	double ms = std::chrono::duration<double, std::milli>(
	                Clock::now() - t0).count();
	if (ms > 1000.0) {
		std::printf("FAIL: dluna_sleep_ms(0) took %.3f ms\n", ms);
		return 1;
	}

	std::printf("PASS sleep_ms\n");
	return 0;
}
