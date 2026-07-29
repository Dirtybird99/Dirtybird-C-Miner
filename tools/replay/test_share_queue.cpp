/* test_share_queue.cpp - the submit queue must not lose concurrent finds.
 *
 * This replaced a single-slot mailbox. That mailbox overwrote unconditionally,
 * so a second find arriving before the network thread drained destroyed the
 * first -- and counted it nowhere, so a rig shed shares with every counter
 * reading healthy. The first case below is the regression test for exactly
 * that: two shares staged without an intervening drain must BOTH survive.
 *
 * Depth is 32, so eviction should never happen in practice with a worker count
 * in the tens. That is what makes submitDrops informative now: a non-zero value
 * means the network thread has stopped draining, not that the queue is small. */
#include <cstdio>
#include <string>

#include "dluna.h"

/* main.cpp owns the real one; this test is not linked against it. */
MinerState G;
bool       g_has_avx2 = false;

static int g_failures = 0;

#define CHECK(cond, ...)                                                      \
	do {                                                                      \
		if (!(cond)) {                                                        \
			std::printf("FAIL %s:%d: ", __FILE__, __LINE__);                  \
			std::printf(__VA_ARGS__);                                         \
			std::printf("\n");                                                \
			++g_failures;                                                     \
		}                                                                     \
	} while (0)

int main(void)
{
	/* 1. THE REGRESSION. Two finds, no drain between them. The old mailbox kept
	 *    only the second; both must now be queued, in order. */
	CHECK(dluna_submit_enqueue(G, "job-A", "aa", 1), "first enqueue reported an eviction");
	CHECK(dluna_submit_enqueue(G, "job-B", "bb", 1), "second enqueue reported an eviction");
	CHECK(G.submitQueue.size() == 2,
	      "queue holds %zu share(s) after two finds, want 2 -- one was lost",
	      G.submitQueue.size());
	CHECK(G.submitDrops.load() == 0, "a non-full queue counted %lld drop(s)",
	      (long long)G.submitDrops.load());
	CHECK(G.enqueued.load() == 2, "enqueued = %lld, want 2", (long long)G.enqueued.load());

	/* 2. FIFO: the oldest find is sent first. */
	CHECK(G.submitQueue.front().jobId == "job-A", "front is %s, want job-A (FIFO)",
	      G.submitQueue.front().jobId.c_str());
	CHECK(G.submitQueue.back().jobId == "job-B", "back is %s, want job-B",
	      G.submitQueue.back().jobId.c_str());

	/* 3. Every entry carries its own epoch, which the drain re-checks per item.
	 *    A shared epoch would let one stale share poison a whole batch. */
	CHECK(dluna_submit_enqueue(G, "job-C", "cc", 7), "third enqueue reported an eviction");
	CHECK(G.submitQueue.front().epoch == 1 && G.submitQueue.back().epoch == 7,
	      "per-item epochs not preserved (front=%llu back=%llu)",
	      (unsigned long long)G.submitQueue.front().epoch,
	      (unsigned long long)G.submitQueue.back().epoch);

	/* 4. Overflow evicts the OLDEST and counts it. Fill to exactly capacity
	 *    first: the queue must accept SUBMIT_QUEUE_MAX entries without ever
	 *    dropping, or the bound is off by one. */
	G.submitQueue.clear();
	G.submitDrops.store(0);
	G.enqueued.store(0);
	for (size_t i = 0; i < MinerState::SUBMIT_QUEUE_MAX; ++i) {
		char id[32];
		std::snprintf(id, sizeof id, "job-%zu", i);
		CHECK(dluna_submit_enqueue(G, id, "xx", 1),
		      "eviction at entry %zu, before the queue was full", i);
	}
	CHECK(G.submitQueue.size() == MinerState::SUBMIT_QUEUE_MAX,
	      "queue holds %zu at capacity, want %zu",
	      G.submitQueue.size(), MinerState::SUBMIT_QUEUE_MAX);
	CHECK(G.submitDrops.load() == 0, "filling to capacity counted %lld drop(s)",
	      (long long)G.submitDrops.load());

	/* One past capacity: oldest goes, size holds, loss is counted. */
	CHECK(!dluna_submit_enqueue(G, "job-overflow", "zz", 1),
	      "overflowing the queue did not report an eviction");
	CHECK(G.submitQueue.size() == MinerState::SUBMIT_QUEUE_MAX,
	      "queue grew past capacity to %zu", G.submitQueue.size());
	CHECK(G.submitDrops.load() == 1, "overflow counted %lld drop(s), want 1",
	      (long long)G.submitDrops.load());
	CHECK(G.submitQueue.front().jobId == "job-1",
	      "front is %s after eviction, want job-1 (oldest dropped)",
	      G.submitQueue.front().jobId.c_str());
	CHECK(G.submitQueue.back().jobId == "job-overflow",
	      "newest share was not kept (back is %s)", G.submitQueue.back().jobId.c_str());

	/* 5. enqueued counts every accepted push including the overflowing one --
	 *    it is a throughput counter, not a success counter; submitDrops is what
	 *    records loss. Conflating them would hide the loss again. */
	CHECK(G.enqueued.load() == (int64_t)MinerState::SUBMIT_QUEUE_MAX + 1,
	      "enqueued = %lld, want %zu", (long long)G.enqueued.load(),
	      MinerState::SUBMIT_QUEUE_MAX + 1);

	if (g_failures == 0) {
		std::printf("share queue: PASS (depth %zu, drops %lld)\n",
		            MinerState::SUBMIT_QUEUE_MAX, (long long)G.submitDrops.load());
		return 0;
	}
	std::printf("share queue: %d FAILURE(S)\n", g_failures);
	return 1;
}
