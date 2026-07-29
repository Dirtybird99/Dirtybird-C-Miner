/* test_share_mailbox.cpp - the single-slot mailbox loses shares silently.
 *
 * The mailbox holds ONE pending share. The network thread drains it once per
 * recv iteration on the 50ms poll, so two discoveries inside that window
 * collide and the earlier one is destroyed. Before submitDrops existed nothing
 * recorded that: staleDrops counts only epoch mismatches, so a rig could shed
 * shares for months with every counter looking healthy.
 *
 * These cases pin the loss so it shows up in -V, and pin that a share is NOT
 * counted as dropped on the ordinary path where the network thread kept up. */
#include <cstdio>
#include <cstring>
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

/* What the network thread does when it picks the share up. */
static void drain(MinerState &st)
{
	st.submitReady.store(false, std::memory_order_release);
}

int main(void)
{
	/* 1. First share into an empty mailbox: kept, nothing destroyed. */
	CHECK(dluna_mailbox_stage(G, "job-A", "aa", 1) == true,
	      "staging into an empty mailbox reported an overwrite");
	CHECK(G.submitDrops.load() == 0, "empty mailbox counted a drop: %lld",
	      (long long)G.submitDrops.load());
	CHECK(G.submitBlob == "aa", "mailbox holds %s, want aa", G.submitBlob.c_str());

	/* 2. Second share before the network thread drains: A is destroyed. This
	 *    is the loss that used to leave no trace at all. */
	CHECK(dluna_mailbox_stage(G, "job-B", "bb", 1) == false,
	      "overwriting a waiting share reported success");
	CHECK(G.submitDrops.load() == 1, "overwrite counted %lld drops, want 1",
	      (long long)G.submitDrops.load());
	CHECK(G.submitBlob == "bb", "mailbox holds %s after overwrite, want bb",
	      G.submitBlob.c_str());
	/* The destroyed share is gone -- it is NOT still queued behind B. Depth 1
	 * is the actual defect; this test only makes its cost countable. */
	CHECK(G.submitJobId == "job-B", "mailbox jobid %s, want job-B",
	      G.submitJobId.c_str());

	/* 3. A third share AFTER a drain must not be counted as a drop: the
	 *    counter has to mean "a share was destroyed", not "a share was sent".
	 *    Getting this wrong would make the number useless in exactly the
	 *    healthy case it is meant to certify. */
	drain(G);
	CHECK(dluna_mailbox_stage(G, "job-C", "cc", 2) == true,
	      "staging after a drain reported an overwrite");
	CHECK(G.submitDrops.load() == 1,
	      "drained mailbox counted an extra drop (now %lld, want 1)",
	      (long long)G.submitDrops.load());

	/* 4. Back-to-back overwrites accumulate rather than saturating at one. */
	CHECK(dluna_mailbox_stage(G, "job-D", "dd", 2) == false, "expected overwrite");
	CHECK(dluna_mailbox_stage(G, "job-E", "ee", 2) == false, "expected overwrite");
	CHECK(G.submitDrops.load() == 3, "want 3 total drops, got %lld",
	      (long long)G.submitDrops.load());

	/* 5. Staging preserves the epoch, which the network thread re-checks
	 *    before sending. A dropped share must not corrupt that. */
	CHECK(G.submitEpoch == 2, "epoch %llu survived staging, want 2",
	      (unsigned long long)G.submitEpoch);

	if (g_failures == 0) {
		std::printf("share mailbox: PASS (drops counted: %lld)\n",
		            (long long)G.submitDrops.load());
		return 0;
	}
	std::printf("share mailbox: %d FAILURE(S)\n", g_failures);
	return 1;
}
