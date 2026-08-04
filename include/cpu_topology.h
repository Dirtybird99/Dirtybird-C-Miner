#pragma once

#include <cstdint>

/* P-core-first thread pinning.
 *
 * AstroBWTv3 is memory-bandwidth bound and its per-thread working set is
 * L2-sized, so thread placement matters on hybrid parts: two workers on the
 * HT siblings of one physical core share an L1/L2 and a memory port and add
 * little over one worker, while an E-core brings its own cache. Preferred
 * placement order is therefore
 *
 *   1. one logical CPU per physical core of the highest efficiency class
 *      (the P-core primaries),
 *   2. every logical CPU of the lower efficiency classes (the E-cores),
 *   3. the SMT siblings of the highest class, last.
 *
 * On an 8P+8E part (i7-13700HX) this yields
 *   [0,2,4,6,8,10,12,14, 16..23, 1,3,5,7,9,11,13,15].
 *
 * Pinning is ON by default; opt out with --no-pin, DLUNA_NO_PIN=1, or
 * "pin": false in config.json. */

/* One physical core: its efficiency class (higher = faster core on hybrid
 * parts, all-equal on homogeneous parts) and its logical CPU ids. */
struct DlunaCoreInfo {
    int efficiency_class;
    int primary_cpu;
    int sibling_cpu;   /* -1 when the core has no SMT sibling */
};

struct DlunaCpuOrder {
    int count;              /* number of valid entries in order[] */
    uint8_t order[64];      /* logical CPU ids, best placement first */
    bool detected;          /* false = topology unknown; order is identity */
};

/* Pure placement policy (unit-tested): cores -> preferred logical CPU order.
 * nlogical is the identity-fallback width used when ncores is 0. */
DlunaCpuOrder dluna_pin_order_from_cores(const DlunaCoreInfo *cores, int ncores,
                                         int nlogical);

/* Detected order for this machine (cached after the first call). */
const DlunaCpuOrder &dluna_cpu_order();

/* False when DLUNA_NO_PIN is set to anything but "0". */
bool dluna_pin_enabled();

/* Pin the calling mining thread to dluna_cpu_order().order[tid % count].
 * Honors dluna_pin_enabled(). Returns 1 if a pin syscall was made, else 0. */
uint32_t dluna_pin_mining_thread(int tid);
