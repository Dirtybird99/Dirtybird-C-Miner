/**
 * Global mimalloc override for operator new/delete
 *
 * This file replaces the standard C++ operator new/delete with mimalloc.
 * Since this defines global replacements, ALL code in the program
 * (including precompiled libraries like SPSA) will use mimalloc for
 * heap allocations via new/delete.
 *
 * Benefits:
 * - 15-30% faster allocation for small objects
 * - Better cache locality (thread-local heaps)
 * - Lower contention in multi-threaded mining
 * - Large OS pages (2MB) for reduced TLB misses
 */

#ifdef USE_MIMALLOC
#include <mimalloc.h>
#include <mimalloc-new-delete.h>

// Enable large (2MB) OS pages for all mimalloc allocations at startup.
// This reduces TLB misses for SPSA's heap allocations.
// Uses C++ static initialization to run before main().
namespace {
struct MimallocInit {
    MimallocInit() {
        mi_option_set(mi_option_allow_large_os_pages, 1);
        mi_option_set(mi_option_eager_commit, 1);
    }
} g_mimalloc_init;
}
#endif
