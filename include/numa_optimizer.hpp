#ifndef NUMA_OPTIMIZER_H
#define NUMA_OPTIMIZER_H

#include <iostream>
#include <stdexcept>
#include "dirtybird-hugepages.hpp"

#ifdef __linux__
#include <numa.h>
#include <sched.h>
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#ifdef DIRTYBIRD_RANDOMX
extern "C" {
#include <randomx/virtual_memory.hpp>
}
#endif

// ============================================================================
// Thread-local NUMA context
// ============================================================================

struct DirtybirdAllocInfo {
    DirtybirdPageType page_type;
    int numa_node;
    bool is_locked;
};

// DERO Miner: Use thread_local for all compilers (C++11 standard)
// __declspec(thread) is MSVC-specific and ignored by Clang
inline thread_local int dirtybird_tls_numa_node = -1;
inline thread_local DirtybirdAllocInfo dirtybird_tls_last_alloc = {DIRTYBIRD_PAGE_REGULAR, -1, false};

inline void dirtybird_setNumaNode(int node) {
    dirtybird_tls_numa_node = node;
}

inline void dirtybird_clearNumaNode() {
    dirtybird_tls_numa_node = -1;
}

inline int dirtybird_getNumaNode() {
    return dirtybird_tls_numa_node;
}

inline DirtybirdAllocInfo dirtybird_getLastAllocInfo() {
    return dirtybird_tls_last_alloc;
}

inline void dirtybird_setLastAllocInfo(DirtybirdPageType type, int node, bool locked) {
    dirtybird_tls_last_alloc.page_type = type;
    dirtybird_tls_last_alloc.numa_node = node;
    dirtybird_tls_last_alloc.is_locked = locked;
}

// ============================================================================
// NUMAOptimizer class
// ============================================================================

class NUMAOptimizer {
public:
    struct NodeInfo {
        int node_id;
        int num_cpus;
        long memory_size_mb;
        bool has_memory;
    };

    static bool initialize();
    static int getMemoryNodes();
    static int getTotalCPUs();
    static void* allocateLocal(size_t size);
    static void* allocateOnNode(size_t size, int node);
    static void deallocate(void* ptr, size_t size);
    static void optimizeMemoryForMining(void* ptr, size_t size);
    static void printThreadBinding(int thread_id);
    static bool isAvailable();
    
    static const HugePagesInfo& getCachedHugePagesInfo() {
        static HugePagesInfo info = getHugePagesInfo();
        return info;
    }

    static bool isOneGbPagesAvailable() {
        const auto& info = getCachedHugePagesInfo();
        return info.page_size_1gb > 0 && info.free_1gb > 0;
    }

    static bool isHugePagesAvailable() {
        const auto& info = getCachedHugePagesInfo();
        return (info.page_size_2mb > 0 && info.free_2mb > 0) || 
               (info.page_size_1gb > 0 && info.free_1gb > 0);
    }

    static bool setMemoryPolicy(int node);
    static void restoreMemoryPolicy();
    
    static DirtybirdAllocInfo getLastAllocInfo() { return dirtybird_getLastAllocInfo(); }
    static void printAllocInfo(const DirtybirdAllocInfo& info);

    class ScopedMemoryPolicy {
    private:
        bool need_restore;
        int target_node;
    public:
        ScopedMemoryPolicy(int node) : need_restore(false), target_node(node) {
            if (node >= 0) {
                dirtybird_setNumaNode(node);
                need_restore = NUMAOptimizer::setMemoryPolicy(node);
            }
        }
        ~ScopedMemoryPolicy() {
            dirtybird_clearNumaNode();
            if (need_restore) {
                NUMAOptimizer::restoreMemoryPolicy();
            }
        }
        int getNode() const { return target_node; }
    };

private:
    static bool numa_initialized;
    static int memory_nodes;
    static int total_cpus;
    static void detectTopology();
};

#endif // NUMA_OPTIMIZER_H