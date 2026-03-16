#include "dero_worker_pool.hpp"

#include "dirtybird-hugepages.hpp"
#include <astrobwtv3/astrobwtv3.h>
#include <astrobwtv3/lookupcompute.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

namespace {

struct DeroWorkerPoolState {
    std::mutex mutex;
    int worker_count = 0;
    std::vector<void*> allocations;
    std::vector<workerData*> workers;
    DeroWorkerPoolStats stats;
};

DeroWorkerPoolState g_dero_worker_pool;
constexpr int kWorkerHugePageAttempts = 5;
constexpr auto kWorkerHugePageRetryDelay = std::chrono::milliseconds(250);

void* allocateWorkerBlock() {
    void* allocation = nullptr;

    for (int attempt = 0; attempt < kWorkerHugePageAttempts; ++attempt) {
        allocation = malloc_huge_pages(sizeof(workerData));
        if (allocation == nullptr) {
            return nullptr;
        }

        if (get_huge_alloc_page_type(allocation) != DIRTYBIRD_PAGE_REGULAR ||
            attempt + 1 == kWorkerHugePageAttempts) {
            return allocation;
        }

        free_huge_pages(allocation);
        std::this_thread::sleep_for(kWorkerHugePageRetryDelay);
    }

    return allocation;
}

void shutdownDeroWorkerPoolLocked() {
    for (void* allocation : g_dero_worker_pool.allocations) {
        if (allocation != nullptr) {
            free_huge_pages(allocation);
        }
    }
    g_dero_worker_pool.worker_count = 0;
    g_dero_worker_pool.allocations.clear();
    g_dero_worker_pool.workers.clear();
    g_dero_worker_pool.stats = {};
}

} // namespace

bool initializeDeroWorkerPool(int worker_count) {
    if (worker_count <= 0) {
        return false;
    }

    std::lock_guard<std::mutex> lock(g_dero_worker_pool.mutex);
    if (!g_dero_worker_pool.workers.empty() && g_dero_worker_pool.worker_count == worker_count) {
        return true;
    }

    shutdownDeroWorkerPoolLocked();

    g_dero_worker_pool.worker_count = worker_count;
    g_dero_worker_pool.allocations.resize(worker_count, nullptr);
    g_dero_worker_pool.workers.resize(worker_count, nullptr);
    g_dero_worker_pool.stats.requested_workers = worker_count;
    g_dero_worker_pool.stats.requested_bytes = static_cast<size_t>(worker_count) * sizeof(workerData);

    for (int i = 0; i < worker_count; ++i) {
        void* allocation = allocateWorkerBlock();
        if (allocation == nullptr) {
            shutdownDeroWorkerPoolLocked();
            return false;
        }

        const DirtybirdPageType page_type = get_huge_alloc_page_type(allocation);
        if (page_type == DIRTYBIRD_PAGE_2MB) {
            ++g_dero_worker_pool.stats.large_2mb_workers;
        } else if (page_type == DIRTYBIRD_PAGE_1GB) {
            ++g_dero_worker_pool.stats.large_1gb_workers;
        } else {
            ++g_dero_worker_pool.stats.regular_workers;
        }
        if (is_huge_alloc_virtual_locked(allocation)) {
            ++g_dero_worker_pool.stats.virtual_locked_workers;
        }

        auto* worker = static_cast<workerData*>(allocation);
        std::memset(worker, 0, sizeof(workerData));
        initWorker(*worker);
        lookupGen(*worker, nullptr, nullptr);
        g_dero_worker_pool.allocations[i] = allocation;
        g_dero_worker_pool.workers[i] = worker;
    }

    return true;
}

workerData* getDeroWorkerForThread(int tid) {
    std::lock_guard<std::mutex> lock(g_dero_worker_pool.mutex);
    const int index = tid - 1;
    if (index < 0 || index >= g_dero_worker_pool.worker_count) {
        return nullptr;
    }
    return g_dero_worker_pool.workers[static_cast<size_t>(index)];
}

const DeroWorkerPoolStats& getDeroWorkerPoolStats() {
    return g_dero_worker_pool.stats;
}

void shutdownDeroWorkerPool() {
    std::lock_guard<std::mutex> lock(g_dero_worker_pool.mutex);
    shutdownDeroWorkerPoolLocked();
}
