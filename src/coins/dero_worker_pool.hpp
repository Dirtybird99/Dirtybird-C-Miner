#pragma once

#include <cstddef>

struct workerData;

struct DeroWorkerPoolStats {
    int requested_workers = 0;
    int large_2mb_workers = 0;
    int large_1gb_workers = 0;
    int regular_workers = 0;
    int virtual_locked_workers = 0;
    size_t requested_bytes = 0;
};

bool initializeDeroWorkerPool(int worker_count);
workerData* getDeroWorkerForThread(int tid);
const DeroWorkerPoolStats& getDeroWorkerPoolStats();
void shutdownDeroWorkerPool();
