#pragma once

#include <atomic>
#include <cstdint>
#include <vector>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#include <mmsystem.h>
#else
#include <thread>
#include <chrono>
#endif

// Duty-Cycle Pacer - Insert micro-sleeps to prevent thermal throttling
//
// Two modes:
//   1. Per-thread hash-count pacing (original): each thread sleeps independently
//   2. Synchronized epoch pacing (new): all threads sleep at the same time
//      to enable CPU C6 cluster power-down state, which drops package power
//      by ~40W during the sleep window.
//
// Thermal governor: hashrate-based throttle detection
//   - Tracks peak hashrate over a sliding window
//   - If current hashrate drops >15% from peak, signals throttle
//   - Returns scaling factor 0-100 (100 = full speed)
//   - Mining threads check this and yield() briefly to prevent cliff-throttling
//
// On Windows, uses timeBeginPeriod(1) + Sleep(N) for accurate ms-level pacing.
// Without timeBeginPeriod, Sleep(1) actually sleeps ~15.6ms.

namespace thermal {

// Global pace configuration (set from main/config)
inline std::atomic<uint32_t> g_pace_interval{0};
inline std::atomic<uint32_t> g_pace_sleep_ms{0};

// Thermal governor state: scaling factor 0-100 (100 = full speed)
// Updated once per second from reporter. Read by mining threads.
inline std::atomic<int> g_thermal_scale{100};
inline std::atomic<bool> g_thermal_yield_enabled{false};

// Peak hashrate observed (hashes/sec), used for throttle detection
inline std::atomic<int64_t> g_peak_hashrate{0};

// Number of consecutive seconds hashrate has been depressed
inline std::atomic<int> g_throttle_count{0};

// Synchronized epoch sleep configuration
// All threads check a shared timestamp and sleep simultaneously when the epoch advances.
// This enables CPU C6 cluster power-down (all cores idle = package enters deep C-state).
// SYNC_PACE_INTERVAL_MS: time between synchronized pauses (e.g., 50ms)
// SYNC_PACE_SLEEP_MS: duration of each pause (e.g., 2ms)
// Duty cycle = SLEEP / INTERVAL (e.g., 2/50 = 4% overhead)
#ifndef SYNC_PACE_INTERVAL_MS
#define SYNC_PACE_INTERVAL_MS 0  // 0 = disabled
#endif
#ifndef SYNC_PACE_SLEEP_MS
#define SYNC_PACE_SLEEP_MS 2
#endif

// Shared epoch timestamp — all threads converge on this
inline std::atomic<uint64_t> g_next_sleep_epoch_ms{0};

// Initialize duty-cycle pacer
// interval: hash count between sleeps (e.g., 32 = sleep every 32 hashes)
// sleep_ms: milliseconds to sleep (e.g., 1 = 1ms)
inline void init_pacer(uint32_t interval, uint32_t sleep_ms) {
    g_pace_interval.store(interval, std::memory_order_relaxed);
    g_pace_sleep_ms.store(sleep_ms, std::memory_order_relaxed);
#ifdef _WIN32
    // Set Windows timer resolution to 1ms for accurate Sleep()
    timeBeginPeriod(1);
#endif

    // Initialize epoch for synchronized pacing
#if SYNC_PACE_INTERVAL_MS > 0
#ifdef _WIN32
    g_next_sleep_epoch_ms.store(GetTickCount64() + SYNC_PACE_INTERVAL_MS, std::memory_order_relaxed);
#else
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
    g_next_sleep_epoch_ms.store(ms + SYNC_PACE_INTERVAL_MS, std::memory_order_relaxed);
#endif
#endif
}

// Called once per second from reporter. Analyzes hashrate trend and returns
// effective scaling % (100 = full speed, lower = throttling detected).
// Uses hashrate-based proxy: if current rate drops >15% from peak, signal throttle.
inline int update(const std::vector<int64_t>& rate_history) {
    if (rate_history.size() < 5) {
        // Not enough data yet — assume full speed
        return 100;
    }

    // Current hashrate: average of last 3 samples for noise reduction
    size_t n = rate_history.size();
    int64_t current = 0;
    int samples = std::min((size_t)3, n);
    for (size_t i = n - samples; i < n; ++i) {
        current += rate_history[i];
    }
    current /= samples;

    // Update peak (only after warmup: first 10 seconds)
    int64_t peak = g_peak_hashrate.load(std::memory_order_relaxed);
    if (n > 10 && current > peak) {
        g_peak_hashrate.store(current, std::memory_order_relaxed);
        peak = current;
    } else if (n <= 10) {
        // During warmup, just track peak but don't throttle
        if (current > peak) {
            g_peak_hashrate.store(current, std::memory_order_relaxed);
        }
        g_thermal_scale.store(100, std::memory_order_relaxed);
        return 100;
    }

    if (peak <= 0) {
        g_thermal_scale.store(100, std::memory_order_relaxed);
        return 100;
    }

    // Calculate drop percentage
    double ratio = (double)current / (double)peak;
    int scale;

    if (ratio >= 0.85) {
        // Within 15% of peak — no throttling
        g_throttle_count.store(0, std::memory_order_relaxed);
        scale = 100;
    } else if (ratio >= 0.70) {
        // 15-30% drop — mild throttle detected, scale proportionally
        int count = g_throttle_count.fetch_add(1, std::memory_order_relaxed) + 1;
        // Only activate after 3 consecutive seconds of depression
        if (count >= 3) {
            scale = static_cast<int>(ratio * 100.0);  // e.g., 75% of peak -> scale=75
        } else {
            scale = 100;
        }
    } else {
        // >30% drop — severe throttle, aggressive backoff
        g_throttle_count.fetch_add(1, std::memory_order_relaxed);
        scale = static_cast<int>(ratio * 100.0);
        if (scale < 50) scale = 50;  // Floor at 50% to avoid stalling
    }

    g_thermal_scale.store(scale, std::memory_order_relaxed);
    return scale;
}

inline void set_thermal_yield_enabled(bool enabled) {
    g_thermal_yield_enabled.store(enabled, std::memory_order_relaxed);
}

inline bool thermal_yield_enabled() {
    return g_thermal_yield_enabled.load(std::memory_order_relaxed);
}

// Thread-side: check if thermal governor signals we should yield.
// This is extremely lightweight — just an atomic load.
inline bool should_thermal_yield() {
    return thermal_yield_enabled() &&
           g_thermal_scale.load(std::memory_order_relaxed) < 100;
}

// Thread-side: perform thermal yield. Brief yield or micro-sleep
// proportional to how far below target we are.
inline void do_thermal_yield() {
    int scale = g_thermal_scale.load(std::memory_order_relaxed);
    if (scale >= 100) return;

    if (scale >= 85) {
        // Mild: just yield timeslice
        std::this_thread::yield();
    } else {
        // Moderate/severe: brief sleep to let CPU cool
        // scale=75 -> 100us, scale=50 -> 250us
        int us = (100 - scale) * 5;
        if (us > 500) us = 500;  // Cap at 500us
#ifdef _WIN32
        // Windows Sleep(0) = yield, Sleep(1) = 1ms minimum with timeBeginPeriod(1)
        // For sub-ms, use yield loop
        if (us < 1000) {
            std::this_thread::yield();
        } else {
            Sleep(1);
        }
#else
        std::this_thread::sleep_for(std::chrono::microseconds(us));
#endif
    }
}

// Thread-side: check if this hash should trigger a pace sleep
inline bool should_pace(uint64_t hash_count) {
    uint32_t interval = g_pace_interval.load(std::memory_order_relaxed);
    if (interval == 0) return false;
    return (hash_count % interval) == 0;
}

// Thread-side: perform the pace sleep (accurate on Windows with timeBeginPeriod)
inline void do_pace_sleep() {
    uint32_t ms = g_pace_sleep_ms.load(std::memory_order_relaxed);
    if (ms > 0) {
#ifdef _WIN32
        Sleep(ms);
#else
        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
#endif
    }
}

// Synchronized epoch sleep: all threads check the same timestamp and sleep together.
// When the epoch advances, every thread that sees it sleeps for SYNC_PACE_SLEEP_MS.
// Because each hash takes ~1.25ms, all 20 threads notice within ~2ms and sleep
// near-simultaneously, enabling CPU C6 cluster power-down.
inline void sync_pace_check() {
#if SYNC_PACE_INTERVAL_MS > 0
#ifdef _WIN32
    uint64_t now = GetTickCount64();
#else
    auto tp = std::chrono::steady_clock::now().time_since_epoch();
    uint64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(tp).count();
#endif
    uint64_t next = g_next_sleep_epoch_ms.load(std::memory_order_relaxed);
    if (now >= next) {
        // Advance the epoch — only one thread wins the CAS, but all threads sleep
        uint64_t new_epoch = now + SYNC_PACE_INTERVAL_MS;
        g_next_sleep_epoch_ms.compare_exchange_strong(next, new_epoch, std::memory_order_relaxed);
#ifdef _WIN32
        Sleep(SYNC_PACE_SLEEP_MS);
#else
        std::this_thread::sleep_for(std::chrono::milliseconds(SYNC_PACE_SLEEP_MS));
#endif
    }
#endif
}

} // namespace thermal
