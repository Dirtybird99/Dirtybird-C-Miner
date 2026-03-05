#pragma once

#include <hip/hip_runtime.h>

// Basic OS detection
#if defined(_WIN32) || defined(_WIN64)
#  define DIRTYBIRD_OS_WINDOWS 1
#else
#  define DIRTYBIRD_OS_WINDOWS 0
#endif

#if defined(__linux__)
#  define DIRTYBIRD_OS_LINUX 1
#else
#  define DIRTYBIRD_OS_LINUX 0
#endif

// ---------------------------------------------------------------------
// ssize_t
// ---------------------------------------------------------------------
#include <cstddef>

#if DIRTYBIRD_OS_WINDOWS && !defined(ssize_t)
#  include <BaseTsd.h>
   typedef SSIZE_T ssize_t;
#endif

// ---------------------------------------------------------------------
// Sleep / nanosleep abstraction
// ---------------------------------------------------------------------
#include <chrono>
#include <thread>

inline void dirtybird_sleep_ns(long long ns) {
#if DIRTYBIRD_OS_WINDOWS
    // Windows Sleep is ms, so approximate
    auto ms = std::chrono::milliseconds((ns + 999999) / 1000000);
    std::this_thread::sleep_for(ms);
#else
    std::this_thread::sleep_for(std::chrono::nanoseconds(ns));
#endif
}

// If you really want a nanosleep-like API:
inline void dirtybird_sleep_timespec(long long sec, long long nsec) {
    long long ns = sec * 1000000000LL + nsec;
    dirtybird_sleep_ns(ns);
}

// ---------------------------------------------------------------------
// Compiler attributes: optnone, noinline, etc.
// ---------------------------------------------------------------------
#if defined(__clang__) && !defined(_MSC_VER)
#  define DIRTYBIRD_OPTNONE __attribute__((optnone))
#  define DIRTYBIRD_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#  define DIRTYBIRD_OPTNONE __declspec(noinline)  // best we can do
#  define DIRTYBIRD_NOINLINE __declspec(noinline)
#else
#  define DIRTYBIRD_OPTNONE
#  define DIRTYBIRD_NOINLINE
#endif

#if defined(_MSC_VER) && defined(__HIP_PLATFORM_NVIDIA__)
    #define DIRTYBIRD_HIP_LAUNCH_KERNEL(kernelName, numBlocks, threadsPerBlock, sharedMem, stream, ...) \
        (kernelName)<<<(numBlocks), (threadsPerBlock), (sharedMem), (stream)>>>(__VA_ARGS__)
#else
    #define DIRTYBIRD_HIP_LAUNCH_KERNEL(kernelName, numBlocks, threadsPerBlock, sharedMem, stream, ...) \
        hipLaunchKernelGGL(kernelName, numBlocks, threadsPerBlock, sharedMem, stream, __VA_ARGS__)
#endif