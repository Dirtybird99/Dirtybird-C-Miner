/*
 * cpu_topology.cpp -- CPU topology detection and P-core-first thread pinning.
 */

#include "cpu_topology.h"

#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(__linux__) && !defined(__ANDROID__)
#include <pthread.h>
#include <sched.h>
#endif

DlunaCpuOrder dluna_pin_order_from_cores(const DlunaCoreInfo *cores, int ncores,
                                         int nlogical)
{
    DlunaCpuOrder out;
    out.count = 0;
    out.detected = false;
    memset(out.order, 0, sizeof(out.order));

    auto identity = [&out](int n) {
        if (n > 64) n = 64;
        if (n < 0) n = 0;
        for (int i = 0; i < n; i++) out.order[i] = (uint8_t)i;
        out.count = n;
    };

    if (!cores || ncores <= 0) {
        identity(nlogical);
        return out;
    }

    int max_class = 0;
    for (int i = 0; i < ncores; i++) {
        if (cores[i].efficiency_class > max_class)
            max_class = cores[i].efficiency_class;
        /* Any logical id we cannot express in a 64-bit affinity mask means the
         * detection result is unusable -- fall back to identity. */
        if (cores[i].primary_cpu < 0 || cores[i].primary_cpu >= 64 ||
            cores[i].sibling_cpu >= 64) {
            identity(nlogical);
            return out;
        }
    }

    auto push = [&out](int cpu) {
        if (cpu >= 0 && out.count < 64)
            out.order[out.count++] = (uint8_t)cpu;
    };

    /* Tier 1: primaries of the highest class. */
    for (int i = 0; i < ncores; i++)
        if (cores[i].efficiency_class == max_class)
            push(cores[i].primary_cpu);

    /* Tier 2: all logicals of the lower classes, best class first. */
    for (int cls = max_class - 1; cls >= 0; cls--) {
        for (int i = 0; i < ncores; i++)
            if (cores[i].efficiency_class == cls)
                push(cores[i].primary_cpu);
        for (int i = 0; i < ncores; i++)
            if (cores[i].efficiency_class == cls)
                push(cores[i].sibling_cpu);
    }

    /* Tier 3: SMT siblings of the highest class. */
    for (int i = 0; i < ncores; i++)
        if (cores[i].efficiency_class == max_class)
            push(cores[i].sibling_cpu);

    if (out.count == 0) {
        identity(nlogical);
        return out;
    }
    out.detected = true;
    return out;
}

#ifdef _WIN32
static DlunaCpuOrder detect_order_windows()
{
    int nlogical = (int)std::thread::hardware_concurrency();

    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || len == 0)
        return dluna_pin_order_from_cores(nullptr, 0, nlogical);

    std::vector<uint8_t> buf(len);
    if (!GetLogicalProcessorInformationEx(
            RelationProcessorCore,
            (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)buf.data(), &len))
        return dluna_pin_order_from_cores(nullptr, 0, nlogical);

    std::vector<DlunaCoreInfo> cores;
    for (DWORD off = 0; off + sizeof(DWORD) <= len;) {
        auto *info = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)(buf.data() + off);
        if (info->Size == 0) break;
        if (info->Relationship == RelationProcessorCore) {
            const PROCESSOR_RELATIONSHIP &p = info->Processor;
            /* One processor group only: a single 64-bit KAFFINITY mask reaches
             * exactly the first group. Multi-group boxes fall back to identity. */
            if (p.GroupCount != 1 || p.GroupMask[0].Group != 0)
                return dluna_pin_order_from_cores(nullptr, 0, nlogical);

            KAFFINITY mask = p.GroupMask[0].Mask;
            int pop = 0;
            for (KAFFINITY m = mask; m; m &= m - 1) pop++;
            /* 1 = no SMT, 2 = an SMT pair; anything else is topology we do not
             * model -- fall back rather than guess. */
            if (pop < 1 || pop > 2)
                return dluna_pin_order_from_cores(nullptr, 0, nlogical);

            DlunaCoreInfo c;
            c.efficiency_class = (int)p.EfficiencyClass;
            c.primary_cpu = -1;
            c.sibling_cpu = -1;
            for (int bit = 0; bit < 64; bit++) {
                if (!(mask & ((KAFFINITY)1 << bit))) continue;
                if (c.primary_cpu < 0) c.primary_cpu = bit;
                else c.sibling_cpu = bit;
            }
            cores.push_back(c);
        }
        off += info->Size;
    }

    return dluna_pin_order_from_cores(cores.data(), (int)cores.size(), nlogical);
}
#endif

const DlunaCpuOrder &dluna_cpu_order()
{
    static const DlunaCpuOrder order = [] {
#ifdef _WIN32
        return detect_order_windows();
#else
        /* Linux (and everything else): identity order. Distinct-physical-first
         * placement needs sysfs topology parsing; identity preserves the
         * historical i%ncores behavior while still pinning each worker. */
        return dluna_pin_order_from_cores(
            nullptr, 0, (int)std::thread::hardware_concurrency());
#endif
    }();
    return order;
}

bool dluna_pin_enabled()
{
    const char *e = getenv("DLUNA_NO_PIN");
    return !(e && e[0] && strcmp(e, "0"));
}

uint32_t dluna_pin_mining_thread(int tid)
{
#if defined(__ANDROID__) || defined(__APPLE__)
    /* No usable per-thread affinity API (Bionic) / scheduler support (Apple). */
    (void)tid;
    return 0;
#else
    if (!dluna_pin_enabled())
        return 0;
    const DlunaCpuOrder &ord = dluna_cpu_order();
    if (ord.count <= 0)
        return 0;
    int cpu = ord.order[tid % ord.count];
#ifdef _WIN32
    if (!SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << cpu))
        return 0;
    return 1;
#else
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0)
        return 0;
    return 1;
#endif
#endif
}
