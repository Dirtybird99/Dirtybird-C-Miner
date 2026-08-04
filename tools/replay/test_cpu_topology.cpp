// test_cpu_topology.cpp - pin-order policy over synthetic topologies.

#include "cpu_topology.h"

#include <cstdio>
#include <cstring>

static int g_failures = 0;

static void expect_order(const char *name, const DlunaCpuOrder &got,
                         const uint8_t *want, int want_count, bool want_detected)
{
    bool ok = got.count == want_count && got.detected == want_detected;
    if (ok)
        for (int i = 0; i < want_count; i++)
            if (got.order[i] != want[i]) { ok = false; break; }
    if (!ok) {
        g_failures++;
        std::printf("FAIL %s: count=%d detected=%d order=[", name, got.count,
                    (int)got.detected);
        for (int i = 0; i < got.count; i++)
            std::printf("%s%d", i ? "," : "", (int)got.order[i]);
        std::printf("]\n");
    } else {
        std::printf("PASS %s\n", name);
    }
}

int main()
{
    /* i7-13700HX shape: 8 P-cores (class 1, SMT pairs on 0..15) then
     * 8 E-cores (class 0, no SMT, 16..23). */
    {
        DlunaCoreInfo cores[16];
        for (int i = 0; i < 8; i++) cores[i] = {1, 2 * i, 2 * i + 1};
        for (int i = 0; i < 8; i++) cores[8 + i] = {0, 16 + i, -1};
        const uint8_t want[24] = {0, 2, 4, 6, 8, 10, 12, 14,
                                  16, 17, 18, 19, 20, 21, 22, 23,
                                  1, 3, 5, 7, 9, 11, 13, 15};
        expect_order("hybrid_8p8e", dluna_pin_order_from_cores(cores, 16, 24),
                     want, 24, true);
    }

    /* Same machine, records enumerated E-cores first: class ordering must
     * still put P primaries ahead of E-cores. */
    {
        DlunaCoreInfo cores[16];
        for (int i = 0; i < 8; i++) cores[i] = {0, 16 + i, -1};
        for (int i = 0; i < 8; i++) cores[8 + i] = {1, 2 * i, 2 * i + 1};
        const uint8_t want[24] = {0, 2, 4, 6, 8, 10, 12, 14,
                                  16, 17, 18, 19, 20, 21, 22, 23,
                                  1, 3, 5, 7, 9, 11, 13, 15};
        expect_order("hybrid_e_first", dluna_pin_order_from_cores(cores, 16, 24),
                     want, 24, true);
    }

    /* Homogeneous 8C/16T: primaries first, HT siblings last. */
    {
        DlunaCoreInfo cores[8];
        for (int i = 0; i < 8; i++) cores[i] = {0, 2 * i, 2 * i + 1};
        const uint8_t want[16] = {0, 2, 4, 6, 8, 10, 12, 14,
                                  1, 3, 5, 7, 9, 11, 13, 15};
        expect_order("homogeneous_8c16t", dluna_pin_order_from_cores(cores, 8, 16),
                     want, 16, true);
    }

    /* Homogeneous no-SMT quad core: identity, still detected. */
    {
        DlunaCoreInfo cores[4] = {{0, 0, -1}, {0, 1, -1}, {0, 2, -1}, {0, 3, -1}};
        const uint8_t want[4] = {0, 1, 2, 3};
        expect_order("no_smt_4c", dluna_pin_order_from_cores(cores, 4, 4),
                     want, 4, true);
    }

    /* Detection failure: identity over nlogical, detected=false. */
    {
        const uint8_t want[6] = {0, 1, 2, 3, 4, 5};
        expect_order("no_cores_fallback", dluna_pin_order_from_cores(nullptr, 0, 6),
                     want, 6, false);
    }

    /* Logical id beyond the 64-bit mask range: fall back to identity. */
    {
        DlunaCoreInfo cores[2] = {{0, 0, 1}, {0, 70, -1}};
        const uint8_t want[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        expect_order("cpu_id_over_63", dluna_pin_order_from_cores(cores, 2, 8),
                     want, 8, false);
    }

    /* nlogical over 64 clamps the identity fallback at 64 entries. */
    {
        DlunaCpuOrder got = dluna_pin_order_from_cores(nullptr, 0, 128);
        if (got.count != 64 || got.detected) {
            g_failures++;
            std::printf("FAIL fallback_clamp_64: count=%d detected=%d\n",
                        got.count, (int)got.detected);
        } else {
            std::printf("PASS fallback_clamp_64\n");
        }
    }

    /* Three efficiency classes: class 2 primaries, then class 1 logicals,
     * then class 0 logicals, then class 2 siblings. */
    {
        DlunaCoreInfo cores[3] = {{2, 0, 1}, {1, 2, 3}, {0, 4, -1}};
        const uint8_t want[5] = {0, 2, 3, 4, 1};
        expect_order("three_classes", dluna_pin_order_from_cores(cores, 3, 5),
                     want, 5, true);
    }

    if (g_failures) {
        std::printf("cpu_topology: %d FAILURES\n", g_failures);
        return 1;
    }
    std::printf("cpu_topology: all tests passed\n");
    return 0;
}
