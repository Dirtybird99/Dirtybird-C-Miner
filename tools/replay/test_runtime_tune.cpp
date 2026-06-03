// test_runtime_tune.cpp - policy checks for the miner priority profile.

#include "runtime_tune.h"

#include <cstdio>

static int failures = 0;

static void expect_true(bool condition, const char* label)
{
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", label);
        ++failures;
    }
}

// Default (no env): NORMAL/smooth, no power-throttle disable.
static void test_default_is_normal_smooth()
{
    DlunaRuntimeTuneOptions options =
        dluna_runtime_tune_options_from_values(nullptr, nullptr);

    expect_true(options.level == DLUNA_PRIO_NORMAL, "default level is NORMAL");
    expect_true(!options.disable_power_throttling,
                "default leaves power throttling untouched");
}

// DLUNA_PRIORITY=max -> aggressive: MAX + power-throttle disable.
static void test_priority_max()
{
    DlunaRuntimeTuneOptions options =
        dluna_runtime_tune_options_from_values("max", nullptr);

    expect_true(options.level == DLUNA_PRIO_MAX, "DLUNA_PRIORITY=max -> MAX");
    expect_true(options.disable_power_throttling,
                "MAX disables power throttling");
}

// DLUNA_PRIORITY=normal explicit -> NORMAL, no throttle disable.
static void test_priority_normal_explicit()
{
    DlunaRuntimeTuneOptions options =
        dluna_runtime_tune_options_from_values("normal", nullptr);

    expect_true(options.level == DLUNA_PRIO_NORMAL, "DLUNA_PRIORITY=normal -> NORMAL");
    expect_true(!options.disable_power_throttling, "NORMAL keeps throttling default");
}

// Case-insensitive parsing.
static void test_case_insensitive()
{
    expect_true(dluna_priority_level_from_string("MAX") == DLUNA_PRIO_MAX,
                "MAX (upper) parses to MAX");
    expect_true(dluna_priority_level_from_string("Max") == DLUNA_PRIO_MAX,
                "Max (mixed) parses to MAX");
    expect_true(dluna_priority_level_from_string("NORMAL") == DLUNA_PRIO_NORMAL,
                "NORMAL (upper) parses to NORMAL");
}

// Unknown / empty / null -> NORMAL (defensive).
static void test_unknown_falls_back_to_normal()
{
    expect_true(dluna_priority_level_from_string("turbo") == DLUNA_PRIO_NORMAL,
                "unknown value -> NORMAL");
    expect_true(dluna_priority_level_from_string("") == DLUNA_PRIO_NORMAL,
                "empty value -> NORMAL");
    expect_true(dluna_priority_level_from_string(nullptr) == DLUNA_PRIO_NORMAL,
                "null value -> NORMAL");

    DlunaRuntimeTuneOptions options =
        dluna_runtime_tune_options_from_values("turbo", nullptr);
    expect_true(options.level == DLUNA_PRIO_NORMAL,
                "invalid DLUNA_PRIORITY -> NORMAL");
}

// Legacy DLUNA_DISABLE_RUNTIME_TUNE=1 (no DLUNA_PRIORITY) -> NORMAL.
static void test_legacy_disable_forces_normal()
{
    DlunaRuntimeTuneOptions options =
        dluna_runtime_tune_options_from_values(nullptr, "1");

    expect_true(options.level == DLUNA_PRIO_NORMAL,
                "legacy DLUNA_DISABLE_RUNTIME_TUNE=1 -> NORMAL");
    expect_true(!options.disable_power_throttling,
                "legacy disable keeps throttling default");
}

// Explicit DLUNA_PRIORITY=max beats the legacy off-switch.
static void test_explicit_priority_beats_legacy_disable()
{
    DlunaRuntimeTuneOptions options =
        dluna_runtime_tune_options_from_values("max", "1");

    expect_true(options.level == DLUNA_PRIO_MAX,
                "explicit DLUNA_PRIORITY=max overrides DLUNA_DISABLE_RUNTIME_TUNE=1");
    expect_true(options.disable_power_throttling, "MAX still disables throttling");
}

// Non-exact legacy disable value is ignored (stays NORMAL by default anyway).
static void test_non_exact_legacy_value_ignored()
{
    DlunaRuntimeTuneOptions options =
        dluna_runtime_tune_options_from_values(nullptr, "false");

    expect_true(options.level == DLUNA_PRIO_NORMAL,
                "non-exact legacy disable value -> default NORMAL");
}

// Level name round-trips.
static void test_level_names()
{
    expect_true(dluna_priority_level_name(DLUNA_PRIO_NORMAL)[0] == 'n',
                "NORMAL name");
    expect_true(dluna_priority_level_name(DLUNA_PRIO_MAX)[0] == 'm', "MAX name");
}

int main()
{
    test_default_is_normal_smooth();
    test_priority_max();
    test_priority_normal_explicit();
    test_case_insensitive();
    test_unknown_falls_back_to_normal();
    test_legacy_disable_forces_normal();
    test_explicit_priority_beats_legacy_disable();
    test_non_exact_legacy_value_ignored();
    test_level_names();

    if (failures != 0)
        return 1;
    std::printf("test_runtime_tune: all OK\n");
    return 0;
}
