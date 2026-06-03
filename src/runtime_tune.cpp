#ifdef _WIN32
#if !defined(_WIN32_WINNT) || (_WIN32_WINNT < 0x0A00)
#undef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#endif

#include "runtime_tune.h"

#include <cstdlib>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <processthreadsapi.h>
#endif

static bool env_is_exactly(const char* value, char expected)
{
    return value && value[0] == expected && value[1] == '\0';
}

static bool str_ieq(const char* a, const char* b)
{
    if (!a || !b) return false;
    for (; *a && *b; ++a, ++b) {
        char ca = *a, cb = *b;
        if (ca >= 'A' && ca <= 'Z') ca = (char)(ca + 32);
        if (cb >= 'A' && cb <= 'Z') cb = (char)(cb + 32);
        if (ca != cb) return false;
    }
    return *a == '\0' && *b == '\0';
}

DlunaPriorityLevel dluna_priority_level_from_string(const char* s)
{
    if (str_ieq(s, "max")) return DLUNA_PRIO_MAX;
    /* "normal", null, empty, or any unrecognized value -> NORMAL (smooth) */
    return DLUNA_PRIO_NORMAL;
}

const char* dluna_priority_level_name(DlunaPriorityLevel level)
{
    return level == DLUNA_PRIO_MAX ? "max" : "normal";
}

DlunaRuntimeTuneOptions dluna_runtime_tune_options_from_values(
    const char* priority,
    const char* disable_runtime_tune)
{
    /* Base profile: NORMAL/smooth — make no syscalls at all. */
    DlunaRuntimeTuneOptions options{DLUNA_PRIO_NORMAL, false};

    const bool have_priority = priority && priority[0] != '\0';
    if (have_priority) {
        options.level = dluna_priority_level_from_string(priority);
    } else if (env_is_exactly(disable_runtime_tune, '1')) {
        /* Legacy off-switch (only honored when DLUNA_PRIORITY is absent). */
        options.level = DLUNA_PRIO_NORMAL;
    }

    /* The power-throttle (EcoQoS) disable is power-adjacent; only the aggressive
     * MAX profile asserts it. NORMAL leaves Windows power management untouched. */
    if (options.level == DLUNA_PRIO_MAX)
        options.disable_power_throttling = true;

    return options;
}

DlunaRuntimeTuneOptions dluna_runtime_tune_options_from_env()
{
    return dluna_runtime_tune_options_from_values(
        std::getenv("DLUNA_PRIORITY"),
        std::getenv("DLUNA_DISABLE_RUNTIME_TUNE"));
}

uint32_t dluna_tune_process_runtime()
{
    const DlunaRuntimeTuneOptions options = dluna_runtime_tune_options_from_env();

    uint32_t applied = DLUNA_RUNTIME_TUNE_NONE;

#ifdef _WIN32
    /* NORMAL: deliberately make NO calls — leave the process at the OS default
     * priority class so the scheduler keeps the desktop responsive. */
    if (options.level == DLUNA_PRIO_MAX) {
        if (SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS)) {
            applied |= DLUNA_RUNTIME_TUNE_PROCESS_PRIORITY;
        }
    }

#if defined(PROCESS_POWER_THROTTLING_CURRENT_VERSION) && \
    defined(PROCESS_POWER_THROTTLING_EXECUTION_SPEED)
    if (options.disable_power_throttling) {
        PROCESS_POWER_THROTTLING_STATE state{};
        state.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
        state.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
        state.StateMask = 0;

        if (SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling,
                                  &state, sizeof(state))) {
            applied |= DLUNA_RUNTIME_TUNE_PROCESS_POWER;
        }
    }
#endif
#endif

    return applied;
}

uint32_t dluna_tune_mining_thread()
{
    const DlunaRuntimeTuneOptions options = dluna_runtime_tune_options_from_env();

    uint32_t applied = DLUNA_RUNTIME_TUNE_NONE;

#ifdef _WIN32
    /* NORMAL: no SetThreadPriority — workers run at the default thread priority
     * (same as DeroLuna), so the OS can preempt them for the UI. */
    if (options.level == DLUNA_PRIO_MAX) {
        if (SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST)) {
            applied |= DLUNA_RUNTIME_TUNE_THREAD_PRIORITY;
        }
    }

#if defined(THREAD_POWER_THROTTLING_CURRENT_VERSION) && \
    defined(THREAD_POWER_THROTTLING_EXECUTION_SPEED)
    if (options.disable_power_throttling) {
        THREAD_POWER_THROTTLING_STATE state{};
        state.Version = THREAD_POWER_THROTTLING_CURRENT_VERSION;
        state.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
        state.StateMask = 0;

        if (SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling,
                                 &state, sizeof(state))) {
            applied |= DLUNA_RUNTIME_TUNE_THREAD_POWER;
        }
    }
#endif
#endif

    return applied;
}
