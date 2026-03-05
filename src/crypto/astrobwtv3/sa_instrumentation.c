/*
 * sa_instrumentation.c - Thread-local counter storage definition
 *
 * This file defines the thread-local SACounters instance.
 * Only compiled when ENABLE_SA_INSTRUMENTATION is defined.
 */

#include "sa_instrumentation.h"

#ifdef ENABLE_SA_INSTRUMENTATION

#ifdef _MSC_VER
/* MSVC: use __declspec(thread) for thread-local */
__declspec(thread) SACounters g_sa_counters = {0};
#else
/* GCC/Clang/MinGW: use __thread for thread-local */
__thread SACounters g_sa_counters = {0};
#endif

#endif /* ENABLE_SA_INSTRUMENTATION */
