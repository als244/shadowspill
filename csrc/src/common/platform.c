#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include "platform.h"

#if defined(_WIN32)

#include <windows.h>

uint64_t shadowspill_monotonic_ns(void) {
    LARGE_INTEGER frequency;
    LARGE_INTEGER counter;
    if (!QueryPerformanceFrequency(&frequency) || frequency.QuadPart <= 0 ||
        !QueryPerformanceCounter(&counter)) {
        return 0U;
    }
    /* Split the division so a long uptime cannot overflow the multiply. */
    const uint64_t ticks = (uint64_t)counter.QuadPart;
    const uint64_t per_second = (uint64_t)frequency.QuadPart;
    return (ticks / per_second) * 1000000000U +
        ((ticks % per_second) * 1000000000U) / per_second;
}

void shadowspill_thread_yield(void) {
    (void)SwitchToThread();
}

void shadowspill_name_current_thread(const char *name) {
    if (name == NULL) {
        return;
    }
    /* SetThreadDescription wants UTF-16 and is not in every SDK; the profiler
     * callback carries the name for anyone who needs it. */
}

#else

#include <pthread.h>
#include <sched.h>
#include <time.h>

uint64_t shadowspill_monotonic_ns(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return 0U;
    }
    return (uint64_t)value.tv_sec * 1000000000U + (uint64_t)value.tv_nsec;
}

void shadowspill_thread_yield(void) {
    (void)sched_yield();
}

void shadowspill_name_current_thread(const char *name) {
    if (name == NULL) {
        return;
    }
#if defined(__linux__)
    /* Linux caps the name at 16 bytes including the terminator. */
    (void)pthread_setname_np(pthread_self(), "shadowspill.wkr");
#endif
}

#endif
