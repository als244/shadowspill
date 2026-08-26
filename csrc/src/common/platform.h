#ifndef SHADOWSPILL_COMMON_PLATFORM_H
#define SHADOWSPILL_COMMON_PLATFORM_H

/*
 * The three things the library needs from the operating system that POSIX and
 * Windows spell differently.
 *
 * Everything else it needs - threads, mutexes, atomics - it takes from
 * pthreads and <stdatomic.h>, which a Windows build supplies through its
 * toolchain rather than through a shim here.
 */

#include <stdint.h>

/* Nanoseconds from an unspecified but monotonic origin; 0 if unavailable. */
uint64_t shadowspill_monotonic_ns(void);

/* Offer the rest of this time slice to another runnable thread. */
void shadowspill_thread_yield(void);

/* Best-effort debugger/profiler label for the calling thread. */
void shadowspill_name_current_thread(const char *name);

/* Logical CPUs available to this process; at least 1. Used to size worker
   counts, which is scheduling only and never changes an answer. */
uint32_t shadowspill_logical_cpu_count(void);

#endif
