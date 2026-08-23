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

#endif
