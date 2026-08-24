#ifndef SHADOWSPILL_H
#define SHADOWSPILL_H

#include <stdint.h>

/*
 * Windows needs the declaration to differ between building the DLL and using
 * it, so the build defines SHADOWSPILL_BUILDING and nothing else has to know.
 * Elsewhere one attribute covers both.
 */
#if defined(_WIN32)
#if defined(SHADOWSPILL_BUILDING)
#define SHADOWSPILL_API __declspec(dllexport)
#else
#define SHADOWSPILL_API __declspec(dllimport)
#endif
#else
#define SHADOWSPILL_API __attribute__((visibility("default")))
#endif

#include <shadowspill/status.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * One ABI version for everything shipped in libshadowspill.
 *
 * The simulator, the planner, the runtime, the admission replay and the
 * descriptor structs they exchange each carried a version of their own. They
 * are built and shipped together, so they cannot skew apart, and seven
 * numbers to bump meant seven chances to forget one - which is how a caller
 * ends up passing a struct one field short of what the library writes.
 *
 * Backends and the PyTorch adapter keep their own versions: those are
 * compiled separately against a contract, and can genuinely differ from the
 * library they load into.
 */
#define SHADOWSPILL_ABI_VERSION 50U

/* The version the loaded library was built with. */
SHADOWSPILL_API uint32_t shadowspill_abi_version(void);

#ifdef __cplusplus
}
#endif

#endif
