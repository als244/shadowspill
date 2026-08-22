#ifndef SHADOWSPILL_H
#define SHADOWSPILL_H

#include <shadowspill/status.h>

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
#define SHADOWSPILL_ABI_VERSION 48U

#endif
