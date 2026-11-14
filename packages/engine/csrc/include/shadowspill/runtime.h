/*
 * The neutral runtime's public API, in the order a program uses it.
 *
 * This header is the umbrella: it includes one header per subsystem, and
 * including it gives the whole API. The parts are vocabulary (statuses,
 * reasons, enumerations), lanes (what moves bytes between two pools), pool
 * memory (where a pool's memory comes from), library (what a separately
 * loaded library offers), descriptions (what a caller declares), diagnostics
 * (what the runtime reports), lifecycle (opening and closing a runtime and a
 * plan), pools, objects, plan admission, task boundaries, telemetry, and
 * timing (work timed against the runtime's own events). Each may also be
 * included on its own.
 */

#ifndef SHADOWSPILL_RUNTIME_H
#define SHADOWSPILL_RUNTIME_H

#include <shadowspill/runtime/vocabulary.h>

#include <shadowspill/runtime/lane.h>
#include <shadowspill/runtime/pool_memory.h>
#include <shadowspill/runtime/library.h>
#include <shadowspill/runtime/descriptions.h>
#include <shadowspill/runtime/diagnostics.h>

#include <shadowspill/runtime/lifecycle.h>
#include <shadowspill/runtime/pools.h>
#include <shadowspill/runtime/objects.h>
#include <shadowspill/runtime/plan.h>
#include <shadowspill/runtime/tasks.h>
#include <shadowspill/runtime/telemetry.h>
#include <shadowspill/runtime/timing.h>

#endif
