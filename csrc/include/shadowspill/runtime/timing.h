/* Timing work on the device, against the runtime's own events. */

#ifndef SHADOWSPILL_RUNTIME_TIMING_H
#define SHADOWSPILL_RUNTIME_TIMING_H

#include <stdint.h>

#include <shadowspill/shadowspill.h>
#include <shadowspill/runtime/vocabulary.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------
 * Timing
 *
 * The runtime times its own transfers with backend events. These calls offer
 * the same events to a caller timing anything else on the same stream, so one
 * step's timeline has one clock rather than two.
 *
 * There is one thing here: a marker, which is where an instant on a stream is
 * recorded. A caller takes markers once, records them around whatever it wants
 * to measure -- a marker records again every time the caller asks, which is how
 * one marker times the same span every step -- and asks for the time between
 * two of them, or whether the device has reached one yet. Nothing is measured
 * until it has, so every read says whether it has, and a caller that must block
 * waits for the marker itself.
 *
 * A caller that must wait for a whole stream rather than one instant on it
 * waits for the stream; that is the only other call here.
 *
 * The compute stream is named by the integer handle its owner already has, and
 * the runtime wraps it through the backend, which is the only thing that knows
 * how.
 */

typedef struct ShadowSpillTimingMarker ShadowSpillTimingMarker;

/* Takes a marker from the runtime, before anything is recorded on it. */
SHADOWSPILL_API ShadowSpillStatus shadowspill_timing_marker_create(
    ShadowSpillRuntime *runtime,
    ShadowSpillTimingMarker **marker
);

/* Records this instant on the given stream, replacing any instant before it. */
SHADOWSPILL_API ShadowSpillStatus shadowspill_timing_marker_record(
    ShadowSpillTimingMarker *marker,
    uint64_t compute_stream
);

/* Whether the device has reached the marker, without waiting for it. */
SHADOWSPILL_API ShadowSpillStatus shadowspill_timing_marker_query(
    const ShadowSpillTimingMarker *marker,
    uint8_t *reached
);

/* Blocks the calling thread until the device reaches the marker. */
SHADOWSPILL_API ShadowSpillStatus shadowspill_timing_marker_wait(
    const ShadowSpillTimingMarker *marker
);

/*
 * The nanoseconds between two markers on the same stream. Answers OK whenever
 * the question could be asked, with `reached` telling whether both markers have
 * been passed; while either is still ahead of the device the time is not
 * written.
 */
SHADOWSPILL_API ShadowSpillStatus shadowspill_timing_elapsed(
    const ShadowSpillTimingMarker *from,
    const ShadowSpillTimingMarker *to,
    uint8_t *reached,
    uint64_t *nanoseconds
);

/* Blocks the calling thread until the device has finished the whole stream. */
SHADOWSPILL_API ShadowSpillStatus shadowspill_timing_stream_wait(
    ShadowSpillRuntime *runtime,
    uint64_t compute_stream
);

/* Gives the marker's event back to the runtime. */
SHADOWSPILL_API void shadowspill_timing_marker_release(
    ShadowSpillTimingMarker *marker
);

#ifdef __cplusplus
}
#endif

#endif
