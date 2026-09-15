#ifndef SHADOWSPILL_RUNTIME_WORKER_INTERNAL_H
#define SHADOWSPILL_RUNTIME_WORKER_INTERNAL_H

/*
 * The worker thread and the queued actions it drives.
 *
 * One action moves through three stages, and this directory holds one file
 * per stage: it is handled (its object lock taken, its kind and state
 * decided), dispatched onto a transfer lane, and completed once the backend
 * says the copy is done. Every function below is called with the action's
 * object lock held and returns with it released, because the stages hand an
 * action to one another mid-flight; the return is the same three-valued
 * answer throughout -- negative for a latched failure, zero for "not yet,
 * try again", one for dispatched, two for finished and unlinked.
 */

#include "../internal.h"

int shadowspill_action_handle(
    ShadowSpillRuntime *runtime,
    ShadowSpillQueuedAction *action
);

/* The in-flight half: commit a completed transfer, or leave it queued. */
int shadowspill_action_finish_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillQueuedAction *action
);

void shadowspill_action_latch_failure(
    ShadowSpillRuntime *runtime,
    const ShadowSpillQueuedAction *action,
    ShadowSpillStatus status,
    ShadowSpillFailureReason reason,
    uint64_t object_id,
    uint64_t allocation_id,
    uint64_t requested_bytes
);

/* Unlinks the action, releases its events, and notifies an idle runtime. */
void shadowspill_action_complete(
    ShadowSpillRuntime *runtime,
    ShadowSpillQueuedAction *action
);

/* Whether the destination a transfer writes into may enter its lane yet. */
int shadowspill_action_destination_ready(ShadowSpillQueuedAction *action);

int shadowspill_action_dispatch_evict_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillQueuedAction *action
);

int shadowspill_action_dispatch_fetch_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillQueuedAction *action
);

#endif
