#define _POSIX_C_SOURCE 200809L

#include "internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int stream_equal(
    ShadowSpillBackendStream left,
    ShadowSpillBackendStream right
) {
    return memcmp(&left, &right, sizeof(left)) == 0;
}

static uint64_t monotonic_nanoseconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return 0U;
    }
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) +
        (uint64_t)value.tv_nsec;
}

int shadowspill_completion_tracker_initialize(
    ShadowSpillCompletionTracker *tracker
) {
    if (tracker == NULL) {
        return -1;
    }
    *tracker = (ShadowSpillCompletionTracker){0};
    return pthread_mutex_init(&tracker->lock, NULL);
}

void shadowspill_completion_tracker_destroy(
    ShadowSpillRuntime *runtime,
    ShadowSpillCompletionTracker *tracker
) {
    if (runtime == NULL || tracker == NULL) {
        return;
    }
    ShadowSpillCompletionStream *stream = tracker->streams;
    while (stream != NULL) {
        ShadowSpillCompletionStream *next_stream = stream->next;
        ShadowSpillCompletionRecord *record = stream->head;
        while (record != NULL) {
            ShadowSpillCompletionRecord *next_record = record->next;
            (void)shadowspill_event_lease_release(runtime, record->event);
            free(record);
            record = next_record;
        }
        free(stream);
        stream = next_stream;
    }
    tracker->streams = NULL;
    tracker->pending = 0U;
    pthread_mutex_destroy(&tracker->lock);
}

ShadowSpillRuntimeStatus shadowspill_completion_submit(
    ShadowSpillRuntime *runtime,
    ShadowSpillBackendStream stream,
    ShadowSpillEventLease *event,
    uint64_t object_id,
    uint64_t allocation_id
) {
    if (runtime == NULL || event == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    ShadowSpillCompletionRecord *record = calloc(1U, sizeof(*record));
    if (record == NULL) {
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    record->event = event;
    record->object_id = object_id;
    record->allocation_id = allocation_id;
    shadowspill_event_lease_retain(event);

    ShadowSpillCompletionTracker *tracker = &runtime->completions;
    pthread_mutex_lock(&tracker->lock);
    ShadowSpillCompletionStream *owner = tracker->streams;
    while (owner != NULL && !stream_equal(owner->stream, stream)) {
        owner = owner->next;
    }
    if (owner == NULL) {
        owner = calloc(1U, sizeof(*owner));
        if (owner == NULL) {
            pthread_mutex_unlock(&tracker->lock);
            (void)shadowspill_event_lease_release(runtime, event);
            free(record);
            return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
        }
        owner->stream = stream;
        owner->next = tracker->streams;
        tracker->streams = owner;
    }
    if (owner->tail == NULL) {
        owner->head = record;
    } else {
        owner->tail->next = record;
    }
    owner->tail = record;
    ++tracker->pending;
    pthread_mutex_unlock(&tracker->lock);
    return SHADOWSPILL_RUNTIME_OK;
}

int shadowspill_completion_poll(
    ShadowSpillRuntime *runtime,
    uint64_t *next_poll_nanoseconds,
    uint64_t *failure_object_id,
    uint64_t *failure_allocation_id
) {
    if (runtime == NULL || next_poll_nanoseconds == NULL ||
        failure_object_id == NULL || failure_allocation_id == NULL) {
        return -1;
    }
    *next_poll_nanoseconds = 0U;
    *failure_object_id = SHADOWSPILL_RUNTIME_NO_ID;
    *failure_allocation_id = SHADOWSPILL_RUNTIME_NO_ID;
    int changed = 0;
    const uint64_t now = monotonic_nanoseconds();
    ShadowSpillCompletionTracker *tracker = &runtime->completions;

    pthread_mutex_lock(&tracker->lock);
    ShadowSpillCompletionStream *stream = tracker->streams;
    pthread_mutex_unlock(&tracker->lock);
    while (stream != NULL) {
        for (;;) {
            pthread_mutex_lock(&tracker->lock);
            ShadowSpillCompletionRecord *record = stream->head;
            const uint64_t due = stream->next_poll_timestamp_ns;
            if (record != NULL && (due == 0U || due <= now)) {
                shadowspill_event_lease_retain(record->event);
            }
            pthread_mutex_unlock(&tracker->lock);
            if (record == NULL) {
                break;
            }
            if (due != 0U && due > now) {
                const uint64_t remaining = due - now;
                if (*next_poll_nanoseconds == 0U ||
                    remaining < *next_poll_nanoseconds) {
                    *next_poll_nanoseconds = remaining;
                }
                break;
            }

            int complete = 0;
            const int query_status = shadowspill_event_lease_query(
                runtime, record->event, &complete
            );
            pthread_mutex_lock(&tracker->lock);
            if (stream->head != record) {
                pthread_mutex_unlock(&tracker->lock);
                (void)shadowspill_event_lease_release(runtime, record->event);
                continue;
            }
            if (query_status != 0) {
                *failure_object_id = record->object_id;
                *failure_allocation_id = record->allocation_id;
                pthread_mutex_unlock(&tracker->lock);
                (void)shadowspill_event_lease_release(runtime, record->event);
                return -1;
            }
            if (!complete) {
                const uint64_t delay = runtime->worker_poll_nanoseconds;
                stream->next_poll_timestamp_ns = now + delay;
                if (*next_poll_nanoseconds == 0U ||
                    delay < *next_poll_nanoseconds) {
                    *next_poll_nanoseconds = delay;
                }
                pthread_mutex_unlock(&tracker->lock);
                (void)shadowspill_event_lease_release(runtime, record->event);
                break;
            }
            stream->head = record->next;
            if (stream->head == NULL) {
                stream->tail = NULL;
            }
            stream->next_poll_timestamp_ns = 0U;
            if (tracker->pending != 0U) {
                --tracker->pending;
            }
            pthread_mutex_unlock(&tracker->lock);
            (void)shadowspill_event_lease_release(runtime, record->event);
            (void)shadowspill_event_lease_release(runtime, record->event);
            free(record);
            changed = 1;
        }
        stream = stream->next;
    }
    return changed;
}
