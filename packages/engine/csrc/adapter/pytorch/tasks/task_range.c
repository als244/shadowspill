#include "internal.h"
#include "../internal.h"

static _Thread_local struct {
    int active;
    ShadowSpillProfilerRange range;
    const char *label;
} task_range;

void shadowspill_pytorch_task_range_begin(const char *name, const char *label) {
    task_range.range =
        name == NULL ? 0U : shadowspill_profiler_range_begin(shadowspill_pytorch_runtime(), name);
    task_range.active = 1;
    task_range.label = label;
}

void shadowspill_pytorch_task_range_end(void) {
    if (task_range.active) {
        shadowspill_profiler_range_end(shadowspill_pytorch_runtime(), task_range.range);
        task_range.active = 0;
        task_range.range = 0U;
    }
    task_range.label = NULL;
}

int shadowspill_pytorch_task_range_active(void) {
    return task_range.active;
}

const char *shadowspill_pytorch_task_range_label(void) {
    return task_range.label;
}
