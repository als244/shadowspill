#include "internal.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int placement_key_compare(
    const ShadowSpillFixedPlacementDescription *left,
    const ShadowSpillFixedPlacementDescription *right
) {
    if (left->kind != right->kind) {
        return left->kind < right->kind ? -1 : 1;
    }
    if (left->task_id != right->task_id) {
        return left->task_id < right->task_id ? -1 : 1;
    }
    if (left->ordinal != right->ordinal) {
        return left->ordinal < right->ordinal ? -1 : 1;
    }
    if (left->object_id != right->object_id) {
        return left->object_id < right->object_id ? -1 : 1;
    }
    return 0;
}

static int placement_compare(const void *left, const void *right) {
    return placement_key_compare(left, right);
}

static int dependency_compare(const void *left_value, const void *right_value) {
    const ShadowSpillFixedRuntimeDependency *left = left_value;
    const ShadowSpillFixedRuntimeDependency *right = right_value;
    const ShadowSpillFixedDependencyDescription *a = &left->description;
    const ShadowSpillFixedDependencyDescription *b = &right->description;
    if (a->successor_kind != b->successor_kind) {
        return a->successor_kind < b->successor_kind ? -1 : 1;
    }
    if (a->successor_task_id != b->successor_task_id) {
        return a->successor_task_id < b->successor_task_id ? -1 : 1;
    }
    if (a->successor_ordinal != b->successor_ordinal) {
        return a->successor_ordinal < b->successor_ordinal ? -1 : 1;
    }
    if (a->predecessor_task_id != b->predecessor_task_id) {
        return a->predecessor_task_id < b->predecessor_task_id ? -1 : 1;
    }
    if (a->predecessor_action_ordinal != b->predecessor_action_ordinal) {
        return a->predecessor_action_ordinal < b->predecessor_action_ordinal
            ? -1 : 1;
    }
    return 0;
}

static int valid_placement(
    const ShadowSpillFixedPlacementDescription *placement,
    uint64_t slice_bytes
) {
    if (placement->kind > SHADOWSPILL_FIXED_ACTION_DESTINATION ||
        placement->bytes == 0U || placement->alignment_bytes == 0U ||
        placement->offset > slice_bytes ||
        placement->bytes > slice_bytes - placement->offset ||
        placement->offset % placement->alignment_bytes != 0U) {
        return 0;
    }
    if (placement->kind == SHADOWSPILL_FIXED_INITIAL_OBJECT) {
        return placement->task_id == SHADOWSPILL_RUNTIME_NO_ID &&
            placement->ordinal == SHADOWSPILL_RUNTIME_NO_ID &&
            placement->object_id != SHADOWSPILL_RUNTIME_NO_ID;
    }
    if (placement->task_id == SHADOWSPILL_RUNTIME_NO_ID ||
        placement->ordinal == SHADOWSPILL_RUNTIME_NO_ID) {
        return 0;
    }
    return placement->kind == SHADOWSPILL_FIXED_TASK_ALLOCATION
        ? placement->object_id == SHADOWSPILL_RUNTIME_NO_ID
        : placement->object_id != SHADOWSPILL_RUNTIME_NO_ID;
}

static int valid_dependency(
    const ShadowSpillFixedDependencyDescription *dependency
) {
    return dependency->predecessor_task_id != SHADOWSPILL_RUNTIME_NO_ID &&
        dependency->predecessor_action_ordinal != SHADOWSPILL_RUNTIME_NO_ID &&
        dependency->successor_task_id != SHADOWSPILL_RUNTIME_NO_ID &&
        dependency->successor_ordinal != SHADOWSPILL_RUNTIME_NO_ID &&
        (dependency->successor_kind == SHADOWSPILL_FIXED_TASK_ALLOCATION ||
         dependency->successor_kind == SHADOWSPILL_FIXED_ACTION_DESTINATION);
}

static void clear_metadata(ShadowSpillFixedLayoutState *layout) {
    free(layout->placements);
    free(layout->dependencies);
    *layout = (ShadowSpillFixedLayoutState){0};
}

/*
 * Own one contiguous execution-pool slice for an admitted physical layout.
 * Individual fixed leases borrow subranges of this slice; only this owner
 * changes the production range allocator's physical accounting.
 */
ShadowSpillRuntimeStatus shadowspill_fixed_layout_reserve_slice(
    ShadowSpillRuntime *runtime,
    uint64_t bytes
) {
    if (runtime == NULL || bytes == 0U) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    ShadowSpillMemoryPool *pool = shadowspill_execution_pool(runtime);
    shadowspill_memory_pool_lock_reservation(pool);
    ShadowSpillRuntimeStatus status = SHADOWSPILL_RUNTIME_OK;
    if (runtime->fixed_layout.active) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
    } else {
        uint64_t offset = 0U;
        const int reserve_status = shadowspill_memory_pool_reserve_locked(
            pool,
            bytes,
            pool->minimum_alignment,
            SHADOWSPILL_MEMORY_BEST_FIT_LOW,
            &offset
        );
        if (reserve_status != 0) {
            status = reserve_status > 0
                ? SHADOWSPILL_RUNTIME_OUT_OF_MEMORY
                : SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
        } else {
            runtime->fixed_layout.slice_offset = offset;
            runtime->fixed_layout.slice_bytes = bytes;
            runtime->fixed_layout.active = 1U;
            shadowspill_publish_execution_geometry_locked(runtime);
        }
    }
    shadowspill_memory_pool_unlock_reservation(pool);
    shadowspill_memory_pool_relinquish_reservation(pool);
    return status;
}

static int has_borrowed_layout_lease(const ShadowSpillMemoryPool *pool) {
    for (const ShadowSpillMemoryLease *lease = pool->leases;
         lease != NULL; lease = lease->pool_next) {
        if (!lease->owns_pool_range) {
            return 1;
        }
    }
    return 0;
}

ShadowSpillRuntimeStatus shadowspill_fixed_layout_clear(
    ShadowSpillRuntime *runtime
) {
    if (runtime == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    ShadowSpillMemoryPool *pool = shadowspill_execution_pool(runtime);
    shadowspill_memory_pool_lock_reservation(pool);
    ShadowSpillRuntimeStatus status = SHADOWSPILL_RUNTIME_OK;
    if (!runtime->fixed_layout.active) {
        /* Clearing an absent layout is intentionally idempotent. */
    } else if (has_borrowed_layout_lease(pool)) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
    } else if (shadowspill_memory_pool_release_locked(
                   pool,
                   runtime->fixed_layout.slice_offset,
                   runtime->fixed_layout.slice_bytes
               ) != 0) {
        status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    } else {
        clear_metadata(&runtime->fixed_layout);
        shadowspill_publish_execution_geometry_locked(runtime);
    }
    shadowspill_memory_pool_unlock_reservation(pool);
    shadowspill_memory_pool_relinquish_reservation(pool);
    return status;
}

void shadowspill_fixed_layout_destroy(ShadowSpillRuntime *runtime) {
    if (runtime != NULL) {
        clear_metadata(&runtime->fixed_layout);
    }
}

ShadowSpillRuntimeStatus shadowspill_fixed_layout_adopt_execution_lease_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryLease *lease,
    uint64_t relative_offset,
    uint64_t bytes,
    uint64_t alignment
) {
    if (runtime == NULL || lease == NULL || !runtime->fixed_layout.active ||
        !runtime->fixed_layout.sealed || bytes == 0U ||
        relative_offset > runtime->fixed_layout.slice_bytes ||
        bytes > runtime->fixed_layout.slice_bytes - relative_offset) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    ShadowSpillMemoryPool *pool = shadowspill_execution_pool(runtime);
    if (alignment < pool->minimum_alignment) {
        alignment = pool->minimum_alignment;
    }
    const uint64_t absolute_offset =
        runtime->fixed_layout.slice_offset + relative_offset;
    if (absolute_offset % alignment != 0U ||
        shadowspill_memory_pool_adopt_borrowed_lease_locked(
            pool, lease, bytes, alignment, absolute_offset
        ) != 0) {
        return SHADOWSPILL_RUNTIME_PLAN_VIOLATION;
    }
    return SHADOWSPILL_RUNTIME_OK;
}

const ShadowSpillFixedPlacementDescription *
shadowspill_fixed_layout_find_placement(
    const ShadowSpillRuntime *runtime,
    uint8_t kind,
    uint64_t task_id,
    uint64_t ordinal,
    uint64_t object_id
) {
    if (runtime == NULL || !runtime->fixed_layout.sealed) {
        return NULL;
    }
    const ShadowSpillFixedPlacementDescription key = {
        .task_id = task_id,
        .ordinal = ordinal,
        .object_id = object_id,
        .kind = kind,
    };
    return bsearch(
        &key,
        runtime->fixed_layout.placements,
        (size_t)runtime->fixed_layout.placement_count,
        sizeof(*runtime->fixed_layout.placements),
        placement_compare
    );
}

static ShadowSpillRuntimeStatus copy_layout_description(
    ShadowSpillRuntime *runtime,
    const ShadowSpillFixedLayoutDescription *description
) {
    ShadowSpillFixedPlacementDescription *placements =
        description->placement_count == 0U
        ? NULL
        : malloc(
              (size_t)description->placement_count * sizeof(*placements)
          );
    ShadowSpillFixedRuntimeDependency *dependencies =
        description->dependency_count == 0U
        ? NULL
        : calloc(
              (size_t)description->dependency_count, sizeof(*dependencies)
          );
    if ((description->placement_count != 0U && placements == NULL) ||
        (description->dependency_count != 0U && dependencies == NULL)) {
        free(placements);
        free(dependencies);
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    if (placements != NULL) {
        memcpy(
            placements,
            description->placements,
            (size_t)description->placement_count * sizeof(*placements)
        );
        qsort(
            placements,
            (size_t)description->placement_count,
            sizeof(*placements),
            placement_compare
        );
    }
    for (uint64_t index = 0U; index < description->dependency_count; ++index) {
        dependencies[index].description = description->dependencies[index];
    }
    if (dependencies != NULL) {
        qsort(
            dependencies,
            (size_t)description->dependency_count,
            sizeof(*dependencies),
            dependency_compare
        );
    }
    runtime->fixed_layout.placements = placements;
    runtime->fixed_layout.placement_count = description->placement_count;
    runtime->fixed_layout.dependencies = dependencies;
    runtime->fixed_layout.dependency_count = description->dependency_count;
    return SHADOWSPILL_RUNTIME_OK;
}

static int descriptions_are_valid(
    const ShadowSpillFixedLayoutDescription *description
) {
    if (description == NULL ||
        description->abi_version != SHADOWSPILL_FIXED_LAYOUT_ABI_VERSION ||
        description->slice_bytes == 0U ||
        (description->placement_count != 0U &&
         description->placements == NULL) ||
        (description->dependency_count != 0U &&
         description->dependencies == NULL) ||
        description->placement_count > SIZE_MAX /
            sizeof(ShadowSpillFixedPlacementDescription) ||
        description->dependency_count > SIZE_MAX /
            sizeof(ShadowSpillFixedRuntimeDependency)) {
        return 0;
    }
    for (uint64_t index = 0U; index < description->placement_count; ++index) {
        if (!valid_placement(
                &description->placements[index], description->slice_bytes
            )) {
            return 0;
        }
    }
    for (uint64_t index = 0U; index < description->dependency_count; ++index) {
        if (!valid_dependency(&description->dependencies[index])) {
            return 0;
        }
    }
    return 1;
}

static int sorted_layout_is_unique(const ShadowSpillFixedLayoutState *layout) {
    for (uint64_t index = 1U; index < layout->placement_count; ++index) {
        if (placement_key_compare(
                &layout->placements[index - 1U],
                &layout->placements[index]
            ) == 0) {
            return 0;
        }
    }
    for (uint64_t index = 1U; index < layout->dependency_count; ++index) {
        if (dependency_compare(
                &layout->dependencies[index - 1U],
                &layout->dependencies[index]
            ) == 0) {
            return 0;
        }
    }
    return 1;
}

ShadowSpillRuntimeStatus shadowspill_admit_fixed_layout(
    ShadowSpillRuntime *runtime,
    const ShadowSpillFixedLayoutDescription *description
) {
    if (runtime == NULL || !descriptions_are_valid(description)) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    if (runtime->fixed_layout.active || runtime->execution.owned_head != NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    ShadowSpillRuntimeStatus status = copy_layout_description(
        runtime, description
    );
    if (status != SHADOWSPILL_RUNTIME_OK) {
        return status;
    }
    if (!sorted_layout_is_unique(&runtime->fixed_layout)) {
        clear_metadata(&runtime->fixed_layout);
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    status = shadowspill_fixed_layout_reserve_slice(
        runtime, description->slice_bytes
    );
    if (status != SHADOWSPILL_RUNTIME_OK) {
        clear_metadata(&runtime->fixed_layout);
    }
    return status;
}

static const ShadowSpillTaskAllocationABIStep *allocation_step(
    const ShadowSpillExecutionRecord *record,
    uint64_t ordinal
) {
    for (uint32_t index = 0U; index < record->allocation_abi_step_count; ++index) {
        const ShadowSpillTaskAllocationABIStep *step =
            &record->allocation_abi_steps[index];
        if (step->operation == SHADOWSPILL_TASK_ALLOCATION_ALLOCATE &&
            step->allocation_ordinal == ordinal) {
            return step;
        }
    }
    return NULL;
}

static ShadowSpillRuntimeStatus validate_resolved_placement(
    ShadowSpillRuntime *runtime,
    const ShadowSpillFixedPlacementDescription *placement
) {
    if (placement->kind == SHADOWSPILL_FIXED_INITIAL_OBJECT) {
        ShadowSpillObject *object = shadowspill_find_object(
            runtime, placement->object_id
        );
        return object != NULL && object->size_bytes == placement->bytes
            ? SHADOWSPILL_RUNTIME_OK : SHADOWSPILL_RUNTIME_PLAN_VIOLATION;
    }
    ShadowSpillExecutionRecord *record = shadowspill_execution_table_acquire(
        &runtime->execution, placement->task_id
    );
    if (record == NULL) {
        return SHADOWSPILL_RUNTIME_PLAN_VIOLATION;
    }
    if (placement->kind == SHADOWSPILL_FIXED_TASK_ALLOCATION) {
        const ShadowSpillTaskAllocationABIStep *step = allocation_step(
            record, placement->ordinal
        );
        return step != NULL && step->charged_bytes == placement->bytes &&
            step->alignment_bytes == placement->alignment_bytes
            ? SHADOWSPILL_RUNTIME_OK : SHADOWSPILL_RUNTIME_PLAN_VIOLATION;
    }
    if (placement->ordinal >= record->action_count) {
        return SHADOWSPILL_RUNTIME_PLAN_VIOLATION;
    }
    const ShadowSpillExecutionAction *action =
        &record->actions[placement->ordinal];
    return action->kind == SHADOWSPILL_RUNTIME_PREFETCH &&
        action->object->object_id == placement->object_id &&
        action->object->size_bytes == placement->bytes
        ? SHADOWSPILL_RUNTIME_OK : SHADOWSPILL_RUNTIME_PLAN_VIOLATION;
}

static ShadowSpillRuntimeStatus resolve_dependency(
    ShadowSpillRuntime *runtime,
    ShadowSpillFixedRuntimeDependency *dependency
) {
    const ShadowSpillFixedDependencyDescription *item =
        &dependency->description;
    ShadowSpillExecutionRecord *predecessor =
        shadowspill_execution_table_acquire(
            &runtime->execution, item->predecessor_task_id
        );
    if (predecessor == NULL ||
        item->predecessor_action_ordinal >= predecessor->action_count ||
        predecessor->actions[item->predecessor_action_ordinal].kind !=
            SHADOWSPILL_RUNTIME_OFFLOAD) {
        return SHADOWSPILL_RUNTIME_PLAN_VIOLATION;
    }
    dependency->predecessor_action =
        &predecessor->queued_actions[item->predecessor_action_ordinal];
    uint64_t successor_object_id = SHADOWSPILL_RUNTIME_NO_ID;
    if (item->successor_kind == SHADOWSPILL_FIXED_ACTION_DESTINATION) {
        ShadowSpillExecutionRecord *successor_record =
            shadowspill_execution_table_acquire(
                &runtime->execution, item->successor_task_id
            );
        if (successor_record == NULL ||
            item->successor_ordinal >= successor_record->action_count) {
            return SHADOWSPILL_RUNTIME_PLAN_VIOLATION;
        }
        successor_object_id = successor_record
            ->actions[item->successor_ordinal].object->object_id;
    }
    const ShadowSpillFixedPlacementDescription *successor =
        shadowspill_fixed_layout_find_placement(
            runtime,
            item->successor_kind,
            item->successor_task_id,
            item->successor_ordinal,
            successor_object_id
        );
    return successor == NULL
        ? SHADOWSPILL_RUNTIME_PLAN_VIOLATION : SHADOWSPILL_RUNTIME_OK;
}

ShadowSpillRuntimeStatus shadowspill_seal_fixed_layout(
    ShadowSpillRuntime *runtime
) {
    if (runtime == NULL || !runtime->fixed_layout.active ||
        runtime->fixed_layout.sealed) {
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    /* Lookup helpers require the immutable layout to be visible while sealing. */
    runtime->fixed_layout.sealed = 1U;
    for (uint64_t index = 0U; index < runtime->fixed_layout.placement_count;
         ++index) {
        ShadowSpillRuntimeStatus status = validate_resolved_placement(
            runtime, &runtime->fixed_layout.placements[index]
        );
        if (status != SHADOWSPILL_RUNTIME_OK) {
            runtime->fixed_layout.sealed = 0U;
            return status;
        }
    }
    for (uint64_t index = 0U; index < runtime->fixed_layout.dependency_count;
         ++index) {
        ShadowSpillRuntimeStatus status = resolve_dependency(
            runtime, &runtime->fixed_layout.dependencies[index]
        );
        if (status != SHADOWSPILL_RUNTIME_OK) {
            runtime->fixed_layout.sealed = 0U;
            return status;
        }
    }
    return SHADOWSPILL_RUNTIME_OK;
}
