/* Whether a bound schedule can run on its program.
 *
 * The rules are the schedule's own, independent of timing: the residency it
 * starts from, the order of its actions, every input on the device when its
 * task reads it, every move one its alias's copies allow, and the residency it
 * promises to end in. They are checked in the order the readable rules state
 * them (`MemorySchedule.validate`), so the first refusal is the same one:
 * the initial residency, then every action's own facts, then the walk over
 * the tasks and their actions, then the final residency.
 */

#include <shadowspill/simulator.h>

#include <stdint.h>
#include <stdlib.h>

#include "internal.h"

/* Where one alias group's copies are, during the walk. */
enum {
    ON_DEVICE = 1U << 0U,
    ON_SPILL = 1U << 1U,
    SPILL_CURRENT = 1U << 2U,
};

static void refuse(
    ShadowSpillScheduleCheck *result, uint32_t refusal, uint32_t index, uint32_t alias
) {
    result->refusal = refusal;
    result->index = index;
    result->alias = alias;
}

static int apply_action(
    const ShadowSpillSimulationProgram *program,
    const uint8_t *flags,
    const uint32_t *last_reader,
    const uint8_t *final_alias,
    uint8_t *state,
    uint32_t action,
    ShadowSpillScheduleCheck *result
) {
    const uint32_t alias = program->action_aliases[action];
    const uint32_t task = program->action_trigger_tasks[action];
    const uint8_t kind = program->action_kinds[action];
    const int retained = (flags[alias] & SHADOWSPILL_ALIAS_RETAINED) != 0U;
    if (kind == SHADOWSPILL_MEMORY_FETCH) {
        if ((state[alias] & (ON_SPILL | SPILL_CURRENT)) != (ON_SPILL | SPILL_CURRENT)) {
            refuse(result, SHADOWSPILL_SCHEDULE_REFUSAL_FETCH_WITHOUT_HOST_COPY, action, alias);
            return 0;
        }
        if ((state[alias] & ON_DEVICE) != 0U) {
            refuse(result, SHADOWSPILL_SCHEDULE_REFUSAL_FETCH_ALREADY_RESIDENT, action, alias);
            return 0;
        }
        state[alias] |= ON_DEVICE;
        if (!retained) {
            state[alias] &= (uint8_t)~(ON_SPILL | SPILL_CURRENT);
        }
        return 1;
    }
    if ((state[alias] & ON_DEVICE) == 0U) {
        refuse(result, SHADOWSPILL_SCHEDULE_REFUSAL_NOT_DEVICE_RESIDENT, action, alias);
        return 0;
    }
    if (kind == SHADOWSPILL_MEMORY_RELEASE) {
        /* Releasing the device copy is safe when a current spill copy remains,
         * or when nothing later reads the value and the final residency does
         * not name it. */
        const int still_needed = final_alias[alias] != 0U ||
            (last_reader[alias] != SHADOWSPILL_SIMULATOR_NO_INDEX &&
             last_reader[alias] > task);
        if ((state[alias] & SPILL_CURRENT) == 0U && still_needed) {
            refuse(result, SHADOWSPILL_SCHEDULE_REFUSAL_RELEASE_ONLY_COPY, action, alias);
            return 0;
        }
        state[alias] &= (uint8_t)~ON_DEVICE;
        if (!retained) {
            state[alias] &= (uint8_t)~(ON_SPILL | SPILL_CURRENT);
        }
        return 1;
    }
    state[alias] |= ON_SPILL | SPILL_CURRENT;
    if (kind == SHADOWSPILL_MEMORY_EVICT) {
        state[alias] &= (uint8_t)~ON_DEVICE;
    }
    return 1;
}

static int check(
    const ShadowSpillSimulationProgram *program,
    const uint8_t *flags,
    uint8_t *state,
    uint32_t *last_reader,
    uint8_t *final_alias,
    ShadowSpillScheduleCheck *result
) {
    /* Who reads each alias last, and which aliases the final residency names. */
    for (uint32_t task = 0; task < program->task_count; ++task) {
        for (uint32_t item = program->input_offsets[task];
             item < program->input_offsets[task + 1U]; ++item) {
            last_reader[program->input_aliases[item]] = task;
        }
        for (uint32_t item = program->mutation_offsets[task];
             item < program->mutation_offsets[task + 1U]; ++item) {
            last_reader[program->mutation_aliases[item]] = task;
        }
    }
    for (uint32_t item = 0; item < program->final_count; ++item) {
        final_alias[program->final_aliases[item]] = 1U;
    }

    /* The starting residency: runtime-owned groups on the device, groups that
     * keep a spill copy current there, then what the schedule declares. */
    for (uint32_t alias = 0; alias < program->alias_count; ++alias) {
        if ((flags[alias] & SHADOWSPILL_ALIAS_SHARED) != 0U) {
            state[alias] |= ON_DEVICE;
        }
        if ((flags[alias] & SHADOWSPILL_ALIAS_RETAINED) != 0U) {
            state[alias] |= ON_SPILL | SPILL_CURRENT;
        }
    }
    for (uint32_t item = 0; item < program->initial_count; ++item) {
        const uint32_t alias = program->initial_aliases[item];
        if ((flags[alias] & SHADOWSPILL_ALIAS_SHARED) != 0U) {
            refuse(result, SHADOWSPILL_SCHEDULE_REFUSAL_INITIAL_RUNTIME_OWNED, item, alias);
            return 0;
        }
        if ((flags[alias] & SHADOWSPILL_ALIAS_ZERO_SIZE) != 0U) {
            continue;
        }
        state[alias] |= program->initial_locations[item] == SHADOWSPILL_MEMORY_DEVICE
            ? ON_DEVICE
            : (uint8_t)(ON_SPILL | SPILL_CURRENT);
    }

    /* Every action's own facts, before any is applied. */
    for (uint32_t action = 0; action < program->action_count; ++action) {
        const uint32_t alias = program->action_aliases[action];
        if ((flags[alias] & SHADOWSPILL_ALIAS_SHARED) != 0U) {
            refuse(result, SHADOWSPILL_SCHEDULE_REFUSAL_ACTION_RUNTIME_OWNED, action, alias);
            return 0;
        }
        if ((flags[alias] & SHADOWSPILL_ALIAS_ZERO_SIZE) != 0U) {
            refuse(result, SHADOWSPILL_SCHEDULE_REFUSAL_ACTION_ZERO_SIZE, action, alias);
            return 0;
        }
        if (action > 0U &&
            program->action_trigger_tasks[action] <
                program->action_trigger_tasks[action - 1U]) {
            refuse(result, SHADOWSPILL_SCHEDULE_REFUSAL_ACTION_ORDER, action, alias);
            return 0;
        }
    }

    /* The walk: each task reads, writes, then runs the actions it triggers. */
    uint32_t action = 0U;
    for (uint32_t task = 0; task < program->task_count; ++task) {
        for (uint32_t item = program->input_offsets[task];
             item < program->input_offsets[task + 1U]; ++item) {
            const uint32_t alias = program->input_aliases[item];
            if ((flags[alias] & SHADOWSPILL_ALIAS_ZERO_SIZE) == 0U &&
                (state[alias] & ON_DEVICE) == 0U) {
                refuse(result, SHADOWSPILL_SCHEDULE_REFUSAL_INPUT_NOT_RESIDENT, task, alias);
                return 0;
            }
        }
        for (uint32_t item = program->output_offsets[task];
             item < program->output_offsets[task + 1U]; ++item) {
            const uint32_t alias = program->output_aliases[item];
            if ((flags[alias] & SHADOWSPILL_ALIAS_ZERO_SIZE) == 0U) {
                state[alias] |= ON_DEVICE;
                state[alias] &= (uint8_t)~SPILL_CURRENT;
            }
        }
        for (uint32_t item = program->mutation_offsets[task];
             item < program->mutation_offsets[task + 1U]; ++item) {
            const uint32_t alias = program->mutation_aliases[item];
            if ((flags[alias] & SHADOWSPILL_ALIAS_ZERO_SIZE) == 0U) {
                state[alias] &= (uint8_t)~SPILL_CURRENT;
            }
        }
        for (; action < program->action_count &&
               program->action_trigger_tasks[action] == task;
             ++action) {
            if (!apply_action(
                    program, flags, last_reader, final_alias, state, action, result)) {
                return 0;
            }
        }
    }

    /* The residency the schedule promised to end in. */
    for (uint32_t item = 0; item < program->final_count; ++item) {
        const uint32_t alias = program->final_aliases[item];
        if ((flags[alias] & SHADOWSPILL_ALIAS_SHARED) != 0U) {
            refuse(result, SHADOWSPILL_SCHEDULE_REFUSAL_FINAL_RUNTIME_OWNED, item, alias);
            return 0;
        }
        if ((flags[alias] & SHADOWSPILL_ALIAS_ZERO_SIZE) != 0U) {
            continue;
        }
        const int reached = program->final_locations[item] == SHADOWSPILL_MEMORY_DEVICE
            ? (state[alias] & ON_DEVICE) != 0U
            : (state[alias] & (ON_SPILL | SPILL_CURRENT)) == (ON_SPILL | SPILL_CURRENT);
        if (!reached) {
            refuse(result, SHADOWSPILL_SCHEDULE_REFUSAL_FINAL_NOT_REACHED, item, alias);
            return 0;
        }
    }
    return 1;
}

ShadowSpillStatus shadowspill_check_schedule(
    const ShadowSpillSimulationProgram *program,
    const uint8_t *alias_flags,
    ShadowSpillScheduleCheck *result
) {
    if (result == NULL || !shadowspill_validate_program(program) ||
        (program->alias_count != 0U && alias_flags == NULL)) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    refuse(result, SHADOWSPILL_SCHEDULE_REFUSAL_NONE, 0U, 0U);
    const size_t aliases = program->alias_count == 0U ? 1U : program->alias_count;
    uint8_t *state = calloc(aliases, sizeof(*state));
    uint32_t *last_reader = malloc(aliases * sizeof(*last_reader));
    uint8_t *final_alias = calloc(aliases, sizeof(*final_alias));
    if (state == NULL || last_reader == NULL || final_alias == NULL) {
        free(state);
        free(last_reader);
        free(final_alias);
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }
    for (uint32_t alias = 0; alias < program->alias_count; ++alias) {
        last_reader[alias] = SHADOWSPILL_SIMULATOR_NO_INDEX;
    }
    (void)check(program, alias_flags, state, last_reader, final_alias, result);
    free(state);
    free(last_reader);
    free(final_alias);
    return SHADOWSPILL_STATUS_OK;
}
