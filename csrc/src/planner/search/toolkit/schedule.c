/* Generic operations on a placed schedule.
 *
 * Nothing here knows which search placed the schedule it is handed: a
 * schedule is a set of actions over a topology however it was found.
 */

#include "../../internal.h"

void shadowspill_bind_indexed_schedule(
    const ShadowSpillSimulationProgram *topology,
    const ShadowSpillIndexedSchedule *schedule,
    ShadowSpillSimulationProgram *program
) {
    *program = *topology;
    program->action_count = schedule->action_count;
    program->action_trigger_tasks = schedule->action_trigger_tasks;
    program->action_aliases = schedule->action_aliases;
    program->action_kinds = schedule->action_kinds;
    program->initial_count = schedule->initial_count;
    program->initial_aliases = schedule->initial_aliases;
    program->initial_locations = schedule->initial_locations;
    program->final_count = schedule->final_count;
    program->final_aliases = schedule->final_aliases;
    program->final_locations = schedule->final_locations;
}
