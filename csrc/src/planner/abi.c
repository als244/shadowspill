/* Structure sizes, so a caller mirroring these layouts checks its mirror
 * at load instead of discovering a mismatch as corrupted fields.
 *
 * This is the one planner file that includes a search's header: the sizes
 * it reports span the generic planner and whichever searches ship, and a
 * size check has to see the real definitions.
 */

#include <shadowspill/planner.h>
#include <shadowspill/pressurefit/pressurefit.h>

uint64_t shadowspill_planner_struct_size(uint32_t which) {
    switch (which) {
    case SHADOWSPILL_PRESSUREFIT_STRUCT_OPTIONS:
        return sizeof(ShadowSpillPressureFitOptions);
    case SHADOWSPILL_PRESSUREFIT_STRUCT_WORK_DIAGNOSTICS:
        return sizeof(ShadowSpillPressureFitWorkDiagnostics);
    case SHADOWSPILL_PRESSUREFIT_STRUCT_CANDIDATE_DIAGNOSTIC:
        return sizeof(ShadowSpillPressureFitCandidateDiagnostic);
    case SHADOWSPILL_PRESSUREFIT_STRUCT_SECTION_TIMING:
        return sizeof(ShadowSpillPressureFitSectionTiming);
    case SHADOWSPILL_PRESSUREFIT_STRUCT_REDUCTION_STEP:
        return sizeof(ShadowSpillPressureFitReductionStep);
    case SHADOWSPILL_STRUCT_ADMISSION_FACTS:
        return sizeof(ShadowSpillAdmissionFacts);
    case SHADOWSPILL_PRESSUREFIT_STRUCT_BEST_PLACED_RECORD:
        return sizeof(ShadowSpillPressureFitBestPlacedRecord);
    case SHADOWSPILL_PRESSUREFIT_STRUCT_RESULT:
        return sizeof(ShadowSpillPressureFitResult);
    default:
        return 0U;
    }
}
