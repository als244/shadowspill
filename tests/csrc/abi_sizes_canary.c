/*
 * Print the size of every struct the Python ctypes mirrors describe.
 *
 * A mirror that drifts from its C struct is not a compile error and not a
 * failed call: it is silent heap corruption, because ctypes writes the size it
 * believes in. Comparing the two sizes is the only cheap way to catch it.
 */

#include <stdio.h>
#include <stdlib.h>

#include <shadowspill/pytorch_adapter.h>
#include <shadowspill/runtime.h>

#define REPORT(name) printf("%s %zu\n", #name, sizeof(name))

int main(void) {
    REPORT(ShadowSpillPytorchPoolConfig);
    REPORT(ShadowSpillPytorchRouteConfig);
    REPORT(ShadowSpillPytorchAdapterConfig);
    REPORT(ShadowSpillPytorchAdapterStatistics);
    REPORT(ShadowSpillPytorchAdapterFailure);
    REPORT(ShadowSpillRuntimeStatistics);
    REPORT(ShadowSpillAllocationEvent);
    REPORT(ShadowSpillTraceConfig);
    REPORT(ShadowSpillTransferRouteKey);
    REPORT(ShadowSpillTransferCalibrationConfig);
    REPORT(ShadowSpillTransferProfile);
    REPORT(ShadowSpillTraceEvent);
    REPORT(ShadowSpillTraceSummary);
    REPORT(ShadowSpillAllocation);
    REPORT(ShadowSpillRuntimeFailure);
    REPORT(ShadowSpillObjectBinding);
    REPORT(ShadowSpillObjectDescription);
    REPORT(ShadowSpillObjectUpdate);
    REPORT(ShadowSpillRuntimeAction);
    REPORT(ShadowSpillTaskPublicationDescription);
    REPORT(ShadowSpillTaskAllocationContractStep);
    REPORT(ShadowSpillTaskDescription);
    REPORT(ShadowSpillFixedPlacementDescription);
    REPORT(ShadowSpillFixedDependencyDescription);
    REPORT(ShadowSpillFixedLayoutDescription);
    REPORT(ShadowSpillObjectSnapshot);
    REPORT(ShadowSpillObjectLocationSnapshot);
    REPORT(ShadowSpillPytorchAdapterCapabilities);
    REPORT(ShadowSpillPytorchPhysicalAdmission);
    REPORT(ShadowSpillPytorchTaskDispatchTiming);
    return EXIT_SUCCESS;
}
