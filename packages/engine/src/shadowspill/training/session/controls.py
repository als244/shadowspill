"""One CPU resolution of prepared capabilities and captured call controls."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from shadowspill.training.preparation.configuration.operations import OperationContract
from shadowspill.training.preparation.configuration.scalars import control_values
from shadowspill.training.preparation.setup import ResolvedSetup
from shadowspill_client.protocol.controls import ControlSchema, Scalar, ScalarControl
from shadowspill_client.protocol.operations import OperationKey
from shadowspill_client.protocol.requests import OperationRequest

_SCALE = ControlSchema({"grad_scale": ScalarControl(1.0)})


@dataclass(frozen=True)
class CallControls:
    optim: Mapping[str, Scalar] = field(default_factory=lambda: MappingProxyType({}))
    loss_params: Mapping[str, Scalar] = field(
        default_factory=lambda: MappingProxyType({})
    )
    grad_scale: float | None = None


def operation_key(setup: ResolvedSetup, request: OperationRequest) -> OperationKey:
    if request.operation not in ("step", "forward_backward", "forward", "optim_step"):
        raise ValueError("state commands do not select a compute capability")
    # Caller order defines the default separately for each operation and mode.
    # The objective's default is used only when resolving omitted setup options;
    # it may not be prepared at all for this runtime call.
    for key in setup.config.operations:
        if (
            key.operation == request.operation
            and key.mode == (request.mode or "training")
            and (request.outputs is None or key.outputs == request.outputs)
        ):
            return key
    raise ValueError("operation/output/mode was not prepared for this client")


def call_controls(
    setup: ResolvedSetup, request: OperationRequest, key: OperationKey
) -> CallControls:
    loss: Mapping[str, Scalar] = MappingProxyType({})
    optim: Mapping[str, Scalar] = MappingProxyType({})
    scale = None
    if key.operation != "optim_step":
        contract = OperationContract(key, setup.objective)
        loss = control_values(
            contract.controls, contract.loss_params(request.loss_params)
        )
    if key.operation in ("step", "optim_step"):
        optimizer = setup.optimizer
        if optimizer is None:
            raise ValueError("optimizer was not prepared for this client")
        optim = control_values(
            optimizer.component.controls,
            optimizer.component.controls.resolve(
                defaults=optimizer.defaults, overrides=request.optim
            ),
        )
        assert request.grad_scale is not None
        scale = float(
            control_values(_SCALE, {"grad_scale": request.grad_scale})["grad_scale"]
        )
    return CallControls(optim, loss, scale)
