"""One CPU resolution of prepared capabilities and captured call controls."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from shadowspill.training.preparation.operations import OperationContract
from shadowspill.training.preparation.scalars import control_values
from shadowspill.training.preparation.setup import ResolvedSetup
from shadowspill_client.protocol.controls import ControlSchema, Scalar, ScalarControl
from shadowspill_client.protocol.operations import OperationKey
from shadowspill_client.protocol.requests import OperationRequest

_SCALE = ControlSchema({"grad_scale": ScalarControl(1.0)})


@dataclass(frozen=True)
class CallControls:
    optim: Mapping[str, Scalar]
    loss_params: Mapping[str, Scalar]
    grad_scale: float | None


def operation_key(setup: ResolvedSetup, request: OperationRequest) -> OperationKey:
    key = OperationKey(
        request.operation,
        None
        if request.operation == "optim_step"
        else request.outputs or setup.objective.definition.default_outputs,
        request.mode or "training",
    )
    if key not in setup.config.operations:
        raise ValueError("operation/output/mode was not prepared for this client")
    return key


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
            optimizer.definition.controls,
            optimizer.definition.controls.resolve(
                defaults=optimizer.defaults, overrides=request.optim
            ),
        )
        assert request.grad_scale is not None
        scale = float(
            control_values(_SCALE, {"grad_scale": request.grad_scale})["grad_scale"]
        )
    return CallControls(optim, loss, scale)
