"""Completion facts constrain terminal outcomes independently of artifact delivery."""

from shadowspill_client.protocol.requests import OperationRequest
from shadowspill_client.protocol.results import ExecutionReceipt, RequestStatus


def validate_receipt(
    request: OperationRequest, status: RequestStatus, receipt: ExecutionReceipt
) -> None:
    if status.state != "running":
        raise ValueError("execution receipt requires dispatched work")
    if receipt.identity != request.identity or receipt.operation != request.operation:
        raise ValueError("execution receipt differs from its request")


def validate_outcome(status: RequestStatus, receipt: ExecutionReceipt | None) -> None:
    if status.result is not None:
        if receipt is None:
            raise ValueError("completion requires a confirmed execution receipt")
        if ExecutionReceipt.from_result(status.result) != receipt:
            raise ValueError("result differs from confirmed execution")
    if status.failure is not None:
        failure = status.failure
        if receipt is None:
            if failure.execution_effect == "completed":
                raise ValueError(
                    "completed failure requires a confirmed execution receipt"
                )
        elif (
            failure.execution_effect != "completed"
            or failure.completed_updates != receipt.completed_updates
            or failure.accumulation != receipt.accumulation
            or failure.history_id != receipt.history_id
        ):
            raise ValueError("failure differs from confirmed execution")
