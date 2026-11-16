"""Installed CPU-only preparation child over the existing private control channel."""

import os
import socket
import sys
from collections.abc import Callable
from functools import partial
from typing import BinaryIO, cast

from shadowspill_client.protocol.checkpoints import WeightsManifest

from .channel import read_message, write_message
from .conversion_records import (
    ConversionCommand,
    ConversionFailed,
    ConversionRequest,
    ConversionWritten,
)


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("checkpoint preparation requires an empty GPU visibility mask")
    with (
        socket.socket(fileno=int(sys.argv[1])) as channel,
        channel.makefile("rwb", buffering=0) as file,
    ):
        stream = cast(BinaryIO, file)
        try:
            request = read_message(stream)
            if not isinstance(request, ConversionRequest):
                raise ValueError("checkpoint preparation requires a conversion request")
            if request.host_bytes < 1 << 20:
                raise ValueError("checkpoint conversion host envelope is too small")
            materialize: Callable[[], ConversionWritten]
            if isinstance(request.source, WeightsManifest):
                from shadowspill.checkpoint.weights_preparation import (
                    materialize as write_weights,
                )
                from shadowspill.checkpoint.weights_preparation import (
                    prepare as prepare_weights,
                )

                weights = prepare_weights(request)
                description = weights.description
                materialize = partial(write_weights, weights)
            else:
                from shadowspill.checkpoint.preparation import (
                    materialize as write_checkpoint,
                )
                from shadowspill.checkpoint.preparation import (
                    prepare as prepare_checkpoint,
                )

                checkpoint = prepare_checkpoint(request)
                description = checkpoint.description
                materialize = partial(write_checkpoint, checkpoint)
            write_message(stream, description)
            command = read_message(stream)
            if not isinstance(command, ConversionCommand):
                raise ValueError("checkpoint preparation requires admission or close")
            if command.action == "write":
                write_message(stream, materialize())
        except Exception as error:
            write_message(
                stream, ConversionFailed(str(error)[:2048] or type(error).__name__)
            )
            raise


if __name__ == "__main__":
    main()
