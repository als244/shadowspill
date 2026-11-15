"""Installed CPU-only preparation child over the existing private control channel."""

import os
import socket
import sys
from typing import BinaryIO, cast

from .channel import read_message, write_message
from .conversion_records import ConversionCommand, ConversionFailed, ConversionRequest


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
            from shadowspill.checkpoint.preparation import materialize, prepare

            prepared = prepare(request)
            write_message(stream, prepared.description)
            command = read_message(stream)
            if not isinstance(command, ConversionCommand):
                raise ValueError("checkpoint preparation requires admission or close")
            if command.action == "write":
                write_message(stream, materialize(prepared))
        except Exception as error:
            write_message(
                stream, ConversionFailed(str(error)[:2048] or type(error).__name__)
            )
            raise


if __name__ == "__main__":
    main()
