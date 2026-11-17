"""Optional creation display; byte cursors and decoders stay bounded per stream."""

from __future__ import annotations

import codecs
import sys
from contextlib import suppress
from datetime import UTC, datetime
from time import monotonic
from typing import TYPE_CHECKING

from .protocol.progress import LOG_CHUNK_BYTES, ProgressPage, ProgressRequest
from .protocol.runs import RunInfo

if TYPE_CHECKING:
    from .service import ServiceClient


class ProgressDisplay:
    def __init__(self, enabled: bool) -> None:
        if type(enabled) is not bool:
            raise TypeError("progress must be boolean")
        self.enabled = enabled
        self._stream = sys.stdout
        self._offsets: dict[str, int] = {}
        self._decoders: dict[str, codecs.IncrementalDecoder] = {}
        self._partial: dict[str, str] = {}
        self._capped: set[str] = set()
        self._stage: str | None = None
        self._next_poll = 0.0

    def _write(self, text: str) -> None:
        self._stream.write(text)
        self._stream.flush()

    def update(
        self, service: ServiceClient, info: RunInfo, *, final: bool = False
    ) -> None:
        if not self.enabled:
            return
        try:
            stage = info.stage or info.state
            changed = stage != self._stage
            if changed:
                self._stage = stage
                stamp = (
                    datetime.now(UTC)
                    .isoformat(timespec="seconds")
                    .replace("+00:00", "Z")
                )
                self._write(
                    f"\n{stamp}  Run {info.run_id[:12]} · {stage.capitalize()}\n"
                )
            if not final and not changed and monotonic() < self._next_poll:
                return
            self._next_poll = monotonic() + 0.5
            while True:
                service._work.check()
                page = service._request(
                    "POST",
                    f"/v1/runs/{info.session_id}/{info.run_id}/progress",
                    ProgressPage,
                    ProgressRequest(self._offsets),
                )
                if (page.session_id, page.run_id) != (info.session_id, info.run_id):
                    raise ValueError("creation logs crossed run scope")
                more = False
                for chunk in page.chunks:
                    name = chunk.stream
                    if chunk.offset != self._offsets.get(name, 0):
                        raise ValueError("creation logs changed their byte cursor")
                    data = chunk.bytes()
                    self._offsets[name] = chunk.offset + len(data)
                    decoder = self._decoders.setdefault(
                        name, codecs.getincrementaldecoder("utf-8")("replace")
                    )
                    text = self._partial.get(name, "") + decoder.decode(
                        data, final=chunk.complete
                    )
                    end = text.rfind("\n") + 1
                    if end:
                        self._write(text[:end])
                    text = text[end:]
                    # Arbitrary worker output may contain an extremely long line.
                    if len(text) > LOG_CHUNK_BYTES or (chunk.complete and text):
                        self._write(text + "\n")
                        text = ""
                    self._partial[name] = text
                    if chunk.capped and name not in self._capped:
                        self._capped.add(name)
                        self._write(
                            f"  ! {name}: log reached its configured size limit.\n"
                        )
                    more |= bool(data) and not chunk.complete
                if not final or not more:
                    break
        except Exception as error:
            # Reporting cannot cancel or replay accepted server work. Run-state
            # polling still reports the actual creation outcome independently.
            self.enabled = False
            with suppress(Exception):
                self._write(
                    f"\n  ! Live progress unavailable ({type(error).__name__}); "
                    "creation continues.\n"
                )
