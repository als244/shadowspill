"""Plan-local tensor views with explicit borrowing of shared model storage."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableMapping

import torch


class StorageViews(MutableMapping[str, torch.Tensor]):
    def __init__(
        self,
        source: StorageViews | None = None,
        aliases: Mapping[str, str] | None = None,
    ):
        self.local: dict[str, torch.Tensor] = {}
        self.source, self.aliases = source, dict(aliases or {})

    def __getitem__(self, key: str) -> torch.Tensor:
        if key in self.aliases:
            assert self.source is not None
            return self.source[self.aliases[key]]
        return self.local[key]

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        if key in self.aliases:
            assert self.source is not None
            self.source[self.aliases[key]] = value
        else:
            self.local[key] = value

    def __delitem__(self, key: str) -> None:
        if key in self.aliases:
            assert self.source is not None
            del self.source[self.aliases[key]]
        else:
            del self.local[key]

    def __iter__(self) -> Iterator[str]:
        yield from self.local
        if self.source is not None:
            yield from (
                key for key, value in self.aliases.items() if value in self.source
            )

    def __len__(self) -> int:
        return sum(1 for _ in self)
