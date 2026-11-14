"""Plan-local tensor views with explicit borrowing of shared model storage."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableMapping

import torch


class StorageViews(MutableMapping[str, torch.Tensor]):
    def __init__(
        self,
        sources: Mapping[str, tuple[StorageViews, str]] | None = None,
    ):
        self.local: dict[str, torch.Tensor] = {}
        self.sources = dict(sources or {})

    def __getitem__(self, key: str) -> torch.Tensor:
        if key in self.sources:
            source, alias = self.sources[key]
            return source[alias]
        return self.local[key]

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        if key in self.sources:
            source, alias = self.sources[key]
            source[alias] = value
        else:
            self.local[key] = value

    def __delitem__(self, key: str) -> None:
        if key in self.sources:
            source, alias = self.sources[key]
            del source[alias]
        else:
            del self.local[key]

    def __iter__(self) -> Iterator[str]:
        yield from self.local
        yield from (
            key for key, (source, alias) in self.sources.items() if alias in source
        )

    def __len__(self) -> int:
        return sum(1 for _ in self)
