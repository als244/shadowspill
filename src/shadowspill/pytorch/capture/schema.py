"""Normalized dispatcher alias metadata used by offline task lowering.

PyTorch's Python ``FunctionSchema`` projection currently drops the element
alias labels from list-valued tensor returns such as ``Tensor(a)[]``.  The
bundled torchgen parser is the source of truth for the same dispatcher schema
text and retains those labels.  This narrow adapter contains that
version-sensitive normalization so the semantic lowering code consumes one
small, explicit contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from torchgen.model import FunctionSchema  # type: ignore[import-untyped]


@dataclass(frozen=True, slots=True)
class SchemaAlias:
    """Alias labels and mutation behavior for one argument or return."""

    name: str
    labels: frozenset[str]
    is_write: bool


@dataclass(frozen=True, slots=True)
class OperatorAliasContract:
    """Complete normalized alias projection of one dispatcher schema."""

    arguments: tuple[SchemaAlias, ...]
    returns: tuple[SchemaAlias, ...]


def operator_alias_contract(schema: Any) -> OperatorAliasContract:
    """Return alias metadata, including list-element return annotations."""

    text = str(schema)
    try:
        contract = _parse_alias_contract(text)
    except (AssertionError, RuntimeError, ValueError):
        # torch.library permits some valid custom schemas that torchgen's
        # code-generation model intentionally rejects.  The dispatcher
        # projection remains authoritative for those ordinary tensor returns.
        contract = _dispatcher_alias_contract(schema)
    argument_names = tuple(argument.name for argument in schema.arguments)
    if argument_names != tuple(argument.name for argument in contract.arguments):
        raise RuntimeError(
            "torchgen and dispatcher argument orders differ for schema " + text
        )
    if len(contract.returns) != len(schema.returns):
        raise RuntimeError(
            "torchgen and dispatcher return arities differ for schema " + text
        )
    return contract


def _dispatcher_alias_contract(schema: Any) -> OperatorAliasContract:
    def alias(argument: Any) -> SchemaAlias:
        info = argument.alias_info
        labels = (
            frozenset()
            if info is None
            else frozenset((*info.before_set, *info.after_set))
        )
        return SchemaAlias(
            argument.name,
            labels,
            bool(info is not None and info.is_write),
        )

    return OperatorAliasContract(
        arguments=tuple(alias(argument) for argument in schema.arguments),
        returns=tuple(alias(result) for result in schema.returns),
    )


@lru_cache(maxsize=4096)
def _parse_alias_contract(text: str) -> OperatorAliasContract:
    parsed = FunctionSchema.parse(text)

    def alias(name: str | None, annotation: Any) -> SchemaAlias:
        if annotation is None:
            return SchemaAlias(name or "", frozenset(), False)
        labels = frozenset((*annotation.alias_set, *annotation.alias_set_after))
        return SchemaAlias(name or "", labels, bool(annotation.is_write))

    return OperatorAliasContract(
        arguments=tuple(
            alias(argument.name, argument.annotation)
            for argument in parsed.arguments.flat_all
        ),
        returns=tuple(
            alias(result.name, result.annotation) for result in parsed.returns
        ),
    )


__all__ = ["OperatorAliasContract", "SchemaAlias", "operator_alias_contract"]
